import cocotb
from cocotb.clock import Clock

from dataclasses import dataclass

from lpu_tb import (
    WB_MEM0,
    WB_MEM1,
    WB_SXM,
    EB_MEM0,
    EB_MXM,
    EB_SXM,
    EB_VXM,
    WC_MXM,
    WC_SXM,
    WC_MEM0,
    EC_SXM,
    EC_VXM,
    EC_MEM0,
    EC_MEM1,
    build_instruction,
    INGRESS_INPUT,
    INGRESS_WGHT,
    preload_mem0_word,
    preload_mem1_word,
    preload_program,
    reset_dut,
    tick,
)


@dataclass(frozen=True)
class AttentionAddressMap:
    x_mem0_base: int = 0
    wq_mem1_base: int = 16
    wk_mem1_base: int = 32
    wv_mem1_base: int = 48
    q_quant_mem0_base: int = 64
    k_quant_mem0_base: int = 80
    v_quant_mem0_base: int = 96
    kt_quant_mem0_base: int = 112
    scores_mem0_base: int = 128
    probs_mem0_base: int = 144
    output_mem0_base: int = 160


@dataclass(frozen=True)
class SelfAttentionFixture:
    x_matrix: list[list[int]]
    w_q: list[list[int]]
    w_k: list[list[int]]
    w_v: list[list[int]]


SOFTMAX_LN2 = 177
SOFTMAX_MAX_BITS = 30
SOFTMAX_OUT_BITS = 8
SOFTMAX_SHIFT = SOFTMAX_MAX_BITS - SOFTMAX_OUT_BITS


def sign_extend(value: int, bits: int) -> int:
    sign_bit = 1 << (bits - 1)
    return (value ^ sign_bit) - sign_bit


def clip_signed_q8(value: int) -> int:
    if value > 127:
        return 127
    if value < -127:
        return -127
    return value


def clip_unsigned_q8(value: int) -> int:
    if value <= 0:
        return 0
    if value >= 255:
        return 255
    return value & 0xFF


def compute_row_shift(lanes: list[int]) -> int:
    max_abs = max(abs(lane) for lane in lanes)
    shift = 0
    while max_abs > 127:
        max_abs >>= 1
        shift += 1
    return shift


def round_shift_signed(value: int, shift: int) -> int:
    if shift == 0:
        return value
    rounding = 1 << (shift - 1)
    if value >= 0:
        return (value + rounding) >> shift
    return (value - rounding) >> shift


def matmul_int(a_matrix: list[list[int]], b_matrix: list[list[int]]) -> list[list[int]]:
    rows = len(a_matrix)
    inner = len(a_matrix[0])
    cols = len(b_matrix[0])
    out = [[0 for _ in range(cols)] for _ in range(rows)]
    for row in range(rows):
        for col in range(cols):
            acc = 0
            for k_idx in range(inner):
                acc += a_matrix[row][k_idx] * b_matrix[k_idx][col]
            out[row][col] = acc
    return out


def quantize_row_int32_to_q8(lanes: list[int]) -> tuple[list[int], int]:
    shift = compute_row_shift(lanes)
    return [clip_signed_q8(round_shift_signed(value, shift)) for value in lanes], shift


def quantize_matrix_int32_to_q8(matrix: list[list[int]]) -> tuple[list[list[int]], list[int]]:
    rows: list[list[int]] = []
    shifts: list[int] = []
    for row in matrix:
        quantized_row, shift = quantize_row_int32_to_q8(row)
        rows.append(quantized_row)
        shifts.append(shift)
    return rows, shifts


def transpose_matrix(matrix: list[list[int]]) -> list[list[int]]:
    return [[matrix[row][col] for row in range(4)] for col in range(4)]


def exp_expected(q_value: int) -> int:
    coeff_a = 92
    coeff_b = 346
    coeff_c = 88

    z_value = (-q_value) // SOFTMAX_LN2
    p_value = q_value + (z_value * SOFTMAX_LN2)
    t_value = p_value + coeff_b
    t_squared = t_value * t_value
    q_poly = ((coeff_a * t_squared) >> 16) + coeff_c
    return q_poly >> z_value


def softmax_expected_row(lanes: list[int]) -> list[int]:
    lane_max = max(lanes)
    lane_sub = [lane - lane_max for lane in lanes]
    lane_exp = [exp_expected(lane) for lane in lane_sub]
    sum_exp = sum(lane_exp)
    quotient = (1 << SOFTMAX_MAX_BITS) // sum_exp
    return [(quotient * lane) >> SOFTMAX_SHIFT for lane in lane_exp]


def scale_softmax_quant_row(lanes: list[int]) -> list[int]:
    scaled_lanes = [lane >> 1 for lane in lanes]
    softmax_lanes = softmax_expected_row(scaled_lanes)
    return [clip_unsigned_q8(lane) for lane in softmax_lanes]


def softmax_quantize_scores(scores: list[list[int]]) -> list[list[int]]:
    return [scale_softmax_quant_row(row) for row in scores]


def pack_q8_row(row: list[int]) -> int:
    word = 0
    for idx, value in enumerate(row):
        word |= (value & 0xFF) << (8 * idx)
    return word


def pack_int32_row(row: list[int]) -> int:
    word = 0
    for idx, value in enumerate(row):
        word |= (value & 0xFFFFFFFF) << (32 * idx)
    return word


def self_attention_golden(
    x_matrix: list[list[int]],
    w_q: list[list[int]],
    w_k: list[list[int]],
    w_v: list[list[int]],
) -> dict[str, list]:
    q_int32 = matmul_int(x_matrix, w_q)
    q_q8, q_scales = quantize_matrix_int32_to_q8(q_int32)

    k_int32 = matmul_int(x_matrix, w_k)
    k_q8, k_scales = quantize_matrix_int32_to_q8(k_int32)
    k_q8_t = transpose_matrix(k_q8)

    v_int32 = matmul_int(x_matrix, w_v)
    v_q8, v_scales = quantize_matrix_int32_to_q8(v_int32)

    scores_int32 = matmul_int(q_q8, k_q8_t)
    probs_q8 = softmax_quantize_scores(scores_int32)
    output_int32 = matmul_int(probs_q8, v_q8)

    return {
        "q_int32": q_int32,
        "q_q8": q_q8,
        "q_scales": q_scales,
        "q_q8_packed": [pack_q8_row(row) for row in q_q8],
        "k_int32": k_int32,
        "k_q8": k_q8,
        "k_scales": k_scales,
        "k_q8_packed": [pack_q8_row(row) for row in k_q8],
        "k_q8_t": k_q8_t,
        "k_q8_t_packed": [pack_q8_row(row) for row in k_q8_t],
        "v_int32": v_int32,
        "v_q8": v_q8,
        "v_scales": v_scales,
        "v_q8_packed": [pack_q8_row(row) for row in v_q8],
        "scores_int32": scores_int32,
        "scores_int32_packed": [pack_int32_row(row) for row in scores_int32],
        "probs_q8": probs_q8,
        "probs_q8_packed": [pack_q8_row(row) for row in probs_q8],
        "output_int32": output_int32,
        "output_int32_packed": [pack_int32_row(row) for row in output_int32],
    }


def matrix_to_mem0_columns(matrix: list[list[int]]) -> list[list[int]]:
    """Pack a 4x4 matrix into the MEM0 column-major layout expected by MXM."""
    return [[matrix[row][col] for row in range(4)] for col in range(4)]


def matrix_to_mem1_rows(matrix: list[list[int]]) -> list[list[int]]:
    """Pack a 4x4 matrix into the MEM1 row-major layout expected by MXM."""
    return [list(row) for row in matrix]


def preload_matrix_into_mem0_columns(dut, *, base_addr: int, matrix: list[list[int]]) -> None:
    for idx, values in enumerate(matrix_to_mem0_columns(matrix)):
        preload_mem0_word(dut, addr=base_addr + idx, values=values)


def preload_matrix_into_mem1_rows(dut, *, base_addr: int, matrix: list[list[int]]) -> None:
    for idx, values in enumerate(matrix_to_mem1_rows(matrix)):
        preload_mem1_word(dut, addr=base_addr + idx, values=values)


def read_packed_q8_rows_from_mem0(dut, *, base_addr: int) -> list[list[int]]:
    rows: list[list[int]] = []
    for row_idx in range(4):
        word = int(dut.u_lpu.u_mem0.sram_array[base_addr + row_idx].value) & 0xFFFFFFFF
        rows.append([sign_extend((word >> (8 * lane)) & 0xFF, 8) for lane in range(4)])
    return rows


def read_packed_u8_rows_from_mem0(dut, *, base_addr: int) -> list[list[int]]:
    rows: list[list[int]] = []
    for row_idx in range(4):
        word = int(dut.u_lpu.u_mem0.sram_array[base_addr + row_idx].value) & 0xFFFFFFFF
        rows.append([(word >> (8 * lane)) & 0xFF for lane in range(4)])
    return rows


def read_full_int32_rows_from_mem0(dut, *, base_addr: int) -> list[list[int]]:
    rows: list[list[int]] = []
    for row_idx in range(4):
        word = int(dut.u_lpu.u_mem0.sram_array[base_addr + row_idx].value)
        rows.append([sign_extend((word >> (32 * lane)) & 0xFFFFFFFF, 32) for lane in range(4)])
    return rows


def read_row_scales_from_mem0(dut, *, base_addr: int) -> list[int]:
    scales: list[int] = []
    for row_idx in range(4):
        word = int(dut.u_lpu.u_mem0.sram_array[base_addr + row_idx].value)
        scales.append((word >> 32) & 0xFF)
    return scales


def build_projection_program(
    *,
    mem0_base: int,
    mem1_base: int,
    target_base: int,
    target_mem: str = "mem0",
    quant_wait_cycles: int = 10,
    store_cycles: int = 4,
) -> list[int]:
    program: list[int] = []

    if target_mem not in {"mem0", "mem1"}:
        raise ValueError(f"unsupported target_mem={target_mem!r}")

    for k_idx in range(4):
        program.extend(
            [
                build_instruction(
                    mem1_read_en=1,
                    mem1_addr=mem1_base + k_idx,
                    westbound_sel=WB_MEM1,
                    westbound_consumer_sel=WC_MXM,
                    mxm_ingress_mode=INGRESS_WGHT,
                    mxm_input_is_signed=1,
                    mxm_wght_is_signed=1,
                ),
                build_instruction(
                    westbound_sel=WB_MEM1,
                    westbound_consumer_sel=WC_MXM,
                    mxm_ingress_mode=INGRESS_WGHT,
                    mxm_input_is_signed=1,
                    mxm_wght_is_signed=1,
                ),
                build_instruction(
                    mem0_read_en=1,
                    mem0_addr=mem0_base + k_idx,
                    westbound_sel=WB_MEM0,
                    westbound_consumer_sel=WC_MXM,
                    mxm_ingress_mode=INGRESS_INPUT,
                    mxm_input_is_signed=1,
                    mxm_wght_is_signed=1,
                ),
                build_instruction(
                    westbound_sel=WB_MEM0,
                    westbound_consumer_sel=WC_MXM,
                    mxm_ingress_mode=INGRESS_INPUT,
                    mxm_input_is_signed=1,
                    mxm_wght_is_signed=1,
                ),
                build_instruction(mxm_start=1, mxm_input_is_signed=1, mxm_wght_is_signed=1),
                build_instruction(mxm_start=1, mxm_input_is_signed=1, mxm_wght_is_signed=1),
            ]
        )

    program.extend([build_instruction(), build_instruction()])

    for row_idx in range(4):
        program.append(
            build_instruction(
                eastbound_sel=EB_MXM,
                eastbound_consumer_sel=EC_VXM,
                mxm_e_row_sel=row_idx,
                mxm_e_valid_in=1,
                vxm_ctrl=0b0000,
                vxm_data_sel=1,
            )
        )
        program.append(
            build_instruction(
                vxm_ctrl=0b0000,
                vxm_data_sel=1,
            )
        )

        for _ in range(quant_wait_cycles):
            program.append(build_instruction())

        for _ in range(store_cycles):
            if target_mem == "mem0":
                program.append(
                    build_instruction(
                        eastbound_sel=EB_VXM,
                        eastbound_consumer_sel=EC_MEM0,
                        mem0_write_en=1,
                        mem0_addr=target_base + row_idx,
                    )
                )
            else:
                program.append(
                    build_instruction(
                        eastbound_sel=EB_VXM,
                        eastbound_consumer_sel=EC_MEM1,
                        mem1_write_en=1,
                        mem1_addr=target_base + row_idx,
                    )
                )

    return program


def build_sxm_transpose_program(*, source_base: int, target_base: int, use_westbound: bool = True) -> list[int]:
    if use_westbound:
        return [
            build_instruction(mem0_read_en=1, mem0_addr=source_base + 0),
            build_instruction(
                mem0_read_en=1,
                mem0_addr=source_base + 1,
                westbound_sel=WB_MEM0,
                westbound_consumer_sel=WC_SXM,
            ),
            build_instruction(
                mem0_read_en=1,
                mem0_addr=source_base + 2,
                westbound_sel=WB_MEM0,
                westbound_consumer_sel=WC_SXM,
                sxm_opcode_input=0x5A5,
            ),
            build_instruction(
                mem0_read_en=1,
                mem0_addr=source_base + 3,
                westbound_sel=WB_MEM0,
                westbound_consumer_sel=WC_SXM,
            ),
            build_instruction(
                westbound_sel=WB_MEM0,
                westbound_consumer_sel=WC_SXM,
            ),
            build_instruction(),
            build_instruction(
                westbound_sel=WB_SXM,
                westbound_consumer_sel=WC_MEM0,
                mem0_write_en=1,
                mem0_addr=target_base + 0,
                sxm_opcode_input=0xA5A,
            ),
            build_instruction(
                westbound_sel=WB_SXM,
                westbound_consumer_sel=WC_MEM0,
                mem0_write_en=1,
                mem0_addr=target_base + 1,
            ),
            build_instruction(
                westbound_sel=WB_SXM,
                westbound_consumer_sel=WC_MEM0,
                mem0_write_en=1,
                mem0_addr=target_base + 2,
            ),
            build_instruction(
                westbound_sel=WB_SXM,
                westbound_consumer_sel=WC_MEM0,
                mem0_write_en=1,
                mem0_addr=target_base + 3,
            ),
        ]

    return [
        build_instruction(mem0_read_en=1, mem0_addr=source_base + 0),
        build_instruction(
            mem0_read_en=1,
            mem0_addr=source_base + 1,
            eastbound_sel=EB_MEM0,
            eastbound_consumer_sel=EC_SXM,
        ),
        build_instruction(
            mem0_read_en=1,
            mem0_addr=source_base + 2,
            eastbound_sel=EB_MEM0,
            eastbound_consumer_sel=EC_SXM,
            sxm_opcode_input=0x5A5,
        ),
        build_instruction(
            mem0_read_en=1,
            mem0_addr=source_base + 3,
            eastbound_sel=EB_MEM0,
            eastbound_consumer_sel=EC_SXM,
        ),
        build_instruction(
            eastbound_sel=EB_MEM0,
            eastbound_consumer_sel=EC_SXM,
        ),
        build_instruction(),
        build_instruction(
            eastbound_sel=EB_SXM,
            eastbound_consumer_sel=EC_MEM0,
            mem0_write_en=1,
            mem0_addr=target_base + 0,
            sxm_opcode_input=0xA5A,
        ),
        build_instruction(
            eastbound_sel=EB_SXM,
            eastbound_consumer_sel=EC_MEM0,
            mem0_write_en=1,
            mem0_addr=target_base + 1,
        ),
        build_instruction(
            eastbound_sel=EB_SXM,
            eastbound_consumer_sel=EC_MEM0,
            mem0_write_en=1,
            mem0_addr=target_base + 2,
        ),
        build_instruction(
            eastbound_sel=EB_SXM,
            eastbound_consumer_sel=EC_MEM0,
            mem0_write_en=1,
            mem0_addr=target_base + 3,
        ),
    ]


def build_scores_softmax_program(*, q_base: int, kt_base: int, scores_base: int, probs_base: int) -> list[int]:
    # Current assumption:
    # - q_base points at MEM0 entries already arranged in the column-major format
    #   MXM expects on its input side.
    # - kt_base points at MEM1 entries already arranged as row-wise K^T vectors.
    program: list[int] = []

    for k_idx in range(4):
        program.extend(
            [
                build_instruction(
                    mem1_read_en=1,
                    mem1_addr=kt_base + k_idx,
                    westbound_sel=WB_MEM1,
                    westbound_consumer_sel=WC_MXM,
                    mxm_ingress_mode=INGRESS_WGHT,
                    mxm_input_is_signed=1,
                    mxm_wght_is_signed=1,
                ),
                build_instruction(
                    westbound_sel=WB_MEM1,
                    westbound_consumer_sel=WC_MXM,
                    mxm_ingress_mode=INGRESS_WGHT,
                    mxm_input_is_signed=1,
                    mxm_wght_is_signed=1,
                ),
                build_instruction(
                    mem0_read_en=1,
                    mem0_addr=q_base + k_idx,
                    westbound_sel=WB_MEM0,
                    westbound_consumer_sel=WC_MXM,
                    mxm_ingress_mode=INGRESS_INPUT,
                    mxm_input_is_signed=1,
                    mxm_wght_is_signed=1,
                ),
                build_instruction(
                    westbound_sel=WB_MEM0,
                    westbound_consumer_sel=WC_MXM,
                    mxm_ingress_mode=INGRESS_INPUT,
                    mxm_input_is_signed=1,
                    mxm_wght_is_signed=1,
                ),
                build_instruction(mxm_start=1, mxm_input_is_signed=1, mxm_wght_is_signed=1),
                build_instruction(mxm_start=1, mxm_input_is_signed=1, mxm_wght_is_signed=1),
            ]
        )

    program.extend([build_instruction(), build_instruction()])

    for row_idx in range(4):
        program.append(
            build_instruction(
                eastbound_sel=EB_MXM,
                eastbound_consumer_sel=EC_MEM0,
                mem0_write_en=1,
                mem0_addr=scores_base + row_idx,
                mxm_e_row_sel=row_idx,
                mxm_e_valid_in=1,
            )
        )
        program.append(
            build_instruction(
                eastbound_sel=EB_MXM,
                eastbound_consumer_sel=EC_VXM,
                mxm_e_row_sel=row_idx,
                mxm_e_valid_in=1,
                vxm_ctrl=0b1100,
                vxm_data_sel=1,
            )
        )
        program.append(
            build_instruction(
                vxm_ctrl=0b1100,
                vxm_data_sel=1,
            )
        )
        for _ in range(10):
            program.append(build_instruction())
        for _ in range(4):
            program.append(
                build_instruction(
                    eastbound_sel=EB_VXM,
                    eastbound_consumer_sel=EC_MEM0,
                    mem0_write_en=1,
                    mem0_addr=probs_base + row_idx,
                )
            )

    return program


def build_output_program(*, probs_base: int, v_base: int, output_base: int) -> list[int]:
    # Current assumption:
    # - probs_base points at MEM0 entries already arranged in the column-major
    #   format expected by MXM.
    # - v_base points at MEM1 entries already arranged as row-wise V_q vectors.
    program: list[int] = []

    for k_idx in range(4):
        program.extend(
            [
                build_instruction(
                    mem1_read_en=1,
                    mem1_addr=v_base + k_idx,
                    westbound_sel=WB_MEM1,
                    westbound_consumer_sel=WC_MXM,
                    mxm_ingress_mode=INGRESS_WGHT,
                    mxm_input_is_signed=0,
                    mxm_wght_is_signed=1,
                ),
                build_instruction(
                    westbound_sel=WB_MEM1,
                    westbound_consumer_sel=WC_MXM,
                    mxm_ingress_mode=INGRESS_WGHT,
                    mxm_input_is_signed=0,
                    mxm_wght_is_signed=1,
                ),
                build_instruction(
                    mem0_read_en=1,
                    mem0_addr=probs_base + k_idx,
                    westbound_sel=WB_MEM0,
                    westbound_consumer_sel=WC_MXM,
                    mxm_ingress_mode=INGRESS_INPUT,
                    mxm_input_is_signed=0,
                    mxm_wght_is_signed=1,
                ),
                build_instruction(
                    westbound_sel=WB_MEM0,
                    westbound_consumer_sel=WC_MXM,
                    mxm_ingress_mode=INGRESS_INPUT,
                    mxm_input_is_signed=0,
                    mxm_wght_is_signed=1,
                ),
                build_instruction(mxm_start=1, mxm_input_is_signed=0, mxm_wght_is_signed=1),
                build_instruction(mxm_start=1, mxm_input_is_signed=0, mxm_wght_is_signed=1),
            ]
        )

    program.extend([build_instruction(), build_instruction()])

    for row_idx in range(4):
        program.append(
            build_instruction(
                eastbound_sel=EB_MXM,
                eastbound_consumer_sel=EC_MEM0,
                mem0_write_en=1,
                mem0_addr=output_base + row_idx,
                mxm_e_row_sel=row_idx,
                mxm_e_valid_in=1,
            )
        )

    return program


async def run_program(dut, program: list[int], *, settle_cycles: int = 8) -> None:
    preload_program(dut, program)
    await reset_dut(dut)
    await tick(dut, len(program) + settle_cycles)


async def run_q_projection_phase(dut, fixture: SelfAttentionFixture, addr: AttentionAddressMap) -> None:
    preload_matrix_into_mem0_columns(dut, base_addr=addr.x_mem0_base, matrix=fixture.x_matrix)
    preload_matrix_into_mem1_rows(dut, base_addr=addr.wq_mem1_base, matrix=fixture.w_q)
    await run_program(
        dut,
        build_projection_program(
            mem0_base=addr.x_mem0_base,
            mem1_base=addr.wq_mem1_base,
            target_base=addr.q_quant_mem0_base,
        ),
    )


async def run_k_projection_phase(dut, fixture: SelfAttentionFixture, addr: AttentionAddressMap) -> None:
    preload_matrix_into_mem0_columns(dut, base_addr=addr.x_mem0_base, matrix=fixture.x_matrix)
    preload_matrix_into_mem1_rows(dut, base_addr=addr.wk_mem1_base, matrix=fixture.w_k)
    await run_program(
        dut,
        build_projection_program(
            mem0_base=addr.x_mem0_base,
            mem1_base=addr.wk_mem1_base,
            target_base=addr.k_quant_mem0_base,
        ),
    )


async def run_v_projection_phase(dut, fixture: SelfAttentionFixture, addr: AttentionAddressMap) -> None:
    preload_matrix_into_mem0_columns(dut, base_addr=addr.x_mem0_base, matrix=fixture.x_matrix)
    preload_matrix_into_mem1_rows(dut, base_addr=addr.wv_mem1_base, matrix=fixture.w_v)
    await run_program(
        dut,
        build_projection_program(
            mem0_base=addr.x_mem0_base,
            mem1_base=addr.wv_mem1_base,
            target_base=addr.v_quant_mem0_base,
        ),
    )


async def run_k_transpose_phase(dut, addr: AttentionAddressMap) -> None:
    await run_program(
        dut,
        build_sxm_transpose_program(
            source_base=addr.k_quant_mem0_base,
            target_base=addr.kt_quant_mem0_base,
            use_westbound=True,
        ),
    )


async def run_scores_softmax_phase(dut, addr: AttentionAddressMap) -> None:
    await run_program(
        dut,
        build_scores_softmax_program(
            q_base=addr.q_quant_mem0_base,
            kt_base=addr.kt_quant_mem0_base,
            scores_base=addr.scores_mem0_base,
            probs_base=addr.probs_mem0_base,
        ),
    )


async def run_output_phase(dut, addr: AttentionAddressMap) -> None:
    await run_program(
        dut,
        build_output_program(
            probs_base=addr.probs_mem0_base,
            v_base=addr.v_quant_mem0_base,
            output_base=addr.output_mem0_base,
        ),
    )


@cocotb.test()
async def test_lpu_self_attention_forward_int8(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    fixture = SelfAttentionFixture(
        x_matrix=[
            [45, 12, -88, 3],
            [10, 115, 34, -52],
            [-76, 22, 95, 14],
            [61, -5, 18, 103],
        ],
        w_q=[
            [24, -11, 38, 12],
            [15, 52, -20, 64],
            [-31, 27, 41, -19],
            [22, -44, 13, 35],
        ],
        w_k=[
            [13, 29, -15, 51],
            [-22, 41, 33, 11],
            [17, -35, 52, 24],
            [44, 10, -26, 48],
        ],
        w_v=[
            [55, 31, -21, 18],
            [-12, 63, 44, 27],
            [28, -19, 71, 36],
            [33, 45, -11, 59],
        ],
    )
    addr = AttentionAddressMap()
    golden = self_attention_golden(
        fixture.x_matrix,
        fixture.w_q,
        fixture.w_k,
        fixture.w_v,
    )

    await run_q_projection_phase(dut, fixture, addr)
    q_rows = read_packed_q8_rows_from_mem0(dut, base_addr=addr.q_quant_mem0_base)
    q_scales = read_row_scales_from_mem0(dut, base_addr=addr.q_quant_mem0_base)
    assert q_rows == golden["q_q8"], f"Q_q mismatch: got={q_rows} expected={golden['q_q8']}"
    assert q_scales == golden["q_scales"], f"Q scale mismatch: got={q_scales} expected={golden['q_scales']}"
    preload_matrix_into_mem0_columns(dut, base_addr=addr.q_quant_mem0_base, matrix=q_rows)

    await run_k_projection_phase(dut, fixture, addr)
    k_rows = read_packed_q8_rows_from_mem0(dut, base_addr=addr.k_quant_mem0_base)
    k_scales = read_row_scales_from_mem0(dut, base_addr=addr.k_quant_mem0_base)
    assert k_rows == golden["k_q8"], f"K_q mismatch: got={k_rows} expected={golden['k_q8']}"
    assert k_scales == golden["k_scales"], f"K scale mismatch: got={k_scales} expected={golden['k_scales']}"

    await run_k_transpose_phase(dut, addr)
    kt_rows = read_packed_q8_rows_from_mem0(dut, base_addr=addr.kt_quant_mem0_base)
    assert kt_rows == golden["k_q8_t"], f"K_q^T mismatch: got={kt_rows} expected={golden['k_q8_t']}"
    preload_matrix_into_mem1_rows(dut, base_addr=addr.kt_quant_mem0_base, matrix=kt_rows)

    await run_v_projection_phase(dut, fixture, addr)
    v_rows = read_packed_q8_rows_from_mem0(dut, base_addr=addr.v_quant_mem0_base)
    v_scales = read_row_scales_from_mem0(dut, base_addr=addr.v_quant_mem0_base)
    assert v_rows == golden["v_q8"], f"V_q mismatch: got={v_rows} expected={golden['v_q8']}"
    assert v_scales == golden["v_scales"], f"V scale mismatch: got={v_scales} expected={golden['v_scales']}"
    preload_matrix_into_mem1_rows(dut, base_addr=addr.v_quant_mem0_base, matrix=v_rows)

    await run_scores_softmax_phase(dut, addr)
    score_rows = read_full_int32_rows_from_mem0(dut, base_addr=addr.scores_mem0_base)
    assert score_rows == golden["scores_int32"], (
        f"scores mismatch: got={score_rows} expected={golden['scores_int32']}"
    )
    probs_rows = read_packed_u8_rows_from_mem0(dut, base_addr=addr.probs_mem0_base)
    assert probs_rows == golden["probs_q8"], (
        f"probabilities mismatch: got={probs_rows} expected={golden['probs_q8']}"
    )
    preload_matrix_into_mem0_columns(dut, base_addr=addr.probs_mem0_base, matrix=probs_rows)

    await run_output_phase(dut, addr)
    output_rows = read_full_int32_rows_from_mem0(dut, base_addr=addr.output_mem0_base)
    assert output_rows == golden["output_int32"], (
        f"output mismatch: got={output_rows} expected={golden['output_int32']}"
    )

    assert int(dut.vxm_input_overflow_dbg.value) == 0, "VXM input FIFO overflowed unexpectedly"
