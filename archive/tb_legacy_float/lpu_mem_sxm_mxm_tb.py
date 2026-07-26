import struct

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


WB_NONE = 0
WB_SXM = 1
WB_MEM0 = 2
WB_MEM1 = 4

WC_NONE = 0
WC_MXM = 1
WC_SXM = 2

INGRESS_NONE = 0
INGRESS_INPUT = 1
INGRESS_WGHT = 2
MXM_SIZE = 8
INPUT_SCALE = -2
WEIGHT_SCALE = -3
OUTPUT_SCALE = INPUT_SCALE + WEIGHT_SCALE

A_RAW = [
    [1, 2, -1, 0, 3, -2, 1, 4],
    [0, -1, 2, 1, -3, 2, 5, -1],
    [3, 0, 1, -2, 2, 1, -1, 0],
    [-2, 4, 0, 3, 1, -1, 2, 1],
    [1, -3, 2, 2, 0, 4, -2, 3],
    [2, 1, -4, 0, 1, 3, 2, -2],
    [0, 2, 3, -1, 4, -3, 1, 2],
    [-1, 0, 2, 5, -2, 1, 3, -4],
]

B_RAW = [
    [2, 0, -1, 3, 1, -2, 4, 1],
    [-1, 3, 2, 0, 1, 1, -2, 2],
    [0, -2, 1, 4, -3, 2, 1, -1],
    [3, 1, 0, -2, 2, -1, 1, 4],
    [1, -1, 3, 2, 0, 4, -3, 2],
    [-2, 2, -1, 1, 3, 0, 2, -4],
    [4, -3, 2, -1, 1, 3, 0, 1],
    [1, 2, -2, 3, -1, 1, 4, 0],
]


def pack_bytes(values):
    word = 0
    for idx, value in enumerate(values):
        word |= (value & 0xFF) << (8 * idx)
    return word


def pack_mem_row(values, scale=0):
    return pack_bytes(values) | ((scale & 0xFF) << 64)


def signed_value(handle):
    return int(handle.value.to_signed())


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def fp8_e5m2_bits(value: float) -> int:
    table = {
        0.0: 0x00,
        0.5: 0x38,
        1.0: 0x3C,
        2.0: 0x40,
        4.0: 0x44,
        8.0: 0x48,
        -0.5: 0xB8,
        -1.0: 0xBC,
        -2.0: 0xC0,
        -4.0: 0xC4,
        -8.0: 0xC8,
    }
    if value not in table:
        raise ValueError(f"unsupported FP8 test value {value}")
    return table[value]


def _set_field(word, value, lsb, width):
    mask = (1 << width) - 1
    return word | ((value & mask) << lsb)


MEM_ADDR_W = 15
OP_TRANSPOSE_LOAD = 0x5A5
OP_TRANSPOSE_EMIT = 0xA5A


def _set_mem_addr(word, value, lsb):
    if value < 0 or value >= (1 << MEM_ADDR_W):
        raise ValueError(f"memory address {value} does not fit in {MEM_ADDR_W} bits")
    return _set_field(word, value, lsb, MEM_ADDR_W)


def build_instruction(
    *,
    westbound_sel=WB_NONE,
    westbound_consumer_sel=WC_NONE,
    mem0_read_en=0,
    mem0_addr=0,
    mem1_read_en=0,
    mem1_addr=0,
    sxm_opcode_input=0,
    sxm_transpose_load=0,
    sxm_transpose_emit=0,
    sxm_load_from_west=0,
    mxm_ingress_mode=INGRESS_NONE,
    mxm_start=0,
    mxm_clear=0,
    mxm_input_is_signed=1,
    mxm_wght_is_signed=1,
    mxm_use_fp=0,
):
    word = 0
    if sxm_opcode_input == OP_TRANSPOSE_LOAD:
        sxm_transpose_load = 1
    elif sxm_opcode_input == OP_TRANSPOSE_EMIT:
        sxm_transpose_emit = 1

    word = _set_field(word, westbound_sel, 0, 3)
    word = _set_field(word, westbound_consumer_sel, 6, 3)
    word = _set_field(word, mem0_read_en, 12, 1)
    word = _set_mem_addr(word, mem0_addr, 14)
    word = _set_field(word, mem1_read_en, 29, 1)
    word = _set_mem_addr(word, mem1_addr, 31)
    word = _set_field(word, mxm_ingress_mode, 46, 2)
    word = _set_field(word, mxm_start, 48, 1)
    word = _set_field(word, mxm_clear, 49, 1)
    word = _set_field(word, mxm_input_is_signed, 57, 1)
    word = _set_field(word, mxm_wght_is_signed, 58, 1)
    word = _set_field(word, mxm_use_fp, 59, 1)
    word = _set_field(word, sxm_transpose_load, 63, 1)
    word = _set_field(word, sxm_transpose_emit, 64, 1)
    word = _set_field(word, sxm_load_from_west, 65, 1)
    return word


async def tick(dut, n=1):
    for _ in range(n):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")


async def reset_dut(dut):
    dut.rst_n.value = 0
    await tick(dut, 2)
    dut.rst_n.value = 1
    await Timer(1, unit="ns")


def preload_instruction(dut, pc, instruction_word):
    dut.u_lpu.u_icu.imem_array[pc].value = instruction_word


def preload_mem0_word(dut, addr, values, scale=0):
    dut.u_lpu.u_mem0.sram_array[addr].value = pack_mem_row(values, scale)


def preload_mem1_word(dut, addr, values, scale=0):
    dut.u_lpu.u_mem1.sram_array[addr].value = pack_mem_row(values, scale)


def preload_program(dut, instructions, *, trailing_nops=32):
    for pc, instruction_word in enumerate(instructions):
        preload_instruction(dut, pc, instruction_word)
    for pc in range(len(instructions), len(instructions) + trailing_nops):
        preload_instruction(dut, pc, build_instruction())


def append_sxm_transpose_load_from_mem1(program, *, source_base):
    program.append(build_instruction(mem1_read_en=1, mem1_addr=source_base + 0))
    program.append(build_instruction(
        mem1_read_en=1,
        mem1_addr=source_base + 1,
        westbound_sel=WB_MEM1,
        westbound_consumer_sel=WC_SXM,
    ))
    program.append(build_instruction(
        mem1_read_en=1,
        mem1_addr=source_base + 2,
        westbound_sel=WB_MEM1,
        westbound_consumer_sel=WC_SXM,
        sxm_opcode_input=0x5A5,
    ))
    for r in range(3, 8):
        program.append(build_instruction(
            mem1_read_en=1,
            mem1_addr=source_base + r,
            westbound_sel=WB_MEM1,
            westbound_consumer_sel=WC_SXM,
        ))
    program.append(build_instruction(
        westbound_sel=WB_MEM1,
        westbound_consumer_sel=WC_SXM,
    ))
    program.append(build_instruction())


def read_mxm_matrix(dut):
    return [
        [signed_value(getattr(dut, f"mxm_out_{r}{c}_dbg")) for c in range(8)]
        for r in range(8)
    ]


def matmul_expected(a, b):
    return [
        [
            sum(a[row][k] * b[k][col] for k in range(MXM_SIZE))
            for col in range(MXM_SIZE)
        ]
        for row in range(MXM_SIZE)
    ]


def format_int_matrix(matrix):
    return "\n".join("    " + " ".join(f"{value:5d}" for value in row) for row in matrix)


def format_decimal_matrix(matrix, scale):
    return "\n".join(
        "    " + " ".join(f"{value * (2.0 ** scale):8.5f}" for value in row)
        for row in matrix
    )


def append_mem1_to_mxm_outer_product(program, *, input_addr, weight_addr):
    program.append(build_instruction(mem1_read_en=1, mem1_addr=input_addr))
    program.append(build_instruction(
        mem1_read_en=1,
        mem1_addr=weight_addr,
        westbound_sel=WB_MEM1,
        westbound_consumer_sel=WC_MXM,
        mxm_ingress_mode=INGRESS_INPUT,
    ))
    program.append(build_instruction(
        westbound_sel=WB_MEM1,
        westbound_consumer_sel=WC_MXM,
        mxm_ingress_mode=INGRESS_WGHT,
    ))
    program.extend([
        build_instruction(mxm_start=1),
        build_instruction(),
        build_instruction(),
    ])


@cocotb.test()
async def test_lpu_mem1_westbound_mxm_8x8_matmul(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    input_base = 64
    weight_base = 128

    for k in range(MXM_SIZE):
        a_col = [A_RAW[row][k] for row in range(MXM_SIZE)]
        preload_mem1_word(dut, input_base + k, a_col, scale=INPUT_SCALE)
        preload_mem1_word(dut, weight_base + k, B_RAW[k], scale=WEIGHT_SCALE)

    program = [build_instruction(mxm_clear=1)]
    for k in range(MXM_SIZE):
        append_mem1_to_mxm_outer_product(
            program,
            input_addr=input_base + k,
            weight_addr=weight_base + k,
        )
    preload_program(dut, program)

    await reset_dut(dut)
    await tick(dut, len(program) + 16)

    observed = read_mxm_matrix(dut)
    expected = matmul_expected(A_RAW, B_RAW)
    observed_scale = int(dut.mxm_out_scale_dbg.value.to_signed())

    dut._log.info("A matrix loaded from MEM1 as columns:\n%s", format_int_matrix(A_RAW))
    dut._log.info(
        "A decimal matrix, scale=%d:\n%s",
        INPUT_SCALE,
        format_decimal_matrix(A_RAW, INPUT_SCALE),
    )
    dut._log.info("B matrix loaded from MEM1 as rows:\n%s", format_int_matrix(B_RAW))
    dut._log.info(
        "B decimal matrix, scale=%d:\n%s",
        WEIGHT_SCALE,
        format_decimal_matrix(B_RAW, WEIGHT_SCALE),
    )
    dut._log.info("Expected A*B:\n%s", format_int_matrix(expected))
    dut._log.info("Observed MXM A*B:\n%s", format_int_matrix(observed))
    dut._log.info(
        "Observed MXM decimal A*B, scale=%d:\n%s",
        observed_scale,
        format_decimal_matrix(observed, observed_scale),
    )

    assert int(dut.input_loaded_dbg.value) == 1
    assert int(dut.wght_loaded_dbg.value) == 1
    assert observed_scale == OUTPUT_SCALE
    assert observed == expected, f"MEM1->westbound->MXM matmul mismatch: got={observed} expected={expected}"


@cocotb.test()
async def test_lpu_mem_sxm_mxm_qkt_single_k_slice(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    q_column0 = [3, -2, 5, 1, 0, 0, 0, 0]
    k_column0 = [7, 4, -3, 2, 0, 0, 0, 0]

    preload_mem0_word(dut, addr=0, values=q_column0)
    for row_idx, lane0_value in enumerate(k_column0):
        preload_mem1_word(dut, addr=row_idx, values=[lane0_value] + [0]*7)

    program = [
        build_instruction(mxm_clear=1),
        build_instruction(
            mem0_read_en=1,
            mem0_addr=0,
            westbound_sel=WB_MEM0,
            westbound_consumer_sel=WC_MXM,
            mxm_ingress_mode=INGRESS_INPUT,
        ),
        build_instruction(
            westbound_sel=WB_MEM0,
            westbound_consumer_sel=WC_MXM,
            mxm_ingress_mode=INGRESS_INPUT,
        ),
    ]
    append_sxm_transpose_load_from_mem1(program, source_base=0)
    program.extend(
        [
            build_instruction(
                westbound_sel=WB_SXM,
                westbound_consumer_sel=WC_MXM,
                mxm_ingress_mode=INGRESS_WGHT,
                sxm_opcode_input=0xA5A,
            ),
            build_instruction(mxm_start=1),
            build_instruction(),
            build_instruction(),
        ]
    )

    preload_program(dut, program)

    await reset_dut(dut)
    await tick(dut, len(program) + 12)

    assert int(dut.input_loaded_dbg.value) == 1
    assert int(dut.wght_loaded_dbg.value) == 1

    expected_scores = [
        [q_value * k_value for k_value in k_column0]
        for q_value in q_column0
    ]
    observed_scores = read_mxm_matrix(dut)
    assert observed_scores == expected_scores, (
        f"mem->sxm->mxm mismatch: got={observed_scores} expected={expected_scores}"
    )


@cocotb.test(skip=True)
async def test_lpu_mem_sxm_mxm_qkt_single_k_slice_fp(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    q_column0 = [1.0, -0.5, 2.0, 0.5, 0.0, 0.0, 0.0, 0.0]
    k_column0 = [2.0, 0.5, -1.0, 4.0, 0.0, 0.0, 0.0, 0.0]

    preload_mem0_word(dut, addr=0, values=[fp8_e5m2_bits(v) for v in q_column0])
    for row_idx, lane0_value in enumerate(k_column0):
        preload_mem1_word(dut, addr=row_idx, values=[fp8_e5m2_bits(lane0_value)] + [0]*7)

    fp_cfg = dict(mxm_use_fp=1, mxm_input_is_signed=0, mxm_wght_is_signed=0)
    program = [
        build_instruction(mxm_clear=1, **fp_cfg),
        build_instruction(
            mem0_read_en=1,
            mem0_addr=0,
            westbound_sel=WB_MEM0,
            westbound_consumer_sel=WC_MXM,
            mxm_ingress_mode=INGRESS_INPUT,
            **fp_cfg,
        ),
        build_instruction(
            westbound_sel=WB_MEM0,
            westbound_consumer_sel=WC_MXM,
            mxm_ingress_mode=INGRESS_INPUT,
            **fp_cfg,
        ),
    ]
    append_sxm_transpose_load_from_mem1(program, source_base=0)
    program.extend(
        [
            build_instruction(
                westbound_sel=WB_SXM,
                westbound_consumer_sel=WC_MXM,
                mxm_ingress_mode=INGRESS_WGHT,
                sxm_opcode_input=0xA5A,
                **fp_cfg,
            ),
            build_instruction(mxm_start=1, **fp_cfg),
            build_instruction(**fp_cfg),
            build_instruction(**fp_cfg),
            build_instruction(**fp_cfg),
            build_instruction(**fp_cfg),
            build_instruction(**fp_cfg),
        ]
    )

    preload_program(dut, program)

    await reset_dut(dut)
    await tick(dut, len(program) + 12)

    expected = [
        [q_value * k_value for k_value in k_column0]
        for q_value in q_column0
    ]
    observed_bits = [
        [int(getattr(dut, f"mxm_out_{r}{c}_dbg").value) & 0xFFFFFFFF for c in range(8)]
        for r in range(8)
    ]

    for r in range(8):
        for c in range(8):
            exp_bits = f32_bits(expected[r][c])
            got_bits = observed_bits[r][c]
            assert got_bits == exp_bits, (
                f"FP mem->sxm->mxm mismatch at ({r}, {c}): got 0x{got_bits:08x}, "
                f"expected 0x{exp_bits:08x}"
            )
