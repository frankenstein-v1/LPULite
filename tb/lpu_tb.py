import math
import struct

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


WB_NONE = 0
WB_SXM = 1
WB_MEM0 = 2
WB_VXM = 3
WB_MEM1 = 4

EB_NONE = 0
EB_MXM = 1
EB_SXM = 2
EB_MEM0 = 3
EB_VXM = 4

WC_NONE = 0
WC_MXM = 1
WC_SXM = 2
WC_MEM0 = 3
WC_VXM = 4

EC_NONE = 0
EC_SXM = 1
EC_MEM0 = 2
EC_VXM = 3
EC_MEM1 = 4

INGRESS_NONE = 0
INGRESS_INPUT = 1
INGRESS_WGHT = 2


#pack 4 bytes into a word 
def pack_bytes(values):
    word = 0
    for idx, value in enumerate(values):
        word |= (value & 0xFF) << (8 * idx)
    return word


def pack_mxm_row(values):
    row = 0
    for idx, value in enumerate(values):
        row |= (value & 0xFFFFFFFF) << (32 * idx)
    return row


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def bits_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFF_FFFF))[0]


def to_f32(value: float) -> float:
    return bits_to_f32(f32_bits(value))


def fp8_e5m2_bits(value: float) -> int:
    bits = f32_bits(value)
    sign = (bits >> 31) & 0x1
    exp = (bits >> 23) & 0xFF
    frac = bits & 0x7FFFFF

    if exp == 0 and frac == 0:
        return sign << 7
    if exp == 0xFF:
        return ((sign << 7) | 0x7D) if frac else ((sign << 7) | 0x7C)
    if exp == 0:
        return sign << 7

    fp8_exp = exp - 127 + 15
    if fp8_exp <= 0:
        return sign << 7

    mantissa_full = (1 << 23) | frac
    mantissa_q = (mantissa_full >> 21) & 0x7
    guard = (mantissa_full >> 20) & 0x1
    sticky = mantissa_full & ((1 << 20) - 1)
    if guard and (sticky or (mantissa_q & 0x1)):
        mantissa_q += 1
    if mantissa_q == 8:
        mantissa_q = 4
        fp8_exp += 1
    if fp8_exp >= 31:
        return (sign << 7) | 0x7C
    return (sign << 7) | ((fp8_exp & 0x1F) << 2) | (mantissa_q & 0x3)


def fp8_e5m2_to_f32(bits: int) -> float:
    sign = -1.0 if (bits & 0x80) else 1.0
    exp = (bits >> 2) & 0x1F
    frac = bits & 0x3

    if exp == 0:
        if frac == 0:
            return -0.0 if sign < 0 else 0.0
        return to_f32(sign * math.ldexp(frac / 4.0, -14))
    if exp == 0x1F:
        return math.nan if frac else (math.inf if sign > 0 else -math.inf)

    mantissa = 1.0 + (frac / 4.0)
    return to_f32(sign * math.ldexp(mantissa, exp - 15))

#read.  signed vector
def signed_value(handle):
    return int(handle.value.to_signed())


def clip_unsigned_q8(value):
    if value <= 0:
        return 0
    if value >= 255:
        return 255
    return value & 0xFF


def lut_softmax_exp_expected(q_value):
    ln2 = 177
    z_value = (-q_value) // ln2
    p_value = q_value + (z_value * ln2)
    lut_addr = -p_value
    if lut_addr < 0 or lut_addr > 177:
        return 0
    lut_value = round((2.718281828459045 ** (-lut_addr / 256.0)) * 256.0)
    return lut_value >> z_value


def softmax_expected(lanes):
    lane_max = max(lanes)
    lane_sub = [lane - lane_max for lane in lanes]
    lane_exp = [lut_softmax_exp_expected(lane) for lane in lane_sub]
    sum_exp = sum(lane_exp)
    quotient = (1 << 30) // sum_exp
    shift = 30 - 8
    return [(quotient * lane) >> shift for lane in lane_exp]


def scale_softmax_quant_expected(lanes):
    scaled_lanes = [lane >> 1 for lane in lanes]
    softmax_lanes = softmax_expected(scaled_lanes)
    return [clip_unsigned_q8(lane) for lane in softmax_lanes]


def fp32_to_q8_8_ref(fp_bits):
    sign_bit = (fp_bits >> 31) & 0x1
    exp_bits = (fp_bits >> 23) & 0xFF
    frac_bits = fp_bits & 0x7FFFFF

    if exp_bits == 0 and frac_bits == 0:
        return 0
    if exp_bits == 0xFF:
        return -0x8000_0000 if sign_bit else 0x7FFF_FFFF

    if exp_bits == 0:
        significand = frac_bits
        exp_unbiased = -126
    else:
        significand = (1 << 23) | frac_bits
        exp_unbiased = exp_bits - 127

    shift_amount = exp_unbiased - 23 + 8
    scaled_value = significand
    if shift_amount >= 0:
        scaled_value = 0x7FFF_FFFF if shift_amount > 30 else (scaled_value << shift_amount)
    else:
        scaled_value = 0 if -shift_amount > 62 else (scaled_value >> (-shift_amount))

    if sign_bit:
        scaled_value = -scaled_value

    if scaled_value > 0x7FFF_FFFF:
        return 0x7FFF_FFFF
    if scaled_value < -0x8000_0000:
        return -0x8000_0000
    return scaled_value


def uq8_8_to_fp32_ref(fixed_value):
    if fixed_value == 0:
        return 0

    msb_idx = max(idx for idx in range(32) if (fixed_value >> idx) & 0x1)
    exponent_bits = msb_idx + 119
    if msb_idx <= 23:
        normalized = fixed_value << (23 - msb_idx)
    else:
        normalized = fixed_value >> (msb_idx - 23)
    return ((exponent_bits & 0xFF) << 23) | (normalized & 0x7FFFFF)


def softmax_fp8_quant_expected(data_floats):
    lane_max = to_f32(max(data_floats))
    delta_bits = [f32_bits(to_f32(value - lane_max)) for value in data_floats]
    exp_bits = [
        uq8_8_to_fp32_ref(lut_softmax_exp_expected(fp32_to_q8_8_ref(bits)))
        for bits in delta_bits
    ]
    exp_values = [bits_to_f32(bits) for bits in exp_bits]
    sum01 = to_f32(exp_values[0] + exp_values[1])
    sum23 = to_f32(exp_values[2] + exp_values[3])
    sum_exp = to_f32(sum01 + sum23)
    prob_floats = [to_f32(value / sum_exp) for value in exp_values]
    return [fp8_e5m2_bits(value) for value in prob_floats]


def _set_field(word, value, lsb, width):
    mask = (1 << width) - 1
    return word | ((value & mask) << lsb)

#builds the instruction for the ICU
def build_instruction(
    *,
    westbound_sel=WB_NONE,
    eastbound_sel=EB_NONE,
    westbound_consumer_sel=WC_NONE,
    eastbound_consumer_sel=EC_NONE,
    mem0_read_en=0,
    mem0_write_en=0,
    mem0_addr=0,
    mem1_read_en=0,
    mem1_write_en=0,
    mem1_addr=0,
    sxm_opcode_input=0,
    sxm_opcode_weight=0,
    vxm_ctrl=0,
    vxm_data_sel=0,
    mxm_ingress_mode=INGRESS_NONE,
    mxm_start=0,
    mxm_clear=0,
    mxm_e_row_sel=0,
    mxm_e_col_sel=0,
    mxm_e_valid_in=0,
    mxm_input_is_signed=1,
    mxm_wght_is_signed=1,
    mxm_use_fp=0,
    fp_quant_mode=0,
    mem_store_fmt=0,
):
    
    word = 0

    # bus control [11:0]
    word = _set_field(word, westbound_sel, 0, 3)
    word = _set_field(word, eastbound_sel, 3, 3)
    word = _set_field(word, westbound_consumer_sel, 6, 3)
    word = _set_field(word, eastbound_consumer_sel, 9, 3)

    # mem0 control [22:12]
    word = _set_field(word, mem0_read_en, 12, 1)
    word = _set_field(word, mem0_write_en, 13, 1)
    word = _set_field(word, mem0_addr, 14, 9)

    # mem1 control [33:23]
    word = _set_field(word, mem1_read_en, 23, 1)
    word = _set_field(word, mem1_write_en, 24, 1)
    word = _set_field(word, mem1_addr, 25, 9)

    # sxm control [57:34]
    word = _set_field(word, sxm_opcode_input, 34, 12)
    word = _set_field(word, sxm_opcode_weight, 46, 12)

    # vxm control [76:71]
    word = _set_field(word, vxm_ctrl, 71, 4)
    word = _set_field(word, vxm_data_sel, 76, 1)

    # mxm control [70:62]
    word = _set_field(word, mxm_ingress_mode, 62, 2)
    word = _set_field(word, mxm_start, 64, 1)
    word = _set_field(word, mxm_clear, 65, 1)
    word = _set_field(word, mxm_e_row_sel, 66, 2)
    word = _set_field(word, mxm_e_col_sel, 68, 2)
    word = _set_field(word, mxm_e_valid_in, 70, 1)
    word = _set_field(word, mxm_input_is_signed, 77, 1)
    word = _set_field(word, mxm_wght_is_signed, 78, 1)
    word = _set_field(word, mxm_use_fp, 79, 1)
    word = _set_field(word, fp_quant_mode, 80, 1)
    word = _set_field(word, mem_store_fmt, 81, 2)

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
    """Write one 96-bit word directly into the ICU instruction memory."""
    dut.u_lpu.u_icu.imem_array[pc].value = instruction_word


def preload_mem0_word(dut, addr, values):
    """Write one packed 32-bit word directly into MEM0 SRAM."""
    dut.u_lpu.u_mem0.sram_array[addr].value = pack_bytes(values)


def preload_mem1_word(dut, addr, values):
    """Write one packed 32-bit word directly into MEM1 SRAM."""
    dut.u_lpu.u_mem1.sram_array[addr].value = pack_bytes(values)


def preload_program(dut, instructions, *, trailing_nops=64):
    for pc, instruction_word in enumerate(instructions):
        preload_instruction(dut, pc, instruction_word)
    for pc in range(len(instructions), len(instructions) + trailing_nops):
        preload_instruction(dut, pc, build_instruction())


def matmul_expected(a_matrix, b_matrix):
    expected = [[0 for _ in range(4)] for _ in range(4)]
    for row in range(4):
        for col in range(4):
            for k_idx in range(4):
                expected[row][col] += a_matrix[row][k_idx] * b_matrix[k_idx][col]
    return expected


def matmul_expected_fp32(a_matrix, b_matrix):
    expected = [[0.0 for _ in range(4)] for _ in range(4)]
    for row in range(4):
        for col in range(4):
            acc = 0.0
            for k_idx in range(4):
                product = to_f32(to_f32(a_matrix[row][k_idx]) * to_f32(b_matrix[k_idx][col]))
                acc = to_f32(acc + product)
            expected[row][col] = acc
    return expected


def transpose_matrix(matrix):
    return [[matrix[row][col] for row in range(4)] for col in range(4)]


def read_mxm_matrix(dut):
    return [
        [
            signed_value(dut.mxm_out_00_dbg),
            signed_value(dut.mxm_out_01_dbg),
            signed_value(dut.mxm_out_02_dbg),
            signed_value(dut.mxm_out_03_dbg),
        ],
        [
            signed_value(dut.mxm_out_10_dbg),
            signed_value(dut.mxm_out_11_dbg),
            signed_value(dut.mxm_out_12_dbg),
            signed_value(dut.mxm_out_13_dbg),
        ],
        [
            signed_value(dut.mxm_out_20_dbg),
            signed_value(dut.mxm_out_21_dbg),
            signed_value(dut.mxm_out_22_dbg),
            signed_value(dut.mxm_out_23_dbg),
        ],
        [
            signed_value(dut.mxm_out_30_dbg),
            signed_value(dut.mxm_out_31_dbg),
            signed_value(dut.mxm_out_32_dbg),
            signed_value(dut.mxm_out_33_dbg),
        ],
    ]


def read_mxm_matrix_bits(dut):
    return [
        [
            int(dut.mxm_out_00_dbg.value) & 0xFFFFFFFF,
            int(dut.mxm_out_01_dbg.value) & 0xFFFFFFFF,
            int(dut.mxm_out_02_dbg.value) & 0xFFFFFFFF,
            int(dut.mxm_out_03_dbg.value) & 0xFFFFFFFF,
        ],
        [
            int(dut.mxm_out_10_dbg.value) & 0xFFFFFFFF,
            int(dut.mxm_out_11_dbg.value) & 0xFFFFFFFF,
            int(dut.mxm_out_12_dbg.value) & 0xFFFFFFFF,
            int(dut.mxm_out_13_dbg.value) & 0xFFFFFFFF,
        ],
        [
            int(dut.mxm_out_20_dbg.value) & 0xFFFFFFFF,
            int(dut.mxm_out_21_dbg.value) & 0xFFFFFFFF,
            int(dut.mxm_out_22_dbg.value) & 0xFFFFFFFF,
            int(dut.mxm_out_23_dbg.value) & 0xFFFFFFFF,
        ],
        [
            int(dut.mxm_out_30_dbg.value) & 0xFFFFFFFF,
            int(dut.mxm_out_31_dbg.value) & 0xFFFFFFFF,
            int(dut.mxm_out_32_dbg.value) & 0xFFFFFFFF,
            int(dut.mxm_out_33_dbg.value) & 0xFFFFFFFF,
        ],
    ]


def append_mxm_k_slice(program, *, k_idx):
    program.extend(
        [
            build_instruction(
                mem1_read_en=1,
                mem1_addr=k_idx,
                westbound_sel=WB_MEM1,
                westbound_consumer_sel=WC_MXM,
                mxm_ingress_mode=INGRESS_WGHT,
            ),
            build_instruction(
                westbound_sel=WB_MEM1,
                westbound_consumer_sel=WC_MXM,
                mxm_ingress_mode=INGRESS_WGHT,
            ),
            build_instruction(
                mem0_read_en=1,
                mem0_addr=k_idx,
                westbound_sel=WB_MEM0,
                westbound_consumer_sel=WC_MXM,
                mxm_ingress_mode=INGRESS_INPUT,
            ),
            build_instruction(
                westbound_sel=WB_MEM0,
                westbound_consumer_sel=WC_MXM,
                mxm_ingress_mode=INGRESS_INPUT,
            ),
            build_instruction(mxm_start=1),
            build_instruction(mxm_start=1),
        ]
    )


def append_mxm_weight_row_load_from_mem1(program, *, addr):
    program.extend(
        [
            build_instruction(
                mem1_read_en=1,
                mem1_addr=addr,
                westbound_sel=WB_MEM1,
                westbound_consumer_sel=WC_MXM,
                mxm_ingress_mode=INGRESS_WGHT,
                mxm_use_fp=1,
                mxm_input_is_signed=0,
                mxm_wght_is_signed=0,
            ),
            build_instruction(
                westbound_sel=WB_MEM1,
                westbound_consumer_sel=WC_MXM,
                mxm_ingress_mode=INGRESS_WGHT,
                mxm_use_fp=1,
                mxm_input_is_signed=0,
                mxm_wght_is_signed=0,
            ),
        ]
    )


def append_mxm_input_column_load_from_mem0(program, *, addr):
    program.extend(
        [
            build_instruction(
                mem0_read_en=1,
                mem0_addr=addr,
                westbound_sel=WB_MEM0,
                westbound_consumer_sel=WC_MXM,
                mxm_ingress_mode=INGRESS_INPUT,
                mxm_use_fp=1,
                mxm_input_is_signed=0,
                mxm_wght_is_signed=0,
            ),
            build_instruction(
                westbound_sel=WB_MEM0,
                westbound_consumer_sel=WC_MXM,
                mxm_ingress_mode=INGRESS_INPUT,
                mxm_use_fp=1,
                mxm_input_is_signed=0,
                mxm_wght_is_signed=0,
            ),
        ]
    )


def append_sxm_transpose_load_from_mem1(program, *, source_base):
    # Match the working westbound transpose schedule used elsewhere, but source
    # rows from MEM1 so SXM can build one K tile in-place.
    program.extend(
        [
            build_instruction(mem1_read_en=1, mem1_addr=source_base + 0),
            build_instruction(
                mem1_read_en=1,
                mem1_addr=source_base + 1,
                westbound_sel=WB_MEM1,
                westbound_consumer_sel=WC_SXM,
            ),
            build_instruction(
                mem1_read_en=1,
                mem1_addr=source_base + 2,
                westbound_sel=WB_MEM1,
                westbound_consumer_sel=WC_SXM,
                sxm_opcode_input=0x5A5,
            ),
            build_instruction(
                mem1_read_en=1,
                mem1_addr=source_base + 3,
                westbound_sel=WB_MEM1,
                westbound_consumer_sel=WC_SXM,
            ),
            build_instruction(
                westbound_sel=WB_MEM1,
                westbound_consumer_sel=WC_SXM,
            ),
            build_instruction(),
        ]
    )


def append_sxm_transpose_load_from_mem0(program, *, source_base):
    program.extend(
        [
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
        ]
    )


def append_sxm_emit_capture_to_mxm(program, *, col_idx, ingress_mode):
    if col_idx == 0:
        program.append(
            build_instruction(
                westbound_sel=WB_SXM,
                westbound_consumer_sel=WC_MXM,
                mxm_ingress_mode=ingress_mode,
                sxm_opcode_input=0xA5A,
                mxm_use_fp=1,
                mxm_input_is_signed=0,
                mxm_wght_is_signed=0,
            )
        )
    else:
        program.append(
            build_instruction(
                sxm_opcode_input=0xA5A,
                mxm_use_fp=1,
                mxm_input_is_signed=0,
                mxm_wght_is_signed=0,
            )
        )
        for _ in range(col_idx - 1):
            program.append(
                build_instruction(
                    mxm_use_fp=1,
                    mxm_input_is_signed=0,
                    mxm_wght_is_signed=0,
                )
            )
        program.append(
            build_instruction(
                westbound_sel=WB_SXM,
                westbound_consumer_sel=WC_MXM,
                mxm_ingress_mode=ingress_mode,
                mxm_use_fp=1,
                mxm_input_is_signed=0,
                mxm_wght_is_signed=0,
            )
        )


def append_vxm_row_store(program, *, row_idx, target_addr, wait_cycles=10, store_cycles=4, fp_quant_mode=0):
    program.append(
        build_instruction(
            eastbound_sel=EB_MXM,
            eastbound_consumer_sel=EC_VXM,
            mxm_e_row_sel=row_idx,
            mxm_e_valid_in=1,
            vxm_ctrl=0b1100,
            vxm_data_sel=1,
            fp_quant_mode=fp_quant_mode,
        )
    )
    program.append(
        build_instruction(
            vxm_ctrl=0b1100,
            vxm_data_sel=1,
            fp_quant_mode=fp_quant_mode,
        )
    )
    for _ in range(wait_cycles):
        program.append(build_instruction(fp_quant_mode=fp_quant_mode))
    for _ in range(store_cycles):
        program.append(
            build_instruction(
                eastbound_sel=EB_VXM,
                eastbound_consumer_sel=EC_MEM0,
                mem0_write_en=1,
                mem0_addr=target_addr,
                fp_quant_mode=fp_quant_mode,
            )
        )


@cocotb.test()
async def test_lpu_wrapper_smoke(dut):
    """Minimal smoke test: clock, reset, and wrapper visibility."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    await reset_dut(dut)

    assert int(dut.pc_dbg.value) >= 0


@cocotb.test()
async def test_lpu_minimal_program_skeleton(dut):
    
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    weights = [3, 0, 0, 0]
    inputs = [2, 0, 0, 0]

    preload_mem1_word(dut, addr=0, values=weights)
    preload_mem0_word(dut, addr=0, values=inputs)

    preload_instruction(
        dut,
        0,
        build_instruction(
            mem1_read_en=1,
            mem1_addr=0,
            westbound_sel=WB_MEM1,
            westbound_consumer_sel=WC_MXM,
            mxm_ingress_mode=INGRESS_WGHT,
        ),
    )

    preload_instruction(
        dut,
        1,
        build_instruction(
            westbound_sel=WB_MEM1,
            westbound_consumer_sel=WC_MXM,
            mxm_ingress_mode=INGRESS_WGHT
        ),
    )

    preload_instruction(
        dut,
        2,
        build_instruction(
            mem0_read_en=1,
            mem0_addr=0, 
            westbound_sel=WB_MEM0,
            westbound_consumer_sel=WC_MXM,
            mxm_ingress_mode= INGRESS_INPUT
        ),
    )

    preload_instruction(
        dut, 
        3,
        build_instruction(
            westbound_sel=WB_MEM0,
            westbound_consumer_sel=WC_MXM, 
            mxm_ingress_mode=INGRESS_INPUT
        ),
    )

    preload_instruction(
        dut, 4, 
        build_instruction(
            mxm_start=1
        ),
    )

    preload_instruction(
        dut, 5,
        build_instruction(
            mxm_start=1
        ),
    )

    preload_instruction(dut, 6, build_instruction())

    preload_instruction(dut, 7, build_instruction())



    await reset_dut(dut)

    # asertions about instruction 0 
    assert int(dut.pc_dbg.value) == 0
    assert int(dut.westbound_sel_dbg.value) == WB_MEM1
    assert int(dut.westbound_consumer_sel_dbg.value) == WC_MXM
    assert int(dut.mxm_ingress_mode_dbg.value) == INGRESS_WGHT

    await tick(dut, 2)

    assert int(dut.wght_loaded_dbg.value) == 1
    assert signed_value(dut.wght_buf0) == 3
    assert signed_value(dut.wght_buf1) == 0
    assert signed_value(dut.wght_buf2) == 0
    assert signed_value(dut.wght_buf3) == 0

    await tick(dut, 1)

    # mem0 input-load instruction phase
    assert int(dut.westbound_sel_dbg.value) == WB_MEM0
    assert int(dut.westbound_consumer_sel_dbg.value) == WC_MXM
    assert int(dut.mxm_ingress_mode_dbg.value) == INGRESS_INPUT

    await tick(dut, 1)

    #check if the mem0 values r correct
    assert int(dut.input_loaded_dbg.value) == 1
    assert signed_value(dut.input_buf0) == 2
    assert signed_value(dut.input_buf1) == 0
    assert signed_value(dut.input_buf2) == 0
    assert signed_value(dut.input_buf3) == 0

    await tick(dut, 1)

    await tick(dut, 2)

    assert signed_value(dut.mxm_out_00_dbg) == 6


@cocotb.test()
async def test_lpu_mxm_to_vxm_softmax_quant_to_mem0(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    weights = [3, 0, 0, 0]
    inputs = [2, 0, 0, 0]
    expected_mxm_row = [6, 0, 0, 0]
    expected_mem0_word = pack_bytes(scale_softmax_quant_expected(expected_mxm_row))
    target_addr = 1

    preload_mem1_word(dut, addr=0, values=weights)
    preload_mem0_word(dut, addr=0, values=inputs)

    program = [
        build_instruction(
            mem1_read_en=1,
            mem1_addr=0,
            westbound_sel=WB_MEM1,
            westbound_consumer_sel=WC_MXM,
            mxm_ingress_mode=INGRESS_WGHT,
        ),
        build_instruction(
            westbound_sel=WB_MEM1,
            westbound_consumer_sel=WC_MXM,
            mxm_ingress_mode=INGRESS_WGHT,
        ),
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
        build_instruction(mxm_start=1),
        build_instruction(mxm_start=1),
        build_instruction(),
        build_instruction(),
        build_instruction(),
        build_instruction(),
        build_instruction(
            eastbound_sel=EB_MXM,
            eastbound_consumer_sel=EC_VXM,
            mxm_e_row_sel=0,
            mxm_e_valid_in=1,
            vxm_ctrl=0b1100,
            vxm_data_sel=1,
        ),
        build_instruction(
            vxm_ctrl=0b1100,
            vxm_data_sel=1,
        ),
        build_instruction(),
        build_instruction(),
        build_instruction(),
        build_instruction(),
        build_instruction(),
        build_instruction(),
        build_instruction(),
        build_instruction(),
        build_instruction(),
        build_instruction(),
        build_instruction(),
        build_instruction(),
        build_instruction(),
        build_instruction(),
        build_instruction(),
        build_instruction(
            eastbound_sel=EB_VXM,
            eastbound_consumer_sel=EC_MEM0,
            mem0_write_en=1,
            mem0_addr=target_addr,
        ),
        build_instruction(
            eastbound_sel=EB_VXM,
            eastbound_consumer_sel=EC_MEM0,
            mem0_write_en=1,
            mem0_addr=target_addr,
        ),
        build_instruction(
            eastbound_sel=EB_VXM,
            eastbound_consumer_sel=EC_MEM0,
            mem0_write_en=1,
            mem0_addr=target_addr,
        ),
        build_instruction(
            eastbound_sel=EB_VXM,
            eastbound_consumer_sel=EC_MEM0,
            mem0_write_en=1,
            mem0_addr=target_addr,
        ),
        build_instruction(
            eastbound_sel=EB_VXM,
            eastbound_consumer_sel=EC_MEM0,
            mem0_write_en=1,
            mem0_addr=target_addr,
        ),
        build_instruction(
            eastbound_sel=EB_VXM,
            eastbound_consumer_sel=EC_MEM0,
            mem0_write_en=1,
            mem0_addr=target_addr,
        ),
        build_instruction(
            eastbound_sel=EB_VXM,
            eastbound_consumer_sel=EC_MEM0,
            mem0_write_en=1,
            mem0_addr=target_addr,
        ),
        build_instruction(
            eastbound_sel=EB_VXM,
            eastbound_consumer_sel=EC_MEM0,
            mem0_write_en=1,
            mem0_addr=target_addr,
        ),
    ]
    preload_program(dut, program)

    await reset_dut(dut)
    await tick(dut, 40)

    observed_word = int(dut.u_lpu.u_mem0.sram_array[target_addr].value)
    assert observed_word == expected_mem0_word, (
        f"MEM0[{target_addr}] mismatch: got 0x{observed_word:08x}, "
        f"expected 0x{expected_mem0_word:08x}"
    )
    assert int(dut.vxm_input_overflow_dbg.value) == 0, "VXM input FIFO overflowed unexpectedly"


@cocotb.test()
async def test_lpu_mxm_row_store_to_mem0_full_width(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    weights = [3, 0, 0, 0]
    inputs = [2, 0, 0, 0]
    target_addr = 8
    expected_row = [6, 0, 0, 0]
    expected_row_word = pack_mxm_row(expected_row)

    preload_mem1_word(dut, addr=0, values=weights)
    preload_mem0_word(dut, addr=0, values=inputs)

    program = [
        build_instruction(
            mem1_read_en=1,
            mem1_addr=0,
            westbound_sel=WB_MEM1,
            westbound_consumer_sel=WC_MXM,
            mxm_ingress_mode=INGRESS_WGHT,
        ),
        build_instruction(
            westbound_sel=WB_MEM1,
            westbound_consumer_sel=WC_MXM,
            mxm_ingress_mode=INGRESS_WGHT,
        ),
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
        build_instruction(mxm_start=1),
        build_instruction(mxm_start=1),
        build_instruction(),
        build_instruction(),
        build_instruction(
            eastbound_sel=EB_MXM,
            eastbound_consumer_sel=EC_MEM0,
            mem0_write_en=1,
            mem0_addr=target_addr,
            mxm_e_row_sel=0,
            mxm_e_valid_in=1,
        ),
        build_instruction(),
    ]
    preload_program(dut, program)

    await reset_dut(dut)
    await tick(dut, len(program) + 4)

    observed_row_word = int(dut.u_lpu.u_mem0.sram_array[target_addr].value)
    assert observed_row_word == expected_row_word, (
        f"MEM0[{target_addr}] full-row mismatch: got 0x{observed_row_word:032x}, "
        f"expected 0x{expected_row_word:032x}"
    )


@cocotb.test()
async def test_lpu_full_mxm_to_vxm_softmax_quant_rows_to_mem0(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    a_matrix = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [2, 0, 1, 3],
        [4, 1, 0, 2],
    ]
    b_matrix = [
        [1, 0, 2, 1],
        [0, 1, 1, 0],
        [3, 1, 0, 2],
        [2, 1, 1, 1],
    ]
    expected_mxm = matmul_expected(a_matrix, b_matrix)
    mem0_store_base = 16

    for k_idx in range(4):
        preload_mem0_word(
            dut,
            addr=k_idx,
            values=[a_matrix[row][k_idx] for row in range(4)],
        )
        preload_mem1_word(
            dut,
            addr=k_idx,
            values=[b_matrix[k_idx][col] for col in range(4)],
        )

    program = []
    for k_idx in range(4):
        append_mxm_k_slice(program, k_idx=k_idx)

    program.extend([build_instruction(), build_instruction()])

    for row_idx in range(4):
        append_vxm_row_store(
            program,
            row_idx=row_idx,
            target_addr=mem0_store_base + row_idx,
        )

    preload_program(dut, program)

    await reset_dut(dut)
    await tick(dut, len(program) + 8)

    observed_mxm = read_mxm_matrix(dut)
    assert observed_mxm == expected_mxm, (
        f"MXM 4x4 mismatch: got={observed_mxm} expected={expected_mxm}"
    )

    for row_idx, expected_row in enumerate(expected_mxm):
        observed_word = int(dut.u_lpu.u_mem0.sram_array[mem0_store_base + row_idx].value)
        expected_word = pack_bytes(scale_softmax_quant_expected(expected_row))
        assert observed_word == expected_word, (
            f"MEM0[{mem0_store_base + row_idx}] mismatch: got 0x{observed_word:08x}, "
            f"expected 0x{expected_word:08x} for row {row_idx}"
        )

    assert int(dut.vxm_input_overflow_dbg.value) == 0, "VXM input FIFO overflowed unexpectedly"


@cocotb.test()
async def test_lpu_fp8_mxm_mem0_mem1_to_fp32_matrix(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    a = [
        [1.0, 0.5, -1.0, 2.0],
        [-0.5, 1.0, 2.0, -1.0],
        [2.0, -1.0, 0.5, 1.0],
        [1.0, 2.0, -0.5, 0.5],
    ]
    b = [
        [2.0, -1.0, 0.5, 4.0],
        [0.5, 4.0, -2.0, 1.0],
        [-1.0, 2.0, 1.0, -0.5],
        [4.0, 0.5, -1.0, 2.0],
    ]

    for k_idx in range(4):
        preload_mem0_word(
            dut,
            addr=k_idx,
            values=[fp8_e5m2_bits(a[row][k_idx]) for row in range(4)],
        )
        preload_mem1_word(
            dut,
            addr=k_idx,
            values=[fp8_e5m2_bits(b[k_idx][col]) for col in range(4)],
        )

    program = [build_instruction(mxm_clear=1, mxm_use_fp=1, mxm_input_is_signed=0, mxm_wght_is_signed=0)]
    for k_idx in range(4):
        program.extend(
            [
                build_instruction(
                    mem1_read_en=1,
                    mem1_addr=k_idx,
                    westbound_sel=WB_MEM1,
                    westbound_consumer_sel=WC_MXM,
                    mxm_ingress_mode=INGRESS_WGHT,
                    mxm_use_fp=1,
                    mxm_input_is_signed=0,
                    mxm_wght_is_signed=0,
                ),
                build_instruction(
                    westbound_sel=WB_MEM1,
                    westbound_consumer_sel=WC_MXM,
                    mxm_ingress_mode=INGRESS_WGHT,
                    mxm_use_fp=1,
                    mxm_input_is_signed=0,
                    mxm_wght_is_signed=0,
                ),
                build_instruction(
                    mem0_read_en=1,
                    mem0_addr=k_idx,
                    westbound_sel=WB_MEM0,
                    westbound_consumer_sel=WC_MXM,
                    mxm_ingress_mode=INGRESS_INPUT,
                    mxm_use_fp=1,
                    mxm_input_is_signed=0,
                    mxm_wght_is_signed=0,
                ),
                build_instruction(
                    westbound_sel=WB_MEM0,
                    westbound_consumer_sel=WC_MXM,
                    mxm_ingress_mode=INGRESS_INPUT,
                    mxm_use_fp=1,
                    mxm_input_is_signed=0,
                    mxm_wght_is_signed=0,
                ),
                build_instruction(
                    mxm_start=1,
                    mxm_use_fp=1,
                    mxm_input_is_signed=0,
                    mxm_wght_is_signed=0,
                ),
            ]
        )
        for _ in range(4):
            program.append(
                build_instruction(
                    mxm_use_fp=1,
                    mxm_input_is_signed=0,
                    mxm_wght_is_signed=0,
                )
            )

    preload_program(dut, program)

    await reset_dut(dut)
    await tick(dut, len(program) + 8)

    expected = matmul_expected(a, b)
    observed_bits = [
        [int(dut.mxm_out_00_dbg.value) & 0xFFFFFFFF, int(dut.mxm_out_01_dbg.value) & 0xFFFFFFFF,
         int(dut.mxm_out_02_dbg.value) & 0xFFFFFFFF, int(dut.mxm_out_03_dbg.value) & 0xFFFFFFFF],
        [int(dut.mxm_out_10_dbg.value) & 0xFFFFFFFF, int(dut.mxm_out_11_dbg.value) & 0xFFFFFFFF,
         int(dut.mxm_out_12_dbg.value) & 0xFFFFFFFF, int(dut.mxm_out_13_dbg.value) & 0xFFFFFFFF],
        [int(dut.mxm_out_20_dbg.value) & 0xFFFFFFFF, int(dut.mxm_out_21_dbg.value) & 0xFFFFFFFF,
         int(dut.mxm_out_22_dbg.value) & 0xFFFFFFFF, int(dut.mxm_out_23_dbg.value) & 0xFFFFFFFF],
        [int(dut.mxm_out_30_dbg.value) & 0xFFFFFFFF, int(dut.mxm_out_31_dbg.value) & 0xFFFFFFFF,
         int(dut.mxm_out_32_dbg.value) & 0xFFFFFFFF, int(dut.mxm_out_33_dbg.value) & 0xFFFFFFFF],
    ]

    for r in range(4):
        for c in range(4):
            exp_bits = f32_bits(expected[r][c])
            got_bits = observed_bits[r][c]
            assert got_bits == exp_bits, (
                f"FP LPU MXM mismatch at ({r}, {c}): got 0x{got_bits:08x}, "
                f"expected 0x{exp_bits:08x}"
            )


@cocotb.test()
async def test_lpu_regular_self_attention(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    q_matrix = [
        [0.35, -0.72, 1.18, 0.49],
        [1.41, -0.58, 0.27, -1.33],
        [-0.91, 0.63, 0.44, 1.57],
        [0.82, -1.26, -0.37, 0.95],
    ]
    k_matrix = [
        [0.68, -0.44, 1.29, 0.53],
        [-1.12, 0.31, 0.77, -0.69],
        [0.26, 1.46, -0.57, 0.88],
        [0.91, -0.38, 0.42, 1.21],
    ]
    v_matrix = [
        [0.57, -1.08, 0.93, 0.24],
        [1.34, 0.41, -0.62, 0.78],
        [-0.49, 1.12, 0.36, -1.27],
        [0.85, -0.33, 1.49, 0.68],
    ]

    q_base = 0
    k_base = 0
    v_base = 16
    softmax_base = 32

    q_bits = [[fp8_e5m2_bits(value) for value in row] for row in q_matrix]
    k_bits = [[fp8_e5m2_bits(value) for value in row] for row in k_matrix]
    v_bits = [[fp8_e5m2_bits(value) for value in row] for row in v_matrix]

    q_fp = [[fp8_e5m2_to_f32(bits) for bits in row] for row in q_bits]
    k_fp = [[fp8_e5m2_to_f32(bits) for bits in row] for row in k_bits]
    v_fp = [[fp8_e5m2_to_f32(bits) for bits in row] for row in v_bits]

    for k_idx in range(4):
        preload_mem0_word(
            dut,
            addr=q_base + k_idx,
            values=[q_bits[row][k_idx] for row in range(4)],
        )
    for row_idx in range(4):
        preload_mem1_word(dut, addr=k_base + row_idx, values=k_bits[row_idx])
        preload_mem1_word(dut, addr=v_base + row_idx, values=v_bits[row_idx])

    score_matrix = matmul_expected_fp32(q_fp, transpose_matrix(k_fp))
    scaled_score_matrix = [
        [to_f32(score * 0.5) for score in score_row]
        for score_row in score_matrix
    ]
    softmax_rows_bits = [softmax_fp8_quant_expected(score_row) for score_row in scaled_score_matrix]
    softmax_rows_fp = [
        [fp8_e5m2_to_f32(bits) for bits in row_bits]
        for row_bits in softmax_rows_bits
    ]
    attention_out = matmul_expected_fp32(softmax_rows_fp, v_fp)

    fp_ctrl = dict(mxm_use_fp=1, mxm_input_is_signed=0, mxm_wght_is_signed=0)
    program = [build_instruction(mxm_clear=1, **fp_ctrl)]

    append_sxm_transpose_load_from_mem1(program, source_base=k_base)
    for k_idx in range(4):
        append_mxm_input_column_load_from_mem0(program, addr=q_base + k_idx)
        append_sxm_emit_capture_to_mxm(program, col_idx=k_idx, ingress_mode=INGRESS_WGHT)
        program.append(build_instruction(mxm_start=1, **fp_ctrl))
        for _ in range(4):
            program.append(build_instruction(**fp_ctrl))

    program.extend([build_instruction(**fp_ctrl), build_instruction(**fp_ctrl)])

    for row_idx in range(4):
        append_vxm_row_store(
            program,
            row_idx=row_idx,
            target_addr=softmax_base + row_idx,
            wait_cycles=36,
            store_cycles=8,
            fp_quant_mode=1,
        )

    program.append(build_instruction(mxm_clear=1, **fp_ctrl))
    append_sxm_transpose_load_from_mem0(program, source_base=softmax_base)
    for k_idx in range(4):
        append_sxm_emit_capture_to_mxm(program, col_idx=k_idx, ingress_mode=INGRESS_INPUT)
        append_mxm_weight_row_load_from_mem1(program, addr=v_base + k_idx)
        program.append(build_instruction(mxm_start=1, **fp_ctrl))
        for _ in range(4):
            program.append(build_instruction(**fp_ctrl))

    preload_program(dut, program)

    await reset_dut(dut)
    await tick(dut, len(program) + 32)

    for row_idx, expected_row in enumerate(softmax_rows_bits):
        observed_word = int(dut.u_lpu.u_mem0.sram_array[softmax_base + row_idx].value) & 0xFFFF_FFFF
        expected_word = pack_bytes(expected_row)
        assert observed_word == expected_word, (
            f"softmax row {row_idx} mismatch: got 0x{observed_word:08x}, "
            f"expected 0x{expected_word:08x}"
        )

    observed_bits = read_mxm_matrix_bits(dut)
    for row_idx in range(4):
        for col_idx in range(4):
            expected_bits = f32_bits(attention_out[row_idx][col_idx])
            assert observed_bits[row_idx][col_idx] == expected_bits, (
                f"regular FP self-attention mismatch at ({row_idx}, {col_idx}): "
                f"got 0x{observed_bits[row_idx][col_idx]:08x}, "
                f"expected 0x{expected_bits:08x}"
            )


@cocotb.test()
async def test_lpu_mem_sxm_mxm_qkt_single_k_slice(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    # Use a single active k-slice so the expected score matrix is still Q @ K^T,
    # but only the first SXM-emitted transpose row needs to reach MXM.
    q_column0 = [3, -2, 5, 1]
    k_column0 = [7, 4, -3, 2]

    preload_mem0_word(dut, addr=0, values=q_column0)
    for row_idx, lane0_value in enumerate(k_column0):
        preload_mem1_word(dut, addr=row_idx, values=[lane0_value, 0, 0, 0])

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
            build_instruction(mxm_start=1),
            build_instruction(),
            build_instruction(),
        ]
    )

    preload_program(dut, program)

    await reset_dut(dut)
    await tick(dut, len(program) + 6)

    assert int(dut.input_loaded_dbg.value) == 1, "Q slice should remain loaded in the MXM input ingress"
    assert int(dut.wght_loaded_dbg.value) == 1, "SXM-emitted K^T slice should load the MXM weight ingress"
    for lane_idx, expected_lane in enumerate(q_column0):
        observed_lane = signed_value(getattr(dut, f"input_buf{lane_idx}"))
        assert observed_lane == expected_lane, f"Q input ingress lane {lane_idx} mismatch"
    for lane_idx, expected_lane in enumerate(k_column0):
        observed_lane = signed_value(getattr(dut, f"wght_buf{lane_idx}"))
        assert observed_lane == expected_lane, f"K^T weight ingress lane {lane_idx} mismatch"

    expected_scores = [
        [q_value * k_value for k_value in k_column0]
        for q_value in q_column0
    ]
    observed_scores = read_mxm_matrix(dut)
    assert observed_scores == expected_scores, (
        f"mem->sxm->mxm Q@K^T single-slice mismatch: got={observed_scores} "
        f"expected={expected_scores}"
    )


async def run_lpu_sxm_transpose_case(dut, *, use_westbound: bool):
    input_matrix = [
        [0x01, 0x02, 0x03, 0x04], # Row 0
        [0x05, 0x06, 0x07, 0x08], # Row 1
        [0x09, 0x0A, 0x0B, 0x0C], # Row 2
        [0x0D, 0x0E, 0x0F, 0x10], # Row 3
    ]
    
    expected_transposed = [
        [0x01, 0x05, 0x09, 0x0D], # Col 0
        [0x02, 0x06, 0x0A, 0x0E], # Col 1
        [0x03, 0x07, 0x0B, 0x0F], # Col 2
        [0x04, 0x08, 0x0C, 0x10], # Col 3
    ]
    
    # Preload inputs into MEM0 at addresses 0..3
    for r in range(4):
        preload_mem0_word(dut, addr=r, values=input_matrix[r])
        
    # Program sequence:
    # PC 0: read MEM0[0]
    # PC 1: read MEM0[1], route MEM0[0] to SXM
    # PC 2: read MEM0[2], route MEM0[1] to SXM, trigger SXM Transpose LOAD
    # PC 3: read MEM0[3], route MEM0[2] to SXM
    # PC 4: route MEM0[3] to SXM
    # PC 5: Idle
    # PC 6: Route SXM to MEM0[10], trigger SXM Transpose EMIT
    # PC 7: Route SXM to MEM0[11]
    # PC 8: Route SXM to MEM0[12]
    # PC 9: Route SXM to MEM0[13]
    if use_westbound:
        program = [
            build_instruction(mem0_read_en=1, mem0_addr=0),
            build_instruction(mem0_read_en=1, mem0_addr=1, westbound_sel=WB_MEM0, westbound_consumer_sel=WC_SXM),
            build_instruction(mem0_read_en=1, mem0_addr=2, westbound_sel=WB_MEM0, westbound_consumer_sel=WC_SXM, sxm_opcode_input=0x5A5),
            build_instruction(mem0_read_en=1, mem0_addr=3, westbound_sel=WB_MEM0, westbound_consumer_sel=WC_SXM),
            build_instruction(westbound_sel=WB_MEM0, westbound_consumer_sel=WC_SXM),
            build_instruction(),
            build_instruction(westbound_sel=WB_SXM, westbound_consumer_sel=WC_MEM0, mem0_write_en=1, mem0_addr=10, sxm_opcode_input=0xA5A),
            build_instruction(westbound_sel=WB_SXM, westbound_consumer_sel=WC_MEM0, mem0_write_en=1, mem0_addr=11),
            build_instruction(westbound_sel=WB_SXM, westbound_consumer_sel=WC_MEM0, mem0_write_en=1, mem0_addr=12),
            build_instruction(westbound_sel=WB_SXM, westbound_consumer_sel=WC_MEM0, mem0_write_en=1, mem0_addr=13),
        ]
    else:
        program = [
            build_instruction(mem0_read_en=1, mem0_addr=0),
            build_instruction(mem0_read_en=1, mem0_addr=1, eastbound_sel=EB_MEM0, eastbound_consumer_sel=EC_SXM),
            build_instruction(mem0_read_en=1, mem0_addr=2, eastbound_sel=EB_MEM0, eastbound_consumer_sel=EC_SXM, sxm_opcode_input=0x5A5),
            build_instruction(mem0_read_en=1, mem0_addr=3, eastbound_sel=EB_MEM0, eastbound_consumer_sel=EC_SXM),
            build_instruction(eastbound_sel=EB_MEM0, eastbound_consumer_sel=EC_SXM),
            build_instruction(),
            build_instruction(eastbound_sel=EB_SXM, eastbound_consumer_sel=EC_MEM0, mem0_write_en=1, mem0_addr=10, sxm_opcode_input=0xA5A),
            build_instruction(eastbound_sel=EB_SXM, eastbound_consumer_sel=EC_MEM0, mem0_write_en=1, mem0_addr=11),
            build_instruction(eastbound_sel=EB_SXM, eastbound_consumer_sel=EC_MEM0, mem0_write_en=1, mem0_addr=12),
            build_instruction(eastbound_sel=EB_SXM, eastbound_consumer_sel=EC_MEM0, mem0_write_en=1, mem0_addr=13),
        ]
    
    preload_program(dut, program)

    await reset_dut(dut)
    await tick(dut, len(program) + 4)

    # Verify that transposed rows are stored in MEM0[10..13]
    for r in range(4):
        observed_word = int(dut.u_lpu.u_mem0.sram_array[10 + r].value)
        expected_word = pack_bytes(expected_transposed[r])
        assert observed_word == expected_word, (
            f"MEM0[10+{r}] transpose mismatch: got 0x{observed_word:08x}, "
            f"expected 0x{expected_word:08x}"
        )
        
    dut._log.info("End-to-end SXM Transpose integration test passed successfully!")


@cocotb.test()
async def test_lpu_sxm_transpose_to_mem0(dut):
    """Verify end-to-end matrix transposition through the eastbound SXM path."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await run_lpu_sxm_transpose_case(dut, use_westbound=False)


@cocotb.test()
async def test_lpu_sxm_transpose_westbound_to_mem0(dut):
    """Verify end-to-end matrix transposition through the westbound SXM path."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await run_lpu_sxm_transpose_case(dut, use_westbound=True)





   

    
