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

VXM_OPERAND_DATA = 0
VXM_OPERAND_BIAS = 1
VXM_OPERAND_GAMMA = 2
VXM_OPERAND_BETA = 3
VXM_OPERAND_ROPE_COS = 4
VXM_OPERAND_ROPE_SIN = 5

VXM_RES_PASS = 0
VXM_RES_CLEAR = 1
VXM_RES_LOAD = 2
VXM_RES_ADD = 3
VXM_RES_EMIT = 4

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


def regular_fp8_row_quant_expected(data_floats):
    absmax = max(abs(value) for value in data_floats)
    if absmax == 0.0:
        scale_exp = 0
    else:
        scale_exp = math.floor(math.log2(absmax))
    scaled = [to_f32(math.ldexp(value, -scale_exp)) for value in data_floats]
    return [fp8_e5m2_bits(value) for value in scaled], scale_exp


def dequantize_regular_fp8_row(row_bits, scale_exp):
    return [to_f32(math.ldexp(fp8_e5m2_to_f32(bits), scale_exp)) for bits in row_bits]


def pack_fp8_row_mem_word(row_bits, scale_exp):
    return pack_bytes(row_bits) | ((scale_exp & 0xFF) << 32)


def unpack_fp8_row_mem_word(word):
    row_bits = [(word >> (8 * idx)) & 0xFF for idx in range(4)]
    scale_exp = (word >> 32) & 0xFF
    if scale_exp & 0x80:
        scale_exp -= 1 << 8
    return row_bits, scale_exp


def layernorm_expected(row, eps=1e-5):
    mean = to_f32(sum(row) / len(row))
    variance = to_f32(sum(to_f32((value - mean) * (value - mean)) for value in row) / len(row))
    inv_std = to_f32(1.0 / math.sqrt(variance + eps))
    return [to_f32(to_f32(value - mean) * inv_std) for value in row]


def drive_layernorm_debug_matrix(dut, matrix, *, valid=1):
    dut.ln_out_valid_dbg.value = valid
    for row_idx in range(4):
        for col_idx in range(4):
            getattr(dut, f"ln_out_{row_idx}{col_idx}_dbg").value = f32_bits(matrix[row_idx][col_idx])


def clear_layernorm_debug_matrix(dut):
    zero_matrix = [[0.0 for _ in range(4)] for _ in range(4)]
    drive_layernorm_debug_matrix(dut, zero_matrix, valid=0)


def _set_field(word, value, lsb, width):
    mask = (1 << width) - 1
    return word | ((value & mask) << lsb)


MEM_ADDR_W = 11
MEM_ADDR_LOW_W = 9
MEM0_ADDR_HI_LSB = 90
MEM1_ADDR_HI_LSB = 92


def _set_mem_addr(word, value, low_lsb, high_lsb):
    if value < 0 or value >= (1 << MEM_ADDR_W):
        raise ValueError(f"memory address {value} does not fit in {MEM_ADDR_W} bits")
    word = _set_field(word, value, low_lsb, MEM_ADDR_LOW_W)
    return _set_field(word, value >> MEM_ADDR_LOW_W, high_lsb, MEM_ADDR_W - MEM_ADDR_LOW_W)

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
    vxm_layernorm_en=0,
    vxm_rope_en=0,
    vxm_residual_op=VXM_RES_PASS,
    vxm_operand_sel=VXM_OPERAND_DATA,
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
    word = _set_mem_addr(word, mem0_addr, 14, MEM0_ADDR_HI_LSB)

    # mem1 control [33:23]
    word = _set_field(word, mem1_read_en, 23, 1)
    word = _set_field(word, mem1_write_en, 24, 1)
    word = _set_mem_addr(word, mem1_addr, 25, MEM1_ADDR_HI_LSB)

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
    word = _set_field(word, vxm_layernorm_en, 83, 1)
    word = _set_field(word, vxm_operand_sel, 84, 3)
    word = _set_field(word, vxm_rope_en, 87, 1)
    word = _set_field(word, vxm_residual_op, 88, 3)

    return word


async def tick(dut, n=1):
    for _ in range(n):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")


async def reset_dut(dut):
    clear_layernorm_debug_matrix(dut)
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


def preload_mem0_raw_word(dut, addr, word):
    """Write one raw MEM0 word, including FP8 row-scale metadata."""
    dut.u_lpu.u_mem0.sram_array[addr].value = word


def preload_mem1_word(dut, addr, values):
    """Write one packed 32-bit word directly into MEM1 SRAM."""
    dut.u_lpu.u_mem1.sram_array[addr].value = pack_bytes(values)


def preload_program(dut, instructions, *, trailing_nops=64):
    for pc, instruction_word in enumerate(instructions):
        preload_instruction(dut, pc, instruction_word)
    for pc in range(len(instructions), len(instructions) + trailing_nops):
        preload_instruction(dut, pc, build_instruction())


async def run_lpu_program(dut, instructions, *, extra_cycles=32):
    preload_program(dut, instructions)
    await reset_dut(dut)
    await tick(dut, len(instructions) + extra_cycles)


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


def append_mxm_weight_row_load_from_mem0(program, *, addr):
    program.extend(
        [
            build_instruction(
                mem0_read_en=1,
                mem0_addr=addr,
                westbound_sel=WB_MEM0,
                westbound_consumer_sel=WC_MXM,
                mxm_ingress_mode=INGRESS_WGHT,
                mxm_use_fp=1,
                mxm_input_is_signed=0,
                mxm_wght_is_signed=0,
            ),
            build_instruction(
                westbound_sel=WB_MEM0,
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


def append_vxm_regular_fp8_row_store(
    program,
    *,
    row_idx,
    target_addr,
    wait_cycles=12,
    store_cycles=8,
    vxm_ctrl=0b0000,
    vxm_layernorm_en=0,
):
    program.append(
        build_instruction(
            eastbound_sel=EB_MXM,
            eastbound_consumer_sel=EC_VXM,
            mxm_e_row_sel=row_idx,
            mxm_e_valid_in=1,
            vxm_ctrl=vxm_ctrl,
            vxm_data_sel=1,
            fp_quant_mode=1,
            vxm_layernorm_en=vxm_layernorm_en,
        )
    )
    program.append(
        build_instruction(
            vxm_ctrl=vxm_ctrl,
            vxm_data_sel=1,
            fp_quant_mode=1,
            vxm_layernorm_en=vxm_layernorm_en,
        )
    )
    for _ in range(wait_cycles):
        program.append(build_instruction(fp_quant_mode=1, vxm_layernorm_en=vxm_layernorm_en))
    for _ in range(store_cycles):
        program.append(
            build_instruction(
                eastbound_sel=EB_VXM,
                eastbound_consumer_sel=EC_MEM0,
                mem0_write_en=1,
                mem0_addr=target_addr,
                fp_quant_mode=1,
                vxm_layernorm_en=vxm_layernorm_en,
            )
        )


def append_fp8_matmul_mem0_cols_mem1_rows_to_mem0(
    program,
    *,
    left_col_base,
    right_row_base,
    output_base,
    output_vxm_ctrl=0b0000,
    vxm_layernorm_en=0,
):
    fp_ctrl = dict(mxm_use_fp=1, mxm_input_is_signed=0, mxm_wght_is_signed=0)
    program.append(build_instruction(mxm_clear=1, **fp_ctrl))
    for k_idx in range(4):
        append_mxm_weight_row_load_from_mem1(program, addr=right_row_base + k_idx)
        append_mxm_input_column_load_from_mem0(program, addr=left_col_base + k_idx)
        program.append(build_instruction(mxm_start=1, **fp_ctrl))
        for _ in range(4):
            program.append(build_instruction(**fp_ctrl))

    program.extend([build_instruction(**fp_ctrl), build_instruction(**fp_ctrl)])

    for row_idx in range(4):
        append_vxm_regular_fp8_row_store(
            program,
            row_idx=row_idx,
            target_addr=output_base + row_idx,
            wait_cycles=16,
            store_cycles=8,
            vxm_ctrl=output_vxm_ctrl,
            vxm_layernorm_en=vxm_layernorm_en,
        )


def append_fp8_matmul_mem0_rows_mem0_rows_to_mem0(
    program,
    *,
    left_row_base,
    right_row_base,
    output_base,
    output_vxm_ctrl=0b0000,
    vxm_layernorm_en=0,
):
    fp_ctrl = dict(mxm_use_fp=1, mxm_input_is_signed=0, mxm_wght_is_signed=0)
    program.append(build_instruction(mxm_clear=1, **fp_ctrl))
    append_sxm_transpose_load_from_mem0(program, source_base=left_row_base)
    for k_idx in range(4):
        append_sxm_emit_capture_to_mxm(program, col_idx=k_idx, ingress_mode=INGRESS_INPUT)
        append_mxm_weight_row_load_from_mem0(program, addr=right_row_base + k_idx)
        program.append(build_instruction(mxm_start=1, **fp_ctrl))
        for _ in range(4):
            program.append(build_instruction(**fp_ctrl))

    program.extend([build_instruction(**fp_ctrl), build_instruction(**fp_ctrl)])

    for row_idx in range(4):
        append_vxm_regular_fp8_row_store(
            program,
            row_idx=row_idx,
            target_addr=output_base + row_idx,
            wait_cycles=16,
            store_cycles=8,
            vxm_ctrl=output_vxm_ctrl,
            vxm_layernorm_en=vxm_layernorm_en,
        )


def quantize_matrix_to_fp8(matrix):
    bits = [[fp8_e5m2_bits(value) for value in row] for row in matrix]
    decoded = [[fp8_e5m2_to_f32(value) for value in row] for row in bits]
    return bits, decoded


def regular_quantize_matrix_for_mem(matrix):
    quantized = [regular_fp8_row_quant_expected(row) for row in matrix]
    row_bits = [bits for bits, _ in quantized]
    scales = [scale for _, scale in quantized]
    raw_fp = [[fp8_e5m2_to_f32(bits) for bits in row] for row in row_bits]
    words = [
        pack_fp8_row_mem_word(row_bits[row_idx], scales[row_idx])
        for row_idx in range(4)
    ]
    return row_bits, scales, raw_fp, words


def preload_mem0_matrix_rows_fp8(dut, base, row_bits, scales=None):
    for row_idx in range(4):
        scale_exp = 0 if scales is None else scales[row_idx]
        preload_mem0_raw_word(
            dut,
            base + row_idx,
            pack_fp8_row_mem_word(row_bits[row_idx], scale_exp),
        )


def read_mem0_quantized_words(dut, base):
    return [
        int(dut.u_lpu.u_mem0.sram_array[base + row_idx].value) & 0xFFFF_FFFF_FFFF_FFFF
        for row_idx in range(4)
    ]


def assert_mem0_words(dut, base, expected_words, label):
    observed_words = read_mem0_quantized_words(dut, base)
    for row_idx, (observed, expected) in enumerate(zip(observed_words, expected_words)):
        assert observed == expected, (
            f"{label} row {row_idx} mismatch: got 0x{observed:016x}, "
            f"expected 0x{expected:016x}"
        )


def format_float_matrix(matrix):
    return "\n".join(
        "    [" + ", ".join(f"{value: .6f}" for value in row) + "]"
        for row in matrix
    )


def toy_lm_head_logits(hidden_row, lm_head):
    return {
        token: to_f32(sum(to_f32(hidden_row[idx] * weight) for idx, weight in enumerate(weights)))
        for token, weights in lm_head.items()
    }


def sorted_logits(logits):
    return sorted(logits.items(), key=lambda item: item[1], reverse=True)


def log_top_tokens(dut, label, logits, top_k=5):
    top = sorted_logits(logits)[:top_k]
    dut._log.info("%s top logits: %s", label, ", ".join(f"{tok}={val:.6f}" for tok, val in top))
    return top[0][0]


def deterministic_weight(label, row, col, scale):
    seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(label))
    raw = math.sin((seed + 37 * row + 101 * col + 17) * 12.9898) * 43758.5453
    frac = raw - math.floor(raw)
    return to_f32((2.0 * frac - 1.0) * scale)


def deterministic_vector(label, scale=0.85):
    return [deterministic_weight(label, 0, col, scale) for col in range(4)]


def xavier_matrix(label, *, residual=0.0):
    matrix = []
    for row in range(4):
        matrix_row = []
        for col in range(4):
            value = deterministic_weight(label, row, col, 0.45)
            if row == col:
                value = to_f32(value + residual)
            matrix_row.append(value)
        matrix.append(matrix_row)
    return matrix


async def run_toy_transformer_tile(
    dut,
    *,
    token_words,
    valid_len,
    prediction_row_idx,
    embedding_table,
    wq_matrix,
    wk_matrix,
    wv_matrix,
    w1_matrix,
    w2_matrix,
    label,
):
    X_ROW_BASE = 0
    WQ_COL_BASE = 0
    WK_COL_BASE = 16
    WV_COL_BASE = 32
    Q_BASE = 64
    K_BASE = 80
    V_BASE = 96
    W1_BASE = 112
    W2_BASE = 128
    P_BASE = 144
    ATTN_BASE = 160
    LN_BASE = 176
    HIDDEN_BASE = 192
    FINAL_BASE = 208

    x_matrix = [embedding_table[token] for token in token_words]

    x_bits, x_fp = quantize_matrix_to_fp8(x_matrix)
    wq_bits, wq_fp = quantize_matrix_to_fp8(wq_matrix)
    wk_bits, wk_fp = quantize_matrix_to_fp8(wk_matrix)
    wv_bits, wv_fp = quantize_matrix_to_fp8(wv_matrix)
    w1_bits, w1_fp = quantize_matrix_to_fp8(w1_matrix)
    w2_bits, w2_fp = quantize_matrix_to_fp8(w2_matrix)

    for row_idx in range(4):
        preload_mem1_word(dut, X_ROW_BASE + row_idx, x_bits[row_idx])
    for col_idx in range(4):
        preload_mem0_word(
            dut,
            WQ_COL_BASE + col_idx,
            [wq_bits[row][col_idx] for row in range(4)],
        )
        preload_mem0_word(
            dut,
            WK_COL_BASE + col_idx,
            [wk_bits[row][col_idx] for row in range(4)],
        )
        preload_mem0_word(
            dut,
            WV_COL_BASE + col_idx,
            [wv_bits[row][col_idx] for row in range(4)],
        )
    preload_mem0_matrix_rows_fp8(dut, W1_BASE, w1_bits)
    preload_mem0_matrix_rows_fp8(dut, W2_BASE, w2_bits)

    q_fp = matmul_expected_fp32(wq_fp, x_fp)
    k_fp = matmul_expected_fp32(wk_fp, x_fp)
    v_fp = matmul_expected_fp32(wv_fp, x_fp)
    _, _, q_raw_fp, q_words = regular_quantize_matrix_for_mem(q_fp)
    _, _, k_raw_fp, k_words = regular_quantize_matrix_for_mem(k_fp)
    _, _, v_raw_fp, v_words = regular_quantize_matrix_for_mem(v_fp)

    cycle_count = 0

    async def run_stage(program, *, extra_cycles=40):
        nonlocal cycle_count
        await run_lpu_program(dut, program, extra_cycles=extra_cycles)
        cycle_count += len(program) + extra_cycles + 2

    for left_base, output_base, expected_words, stage_label in [
        (WQ_COL_BASE, Q_BASE, q_words, f"{label} Q projection"),
        (WK_COL_BASE, K_BASE, k_words, f"{label} K cache projection"),
        (WV_COL_BASE, V_BASE, v_words, f"{label} V cache projection"),
    ]:
        program = []
        append_fp8_matmul_mem0_cols_mem1_rows_to_mem0(
            program,
            left_col_base=left_base,
            right_row_base=X_ROW_BASE,
            output_base=output_base,
        )
        await run_stage(program)
        assert_mem0_words(dut, output_base, expected_words, stage_label)

    score_matrix = matmul_expected_fp32(q_raw_fp, transpose_matrix(k_raw_fp))
    causal_score_matrix = []
    for row_idx, score_row in enumerate(score_matrix):
        max_visible = min(row_idx, valid_len - 1)
        causal_score_matrix.append([
            to_f32(score * 0.5) if col_idx <= max_visible else -1.0e9
            for col_idx, score in enumerate(score_row)
        ])

    p_bits = [softmax_fp8_quant_expected(row) for row in causal_score_matrix]
    p_raw_fp = [[fp8_e5m2_to_f32(bits) for bits in row] for row in p_bits]
    preload_mem0_matrix_rows_fp8(dut, P_BASE, p_bits)

    attention_fp = matmul_expected_fp32(p_raw_fp, v_raw_fp)
    _, _, attn_raw_fp, attn_words = regular_quantize_matrix_for_mem(attention_fp)
    program = []
    append_fp8_matmul_mem0_rows_mem0_rows_to_mem0(
        program,
        left_row_base=P_BASE,
        right_row_base=V_BASE,
        output_base=ATTN_BASE,
    )
    await run_stage(program)
    assert_mem0_words(dut, ATTN_BASE, attn_words, f"{label} causal attention")

    layernorm_fp = [layernorm_expected(row) for row in attention_fp]
    _, _, ln_raw_fp, ln_words = regular_quantize_matrix_for_mem(layernorm_fp)
    program = []
    append_fp8_matmul_mem0_rows_mem0_rows_to_mem0(
        program,
        left_row_base=P_BASE,
        right_row_base=V_BASE,
        output_base=LN_BASE,
        vxm_layernorm_en=1,
    )
    await run_stage(program)
    assert_mem0_words(dut, LN_BASE, ln_words, f"{label} hardware FP32 layernorm")

    hidden_fp = matmul_expected_fp32(ln_raw_fp, w1_fp)
    hidden_relu_fp = [
        [to_f32(value if value > 0.0 else 0.0) for value in row]
        for row in hidden_fp
    ]
    _, _, hidden_raw_fp, hidden_words = regular_quantize_matrix_for_mem(hidden_relu_fp)
    program = []
    append_fp8_matmul_mem0_rows_mem0_rows_to_mem0(
        program,
        left_row_base=LN_BASE,
        right_row_base=W1_BASE,
        output_base=HIDDEN_BASE,
        output_vxm_ctrl=0b0010,
    )
    await run_stage(program)
    assert_mem0_words(dut, HIDDEN_BASE, hidden_words, f"{label} FFN hidden")

    final_fp = matmul_expected_fp32(hidden_raw_fp, w2_fp)
    _, _, final_raw_fp, final_words = regular_quantize_matrix_for_mem(final_fp)
    program = []
    append_fp8_matmul_mem0_rows_mem0_rows_to_mem0(
        program,
        left_row_base=HIDDEN_BASE,
        right_row_base=W2_BASE,
        output_base=FINAL_BASE,
    )
    await run_stage(program)
    assert_mem0_words(dut, FINAL_BASE, final_words, f"{label} FFN output")

    drive_layernorm_debug_matrix(dut, layernorm_fp)

    dut._log.info("%s tokens: %s", label, " ".join(token_words))
    dut._log.info("%s input embedding matrix:\n%s", label, format_float_matrix(x_fp))
    dut._log.info("%s K cache decoded matrix:\n%s", label, format_float_matrix(k_raw_fp))
    dut._log.info("%s V cache decoded matrix:\n%s", label, format_float_matrix(v_raw_fp))
    dut._log.info("%s causal attention output matrix:\n%s", label, format_float_matrix(attn_raw_fp))
    dut._log.info("%s layernorm matrix:\n%s", label, format_float_matrix(ln_raw_fp))
    dut._log.info("%s final FFN output matrix:\n%s", label, format_float_matrix(final_raw_fp))
    dut._log.info(
        "%s prediction row %d hidden: [%s]",
        label,
        prediction_row_idx,
        ", ".join(f"{value:.6f}" for value in final_raw_fp[prediction_row_idx]),
    )
    dut._log.info("%s approximate LPU cycles across staged programs: %d", label, cycle_count)

    return {
        "x_matrix": x_fp,
        "k_cache": k_raw_fp,
        "v_cache": v_raw_fp,
        "attention": attn_raw_fp,
        "layernorm": ln_raw_fp,
        "final": final_raw_fp,
        "prediction_hidden": final_raw_fp[prediction_row_idx],
        "cycles": cycle_count,
    }


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
async def test_lpu_prefill_causal_attention_ffn_mem0(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    token_words = ["the", "cat", "with", "hat"]
    assert token_words == ["the", "cat", "with", "hat"]

    x_matrix = [
        [0.31, -0.47, 0.83, 0.19],
        [0.58, 0.14, -0.62, 0.77],
        [-0.26, 0.91, 0.38, -0.54],
        [0.73, -0.35, 0.49, 0.27],
    ]
    wq_matrix = [
        [0.42, -0.33, 0.18, 0.57],
        [-0.61, 0.29, 0.74, -0.22],
        [0.36, 0.68, -0.41, 0.15],
        [-0.27, 0.52, 0.33, -0.49],
    ]
    wk_matrix = [
        [-0.38, 0.64, -0.21, 0.46],
        [0.55, -0.17, 0.39, 0.28],
        [0.12, 0.73, -0.58, -0.31],
        [0.67, -0.44, 0.25, 0.16],
    ]
    wv_matrix = [
        [0.49, 0.23, -0.71, 0.34],
        [-0.52, 0.81, 0.17, -0.29],
        [0.37, -0.63, 0.56, 0.11],
        [0.28, 0.45, -0.24, 0.69],
    ]
    w1_matrix = [
        [0.44, -0.26, 0.61, 0.18],
        [0.35, 0.72, -0.48, 0.27],
        [-0.57, 0.16, 0.39, 0.64],
        [0.22, -0.69, 0.53, -0.31],
    ]
    w2_matrix = [
        [0.58, -0.37, 0.24, 0.46],
        [-0.15, 0.66, 0.41, -0.52],
        [0.73, 0.28, -0.33, 0.19],
        [-0.47, 0.54, 0.12, 0.62],
    ]

    WQ_COL_BASE = 0
    WK_COL_BASE = 16
    WV_COL_BASE = 32
    X_ROW_BASE = 0
    Q_BASE = 64
    K_BASE = 80
    V_BASE = 96
    W1_BASE = 112
    W2_BASE = 128
    P_BASE = 144
    ATTN_BASE = 160
    LN_BASE = 176
    HIDDEN_BASE = 192
    FINAL_BASE = 208

    x_bits, x_fp = quantize_matrix_to_fp8(x_matrix)
    wq_bits, wq_fp = quantize_matrix_to_fp8(wq_matrix)
    wk_bits, wk_fp = quantize_matrix_to_fp8(wk_matrix)
    wv_bits, wv_fp = quantize_matrix_to_fp8(wv_matrix)
    w1_bits, w1_fp = quantize_matrix_to_fp8(w1_matrix)
    w2_bits, w2_fp = quantize_matrix_to_fp8(w2_matrix)

    for row_idx in range(4):
        preload_mem1_word(dut, X_ROW_BASE + row_idx, x_bits[row_idx])
    for col_idx in range(4):
        preload_mem0_word(
            dut,
            WQ_COL_BASE + col_idx,
            [wq_bits[row][col_idx] for row in range(4)],
        )
        preload_mem0_word(
            dut,
            WK_COL_BASE + col_idx,
            [wk_bits[row][col_idx] for row in range(4)],
        )
        preload_mem0_word(
            dut,
            WV_COL_BASE + col_idx,
            [wv_bits[row][col_idx] for row in range(4)],
        )
    preload_mem0_matrix_rows_fp8(dut, W1_BASE, w1_bits)
    preload_mem0_matrix_rows_fp8(dut, W2_BASE, w2_bits)

    q_fp = matmul_expected_fp32(wq_fp, x_fp)
    k_fp = matmul_expected_fp32(wk_fp, x_fp)
    v_fp = matmul_expected_fp32(wv_fp, x_fp)
    q_bits, q_scales, q_raw_fp, q_words = regular_quantize_matrix_for_mem(q_fp)
    k_bits, k_scales, k_raw_fp, k_words = regular_quantize_matrix_for_mem(k_fp)
    v_bits, v_scales, v_raw_fp, v_words = regular_quantize_matrix_for_mem(v_fp)

    for left_base, output_base, expected_words, label in [
        (WQ_COL_BASE, Q_BASE, q_words, "Q projection"),
        (WK_COL_BASE, K_BASE, k_words, "K projection"),
        (WV_COL_BASE, V_BASE, v_words, "V projection"),
    ]:
        program = []
        append_fp8_matmul_mem0_cols_mem1_rows_to_mem0(
            program,
            left_col_base=left_base,
            right_row_base=X_ROW_BASE,
            output_base=output_base,
        )
        await run_lpu_program(dut, program, extra_cycles=40)
        assert_mem0_words(dut, output_base, expected_words, label)

    score_matrix = matmul_expected_fp32(q_raw_fp, transpose_matrix(k_raw_fp))
    causal_score_matrix = []
    for row_idx, score_row in enumerate(score_matrix):
        causal_score_matrix.append([
            to_f32(score * 0.5) if col_idx <= row_idx else -1.0e9
            for col_idx, score in enumerate(score_row)
        ])
    p_bits = [softmax_fp8_quant_expected(row) for row in causal_score_matrix]
    p_raw_fp = [[fp8_e5m2_to_f32(bits) for bits in row] for row in p_bits]
    preload_mem0_matrix_rows_fp8(dut, P_BASE, p_bits)

    attention_fp = matmul_expected_fp32(p_raw_fp, v_raw_fp)
    attn_bits, attn_scales, attn_raw_fp, attn_words = regular_quantize_matrix_for_mem(attention_fp)
    program = []
    append_fp8_matmul_mem0_rows_mem0_rows_to_mem0(
        program,
        left_row_base=P_BASE,
        right_row_base=V_BASE,
        output_base=ATTN_BASE,
    )
    await run_lpu_program(dut, program, extra_cycles=40)
    assert_mem0_words(dut, ATTN_BASE, attn_words, "causal attention")

    layernorm_fp = [layernorm_expected(row) for row in attention_fp]
    ln_bits, ln_scales, ln_raw_fp, ln_words = regular_quantize_matrix_for_mem(layernorm_fp)
    program = []
    append_fp8_matmul_mem0_rows_mem0_rows_to_mem0(
        program,
        left_row_base=P_BASE,
        right_row_base=V_BASE,
        output_base=LN_BASE,
        vxm_layernorm_en=1,
    )
    await run_lpu_program(dut, program, extra_cycles=40)
    assert_mem0_words(dut, LN_BASE, ln_words, "hardware FP32 layernorm")

    hidden_fp = matmul_expected_fp32(ln_raw_fp, w1_fp)
    hidden_relu_fp = [
        [to_f32(value if value > 0.0 else 0.0) for value in row]
        for row in hidden_fp
    ]
    hidden_bits, hidden_scales, hidden_raw_fp, hidden_words = regular_quantize_matrix_for_mem(hidden_relu_fp)
    program = []
    append_fp8_matmul_mem0_rows_mem0_rows_to_mem0(
        program,
        left_row_base=LN_BASE,
        right_row_base=W1_BASE,
        output_base=HIDDEN_BASE,
        output_vxm_ctrl=0b0010,
    )
    await run_lpu_program(dut, program, extra_cycles=40)
    assert_mem0_words(dut, HIDDEN_BASE, hidden_words, "FFN hidden")

    final_fp = matmul_expected_fp32(hidden_raw_fp, w2_fp)
    final_bits, final_scales, final_raw_fp, final_words = regular_quantize_matrix_for_mem(final_fp)
    program = []
    append_fp8_matmul_mem0_rows_mem0_rows_to_mem0(
        program,
        left_row_base=HIDDEN_BASE,
        right_row_base=W2_BASE,
        output_base=FINAL_BASE,
    )
    await run_lpu_program(dut, program, extra_cycles=40)
    assert_mem0_words(dut, FINAL_BASE, final_words, "prefill final output")

    drive_layernorm_debug_matrix(dut, layernorm_fp)
    assert final_raw_fp is not None
    assert int(dut.vxm_input_overflow_dbg.value) == 0, "VXM input FIFO overflowed unexpectedly"


@cocotb.test()
async def test_lpu_toy_prefill_decode_back_to_back(dut):
    """Toy 4-wide LLM tile: prefill "Lebron is the" then decode one token."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    vocab = ["Lebron", "is", "the", "king", "goat", "player", "court", "hat", "."]
    embedding_table = {"<pad>": [0.0, 0.0, 0.0, 0.0]}
    embedding_table.update({token: deterministic_vector(f"embed:{token}") for token in vocab})

    # This is intentionally a deterministic random-init toy model, not a trained
    # model and not a hand-biased prompt completion. The LM head is tied to the
    # input embeddings, which is a common transformer design choice.
    wq_matrix = xavier_matrix("wq", residual=0.20)
    wk_matrix = xavier_matrix("wk", residual=0.20)
    wv_matrix = xavier_matrix("wv", residual=0.20)
    w1_matrix = xavier_matrix("w1", residual=0.10)
    w2_matrix = xavier_matrix("w2", residual=0.10)
    lm_head = {token: embedding_table[token] for token in vocab}

    dut._log.info("Toy vocab order: %s", ", ".join(vocab))
    dut._log.info("Toy embedding table is deterministic random-init, not trained or hand-biased.")

    prefill = await run_toy_transformer_tile(
        dut,
        token_words=["Lebron", "is", "the", "<pad>"],
        valid_len=3,
        prediction_row_idx=2,
        embedding_table=embedding_table,
        wq_matrix=wq_matrix,
        wk_matrix=wk_matrix,
        wv_matrix=wv_matrix,
        w1_matrix=w1_matrix,
        w2_matrix=w2_matrix,
        label="PREFILL",
    )
    prefill_logits = toy_lm_head_logits(prefill["prediction_hidden"], lm_head)
    prefill_token = log_top_tokens(dut, "PREFILL next-token", prefill_logits)
    assert prefill_token in vocab

    decode = await run_toy_transformer_tile(
        dut,
        token_words=["Lebron", "is", "the", prefill_token],
        valid_len=4,
        prediction_row_idx=3,
        embedding_table=embedding_table,
        wq_matrix=wq_matrix,
        wk_matrix=wk_matrix,
        wv_matrix=wv_matrix,
        w1_matrix=w1_matrix,
        w2_matrix=w2_matrix,
        label="DECODE",
    )
    decode_logits = toy_lm_head_logits(decode["prediction_hidden"], lm_head)
    decode_token = log_top_tokens(dut, "DECODE next-token", decode_logits)
    assert decode_token in vocab

    generated_text = " ".join(["Lebron", "is", "the", prefill_token, decode_token]).replace(" .", ".")
    dut._log.info('Toy generated text after prefill+decode: "%s"', generated_text)
    dut._log.info(
        "Toy total approximate LPU cycles: %d",
        prefill["cycles"] + decode["cycles"],
    )
    assert int(dut.vxm_input_overflow_dbg.value) == 0, "VXM input FIFO overflowed unexpectedly"


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
async def test_lpu_regular_self_attention_quantized_output_layernorm_sw(dut):
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
    final_quant_base = 48

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
    quantized_attention = [regular_fp8_row_quant_expected(row) for row in attention_out]
    quantized_attention_rows = [row_bits for row_bits, _ in quantized_attention]
    quantized_attention_scales = [scale_exp for _, scale_exp in quantized_attention]
    dequant_attention = [
        dequantize_regular_fp8_row(row_bits, scale_exp)
        for row_bits, scale_exp in quantized_attention
    ]
    expected_layernorm = [layernorm_expected(row) for row in dequant_attention]

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

    program.extend([build_instruction(**fp_ctrl), build_instruction(**fp_ctrl)])

    for row_idx in range(4):
        append_vxm_regular_fp8_row_store(
            program,
            row_idx=row_idx,
            target_addr=final_quant_base + row_idx,
        )

    preload_program(dut, program)

    await reset_dut(dut)
    await tick(dut, len(program) + 32)

    observed_bits = read_mxm_matrix_bits(dut)
    for row_idx in range(4):
        for col_idx in range(4):
            expected_bits = f32_bits(attention_out[row_idx][col_idx])
            assert observed_bits[row_idx][col_idx] == expected_bits, (
                f"regular FP self-attention mismatch at ({row_idx}, {col_idx}): "
                f"got 0x{observed_bits[row_idx][col_idx]:08x}, "
                f"expected 0x{expected_bits:08x}"
            )

    observed_quant_rows = []
    observed_quant_scales = []
    for row_idx in range(4):
        observed_word = int(dut.u_lpu.u_mem0.sram_array[final_quant_base + row_idx].value)
        expected_word = pack_fp8_row_mem_word(
            quantized_attention_rows[row_idx],
            quantized_attention_scales[row_idx],
        )
        assert observed_word == expected_word, (
            f"final quantized row {row_idx} mismatch: got 0x{observed_word:016x}, "
            f"expected 0x{expected_word:016x}"
        )
        row_bits, scale_exp = unpack_fp8_row_mem_word(observed_word)
        observed_quant_rows.append(row_bits)
        observed_quant_scales.append(scale_exp)

    observed_dequant = [
        dequantize_regular_fp8_row(row_bits, scale_exp)
        for row_bits, scale_exp in zip(observed_quant_rows, observed_quant_scales)
    ]
    observed_layernorm = [layernorm_expected(row) for row in observed_dequant]
    drive_layernorm_debug_matrix(dut, observed_layernorm)

    for row_idx in range(4):
        for col_idx in range(4):
            got_value = observed_layernorm[row_idx][col_idx]
            exp_value = expected_layernorm[row_idx][col_idx]
            assert abs(got_value - exp_value) < 1e-6, (
                f"software layernorm mismatch at ({row_idx}, {col_idx}): "
                f"got {got_value} expected {exp_value}"
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





   

    
