import math
import struct
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import NextTimeStep, ReadOnly, RisingEdge, Timer


LANES = 4
LANE_W = 32
BIAS_RELU_CTRL = 0b0011
SCALE_SOFTMAX_CTRL = 0b1100
SOFTMAX_ONLY_CTRL = 0b1000
def pack_lanes(values):
    packed = 0
    for i, value in enumerate(values):
        packed |= (value & 0xFFFF_FFFF) << (i * LANE_W)
    return packed


def pack_float_lanes(values):
    packed = 0
    for i, value in enumerate(values):
        packed |= float_to_bits(value) << (i * LANE_W)
    return packed


def unpack_signed_lanes(value):
    lanes = []
    mask = (1 << LANE_W) - 1
    for i in range(LANES):
        lane = (int(value) >> (i * LANE_W)) & mask
        if lane & (1 << (LANE_W - 1)):
            lane -= 1 << LANE_W
        lanes.append(lane)
    return lanes


def unpack_u32_lanes(value):
    lanes = []
    mask = (1 << LANE_W) - 1
    for i in range(LANES):
        lanes.append((int(value) >> (i * LANE_W)) & mask)
    return lanes


def unpack_q8_lanes(value, *, signed):
    lanes = []
    for i in range(LANES):
        lane = (int(value) >> (i * 8)) & 0xFF
        if signed and lane & 0x80:
            lane -= 1 << 8
        lanes.append(lane)
    return lanes


def relu(value):
    return value if value > 0 else 0


def bias_relu_expected(data_lanes, bias_lanes):
    return [relu(data + bias) for data, bias in zip(data_lanes, bias_lanes)]


def clip_signed_q8(value):
    if value > 127:
        return 127
    if value < -127:
        return -127
    return value


def quantize_regular_expected(lanes):
    max_abs_value = max(abs(lane) for lane in lanes)
    row_shift = 0
    shifted_max_abs_value = max_abs_value
    while shifted_max_abs_value > 127:
        shifted_max_abs_value >>= 1
        row_shift += 1

    quantized = []
    for lane in lanes:
        if row_shift == 0:
            shifted_lane = lane
        else:
            rounding_step = 1 << (row_shift - 1)
            adjusted_lane = lane + rounding_step if lane >= 0 else lane - rounding_step
            shifted_lane = adjusted_lane >> row_shift
        quantized.append(clip_signed_q8(shifted_lane))
    return quantized


def softmax_expected(lanes):
    lane_max = max(lanes)
    lane_sub = [lane - lane_max for lane in lanes]
    lane_exp = [lut_softmax_exp_expected(lane) for lane in lane_sub]
    sum_exp = sum(lane_exp)
    quotient = (1 << 30) // sum_exp
    shift = 30 - 8
    return [(quotient * lane) >> shift for lane in lane_exp]


def quantize_softmax_expected(lanes):
    quantized = []
    for lane in lanes:
        if lane <= 0:
            quantized.append(0)
        elif lane >= 255:
            quantized.append(255)
        else:
            quantized.append(lane & 0xFF)
    return quantized


def scale_softmax_quant_expected(data_lanes):
    scaled_lanes = [lane >> 1 for lane in data_lanes]
    softmax_lanes = softmax_expected(scaled_lanes)
    return quantize_softmax_expected(softmax_lanes)


def float_to_bits(value):
    return struct.unpack(">I", struct.pack(">f", float(value)))[0]


def bits_to_float(bits):
    return struct.unpack(">f", struct.pack(">I", bits & 0xFFFF_FFFF))[0]


def to_f32(value):
    return bits_to_float(float_to_bits(value))


# def layernorm_fp32_expected(row, gamma=None, beta=None, eps=1e-5):
#     if gamma is None:
#         gamma = [1.0 for _ in row]
#     if beta is None:
#         beta = [0.0 for _ in row]
#     mean = to_f32(sum(row) / len(row))
#     variance = to_f32(sum(to_f32((value - mean) * (value - mean)) for value in row) / len(row))
#     inv_std = to_f32(1.0 / math.sqrt(variance + eps))
#     return [
#         to_f32(to_f32(to_f32(value - mean) * inv_std) * gamma_i + beta_i)
#         for value, gamma_i, beta_i in zip(row, gamma, beta)
#     ]


def rmsnorm_fp32_expected(row, gamma=None, eps=1e-5):
    if gamma is None:
        gamma = [1.0 for _ in row]
    rms_sq = to_f32(sum(to_f32(value * value) for value in row) / len(row))
    inv_rms = to_f32(1.0 / math.sqrt(rms_sq + eps))
    return [
        to_f32(to_f32(value * inv_rms) * gamma_i)
        for value, gamma_i in zip(row, gamma)
    ]


def lut_softmax_exp_expected(q_value):
    ln2 = 177
    z_value = (-q_value) // ln2
    p_value = q_value + (z_value * ln2)
    lut_addr = -p_value
    if lut_addr < 0 or lut_addr > 177:
        return 0
    lut_value = round(math.exp(-lut_addr / 256.0) * 256.0)
    return lut_value >> z_value


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


def softmax_fp_quant_expected(data_floats):
    lane_max = to_f32(max(data_floats))
    delta_bits = [float_to_bits(to_f32(value - lane_max)) for value in data_floats]
    exp_bits = [
        uq8_8_to_fp32_ref(lut_softmax_exp_expected(fp32_to_q8_8_ref(bits)))
        for bits in delta_bits
    ]
    exp_values = [bits_to_float(bits) for bits in exp_bits]
    sum01 = to_f32(exp_values[0] + exp_values[1])
    sum23 = to_f32(exp_values[2] + exp_values[3])
    sum_exp = to_f32(sum01 + sum23)
    probs = [float_to_bits(to_f32(value / sum_exp)) for value in exp_values]
    return [fp32_to_uq0_8_ref(prob_bits) for prob_bits in probs]


def fp32_to_uq0_8_ref(fp_bits):
    if (fp_bits >> 31) & 0x1:
        return 0

    exp_bits = (fp_bits >> 23) & 0xFF
    frac_bits = fp_bits & 0x7FFFFF

    if exp_bits == 0 and frac_bits == 0:
        return 0
    if exp_bits == 0xFF:
        return 255

    if exp_bits == 0:
        significand = frac_bits
        exp_unbiased = -126
    else:
        significand = (1 << 23) | frac_bits
        exp_unbiased = exp_bits - 127

    shift_amount = exp_unbiased - 23 + 8
    scaled_value = significand
    if shift_amount >= 0:
        rounded_value = 255 if shift_amount > 8 else (scaled_value << shift_amount)
    elif -shift_amount > 62:
        rounded_value = 0
    else:
        rounded_value = scaled_value + (1 << ((-shift_amount) - 1))
        rounded_value >>= -shift_amount

    return min(255, rounded_value)


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


def fp8_e5m2_bits_reference(value):
    bits = float_to_bits(value)
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


def softmax_fp8_quant_expected(data_floats):
    lane_max = to_f32(max(data_floats))
    delta_bits = [float_to_bits(to_f32(value - lane_max)) for value in data_floats]
    exp_bits = [
        uq8_8_to_fp32_ref(lut_softmax_exp_expected(fp32_to_q8_8_ref(bits)))
        for bits in delta_bits
    ]
    exp_values = [bits_to_float(bits) for bits in exp_bits]
    sum01 = to_f32(exp_values[0] + exp_values[1])
    sum23 = to_f32(exp_values[2] + exp_values[3])
    sum_exp = to_f32(sum01 + sum23)
    prob_floats = [to_f32(value / sum_exp) for value in exp_values]
    return [fp8_e5m2_bits_reference(value) for value in prob_floats]


def regular_fp8_row_quant_expected(data_floats):
    absmax = max(abs(value) for value in data_floats)
    if absmax == 0.0:
        scale_exp = 0
    else:
        scale_exp = math.floor(math.log2(absmax))
    scaled = [math.ldexp(value, -scale_exp) for value in data_floats]
    return [fp8_e5m2_bits_reference(value) for value in scaled], scale_exp


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.stream_in_data.value = 0
    dut.stream_in_bias.value = 0
    dut.in_valid.value = 0
    dut.vxm_ctrl.value = 0
    dut.fp_quant_mode.value = 0
    if hasattr(dut, "rope_en"):
        dut.rope_en.value = 0
        dut.rope_cos_fp8.value = 0x3C3C_3C3C
        dut.rope_sin_fp8.value = 0
    if hasattr(dut, "residual_op"):
        dut.residual_op.value = 0
    dut.out_ready.value = 1
    if hasattr(dut, "layernorm_bypass"):
        dut.layernorm_bypass.value = 1
        dut.layernorm_gamma.value = pack_float_lanes([1.0, 1.0, 1.0, 1.0])
        dut.layernorm_beta.value = pack_float_lanes([0.0, 0.0, 0.0, 0.0])
    if hasattr(dut, "rmsnorm_bypass"):
        dut.rmsnorm_bypass.value = 1
        dut.rmsnorm_gamma.value = pack_float_lanes([1.0, 1.0, 1.0, 1.0])
        dut.rmsnorm_beta.value = pack_float_lanes([0.0, 0.0, 0.0, 0.0])
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def drive_vector(dut, data_lanes, bias_lanes, *, valid=1, ctrl=BIAS_RELU_CTRL, fp_quant_mode=0):
    dut.stream_in_data.value = pack_lanes(data_lanes)
    dut.stream_in_bias.value = pack_lanes(bias_lanes)
    dut.vxm_ctrl.value = ctrl
    dut.fp_quant_mode.value = fp_quant_mode
    dut.in_valid.value = valid


async def sample_after_edge(dut):
    await RisingEdge(dut.clk)


async def wait_for_input_ready(dut, *, timeout_cycles=20):
    for _ in range(timeout_cycles):
        await ReadOnly()
        if int(dut.in_ready.value) == 1:
            await NextTimeStep()
            return
        await RisingEdge(dut.clk)
    assert False, "VXM never became ready to accept an input row"


async def wait_for_output_handshake(dut, *, timeout_cycles=200, signed=False):
    for _ in range(timeout_cycles):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.out_valid.value) == 1 and int(dut.out_ready.value) == 1:
            return unpack_q8_lanes(dut.stream_out.value, signed=signed)
    assert False, "Timed out waiting for VXM output handshake"


async def run_vxm_row(
    dut,
    data_lanes,
    bias_lanes,
    *,
    ctrl,
    signed_output,
    fp_quant_mode=0,
    input_timeout_cycles=20,
    output_timeout_cycles=200,
):
    await wait_for_input_ready(dut, timeout_cycles=input_timeout_cycles)
    await drive_vector(dut, data_lanes, bias_lanes, valid=1, ctrl=ctrl, fp_quant_mode=fp_quant_mode)
    await RisingEdge(dut.clk)
    dut.in_valid.value = 0
    dut.fp_quant_mode.value = 0
    return await wait_for_output_handshake(
        dut,
        timeout_cycles=output_timeout_cycles,
        signed=signed_output,
    )


async def drive_rows_with_backpressure(
    dut,
    rows,
    *,
    ctrl,
    fp_quant_mode=0,
    timeout_cycles=1000,
):
    accepted_rows = 0
    observed_outputs = []
    expected_outputs = [scale_softmax_quant_expected(data) for data, _ in rows]

    for _ in range(timeout_cycles):
        output_complete = False
        if accepted_rows < len(rows):
            data_lanes, bias_lanes = rows[accepted_rows]
            dut.stream_in_data.value = pack_lanes(data_lanes)
            dut.stream_in_bias.value = pack_lanes(bias_lanes)
            dut.vxm_ctrl.value = ctrl
            dut.fp_quant_mode.value = fp_quant_mode
            dut.in_valid.value = 1
        else:
            dut.in_valid.value = 0
            dut.fp_quant_mode.value = 0

        await RisingEdge(dut.clk)
        await ReadOnly()

        if accepted_rows < len(rows) and int(dut.in_valid.value) == 1 and int(dut.in_ready.value) == 1:
            accepted_rows += 1

        if int(dut.out_valid.value) == 1 and int(dut.out_ready.value) == 1:
            observed_outputs.append(unpack_q8_lanes(dut.stream_out.value, signed=False))
            if len(observed_outputs) == len(rows):
                output_complete = True

        await NextTimeStep()
        if output_complete:
            dut.in_valid.value = 0
            dut.fp_quant_mode.value = 0
            break

    assert accepted_rows == len(rows), (
        f"Only accepted {accepted_rows} of {len(rows)} rows before timeout"
    )
    assert len(observed_outputs) == len(rows), (
        f"Only observed {len(observed_outputs)} of {len(rows)} outputs before timeout"
    )
    assert observed_outputs == expected_outputs, (
        "Stress test output mismatch. "
        f"Observed={observed_outputs}, expected={expected_outputs}"
    )


async def observe_rows_with_backpressure(
    dut,
    rows,
    *,
    ctrl,
    fp_quant_mode=0,
    observe_cycles=250,
):
    accepted_rows = 0

    for _ in range(observe_cycles):
        if accepted_rows < len(rows):
            data_lanes, bias_lanes = rows[accepted_rows]
            dut.stream_in_data.value = pack_lanes(data_lanes)
            dut.stream_in_bias.value = pack_lanes(bias_lanes)
            dut.vxm_ctrl.value = ctrl
            dut.fp_quant_mode.value = fp_quant_mode
            dut.in_valid.value = 1
        else:
            dut.in_valid.value = 0
            dut.fp_quant_mode.value = 0

        await RisingEdge(dut.clk)
        await ReadOnly()

        if accepted_rows < len(rows) and int(dut.in_valid.value) == 1 and int(dut.in_ready.value) == 1:
            accepted_rows += 1

        await NextTimeStep()

    dut.in_valid.value = 0
    await RisingEdge(dut.clk)


async def drive_rows_for_observation(
    dut,
    rows,
    *,
    ctrl,
    fp_quant_mode=0,
):
    accepted_rows = 0

    while accepted_rows < len(rows):
        data_lanes, bias_lanes = rows[accepted_rows]
        dut.stream_in_data.value = pack_lanes(data_lanes)
        dut.stream_in_bias.value = pack_lanes(bias_lanes)
        dut.vxm_ctrl.value = ctrl
        dut.fp_quant_mode.value = fp_quant_mode
        dut.in_valid.value = 1

        await RisingEdge(dut.clk)
        await ReadOnly()

        if int(dut.in_ready.value) == 1:
            accepted_rows += 1

        await NextTimeStep()

    dut.in_valid.value = 0
    dut.fp_quant_mode.value = 0


@cocotb.test()
async def test_bias_relu_pipeline_overlap(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    vec_a_data = [-16, -4, 7, 100]
    vec_a_bias = [3, 10, -2, -50]
   
    vec_b_data = [9, -9, 5, -5]
    vec_b_bias = [1, 20, -10, 8]

    vec_c_data = [-40, 50, 67, -41]
    vec_c_bias = [3,10,-2,-20]

    vec_d_data = [67, 41, 67, 41]
    vec_d_bias = [1, 20, -10, 8]



    expected_bias_a = [data + bias for data, bias in zip(vec_a_data, vec_a_bias)]
    expected_bias_b = [data + bias for data, bias in zip(vec_b_data, vec_b_bias)]
    expected_bias_c = [data + bias for data, bias in zip(vec_c_data, vec_c_bias)]
    expected_bias_d = [ data + bias for data, bias in zip(vec_d_data, vec_d_bias)]


    expected_relu_a = [relu(value) for value in expected_bias_a]
    expected_relu_b = [relu(value) for value in expected_bias_b]
    expected_relu_c = [relu(value) for value in expected_bias_c]

    await drive_vector(dut, vec_a_data, vec_a_bias)
    await sample_after_edge(dut)  # A enters stage 0

    await drive_vector(dut, vec_b_data, vec_b_bias)
    await sample_after_edge(dut)  # A enters stage 1, B enters stage 0

    dut.in_valid.value = 0
    await sample_after_edge(dut)  # A enters stage 2, B enters stage 1
    await ReadOnly()

    observed_stage1 = unpack_signed_lanes(dut.s1_bias_reg.value)
    observed_stage2 = unpack_signed_lanes(dut.s2_relu_reg.value)

    assert observed_stage1 == expected_bias_b, (
        f"Stage 1 should be bias-adding vector B while Stage 2 holds vector A. "
        f"Observed stage1={observed_stage1}, expected={expected_bias_b}"
    )
    assert observed_stage2 == expected_relu_a, (
        f"Stage 2 should contain ReLU(vector A + bias A). "
        f"Observed stage2={observed_stage2}, expected={expected_relu_a}"
    )


@cocotb.test()
async def test_bias_relu_pipeline_throughput(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    vectors = [
        ([-16, -4, 7, 100], [3, 10, -2, -50]),
        ([9, -9, 5, -5], [1, 20, -10, 8]),
        ([0, 1, -1, 2], [0, -2, 3, -4]),
        ([15, -32, 64, -1], [-5, 31, -80, 2]),
    ]
    expected_rows = [
        quantize_regular_expected(bias_relu_expected(data, bias))
        for data, bias in vectors
    ]
    max_cycles = len(vectors) + 8
    first_output_cycle = None
    next_expected_index = 0

    for cycle in range(max_cycles):
        if cycle < len(vectors):
            data_lanes, bias_lanes = vectors[cycle]
            await drive_vector(dut, data_lanes, bias_lanes)
        else:
            dut.in_valid.value = 0

        await sample_after_edge(dut)

        if int(dut.out_valid.value) == 1:
            if first_output_cycle is None:
                first_output_cycle = cycle
            expected_lanes = expected_rows[next_expected_index]
            observed_lanes = unpack_q8_lanes(dut.stream_out.value, signed=True)
            assert observed_lanes == expected_lanes, (
                f"Vector {next_expected_index} output mismatch. "
                f"Observed={observed_lanes}, expected={expected_lanes}"
            )
            next_expected_index += 1

            if next_expected_index == len(vectors):
                break
        elif first_output_cycle is not None and next_expected_index < len(vectors):
            assert False, (
                f"Pipeline inserted a bubble on cycle {cycle} after first output at cycle "
                f"{first_output_cycle}"
            )

    assert first_output_cycle is not None, "Pipeline never produced an output"
    assert next_expected_index == len(vectors), "Pipeline did not emit all queued vectors"

    await sample_after_edge(dut)
    assert int(dut.out_valid.value) == 0, "Pipeline should drain after the last queued vector"


@cocotb.test()
async def test_full_row_scale_softmax_quant(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    data_lanes = [16, 8, 4, 0]
    bias_lanes = [0, 0, 0, 0]
    expected_lanes = scale_softmax_quant_expected(data_lanes)

    observed_lanes = await run_vxm_row(
        dut,
        data_lanes,
        bias_lanes,
        ctrl=SCALE_SOFTMAX_CTRL,
        signed_output=False,
    )

    assert observed_lanes == expected_lanes, (
        "Full VXM scale->softmax->quant path mismatch. "
        f"Observed={observed_lanes}, expected={expected_lanes}"
    )


@cocotb.test()
async def test_fp_row_softmax_quant(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    data_lanes = [
        float_to_bits(1.0),
        float_to_bits(0.5),
        float_to_bits(-0.5),
        float_to_bits(2.0),
    ]
    bias_lanes = [0, 0, 0, 0]
    expected_lanes = softmax_fp8_quant_expected([1.0, 0.5, -0.5, 2.0])

    observed_lanes = await run_vxm_row(
        dut,
        data_lanes,
        bias_lanes,
        ctrl=SOFTMAX_ONLY_CTRL,
        signed_output=False,
        fp_quant_mode=1,
    )

    assert observed_lanes == expected_lanes, (
        "FP VXM softmax->quant path mismatch. "
        f"Observed={observed_lanes}, expected={expected_lanes}"
    )


@cocotb.test()
async def test_fp_bias_relu_scale_internal_registers(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    data_floats = [1.0, -2.0, 4.0, -1.0]
    bias_floats = [0.5, 0.25, -1.0, 2.0]
    data_lanes = [float_to_bits(value) for value in data_floats]
    bias_lanes = [float_to_bits(value) for value in bias_floats]

    expected_bias = [float_to_bits(data + bias) for data, bias in zip(data_floats, bias_floats)]
    expected_relu = [float_to_bits(max(data + bias, 0.0)) for data, bias in zip(data_floats, bias_floats)]
    expected_scale = [
        float_to_bits(max(data + bias, 0.0) * 0.5)
        for data, bias in zip(data_floats, bias_floats)
    ]

    await wait_for_input_ready(dut)
    await drive_vector(dut, data_lanes, bias_lanes, valid=1, ctrl=0b0111, fp_quant_mode=1)
    await RisingEdge(dut.clk)
    dut.in_valid.value = 0
    dut.fp_quant_mode.value = 0

    observed_bias = None
    for _ in range(20):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.s1_valid.value) == 1:
            observed_bias = unpack_u32_lanes(dut.s1_bias_reg.value)
            break
    assert observed_bias is not None, "FP VXM pipeline never produced a bias stage result"

    observed_relu = None
    for _ in range(20):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.s2_valid.value) == 1:
            observed_relu = unpack_u32_lanes(dut.s2_relu_reg.value)
            break
    assert observed_relu is not None, "FP VXM pipeline never produced a ReLU stage result"

    observed_scale = None
    for _ in range(20):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.s3_valid.value) == 1:
            observed_scale = unpack_u32_lanes(dut.s3_scale_reg.value)
            break
    assert observed_scale is not None, "FP VXM pipeline never produced a scaled row in s3"

    assert observed_bias == expected_bias, (
        f"FP bias-add mismatch. Observed={observed_bias}, expected={expected_bias}"
    )
    assert observed_relu == expected_relu, (
        f"FP ReLU mismatch. Observed={observed_relu}, expected={expected_relu}"
    )
    assert observed_scale == expected_scale, (
        f"FP scale mismatch. Observed={observed_scale}, expected={expected_scale}"
    )


@cocotb.test()
async def test_fp_full_row_bias_relu_scale_quant(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    data_floats = [1.0, -2.0, 4.0, -1.0]
    bias_floats = [0.5, 0.25, -1.0, 2.0]
    data_lanes = [float_to_bits(value) for value in data_floats]
    bias_lanes = [float_to_bits(value) for value in bias_floats]

    final_floats = [max(data + bias, 0.0) * 0.5 for data, bias in zip(data_floats, bias_floats)]
    expected_lanes, expected_scale = regular_fp8_row_quant_expected(final_floats)

    observed_lanes = await run_vxm_row(
        dut,
        data_lanes,
        bias_lanes,
        ctrl=0b0111,
        signed_output=False,
        fp_quant_mode=1,
    )

    assert observed_lanes == expected_lanes, (
        "FP VXM bias->relu->scale->quant path mismatch. "
        f"Observed={observed_lanes}, expected={expected_lanes}"
    )
    assert int(dut.stream_out_scale.value) == expected_scale, (
        f"FP VXM row scale mismatch. Observed={int(dut.stream_out_scale.value)}, expected={expected_scale}"
    )


@cocotb.test()
async def test_four_row_scale_softmax_quant_stress(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    rows = [
        ([16, 8, 4, 0], [0, 0, 0, 0]),
        ([20, 10, 5, 0], [0, 0, 0, 0]),
        ([12, 6, 3, 0], [0, 0, 0, 0]),
        ([28, 14, 7, 0], [0, 0, 0, 0]),
    ]

    await drive_rows_with_backpressure(
        dut,
        rows,
        ctrl=SCALE_SOFTMAX_CTRL,
    )


@cocotb.test()
async def test_four_row_scale_softmax_quant_observe(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    rows = [
        ([16, 8, 4, 0], [0, 0, 0, 0]),
        ([20, 10, 5, 0], [0, 0, 0, 0]),
        ([12, 6, 3, 0], [0, 0, 0, 0]),
        ([28, 14, 7, 0], [0, 0, 0, 0]),
    ]

    driver_task = cocotb.start_soon(
        drive_rows_for_observation(
            dut,
            rows,
            ctrl=SCALE_SOFTMAX_CTRL,
        )
    )
    await Timer(2500, unit="ns")
    driver_task.cancel()
    dut.in_valid.value = 0


# @cocotb.test()
# async def test_layernorm_normal(dut):
#     cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
#     await reset_dut(dut)
# 
#     data_lanes = [0.35, -0.72, 1.18, 0.49]
#     gamma_lanes = [1.0, 1.0, 1.0, 1.0]
#     beta_lanes = [0.0, 0.0, 0.0, 0.0]
#     expected_lanes, _ = regular_fp8_row_quant_expected(
#         layernorm_fp32_expected(data_lanes, gamma_lanes, beta_lanes)
#     )
# 
#     dut.layernorm_bypass.value = 0
#     dut.layernorm_gamma.value = pack_float_lanes(gamma_lanes)
#     dut.layernorm_beta.value = pack_float_lanes(beta_lanes)
# 
#     observed = await run_vxm_row(
#         dut,
#         [float_to_bits(value) for value in data_lanes],
#         [float_to_bits(0.0) for _ in range(LANES)],
#         ctrl=0b0000, # no bias, no relu, no scale, no softmax
#         signed_output=False,
#         fp_quant_mode=1,
#     )
#     assert observed == expected_lanes, (
#         f"FP32 LayerNorm output mismatch: got {observed}, expected {expected_lanes}"
#     )
# 
# 
# @cocotb.test()
# async def test_layernorm_zero_variance(dut):
#     cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
#     await reset_dut(dut)
# 
#     data_lanes = [0.73, 0.73, 0.73, 0.73]
#     gamma_lanes = [1.5, 0.75, 1.25, 0.5]
#     beta_lanes = [0.13, -0.21, 0.37, -0.49]
#     expected_lanes, _ = regular_fp8_row_quant_expected(
#         layernorm_fp32_expected(data_lanes, gamma_lanes, beta_lanes)
#     )
# 
#     dut.layernorm_bypass.value = 0
#     dut.layernorm_gamma.value = pack_float_lanes(gamma_lanes)
#     dut.layernorm_beta.value = pack_float_lanes(beta_lanes)
# 
#     observed = await run_vxm_row(
#         dut,
#         [float_to_bits(value) for value in data_lanes],
#         [float_to_bits(0.0) for _ in range(LANES)],
#         ctrl=0b0000,
#         signed_output=False,
#         fp_quant_mode=1,
#     )
#     assert observed == expected_lanes, (
#         f"FP32 LayerNorm zero variance output mismatch: got {observed}, expected {expected_lanes}"
#     )
# 
# 
# import random
# import math
# 
# @cocotb.test()
# async def test_layernorm_random(dut):
#     cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
#     await reset_dut(dut)
# 
#     random.seed(42)
#     data_lanes = [random.uniform(-1.75, 1.75) for _ in range(LANES)]
#     gamma_lanes = [random.uniform(0.35, 1.65) for _ in range(LANES)]
#     beta_lanes = [random.uniform(-0.55, 0.55) for _ in range(LANES)]
#     expected_lanes, _ = regular_fp8_row_quant_expected(
#         layernorm_fp32_expected(data_lanes, gamma_lanes, beta_lanes)
#     )
# 
#     dut.layernorm_bypass.value = 0
#     dut.layernorm_gamma.value = pack_float_lanes(gamma_lanes)
#     dut.layernorm_beta.value = pack_float_lanes(beta_lanes)
# 
#     observed = await run_vxm_row(
#         dut,
#         [float_to_bits(value) for value in data_lanes],
#         [float_to_bits(0.0) for _ in range(LANES)],
#         ctrl=0b0000, # no bias, no relu, no scale, no softmax
#         signed_output=False,
#         fp_quant_mode=1,
#     )
# 
#     assert observed == expected_lanes, (
#         f"FP32 LayerNorm random mismatch: got {observed}, expected {expected_lanes}"
#     )

@cocotb.test()
async def test_rmsnorm_normal(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    data_lanes = [0.35, -0.72, 1.18, 0.49]
    gamma_lanes = [1.0, 1.0, 1.0, 1.0]
    expected_lanes, _ = regular_fp8_row_quant_expected(
        rmsnorm_fp32_expected(data_lanes, gamma_lanes)
    )

    dut.rmsnorm_bypass.value = 0
    dut.rmsnorm_gamma.value = pack_float_lanes(gamma_lanes)
    dut.rmsnorm_beta.value = pack_float_lanes([0.0, 0.0, 0.0, 0.0])

    observed = await run_vxm_row(
        dut,
        [float_to_bits(value) for value in data_lanes],
        [float_to_bits(0.0) for _ in range(LANES)],
        ctrl=0b0000, # no bias, no relu, no scale, no softmax
        signed_output=False,
        fp_quant_mode=1,
    )
    assert observed == expected_lanes, (
        f"FP32 RMSNorm output mismatch: got {observed}, expected {expected_lanes}"
    )


@cocotb.test()
async def test_rmsnorm_zero_variance(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    data_lanes = [0.0, 0.0, 0.0, 0.0]
    gamma_lanes = [1.5, 0.75, 1.25, 0.5]
    expected_lanes, _ = regular_fp8_row_quant_expected(
        rmsnorm_fp32_expected(data_lanes, gamma_lanes)
    )

    dut.rmsnorm_bypass.value = 0
    dut.rmsnorm_gamma.value = pack_float_lanes(gamma_lanes)
    dut.rmsnorm_beta.value = pack_float_lanes([0.0, 0.0, 0.0, 0.0])

    observed = await run_vxm_row(
        dut,
        [float_to_bits(value) for value in data_lanes],
        [float_to_bits(0.0) for _ in range(LANES)],
        ctrl=0b0000,
        signed_output=False,
        fp_quant_mode=1,
    )
    assert observed == expected_lanes, (
        f"FP32 RMSNorm zero variance output mismatch: got {observed}, expected {expected_lanes}"
    )


@cocotb.test()
async def test_rmsnorm_random(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    random.seed(42)
    data_lanes = [random.uniform(-1.75, 1.75) for _ in range(LANES)]
    gamma_lanes = [random.uniform(0.35, 1.65) for _ in range(LANES)]
    expected_lanes, _ = regular_fp8_row_quant_expected(
        rmsnorm_fp32_expected(data_lanes, gamma_lanes)
    )

    dut.rmsnorm_bypass.value = 0
    dut.rmsnorm_gamma.value = pack_float_lanes(gamma_lanes)
    dut.rmsnorm_beta.value = pack_float_lanes([0.0, 0.0, 0.0, 0.0])

    observed = await run_vxm_row(
        dut,
        [float_to_bits(value) for value in data_lanes],
        [float_to_bits(0.0) for _ in range(LANES)],
        ctrl=0b0000, # no bias, no relu, no scale, no softmax
        signed_output=False,
        fp_quant_mode=1,
    )

    assert observed == expected_lanes, (
        f"FP32 RMSNorm random mismatch: got {observed}, expected {expected_lanes}"
    )
