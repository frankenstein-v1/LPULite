import struct

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


LANES = 4
LANE_W = 32


def float_to_bits(value: float) -> int:
    return struct.unpack(">I", struct.pack(">f", float(value)))[0]


def bits_to_float(bits: int) -> float:
    return struct.unpack(">f", struct.pack(">I", bits & 0xFFFF_FFFF))[0]


def to_f32(value: float) -> float:
    return bits_to_float(float_to_bits(value))


def pack_lanes(values: list[int]) -> int:
    packed = 0
    for i, value in enumerate(values):
        packed |= (value & 0xFFFF_FFFF) << (i * LANE_W)
    return packed


def unpack_lanes(word: int) -> list[int]:
    lanes = []
    for i in range(LANES):
        lanes.append((word >> (i * LANE_W)) & 0xFFFF_FFFF)
    return lanes


def lut_softmax_exp_expected(q_value: int) -> int:
    ln2 = 177
    z_value = (-q_value) // ln2
    p_value = q_value + (z_value * ln2)
    lut_addr = -p_value
    if lut_addr < 0 or lut_addr > 177:
        return 0
    lut_value = round((2.718281828459045 ** (-lut_addr / 256.0)) * 256.0)
    return lut_value >> z_value


def fp32_to_q8_8(fp_bits: int) -> int:
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
    elif -shift_amount > 62:
        scaled_value = 0
    else:
        scaled_value >>= -shift_amount

    if sign_bit:
        scaled_value = -scaled_value

    if scaled_value > 0x7FFF_FFFF:
        return 0x7FFF_FFFF
    if scaled_value < -0x8000_0000:
        return -0x8000_0000
    return scaled_value


def uq8_8_to_fp32(fixed_value: int) -> int:
    if fixed_value == 0:
        return 0

    msb_idx = max(idx for idx in range(32) if (fixed_value >> idx) & 0x1)
    exponent_bits = msb_idx + 119
    if msb_idx <= 23:
        normalized = fixed_value << (23 - msb_idx)
    else:
        normalized = fixed_value >> (msb_idx - 23)
    return ((exponent_bits & 0xFF) << 23) | (normalized & 0x7FFFFF)


def softmax_fp_expected(float_values: list[float]) -> list[int]:
    lane_max = to_f32(max(float_values))
    delta_bits = [float_to_bits(to_f32(value - lane_max)) for value in float_values]
    exp_bits = [
        uq8_8_to_fp32(lut_softmax_exp_expected(fp32_to_q8_8(bits)))
        for bits in delta_bits
    ]
    exp_values = [bits_to_float(bits) for bits in exp_bits]

    sum01 = to_f32(exp_values[0] + exp_values[1])
    sum23 = to_f32(exp_values[2] + exp_values[3])
    sum_exp = to_f32(sum01 + sum23)

    return [float_to_bits(to_f32(value / sum_exp)) for value in exp_values]


def softmax_int_expected(lanes: list[int]) -> list[int]:
    lane_max = max(lanes)
    lane_sub = [lane - lane_max for lane in lanes]
    lane_exp = [lut_softmax_exp_expected(lane) for lane in lane_sub]
    sum_exp = sum(lane_exp)
    quotient = (1 << 30) // sum_exp
    return [((quotient * lane) >> 22) & 0xFFFF_FFFF for lane in lane_exp]


async def reset_dut(dut) -> None:
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.input_mode_fp.value = 0
    dut.x_in.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


async def drive_softmax_row(dut, *, input_mode_fp: int, lanes: list[int]) -> list[int]:
    dut.in_valid.value = 1
    dut.input_mode_fp.value = input_mode_fp
    dut.x_in.value = pack_lanes(lanes)
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    dut.in_valid.value = 0
    dut.input_mode_fp.value = 0

    for _ in range(20):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        if int(dut.out_valid.value) == 1:
            return unpack_lanes(int(dut.y_out.value))
    assert False, "Timed out waiting for softmax output"


@cocotb.test()
async def test_softmax_fp_row(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    values = [1.0, 0.5, -0.5, 2.0]
    observed = await drive_softmax_row(
        dut,
        input_mode_fp=1,
        lanes=[float_to_bits(value) for value in values],
    )
    expected = softmax_fp_expected(values)

    for obs_bits, exp_bits in zip(observed, expected):
        obs_value = bits_to_float(obs_bits)
        exp_value = bits_to_float(exp_bits)
        assert abs(obs_value - exp_value) < 1e-6, (
            f"softmax fp mismatch: got {obs_value} expected {exp_value}"
        )
    assert int(dut.out_mode_fp.value) == 1, "fp mode should be reflected on output"


@cocotb.test()
async def test_softmax_integer_mode_regression(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    lanes = [16, 8, 4, 0]
    observed = await drive_softmax_row(dut, input_mode_fp=0, lanes=lanes)
    expected = softmax_int_expected(lanes)

    assert observed == expected, (
        f"softmax integer regression mismatch: got {observed}, expected {expected}"
    )
    assert int(dut.out_mode_fp.value) == 0, "integer mode should clear output fp flag"
