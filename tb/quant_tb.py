import struct

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


LANES = 4
LANE_W = 32
SOFTMAX_PROB_SCALE = -7
QUANT_SIGNED_INT8 = 0
QUANT_SOFTMAX_U8 = 1
QUANT_FP8_E5M2 = 2


def pack_input_lanes(lanes: list[int]) -> int:
    assert len(lanes) == LANES
    word = 0
    for i, lane in enumerate(lanes):
        word |= (lane & 0xFFFF_FFFF) << (i * LANE_W)
    return word


def pack_output_bytes(values: list[int]) -> int:
    assert len(values) == LANES
    word = 0
    for i, value in enumerate(values):
        word |= (value & 0xFF) << (i * 8)
    return word


def unpack_output_bytes(word: int) -> list[int]:
    return [(word >> (i * 8)) & 0xFF for i in range(LANES)]


def unpack_output_signed_bytes(word: int) -> list[int]:
    lanes = unpack_output_bytes(word)
    return [lane - 256 if lane & 0x80 else lane for lane in lanes]


def signed_scale(dut) -> int:
    return int(dut.q_scale_out.value.to_signed())


def clip_signed_q8(value: int) -> int:
    if value > 127:
        return 127
    if value < -127:
        return -127
    return value


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


def regular_quant_row_reference(lanes: list[int]) -> tuple[list[int], int]:
    shift = compute_row_shift(lanes)
    return [clip_signed_q8(round_shift_signed(lane, shift)) for lane in lanes], shift


def softmax_quant_reference(p_value: int) -> int:
    if p_value <= 0:
        return 0
    if p_value >= 255:
        return 255
    return p_value & 0xFF


def float_to_bits(value: float) -> int:
    return struct.unpack(">I", struct.pack(">f", float(value)))[0]


def fp8_e5m2_bits_reference(value: float) -> int:
    bits = float_to_bits(value)
    sign = (bits >> 31) & 0x1
    exp = (bits >> 23) & 0xFF
    frac = bits & 0x7FFFFF

    if exp == 0 and frac == 0:
        return sign << 7
    if exp == 0xFF:
        return (sign << 7) | 0x7D if frac else (sign << 7) | 0x7C
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


async def reset_dut(dut) -> None:
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.quant_mode_i.value = QUANT_SIGNED_INT8
    dut.x_input.value = 0

    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")

    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


async def drive_transaction(dut, *, quant_mode: int, lanes: list[int]) -> int:
    dut.in_valid.value = 1
    dut.quant_mode_i.value = quant_mode
    dut.x_input.value = pack_input_lanes(lanes)

    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")

    dut.in_valid.value = 0
    dut.quant_mode_i.value = QUANT_SIGNED_INT8
    dut.x_input.value = 0

    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")

    assert int(dut.out_valid.value) == 1, "out_valid should assert one cycle after ingress capture"
    return int(dut.q_row_out.value)


@cocotb.test()
async def test_reset_clears_outputs(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    assert int(dut.out_valid.value) == 0, "reset should clear out_valid"
    assert int(dut.q_row_out.value) == 0, "reset should clear q_row_out"
    assert int(dut.q_scale_out.value) == 0, "reset should clear q_scale_out"


@cocotb.test()
async def test_regular_quant_mode_vector(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    input_lanes = [-5000, -4096, 3096, 5000]
    observed_word = await drive_transaction(dut, quant_mode=QUANT_SIGNED_INT8, lanes=input_lanes)
    observed_scale = int(dut.q_scale_out.value)

    expected_lanes, expected_shift = regular_quant_row_reference(input_lanes)
    observed_lanes = unpack_output_signed_bytes(observed_word)

    assert observed_lanes == expected_lanes, (
        f"regular mode mismatch: got {observed_lanes}, expected {expected_lanes}"
    )
    assert observed_scale == expected_shift, (
        f"regular mode scale mismatch: got {observed_scale}, expected {expected_shift}"
    )


@cocotb.test()
async def test_softmax_quant_mode_vector(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Softmax is expected to emit nonnegative probability-like values already
    # scaled near Q0.8, so realistic vectors stay within [0, 255] and sum near 256.
    input_lanes = [12, 37, 81, 126]
    observed_word = await drive_transaction(dut, quant_mode=QUANT_SOFTMAX_U8, lanes=input_lanes)

    expected_lanes = [softmax_quant_reference(lane) for lane in input_lanes]
    expected_word = pack_output_bytes(expected_lanes)

    assert observed_word == expected_word, (
        f"softmax mode mismatch: got 0x{observed_word:08x}, expected 0x{expected_word:08x}"
    )
    assert signed_scale(dut) == SOFTMAX_PROB_SCALE


@cocotb.test()
async def test_back_to_back_mode_switching(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    regular_lanes = [-4096, -2048, 2048, 4096]
    softmax_lanes = [4, 28, 96, 128]

    dut.in_valid.value = 1
    dut.quant_mode_i.value = QUANT_SIGNED_INT8
    dut.x_input.value = pack_input_lanes(regular_lanes)
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")

    dut.in_valid.value = 1
    dut.quant_mode_i.value = QUANT_SOFTMAX_U8
    dut.x_input.value = pack_input_lanes(softmax_lanes)
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")

    assert int(dut.out_valid.value) == 1, "first back-to-back result should be valid"
    observed_regular = unpack_output_signed_bytes(int(dut.q_row_out.value))
    expected_regular, expected_regular_shift = regular_quant_row_reference(regular_lanes)
    assert observed_regular == expected_regular, (
        f"first result mismatch: got {observed_regular}, expected {expected_regular}"
    )
    assert int(dut.q_scale_out.value) == expected_regular_shift, (
        f"first scale mismatch: got {int(dut.q_scale_out.value)}, expected {expected_regular_shift}"
    )

    dut.in_valid.value = 0
    dut.quant_mode_i.value = QUANT_SIGNED_INT8
    dut.x_input.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")

    assert int(dut.out_valid.value) == 1, "second back-to-back result should be valid"
    observed_softmax = int(dut.q_row_out.value)
    expected_softmax = pack_output_bytes(
        [softmax_quant_reference(lane) for lane in softmax_lanes]
    )
    assert observed_softmax == expected_softmax, (
        f"second result mismatch: got 0x{observed_softmax:08x}, expected 0x{expected_softmax:08x}"
    )
    assert signed_scale(dut) == SOFTMAX_PROB_SCALE


@cocotb.test()
async def test_softmax_quant_peaked_distribution(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # A peaked softmax row is also realistic: one dominant class and three small tails.
    input_lanes = [3, 7, 18, 228]
    observed_word = await drive_transaction(dut, quant_mode=QUANT_SOFTMAX_U8, lanes=input_lanes)

    expected_word = pack_output_bytes(
        [softmax_quant_reference(lane) for lane in input_lanes]
    )
    assert observed_word == expected_word, (
        f"peaked softmax mismatch: got 0x{observed_word:08x}, expected 0x{expected_word:08x}"
    )
    assert signed_scale(dut) == SOFTMAX_PROB_SCALE


@cocotb.test()
async def test_softmax_u8_probability_vector(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    prob_lanes = [16, 32, 64, 128]
    observed_word = await drive_transaction(dut, quant_mode=QUANT_SOFTMAX_U8, lanes=prob_lanes)

    expected_word = pack_output_bytes(prob_lanes)
    assert observed_word == expected_word, (
        f"u8 softmax quant mismatch: got 0x{observed_word:08x}, expected 0x{expected_word:08x}"
    )
    assert signed_scale(dut) == SOFTMAX_PROB_SCALE


@cocotb.test()
async def test_regular_fp8_quant_mode_vector(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # absmax = 8.0 => shared row scale exponent = +3, scaled row = [1.0, -0.5, 0.5, -1.0]
    input_lanes = [
        float_to_bits(8.0),
        float_to_bits(-4.0),
        float_to_bits(4.0),
        float_to_bits(-8.0),
    ]
    observed_word = await drive_transaction(
        dut,
        quant_mode=QUANT_FP8_E5M2,
        lanes=input_lanes,
    )

    expected_word = pack_output_bytes([
        fp8_e5m2_bits_reference(1.0),
        fp8_e5m2_bits_reference(-0.5),
        fp8_e5m2_bits_reference(0.5),
        fp8_e5m2_bits_reference(-1.0),
    ])
    assert observed_word == expected_word, (
        f"regular fp8 mode mismatch: got 0x{observed_word:08x}, expected 0x{expected_word:08x}"
    )
    assert int(dut.q_scale_out.value) == 3, (
        f"regular fp8 scale mismatch: got {int(dut.q_scale_out.value)}, expected 3"
    )
