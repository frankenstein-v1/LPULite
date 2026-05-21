import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


LANES = 4
LANE_W = 32
SHIFT = 16
MULTIPLIER = 2032
ROUNDING_OFFSET = 1 << (SHIFT - 1)


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


def regular_quant_reference(acc_value: int) -> int:
    product = acc_value * MULTIPLIER

    if product >= 0:
        rounded = product + ROUNDING_OFFSET
    else:
        rounded = product - ROUNDING_OFFSET

    shifted = rounded >> SHIFT

    if shifted > 127:
        return 127
    if shifted < -127:
        return -127
    return shifted


def softmax_quant_reference(p_value: int) -> int:
    if p_value <= 0:
        return 0
    if p_value >= 255:
        return 255
    return p_value & 0xFF


async def reset_dut(dut) -> None:
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.mode_softmax.value = 0
    dut.x_input.value = 0

    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")

    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


async def drive_transaction(dut, *, mode_softmax: int, lanes: list[int]) -> int:
    dut.in_valid.value = 1
    dut.mode_softmax.value = mode_softmax
    dut.x_input.value = pack_input_lanes(lanes)

    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")

    dut.in_valid.value = 0
    dut.mode_softmax.value = 0
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


@cocotb.test()
async def test_regular_quant_mode_vector(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    input_lanes = [-5000, -4096, 3096, 5000]
    observed_word = await drive_transaction(dut, mode_softmax=0, lanes=input_lanes)

    expected_lanes = [regular_quant_reference(lane) for lane in input_lanes]
    observed_lanes = unpack_output_signed_bytes(observed_word)

    assert observed_lanes == expected_lanes, (
        f"regular mode mismatch: got {observed_lanes}, expected {expected_lanes}"
    )


@cocotb.test()
async def test_softmax_quant_mode_vector(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Softmax is expected to emit nonnegative probability-like values already
    # scaled near Q0.8, so realistic vectors stay within [0, 255] and sum near 256.
    input_lanes = [12, 37, 81, 126]
    observed_word = await drive_transaction(dut, mode_softmax=1, lanes=input_lanes)

    expected_lanes = [softmax_quant_reference(lane) for lane in input_lanes]
    expected_word = pack_output_bytes(expected_lanes)

    assert observed_word == expected_word, (
        f"softmax mode mismatch: got 0x{observed_word:08x}, expected 0x{expected_word:08x}"
    )


@cocotb.test()
async def test_back_to_back_mode_switching(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    regular_lanes = [-4096, -2048, 2048, 4096]
    softmax_lanes = [4, 28, 96, 128]

    dut.in_valid.value = 1
    dut.mode_softmax.value = 0
    dut.x_input.value = pack_input_lanes(regular_lanes)
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")

    dut.in_valid.value = 1
    dut.mode_softmax.value = 1
    dut.x_input.value = pack_input_lanes(softmax_lanes)
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")

    assert int(dut.out_valid.value) == 1, "first back-to-back result should be valid"
    observed_regular = unpack_output_signed_bytes(int(dut.q_row_out.value))
    expected_regular = [regular_quant_reference(lane) for lane in regular_lanes]
    assert observed_regular == expected_regular, (
        f"first result mismatch: got {observed_regular}, expected {expected_regular}"
    )

    dut.in_valid.value = 0
    dut.mode_softmax.value = 0
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


@cocotb.test()
async def test_softmax_quant_peaked_distribution(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # A peaked softmax row is also realistic: one dominant class and three small tails.
    input_lanes = [3, 7, 18, 228]
    observed_word = await drive_transaction(dut, mode_softmax=1, lanes=input_lanes)

    expected_word = pack_output_bytes(
        [softmax_quant_reference(lane) for lane in input_lanes]
    )
    assert observed_word == expected_word, (
        f"peaked softmax mismatch: got 0x{observed_word:08x}, expected 0x{expected_word:08x}"
    )
