import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


LANES = 4
LANE_W = 32
PROB_SCALE = -7


def pack_lanes(values: list[int]) -> int:
    packed = 0
    for i, value in enumerate(values):
        packed |= (value & 0xFFFF_FFFF) << (i * LANE_W)
    return packed


def unpack_lanes(word: int) -> list[int]:
    return [(word >> (i * LANE_W)) & 0xFFFF_FFFF for i in range(LANES)]


def to_signed32(value: int) -> int:
    value &= 0xFFFF_FFFF
    return value - (1 << 32) if value & 0x8000_0000 else value


def lut_softmax_exp_expected(q_value: int) -> int:
    ln2 = 177
    z_value = (-q_value) // ln2
    p_value = q_value + (z_value * ln2)
    lut_addr = -p_value
    if lut_addr < 0 or lut_addr > 177:
        return 0
    lut_value = round((2.718281828459045 ** (-lut_addr / 256.0)) * 256.0)
    return lut_value >> z_value


def align_int32_to_q8_8(raw_value: int, scale: int) -> int:
    raw_value = to_signed32(raw_value)
    shift_amount = scale + 8
    if shift_amount >= 0:
        if shift_amount > 30:
            scaled = -(1 << 31) if raw_value < 0 else (1 << 31) - 1
        else:
            scaled = raw_value << shift_amount
    elif -shift_amount > 62:
        scaled = -1 if raw_value < 0 else 0
    else:
        scaled = raw_value >> (-shift_amount)

    if scaled > (1 << 31) - 1:
        return (1 << 31) - 1
    if scaled < -(1 << 31):
        return -(1 << 31)
    return scaled


def softmax_fixed_expected(lanes: list[int], scale: int) -> list[int]:
    aligned = [align_int32_to_q8_8(value, scale) for value in lanes]
    lane_max = max(aligned)
    lane_exp = [lut_softmax_exp_expected(value - lane_max) for value in aligned]
    sum_exp = sum(lane_exp)
    if sum_exp == 0:
        return [0 for _ in lanes]
    return [min(255, ((value << (-PROB_SCALE)) + (sum_exp >> 1)) // sum_exp) for value in lane_exp]


async def reset_dut(dut) -> None:
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.x_in.value = 0
    dut.x_scale_i.value = 0
    dut.out_ready.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


async def drive_softmax_row(dut, *, scale: int, lanes: list[int]) -> list[int]:
    dut.in_valid.value = 1
    dut.x_scale_i.value = scale
    dut.x_in.value = pack_lanes(lanes)
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    dut.in_valid.value = 0
    dut.x_scale_i.value = 0

    for _ in range(20):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        if int(dut.out_valid.value) == 1:
            return unpack_lanes(int(dut.y_out.value))
    assert False, "Timed out waiting for softmax output"


@cocotb.test()
async def test_softmax_boxed_scale_fixed_row(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    lanes = [4, 1, -2, 0]
    scale = -2
    observed_words = await drive_softmax_row(dut, scale=scale, lanes=lanes)
    observed = [word & 0xFF for word in observed_words]
    expected = softmax_fixed_expected(lanes, scale)

    dut._log.info("softmax raw logits=%s scale=%d", lanes, scale)
    dut._log.info("softmax probability bytes observed=%s expected=%s", observed, expected)

    assert observed == expected
    assert int(dut.y_scale_o.value.to_signed()) == PROB_SCALE


@cocotb.test()
async def test_softmax_boxed_scale_peaked_distribution(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    lanes = [18, -4, -8, -12]
    scale = -3
    observed_words = await drive_softmax_row(dut, scale=scale, lanes=lanes)
    observed = [word & 0xFF for word in observed_words]
    expected = softmax_fixed_expected(lanes, scale)

    dut._log.info("peaked softmax probability bytes observed=%s expected=%s", observed, expected)

    assert observed == expected
    assert int(dut.y_scale_o.value.to_signed()) == PROB_SCALE
