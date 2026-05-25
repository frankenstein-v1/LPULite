import cocotb
from cocotb.clock import Clock
from cocotb.triggers import NextTimeStep, RisingEdge, ReadOnly, Timer


DW = 32
ADDR_BITS = 8
RECIP_FRAC_BITS = 30
RECIP_DIVIDEND = 1 << RECIP_FRAC_BITS


def leading_one_idx(value: int) -> int:
    assert value > 0
    return value.bit_length() - 1


def lut_addr_from_divisor(divisor: int, *, addr_bits: int = ADDR_BITS) -> int:
    msb_idx = leading_one_idx(divisor)
    addr = 0

    for frac_idx in range(addr_bits):
        if msb_idx > frac_idx:
            bit = (divisor >> (msb_idx - 1 - frac_idx)) & 1
            addr |= bit << (addr_bits - 1 - frac_idx)

    return addr


def recip_lut_value(addr: int) -> int:
    mantissa = 1.0 + (addr + 0.5) / (1 << ADDR_BITS)
    return int((1 << RECIP_FRAC_BITS) / mantissa + 0.5)


def lut_div_reference(dividend: int, divisor: int) -> int:
    if divisor == 0:
        return 0

    msb_idx = leading_one_idx(divisor)
    addr = lut_addr_from_divisor(divisor)
    reciprocal_q30 = recip_lut_value(addr) >> msb_idx
    return (dividend * reciprocal_q30) >> RECIP_FRAC_BITS


async def reset_dut(dut) -> None:
    dut.rst.value = 1
    dut.start.value = 0
    dut.dividend.value = 0
    dut.divisor.value = 0

    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")

    dut.rst.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


async def launch_transaction(dut, *, dividend: int, divisor: int) -> None:
    dut.dividend.value = dividend
    dut.divisor.value = divisor
    dut.start.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    dut.start.value = 0


async def wait_cycles_and_sample(dut, cycles: int) -> tuple[int, int, int]:
    for _ in range(cycles):
        await RisingEdge(dut.clk)
        await ReadOnly()
    sampled = (int(dut.done.value), int(dut.quotient.value), int(dut.remainder.value))
    await NextTimeStep()
    return sampled


@cocotb.test()
async def test_reset_clears_outputs(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    assert int(dut.done.value) == 0, "reset should leave done low"
    assert int(dut.quotient.value) == 0, "reset should clear quotient"
    assert int(dut.remainder.value) == 0, "reset should clear remainder"


@cocotb.test()
async def test_done_pulses_two_cycles_after_start(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await launch_transaction(dut, dividend=RECIP_DIVIDEND, divisor=0x0000012C)

    done_1, quotient_1, remainder_1 = await wait_cycles_and_sample(dut, 1)
    assert done_1 == 1, "done should assert on the cycle after the launch edge"
    assert quotient_1 == lut_div_reference(RECIP_DIVIDEND, 0x0000012C), (
        "quotient should match the LUT divider model when done asserts"
    )
    assert remainder_1 == 0, "remainder is unused in the LUT divider and should stay zero"

    done_2, _, _ = await wait_cycles_and_sample(dut, 1)
    assert done_2 == 0, "done should be a one-cycle pulse"


@cocotb.test()
async def test_reference_match_across_softmax_like_divisors(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    divisors = [
        88,
        176,
        255,
        256,
        511,
        512,
        777,
        1024,
        2047,
        4096,
        12345,
        65535,
    ]

    for divisor in divisors:
        await launch_transaction(dut, dividend=RECIP_DIVIDEND, divisor=divisor)
        done, quotient, remainder = await wait_cycles_and_sample(dut, 1)

        expected = lut_div_reference(RECIP_DIVIDEND, divisor)
        assert done == 1, f"done missing for divisor={divisor}"
        assert quotient == expected, (
            f"quotient mismatch for divisor={divisor}: got {quotient}, expected {expected}"
        )
        assert remainder == 0, f"remainder should stay zero for divisor={divisor}"

        await wait_cycles_and_sample(dut, 1)


@cocotb.test()
async def test_dividend_scaling_is_preserved(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    divisor = 777
    dividends = [
        RECIP_DIVIDEND,
        RECIP_DIVIDEND >> 1,
        RECIP_DIVIDEND >> 2,
        123456789,
    ]

    for dividend in dividends:
        await launch_transaction(dut, dividend=dividend, divisor=divisor)
        done, quotient, _ = await wait_cycles_and_sample(dut, 1)

        expected = lut_div_reference(dividend, divisor)
        assert done == 1, f"done missing for dividend={dividend}"
        assert quotient == expected, (
            f"scaled quotient mismatch for dividend={dividend}: got {quotient}, expected {expected}"
        )

        await wait_cycles_and_sample(dut, 1)
