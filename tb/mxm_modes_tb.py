import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


def as_signed8(value: int) -> int:
    value &= 0xFF
    return value - 256 if value & 0x80 else value


def operand_value(value: int, *, is_signed: bool) -> int:
    return as_signed8(value) if is_signed else (value & 0xFF)


async def tick(dut, n=1):
    for _ in range(n):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")


async def reset_dut(dut):
    dut.rst.value = 1
    dut.mxm_clear.value = 0
    dut.mxm_start.value = 0
    dut.input0.value = 0
    dut.input1.value = 0
    dut.input_is_signed.value = 1
    dut.wght_load0.value = 0
    dut.wght_load1.value = 0
    dut.wght_val0.value = 0
    dut.wght_val1.value = 0
    dut.weight_is_signed.value = 1

    await tick(dut, 2)
    dut.rst.value = 0
    dut.mxm_clear.value = 1
    await tick(dut, 1)
    dut.mxm_clear.value = 0


async def run_single_product(dut, *, input_value: int, weight_value: int, input_is_signed: int, weight_is_signed: int) -> int:
    dut.input0.value = input_value & 0xFF
    dut.input1.value = 0
    dut.input_is_signed.value = input_is_signed
    dut.wght_val0.value = weight_value & 0xFF
    dut.wght_val1.value = 0
    dut.weight_is_signed.value = weight_is_signed
    dut.wght_load0.value = 1
    dut.wght_load1.value = 0
    dut.mxm_start.value = 1

    await tick(dut, 1)

    dut.wght_load0.value = 0
    await tick(dut, 1)
    await tick(dut, 1)

    dut.mxm_start.value = 0
    dut.input0.value = 0
    await tick(dut, 1)

    return int(dut.c00.value.to_signed())


@cocotb.test()
async def test_mxm_signed_signed_mode(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    observed = await run_single_product(
        dut,
        input_value=0xFE,
        weight_value=0x03,
        input_is_signed=1,
        weight_is_signed=1,
    )
    expected = operand_value(0xFE, is_signed=True) * operand_value(0x03, is_signed=True)
    assert observed == expected, f"signed*signed mismatch: got {observed}, expected {expected}"


@cocotb.test()
async def test_mxm_unsigned_unsigned_mode(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    observed = await run_single_product(
        dut,
        input_value=0xFF,
        weight_value=0x02,
        input_is_signed=0,
        weight_is_signed=0,
    )
    expected = operand_value(0xFF, is_signed=False) * operand_value(0x02, is_signed=False)
    assert observed == expected, f"unsigned*unsigned mismatch: got {observed}, expected {expected}"


@cocotb.test()
async def test_mxm_signed_unsigned_mode(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    observed = await run_single_product(
        dut,
        input_value=0xFF,
        weight_value=0x02,
        input_is_signed=1,
        weight_is_signed=0,
    )
    expected = operand_value(0xFF, is_signed=True) * operand_value(0x02, is_signed=False)
    assert observed == expected, f"signed*unsigned mismatch: got {observed}, expected {expected}"


@cocotb.test()
async def test_mxm_unsigned_signed_mode(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    observed = await run_single_product(
        dut,
        input_value=0xC8,
        weight_value=0xFC,
        input_is_signed=0,
        weight_is_signed=1,
    )
    expected = operand_value(0xC8, is_signed=False) * operand_value(0xFC, is_signed=True)
    assert observed == expected, f"unsigned*signed mismatch: got {observed}, expected {expected}"
