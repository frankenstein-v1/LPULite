import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


def signed_value(value, width):
    mask = (1 << width) - 1
    value = int(value) & mask
    sign_bit = 1 << (width - 1)
    return value - (1 << width) if value & sign_bit else value


def real_from_scaled(raw, scale):
    return raw * (2.0 ** scale)


async def reset_dut(dut):
    dut.rst.value = 1
    dut.clear.value = 0
    dut.en.value = 0
    dut.input_i.value = 0
    dut.weight_i.value = 0
    dut.input_scale_i.value = 0
    dut.weight_scale_i.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


@cocotb.test()
async def test_mac_scaled_decimal_operands(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # 32 * 2^-6 = 0.5
    # 64 * 2^-6 = 1.0
    input_raw = 32
    weight_raw = 64
    input_scale = -6
    weight_scale = -6
    expected_product = input_raw * weight_raw
    expected_scale = input_scale + weight_scale
    expected_real = real_from_scaled(expected_product, expected_scale)

    dut.input_i.value = input_raw
    dut.weight_i.value = weight_raw
    dut.input_scale_i.value = input_scale
    dut.weight_scale_i.value = weight_scale
    dut.en.value = 1

    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")

    dut.en.value = 0

    observed_product = signed_value(dut.product_o.value, 16)
    observed_acc = signed_value(dut.acc_o.value, 32)
    observed_scale = signed_value(dut.acc_scale_o.value, 8)
    observed_real = real_from_scaled(observed_acc, observed_scale)

    assert observed_product == expected_product
    assert observed_acc == expected_product
    assert observed_scale == expected_scale
    assert abs(observed_real - expected_real) < 1e-12
    assert abs(observed_real - 0.5) < 1e-12

    dut._log.info(
        "MAC decimal operands: (%d * 2^%d = %.6f) * "
        "(%d * 2^%d = %.6f) -> acc=%d scale=%d real=%.6f",
        input_raw,
        input_scale,
        real_from_scaled(input_raw, input_scale),
        weight_raw,
        weight_scale,
        real_from_scaled(weight_raw, weight_scale),
        observed_acc,
        observed_scale,
        observed_real,
    )


@cocotb.test()
async def test_mac_accumulates_scaled_decimal_products(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # First product: 0.5 * 1.0 = 0.5
    dut.input_i.value = 32
    dut.weight_i.value = 64
    dut.input_scale_i.value = -6
    dut.weight_scale_i.value = -6
    dut.en.value = 1
    await RisingEdge(dut.clk)

    # Second product with the same combined scale:
    # 0.25 * 0.5 = 0.125
    # raw product = 16 * 32 = 512, scale = -12
    dut.input_i.value = 16
    dut.weight_i.value = 32
    dut.input_scale_i.value = -6
    dut.weight_scale_i.value = -6
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")

    dut.en.value = 0

    expected_acc = (32 * 64) + (16 * 32)
    expected_scale = -12
    expected_real = 0.5 + 0.125

    observed_acc = signed_value(dut.acc_o.value, 32)
    observed_scale = signed_value(dut.acc_scale_o.value, 8)
    observed_real = real_from_scaled(observed_acc, observed_scale)

    assert observed_acc == expected_acc
    assert observed_scale == expected_scale
    assert abs(observed_real - expected_real) < 1e-12

    dut._log.info(
        "MAC accumulated decimal products: acc=%d scale=%d real=%.6f",
        observed_acc,
        observed_scale,
        observed_real,
    )
