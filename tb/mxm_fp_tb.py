import struct

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def fp8_e5m2_bits(value: float) -> int:
    # Restrict the test vectors to values that are exactly representable in
    # E5M2 so this encoder can stay simple and deterministic.
    table = {
        0.0: 0x00,
        0.5: 0x38,
        1.0: 0x3C,
        2.0: 0x40,
        4.0: 0x44,
        8.0: 0x48,
        -0.5: 0xB8,
        -1.0: 0xBC,
        -2.0: 0xC0,
        -4.0: 0xC4,
        -8.0: 0xC8,
    }
    if value not in table:
        raise ValueError(f"unsupported FP8 test value {value}")
    return table[value]


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
    dut.wght_load0.value = 0
    dut.wght_load1.value = 0
    dut.wght_val0.value = 0
    dut.wght_val1.value = 0
    await tick(dut, 2)
    dut.rst.value = 0
    dut.mxm_clear.value = 1
    await tick(dut, 1)
    dut.mxm_clear.value = 0


async def load_weight_row(dut, row_vals):
    dut.wght_val0.value = row_vals[0]
    dut.wght_val1.value = row_vals[1]
    dut.wght_load0.value = 1
    dut.wght_load1.value = 1
    await tick(dut, 1)
    dut.wght_load0.value = 0
    dut.wght_load1.value = 0


async def launch_outer_product(dut, col_vals):
    dut.input0.value = col_vals[0]
    dut.input1.value = col_vals[1]
    dut.mxm_start.value = 1
    await tick(dut, 1)
    dut.mxm_start.value = 0
    # The FP MAC path walks IDLE -> CAST -> FMA -> WRITEBACK -> IDLE.
    await tick(dut, 4)


def matmul_reference(a, b):
    out = [[0.0 for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            out[r][c] = sum(a[r][k] * b[k][c] for k in range(2))
    return out


@cocotb.test()
async def test_mxm_fp8_matmul_to_fp32_matrix(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Choose values that are exactly representable in E5M2 and whose products
    # accumulate to exactly representable FP32 values.
    a = [
        [1.0, 0.5],
        [-1.0, 2.0],
    ]
    b = [
        [2.0, -1.0],
        [0.5, 4.0],
    ]

    expected = matmul_reference(a, b)

    # Outer-product schedule: load B[k, :] as weights, then launch A[:, k].
    for k in range(2):
        await load_weight_row(
            dut,
            [fp8_e5m2_bits(b[k][0]), fp8_e5m2_bits(b[k][1])],
        )
        await launch_outer_product(
            dut,
            [fp8_e5m2_bits(a[0][k]), fp8_e5m2_bits(a[1][k])],
        )

    observed_bits = [
        [int(dut.c00.value), int(dut.c01.value)],
        [int(dut.c10.value), int(dut.c11.value)],
    ]
    observed = [[f32_from_bits(bits) for bits in row] for row in observed_bits]

    for r in range(2):
        for c in range(2):
            exp_bits = f32_bits(expected[r][c])
            got_bits = observed_bits[r][c]
            assert got_bits == exp_bits, (
                f"FP MXM mismatch at ({r}, {c}): got bits 0x{got_bits:08x}"
                f" ({observed[r][c]}), expected 0x{exp_bits:08x}"
                f" ({expected[r][c]})"
            )
