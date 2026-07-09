import math
import random
import struct

import cocotb
from cocotb.triggers import Timer


LANES = 4
LANE_W = 32


def float_to_bits(value):
    return struct.unpack(">I", struct.pack(">f", float(value)))[0]


def bits_to_float(bits):
    return struct.unpack(">f", struct.pack(">I", bits & 0xFFFF_FFFF))[0]


def to_f32(value):
    return bits_to_float(float_to_bits(value))


def pack_float_lanes(values):
    packed = 0
    for i, value in enumerate(values):
        packed |= float_to_bits(value) << (i * LANE_W)
    return packed


def unpack_float_lanes(value):
    lanes = []
    for i in range(LANES):
        bits = (int(value) >> (i * LANE_W)) & 0xFFFF_FFFF
        lanes.append(bits_to_float(bits))
    return lanes


def layernorm_reference(values, gamma, beta, eps=1e-5):
    mean = to_f32(sum(values) / len(values))
    variance = to_f32(sum(to_f32((x - mean) * (x - mean)) for x in values) / len(values))
    inv_std = to_f32(1.0 / math.sqrt(variance + eps))
    return [
        to_f32(to_f32(to_f32(x - mean) * inv_std) * g + b)
        for x, g, b in zip(values, gamma, beta)
    ]


async def check_layernorm(dut, values, gamma, beta):
    dut.x_in.value = pack_float_lanes(values)
    dut.gamma.value = pack_float_lanes(gamma)
    dut.beta.value = pack_float_lanes(beta)

    await Timer(1, unit="ns")

    observed = unpack_float_lanes(dut.y_out.value)
    expected = layernorm_reference(values, gamma, beta)

    for idx, (got, exp) in enumerate(zip(observed, expected)):
        assert abs(got - exp) < 1e-5, (
            f"FP32 layernorm lane {idx} mismatch: got {got}, expected {exp}"
        )


@cocotb.test()
async def test_layernorm_fp32_combinatorial_isolated(dut):
    await check_layernorm(
        dut,
        values=[0.35, -0.72, 1.18, 0.49],
        gamma=[1.0, 1.0, 1.0, 1.0],
        beta=[0.0, 0.0, 0.0, 0.0],
    )


@cocotb.test()
async def test_layernorm_fp32_gamma_beta(dut):
    await check_layernorm(
        dut,
        values=[1.41, -0.58, 0.27, -1.33],
        gamma=[0.75, 1.25, -0.50, 1.50],
        beta=[0.13, -0.21, 0.37, -0.49],
    )


@cocotb.test()
async def test_layernorm_fp32_random_sweep(dut):
    random.seed(42)
    for _ in range(10):
        values = [random.uniform(-1.75, 1.75) for _ in range(LANES)]
        gamma = [random.uniform(0.35, 1.65) for _ in range(LANES)]
        beta = [random.uniform(-0.55, 0.55) for _ in range(LANES)]
        await check_layernorm(dut, values, gamma, beta)
