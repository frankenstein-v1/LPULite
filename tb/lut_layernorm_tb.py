import cocotb
from cocotb.triggers import Timer
import random
import math

LANES = 4
LANE_W = 32

def pack_lanes(values):
    packed = 0
    for i, value in enumerate(values):
        packed |= (value & 0xFFFF_FFFF) << (i * LANE_W)
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

@cocotb.test()
async def test_layernorm_combinatorial_isolated(dut):
    # Fixed seed for repeatable explanation display
    random.seed(42)

    # 1. Select clean test inputs
    inputs = [15, -25, 40, -10]
    gamma = [3, 2, 4, 1]
    beta = [5, -5, 10, -2]

    # Calculate expected values in Python
    sum_x = sum(inputs)
    mean_u = sum_x >> 2 # Arithmetic shift division by 4
    
    diffs = [x - mean_u for x in inputs]
    sqs = [d * d for d in diffs]
    sum_sq = sum(sqs)
    variance = sum_sq >> 2 # Arithmetic shift division by 4
    
    sigma2_idx = min(variance, 65535)
    
    if sigma2_idx == 0:
        val = 0.00001
    else:
        val = float(sigma2_idx)
        
    inv_sqrt = 1.0 / math.sqrt(val)
    lut_val = int(inv_sqrt * 65536.0 + 0.5)
    
    expected_out = []
    for d, g, b in zip(diffs, gamma, beta):
        x_hat_large = d * lut_val
        scaled_large = x_hat_large * g
        scaled_shifted = scaled_large >> 16
        out_lane = scaled_shifted + b
        expected_out.append(out_lane)

    # 2. Write to DUT inputs
    dut.x_in.value = pack_lanes(inputs)
    dut.gamma.value = pack_lanes(gamma)
    dut.beta.value = pack_lanes(beta)

    # Wait a small combinatorial step
    await Timer(1, unit="ns")

    # Read output
    observed = unpack_signed_lanes(dut.y_out.value)

    # Print outputs clearly to console
    print(f"\n==============================================")
    print(f"--- ISOLATED LAYERNORM TEST ---")
    print(f"Inputs (x):   {inputs}")
    print(f"Gamma (g):    {gamma}")
    print(f"Beta (b):     {beta}")
    print(f"Calculated Mean (u): {mean_u}")
    print(f"Calculated Variance: {variance} (Index: {sigma2_idx})")
    print(f"Lookup Inverse Sqrt: {inv_sqrt:.6f} -> LUT Value: {lut_val}")
    print(f"Expected Output:      {expected_out}")
    print(f"Observed Output:      {observed}")
    print(f"==============================================\n")

    assert observed == expected_out, f"Mismatch: got {observed}, expected {expected_out}"


@cocotb.test()
async def test_layernorm_combinatorial_random_sweep(dut):
    # Run 10 sweeps to test random configurations
    print(f"\n--- RANDOM SWEEP RESULTS ---")
    for sweep in range(1, 11):
        inputs = [random.randint(-100, 100) for _ in range(4)]
        gamma = [random.randint(1, 10) for _ in range(4)]
        beta = [random.randint(-20, 20) for _ in range(4)]

        sum_x = sum(inputs)
        mean_u = sum_x >> 2
        diffs = [x - mean_u for x in inputs]
        sqs = [d * d for d in diffs]
        sum_sq = sum(sqs)
        variance = sum_sq >> 2
        
        sigma2_idx = min(variance, 65535)
        val = 0.00001 if sigma2_idx == 0 else float(sigma2_idx)
        inv_sqrt = 1.0 / math.sqrt(val)
        lut_val = int(inv_sqrt * 65536.0 + 0.5)
        
        expected_out = []
        for d, g, b in zip(diffs, gamma, beta):
            x_hat_large = d * lut_val
            scaled_large = x_hat_large * g
            scaled_shifted = scaled_large >> 16
            out_lane = scaled_shifted + b
            expected_out.append(out_lane)

        dut.x_in.value = pack_lanes(inputs)
        dut.gamma.value = pack_lanes(gamma)
        dut.beta.value = pack_lanes(beta)

        await Timer(1, unit="ns")
        observed = unpack_signed_lanes(dut.y_out.value)

        print(f"Sweep {sweep:2d}: Inputs={inputs} -> Observed={observed} | Expected={expected_out}")
        assert observed == expected_out, f"Sweep {sweep} mismatch: got {observed}, expected {expected_out}"
    print(f"----------------------------\n")
