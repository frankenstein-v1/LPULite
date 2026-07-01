import cocotb
from cocotb.clock import Clock
from cocotb.triggers import NextTimeStep, ReadOnly, RisingEdge, Timer


LANES = 4
LANE_W = 32
BIAS_RELU_CTRL = 0b0011
SCALE_SOFTMAX_CTRL = 0b1100
QUANT_MULTIPLIER = 2032
QUANT_SHIFT = 16


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


def unpack_q8_lanes(value, *, signed):
    lanes = []
    for i in range(LANES):
        lane = (int(value) >> (i * 8)) & 0xFF
        if signed and lane & 0x80:
            lane -= 1 << 8
        lanes.append(lane)
    return lanes


def relu(value):
    return value if value > 0 else 0


def bias_relu_expected(data_lanes, bias_lanes):
    return [relu(data + bias) for data, bias in zip(data_lanes, bias_lanes)]


def clip_signed_q8(value):
    if value > 127:
        return 127
    if value < -127:
        return -127
    return value


def quantize_regular_expected(lanes):
    quantized = []
    for lane in lanes:
        product = lane * QUANT_MULTIPLIER
        if product >= 0:
            rounded = product + (1 << (QUANT_SHIFT - 1))
        else:
            rounded = product - (1 << (QUANT_SHIFT - 1))
        scaled = rounded >> QUANT_SHIFT
        quantized.append(clip_signed_q8(scaled))
    return quantized


def exp_expected(q_value):
    ln2 = 177
    coeff_a = 92
    coeff_b = 346
    coeff_c = 88

    z_value = (-q_value) // ln2
    p_value = q_value + (z_value * ln2)
    t_value = p_value + coeff_b
    t_squared = t_value * t_value
    q_poly = ((coeff_a * t_squared) >> 16) + coeff_c
    return q_poly >> z_value


def softmax_expected(lanes):
    lane_max = max(lanes)
    lane_sub = [lane - lane_max for lane in lanes]
    lane_exp = [exp_expected(lane) for lane in lane_sub]
    sum_exp = sum(lane_exp)
    quotient = (1 << 30) // sum_exp
    shift = 30 - 8
    return [(quotient * lane) >> shift for lane in lane_exp]


def quantize_softmax_expected(lanes):
    quantized = []
    for lane in lanes:
        if lane <= 0:
            quantized.append(0)
        elif lane >= 255:
            quantized.append(255)
        else:
            quantized.append(lane & 0xFF)
    return quantized


def scale_softmax_quant_expected(data_lanes):
    scaled_lanes = [lane >> 1 for lane in data_lanes]
    softmax_lanes = softmax_expected(scaled_lanes)
    return quantize_softmax_expected(softmax_lanes)


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.stream_in_data.value = 0
    dut.stream_in_bias.value = 0
    dut.in_valid.value = 0
    dut.vxm_ctrl.value = 0
    dut.out_ready.value = 1
    if hasattr(dut, "layernorm_bypass"):
        dut.layernorm_bypass.value = 1
        dut.layernorm_gamma.value = pack_lanes([1, 1, 1, 1])
        dut.layernorm_beta.value = pack_lanes([0, 0, 0, 0])
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def drive_vector(dut, data_lanes, bias_lanes, *, valid=1, ctrl=BIAS_RELU_CTRL):
    dut.stream_in_data.value = pack_lanes(data_lanes)
    dut.stream_in_bias.value = pack_lanes(bias_lanes)
    dut.vxm_ctrl.value = ctrl
    dut.in_valid.value = valid


async def sample_after_edge(dut):
    await RisingEdge(dut.clk)


async def wait_for_input_ready(dut, *, timeout_cycles=20):
    for _ in range(timeout_cycles):
        await ReadOnly()
        if int(dut.in_ready.value) == 1:
            await NextTimeStep()
            return
        await RisingEdge(dut.clk)
    assert False, "VXM never became ready to accept an input row"


async def wait_for_output_handshake(dut, *, timeout_cycles=200, signed=False):
    for _ in range(timeout_cycles):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.out_valid.value) == 1 and int(dut.out_ready.value) == 1:
            return unpack_q8_lanes(dut.stream_out.value, signed=signed)
    assert False, "Timed out waiting for VXM output handshake"


async def run_vxm_row(
    dut,
    data_lanes,
    bias_lanes,
    *,
    ctrl,
    signed_output,
    input_timeout_cycles=20,
    output_timeout_cycles=200,
):
    await wait_for_input_ready(dut, timeout_cycles=input_timeout_cycles)
    await drive_vector(dut, data_lanes, bias_lanes, valid=1, ctrl=ctrl)
    await RisingEdge(dut.clk)
    dut.in_valid.value = 0
    return await wait_for_output_handshake(
        dut,
        timeout_cycles=output_timeout_cycles,
        signed=signed_output,
    )


async def drive_rows_with_backpressure(
    dut,
    rows,
    *,
    ctrl,
    timeout_cycles=1000,
):
    accepted_rows = 0
    observed_outputs = []
    expected_outputs = [scale_softmax_quant_expected(data) for data, _ in rows]

    for _ in range(timeout_cycles):
        if accepted_rows < len(rows):
            data_lanes, bias_lanes = rows[accepted_rows]
            dut.stream_in_data.value = pack_lanes(data_lanes)
            dut.stream_in_bias.value = pack_lanes(bias_lanes)
            dut.vxm_ctrl.value = ctrl
            dut.in_valid.value = 1
        else:
            dut.in_valid.value = 0

        await RisingEdge(dut.clk)
        await ReadOnly()

        if accepted_rows < len(rows) and int(dut.in_valid.value) == 1 and int(dut.in_ready.value) == 1:
            accepted_rows += 1

        if int(dut.out_valid.value) == 1 and int(dut.out_ready.value) == 1:
            observed_outputs.append(unpack_q8_lanes(dut.stream_out.value, signed=False))
            if len(observed_outputs) == len(rows):
                dut.in_valid.value = 0
                break

        await NextTimeStep()

    assert accepted_rows == len(rows), (
        f"Only accepted {accepted_rows} of {len(rows)} rows before timeout"
    )
    assert len(observed_outputs) == len(rows), (
        f"Only observed {len(observed_outputs)} of {len(rows)} outputs before timeout"
    )
    assert observed_outputs == expected_outputs, (
        "Stress test output mismatch. "
        f"Observed={observed_outputs}, expected={expected_outputs}"
    )


async def observe_rows_with_backpressure(
    dut,
    rows,
    *,
    ctrl,
    observe_cycles=250,
):
    accepted_rows = 0

    for _ in range(observe_cycles):
        if accepted_rows < len(rows):
            data_lanes, bias_lanes = rows[accepted_rows]
            dut.stream_in_data.value = pack_lanes(data_lanes)
            dut.stream_in_bias.value = pack_lanes(bias_lanes)
            dut.vxm_ctrl.value = ctrl
            dut.in_valid.value = 1
        else:
            dut.in_valid.value = 0

        await RisingEdge(dut.clk)
        await ReadOnly()

        if accepted_rows < len(rows) and int(dut.in_valid.value) == 1 and int(dut.in_ready.value) == 1:
            accepted_rows += 1

        await NextTimeStep()

    dut.in_valid.value = 0
    await RisingEdge(dut.clk)


async def drive_rows_for_observation(
    dut,
    rows,
    *,
    ctrl,
):
    accepted_rows = 0

    while accepted_rows < len(rows):
        data_lanes, bias_lanes = rows[accepted_rows]
        dut.stream_in_data.value = pack_lanes(data_lanes)
        dut.stream_in_bias.value = pack_lanes(bias_lanes)
        dut.vxm_ctrl.value = ctrl
        dut.in_valid.value = 1

        await RisingEdge(dut.clk)
        await ReadOnly()

        if int(dut.in_ready.value) == 1:
            accepted_rows += 1

        await NextTimeStep()

    dut.in_valid.value = 0


@cocotb.test()
async def test_bias_relu_pipeline_overlap(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    vec_a_data = [-16, -4, 7, 100]
    vec_a_bias = [3, 10, -2, -50]
   
    vec_b_data = [9, -9, 5, -5]
    vec_b_bias = [1, 20, -10, 8]

    vec_c_data = [-40, 50, 67, -41]
    vec_c_bias = [3,10,-2,-20]

    vec_d_data = [67, 41, 67, 41]
    vec_d_bias = [1, 20, -10, 8]



    expected_bias_a = [data + bias for data, bias in zip(vec_a_data, vec_a_bias)]
    expected_bias_b = [data + bias for data, bias in zip(vec_b_data, vec_b_bias)]
    expected_bias_c = [data + bias for data, bias in zip(vec_c_data, vec_c_bias)]
    expected_bias_d = [ data + bias for data, bias in zip(vec_d_data, vec_d_bias)]


    expected_relu_a = [relu(value) for value in expected_bias_a]
    expected_relu_b = [relu(value) for value in expected_bias_b]
    expected_relu_c = [relu(value) for value in expected_bias_c]

    await drive_vector(dut, vec_a_data, vec_a_bias)
    await sample_after_edge(dut)  # A enters stage 0

    await drive_vector(dut, vec_b_data, vec_b_bias)
    await sample_after_edge(dut)  # A enters stage 1, B enters stage 0

    dut.in_valid.value = 0
    await sample_after_edge(dut)  # A enters stage 2, B enters stage 1
    await ReadOnly()

    observed_stage1 = unpack_signed_lanes(dut.s1_bias_reg.value)
    observed_stage2 = unpack_signed_lanes(dut.s2_relu_reg.value)

    assert observed_stage1 == expected_bias_b, (
        f"Stage 1 should be bias-adding vector B while Stage 2 holds vector A. "
        f"Observed stage1={observed_stage1}, expected={expected_bias_b}"
    )
    assert observed_stage2 == expected_relu_a, (
        f"Stage 2 should contain ReLU(vector A + bias A). "
        f"Observed stage2={observed_stage2}, expected={expected_relu_a}"
    )


@cocotb.test()
async def test_bias_relu_pipeline_throughput(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    vectors = [
        ([-16, -4, 7, 100], [3, 10, -2, -50]),
        ([9, -9, 5, -5], [1, 20, -10, 8]),
        ([0, 1, -1, 2], [0, -2, 3, -4]),
        ([15, -32, 64, -1], [-5, 31, -80, 2]),
    ]
    expected_rows = [
        quantize_regular_expected(bias_relu_expected(data, bias))
        for data, bias in vectors
    ]
    max_cycles = len(vectors) + 8
    first_output_cycle = None
    next_expected_index = 0

    for cycle in range(max_cycles):
        if cycle < len(vectors):
            data_lanes, bias_lanes = vectors[cycle]
            await drive_vector(dut, data_lanes, bias_lanes)
        else:
            dut.in_valid.value = 0

        await sample_after_edge(dut)

        if int(dut.out_valid.value) == 1:
            if first_output_cycle is None:
                first_output_cycle = cycle
            expected_lanes = expected_rows[next_expected_index]
            observed_lanes = unpack_q8_lanes(dut.stream_out.value, signed=True)
            assert observed_lanes == expected_lanes, (
                f"Vector {next_expected_index} output mismatch. "
                f"Observed={observed_lanes}, expected={expected_lanes}"
            )
            next_expected_index += 1

            if next_expected_index == len(vectors):
                break
        elif first_output_cycle is not None and next_expected_index < len(vectors):
            assert False, (
                f"Pipeline inserted a bubble on cycle {cycle} after first output at cycle "
                f"{first_output_cycle}"
            )

    assert first_output_cycle is not None, "Pipeline never produced an output"
    assert next_expected_index == len(vectors), "Pipeline did not emit all queued vectors"

    await sample_after_edge(dut)
    assert int(dut.out_valid.value) == 0, "Pipeline should drain after the last queued vector"


@cocotb.test()
async def test_full_row_scale_softmax_quant(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    data_lanes = [16, 8, 4, 0]
    bias_lanes = [0, 0, 0, 0]
    expected_lanes = scale_softmax_quant_expected(data_lanes)

    observed_lanes = await run_vxm_row(
        dut,
        data_lanes,
        bias_lanes,
        ctrl=SCALE_SOFTMAX_CTRL,
        signed_output=False,
    )

    assert observed_lanes == expected_lanes, (
        "Full VXM scale->softmax->quant path mismatch. "
        f"Observed={observed_lanes}, expected={expected_lanes}"
    )


@cocotb.test()
async def test_four_row_scale_softmax_quant_stress(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    rows = [
        ([16, 8, 4, 0], [0, 0, 0, 0]),
        ([20, 10, 5, 0], [0, 0, 0, 0]),
        ([12, 6, 3, 0], [0, 0, 0, 0]),
        ([28, 14, 7, 0], [0, 0, 0, 0]),
    ]

    await drive_rows_with_backpressure(
        dut,
        rows,
        ctrl=SCALE_SOFTMAX_CTRL,
    )


@cocotb.test()
async def test_four_row_scale_softmax_quant_observe(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    rows = [
        ([16, 8, 4, 0], [0, 0, 0, 0]),
        ([20, 10, 5, 0], [0, 0, 0, 0]),
        ([12, 6, 3, 0], [0, 0, 0, 0]),
        ([28, 14, 7, 0], [0, 0, 0, 0]),
    ]

    driver_task = cocotb.start_soon(
        drive_rows_for_observation(
            dut,
            rows,
            ctrl=SCALE_SOFTMAX_CTRL,
        )
    )
    await Timer(2500, unit="ns")
    driver_task.cancel()
    dut.in_valid.value = 0


@cocotb.test()
async def test_layernorm_normal(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Enable LayerNorm (bypass = 0)
    dut.layernorm_bypass.value = 0
    # Set gamma = [2, 2, 2, 2], beta = [3, 3, 3, 3]
    dut.layernorm_gamma.value = pack_lanes([2, 2, 2, 2])
    dut.layernorm_beta.value = pack_lanes([3, 3, 3, 3])

    # Input: [20, 10, 5, 1], bias: [0, 0, 0, 0]
    # Expect output: [6, 3, 1, 0]
    observed = await run_vxm_row(
        dut,
        [20, 10, 5, 1],
        [0, 0, 0, 0],
        ctrl=0b0000, # no bias, no relu, no scale, no softmax
        signed_output=True,
    )
    assert observed == [6, 3, 1, 0], f"LayerNorm output mismatch: got {observed}, expected [6, 3, 1, 0]"


@cocotb.test()
async def test_layernorm_zero_variance(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Enable LayerNorm (bypass = 0)
    dut.layernorm_bypass.value = 0
    # Set gamma = [2, 2, 2, 2], beta = [3, 3, 3, 3]
    dut.layernorm_gamma.value = pack_lanes([2, 2, 2, 2])
    dut.layernorm_beta.value = pack_lanes([3, 3, 3, 3])

    # Input with 0 variance: [5, 5, 5, 5]
    observed = await run_vxm_row(
        dut,
        [5, 5, 5, 5],
        [0, 0, 0, 0],
        ctrl=0b0000,
        signed_output=True,
    )
    assert observed == [3, 3, 3, 3], f"LayerNorm zero variance output mismatch: got {observed}, expected [3, 3, 3, 3]"

