import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


def pack_signed_lanes(values, lane_w):
    mask = (1 << lane_w) - 1
    packed = 0
    for idx, value in enumerate(values):
        packed |= (value & mask) << (idx * lane_w)
    return packed


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.stream_in_data.value = 0
    dut.stream_in_param.value = 0
    dut.in_valid.value = 0
    dut.math_op.value = 0
    dut.accum_en.value = 0
    dut.emit_result.value = 0
    dut.flush.value = 0
    dut.clear_accum.value = 0
    dut.out_ready.value = 1
    await Timer(10, unit="ns")
    dut.rst_n.value = 1
    await Timer(10, unit="ns")


async def drive_input(dut, data_lanes, param_lanes, *, math_op, accum_en=0, emit_result=1):
    dut.stream_in_data.value = pack_signed_lanes(data_lanes, 8)
    dut.stream_in_param.value = pack_signed_lanes(param_lanes, 8)
    dut.math_op.value = math_op
    dut.accum_en.value = accum_en
    dut.emit_result.value = emit_result
    dut.in_valid.value = 1

    while True:
        await RisingEdge(dut.clk)
        if int(dut.in_ready.value):
            break

    dut.in_valid.value = 0
    dut.accum_en.value = 0
    dut.emit_result.value = 0


async def flush_accumulator(dut, *, clear_accum=0):
    dut.flush.value = 1
    dut.clear_accum.value = clear_accum
    await RisingEdge(dut.clk)
    dut.flush.value = 0
    dut.clear_accum.value = 0


@cocotb.test()
async def test_vxm_relu_emits_result(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await drive_input(
        dut,
        data_lanes=[-5, -4, 7, -1],
        param_lanes=[0, 0, 0, 0],
        math_op=0,
    )

    expected = pack_signed_lanes([0, 0, 7, 0], 20)
    assert int(dut.out_valid.value) == 1, "VXM should mark emitted ALU output valid"
    assert int(dut.stream_out.value) == expected, f"Expected {hex(expected)}, got {hex(int(dut.stream_out.value))}"

    await RisingEdge(dut.clk)
    assert int(dut.out_valid.value) == 0, "VXM output should drain once accepted"


@cocotb.test()
async def test_vxm_multiply_keeps_wide_result(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await drive_input(
        dut,
        data_lanes=[20, -7, 12, -3],
        param_lanes=[20, 10, -5, -11],
        math_op=2,
    )

    expected = pack_signed_lanes([400, -70, -60, 33], 20)
    assert int(dut.out_valid.value) == 1, "Multiply result should be emitted"
    assert int(dut.stream_out.value) == expected, f"Expected {hex(expected)}, got {hex(int(dut.stream_out.value))}"


@cocotb.test()
async def test_vxm_flush_and_clear_are_separate(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await drive_input(
        dut,
        data_lanes=[1, 1, 1, 1],
        param_lanes=[1, 1, 1, 1],
        math_op=1,
        accum_en=1,
        emit_result=0,
    )

    await drive_input(
        dut,
        data_lanes=[2, 2, 2, 2],
        param_lanes=[3, 3, 3, 3],
        math_op=2,
        accum_en=1,
        emit_result=0,
    )

    await flush_accumulator(dut, clear_accum=1)

    expected = pack_signed_lanes([8, 8, 8, 8], 20)
    assert int(dut.out_valid.value) == 1, "Flushing should emit the accumulator snapshot"
    assert int(dut.stream_out.value) == expected, f"Expected {hex(expected)}, got {hex(int(dut.stream_out.value))}"

    await RisingEdge(dut.clk)

    await flush_accumulator(dut, clear_accum=0)
    expected_cleared = pack_signed_lanes([0, 0, 0, 0], 20)
    assert int(dut.out_valid.value) == 1, "Accumulator should still be flushable after clear"
    assert int(dut.stream_out.value) == expected_cleared, "Accumulator clear should reset all lanes"


@cocotb.test()
async def test_vxm_backpressure_holds_output_and_stalls_emitters(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    dut.out_ready.value = 0

    dut.stream_in_data.value = pack_signed_lanes([4, 3, 2, 1], 8)
    dut.stream_in_param.value = 0
    dut.math_op.value = 3
    dut.emit_result.value = 1
    dut.in_valid.value = 1

    await RisingEdge(dut.clk)
    expected_first = pack_signed_lanes([4, 3, 2, 1], 20)
    assert int(dut.out_valid.value) == 1, "Output should become valid even when downstream stalls"
    assert int(dut.stream_out.value) == expected_first, "Pass-through output mismatch"

    await RisingEdge(dut.clk)
    assert int(dut.in_ready.value) == 0, "Emitter should stall while the output register is full"
    assert int(dut.out_valid.value) == 1, "Held output must remain valid under backpressure"
    assert int(dut.stream_out.value) == expected_first, "Held output changed while stalled"

    dut.out_ready.value = 1
    await RisingEdge(dut.clk)
    dut.in_valid.value = 0
    dut.emit_result.value = 0

    assert int(dut.out_valid.value) == 0, "Output should drain once downstream is ready"
