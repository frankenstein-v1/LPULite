import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


LANES = 4
LANE_W = 32


def pack_lanes(values):
    packed = 0
    for i, value in enumerate(values):
        packed |= (value & 0xFFFF_FFFF) << (i * LANE_W)
    return packed


def shift_lane(value):
    return (value >> 1) & 0xFFFF_FFFF


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.stream_in_data.value = 0
    dut.stream_in_bias.value = 0
    dut.in_valid.value = 0
    dut.vxm_ctrl.value = 0
    dut.out_ready.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_raw_input_then_scale_into_row_collector(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    data_lanes = [10, -8, 7, -3]
    bias_lanes = [1000, -1000, 2000, -2000]
    expected_lanes = [shift_lane(value) for value in data_lanes]
    expected_row = pack_lanes(expected_lanes)

    dut.stream_in_data.value = pack_lanes(data_lanes)
    dut.stream_in_bias.value = pack_lanes(bias_lanes)
    dut.vxm_ctrl.value = 0b100
    dut.in_valid.value = 1
    dut.out_ready.value = 1

    await Timer(1, unit="ns")

    assert int(dut.in_ready.value) == 1, "VXM should pass out_ready through to in_ready"
    assert int(dut.out_valid.value) == 1, "VXM should pass in_valid through to out_valid"
    assert int(dut.stream_out.value) == expected_row, "Row collector output mismatch for raw->scale path"

    for i, expected in enumerate(expected_lanes):
        actual = dut.stream_out.value[(i + 1) * LANE_W - 1 : i * LANE_W].to_signed()
        expected_signed = expected if expected < (1 << 31) else expected - (1 << 32)
        assert actual == expected_signed, f"lane {i} mismatch in row collector packing"


@cocotb.test()
async def test_bias_and_relu_are_bypassed_on_raw_scale_path(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    data_lanes = [-16, -1, 33, 1024]
    bias_lanes = [123, 456, -789, -2048]
    dut.stream_in_data.value = pack_lanes(data_lanes)
    dut.stream_in_bias.value = pack_lanes(bias_lanes)
    dut.vxm_ctrl.value = 0b100
    dut.in_valid.value = 1
    dut.out_ready.value = 1

    await Timer(1, unit="ns")

    for i, raw_value in enumerate(data_lanes):
        observed = dut.stream_out.value[(i + 1) * LANE_W - 1 : i * LANE_W].to_signed()
        expected = raw_value >> 1
        assert observed == expected, f"lane {i} should scale the raw input, not bias-add or ReLU it"

@cocotb.test()
async def test_scale_into_softmax(dut):
    """Test VXM Scale followed by Softmax Module"""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    
    # Drive inputs with the 4 test vectors from user
    data_lanes = [10, -8, 7, -3]
    
    dut.stream_in_data.value = pack_lanes(data_lanes)
    dut.stream_in_bias.value = 0
    # vxm_ctrl: 1100 (0xC)
    # [3]: Softmax Bypass Sel = 1 (Select Softmax)
    # [2]: Mux3 Sel = 1 (Activate Scale)
    # [1]: Mux2 Sel = 0 (Bypass ReLU)
    # [0]: Mux1 Sel = 0 (Bypass Bias Add)
    dut.vxm_ctrl.value = 0b1100
    dut.in_valid.value = 1
    dut.out_ready.value = 1
    
    await RisingEdge(dut.clk)
    dut.in_valid.value = 0
    
    dut._log.info("Inputs driven. Waiting for Softmax calculation to finish (takes ~32 cycles due to divider)...")
    
    # Wait for output to be valid (Softmax takes ~32 cycles)
    timeout = 0
    while int(dut.out_valid.value) == 0:
        await RisingEdge(dut.clk)
        timeout += 1
        if timeout > 100:
            assert False, "Timeout waiting for Softmax output!"
            
    dut._log.info(f"Softmax finished in {timeout} cycles.")
    
    for i in range(LANES):
        observed = dut.stream_out.value[(i + 1) * LANE_W - 1 : i * LANE_W].to_signed()
        dut._log.info(f"Lane {i} Softmax Output: {observed}")
