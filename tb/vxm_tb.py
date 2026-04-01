import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.stream_in_data.value = 0
    dut.stream_in_param.value = 0
    dut.math_op.value = 0
    dut.accum_en.value = 0
    dut.flush.value = 0
    await Timer(10, unit="ns")
    dut.rst_n.value = 1
    await Timer(10, unit="ns")

@cocotb.test()
async def test_vxm_pass(dut):
    """Test pass-through operation"""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await RisingEdge(dut.clk)
    
    # Send some data
    # 4 lanes, e.g., 0x01, 0x02, 0x03, 0x04
    dut.stream_in_data.value = 0x04030201
    dut.stream_in_param.value = 0x00000000
    dut.math_op.value = 0 # Pass
    dut.accum_en.value = 1
    dut.flush.value = 0
    
    await RisingEdge(dut.clk)
    dut.accum_en.value = 0
    dut.flush.value = 1
    
    await RisingEdge(dut.clk)
    dut.flush.value = 0
    
    assert dut.stream_out.value == 0x04030201, f"Expected 0x04030201, got {hex(dut.stream_out.value)}"

@cocotb.test()
async def test_vxm_add(dut):
    """Test addition operation"""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await RisingEdge(dut.clk)
    
    dut.stream_in_data.value  = 0x04030201
    dut.stream_in_param.value = 0x01020304
    dut.math_op.value = 1 # Add
    dut.accum_en.value = 1
    dut.flush.value = 0
    
    await RisingEdge(dut.clk)
    dut.accum_en.value = 0
    dut.flush.value = 1
    
    await RisingEdge(dut.clk)
    dut.flush.value = 0
    
    expected = 0x05050505
    assert dut.stream_out.value == expected, f"Expected {hex(expected)}, got {hex(dut.stream_out.value)}"

@cocotb.test()
async def test_vxm_multiply(dut):
    """Test multiplication operation"""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await RisingEdge(dut.clk)
    
    dut.stream_in_data.value  = 0x02030405
    dut.stream_in_param.value = 0x05040302
    dut.math_op.value = 2 # Multiply
    dut.accum_en.value = 1
    dut.flush.value = 0
    
    await RisingEdge(dut.clk)
    dut.accum_en.value = 0
    dut.flush.value = 1
    
    await RisingEdge(dut.clk)
    dut.flush.value = 0
    
    # 2*5=10(0x0a), 3*4=12(0x0c), 4*3=12(0x0c), 5*2=10(0x0a)
    expected = 0x0a0c0c0a
    assert dut.stream_out.value == expected, f"Expected {hex(expected)}, got {hex(dut.stream_out.value)}"

@cocotb.test()
async def test_vxm_accumulate(dut):
    """Test accumulation across multiple cycles"""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await RisingEdge(dut.clk)
    
    # Cycle 1: Add
    dut.stream_in_data.value  = 0x01010101
    dut.stream_in_param.value = 0x01010101
    dut.math_op.value = 1 # Add -> result = 0x02020202
    dut.accum_en.value = 1
    dut.flush.value = 0
    
    await RisingEdge(dut.clk)
    
    # Cycle 2: Multiply
    dut.stream_in_data.value  = 0x02020202
    dut.stream_in_param.value = 0x03030303
    dut.math_op.value = 2 # Multiply -> result = 0x06060606
    # Total accum should become 0x02...+0x06... = 0x08...
    dut.accum_en.value = 1
    
    await RisingEdge(dut.clk)
    
    dut.accum_en.value = 0
    dut.flush.value = 1
    
    await RisingEdge(dut.clk)
    dut.flush.value = 0
    
    expected = 0x08080808
    assert dut.stream_out.value == expected, f"Expected {hex(expected)}, got {hex(dut.stream_out.value)}"
