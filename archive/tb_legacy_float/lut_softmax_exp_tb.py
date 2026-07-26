import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, ReadOnly
from pathlib import Path
import re

def parse_lut_from_sv() -> dict[int, int]:
    """Parses the lookup table values directly from the SystemVerilog file to guarantee alignment."""
    sv_file = Path(__file__).parent / "../src/lut_softmax_exp.sv"
    lut = {}
    pattern = re.compile(r"8'h([0-9a-fA-F]+):\s*exp_lut\s*=\s*\d+'d(\d+);")
    
    with open(sv_file, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                addr = int(m.group(1), 16)
                val = int(m.group(2))
                lut[addr] = val
    return lut

# Load the LUT values once
LUT = parse_lut_from_sv()

def lut_softmax_exp_reference(q: int, dw: int = 32) -> int:
    """Python reference model of lut_softmax_exp matched to SystemVerilog arithmetic."""
    LN2 = 177
    
    # SystemVerilog integer division truncates towards zero.
    # In Python, we simulate this by converting to float division and casting to int.
    z = int((-q) / LN2)
    
    # Calculate remainder p
    p = q + (z * LN2)
    
    # Address is 8-bit unsigned representation of -p
    addr = (-p) & 0xFF
    
    # Get LUT value, defaults to 0 if out of bounds (addr > 177)
    lut_value = LUT.get(addr, 0)
    
    # Arithmetic right shift by z
    if z < 0:
        # Positive inputs are not officially supported and can produce undefined shift results in SV.
        return 0
    elif z >= dw:
        return 0
    else:
        return lut_value >> z

async def reset_dut(dut) -> None:
    """Applies a reset pulse to the DUT."""
    dut.rst.value = 1
    dut.q.value = 0
    await Timer(20, unit="ns")
    dut.rst.value = 0
    await Timer(10, unit="ns")

@cocotb.test()
async def test_reset_behavior(dut):
    """Verifies that reset functions correctly and the output is computed combinationally."""
    await reset_dut(dut)
    
    # Drive q = 0
    dut.q.value = 0
    await Timer(1, unit="ns")
    assert int(dut.q_out.value) == 256, f"Expected 256 at q=0, got {int(dut.q_out.value)}"

@cocotb.test()
async def test_specific_cases(dut):
    """Tests typical, boundary, and underflow inputs with specific assertions."""
    await reset_dut(dut)
    
    # Specific test cases: (q_input, expected_q_out)
    cases = [
        (0, 256),       # exp(0) * 256 = 256
        (-100, 173),    # Typical negative value
        (-177, 128),    # Exactly -LN2, expected exp(-ln2)*256 = 128
        (-178, 127),    # Just past -LN2
        (-300, 79),     # More negative typical value
        (-354, 64),     # Exactly -2 * LN2
        (-1593, 0),     # Underflow boundary (z = 9)
        (-2000, 0),     # Deep underflow
    ]
    
    for q_in, expected_out in cases:
        dut.q.value = q_in
        await Timer(1, unit="ns")
        observed = int(dut.q_out.value)
        assert observed == expected_out, (
            f"Mismatch for q={q_in}: expected {expected_out}, got {observed}"
        )
        dut._log.info(f"Verified q={q_in:5d} -> q_out={observed:3d} (expected {expected_out})")

@cocotb.test()
async def test_sweep(dut):
    """Sweeps all negative inputs from 0 down to -2000 to ensure 100% correctness."""
    await reset_dut(dut)
    
    mismatches = 0
    for q_in in range(0, -2001, -1):
        dut.q.value = q_in
        await Timer(1, unit="ns")
        
        observed = int(dut.q_out.value)
        expected = lut_softmax_exp_reference(q_in)
        
        if observed != expected:
            dut._log.error(f"Mismatch at q={q_in}: expected {expected}, got {observed}")
            mismatches += 1
            if mismatches >= 10:
                assert False, "Too many mismatches. Aborting."
                
    assert mismatches == 0, f"Found {mismatches} mismatches during sweep."
    dut._log.info("Successfully completed 100% sweep verification from q=0 down to q=-2000!")
