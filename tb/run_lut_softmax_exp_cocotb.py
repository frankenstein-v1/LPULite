import os
from pathlib import Path
from cocotb_tools.runner import get_runner

def test_lut_softmax_exp():
    # Source files
    src_dir = (Path(__file__).parent / "../src").resolve()
    sources = [
        src_dir / "lut_softmax_exp.sv"
    ]
    
    runner = get_runner("icarus")
    
    # We build the simulation
    runner.build(
        sources=sources,
        hdl_toplevel="lut_softmax_exp",
        always=True,
        waves=True,
    )
    
    # We run the specific test module
    runner.test(
        hdl_toplevel="lut_softmax_exp",
        test_module="lut_softmax_exp_tb",
        waves=True,
    )

if __name__ == "__main__":
    test_lut_softmax_exp()
