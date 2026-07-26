import os
from pathlib import Path
from cocotb_tools.runner import get_runner

def test_sxm():
    # Source files
    src_dir = (Path(__file__).parent / "../src").resolve()
    sources = [
        src_dir / "lpu_pkg.sv",
        src_dir / "sxm.sv"
    ]
    
    runner = get_runner("icarus")
    
    # We build the simulation with wave dumping enabled
    runner.build(
        sources=sources,
        hdl_toplevel="sxm",
        includes=[src_dir],
        always=True,
        waves=True,
    )
    
    # We run the specific test module with wave dumping enabled
    runner.test(
        hdl_toplevel="sxm",
        test_module="sxm_tb",
        waves=True,
    )

if __name__ == "__main__":
    test_sxm()
