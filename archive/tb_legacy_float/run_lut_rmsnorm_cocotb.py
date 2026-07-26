import os
from pathlib import Path
from cocotb_tools.runner import get_runner

def test_lut_rmsnorm():
    src_dir = (Path(__file__).parent / "../src").resolve()
    sources = [
        src_dir / "lut_rmsnorm.sv"
    ]
    
    runner = get_runner("icarus")
    
    runner.build(
        sources=sources,
        includes=[src_dir],
        hdl_toplevel="lut_rmsnorm",
        always=True,
    )
    
    runner.test(
        hdl_toplevel="lut_rmsnorm",
        test_module="lut_rmsnorm_tb",
    )

if __name__ == "__main__":
    test_lut_rmsnorm()
