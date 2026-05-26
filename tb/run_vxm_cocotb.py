import os
from pathlib import Path
from cocotb_tools.runner import get_runner

def test_vxm():
    # Source files
    src_dir = (Path(__file__).parent / "../src").resolve()
    sources = [
        src_dir / "vxm.sv",
        src_dir / "softmax.sv",
        src_dir / "quant.sv",
        src_dir / "lut_softmax_exp.sv",
        src_dir / "lut_softmax_div.sv"
    ]
    
    runner = get_runner("icarus")
    
    # We build the simulation
    runner.build(
        sources=sources,
        hdl_toplevel="vxm",
        always=True,
    )
    
    # We run the specific test module
    runner.test(
        hdl_toplevel="vxm",
        test_module="vxm_tb",
    )

if __name__ == "__main__":
    test_vxm()
