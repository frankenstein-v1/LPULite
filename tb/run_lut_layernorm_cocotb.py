import os
from pathlib import Path
from cocotb_tools.runner import get_runner

def test_lut_layernorm():
    src_dir = (Path(__file__).parent / "../src").resolve()
    sources = [
        src_dir / "lut_layernorm.sv"
    ]
    
    runner = get_runner("icarus")
    
    # We compile only the LayerNorm module as top-level
    runner.build(
        sources=sources,
        hdl_toplevel="lut_layernorm",
        always=True,
    )
    
    # We run the test module
    runner.test(
        hdl_toplevel="lut_layernorm",
        test_module="lut_layernorm_tb",
    )

if __name__ == "__main__":
    test_lut_layernorm()
