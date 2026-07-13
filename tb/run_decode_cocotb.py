import os
from pathlib import Path
from cocotb_tools.runner import get_runner

def test_decode():
    src_dir = (Path(__file__).parent / "../src").resolve()
    tb_dir = Path(__file__).parent.resolve()
    
    sources = [
        src_dir / "lpu_pkg.sv",
        src_dir / "mac.sv",
        src_dir / "acc.sv",
        src_dir / "mem.sv",
        src_dir / "mxm.sv",
        src_dir / "sxm.sv",
        src_dir / "int_mac.sv",
        src_dir / "cvfpu_fp32_fma.sv",
        src_dir / "cvfpu_fp8_to_fp32_cast.sv",
        src_dir / "row_fifo.sv",
        src_dir / "lut_softmax_exp.sv",
        src_dir / "lut_softmax_div.sv",
        src_dir / "softmax.sv",
        src_dir / "quant.sv",
        src_dir / "lut_layernorm.sv",
        src_dir / "vxm_rope.sv",
        src_dir / "residual_add.sv",
        src_dir / "vxm.sv",
        src_dir / "icu.sv",
        src_dir / "westbound_bus/westbound_bus.sv",
        src_dir / "westbound_bus/westbound_consumer_decode.sv",
        src_dir / "eastbound_bus/eastbound_bus.sv",
        src_dir / "eastbound_bus/eastbound_consumer_decode.sv",
        src_dir / "eastbound_bus/mxm_eastbound_adapter.sv",
        src_dir / "lpu.sv",
        tb_dir / "LPU_tb.sv"
    ]
    
    runner = get_runner("icarus")
    
    runner.build(
        sources=sources,
        includes=[src_dir],
        hdl_toplevel="lpu_cocotb_top",
        always=True,
    )
    
    runner.test(
        hdl_toplevel="lpu_cocotb_top",
        test_module="tb_decode",
    )

if __name__ == "__main__":
    test_decode()
