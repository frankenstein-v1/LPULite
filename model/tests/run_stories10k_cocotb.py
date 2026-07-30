import os
import logging
from pathlib import Path
from cocotb_tools.runner import get_runner

logging.basicConfig(level=logging.INFO)

def run_stories10k_inference():
    root_dir = Path(__file__).resolve().parents[2]
    src_dir = root_dir / "src"
    tb_dir = Path(__file__).parent.resolve()
    sim = os.getenv("SIM", "icarus")
    build_dir = (tb_dir / f"sim_build_stories10k_{sim}").resolve()
    
    sources = [
        src_dir / "lpu_pkg.sv",
        src_dir / "mac.sv",
        src_dir / "acc.sv",
        src_dir / "mem.sv",
        src_dir / "mem_row_dequant.sv",
        src_dir / "mxm.sv",
        src_dir / "sxm.sv",
        src_dir / "row_fifo.sv",
        src_dir / "lut_rmsnorm.sv",
        src_dir / "lut_softmax_exp.sv",
        src_dir / "lut_softmax_div.sv",
        src_dir / "softmax.sv",
        src_dir / "quant.sv",
        src_dir / "rmsnorm.sv",
        src_dir / "vxm_rope.sv",
        src_dir / "residual_add.sv",
        src_dir / "vxm.sv",
        src_dir / "icu.sv",
        src_dir / "shared_bus_mux.sv",
        src_dir / "westbound_bus/westbound_bus.sv",
        src_dir / "westbound_bus/westbound_consumer_decode.sv",
        src_dir / "eastbound_bus/eastbound_bus.sv",
        src_dir / "eastbound_bus/eastbound_consumer_decode.sv",
        src_dir / "eastbound_bus/mxm_eastbound_adapter.sv",
        src_dir / "lpu.sv",
        root_dir / "tb" / "LPU_tb.sv"
    ]
    
    runner = get_runner(sim)
    
    build_args = ["-g2012"]
    
    runner.build(
        sources=sources,
        includes=[src_dir],
        hdl_toplevel="lpu_cocotb_top",
        always=True,
        waves=False,
        build_args=build_args,
        build_dir=build_dir,
        verbose=True,
    )
    
    runner.test(
        hdl_toplevel="lpu_cocotb_top",
        test_module="stories10k_tb",
        testcase="test_stories10k_continuous_inference",
        build_dir=build_dir,
        waves=False,
    )

if __name__ == "__main__":
    run_stories10k_inference()
