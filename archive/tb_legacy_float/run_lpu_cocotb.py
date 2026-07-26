import os
import logging
from pathlib import Path
from cocotb_tools.runner import get_runner

logging.basicConfig(level=logging.INFO)

def test_lpu():
    src_dir = (Path(__file__).parent / "../src").resolve()
    tb_dir = Path(__file__).parent.resolve()
    
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
        tb_dir / "LPU_tb.sv"
    ]
    
    runner = get_runner("icarus")
    
    build_args = ["-g2012"]
    rmsnorm_chunks = os.getenv("LPU_RMSNORM_CHUNKS")
    if rmsnorm_chunks:
        build_args.append(f"-Plpu_cocotb_top.RMSNORM_CHUNKS={int(rmsnorm_chunks)}")
    softmax_chunks = os.getenv("LPU_SOFTMAX_CHUNKS")
    if softmax_chunks:
        build_args.append(f"-Plpu_cocotb_top.SOFTMAX_CHUNKS={int(softmax_chunks)}")

    runner.build(
        sources=sources,
        includes=[src_dir],
        hdl_toplevel="lpu_cocotb_top",
        always=True,
        waves=True,
        build_args=build_args,
        verbose=True,
    )
    
    testcase = os.getenv("TESTCASE")
    default_module = "lpu_tb"
    if testcase:
        lpu_test_path = tb_dir / "lpu_test.py"
        if lpu_test_path.exists():
            try:
                with open(lpu_test_path, "r", encoding="utf-8") as f:
                    if f"async def {testcase}" in f.read():
                        default_module = "lpu_test"
            except Exception:
                pass

    runner.test(
        hdl_toplevel="lpu_cocotb_top",
        test_module=os.getenv("TEST_MODULE", default_module),
        testcase=testcase,
        waves=True,
    )

if __name__ == "__main__":
    test_lpu()
