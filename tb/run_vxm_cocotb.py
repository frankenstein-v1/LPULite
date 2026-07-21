from pathlib import Path
import os
from cocotb_tools.runner import get_runner

def test_vxm():
    src_dir = (Path(__file__).parent / "../src").resolve()
    tb_dir = Path(__file__).parent.resolve()
    sim = os.getenv("SIM", "verilator")
    build_dir = (tb_dir / f"sim_build_vxm_{sim}").resolve()

    sources = [
        src_dir / "lpu_pkg.sv",
        src_dir / "lut_rmsnorm.sv",
        src_dir / "rmsnorm.sv",
        src_dir / "vxm_rope.sv",
        src_dir / "residual_add.sv",
        src_dir / "vxm.sv",
        src_dir / "softmax.sv",
        src_dir / "quant.sv",
        src_dir / "lut_softmax_exp.sv",
        src_dir / "lut_softmax_div.sv"
    ]

    runner = get_runner(sim)

    build_kwargs = dict(
        sources=sources,
        includes=[src_dir],
        hdl_toplevel="vxm",
        always=True,
        build_dir=build_dir,
    )

    if sim == "verilator":
        build_kwargs["build_args"] = ["--Wno-fatal"]
        build_kwargs["waves"] = True
    
    runner.build(**build_kwargs)

    runner.test(
        hdl_toplevel="vxm",
        test_module="vxm_tb",
        testcase=os.getenv("TESTCASE"),
        build_dir=build_dir,
        waves=(sim == "verilator"),
    )

if __name__ == "__main__":
    test_vxm()
