from pathlib import Path
import os

from cocotb_tools.runner import get_runner


def test_softmax():
    src_dir = (Path(__file__).parent / "../src").resolve()
    build_dir = (Path(__file__).parent / "sim_build_softmax_fp").resolve()
    sources = [
        src_dir / "cvfpu_fp32_addsub.sv",
        src_dir / "cvfpu_fp32_div.sv",
        src_dir / "cvfpu_fp32_cmp.sv",
        src_dir / "softmax.sv",
        src_dir / "lut_softmax_exp.sv",
        src_dir / "lut_softmax_div.sv",
    ]

    runner = get_runner("icarus")
    runner.build(
        sources=sources,
        hdl_toplevel="softmax",
        always=True,
        build_dir=build_dir,
    )
    runner.test(
        hdl_toplevel="softmax",
        test_module="softmax_fp_tb",
        testcase=os.getenv("TESTCASE"),
        build_dir=build_dir,
    )


if __name__ == "__main__":
    test_softmax()
