from pathlib import Path
import os

from cocotb_tools.runner import get_runner


def test_mxm8():
    src_dir = (Path(__file__).parent / "../src").resolve()
    tb_dir = Path(__file__).parent.resolve()
    build_dir = (tb_dir / "sim_build_mxm8").resolve()
    sources = [
        src_dir / "mac.sv",
        src_dir / "mxm.sv",
        tb_dir / "mxm_8x8_cocotb_top.sv",
    ]

    runner = get_runner("icarus")
    runner.build(
        sources=sources,
        hdl_toplevel="mxm_8x8_cocotb_top",
        always=True,
        build_dir=build_dir,
        waves=True,
    )
    runner.test(
        hdl_toplevel="mxm_8x8_cocotb_top",
        test_module="mxm_8x8_tb",
        testcase=os.getenv("TESTCASE"),
        build_dir=build_dir,
        waves=True,
    )


if __name__ == "__main__":
    test_mxm8()
