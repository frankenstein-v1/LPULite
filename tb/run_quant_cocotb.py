from pathlib import Path
import os

from cocotb_tools.runner import get_runner


def test_quant():
    src_dir = (Path(__file__).parent / "../src").resolve()
    build_dir = (Path(__file__).parent / "sim_build_quant").resolve()
    sources = [
        src_dir / "quant.sv",
    ]

    runner = get_runner("icarus")
    runner.build(
        sources=sources,
        hdl_toplevel="quant",
        always=True,
        build_dir=build_dir,
    )
    runner.test(
        hdl_toplevel="quant",
        test_module="quant_tb",
        testcase=os.getenv("TESTCASE"),
        build_dir=build_dir,
    )


if __name__ == "__main__":
    test_quant()
