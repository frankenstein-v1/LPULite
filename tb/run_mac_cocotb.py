from pathlib import Path
import os

from cocotb_tools.runner import get_runner


def test_mac():
    src_dir = (Path(__file__).parent / "../src").resolve()
    build_dir = (Path(__file__).parent / "sim_build_mac").resolve()
    sources = [
        src_dir / "mac.sv",
    ]

    runner = get_runner("icarus")
    runner.build(
        sources=sources,
        hdl_toplevel="mac",
        always=True,
        build_dir=build_dir,
        waves=True,
    )
    runner.test(
        hdl_toplevel="mac",
        test_module="mac_tb",
        testcase=os.getenv("TESTCASE"),
        build_dir=build_dir,
        waves=True,
    )


if __name__ == "__main__":
    test_mac()
