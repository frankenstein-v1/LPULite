#!/usr/bin/env python3

import os
from pathlib import Path

from cocotb_tools.runner import get_runner


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    tests = Path(__file__).resolve().parent
    simulator = os.getenv("SIM", "icarus")
    sources = [src / "mac.sv", src / "mxm.sv", tests / "microgpt_lpu_top.sv"]
    runner = get_runner(simulator)
    build_dir = tests / f"sim_build_microgpt_lpu_{simulator}"
    runner.build(
        sources=sources,
        includes=[src],
        hdl_toplevel="microgpt_lpu_top",
        build_dir=build_dir,
        build_args=["-g2012"] if simulator == "icarus" else [],
        always=True,
        waves=False,
    )
    runner.test(
        hdl_toplevel="microgpt_lpu_top",
        test_module="microgpt_lpu_tb",
        build_dir=build_dir,
        waves=False,
    )


if __name__ == "__main__":
    main()
