#!/usr/bin/env python3
"""Run the DE1-SoC HPS/LPU wrapper MicroGPT simulation.

This is the fast pre-Quartus integration test: it drives the same Avalon-MM
register/memory map used by the Linux HPS runtime, loads the generated MEM1 and
VLIW images, runs the static prefix/suffix pages, selects the requested host or
FPGA attention path, and checks the next token for a prompt.
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from cocotb_tools.runner import get_runner


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    simulator = os.getenv("SIM", "icarus")
    runner = get_runner(simulator)

    sources = [
        root / "synthesis" / "rtl" / "lpu_de1_soc_wrapper.sv",
    ]
    sources.extend(sorted((root / "src").glob("*.sv")))
    sources.extend(sorted((root / "src" / "eastbound_bus").glob("*.sv")))
    sources.extend(sorted((root / "src" / "westbound_bus").glob("*.sv")))

    build_dir = root / "synthesis" / "tests" / f"sim_build_microgpt_wrapper_{simulator}"
    runner.build(
        sources=sources,
        includes=[root / "src"],
        hdl_toplevel="lpu_de1_soc_wrapper",
        build_dir=build_dir,
        build_args=["-g2012"] if simulator == "icarus" else [],
        always=True,
        waves=os.getenv("WAVES", "0") == "1",
    )
    results = build_dir / "results.xml"
    results.unlink(missing_ok=True)
    runner.test(
        hdl_toplevel="lpu_de1_soc_wrapper",
        test_module=os.getenv("TEST_MODULE", "microgpt_wrapper_tb"),
        build_dir=build_dir,
        waves=os.getenv("WAVES", "0") == "1",
    )

    if not results.exists():
        print(f"{results}: cocotb did not produce a results file")
        sys.exit(1)
    tree = ET.parse(results)
    failures = tree.findall(".//failure")
    errors = tree.findall(".//error")
    if failures or errors:
        print(f"{results}: {len(failures)} failure(s), {len(errors)} error(s)")
        sys.exit(1)


if __name__ == "__main__":
    main()
