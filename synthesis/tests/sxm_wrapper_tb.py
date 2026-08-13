"""Focused SXM broadcast test through the synthesizable HPS/LPU wrapper."""

from __future__ import annotations

import json

import cocotb
from cocotb.clock import Clock

from microgpt_wrapper_tb import (
    ARTIFACT_DIR,
    AvalonDriver,
    MEM0_BASE,
    SCHEDULE_PATH,
    VLIW_PATH,
    find_trace_pc,
    pack_quant_row,
    read_hex,
    run_program,
    unpack_row,
)


@cocotb.test()
async def test_sxm_broadcast_through_wrapper(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    driver = AvalonDriver(dut)
    await driver.init()

    schedule = json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))
    trace = json.loads((ARTIFACT_DIR / "microgpt_decode_trace.json").read_text(encoding="utf-8"))
    vliw = read_hex(VLIW_PATH)
    start = find_trace_pc(trace, "broadcast attention input row0")
    end = find_trace_pc(trace, "broadcast attention input row1")
    page_size = int(schedule["imem"]["page_size"])

    source_lanes = [-91, -37, -5, 0, 7, 29, 63, 111]
    source_scale = -6
    await driver.write_row(MEM0_BASE, 10, pack_quant_row(source_lanes, source_scale))
    await run_program(driver, vliw, start, end - start, page_size)

    for lane, expected in enumerate(source_lanes):
        got_lanes, got_scale = unpack_row(await driver.read_row(MEM0_BASE, 32 + lane))
        assert got_lanes == [expected] * 8, (
            f"broadcast lane {lane}: got {got_lanes}, expected {[expected] * 8}"
        )
        assert got_scale == source_scale, (
            f"broadcast lane {lane}: scale {got_scale}, expected {source_scale}"
        )
