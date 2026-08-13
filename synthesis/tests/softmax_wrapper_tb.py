"""Exercise the 16-chunk VXM softmax through the HPS wrapper and VLIW path."""

from __future__ import annotations

from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from microgpt_wrapper_tb import (
    AvalonDriver,
    MEM0_BASE,
    pack_quant_row,
    read_hex,
    run_program,
    unpack_row,
)

ROOT = Path(__file__).resolve().parents[2]
SOFTMAX_PATH = ROOT / "model" / "artifacts" / "fpga_microgpt" / "microgpt_softmax_vliw.hex"
SRC_BASE = 600
DST_BASE = 640
CHUNKS = 16


@cocotb.test()
async def test_vxm_softmax_through_wrapper(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    driver = AvalonDriver(dut)
    await driver.init()

    for chunk in range(CHUNKS):
        await driver.write_row(MEM0_BASE, SRC_BASE + chunk, pack_quant_row([0] * 8, 0))

    events = {
        "softmax": 0,
        "residual_start": 0,
        "residual_row": 0,
        "quant_issue": 0,
        "quant_valid": 0,
        "fifo_write": 0,
        "fifo_read": 0,
        "input_overflow": 0,
    }
    monitor_done = False

    def bit(signal) -> int:
        try:
            return int(signal.value)
        except ValueError:
            return 0

    async def monitor() -> None:
        while not monitor_done:
            await RisingEdge(dut.clk)
            events["softmax"] += bit(dut.u_lpu.u_vxm.chunked_softmax_out_valid) & bit(
                dut.u_lpu.u_vxm.chunked_softmax_out_ready
            )
            events["residual_start"] += bit(dut.u_lpu.u_vxm.residual_start)
            events["residual_row"] += bit(dut.u_lpu.u_vxm.residual_row_valid)
            events["quant_issue"] += bit(dut.u_lpu.u_vxm.quant_issue)
            events["quant_valid"] += bit(dut.u_lpu.u_vxm.quantize_valid)
            events["fifo_write"] += bit(dut.u_lpu.vxm_result_wr_en)
            events["fifo_read"] += bit(dut.u_lpu.vxm_result_rd_en)
            events["input_overflow"] |= bit(dut.u_lpu.vxm_input_overflow)

    monitor_task = cocotb.start_soon(monitor())
    program = read_hex(SOFTMAX_PATH)
    await run_program(driver, program, 0, len(program), 900)
    monitor_done = True
    await RisingEdge(dut.clk)
    await monitor_task
    dut._log.info("softmax wrapper events: %s", events)
    dut._log.info(
        "softmax final state=%s softmax_result=%s residual_busy=%s residual_result=%s "
        "quant_inflight=%s stream_valid=%s fifo_count=%s",
        dut.u_lpu.u_vxm.softmax_inst.state_q.value,
        dut.u_lpu.u_vxm.softmax_result_valid.value,
        dut.u_lpu.u_vxm.residual_busy.value,
        dut.u_lpu.u_vxm.residual_result_valid.value,
        dut.u_lpu.u_vxm.quant_inflight.value,
        dut.u_lpu.u_vxm.stream_out_valid_reg.value,
        dut.u_lpu.u_vxm_output_fifo.count.value,
    )

    for chunk in range(CHUNKS):
        lanes, scale = unpack_row(await driver.read_row(MEM0_BASE, DST_BASE + chunk))
        assert lanes == [1] * 8, f"softmax chunk {chunk}: lanes={lanes}, expected eight 1s"
        assert scale == -7, f"softmax chunk {chunk}: scale={scale}, expected -7"
