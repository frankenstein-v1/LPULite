"""Focused resident-MXM attention microkernel checks through the HPS wrapper."""

from __future__ import annotations

import json

import cocotb
from cocotb.clock import Clock

from microgpt_wrapper_tb import (
    ARTIFACT_DIR,
    ATTENTION_VLIW_PATH,
    CTRL_PC_LOAD,
    MEM0_BASE,
    MEM1_BASE,
    AvalonDriver,
    load_imem_page,
    pack_quant_row,
    read_hex,
    unpack_row,
)


@cocotb.test()
async def test_attention_k_transpose_uses_sxm(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    driver = AvalonDriver(dut)
    await driver.init()

    schedule = json.loads(
        (ARTIFACT_DIR / "microgpt_decode_schedule.json").read_text(encoding="utf-8")
    )
    abi = schedule["mem0_abi"]
    meta = schedule["microkernels"]["attention"]
    sections = meta["sections"]
    image = read_hex(ATTENTION_VLIW_PATH)
    await load_imem_page(driver, image, 0, len(image))
    await driver.write32(CTRL_PC_LOAD, len(image) - 1)

    src_base = int(abi["attention_k_tile_input_rows"][0])
    dst_base = int(meta["mem1_k_transpose_stage_rows"][0])
    source = [[row * 8 + lane - 31 for lane in range(8)] for row in range(8)]
    for block in range(2):
        for row, lanes in enumerate(source):
            await driver.write_row(MEM0_BASE, src_base + row, pack_quant_row(lanes, -4))

        section = sections[f"k_transpose_block{block}"]
        requested = int(section["instructions"])
        actual = await driver.run_cycles(requested, int(section["start"]))
        assert actual == requested
        for dim in range(4):
            lanes, scale = unpack_row(
                await driver.read_row(MEM1_BASE, dst_base + dim * 2 + block)
            )
            expected = [source[pos][dim] for pos in range(8)]
            got_values = [float(value) * (2.0 ** scale) for value in lanes]
            expected_values = [float(value) * (2.0 ** -4) for value in expected]
            assert got_values == expected_values, (
                f"SXM K^T block {block} dim {dim}: got {got_values}, "
                f"expected {expected_values} (packed lanes={lanes}, scale={scale})"
            )


@cocotb.test()
async def test_attention_qk_uses_mxm(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    driver = AvalonDriver(dut)
    await driver.init()

    schedule = json.loads(
        (ARTIFACT_DIR / "microgpt_decode_schedule.json").read_text(encoding="utf-8")
    )
    abi = schedule["mem0_abi"]
    meta = schedule["microkernels"]["attention"]
    section = meta["sections"]["qk"]
    image = read_hex(ATTENTION_VLIW_PATH)
    await load_imem_page(driver, image, 0, len(image))
    await driver.write32(CTRL_PC_LOAD, len(image) - 1)

    q_base = int(abi["attention_q_broadcast_rows"][0])
    score_base = int(abi["attention_qk_score_rows"][0])
    kt_base = int(meta["mem1_k_transpose_stage_rows"][0])
    for dim, q in enumerate([1, 2, 3, 4]):
        await driver.write_row(MEM0_BASE, q_base + dim, pack_quant_row([q] * 8, 0))
        for block in range(2):
            factors = [block * 8 + lane + 1 for lane in range(8)]
            await driver.write_row(MEM1_BASE, kt_base + dim * 2 + block, pack_quant_row(factors, 0))

    for dim, q in enumerate([1, 2, 3, 4]):
        assert await driver.read_row(MEM0_BASE, q_base + dim) == pack_quant_row([q] * 8, 0)
    for dim in range(4):
        for block in range(2):
            factors = [block * 8 + lane + 1 for lane in range(8)]
            assert await driver.read_row(MEM1_BASE, kt_base + dim * 2 + block) == pack_quant_row(factors, 0)

    requested = int(section["instructions"])
    actual = await driver.run_cycles(requested, int(section["start"]))
    assert actual == requested

    got: list[float] = []
    for block in range(2):
        readback = await driver.read_row(MEM0_BASE, score_base + block)
        lanes, scale = unpack_row(readback)
        got.extend(float(value) * (2.0 ** scale) for value in lanes)
    expected = [10.0 * (pos + 1) for pos in range(16)]
    dut._log.info("focused MXM QK got=%s expected=%s", got, expected)
    for actual_value, expected_value in zip(got, expected):
        assert abs(actual_value - expected_value) <= 2.0
