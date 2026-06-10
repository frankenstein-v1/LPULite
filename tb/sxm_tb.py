import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, NextTimeStep


def pack_superlane(lanes: list[int]) -> int:
    assert len(lanes) == 4
    value = 0
    for idx, lane in enumerate(lanes):
        value |= (lane & 0xFF) << (idx * 8)
    return value


def unpack_superlane(value: int) -> list[int]:
    return [(int(value) >> (idx * 8)) & 0xFF for idx in range(4)]


async def reset_dut(dut) -> None:
    dut.rst_n.value = 0
    dut.opcode_input.value = 0
    dut.opcode_weight.value = 0
    dut.load_from_west.value = 0
    dut.eastbound_in.value = 0
    dut.westbound_in.value = 0
    await Timer(20, unit="ns")
    dut.rst_n.value = 1
    await Timer(10, unit="ns")


@cocotb.test()
async def test_sxm_idle_outputs_zero(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    await reset_dut(dut)

    dut.eastbound_in.value = pack_superlane([0xAA, 0xBB, 0xCC, 0xDD])
    dut.westbound_in.value = pack_superlane([0x11, 0x22, 0x33, 0x44])
    dut.opcode_input.value = 0
    dut.opcode_weight.value = 0xFFF
    dut.load_from_west.value = 0
    await Timer(1, unit="ns")

    assert int(dut.eastbound_out.value) == 0, "eastbound_out should idle low outside transpose emit"
    assert int(dut.westbound_out.value) == 0, "westbound_out should idle low outside transpose emit"
    assert int(dut.emit_valid.value) == 0, "emit_valid should idle low outside transpose emit"


async def run_transpose_case(dut, *, load_from_west: bool):
    input_matrix = [
        [0x01, 0x02, 0x03, 0x04],
        [0x05, 0x06, 0x07, 0x08],
        [0x09, 0x0A, 0x0B, 0x0C],
        [0x0D, 0x0E, 0x0F, 0x10],
    ]

    expected_transposed = [
        [0x01, 0x05, 0x09, 0x0D],
        [0x02, 0x06, 0x0A, 0x0E],
        [0x03, 0x07, 0x0B, 0x0F],
        [0x04, 0x08, 0x0C, 0x10],
    ]

    dut.opcode_input.value = 0x5A5
    dut.load_from_west.value = int(load_from_west)
    if load_from_west:
        dut.westbound_in.value = pack_superlane(input_matrix[0])
    else:
        dut.eastbound_in.value = pack_superlane(input_matrix[0])
    await RisingEdge(dut.clk)
    await NextTimeStep()

    for row_idx in range(1, 4):
        dut.opcode_input.value = 0
        if load_from_west:
            dut.westbound_in.value = pack_superlane(input_matrix[row_idx])
        else:
            dut.eastbound_in.value = pack_superlane(input_matrix[row_idx])
        await RisingEdge(dut.clk)
        await NextTimeStep()

    dut.opcode_input.value = 0xA5A
    await Timer(1, unit="ns")
    assert int(dut.emit_valid.value) == 1, "emit_valid should assert on the first emit cycle"
    observed_row0_e = unpack_superlane(int(dut.eastbound_out.value))
    observed_row0_w = unpack_superlane(int(dut.westbound_out.value))
    assert observed_row0_e == expected_transposed[0], (
        f"eastbound row 0 mismatch: expected {expected_transposed[0]}, got {observed_row0_e}"
    )
    assert observed_row0_w == expected_transposed[0], (
        f"westbound row 0 mismatch: expected {expected_transposed[0]}, got {observed_row0_w}"
    )
    await RisingEdge(dut.clk)
    await NextTimeStep()

    for col_idx in range(1, 4):
        dut.opcode_input.value = 0
        await Timer(1, unit="ns")
        assert int(dut.emit_valid.value) == 1, f"emit_valid should stay high on emit cycle {col_idx}"
        observed_e = unpack_superlane(int(dut.eastbound_out.value))
        observed_w = unpack_superlane(int(dut.westbound_out.value))
        assert observed_e == expected_transposed[col_idx], (
            f"eastbound row {col_idx} mismatch: expected {expected_transposed[col_idx]}, got {observed_e}"
        )
        assert observed_w == expected_transposed[col_idx], (
            f"westbound row {col_idx} mismatch: expected {expected_transposed[col_idx]}, got {observed_w}"
        )
        await RisingEdge(dut.clk)
        await NextTimeStep()

    await Timer(1, unit="ns")
    assert int(dut.emit_valid.value) == 0, "emit_valid should deassert after the final emit cycle"


@cocotb.test()
async def test_sxm_transpose_from_eastbound(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    await reset_dut(dut)
    await run_transpose_case(dut, load_from_west=False)


@cocotb.test()
async def test_sxm_transpose_from_westbound(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    await reset_dut(dut)
    await run_transpose_case(dut, load_from_west=True)
