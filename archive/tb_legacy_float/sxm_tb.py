import cocotb
from cocotb.clock import Clock
from cocotb.triggers import NextTimeStep, RisingEdge, Timer


def pack_superlane(lanes: list[int]) -> int:
    assert len(lanes) == 8
    value = 0
    for idx, lane in enumerate(lanes):
        value |= (lane & 0xFF) << (idx * 8)
    return value


def unpack_superlane(value: int, num_lanes: int = 8) -> list[int]:
    return [(int(value) >> (idx * 8)) & 0xFF for idx in range(num_lanes)]


def unpack_superlane_signed(value: int, num_lanes: int = 8) -> list[int]:
    raw = unpack_superlane(value, num_lanes)
    return [v - 256 if v >= 128 else v for v in raw]


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

    dut.eastbound_in.value = pack_superlane([0xAA, 0xBB, 0xCC, 0xDD, 0x11, 0x22, 0x33, 0x44])
    dut.westbound_in.value = pack_superlane([0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88])
    dut.opcode_input.value = 0
    dut.opcode_weight.value = 0xFFF
    dut.load_from_west.value = 0
    await Timer(1, unit="ns")

    assert int(dut.eastbound_out.value) == 0, "eastbound_out should idle low outside transpose emit"
    assert int(dut.westbound_out.value) == 0, "westbound_out should idle low outside transpose emit"
    assert int(dut.emit_valid.value) == 0, "emit_valid should idle low outside transpose emit"


async def run_transpose_case(dut, *, load_from_west: bool):
    # 8x8 matrix of signed INT8 values
    input_matrix = [
        [  1, -12,   3, -40,  55,  -6,   7,  -8],
        [ -9,  10, -11,  12, -13,  14, -15,  16],
        [ 17, -18,  19, -20,  21, -22,  23, -24],
        [-25,  26, -27,  28, -29,  30, -31,  32],
        [ 33, -34,  35, -36,  37, -38,  39, -40],
        [-41,  42, -43,  44, -45,  46, -47,  48],
        [ 49, -50,  51, -52,  53, -54,  55, -56],
        [-57,  58, -59,  60, -61,  62, -63,  64],
    ]

    expected_transposed = [
        [input_matrix[r][c] for r in range(8)]
        for c in range(8)
    ]

    # Load row 0 with OP_TRANSPOSE_LOAD pulse
    dut.opcode_input.value = 0x5A5
    dut.load_from_west.value = int(load_from_west)
    if load_from_west:
        dut.westbound_in.value = pack_superlane(input_matrix[0])
    else:
        dut.eastbound_in.value = pack_superlane(input_matrix[0])
    await RisingEdge(dut.clk)
    await NextTimeStep()

    # Load remaining 7 rows
    for row_idx in range(1, 8):
        dut.opcode_input.value = 0
        if load_from_west:
            dut.westbound_in.value = pack_superlane(input_matrix[row_idx])
        else:
            dut.eastbound_in.value = pack_superlane(input_matrix[row_idx])
        await RisingEdge(dut.clk)
        await NextTimeStep()

    # Pulse OP_TRANSPOSE_EMIT
    dut.opcode_input.value = 0xA5A
    await Timer(1, unit="ns")
    assert int(dut.emit_valid.value) == 1, "emit_valid should assert on the first emit cycle"
    observed_row0_e = unpack_superlane_signed(int(dut.eastbound_out.value))
    observed_row0_w = unpack_superlane_signed(int(dut.westbound_out.value))
    assert observed_row0_e == expected_transposed[0], (
        f"eastbound row 0 mismatch: expected {expected_transposed[0]}, got {observed_row0_e}"
    )
    assert observed_row0_w == expected_transposed[0], (
        f"westbound row 0 mismatch: expected {expected_transposed[0]}, got {observed_row0_w}"
    )
    await RisingEdge(dut.clk)
    await NextTimeStep()

    # Emit remaining 7 transposed rows
    for col_idx in range(1, 8):
        dut.opcode_input.value = 0
        await Timer(1, unit="ns")
        assert int(dut.emit_valid.value) == 1, f"emit_valid should stay high on emit cycle {col_idx}"
        observed_e = unpack_superlane_signed(int(dut.eastbound_out.value))
        observed_w = unpack_superlane_signed(int(dut.westbound_out.value))
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
