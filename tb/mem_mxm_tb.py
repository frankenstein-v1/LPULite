import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

WB_NONE = 0  # no westbound producer selected
WB_MEM0 = 2  # mem0 selected as the westbound producer

WC_NONE = 0  # no westbound consumer selected
WC_MXM = 1  # MXM selected as westbound consumer
WC_SXM = 2  # SXM selected as westbound consumer

INGRESS_NONE = 0  # MXM ingress disabled
INGRESS_ACT = 1  # treat westbound payload as activation data


def pack_bytes(values):
    """Pack four 8-bit lane values into a 32-bit word, lane 0 in bits [7:0]."""
    word = 0
    for idx, value in enumerate(values):
        word |= (value & 0xFF) << (8 * idx)
    return word


def signed_value(handle):
    """Read a signed logic vector using the cocotb 2.x API."""
    return int(handle.value.to_signed())


async def tick(dut, n=1):
    """Advance simulation by n clock cycles and allow signals to settle."""
    for _ in range(n):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")


async def reset_dut(dut):
    """Drive the wrapper to a known state and release reset cleanly."""
    dut.rst_n.value = 0
    dut.mem_write_en.value = 0
    dut.mem_read_en.value = 0
    dut.mem_addr.value = 0
    dut.mem_write_data.value = 0
    dut.westbound_sel.value = WB_NONE
    dut.westbound_consumer_sel.value = WC_NONE
    dut.mxm_ingress_mode.value = INGRESS_NONE
    dut.mxm_clear.value = 0
    dut.mxm_start.value = 0

    await tick(dut, 2)
    dut.rst_n.value = 1

    # Give the MXM one explicit clear pulse after reset deassertion.
    dut.mxm_clear.value = 1
    await tick(dut, 1)
    dut.mxm_clear.value = 0

    await tick(dut, 1)


async def write_mem_word(dut, addr, values):
    """Synchronously write one 32-bit word into MEM0."""
    assert len(values) == 4, f"expected 4 lanes, got {len(values)}"

    dut.mem_addr.value = addr
    dut.mem_write_data.value = pack_bytes(values)
    dut.mem_write_en.value = 1
    await tick(dut, 1)

    dut.mem_write_en.value = 0
    dut.mem_write_data.value = 0


async def issue_mem0_read_to_mxm_act(dut, addr):
    """
    Read one word from MEM0 and route it westbound into the MXM activation ingress.

    The memory read data and mem0_valid are both registered, so the bus carries
    valid payload after the read edge and the MXM captures it on the following edge.
    """
    dut.mem_addr.value = addr
    dut.westbound_sel.value = WB_MEM0
    dut.westbound_consumer_sel.value = WC_MXM
    dut.mxm_ingress_mode.value = INGRESS_ACT

    dut.mem_read_en.value = 1
    await tick(dut, 1)

    dut.mem_read_en.value = 0
    await tick(dut, 1)


@cocotb.test()
async def test_mem0_westbound_path_loads_act_buffer(dut):
    """A MEM0 westbound read with MXM selected should load the MXM act ingress."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    test_word = [11, 22, 33, 44]
    expected_word = pack_bytes(test_word)

    await reset_dut(dut)
    await write_mem_word(dut, addr=7, values=test_word)

    dut.mem_addr.value = 7
    dut.westbound_sel.value = WB_MEM0
    dut.westbound_consumer_sel.value = WC_MXM
    dut.mxm_ingress_mode.value = INGRESS_ACT

    # First edge performs the synchronous memory read.
    dut.mem_read_en.value = 1
    await tick(dut, 1)

    assert int(dut.westbound_valid_dbg.value) == 1, "westbound bus should be valid after MEM0 read"
    assert int(dut.westbound_payload_dbg.value) == expected_word, "westbound payload mismatch"
    assert int(dut.mxm_west_en_dbg.value) == 1, "MXM consumer enable should assert for WC_MXM"
    assert int(dut.act_loaded_dbg.value) == 0, "MXM should not capture until the next clock edge"

    # Second edge captures the bus word into the MXM activation ingress registers.
    dut.mem_read_en.value = 0
    await tick(dut, 1)

    assert int(dut.act_loaded_dbg.value) == 1, "MXM activation ingress should be marked loaded"
    assert signed_value(dut.act_buf0) == test_word[0], "act_buf0 mismatch"
    assert signed_value(dut.act_buf1) == test_word[1], "act_buf1 mismatch"
    assert signed_value(dut.act_buf2) == test_word[2], "act_buf2 mismatch"
    assert signed_value(dut.act_buf3) == test_word[3], "act_buf3 mismatch"


@cocotb.test()
async def test_mem0_read_Does_not_load_mxm_when_consumner_isnt_mxm(dut):
    """The bus may be valid, but MXM must ignore it if another consumer is selected."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    test_word = [9, 8, 7, 6]
    expected_word = pack_bytes(test_word)

    await reset_dut(dut)
    await write_mem_word(dut, addr=12, values=test_word)

    dut.mem_addr.value = 12
    dut.westbound_sel.value = WB_MEM0
    dut.westbound_consumer_sel.value = WC_SXM
    dut.mxm_ingress_mode.value = INGRESS_ACT

    dut.mem_read_en.value = 1
    await tick(dut, 1)

    assert int(dut.westbound_valid_dbg.value) == 1, "westbound bus should still carry the MEM0 read"
    assert int(dut.westbound_payload_dbg.value) == expected_word, "westbound payload mismatch"
    assert int(dut.mxm_west_en_dbg.value) == 0, "MXM consumer enable must stay low when SXM is selected"
    assert int(dut.act_loaded_dbg.value) == 0, "MXM should not have captured anything yet"

    dut.mem_read_en.value = 0
    await tick(dut, 1)

    assert int(dut.act_loaded_dbg.value) == 0, "MXM activation ingress must remain unloaded"
    assert signed_value(dut.act_buf0) == 0, "act_buf0 should remain cleared"
    assert signed_value(dut.act_buf1) == 0, "act_buf1 should remain cleared"
    assert signed_value(dut.act_buf2) == 0, "act_buf2 should remain cleared"
    assert signed_value(dut.act_buf3) == 0, "act_buf3 should remain cleared"
