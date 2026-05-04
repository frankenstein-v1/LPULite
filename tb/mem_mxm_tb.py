import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

WB_NONE = 0  # no westbound producer selected
WB_MEM0 = 2  # mem0 selected as the westbound producer

WC_NONE = 0  # no westbound consumer selected
WC_MXM = 1  # MXM selected as westbound consumer
WC_SXM = 2  # SXM selected as westbound consumer

INGRESS_NONE = 0  # MXM ingress disabled
INGRESS_INPUT = 1  # treat westbound payload as input data
INGRESS_WGHT = 2

WB_MEM1 = 4


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
    dut.mem1_write_en.value = 0
    dut.mem1_read_en.value = 0
    dut.mem1_addr.value = 0
    dut.mem_write_data.value = 0
    dut.mem1_write_data.value = 0
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

async def write_mem1_word(dut, addr, values):
    """Synchronously write one 32-bit word into MEM1."""
    assert len(values) == 4, f"expected 4 lanes, got {len(values)}"

    dut.mem1_addr.value = addr
    dut.mem1_write_data.value = pack_bytes(values)
    dut.mem1_write_en.value = 1
    await tick(dut, 1)

    dut.mem1_write_en.value = 0
    dut.mem1_write_data.value = 0


async def issue_mem1_read_to_mxm_wght(dut, addr):
    """Read one MEM1 word westbound into the MXM weight ingress."""
    dut.mem1_addr.value = addr
    dut.westbound_sel.value = WB_MEM1
    dut.westbound_consumer_sel.value = WC_MXM
    dut.mxm_ingress_mode.value = INGRESS_WGHT

    dut.mem1_read_en.value = 1
    await tick(dut, 1)

    dut.mem1_read_en.value = 0
    await tick(dut, 1)


def read_mxm_matrix(dut):
    """Read the exposed 4x4 MXM output matrix from debug taps."""
    return [
        [
            signed_value(dut.mxm_out_00_dbg),
            signed_value(dut.mxm_out_01_dbg),
            signed_value(dut.mxm_out_02_dbg),
            signed_value(dut.mxm_out_03_dbg),
        ],
        [
            signed_value(dut.mxm_out_10_dbg),
            signed_value(dut.mxm_out_11_dbg),
            signed_value(dut.mxm_out_12_dbg),
            signed_value(dut.mxm_out_13_dbg),
        ],
        [
            signed_value(dut.mxm_out_20_dbg),
            signed_value(dut.mxm_out_21_dbg),
            signed_value(dut.mxm_out_22_dbg),
            signed_value(dut.mxm_out_23_dbg),
        ],
        [
            signed_value(dut.mxm_out_30_dbg),
            signed_value(dut.mxm_out_31_dbg),
            signed_value(dut.mxm_out_32_dbg),
            signed_value(dut.mxm_out_33_dbg),
        ],
    ]


async def issue_mem0_read_to_mxm_input(dut, addr):
    """
    Read one word from MEM0 and route it westbound into the MXM input ingress.

    The memory read data and mem0_valid are both registered, so the bus carries
    valid payload after the read edge and the MXM captures it on the following edge.
    """
    dut.mem_addr.value = addr
    dut.westbound_sel.value = WB_MEM0
    dut.westbound_consumer_sel.value = WC_MXM
    dut.mxm_ingress_mode.value = INGRESS_INPUT

    dut.mem_read_en.value = 1
    await tick(dut, 1)

    dut.mem_read_en.value = 0
    await tick(dut, 1)


@cocotb.test()
async def test_mem0_westbound_path_loads_input_buffer(dut):
    """A MEM0 westbound read with MXM selected should load the MXM input ingress."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    test_word = [11, 22, 33, 44]
    expected_word = pack_bytes(test_word)

    await reset_dut(dut)
    await write_mem_word(dut, addr=7, values=test_word)

    dut.mem_addr.value = 7
    dut.westbound_sel.value = WB_MEM0
    dut.westbound_consumer_sel.value = WC_MXM
    dut.mxm_ingress_mode.value = INGRESS_INPUT

    # First edge performs the synchronous memory read.
    dut.mem_read_en.value = 1
    await tick(dut, 1)

    assert int(dut.westbound_valid_dbg.value) == 1, "westbound bus should be valid after MEM0 read"
    assert int(dut.westbound_payload_dbg.value) == expected_word, "westbound payload mismatch"
    assert int(dut.mxm_west_en_dbg.value) == 1, "MXM consumer enable should assert for WC_MXM"
    assert int(dut.input_loaded_dbg.value) == 0, "MXM should not capture until the next clock edge"

    # Second edge captures the bus word into the MXM input ingress registers.
    dut.mem_read_en.value = 0
    await tick(dut, 1)

    assert int(dut.input_loaded_dbg.value) == 1, "MXM input ingress should be marked loaded"
    assert signed_value(dut.input_buf0) == test_word[0], "input_buf0 mismatch"
    assert signed_value(dut.input_buf1) == test_word[1], "input_buf1 mismatch"
    assert signed_value(dut.input_buf2) == test_word[2], "input_buf2 mismatch"
    assert signed_value(dut.input_buf3) == test_word[3], "input_buf3 mismatch"


@cocotb.test()
async def mem1_to_mxm(dut):
    """A MEM1 westbound read with MXM selected should load the MXM weight ingress."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    weights = [10, 20, 30, 40]
    expected_word = pack_bytes(weights)

    await reset_dut(dut)
    await write_mem1_word(dut, addr=7, values=weights)

    dut.mem1_addr.value = 7
    dut.westbound_sel.value = WB_MEM1
    dut.westbound_consumer_sel.value = WC_MXM
    dut.mxm_ingress_mode.value = INGRESS_WGHT

    # First edge performs the synchronous memory read.
    dut.mem1_read_en.value = 1
    await tick(dut, 1)

    assert int(dut.westbound_valid_dbg.value) == 1, "westbound bus should be valid after MEM1 read"
    assert int(dut.westbound_payload_dbg.value) == expected_word, "westbound payload mismatch"
    assert int(dut.mxm_west_en_dbg.value) == 1, "MXM consumer enable should assert for WC_MXM"
    assert int(dut.wght_loaded_dbg.value) == 0, "MXM should not capture until the next clock edge"

    # Second edge captures the bus word into the MXM weight ingress registers.
    dut.mem1_read_en.value = 0
    await tick(dut, 1)

    assert int(dut.wght_loaded_dbg.value) == 1, "MXM weight ingress should be marked loaded"
    assert signed_value(dut.wght_buf0) == weights[0], "wght_buf0 mismatch"
    assert signed_value(dut.wght_buf1) == weights[1], "wght_buf1 mismatch"
    assert signed_value(dut.wght_buf2) == weights[2], "wght_buf2 mismatch"
    assert signed_value(dut.wght_buf3) == weights[3], "wght_buf3 mismatch"

@cocotb.test()
async def inputs_weights_loaded(dut):
    """Load MEM1 weights and MEM0 inputs into MXM without starting compute."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    
    #initialize weights and inputs
    weights = [55,66,77,88]
    inputs = [11,22,33,44]

    expected_word_weights = pack_bytes(weights)
    expected_word_inputs = pack_bytes(inputs)


    #pack weights into a word
    await reset_dut(dut)
    await write_mem1_word(dut, addr=7, values=weights)
    await write_mem_word(dut, addr=7, values=inputs)

    dut.mem1_addr.value = 7
    dut.westbound_sel.value = WB_MEM1
    dut.westbound_consumer_sel.value = WC_MXM
    dut.mxm_ingress_mode.value = INGRESS_WGHT

    dut.mem1_read_en.value = 1
    await tick(dut, 1)

    assert int(dut.westbound_valid_dbg.value) == 1, "westbound bus should be valid after MEM1 read"
    assert int(dut.westbound_payload_dbg.value) == expected_word_weights, "westbound payload mismatch"
    assert int(dut.mxm_west_en_dbg.value) == 1, "MXM consumer enable should assert for WC_MXM"
    assert int(dut.wght_loaded_dbg.value) == 0, "MXM should not capture until the next clock edge"

    dut.mem1_read_en.value = 0
    await tick(dut, 1)

    assert int(dut.wght_loaded_dbg.value) == 1, "MXM weight ingress should be marked loaded"
    assert signed_value(dut.wght_buf0) == weights[0], "wght_buf0 mismatch"
    assert signed_value(dut.wght_buf1) == weights[1], "wght_buf1 mismatch"
    assert signed_value(dut.wght_buf2) == weights[2], "wght_buf2 mismatch"
    assert signed_value(dut.wght_buf3) == weights[3], "wght_buf3 mismatch"

    dut.mem_addr.value = 7
    dut.westbound_sel.value = WB_MEM0
    dut.westbound_consumer_sel.value = WC_MXM
    dut.mxm_ingress_mode.value = INGRESS_INPUT

    dut.mem_read_en.value = 1
    await tick(dut, 1)

    assert int(dut.westbound_valid_dbg.value) == 1, "westbound bus should be valid after MEM0 read"
    assert int(dut.westbound_payload_dbg.value) == expected_word_inputs, "westbound payload mismatch"
    assert int(dut.mxm_west_en_dbg.value) == 1, "MXM consumer enable should assert for WC_MXM"
    assert int(dut.input_loaded_dbg.value) == 0, "MXM should not capture until the next clock edge"

    # Second edge captures the bus word into the MXM input ingress registers.
    dut.mem_read_en.value = 0
    await tick(dut, 1)

    assert int(dut.input_loaded_dbg.value) == 1, "MXM input ingress should be marked loaded"
    assert signed_value(dut.input_buf0) == inputs[0], "input_buf0 mismatch"
    assert signed_value(dut.input_buf1) == inputs[1], "input_buf1 mismatch"
    assert signed_value(dut.input_buf2) == inputs[2], "input_buf2 mismatch"
    assert signed_value(dut.input_buf3) == inputs[3], "input_buf3 mismatch"






@cocotb.test()
async def mxm_computation(dut):
    """Load MEM1 weights and MEM0 inputs, then verify the first MXM output."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    
    #initialize weights and inputs
    weights = [3,0,0,0]
    inputs = [2,0,0,0]

    expected_word_weights = pack_bytes(weights)
    expected_word_inputs = pack_bytes(inputs)


    #pack weights into a word
    await reset_dut(dut)
    await write_mem1_word(dut, addr=7, values=weights)
    await write_mem_word(dut, addr=7, values=inputs)

    dut.mem1_addr.value = 7
    dut.westbound_sel.value = WB_MEM1
    dut.westbound_consumer_sel.value = WC_MXM
    dut.mxm_ingress_mode.value = INGRESS_WGHT

    dut.mem1_read_en.value = 1
    await tick(dut, 1)

    assert int(dut.westbound_valid_dbg.value) == 1, "westbound bus should be valid after MEM1 read"
    assert int(dut.westbound_payload_dbg.value) == expected_word_weights, "westbound payload mismatch"
    assert int(dut.mxm_west_en_dbg.value) == 1, "MXM consumer enable should assert for WC_MXM"
    assert int(dut.wght_loaded_dbg.value) == 0, "MXM should not capture weights until the next clock edge"

    dut.mem1_read_en.value = 0
    await tick(dut, 1)

    assert int(dut.wght_loaded_dbg.value) == 1, "MXM weight ingress should be marked loaded"
    assert signed_value(dut.wght_buf0) == weights[0], "wght_buf0 mismatch"
    assert signed_value(dut.wght_buf1) == weights[1], "wght_buf1 mismatch"
    assert signed_value(dut.wght_buf2) == weights[2], "wght_buf2 mismatch"
    assert signed_value(dut.wght_buf3) == weights[3], "wght_buf3 mismatch"

    dut.mem_addr.value = 7
    dut.westbound_sel.value = WB_MEM0
    dut.westbound_consumer_sel.value = WC_MXM
    dut.mxm_ingress_mode.value = INGRESS_INPUT

    dut.mem_read_en.value = 1
    await tick(dut, 1)

    assert int(dut.westbound_valid_dbg.value) == 1, "westbound bus should be valid after MEM0 read"
    assert int(dut.westbound_payload_dbg.value) == expected_word_inputs, "westbound payload mismatch"
    assert int(dut.mxm_west_en_dbg.value) == 1, "MXM consumer enable should assert for WC_MXM"
    assert int(dut.input_loaded_dbg.value) == 0, "MXM should not capture until the next clock edge"

    # Second edge captures the bus word into the MXM input ingress registers.
    dut.mem_read_en.value = 0
    await tick(dut, 1)

    assert int(dut.input_loaded_dbg.value) == 1, "MXM input ingress should be marked loaded"
    assert signed_value(dut.input_buf0) == inputs[0], "input_buf0 mismatch"
    assert signed_value(dut.input_buf1) == inputs[1], "input_buf1 mismatch"
    assert signed_value(dut.input_buf2) == inputs[2], "input_buf2 mismatch"
    assert signed_value(dut.input_buf3) == inputs[3], "input_buf3 mismatch"

    expected_output = weights[0] * inputs[0]

    assert signed_value(dut.mxm_out_00_dbg) == 0, "MXM output should start cleared"

    dut.mxm_start.value = 1
    await tick(dut, 1)

    assert signed_value(dut.mxm_out_00_dbg) == 0, "first compute cycle should only form the MAC product"

    await tick(dut, 1)
    dut.mxm_start.value = 0

    assert signed_value(dut.mxm_out_00_dbg) == expected_output, "mxm_out_00_dbg mismatch"


@cocotb.test()
async def mxm_4x4_matmul(dut):
    """Run a full 4x4-by-4x4 matmul by streaming one k-slice per iteration."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    a_matrix = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [2, 0, 1, 3],
        [4, 1, 0, 2],
    ]
    b_matrix = [
        [1, 0, 2, 1],
        [0, 1, 1, 0],
        [3, 1, 0, 2],
        [2, 1, 1, 1],
    ]

    expected = [[0 for _ in range(4)] for _ in range(4)]
    for row in range(4):
        for col in range(4):
            for k_idx in range(4):
                expected[row][col] += a_matrix[row][k_idx] * b_matrix[k_idx][col]

    await reset_dut(dut)
    for k_idx in range(4):
        input_vector = [a_matrix[row][k_idx] for row in range(4)]
        wght_vector = [b_matrix[k_idx][col] for col in range(4)]

        await write_mem1_word(dut, addr=k_idx, values=wght_vector)
        await write_mem_word(dut, addr=k_idx, values=input_vector)

    for k_idx in range(4):
        input_vector = [a_matrix[row][k_idx] for row in range(4)]
        wght_vector = [b_matrix[k_idx][col] for col in range(4)]

        dut.mem1_addr.value = k_idx
        dut.westbound_sel.value = WB_MEM1
        dut.westbound_consumer_sel.value = WC_MXM
        dut.mxm_ingress_mode.value = INGRESS_WGHT
        dut.mem1_read_en.value = 1
        await tick(dut, 1)

        assert int(dut.westbound_valid_dbg.value) == 1, f"MEM1 westbound valid missing for k={k_idx}"
        assert int(dut.westbound_payload_dbg.value) == pack_bytes(wght_vector), f"MEM1 payload mismatch for k={k_idx}"

        dut.mem1_read_en.value = 0
        await tick(dut, 1)

        assert int(dut.wght_loaded_dbg.value) == 1, f"weight ingress not marked loaded for k={k_idx}"
        for lane_idx, expected_lane in enumerate(wght_vector):
            observed_lane = signed_value(getattr(dut, f"wght_buf{lane_idx}"))
            assert observed_lane == expected_lane, f"wght_buf{lane_idx} mismatch for k={k_idx}"

        dut.mem_addr.value = k_idx
        dut.westbound_sel.value = WB_MEM0
        dut.westbound_consumer_sel.value = WC_MXM
        dut.mxm_ingress_mode.value = INGRESS_INPUT
        dut.mem_read_en.value = 1
        await tick(dut, 1)

        assert int(dut.westbound_valid_dbg.value) == 1, f"MEM0 westbound valid missing for k={k_idx}"
        assert int(dut.westbound_payload_dbg.value) == pack_bytes(input_vector), f"MEM0 payload mismatch for k={k_idx}"

        dut.mem_read_en.value = 0
        await tick(dut, 1)

        assert int(dut.input_loaded_dbg.value) == 1, f"input ingress not marked loaded for k={k_idx}"
        for lane_idx, expected_lane in enumerate(input_vector):
            observed_lane = signed_value(getattr(dut, f"input_buf{lane_idx}"))
            assert observed_lane == expected_lane, f"input_buf{lane_idx} mismatch for k={k_idx}"

        dut.mxm_start.value = 1
        await tick(dut, 2)
        dut.mxm_start.value = 0

    await tick(dut, 1)
    observed = read_mxm_matrix(dut)
    assert observed == expected, f"4x4 matmul mismatch: got={observed} expected={expected}"
