import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


WB_NONE = 0
WB_SXM = 1
WB_MEM0 = 2
WB_VXM = 3
WB_MEM1 = 4

EB_NONE = 0
EB_MXM = 1
EB_SXM = 2
EB_MEM0 = 3
EB_VXM = 4

WC_NONE = 0
WC_MXM = 1
WC_SXM = 2
WC_MEM0 = 3
WC_VXM = 4

EC_NONE = 0
EC_SXM = 1
EC_MEM0 = 2
EC_VXM = 3
EC_MEM1 = 4

INGRESS_NONE = 0
INGRESS_INPUT = 1
INGRESS_WGHT = 2


#pack 4 bytes into a word 
def pack_bytes(values):
    word = 0
    for idx, value in enumerate(values):
        word |= (value & 0xFF) << (8 * idx)
    return word


def pack_mxm_row(values):
    row = 0
    for idx, value in enumerate(values):
        row |= (value & 0xFFFFFFFF) << (32 * idx)
    return row

#read.  signed vector
def signed_value(handle):
    return int(handle.value.to_signed())


def clip_unsigned_q8(value):
    if value <= 0:
        return 0
    if value >= 255:
        return 255
    return value & 0xFF


def exp_expected(q_value):
    ln2 = 177
    coeff_a = 92
    coeff_b = 346
    coeff_c = 88

    z_value = (-q_value) // ln2
    p_value = q_value + (z_value * ln2)
    t_value = p_value + coeff_b
    t_squared = t_value * t_value
    q_poly = ((coeff_a * t_squared) >> 16) + coeff_c
    return q_poly >> z_value


def softmax_expected(lanes):
    lane_max = max(lanes)
    lane_sub = [lane - lane_max for lane in lanes]
    lane_exp = [exp_expected(lane) for lane in lane_sub]
    sum_exp = sum(lane_exp)
    quotient = (1 << 30) // sum_exp
    shift = 30 - 8
    return [(quotient * lane) >> shift for lane in lane_exp]


def scale_softmax_quant_expected(lanes):
    scaled_lanes = [lane >> 1 for lane in lanes]
    softmax_lanes = softmax_expected(scaled_lanes)
    return [clip_unsigned_q8(lane) for lane in softmax_lanes]


def _set_field(word, value, lsb, width):
    mask = (1 << width) - 1
    return word | ((value & mask) << lsb)

#builds the instruction for the ICU
def build_instruction(
    *,
    westbound_sel=WB_NONE,
    eastbound_sel=EB_NONE,
    westbound_consumer_sel=WC_NONE,
    eastbound_consumer_sel=EC_NONE,
    mem0_read_en=0,
    mem0_write_en=0,
    mem0_addr=0,
    mem1_read_en=0,
    mem1_write_en=0,
    mem1_addr=0,
    sxm_opcode_input=0,
    sxm_opcode_weight=0,
    vxm_ctrl=0,
    vxm_data_sel=0,
    mxm_ingress_mode=INGRESS_NONE,
    mxm_start=0,
    mxm_clear=0,
    mxm_e_row_sel=0,
    mxm_e_col_sel=0,
    mxm_e_valid_in=0,
):
    
    word = 0

    # bus control [11:0]
    word = _set_field(word, westbound_sel, 0, 3)
    word = _set_field(word, eastbound_sel, 3, 3)
    word = _set_field(word, westbound_consumer_sel, 6, 3)
    word = _set_field(word, eastbound_consumer_sel, 9, 3)

    # mem0 control [22:12]
    word = _set_field(word, mem0_read_en, 12, 1)
    word = _set_field(word, mem0_write_en, 13, 1)
    word = _set_field(word, mem0_addr, 14, 9)

    # mem1 control [33:23]
    word = _set_field(word, mem1_read_en, 23, 1)
    word = _set_field(word, mem1_write_en, 24, 1)
    word = _set_field(word, mem1_addr, 25, 9)

    # sxm control [57:34]
    word = _set_field(word, sxm_opcode_input, 34, 12)
    word = _set_field(word, sxm_opcode_weight, 46, 12)

    # vxm control [76:71]
    word = _set_field(word, vxm_ctrl, 71, 4)
    word = _set_field(word, vxm_data_sel, 76, 1)

    # mxm control [70:62]
    word = _set_field(word, mxm_ingress_mode, 62, 2)
    word = _set_field(word, mxm_start, 64, 1)
    word = _set_field(word, mxm_clear, 65, 1)
    word = _set_field(word, mxm_e_row_sel, 66, 2)
    word = _set_field(word, mxm_e_col_sel, 68, 2)
    word = _set_field(word, mxm_e_valid_in, 70, 1)

    return word


async def tick(dut, n=1):
    for _ in range(n):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")


async def reset_dut(dut):
    dut.rst_n.value = 0
    await tick(dut, 2)
    dut.rst_n.value = 1
    await Timer(1, unit="ns")


def preload_instruction(dut, pc, instruction_word):
    """Write one 96-bit word directly into the ICU instruction memory."""
    dut.u_lpu.u_icu.imem_array[pc].value = instruction_word


def preload_mem0_word(dut, addr, values):
    """Write one packed 32-bit word directly into MEM0 SRAM."""
    dut.u_lpu.u_mem0.sram_array[addr].value = pack_bytes(values)


def preload_mem1_word(dut, addr, values):
    """Write one packed 32-bit word directly into MEM1 SRAM."""
    dut.u_lpu.u_mem1.sram_array[addr].value = pack_bytes(values)


def preload_program(dut, instructions, *, trailing_nops=64):
    for pc, instruction_word in enumerate(instructions):
        preload_instruction(dut, pc, instruction_word)
    for pc in range(len(instructions), len(instructions) + trailing_nops):
        preload_instruction(dut, pc, build_instruction())


def matmul_expected(a_matrix, b_matrix):
    expected = [[0 for _ in range(4)] for _ in range(4)]
    for row in range(4):
        for col in range(4):
            for k_idx in range(4):
                expected[row][col] += a_matrix[row][k_idx] * b_matrix[k_idx][col]
    return expected


def read_mxm_matrix(dut):
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


def append_mxm_k_slice(program, *, k_idx):
    program.extend(
        [
            build_instruction(
                mem1_read_en=1,
                mem1_addr=k_idx,
                westbound_sel=WB_MEM1,
                westbound_consumer_sel=WC_MXM,
                mxm_ingress_mode=INGRESS_WGHT,
            ),
            build_instruction(
                westbound_sel=WB_MEM1,
                westbound_consumer_sel=WC_MXM,
                mxm_ingress_mode=INGRESS_WGHT,
            ),
            build_instruction(
                mem0_read_en=1,
                mem0_addr=k_idx,
                westbound_sel=WB_MEM0,
                westbound_consumer_sel=WC_MXM,
                mxm_ingress_mode=INGRESS_INPUT,
            ),
            build_instruction(
                westbound_sel=WB_MEM0,
                westbound_consumer_sel=WC_MXM,
                mxm_ingress_mode=INGRESS_INPUT,
            ),
            build_instruction(mxm_start=1),
            build_instruction(mxm_start=1),
        ]
    )


def append_vxm_row_store(program, *, row_idx, target_addr, wait_cycles=10, store_cycles=4):
    program.append(
        build_instruction(
            eastbound_sel=EB_MXM,
            eastbound_consumer_sel=EC_VXM,
            mxm_e_row_sel=row_idx,
            mxm_e_valid_in=1,
            vxm_ctrl=0b1100,
            vxm_data_sel=1,
        )
    )
    program.append(
        build_instruction(
            vxm_ctrl=0b1100,
            vxm_data_sel=1,
        )
    )
    for _ in range(wait_cycles):
        program.append(build_instruction())
    for _ in range(store_cycles):
        program.append(
            build_instruction(
                eastbound_sel=EB_VXM,
                eastbound_consumer_sel=EC_MEM0,
                mem0_write_en=1,
                mem0_addr=target_addr,
            )
        )


@cocotb.test()
async def test_lpu_wrapper_smoke(dut):
    """Minimal smoke test: clock, reset, and wrapper visibility."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    await reset_dut(dut)

    assert int(dut.pc_dbg.value) >= 0


@cocotb.test()
async def test_lpu_minimal_program_skeleton(dut):
    
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    weights = [3, 0, 0, 0]
    inputs = [2, 0, 0, 0]

    preload_mem1_word(dut, addr=0, values=weights)
    preload_mem0_word(dut, addr=0, values=inputs)

    preload_instruction(
        dut,
        0,
        build_instruction(
            mem1_read_en=1,
            mem1_addr=0,
            westbound_sel=WB_MEM1,
            westbound_consumer_sel=WC_MXM,
            mxm_ingress_mode=INGRESS_WGHT,
        ),
    )

    preload_instruction(
        dut,
        1,
        build_instruction(
            westbound_sel=WB_MEM1,
            westbound_consumer_sel=WC_MXM,
            mxm_ingress_mode=INGRESS_WGHT
        ),
    )

    preload_instruction(
        dut,
        2,
        build_instruction(
            mem0_read_en=1,
            mem0_addr=0, 
            westbound_sel=WB_MEM0,
            westbound_consumer_sel=WC_MXM,
            mxm_ingress_mode= INGRESS_INPUT
        ),
    )

    preload_instruction(
        dut, 
        3,
        build_instruction(
            westbound_sel=WB_MEM0,
            westbound_consumer_sel=WC_MXM, 
            mxm_ingress_mode=INGRESS_INPUT
        ),
    )

    preload_instruction(
        dut, 4, 
        build_instruction(
            mxm_start=1
        ),
    )

    preload_instruction(
        dut, 5,
        build_instruction(
            mxm_start=1
        ),
    )

    preload_instruction(dut, 6, build_instruction())

    preload_instruction(dut, 7, build_instruction())



    await reset_dut(dut)

    # asertions about instruction 0 
    assert int(dut.pc_dbg.value) == 0
    assert int(dut.westbound_sel_dbg.value) == WB_MEM1
    assert int(dut.westbound_consumer_sel_dbg.value) == WC_MXM
    assert int(dut.mxm_ingress_mode_dbg.value) == INGRESS_WGHT

    await tick(dut, 2)

    assert int(dut.wght_loaded_dbg.value) == 1
    assert signed_value(dut.wght_buf0) == 3
    assert signed_value(dut.wght_buf1) == 0
    assert signed_value(dut.wght_buf2) == 0
    assert signed_value(dut.wght_buf3) == 0

    await tick(dut, 1)

    # mem0 input-load instruction phase
    assert int(dut.westbound_sel_dbg.value) == WB_MEM0
    assert int(dut.westbound_consumer_sel_dbg.value) == WC_MXM
    assert int(dut.mxm_ingress_mode_dbg.value) == INGRESS_INPUT

    await tick(dut, 1)

    #check if the mem0 values r correct
    assert int(dut.input_loaded_dbg.value) == 1
    assert signed_value(dut.input_buf0) == 2
    assert signed_value(dut.input_buf1) == 0
    assert signed_value(dut.input_buf2) == 0
    assert signed_value(dut.input_buf3) == 0

    await tick(dut, 1)

    await tick(dut, 2)

    assert signed_value(dut.mxm_out_00_dbg) == 6


@cocotb.test()
async def test_lpu_mxm_to_vxm_softmax_quant_to_mem0(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    weights = [3, 0, 0, 0]
    inputs = [2, 0, 0, 0]
    expected_mxm_row = [6, 0, 0, 0]
    expected_mem0_word = pack_bytes(scale_softmax_quant_expected(expected_mxm_row))
    target_addr = 1

    preload_mem1_word(dut, addr=0, values=weights)
    preload_mem0_word(dut, addr=0, values=inputs)

    program = [
        build_instruction(
            mem1_read_en=1,
            mem1_addr=0,
            westbound_sel=WB_MEM1,
            westbound_consumer_sel=WC_MXM,
            mxm_ingress_mode=INGRESS_WGHT,
        ),
        build_instruction(
            westbound_sel=WB_MEM1,
            westbound_consumer_sel=WC_MXM,
            mxm_ingress_mode=INGRESS_WGHT,
        ),
        build_instruction(
            mem0_read_en=1,
            mem0_addr=0,
            westbound_sel=WB_MEM0,
            westbound_consumer_sel=WC_MXM,
            mxm_ingress_mode=INGRESS_INPUT,
        ),
        build_instruction(
            westbound_sel=WB_MEM0,
            westbound_consumer_sel=WC_MXM,
            mxm_ingress_mode=INGRESS_INPUT,
        ),
        build_instruction(mxm_start=1),
        build_instruction(mxm_start=1),
        build_instruction(),
        build_instruction(),
        build_instruction(),
        build_instruction(),
        build_instruction(
            eastbound_sel=EB_MXM,
            eastbound_consumer_sel=EC_VXM,
            mxm_e_row_sel=0,
            mxm_e_valid_in=1,
            vxm_ctrl=0b1100,
            vxm_data_sel=1,
        ),
        build_instruction(
            vxm_ctrl=0b1100,
            vxm_data_sel=1,
        ),
        build_instruction(),
        build_instruction(),
        build_instruction(),
        build_instruction(),
        build_instruction(),
        build_instruction(),
        build_instruction(),
        build_instruction(),
        build_instruction(),
        build_instruction(),
        build_instruction(),
        build_instruction(),
        build_instruction(),
        build_instruction(),
        build_instruction(),
        build_instruction(
            eastbound_sel=EB_VXM,
            eastbound_consumer_sel=EC_MEM0,
            mem0_write_en=1,
            mem0_addr=target_addr,
        ),
        build_instruction(
            eastbound_sel=EB_VXM,
            eastbound_consumer_sel=EC_MEM0,
            mem0_write_en=1,
            mem0_addr=target_addr,
        ),
        build_instruction(
            eastbound_sel=EB_VXM,
            eastbound_consumer_sel=EC_MEM0,
            mem0_write_en=1,
            mem0_addr=target_addr,
        ),
        build_instruction(
            eastbound_sel=EB_VXM,
            eastbound_consumer_sel=EC_MEM0,
            mem0_write_en=1,
            mem0_addr=target_addr,
        ),
        build_instruction(
            eastbound_sel=EB_VXM,
            eastbound_consumer_sel=EC_MEM0,
            mem0_write_en=1,
            mem0_addr=target_addr,
        ),
        build_instruction(
            eastbound_sel=EB_VXM,
            eastbound_consumer_sel=EC_MEM0,
            mem0_write_en=1,
            mem0_addr=target_addr,
        ),
        build_instruction(
            eastbound_sel=EB_VXM,
            eastbound_consumer_sel=EC_MEM0,
            mem0_write_en=1,
            mem0_addr=target_addr,
        ),
        build_instruction(
            eastbound_sel=EB_VXM,
            eastbound_consumer_sel=EC_MEM0,
            mem0_write_en=1,
            mem0_addr=target_addr,
        ),
    ]
    preload_program(dut, program)

    await reset_dut(dut)
    await tick(dut, 40)

    observed_word = int(dut.u_lpu.u_mem0.sram_array[target_addr].value)
    assert observed_word == expected_mem0_word, (
        f"MEM0[{target_addr}] mismatch: got 0x{observed_word:08x}, "
        f"expected 0x{expected_mem0_word:08x}"
    )
    assert int(dut.vxm_input_overflow_dbg.value) == 0, "VXM input FIFO overflowed unexpectedly"


@cocotb.test()
async def test_lpu_mxm_row_store_to_mem0_full_width(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    weights = [3, 0, 0, 0]
    inputs = [2, 0, 0, 0]
    target_addr = 8
    expected_row = [6, 0, 0, 0]
    expected_row_word = pack_mxm_row(expected_row)

    preload_mem1_word(dut, addr=0, values=weights)
    preload_mem0_word(dut, addr=0, values=inputs)

    program = [
        build_instruction(
            mem1_read_en=1,
            mem1_addr=0,
            westbound_sel=WB_MEM1,
            westbound_consumer_sel=WC_MXM,
            mxm_ingress_mode=INGRESS_WGHT,
        ),
        build_instruction(
            westbound_sel=WB_MEM1,
            westbound_consumer_sel=WC_MXM,
            mxm_ingress_mode=INGRESS_WGHT,
        ),
        build_instruction(
            mem0_read_en=1,
            mem0_addr=0,
            westbound_sel=WB_MEM0,
            westbound_consumer_sel=WC_MXM,
            mxm_ingress_mode=INGRESS_INPUT,
        ),
        build_instruction(
            westbound_sel=WB_MEM0,
            westbound_consumer_sel=WC_MXM,
            mxm_ingress_mode=INGRESS_INPUT,
        ),
        build_instruction(mxm_start=1),
        build_instruction(mxm_start=1),
        build_instruction(),
        build_instruction(),
        build_instruction(
            eastbound_sel=EB_MXM,
            eastbound_consumer_sel=EC_MEM0,
            mem0_write_en=1,
            mem0_addr=target_addr,
            mxm_e_row_sel=0,
            mxm_e_valid_in=1,
        ),
        build_instruction(),
    ]
    preload_program(dut, program)

    await reset_dut(dut)
    await tick(dut, len(program) + 4)

    observed_row_word = int(dut.u_lpu.u_mem0.sram_array[target_addr].value)
    assert observed_row_word == expected_row_word, (
        f"MEM0[{target_addr}] full-row mismatch: got 0x{observed_row_word:032x}, "
        f"expected 0x{expected_row_word:032x}"
    )


@cocotb.test()
async def test_lpu_full_mxm_to_vxm_softmax_quant_rows_to_mem0(dut):
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
    expected_mxm = matmul_expected(a_matrix, b_matrix)
    mem0_store_base = 16

    for k_idx in range(4):
        preload_mem0_word(
            dut,
            addr=k_idx,
            values=[a_matrix[row][k_idx] for row in range(4)],
        )
        preload_mem1_word(
            dut,
            addr=k_idx,
            values=[b_matrix[k_idx][col] for col in range(4)],
        )

    program = []
    for k_idx in range(4):
        append_mxm_k_slice(program, k_idx=k_idx)

    program.extend([build_instruction(), build_instruction()])

    for row_idx in range(4):
        append_vxm_row_store(
            program,
            row_idx=row_idx,
            target_addr=mem0_store_base + row_idx,
        )

    preload_program(dut, program)

    await reset_dut(dut)
    await tick(dut, len(program) + 8)

    observed_mxm = read_mxm_matrix(dut)
    assert observed_mxm == expected_mxm, (
        f"MXM 4x4 mismatch: got={observed_mxm} expected={expected_mxm}"
    )

    for row_idx, expected_row in enumerate(expected_mxm):
        observed_word = int(dut.u_lpu.u_mem0.sram_array[mem0_store_base + row_idx].value)
        expected_word = pack_bytes(scale_softmax_quant_expected(expected_row))
        assert observed_word == expected_word, (
            f"MEM0[{mem0_store_base + row_idx}] mismatch: got 0x{observed_word:08x}, "
            f"expected 0x{expected_word:08x} for row {row_idx}"
        )

    assert int(dut.vxm_input_overflow_dbg.value) == 0, "VXM input FIFO overflowed unexpectedly"


@cocotb.test()
async def test_lpu_sxm_transpose_to_mem0(dut):
    """Verify end-to-end matrix transposition using the SXM module and buses."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    input_matrix = [
        [0x01, 0x02, 0x03, 0x04], # Row 0
        [0x05, 0x06, 0x07, 0x08], # Row 1
        [0x09, 0x0A, 0x0B, 0x0C], # Row 2
        [0x0D, 0x0E, 0x0F, 0x10], # Row 3
    ]
    
    expected_transposed = [
        [0x01, 0x05, 0x09, 0x0D], # Col 0
        [0x02, 0x06, 0x0A, 0x0E], # Col 1
        [0x03, 0x07, 0x0B, 0x0F], # Col 2
        [0x04, 0x08, 0x0C, 0x10], # Col 3
    ]
    
    # Preload inputs into MEM0 at addresses 0..3
    for r in range(4):
        preload_mem0_word(dut, addr=r, values=input_matrix[r])
        
    # Program sequence:
    # PC 0: read MEM0[0]
    # PC 1: read MEM0[1], route MEM0[0] to SXM
    # PC 2: read MEM0[2], route MEM0[1] to SXM, trigger SXM Transpose LOAD
    # PC 3: read MEM0[3], route MEM0[2] to SXM
    # PC 4: route MEM0[3] to SXM
    # PC 5: Idle
    # PC 6: Route SXM to MEM0[10], trigger SXM Transpose EMIT
    # PC 7: Route SXM to MEM0[11]
    # PC 8: Route SXM to MEM0[12]
    # PC 9: Route SXM to MEM0[13]
    program = [
        # PC 0
        build_instruction(mem0_read_en=1, mem0_addr=0),
        # PC 1
        build_instruction(mem0_read_en=1, mem0_addr=1, eastbound_sel=EB_MEM0, eastbound_consumer_sel=EC_SXM),
        # PC 2
        build_instruction(mem0_read_en=1, mem0_addr=2, eastbound_sel=EB_MEM0, eastbound_consumer_sel=EC_SXM, sxm_opcode_input=0x5A5),
        # PC 3
        build_instruction(mem0_read_en=1, mem0_addr=3, eastbound_sel=EB_MEM0, eastbound_consumer_sel=EC_SXM),
        # PC 4
        build_instruction(eastbound_sel=EB_MEM0, eastbound_consumer_sel=EC_SXM),
        # PC 5
        build_instruction(),
        # PC 6
        build_instruction(eastbound_sel=EB_SXM, eastbound_consumer_sel=EC_MEM0, mem0_write_en=1, mem0_addr=10, sxm_opcode_input=0xA5A),
        # PC 7
        build_instruction(eastbound_sel=EB_SXM, eastbound_consumer_sel=EC_MEM0, mem0_write_en=1, mem0_addr=11),
        # PC 8
        build_instruction(eastbound_sel=EB_SXM, eastbound_consumer_sel=EC_MEM0, mem0_write_en=1, mem0_addr=12),
        # PC 9
        build_instruction(eastbound_sel=EB_SXM, eastbound_consumer_sel=EC_MEM0, mem0_write_en=1, mem0_addr=13),
    ]
    
    preload_program(dut, program)

    await reset_dut(dut)
    await tick(dut, len(program) + 4)

    # Verify that transposed rows are stored in MEM0[10..13]
    for r in range(4):
        observed_word = int(dut.u_lpu.u_mem0.sram_array[10 + r].value)
        expected_word = pack_bytes(expected_transposed[r])
        assert observed_word == expected_word, (
            f"MEM0[10+{r}] transpose mismatch: got 0x{observed_word:08x}, "
            f"expected 0x{expected_word:08x}"
        )
        
    dut._log.info("End-to-end SXM Transpose integration test passed successfully!")





   

    
