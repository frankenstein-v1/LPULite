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

#read.  signed vector
def signed_value(handle):
    return int(handle.value.to_signed())


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
    vxm_math_op=0,
    vxm_accum_en=0,
    vxm_flush=0,
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

    # vxm control [61:58]
    word = _set_field(word, vxm_math_op, 58, 2)
    word = _set_field(word, vxm_accum_en, 60, 1)
    word = _set_field(word, vxm_flush, 61, 1)

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




   

    
