import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import math

# Producer/Consumer codes matching lpu_pkg.sv definitions
WB_NONE = 0; WB_SXM = 1; WB_MEM0 = 2; WB_VXM = 3; WB_MEM1 = 4
EB_NONE = 0; EB_MXM = 1; EB_SXM = 2; EB_MEM0 = 3; EB_VXM = 4
WC_NONE = 0; WC_MXM = 1; WC_SXM = 2; WC_MEM0 = 3; WC_VXM = 4
EC_NONE = 0; EC_SXM = 1; EC_MEM0 = 2; EC_VXM = 3; EC_MEM1 = 4
INGRESS_NONE = 0; INGRESS_INPUT = 1; INGRESS_WGHT = 2

# Helper function to pack 8 bytes into a 64-bit word
def pack_bytes(values):
    word = 0
    for idx, value in enumerate(values):
        word |= (value & 0xFF) << (8 * idx)
    return word

def _set_field(word, value, lsb, width):
    mask = (1 << width) - 1
    return word | ((value & mask) << lsb)


MEM_ADDR_W = 15
OP_TRANSPOSE_LOAD = 0x5A5
OP_TRANSPOSE_EMIT = 0xA5A


def _set_mem_addr(word, value, lsb):
    if value < 0 or value >= (1 << MEM_ADDR_W):
        raise ValueError(f"memory address {value} does not fit in {MEM_ADDR_W} bits")
    return _set_field(word, value, lsb, MEM_ADDR_W)

# Encodes fields into the 96-bit instruction format expected by the ICU
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
    sxm_transpose_load=0,
    sxm_transpose_emit=0,
    sxm_load_from_west=0,
    vxm_ctrl=0,
    vxm_data_sel=0,
    mxm_ingress_mode=INGRESS_NONE,
    mxm_start=0,
    mxm_clear=0,
    mxm_e_row_sel=0,
    mxm_e_col_sel=0,
    mxm_e_valid_in=0,
    mxm_input_is_signed=1,
    mxm_wght_is_signed=1,
    mxm_use_fp=0,
    fp_quant_mode=0,
    mem_store_fmt=0,
    vxm_layernorm_en=0,
    vxm_operand_sel=0,
    vxm_softmax_chunked_en=0,
):
    word = 0
    if sxm_opcode_input == OP_TRANSPOSE_LOAD:
        sxm_transpose_load = 1
    elif sxm_opcode_input == OP_TRANSPOSE_EMIT:
        sxm_transpose_emit = 1

    word = _set_field(word, westbound_sel, 0, 3)
    word = _set_field(word, eastbound_sel, 3, 3)
    word = _set_field(word, westbound_consumer_sel, 6, 3)
    word = _set_field(word, eastbound_consumer_sel, 9, 3)
    word = _set_field(word, mem0_read_en, 12, 1)
    word = _set_field(word, mem0_write_en, 13, 1)
    word = _set_mem_addr(word, mem0_addr, 14)
    word = _set_field(word, mem1_read_en, 29, 1)
    word = _set_field(word, mem1_write_en, 30, 1)
    word = _set_mem_addr(word, mem1_addr, 31)
    word = _set_field(word, mxm_ingress_mode, 46, 2)
    word = _set_field(word, mxm_start, 48, 1)
    word = _set_field(word, mxm_clear, 49, 1)
    word = _set_field(word, mxm_e_row_sel, 50, 3)
    word = _set_field(word, mxm_e_col_sel, 53, 3)
    word = _set_field(word, mxm_e_valid_in, 56, 1)
    word = _set_field(word, mxm_input_is_signed, 57, 1)
    word = _set_field(word, mxm_wght_is_signed, 58, 1)
    word = _set_field(word, mxm_use_fp, 59, 1)
    word = _set_field(word, fp_quant_mode, 60, 1)
    word = _set_field(word, mem_store_fmt, 61, 2)
    word = _set_field(word, sxm_transpose_load, 63, 1)
    word = _set_field(word, sxm_transpose_emit, 64, 1)
    word = _set_field(word, sxm_load_from_west, 65, 1)
    word = _set_field(word, vxm_ctrl, 66, 4)
    word = _set_field(word, vxm_data_sel, 70, 1)
    word = _set_field(word, vxm_operand_sel, 71, 3)
    word = _set_field(word, vxm_layernorm_en, 74, 1)
    word = _set_field(word, vxm_softmax_chunked_en, 79, 1)
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

def preload_instruction(dut, pc, word):
    dut.u_lpu.u_icu.imem_array[pc].value = word

def preload_mem0_row(dut, addr, values):
    dut.u_lpu.u_mem0.sram_array[addr].value = pack_bytes(values)

def preload_mem1_row(dut, addr, values):
    dut.u_lpu.u_mem1.sram_array[addr].value = pack_bytes(values)


@cocotb.test()
async def test_decode_mat_selection(dut):
    # Initialize clock
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # Enable LayerNorm block in the vxm module
    dut.u_lpu.u_vxm.layernorm_bypass.value = 0

    # 1. KV cache Injection: 8 key vectors of dimension 8
    # Keys K_0..K_7 stored in MEM0[1..8] column-major
    K = [
        [12, -8, 5, 2, 1, 3, -4, 2],       # K_0
        [-15, 22, -10, 8, 4, 2, 1, 0],      # K_1
        [8, -14, 18, -9, 0, 1, 2, 3],       # K_2
        [50, -42, 38, -12, 10, 20, 5, 2],   # K_3 (Produces high positive dot product with q)
        [2, 4, 6, 8, 10, 12, 14, 16],       # K_4
        [-1, -2, -3, -4, -5, -6, -7, -8],   # K_5
        [0, 1, 0, 1, 0, 1, 0, 1],           # K_6
        [5, 5, 5, 5, 5, 5, 5, 5],           # K_7
    ]
    for k in range(8):
        preload_mem0_row(dut, addr=1 + k, values=[K[col][k] for col in range(8)])
    preload_mem0_row(dut, addr=9, values=[0]*8)

    # Values V_0..V_7 in MEM0[11..18], row-major
    V = [
        [8, -12, 14, -6, 2, 4, -1, 3],       # V_0
        [-15, 20, -8, 12, 0, 1, -2, 2],      # V_1
        [25, -18, 30, -15, 1, 2, 3, 4],      # V_2
        [42, -22, 50, -18, 3, 6, -9, 12],    # V_3
        [10, 10, 10, 10, 10, 10, 10, 10],    # V_4
        [1, 2, 3, 4, 5, 6, 7, 8],            # V_5
        [-2, -4, -3, -1, -5, 0, 1, 2],       # V_6
        [5, -5, 5, -5, 5, -5, 5, -5],        # V_7
    ]
    for k in range(8):
        preload_mem0_row(dut, addr=11 + k, values=V[k])
    preload_mem0_row(dut, addr=19, values=[0]*8)

    # 2. Query vector q = [2, -1, 3, 0, 1, -2, 4, 0] stored column-major in MEM1[0..7]
    q_vals = [2, -1, 3, 0, 1, -2, 4, 0]
    for k in range(8):
        preload_mem1_row(dut, addr=k, values=[q_vals[k]] + [0]*7)

    # LM Head weights in MEM1[30..37] (8 tokens: "the", "mat", "sofa", "rug", "cat", "sat", "on", "floor")
    # W is stored column-major. We set column 1 ("mat") to have a strongly positive correlation
    # with the expected LayerNorm output.
    W = [
        [5, 30, 2, -1, 0, 1, 2, 3],   # row 0
        [-2, -30, -4, 3, 0, -1, 1, 0], # row 1
        [8, 30, 4, -2, 0, 2, -1, 1],  # row 2
        [-1, 30, -3, 4, 0, 1, 1, -1], # row 3
        [1, -30, 1, 0, 0, -2, 1, 1],  # row 4
        [3, 30, 4, -1, 0, 1, 2, -1],  # row 5
        [-2, -30, 0, 2, 0, 0, 1, 0],  # row 6
        [0, 30, -1, 3, 0, 2, 1, -2]   # row 7
    ]
    for k in range(8):
        preload_mem1_row(dut, addr=30 + k, values=[W[row][k] for row in range(8)])

    # 3. Compile the Decode Phase Program:
    program = []

    # --- Phase 1: Attention Logits Calculation (q K^T) ---
    program.append(build_instruction(mxm_clear=1))

    for k in range(8):
        # Load Key column k from MEM0
        program.append(build_instruction(mem0_read_en=1, mem0_addr=1 + k))
        program.append(build_instruction(westbound_sel=WB_MEM0, westbound_consumer_sel=WC_MXM, mxm_ingress_mode=INGRESS_WGHT))
        # Load Query element k from MEM1
        program.append(build_instruction(mem1_read_en=1, mem1_addr=k))
        program.append(build_instruction(westbound_sel=WB_MEM1, westbound_consumer_sel=WC_MXM, mxm_ingress_mode=INGRESS_INPUT))
        # Accumulate
        program.append(build_instruction(mxm_start=1))
        program.append(build_instruction(mxm_start=1))

    # NOPs to settle MXM output
    program.extend([build_instruction(), build_instruction()])

    # Stream scores from MXM through VXM Softmax to MEM0[20] (Row 0 only)
    program.append(build_instruction(eastbound_sel=EB_MXM, eastbound_consumer_sel=EC_VXM, mxm_e_row_sel=0, mxm_e_valid_in=1, vxm_ctrl=0b1100, vxm_data_sel=1))
    program.append(build_instruction(vxm_ctrl=0b1100, vxm_data_sel=1))
    for _ in range(10):
        program.append(build_instruction())
    for _ in range(8):
        program.append(build_instruction(eastbound_sel=EB_VXM, eastbound_consumer_sel=EC_MEM0, mem0_write_en=1, mem0_addr=20))

    # --- Phase 2: Context vector calculation (S V) & LayerNorm ---
    program.append(build_instruction(mxm_clear=1))

    for k in range(8):
        # Load Value row k from MEM0
        program.append(build_instruction(mem0_read_en=1, mem0_addr=11 + k))
        program.append(build_instruction(westbound_sel=WB_MEM0, westbound_consumer_sel=WC_MXM, mxm_ingress_mode=INGRESS_WGHT))
        # Load Softmax weights S from MEM0[20]
        program.append(build_instruction(mem0_read_en=1, mem0_addr=20))
        program.append(build_instruction(westbound_sel=WB_MEM0, westbound_consumer_sel=WC_MXM, mxm_ingress_mode=INGRESS_INPUT))
        # Accumulate
        program.append(build_instruction(mxm_start=1))
        program.append(build_instruction(mxm_start=1))

    # NOPs to settle MXM output
    program.extend([build_instruction(), build_instruction()])

    # Stream context vector from MXM through VXM LayerNorm to MEM1[10] (Row 0 only)
    program.append(build_instruction(eastbound_sel=EB_MXM, eastbound_consumer_sel=EC_VXM, mxm_e_row_sel=0, mxm_e_valid_in=1, vxm_ctrl=0b0000, vxm_data_sel=1))
    program.append(build_instruction(vxm_ctrl=0b0000, vxm_data_sel=1))
    for _ in range(10):
        program.append(build_instruction())
    for _ in range(8):
        program.append(build_instruction(eastbound_sel=EB_VXM, eastbound_consumer_sel=EC_MEM1, mem1_write_en=1, mem1_addr=10))

    # --- Phase 3: LM Head Projection (x W) ---
    program.append(build_instruction(mxm_clear=1))

    for k in range(8):
        # Load weight column k from MEM1 (W)
        program.append(build_instruction(mem1_read_en=1, mem1_addr=30 + k))
        program.append(build_instruction(westbound_sel=WB_MEM1, westbound_consumer_sel=WC_MXM, mxm_ingress_mode=INGRESS_WGHT))
        # Load normalized input element k from MEM1 (x)
        program.append(build_instruction(mem1_read_en=1, mem1_addr=10))
        program.append(build_instruction(westbound_sel=WB_MEM1, westbound_consumer_sel=WC_MXM, mxm_ingress_mode=INGRESS_INPUT))
        # Accumulate
        program.append(build_instruction(mxm_start=1))
        program.append(build_instruction(mxm_start=1))

    # NOPs to settle
    program.extend([build_instruction(), build_instruction()])

    # Write output logits to MEM1[41] (Row 0 only)
    for _ in range(8):
        program.append(build_instruction(eastbound_sel=EB_MXM, eastbound_consumer_sel=EC_MEM1, mem1_write_en=1, mem1_addr=41, mxm_e_row_sel=0, mxm_e_valid_in=1))

    # 4. Preload instruction program into the ICU
    for pc, inst in enumerate(program):
        preload_instruction(dut, pc, inst)
    for pc in range(len(program), 320):
        preload_instruction(dut, pc, build_instruction())

    # 5. Execute Program
    await reset_dut(dut)
    total_cycles = len(program) + 15
    await tick(dut, total_cycles)

    # 6. Read and log all intermediate hardware states
    scores_word = int(dut.u_lpu.u_mem0.sram_array[20].value)
    scores = [(scores_word >> (8 * i)) & 0xFF for i in range(8)]

    layernorm_out_word = int(dut.u_lpu.u_mem1.sram_array[10].value)
    layernorm_out = []
    for lane in range(8):
        val = (layernorm_out_word >> (8 * lane)) & 0xFF
        if val & 0x80:
            val -= 256
        layernorm_out.append(val)

    # Read final logits from MEM1[41]
    word = int(dut.u_lpu.u_mem1.sram_array[41].value)
    logit_mask = 0xFFFFFFFF
    logits = []
    for lane in range(4):
        val = (word >> (32 * lane)) & logit_mask
        if val & 0x80000000:
            val -= 0x100000000
        logits.append(val)

    dictionary = ["the", "mat", "sofa", "rug", "cat", "sat", "on", "floor"]
    max_idx = logits.index(max(logits))
    predicted_token = dictionary[max_idx]

    dut._log.info(f"\n=======================================================")
    dut._log.info(f"--- INTERMEDIATE DECODE PHASE PROCESS LOGS ---")
    dut._log.info(f"2. Phase 1: Attention & Softmax:")
    dut._log.info(f"   - Softmax Weights (MEM0[20]):  {scores} (Unsigned Q8)")
    dut._log.info(f"")
    dut._log.info(f"3. Phase 2: Context Mixing & LayerNorm:")
    dut._log.info(f"   - Normalized Output (MEM1[10]): {layernorm_out} (Signed 8-bit)")
    dut._log.info(f"")
    dut._log.info(f"4. Phase 3: LM Head Projection & Output Selection:")
    dut._log.info(f"   - Final Logits (MEM1[41]):      {logits} (Corresponding to: {dictionary[:4]})")
    dut._log.info(f"   - Winner Token Selection:       '{predicted_token}' (Logit: {logits[max_idx]})")
    dut._log.info(f"")
    dut._log.info(f"Simulation Status: SUCCESS (Cycles: {total_cycles})")
    dut._log.info(f"=======================================================\n")

    assert predicted_token == "mat", f"Expected 'mat' to be selected, but got '{predicted_token}'"
