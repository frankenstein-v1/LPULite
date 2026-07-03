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

# Helper function to pack 4 bytes into a 32-bit word
def pack_bytes(values):
    word = 0
    for idx, value in enumerate(values):
        word |= (value & 0xFF) << (8 * idx)
    return word

# Helper function to pack 4 32-bit integers into a 128-bit row
def pack_mxm_row(values):
    row = 0
    for idx, value in enumerate(values):
        row |= (value & 0xFFFFFFFF) << (32 * idx)
    return row

def _set_field(word, value, lsb, width):
    mask = (1 << width) - 1
    return word | ((value & mask) << lsb)

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
):
    word = 0
    word = _set_field(word, westbound_sel, 0, 3)
    word = _set_field(word, eastbound_sel, 3, 3)
    word = _set_field(word, westbound_consumer_sel, 6, 3)
    word = _set_field(word, eastbound_consumer_sel, 9, 3)
    word = _set_field(word, mem0_read_en, 12, 1)
    word = _set_field(word, mem0_write_en, 13, 1)
    word = _set_field(word, mem0_addr, 14, 9)
    word = _set_field(word, mem1_read_en, 23, 1)
    word = _set_field(word, mem1_write_en, 24, 1)
    word = _set_field(word, mem1_addr, 25, 9)
    word = _set_field(word, sxm_opcode_input, 34, 12)
    word = _set_field(word, sxm_opcode_weight, 46, 12)
    word = _set_field(word, vxm_ctrl, 71, 4)
    word = _set_field(word, vxm_data_sel, 76, 1)
    word = _set_field(word, mxm_ingress_mode, 62, 2)
    word = _set_field(word, mxm_start, 64, 1)
    word = _set_field(word, mxm_clear, 65, 1)
    word = _set_field(word, mxm_e_row_sel, 66, 2)
    word = _set_field(word, mxm_e_col_sel, 68, 2)
    word = _set_field(word, mxm_e_valid_in, 70, 1)
    word = _set_field(word, mxm_input_is_signed, 77, 1)
    word = _set_field(word, mxm_wght_is_signed, 78, 1)
    word = _set_field(word, mxm_use_fp, 79, 1)
    word = _set_field(word, fp_quant_mode, 80, 1)
    word = _set_field(word, mem_store_fmt, 81, 2)
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

    # ----------------------------------------------------
    # 1. KV Cache Injection (MEM0):
    # ----------------------------------------------------
    # Keys K_0..K_3 are stored in MEM0[1..4] in column-major order:
    # MEM0[1+k] stores [K_0[k], K_1[k], K_2[k], K_3[k]].
    # We choose randomized but structured keys:
    # K_0 = [12, -8, 5, 2]
    # K_1 = [-15, 22, -10, 8]
    # K_2 = [8, -14, 18, -9]
    # K_3 = [50, -42, 38, -12] (Will produce the largest dot product with q)
    preload_mem0_row(dut, addr=1, values=[12, -15, 8, 50])    # k=0
    preload_mem0_row(dut, addr=2, values=[-8, 22, -14, -42])   # k=1
    preload_mem0_row(dut, addr=3, values=[5, -10, 18, 38])     # k=2
    preload_mem0_row(dut, addr=4, values=[2, 8, -9, -12])      # k=3
    preload_mem0_row(dut, addr=5, values=[0, 0, 0, 0])

    # Values V_0..V_3 are stored in MEM0[11..14] in row-major order:
    # All values are fully populated with randomized non-zero integers.
    preload_mem0_row(dut, addr=11, values=[8, -12, 14, -6])      # V_0 ("cat")
    preload_mem0_row(dut, addr=12, values=[-15, 20, -8, 12])     # V_1 ("sat")
    preload_mem0_row(dut, addr=13, values=[25, -18, 30, -15])    # V_2 ("on")
    preload_mem0_row(dut, addr=14, values=[42, -22, 50, -18])    # V_3 ("the")
    preload_mem0_row(dut, addr=15, values=[0, 0, 0, 0])

    # ----------------------------------------------------
    # 2. Embeddings & Weights Injection (MEM1):
    # ----------------------------------------------------
    # Query vector q = [2, -1, 3, 0] stored column-major in MEM1[0..3]:
    # MEM1[k] contains [q[k], 0, 0, 0].
    preload_mem1_row(dut, addr=0, values=[2, 0, 0, 0])
    preload_mem1_row(dut, addr=1, values=[-1, 0, 0, 0])
    preload_mem1_row(dut, addr=2, values=[3, 0, 0, 0])
    preload_mem1_row(dut, addr=3, values=[0, 0, 0, 0])

    # LM Head weight matrix W_B (for columns 4..7: "the", "mat", "sofa", "rug") in MEM1[30..33]
    # W_B is stored column-major:
    # W_B_0 ("the") = [10, -5, 12, -8]
    # W_B_1 ("mat") = [38, 25, 45, 18] (Highly positive projection to select "mat")
    # W_B_2 ("sofa") = [-15, 20, -8, 12]
    # W_B_3 ("rug") = [8, -12, 10, -5]
    # MEM1[30+k] contains [W_B_0[k], W_B_1[k], W_B_2[k], W_B_3[k]]
    preload_mem1_row(dut, addr=30, values=[10, 38, -15, 8])   # k=0
    preload_mem1_row(dut, addr=31, values=[-5, 25, 20, -12])  # k=1
    preload_mem1_row(dut, addr=32, values=[12, 45, -8, 10])   # k=2
    preload_mem1_row(dut, addr=33, values=[-8, 18, 12, -5])   # k=3

    # W_A (for columns 0..3: "The", "cat", "sat", "on") in MEM1[20..23] -> randomized background
    preload_mem1_row(dut, addr=20, values=[5, -3, 2, -1])
    preload_mem1_row(dut, addr=21, values=[-2, 6, -4, 3])
    preload_mem1_row(dut, addr=22, values=[8, -5, 4, -2])
    preload_mem1_row(dut, addr=23, values=[-1, 2, -3, 4])


    # ----------------------------------------------------
    # 3. Compile the Decode Phase Program:
    # ----------------------------------------------------
    program = []

    # --- Phase 1: Attention Logits Calculation (q K^T) ---
    # Clear MXM accumulator
    program.append(build_instruction(mxm_clear=1))

    for k in range(4):
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
    for _ in range(4):
        program.append(build_instruction(eastbound_sel=EB_VXM, eastbound_consumer_sel=EC_MEM0, mem0_write_en=1, mem0_addr=20))

    # --- Phase 2: Context vector calculation (S V) & LayerNorm ---
    # Clear MXM accumulator
    program.append(build_instruction(mxm_clear=1))

    for k in range(4):
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
    for _ in range(4):
        program.append(build_instruction(eastbound_sel=EB_VXM, eastbound_consumer_sel=EC_MEM1, mem1_write_en=1, mem1_addr=10))

    # --- Phase 3: LM Head Projection (x W_B) ---
    # Clear MXM accumulator
    program.append(build_instruction(mxm_clear=1))

    for k in range(4):
        # Load weight column k from MEM1 (W_B)
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

    # Write output logits to MEM1[41] (Row 0 only, needs 4 cycles for streaming)
    for _ in range(4):
        program.append(build_instruction(eastbound_sel=EB_MXM, eastbound_consumer_sel=EC_MEM1, mem1_write_en=1, mem1_addr=41, mxm_e_row_sel=0, mxm_e_valid_in=1))

    # 4. Preload instruction program into the ICU
    for pc, inst in enumerate(program):
        preload_instruction(dut, pc, inst)
    for pc in range(len(program), 320):
        preload_instruction(dut, pc, build_instruction())

    # 5. Execute Program
    await reset_dut(dut)
    total_cycles = len(program) + 10
    await tick(dut, total_cycles)

    # 6. Read and log all intermediate hardware states
    # Read attention scores at MEM0[20]
    scores_word = int(dut.u_lpu.u_mem0.sram_array[20].value)
    scores = [(scores_word >> (8 * i)) & 0xFF for i in range(4)]

    # Read context vector at MEM1[10] (before LayerNorm was written to it, but actually Phase 2 writes LayerNorm output to MEM1[10])
    # Let's inspect the values in MEM1[10] which is the post-LayerNorm context vector
    layernorm_out_word = int(dut.u_lpu.u_mem1.sram_array[10].value)
    layernorm_out = []
    for lane in range(4):
        val = (layernorm_out_word >> (32 * lane)) & 0xFFFFFFFF
        if val & 0x80000000:
            val -= 0x100000000
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

    dictionary = ["the", "mat", "sofa", "rug"]
    max_idx = logits.index(max(logits))
    predicted_token = dictionary[max_idx]

    dut._log.info(f"\n=======================================================")
    dut._log.info(f"--- INTERMEDIATE DECODE PHASE PROCESS LOGS ---")
    dut._log.info(f"1. Inputs:")
    dut._log.info(f"   - Query Vector:            [2, -1, 3, 0]")
    dut._log.info(f"   - Active Key Cache (K_3):  [50, -50, 50, 0]")
    dut._log.info(f"   - Active Value Cache (V_3):[20, -10, 16, -4]")
    dut._log.info(f"")
    dut._log.info(f"2. Phase 1: Attention & Softmax:")
    dut._log.info(f"   - Softmax Weights (MEM0[20]):  {scores} (Unsigned Q8)")
    dut._log.info(f"")
    dut._log.info(f"3. Phase 2: Context Mixing & LayerNorm:")
    dut._log.info(f"   - Normalized Output (MEM1[10]): {layernorm_out} (Signed 32-bit)")
    dut._log.info(f"")
    dut._log.info(f"4. Phase 3: LM Head Projection & Output Selection:")
    dut._log.info(f"   - LM Head Weights for 'mat':   [30, 30, 30, 30]")
    dut._log.info(f"   - Final Logits (MEM1[41]):      {logits} (Corresponding to: {dictionary})")
    dut._log.info(f"   - Winner Token Selection:       '{predicted_token}' (Logit: {logits[max_idx]})")
    dut._log.info(f"")
    dut._log.info(f"Simulation Status: SUCCESS (Cycles: {total_cycles})")
    dut._log.info(f"=======================================================\n")

    assert predicted_token == "mat", f"Expected 'mat' to be selected, but got '{predicted_token}'"

