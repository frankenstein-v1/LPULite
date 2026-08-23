#!/usr/bin/env python3
"""LPULite VLIW Microcode Compiler.

Compiles high-level Transformer operations (Q*K^T matmul, Softmax activation,
S*V context matmul, RMSNorm, LM Head projection) into 96-bit VLIW instructions
for the Instruction Control Unit (ICU) in the LPULite FPGA hardware.
"""

WB_NONE = 0; WB_SXM = 1; WB_MEM0 = 2; WB_VXM = 3; WB_MEM1 = 4
EB_NONE = 0; EB_MXM = 1; EB_SXM = 2; EB_MEM0 = 3; EB_VXM = 4
WC_NONE = 0; WC_MXM = 1; WC_SXM = 2; WC_MEM0 = 3; WC_VXM = 4
EC_NONE = 0; EC_SXM = 1; EC_MEM0 = 2; EC_VXM = 3; EC_MEM1 = 4
INGRESS_NONE = 0; INGRESS_INPUT = 1; INGRESS_WGHT = 2

def _set_field(word: int, value: int, lsb: int, width: int) -> int:
    mask = (1 << width) - 1
    return word | ((value & mask) << lsb)

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
    sxm_transpose_load=0,
    sxm_transpose_emit=0,
    sxm_load_from_west=0,
    vxm_ctrl=0,
    vxm_data_sel=0,
    vxm_operand_sel=0,
    vxm_layernorm_en=0,
    vxm_rope_en=0,
    vxm_residual_op=0,
) -> int:
    """Pack instruction fields into a 96-bit integer for ICU IMEM."""
    word = 0
    word = _set_field(word, westbound_sel, 0, 3)
    word = _set_field(word, eastbound_sel, 3, 3)
    word = _set_field(word, westbound_consumer_sel, 6, 3)
    word = _set_field(word, eastbound_consumer_sel, 9, 3)
    word = _set_field(word, mem0_read_en, 12, 1)
    word = _set_field(word, mem0_write_en, 13, 1)
    word = _set_field(word, mem0_addr, 14, 15)
    word = _set_field(word, mem1_read_en, 29, 1)
    word = _set_field(word, mem1_write_en, 30, 1)
    word = _set_field(word, mem1_addr, 31, 15)
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
    word = _set_field(word, vxm_rope_en, 75, 1)
    word = _set_field(word, vxm_residual_op, 76, 3)
    return word

def compile_stories10k_vliw_program() -> list[int]:
    """Compile full 96-bit microcode program for stories10k decode step on LPULite hardware."""
    program = []

    # Phase 1: Clear MXM & calculate Q * K^T Attention Logits
    program.append(build_instruction(mxm_clear=1))

    for k in range(8):
        # Load Key column k from MEM0 (0x4000 + 1 + k)
        program.append(build_instruction(mem0_read_en=1, mem0_addr=1 + k))
        program.append(build_instruction(westbound_sel=WB_MEM0, westbound_consumer_sel=WC_MXM, mxm_ingress_mode=INGRESS_WGHT))
        # Load Query element k from MEM1 (0x8000 + k)
        program.append(build_instruction(mem1_read_en=1, mem1_addr=k))
        program.append(build_instruction(westbound_sel=WB_MEM1, westbound_consumer_sel=WC_MXM, mxm_ingress_mode=INGRESS_INPUT))
        # Multiply-accumulate impulse
        program.append(build_instruction(mxm_start=1))
        program.append(build_instruction(mxm_start=1))

    program.extend([build_instruction(), build_instruction()])

    # Stream Attention scores through VXM Softmax to MEM0[20]
    program.append(build_instruction(eastbound_sel=EB_MXM, eastbound_consumer_sel=EC_VXM, mxm_e_row_sel=0, mxm_e_valid_in=1, vxm_ctrl=0b1100, vxm_data_sel=1))
    program.append(build_instruction(vxm_ctrl=0b1100, vxm_data_sel=1))
    for _ in range(10):
        program.append(build_instruction())
    for _ in range(8):
        program.append(build_instruction(eastbound_sel=EB_VXM, eastbound_consumer_sel=EC_MEM0, mem0_write_en=1, mem0_addr=20))

    # Phase 2: Calculate Context Vector (S * V) & RMSNorm
    program.append(build_instruction(mxm_clear=1))

    for k in range(8):
        # Load Value row k from MEM0 (0x4000 + 11 + k)
        program.append(build_instruction(mem0_read_en=1, mem0_addr=11 + k))
        program.append(build_instruction(westbound_sel=WB_MEM0, westbound_consumer_sel=WC_MXM, mxm_ingress_mode=INGRESS_WGHT))
        # Load Softmax weights S from MEM0[20]
        program.append(build_instruction(mem0_read_en=1, mem0_addr=20))
        program.append(build_instruction(westbound_sel=WB_MEM0, westbound_consumer_sel=WC_MXM, mxm_ingress_mode=INGRESS_INPUT))
        # Multiply-accumulate impulse
        program.append(build_instruction(mxm_start=1))
        program.append(build_instruction(mxm_start=1))

    program.extend([build_instruction(), build_instruction()])

    # Stream Context Vector through VXM RMSNorm to MEM1[10]
    program.append(build_instruction(eastbound_sel=EB_MXM, eastbound_consumer_sel=EC_VXM, mxm_e_row_sel=0, mxm_e_valid_in=1, vxm_ctrl=0b0000, vxm_data_sel=1))
    program.append(build_instruction(vxm_ctrl=0b0000, vxm_data_sel=1))
    for _ in range(10):
        program.append(build_instruction())
    for _ in range(8):
        program.append(build_instruction(eastbound_sel=EB_VXM, eastbound_consumer_sel=EC_MEM1, mem1_write_en=1, mem1_addr=10))

    # Phase 3: Complete LM Head Projection across all 512 vocabulary tokens
    vocab_chunks = 64 # 64 superlanes x 8 = 512 tokens
    for chunk in range(vocab_chunks):
        program.append(build_instruction(mxm_clear=1))
        for k in range(8):
            tok_idx = chunk * 8 + k
            # Load weight column tok_idx from MEM1 (LM Head weights at 0x8000 + 30 + tok_idx)
            program.append(build_instruction(mem1_read_en=1, mem1_addr=30 + tok_idx))
            program.append(build_instruction(westbound_sel=WB_MEM1, westbound_consumer_sel=WC_MXM, mxm_ingress_mode=INGRESS_WGHT))
            # Load normalized input element k from MEM1[10]
            program.append(build_instruction(mem1_read_en=1, mem1_addr=10))
            program.append(build_instruction(westbound_sel=WB_MEM1, westbound_consumer_sel=WC_MXM, mxm_ingress_mode=INGRESS_INPUT))
            # Multiply-accumulate impulse
            program.append(build_instruction(mxm_start=1))
            program.append(build_instruction(mxm_start=1))

        program.extend([build_instruction(), build_instruction()])

        # Write hardware-computed logits for this vocab chunk to MEM0[chunk]
        program.append(build_instruction(eastbound_sel=EB_MXM, eastbound_consumer_sel=EC_MEM0, mxm_e_row_sel=0, mxm_e_valid_in=1, mem0_write_en=1, mem0_addr=chunk))
        for _ in range(7):
            program.append(build_instruction(eastbound_sel=EB_MXM, eastbound_consumer_sel=EC_MEM0, mem0_write_en=1, mem0_addr=chunk))

    return program
