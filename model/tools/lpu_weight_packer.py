#!/usr/bin/env python3
"""LPULite Weight Memory Packer.

Formats model quantized weight matrices (Q/K/V projections, Out projection,
RMSNorm scale values, LM Head matrix) into packed 32-bit words for FPGA MEM1 SRAM (0x8000).
"""

import json
from pathlib import Path

def pack_int8_row_to_words(row_bytes: list[int]) -> list[int]:
    """Pack an array of signed 8-bit integers into 32-bit words (4 bytes per 32-bit word)."""
    words = []
    for i in range(0, len(row_bytes), 4):
        chunk = row_bytes[i:i+4]
        word = 0
        for byte_idx, val in enumerate(chunk):
            ival = int(round(val)) & 0xFF
            word |= (ival << (byte_idx * 8))
        words.append(word)
    return words

def prepare_mem1_weights(weights: dict, config: dict) -> dict[int, list[int]]:
    """Convert PyTorch float / INT8 weights into FPGA MEM1 word rows mapped by row address."""
    mem1_map = {}

    def get_tensor_row(key, row_idx):
        val = weights.get(key)
        if val is None: return [0] * 64
        if hasattr(val, 'tolist'): val = val.tolist()
        if row_idx < len(val):
            r = val[row_idx]
            if hasattr(r, 'tolist'): r = r.tolist()
            return [int(round(x * 127.0)) if isinstance(x, (float, int)) else 0 for x in r]
        return [0] * 64

    # 1. Query Head vector initialization (MEM1[0..7])
    for k in range(8):
        col_vals = []
        for r in range(8):
            row_data = get_tensor_row("blocks.0.attn.q_proj.weight", r)
            col_vals.append(row_data[k] if k < len(row_data) else 0)
        mem1_map[k] = pack_int8_row_to_words(col_vals)

    # 2. LM Head Weight Matrix (MEM1[30..541] for up to 512 vocabulary tokens)
    lm_head = weights.get("lm_head.weight")
    vocab_size = len(lm_head) if lm_head is not None else 512

    for tok_idx in range(vocab_size):
        row_vals = get_tensor_row("lm_head.weight", tok_idx)[:64]
        mem1_map[30 + tok_idx] = pack_int8_row_to_words(row_vals)

    return mem1_map
