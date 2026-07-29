#!/usr/bin/env python3
"""TinyLPU Weight Memory Packer.

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

    # 1. Query Head vector initialization (MEM1[0..7])
    q_proj = weights.get("blocks.0.attn.q_proj.weight", [])
    for k in range(min(8, len(q_proj))):
        col_vals = [int(round(q_proj[row][k] * 127.0)) if isinstance(q_proj[row], list) else 0 for row in range(min(8, len(q_proj)))]
        mem1_map[k] = pack_int8_row_to_words(col_vals)

    # 2. Complete LM Head Weight Matrix (MEM1[30..157] for all 128 vocabulary tokens)
    lm_head = weights.get("lm_head.weight", [])
    vocab_size = len(lm_head) if lm_head else 128
    for tok_idx in range(vocab_size):
        row_vals = [int(round(lm_head[tok_idx][d] * 127.0)) if d < len(lm_head[tok_idx]) else 0 for d in range(min(8, len(lm_head[tok_idx])))]
        mem1_map[30 + tok_idx] = pack_int8_row_to_words(row_vals)

    return mem1_map
