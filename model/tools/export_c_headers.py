#!/usr/bin/env python3
"""LPULite C Header Exporter.

Converts PyTorch exported weights and VLIW microcode instructions into C header files
(synthesis/driver/include/*.h) for compile-time embedding in the ARM C driver.
"""

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
WEIGHTS_PATH = ROOT_DIR / "model" / "datasets" / "stories10k" / "stories10k_weights_export.json"
VOCAB_PATH = ROOT_DIR / "model" / "datasets" / "stories10k" / "vocab.json"
INCLUDE_DIR = ROOT_DIR / "synthesis" / "driver" / "include"

if str(ROOT_DIR / "model" / "tools") not in sys.path:
    sys.path.append(str(ROOT_DIR / "model" / "tools"))

from lpu_vliw_compiler import compile_stories10k_vliw_program
from lpu_weight_packer import prepare_mem1_weights

def export_vliw_header(vliw_program: list[int], out_file: Path):
    """Export 96-bit VLIW microcode program into include/lpu_vliw.h as a 2D uint32_t array."""
    lines = [
        "/* Auto-generated LPULite VLIW Microcode Header */",
        "#ifndef LPU_VLIW_H",
        "#define LPU_VLIW_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define VLIW_INSTRUCTION_COUNT {len(vliw_program)}",
        "",
        "static const uint32_t g_vliw_program[VLIW_INSTRUCTION_COUNT][3] = {"
    ]
    for inst in vliw_program:
        w0 = inst & 0xFFFFFFFF
        w1 = (inst >> 32) & 0xFFFFFFFF
        w2 = (inst >> 64) & 0xFFFFFFFF
        lines.append(f"    {{ 0x{w0:08X}U, 0x{w1:08X}U, 0x{w2:08X}U }},")
    lines.append("};")
    lines.append("")
    lines.append("#endif /* LPU_VLIW_H */")
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Exported {len(vliw_program)} VLIW instructions to {out_file.relative_to(ROOT_DIR)}")

def export_weights_header(weights: dict, vocab: dict, out_file: Path):
    """Export packed MEM1 weights and token embedding lookup arrays to include/lpu_weights.h."""
    token_emb = weights.get("token_emb.weight")
    if hasattr(token_emb, "tolist"): token_emb = token_emb.tolist()
    if token_emb is None: token_emb = []

    mem1_weights = prepare_mem1_weights(weights, {})

    vocab_size = len(token_emb) if token_emb else 512
    embed_dim = len(token_emb[0]) if token_emb and len(token_emb) > 0 else 64

    # Build vocab mapping tables
    id_to_word = {}
    if isinstance(vocab, dict):
        for k, v in vocab.items():
            if isinstance(v, dict):
                id_to_word[v.get("id", 0)] = k
            elif isinstance(v, int):
                id_to_word[v] = k
            else:
                id_to_word[k] = v

    lines = [
        "/* Auto-generated LPULite Model Weights & Vocabulary Header */",
        "#ifndef LPU_WEIGHTS_H",
        "#define LPU_WEIGHTS_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define LPU_VOCAB_SIZE {vocab_size}",
        f"#define LPU_EMBED_DIM  {embed_dim}",
        f"#define LPU_MEM1_ROWS  600",
        "",
        "/* Packed MEM1 Weights (Row Address 0..599, 3 x 32-bit words per 96-bit row) */",
        "static const uint32_t g_mem1_weights[LPU_MEM1_ROWS][3] = {"
    ]

    for row in range(600):
        words = mem1_weights.get(row, [0, 0, 0])
        w0 = words[0] if len(words) > 0 else 0
        w1 = words[1] if len(words) > 1 else 0
        w2 = words[2] if len(words) > 2 else 0
        lines.append(f"    /* Row {row:3d} */ {{ 0x{w0:08X}U, 0x{w1:08X}U, 0x{w2:08X}U }},")
    lines.append("};")
    lines.append("")

    lines.append("/* Quantized INT8 Token Embeddings Array [512][64] */")
    lines.append("static const int8_t g_token_embeddings[LPU_VOCAB_SIZE][LPU_EMBED_DIM] = {")
    for tok_id in range(vocab_size):
        row_vec = token_emb[tok_id] if tok_id < len(token_emb) else [0] * embed_dim
        if hasattr(row_vec, "tolist"): row_vec = row_vec.tolist()
        row_vals = [max(-128, min(127, int(round(row_vec[d] * 127.0)))) if d < len(row_vec) else 0 for d in range(embed_dim)]
        str_vals = ", ".join(f"{v:4d}" for v in row_vals)
        lines.append(f"    /* Token {tok_id:3d} */ {{ {str_vals} }},")
    lines.append("};")
    lines.append("")

    lines.append("/* Vocabulary ID to Token String Mapping */")
    lines.append("static const char *g_vocab_words[LPU_VOCAB_SIZE] = {")
    for tok_id in range(vocab_size):
        word = id_to_word.get(tok_id, f"<token_{tok_id}>")
        escaped_word = word.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
        lines.append(f'    "{escaped_word}",')
    lines.append("};")
    lines.append("")
    lines.append("#endif /* LPU_WEIGHTS_H */")

    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Exported packed weights & vocab to {out_file.relative_to(ROOT_DIR)}")

def main():
    INCLUDE_DIR.mkdir(parents=True, exist_ok=True)
    pt_path = ROOT_DIR / "model" / "stories288k" / "stories288k_model.pt"
    
    if pt_path.is_file():
        import torch
        ckpt = torch.load(pt_path, map_location="cpu")
        weights = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        vocab_raw = ckpt.get("vocab", {})
        if isinstance(vocab_raw, dict):
            vocab = {v if not isinstance(v, dict) else k: k if not isinstance(v, dict) else v.get("id", i) for i, (k, v) in enumerate(vocab_raw.items())}
        else:
            vocab = {tok: i for i, tok in enumerate(vocab_raw)}
    else:
        with open(WEIGHTS_PATH, "r", encoding="utf-8") as f:
            export = json.load(f)
        weights = export["weights"]
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            vocab = json.load(f)

    vliw_prog = compile_stories10k_vliw_program()
    export_vliw_header(vliw_prog, INCLUDE_DIR / "lpu_vliw.h")
    export_weights_header(weights, vocab, INCLUDE_DIR / "lpu_weights.h")

if __name__ == "__main__":
    main()
