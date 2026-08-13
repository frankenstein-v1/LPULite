#!/usr/bin/env python3
"""Export the current FPGA MicroGPT image as C headers for ARM/HPS Linux.

The source of truth remains model/artifacts/fpga_microgpt:

* microgpt_scheduler_mem1.hex
* microgpt_decode_vliw.hex
* microgpt_decode_schedule.json
* microgpt_decode_trace.json

This script only repackages those files for a small userspace C runtime that
talks to the TinyLPU over the DE1-SoC HPS-to-FPGA lightweight bridge.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = ROOT / "model" / "artifacts" / "fpga_microgpt"
DEFAULT_OUTPUT = ROOT / "synthesis" / "linux" / "include" / "microgpt_hps_image.h"
DEFAULT_CHECKPOINT = ROOT / "model" / "artifacts" / "microgpt_weights_int8.json"


def read_hex_rows(path: Path, width: int) -> list[int]:
    limit = (1 << width) - 1
    rows: list[int] = []
    for line_no, raw in enumerate(path.read_text(encoding="ascii").splitlines(), start=1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        value = int(text.removeprefix("0x"), 16)
        if value > limit:
            raise ValueError(f"{path}:{line_no}: value is wider than {width} bits")
        rows.append(value)
    return rows


def row_words(value: int) -> tuple[int, int, int]:
    return (
        value & 0xFFFFFFFF,
        (value >> 32) & 0xFFFFFFFF,
        (value >> 64) & 0xFFFFFFFF,
    )


def find_split_pc(trace_path: Path) -> int:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    for entry in trace:
        note = entry.get("note", "")
        if isinstance(note, str) and note.startswith("broadcast staged attention row0"):
            return int(entry["pc"])
    raise RuntimeError("could not find prefix/suffix split in decode trace")


def find_pc(trace: list[dict], prefix: str) -> int:
    for entry in trace:
        note = entry.get("note", "")
        if isinstance(note, str) and note.startswith(prefix):
            return int(entry["pc"])
    raise RuntimeError(f"could not find trace entry starting with {prefix!r}")


def find_broadcast_end_pc(trace: list[dict], prefix: str) -> int:
    return find_pc(trace, f"{prefix}: write broadcast lane 7") + 1


def c_array_rows(name: str, rows: list[int]) -> str:
    lines = [f"static const tinylpu_row96_t {name}[] = {{"]
    for value in rows:
        w0, w1, w2 = row_words(value)
        lines.append(f"    {{0x{w0:08X}u, 0x{w1:08X}u, 0x{w2:08X}u}},")
    lines.append("};")
    return "\n".join(lines)


def c_string_array(name: str, values: list[str]) -> str:
    lines = [f"static const char *const {name}[] = {{"]
    for value in values:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'    "{escaped}",')
    lines.append("};")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    artifact_dir = args.artifact_dir
    schedule_path = artifact_dir / "microgpt_decode_schedule.json"
    trace_path = artifact_dir / "microgpt_decode_trace.json"
    imem_path = artifact_dir / "microgpt_decode_vliw.hex"
    mem1_path = artifact_dir / "microgpt_scheduler_mem1.hex"
    softmax_path = artifact_dir / "microgpt_softmax_vliw.hex"
    attention_path = artifact_dir / "microgpt_attention_vliw.hex"

    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    imem = read_hex_rows(imem_path, 96)
    mem1 = read_hex_rows(mem1_path, 72)
    softmax_imem = read_hex_rows(softmax_path, 96)
    attention_imem = read_hex_rows(attention_path, 96)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    split_pc = find_split_pc(trace_path)
    prefix_attention_bcast_start = find_pc(trace, "broadcast attention input row0")
    prefix_wq_start = find_pc(trace, "layer0.attn_wq: clear MXM output block 0")
    suffix_attention_proj_start = find_pc(trace, "attn output projection: clear MXM output block 0")
    suffix_mlp_bcast_start = find_pc(trace, "broadcast mlp row0")
    suffix_mlp_fc1_start = find_pc(trace, "mlp fc1: clear MXM output block 0")
    hidden_bcast_starts = [find_pc(trace, f"broadcast mlp hidden {idx}") for idx in range(8)]
    hidden_bcast_ends = [find_broadcast_end_pc(trace, f"broadcast mlp hidden {idx}") for idx in range(8)]
    final_bcast_start = find_pc(trace, "broadcast final row0")
    lm_head_start = find_pc(trace, "lm head: clear MXM output block 0")
    mem0 = schedule["mem0_abi"]
    symbols = schedule["mem1"]["symbols"]
    config = schedule["config"]
    tokenizer = schedule["tokenizer"]
    attention_meta = schedule["microkernels"]["attention"]
    attention_sections = attention_meta["sections"]
    target_names: list[str] = []
    if DEFAULT_CHECKPOINT.is_file():
        checkpoint = json.loads(DEFAULT_CHECKPOINT.read_text(encoding="utf-8"))
        names = checkpoint.get("training", {}).get("target_names", [])
        if isinstance(names, list):
            target_names = [str(name).lower() for name in names]

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    chars = "".join(tokenizer["characters"])

    contents = f"""/* Auto-generated by synthesis/scripts/export_microgpt_hps_headers.py.
 * Source artifacts: {artifact_dir.as_posix()}
 * Do not edit this file by hand; regenerate it after rebuilding the schedule.
 */
#ifndef MICROGPT_HPS_IMAGE_H
#define MICROGPT_HPS_IMAGE_H

#include <stdint.h>
#include <stddef.h>

typedef struct {{
    uint32_t w0;
    uint32_t w1;
    uint32_t w2;
}} tinylpu_row96_t;

#define MICROGPT_N_LAYER {int(config["n_layer"])}
#define MICROGPT_N_EMBD {int(config["n_embd"])}
#define MICROGPT_BLOCK_SIZE {int(config["block_size"])}
#define MICROGPT_N_HEAD {int(config["n_head"])}
#define MICROGPT_HEAD_DIM (MICROGPT_N_EMBD / MICROGPT_N_HEAD)
#define MICROGPT_LANES 8
#define MICROGPT_ROWS_PER_VEC (MICROGPT_N_EMBD / MICROGPT_LANES)
#define MICROGPT_VOCAB_SIZE {int(config["vocab_size"])}
#define MICROGPT_BOS_TOKEN_ID {int(tokenizer["bos_token_id"])}
#define MICROGPT_TOKEN_CHARS "{chars}"
#define MICROGPT_TARGET_NAME_COUNT {len(target_names)}

#define MICROGPT_IMEM_INSTRUCTIONS {len(imem)}
#define MICROGPT_PREFIX_INSTRUCTIONS {split_pc}
#define MICROGPT_SUFFIX_INSTRUCTIONS {len(imem) - split_pc}
#define MICROGPT_IMEM_PAGE_SIZE {int(schedule["imem"]["page_size"])}
#define MICROGPT_MEM1_ROWS {len(mem1)}
#define MICROGPT_SOFTMAX_IN_BASE {int(mem0["softmax_input_rows"][0])}
#define MICROGPT_SOFTMAX_OUT_BASE {int(mem0["softmax_output_rows"][0])}
#define MICROGPT_SOFTMAX_CHUNKS 16
#define MICROGPT_SOFTMAX_INSTRUCTIONS {len(softmax_imem)}
#define MICROGPT_ATTENTION_INSTRUCTIONS {len(attention_imem)}
#define MICROGPT_ATTN_K_TRANSPOSE_BLOCK0_START {int(attention_sections["k_transpose_block0"]["start"])}
#define MICROGPT_ATTN_K_TRANSPOSE_BLOCK0_INSTRUCTIONS {int(attention_sections["k_transpose_block0"]["instructions"])}
#define MICROGPT_ATTN_K_TRANSPOSE_BLOCK1_START {int(attention_sections["k_transpose_block1"]["start"])}
#define MICROGPT_ATTN_K_TRANSPOSE_BLOCK1_INSTRUCTIONS {int(attention_sections["k_transpose_block1"]["instructions"])}
#define MICROGPT_ATTN_QK_START {int(attention_sections["qk"]["start"])}
#define MICROGPT_ATTN_QK_INSTRUCTIONS {int(attention_sections["qk"]["instructions"])}
#define MICROGPT_ATTN_SOFTMAX_START {int(attention_sections["softmax"]["start"])}
#define MICROGPT_ATTN_SOFTMAX_INSTRUCTIONS {int(attention_sections["softmax"]["instructions"])}
#define MICROGPT_ATTN_PV_START {int(attention_sections["pv"]["start"])}
#define MICROGPT_ATTN_PV_INSTRUCTIONS {int(attention_sections["pv"]["instructions"])}
#define MICROGPT_ATTN_MERGE_START {int(attention_sections["merge"]["start"])}
#define MICROGPT_ATTN_MERGE_INSTRUCTIONS {int(attention_sections["merge"]["instructions"])}

#define MICROGPT_PREFIX_ATTN_BCAST_START {prefix_attention_bcast_start}
#define MICROGPT_PREFIX_WQ_START {prefix_wq_start}
#define MICROGPT_SUFFIX_ATTN_PROJ_START {suffix_attention_proj_start}
#define MICROGPT_SUFFIX_MLP_BCAST_START {suffix_mlp_bcast_start}
#define MICROGPT_SUFFIX_MLP_FC1_START {suffix_mlp_fc1_start}
#define MICROGPT_SUFFIX_FINAL_BCAST_START {final_bcast_start}
#define MICROGPT_SUFFIX_LM_HEAD_START {lm_head_start}

#define MICROGPT_MEM0_TOKEN_ROW0 {int(mem0["token_embedding_rows"][0])}
#define MICROGPT_MEM0_TOKEN_ROW1 {int(mem0["token_embedding_rows"][1])}
#define MICROGPT_MEM0_POS_ROW0 {int(mem0["position_embedding_rows"][0])}
#define MICROGPT_MEM0_POS_ROW1 {int(mem0["position_embedding_rows"][1])}
#define MICROGPT_MEM0_ATTN_ROW0 {int(mem0["staged_attention_rows"][0])}
#define MICROGPT_MEM0_ATTN_ROW1 {int(mem0["staged_attention_rows"][1])}
#define MICROGPT_MEM0_Q_ROW0 {int(mem0["q_rows"][0])}
#define MICROGPT_MEM0_Q_ROW1 {int(mem0["q_rows"][1])}
#define MICROGPT_MEM0_K_ROW0 {int(mem0["k_rows"][0])}
#define MICROGPT_MEM0_K_ROW1 {int(mem0["k_rows"][1])}
#define MICROGPT_MEM0_V_ROW0 {int(mem0["v_rows"][0])}
#define MICROGPT_MEM0_V_ROW1 {int(mem0["v_rows"][1])}
#define MICROGPT_MEM0_LOGIT_ROW0 {int(mem0["logit_rows"][0])}
#define MICROGPT_MEM0_LOGIT_ROW1 {int(mem0["logit_rows"][1])}
#define MICROGPT_MEM0_LOGIT_ROW2 {int(mem0["logit_rows"][2])}
#define MICROGPT_MEM0_LOGIT_ROW3 {int(mem0["logit_rows"][3])}

#define MICROGPT_MEM0_XN_ROW0 10
#define MICROGPT_MEM0_XN_ROW1 11
#define MICROGPT_MEM0_X_ROW0 8
#define MICROGPT_MEM0_X_ROW1 9
#define MICROGPT_MEM0_X_BCAST_BASE 32
#define MICROGPT_MEM0_ATTN_BCAST_BASE 112
#define MICROGPT_MEM0_MLP_BCAST_BASE 192
#define MICROGPT_MEM0_MLP_H_BASE 256
#define MICROGPT_MEM0_MLP_H_BCAST_BASE 320
#define MICROGPT_MEM0_ATTN_Q_BCAST_BASE {int(mem0["attention_q_broadcast_rows"][0])}
#define MICROGPT_MEM0_ATTN_QK_SCORE_BASE {int(mem0["attention_qk_score_rows"][0])}
#define MICROGPT_MEM0_ATTN_PV_PROB_BASE {int(mem0["attention_pv_probability_rows"][0])}
#define MICROGPT_MEM0_ATTN_PV_OUT_ROW {int(mem0["attention_pv_output_row"])}
#define MICROGPT_MEM0_ATTN_HEAD_OUT_BASE {int(mem0["attention_head_output_rows"][0])}
#define MICROGPT_MEM0_ATTN_K_TILE_IN_BASE {int(mem0["attention_k_tile_input_rows"][0])}

static const uint32_t g_microgpt_hidden_bcast_start[8] = {{
{''.join(f"    {value}u,\n" for value in hidden_bcast_starts)}}};

static const uint32_t g_microgpt_hidden_bcast_end[8] = {{
{''.join(f"    {value}u,\n" for value in hidden_bcast_ends)}}};

#define MICROGPT_MEM1_WTE_BASE {int(symbols["wte"]["base_row"])}
#define MICROGPT_MEM1_WPE_BASE {int(symbols["wpe"]["base_row"])}

#define MICROGPT_K_CACHE_BASE 1024
#define MICROGPT_V_CACHE_BASE 1024
#define MICROGPT_MEM1_ATTN_KT_STAGE_BASE {int(attention_meta["mem1_k_transpose_stage_rows"][0])}
#define MICROGPT_MEM1_ATTN_V_STAGE_BASE {int(attention_meta["mem1_v_stage_rows"][0])}

{c_array_rows("g_microgpt_vliw", imem)}

{c_array_rows("g_microgpt_softmax_vliw", softmax_imem)}

{c_array_rows("g_microgpt_attention_vliw", attention_imem)}

{c_array_rows("g_microgpt_mem1", mem1)}

{c_string_array("g_microgpt_target_names", target_names)}

#endif /* MICROGPT_HPS_IMAGE_H */
"""

    output.write_text(contents, encoding="utf-8", newline="\n")
    print(f"wrote {output.relative_to(ROOT)}")
    print(f"  VLIW instructions: {len(imem)} (prefix {split_pc}, suffix {len(imem) - split_pc})")
    print(f"  MEM1 rows: {len(mem1)}")
    print(f"  softmax kernel: {len(softmax_imem)} instructions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
