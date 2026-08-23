#!/usr/bin/env python3
"""Host-side MicroGPT testbench that drives LPULite over JTAG.

This is intentionally a testbench/runtime, not a compiler.  It treats the FPGA
as an accelerator with visible SRAMs and drives it through Intel System Console:

1. stage token/position embedding rows into MEM0
2. run the existing Q/K/V prefix VLIW pages on the FPGA
3. copy FPGA-produced K/V rows into a KV cache region in FPGA MEM0
4. stage an attention context row pair
5. run the existing post-attention suffix pages on the FPGA
6. read logits from MEM0 and choose the next character

``--attention-mode host`` computes only the tiny causal attention context in
Python from FPGA-produced Q/K/V rows.  That is useful as a host-side TB when
you want software to own causal length/masking.  ``--attention-mode current``
does no attention math on the host; it simply stages the current V rows.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HOST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HOST_DIR))

from microgpt_jtag_terminal import (  # noqa: E402
    DEFAULT_IMEM,
    DEFAULT_MEM1,
    DEFAULT_SCHEDULE,
    DEFAULT_SYSTEM_CONSOLE,
    DEFAULT_TRACE,
    MEM0_BASE,
    MEM1_BASE,
    MicroGPTJTAGError,
    MicroGPTTerminal,
    append_program,
    append_row_copy,
    append_row_write,
    build_session_tcl,
    parse_read_rows,
    run_system_console,
    unpack_row,
)


LANES = 8
HIDDEN = 16
HEADS = 4
HEAD_DIM = HIDDEN // HEADS
ROWS_PER_VEC = HIDDEN // LANES
K_CACHE_BASE = 1024
V_CACHE_BASE = 2048


def pack_row(lanes: list[int], scale: int) -> int:
    padded = lanes[:LANES] + [0] * max(0, LANES - len(lanes))
    value = 0
    for idx, lane in enumerate(padded[:LANES]):
        value |= (int(lane) & 0xFF) << (idx * 8)
    value |= (int(scale) & 0xFF) << 64
    return value


def quantize_block(values: list[float]) -> int:
    absmax = max((abs(v) for v in values), default=0.0)
    scale = 0 if absmax == 0.0 else math.ceil(math.log2(absmax / 127.0))
    scale = max(-128, min(127, scale))
    inv = math.ldexp(1.0, -scale)
    lanes = [max(-127, min(127, int(round(v * inv)))) for v in values]
    return pack_row(lanes, scale)


def rows_to_vector(rows: list[int]) -> list[float]:
    values: list[float] = []
    for row in rows:
        lanes, scale = unpack_row(row)
        values.extend(float(lane) * (2.0 ** scale) for lane in lanes)
    return values[:HIDDEN]


def vector_to_rows(values: list[float]) -> list[int]:
    return [
        quantize_block(values[0:LANES]),
        quantize_block(values[LANES:2 * LANES]),
    ]


def softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    max_v = max(values)
    exps = [math.exp(v - max_v) for v in values]
    total = sum(exps)
    return [v / total for v in exps]


def host_attention_context(q_rows: list[int], k_cache: list[list[int]], v_cache: list[list[int]]) -> list[int]:
    q = rows_to_vector(q_rows)
    keys = [rows_to_vector(rows) for rows in k_cache]
    values = [rows_to_vector(rows) for rows in v_cache]
    output = [0.0] * HIDDEN
    for head in range(HEADS):
        base = head * HEAD_DIM
        scores = []
        for key in keys:
            dot = sum(q[base + i] * key[base + i] for i in range(HEAD_DIM))
            scores.append(dot / math.sqrt(float(HEAD_DIM)))
        weights = softmax(scores)
        for i in range(HEAD_DIM):
            output[base + i] = sum(weights[t] * values[t][base + i] for t in range(len(values)))
    return vector_to_rows(output)


class MicroGPTFPGATB(MicroGPTTerminal):
    """JTAG testbench with explicit FPGA-resident KV cache management."""

    def read_mem0_rows(self, rows: list[int], tag: str = "TB_READ_ROWS") -> list[int]:
        body = [f"set {tag.lower()} \"\""]
        for row in rows:
            addr = MEM0_BASE + row * 12
            body.append(f"set _dummy [master_read_32 $m 0x{addr:04X} 1]")
            body.append(f"set part [master_read_32 $m 0x{addr:04X} 3]")
            body.append(f"append {tag.lower()} \" $part\"")
        body.append(f"puts \"{tag}:${tag.lower()}\"")
        output = run_system_console(self.system_console, build_session_tcl(body), timeout=120)
        return parse_read_rows(output, tag)

    def write_mem0_rows(self, mapping: dict[int, int]) -> None:
        body: list[str] = []
        for row, value in mapping.items():
            append_row_write(body, MEM0_BASE, row, value)
        run_system_console(self.system_console, build_session_tcl(body), timeout=120)

    def load_step_inputs(self, token_id: int, pos_id: int) -> None:
        tok0, tok1, pos0, pos1 = self.embedding_rows(token_id, pos_id)
        token_rows = [int(row) for row in self.mem0["token_embedding_rows"]]
        pos_rows = [int(row) for row in self.mem0["position_embedding_rows"]]
        self.write_mem0_rows({
            token_rows[0]: tok0,
            token_rows[1]: tok1,
            pos_rows[0]: pos0,
            pos_rows[1]: pos1,
        })

    def run_prefix(self) -> None:
        body: list[str] = []
        append_program(body, self.prefix_program, self.page_size, self.after_ms, "PREFIX")
        run_system_console(self.system_console, build_session_tcl(body), timeout=240)

    def run_suffix_and_read_logits(self) -> list[float]:
        logit_rows = [int(row) for row in self.mem0["logit_rows"]]
        body: list[str] = []
        append_program(body, self.suffix_program, self.page_size, self.after_ms, "SUFFIX")
        body.append("set logits \"\"")
        for row in logit_rows:
            addr = MEM0_BASE + row * 12
            body.append(f"set _dummy [master_read_32 $m 0x{addr:04X} 1]")
            body.append(f"set part [master_read_32 $m 0x{addr:04X} 3]")
            body.append("append logits \" $part\"")
        body.append("puts \"TB_LOGITS:$logits\"")
        output = run_system_console(self.system_console, build_session_tcl(body), timeout=300)
        return self.decode_logits(parse_read_rows(output, "TB_LOGITS"))

    def cache_current_kv_on_fpga(self, pos_id: int) -> None:
        if not 0 <= pos_id < 16:
            raise MicroGPTJTAGError("MicroGPT block_size is 16; pos_id must be 0..15")
        k_rows = [int(row) for row in self.mem0["k_rows"]]
        v_rows = [int(row) for row in self.mem0["v_rows"]]
        body: list[str] = []
        for lane_row in range(ROWS_PER_VEC):
            append_row_copy(
                body,
                MEM0_BASE,
                k_rows[lane_row],
                MEM0_BASE,
                K_CACHE_BASE + pos_id * ROWS_PER_VEC + lane_row,
            )
            append_row_copy(
                body,
                MEM0_BASE,
                v_rows[lane_row],
                MEM0_BASE,
                V_CACHE_BASE + pos_id * ROWS_PER_VEC + lane_row,
            )
        run_system_console(self.system_console, build_session_tcl(body), timeout=120)

    def read_kv_cache_from_fpga(self, through_pos: int) -> tuple[list[list[int]], list[list[int]]]:
        k_cache: list[list[int]] = []
        v_cache: list[list[int]] = []
        for pos in range(through_pos + 1):
            k_cache.append(self.read_mem0_rows([K_CACHE_BASE + pos * ROWS_PER_VEC + row for row in range(ROWS_PER_VEC)]))
            v_cache.append(self.read_mem0_rows([V_CACHE_BASE + pos * ROWS_PER_VEC + row for row in range(ROWS_PER_VEC)]))
        return k_cache, v_cache

    def stage_attention(self, pos_id: int, mode: str) -> None:
        staged_rows = [int(row) for row in self.mem0["staged_attention_rows"]]
        if mode == "current":
            v_rows = [int(row) for row in self.mem0["v_rows"]]
            body: list[str] = []
            for lane_row in range(ROWS_PER_VEC):
                append_row_copy(body, MEM0_BASE, v_rows[lane_row], MEM0_BASE, staged_rows[lane_row])
            run_system_console(self.system_console, build_session_tcl(body), timeout=120)
            return

        if mode != "host":
            raise MicroGPTJTAGError(f"unknown attention mode: {mode}")
        q_rows = self.read_mem0_rows([int(row) for row in self.mem0["q_rows"]])
        k_cache, v_cache = self.read_kv_cache_from_fpga(pos_id)
        context_rows = host_attention_context(q_rows, k_cache, v_cache)
        self.write_mem0_rows({staged_rows[row]: context_rows[row] for row in range(ROWS_PER_VEC)})

    def run_token(self, token_id: int, pos_id: int, attention_mode: str) -> list[float]:
        self.load_step_inputs(token_id, pos_id)
        self.run_prefix()
        self.cache_current_kv_on_fpga(pos_id)
        self.stage_attention(pos_id, attention_mode)
        return self.run_suffix_and_read_logits()

    def generate_with_tb(self, prompt: str, max_new_tokens: int, attention_mode: str) -> str:
        ids = self.encode_prompt(prompt)
        text = "".join(self.token_to_text(token_id) for token_id in ids if token_id != self.bos_id)
        logits: list[float] | None = None
        for pos_id, token_id in enumerate(ids):
            logits = self.run_token(token_id, min(pos_id, 15), attention_mode)
        print(text, end="", flush=True)
        for step in range(max_new_tokens):
            if logits is None:
                logits = self.run_token(self.bos_id, 0, attention_mode)
            next_id = max(range(len(logits)), key=lambda idx: logits[idx])
            if next_id == self.bos_id:
                break
            ch = self.token_to_text(next_id)
            print(ch, end="", flush=True)
            text += ch
            next_pos = min(len(ids) + step, 15)
            logits = self.run_token(next_id, next_pos, attention_mode)
        print()
        return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-console", type=Path, default=DEFAULT_SYSTEM_CONSOLE)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--imem", type=Path, default=DEFAULT_IMEM)
    parser.add_argument("--mem1", type=Path, default=DEFAULT_MEM1)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--page-size", type=int, default=900)
    parser.add_argument("--after-ms", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--attention-mode", choices=("host", "current"), default="host")
    parser.add_argument("--no-load-weights", action="store_true")
    args = parser.parse_args()

    tb = MicroGPTFPGATB(
        schedule_path=args.schedule,
        imem_path=args.imem,
        mem1_path=args.mem1,
        trace_path=args.trace,
        system_console=args.system_console,
        page_size=args.page_size,
        after_ms=args.after_ms,
    )
    print("MicroGPT FPGA JTAG TB")
    print(f"KV cache in FPGA MEM0: K base={K_CACHE_BASE}, V base={V_CACHE_BASE}, rows/token={ROWS_PER_VEC}")
    print(f"attention mode: {args.attention_mode}")
    if not args.no_load_weights:
        print("Loading MEM1 weights/model rows over JTAG...")
        tb.load_weights()
        print("MEM1 load complete.")
    print("Type a-z prompts, or 'exit'.")
    while True:
        prompt = input("Prompt > ").strip()
        if prompt.lower() in {"exit", "quit"}:
            break
        if not prompt:
            continue
        tb.generate_with_tb(prompt, args.max_new_tokens, args.attention_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
