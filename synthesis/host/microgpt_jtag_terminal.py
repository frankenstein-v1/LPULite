#!/usr/bin/env python3
"""Interactive MicroGPT runner for the DE1-SoC TinyLPU over USB JTAG.

This is a host terminal around the existing JTAG-to-Avalon bridge.  It does
not use UART and it does not modify the FPGA design.  Tensor-heavy stages are
run by the FPGA LPU from generated VLIW pages; the host performs JTAG data
movement, terminal I/O, and greedy token selection from the logits it reads
back.

The current RTL cannot update a dynamic K/V cache or form exact multi-token
attention without host staging.  By default this tool uses the hardware-produced
current-token V row as the attention context.  That is exact for position 0 and
is a useful bring-up path for proving prompt -> MEM0 -> LPU -> logits -> text.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SYSTEM_CONSOLE = Path(r"C:\altera_lite\25.1std\quartus\sopc_builder\bin\system-console.exe")
DEFAULT_SCHEDULE = ROOT / "model" / "artifacts" / "fpga_microgpt" / "microgpt_decode_schedule.json"
DEFAULT_IMEM = ROOT / "model" / "artifacts" / "fpga_microgpt" / "microgpt_decode_vliw.hex"
DEFAULT_MEM1 = ROOT / "model" / "artifacts" / "fpga_microgpt" / "microgpt_scheduler_mem1.hex"
DEFAULT_TRACE = ROOT / "model" / "artifacts" / "fpga_microgpt" / "microgpt_decode_trace.json"

IMEM_BASE = 0x0000
MEM0_BASE = 0x4000
MEM1_BASE = 0x8000
CTRL_RUN = 0xC000
CTRL_PC_LOAD = 0xC004
IMEM_WORDS = 1024
LANES = 8


class MicroGPTJTAGError(RuntimeError):
    pass


def read_hex_rows(path: Path, width: int) -> list[int]:
    limit = (1 << width) - 1
    rows: list[int] = []
    for line_no, raw in enumerate(path.read_text(encoding="ascii").splitlines(), start=1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        value = int(text.removeprefix("0x"), 16)
        if value > limit:
            raise MicroGPTJTAGError(f"{path}:{line_no}: row wider than {width} bits")
        rows.append(value)
    return rows


def signed8(value: int) -> int:
    value &= 0xFF
    return value - 0x100 if value & 0x80 else value


def unpack_row(row: int) -> tuple[list[int], int]:
    lanes = [signed8((row >> (8 * lane)) & 0xFF) for lane in range(LANES)]
    return lanes, signed8((row >> 64) & 0xFF)


def row_words(value: int) -> tuple[int, int, int]:
    return value & 0xFFFFFFFF, (value >> 32) & 0xFFFFFFFF, (value >> 64) & 0xFFFFFFFF


def append_row_write(lines: list[str], base: int, row: int, value: int) -> None:
    for word_index, word in enumerate(row_words(value)):
        lines.append(f"master_write_32 $m 0x{base + row * 12 + word_index * 4:04X} 0x{word:08X}")


def append_row_copy(lines: list[str], src_base: int, src_row: int, dst_base: int, dst_row: int) -> None:
    src = src_base + src_row * 12
    dst = dst_base + dst_row * 12
    lines += [
        f"set _dummy [master_read_32 $m 0x{src:04X} 1]",
        f"set row_words [master_read_32 $m 0x{src:04X} 3]",
        f"master_write_32 $m 0x{dst:04X} [lindex $row_words 0]",
        f"master_write_32 $m 0x{dst + 4:04X} [lindex $row_words 1]",
        f"master_write_32 $m 0x{dst + 8:04X} [lindex $row_words 2]",
    ]


def append_page(lines: list[str], page: list[int], after_ms: int) -> None:
    if len(page) > IMEM_WORDS:
        raise MicroGPTJTAGError(f"page has {len(page)} instructions; IMEM holds {IMEM_WORDS}")
    for row in range(IMEM_WORDS):
        append_row_write(lines, IMEM_BASE, row, page[row] if row < len(page) else 0)
    lines += [
        f"master_write_32 $m 0x{CTRL_PC_LOAD:04X} 0x00000000",
        f"master_write_32 $m 0x{CTRL_RUN:04X} 0x00000001",
        f"after {after_ms}",
        f"master_write_32 $m 0x{CTRL_RUN:04X} 0x00000000",
    ]


def append_program(lines: list[str], program: list[int], page_size: int, after_ms: int, label: str) -> None:
    for page_index, start in enumerate(range(0, len(program), page_size)):
        lines.append(f"puts \"MICROGPT_{label}_PAGE_BEGIN:{page_index}\"")
        append_page(lines, program[start:start + page_size], after_ms)
        lines.append(f"puts \"MICROGPT_{label}_PAGE_DONE:{page_index}\"")


def build_session_tcl(body: list[str]) -> str:
    lines = [
        "refresh_connections",
        "set masters [get_service_paths master]",
        "if {[llength $masters] == 0} { error \"No JTAG-to-Avalon master found\" }",
        "set m [lindex $masters 0]",
        "open_service master $m",
    ]
    lines.extend(body)
    lines += ["close_service master $m", "puts \"MICROGPT_TCL_DONE\""]
    return "\n".join(lines) + "\n"


def run_system_console(system_console: Path, tcl: str, timeout: int = 120) -> str:
    if not system_console.is_file():
        raise MicroGPTJTAGError(f"System Console was not found: {system_console}")
    with tempfile.TemporaryDirectory(prefix="tinylpu-microgpt-") as directory:
        script = Path(directory) / "microgpt_jtag.tcl"
        script.write_text(tcl, encoding="ascii")
        completed = subprocess.run(
            [str(system_console), f"--script={script}"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout,
        )
    if completed.returncode != 0 or "MICROGPT_TCL_DONE" not in completed.stdout:
        raise MicroGPTJTAGError(f"System Console failed:\n{completed.stdout}")
    return completed.stdout


def find_split_pc(trace_path: Path) -> int:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    for entry in trace:
        note = entry.get("note", "")
        if isinstance(note, str) and note.startswith("broadcast staged attention row0"):
            return int(entry["pc"])
    raise MicroGPTJTAGError("could not find staged-attention split in trace")


def parse_read_rows(output: str, tag: str) -> list[int]:
    for line in output.splitlines():
        if line.startswith(f"{tag}:"):
            parts = line.split(":", 1)[1].strip().split()
            words = [int(item, 0) for item in parts]
            if len(words) % 3 != 0:
                raise MicroGPTJTAGError(f"{tag} returned {len(words)} words, not a multiple of 3")
            rows = []
            for index in range(0, len(words), 3):
                rows.append(words[index] | (words[index + 1] << 32) | (words[index + 2] << 64))
            return rows
    raise MicroGPTJTAGError(f"missing {tag} readback in System Console output")


class MicroGPTTerminal:
    def __init__(
        self,
        *,
        schedule_path: Path,
        imem_path: Path,
        mem1_path: Path,
        trace_path: Path,
        system_console: Path,
        page_size: int,
        after_ms: int,
    ) -> None:
        self.schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
        self.program = read_hex_rows(imem_path, 96)
        self.mem1_rows = read_hex_rows(mem1_path, 72)
        split_pc = find_split_pc(trace_path)
        self.prefix_program = self.program[:split_pc]
        self.suffix_program = self.program[split_pc:]
        self.system_console = system_console
        self.page_size = page_size
        self.after_ms = after_ms
        tokenizer = self.schedule["tokenizer"]
        self.chars = tokenizer["characters"]
        self.bos_id = int(tokenizer["bos_token_id"])
        self.char_to_id = {char: index for index, char in enumerate(self.chars)}
        self.mem0 = self.schedule["mem0_abi"]
        self.symbols = self.schedule["mem1"]["symbols"]

    def load_weights(self) -> None:
        body = ["puts \"MICROGPT_LOAD_MEM1_BEGIN\""]
        for row, value in enumerate(self.mem1_rows):
            append_row_write(body, MEM1_BASE, row, value)
        body.append("puts \"MICROGPT_LOAD_MEM1_DONE\"")
        run_system_console(self.system_console, build_session_tcl(body), timeout=120)

    def embedding_rows(self, token_id: int, pos_id: int) -> tuple[int, int, int, int]:
        wte_base = int(self.symbols["wte"]["base_row"])
        wpe_base = int(self.symbols["wpe"]["base_row"])
        return (
            self.mem1_rows[wte_base + token_id * 2],
            self.mem1_rows[wte_base + token_id * 2 + 1],
            self.mem1_rows[wpe_base + pos_id * 2],
            self.mem1_rows[wpe_base + pos_id * 2 + 1],
        )

    def run_step(self, token_id: int, pos_id: int) -> list[float]:
        tok0, tok1, pos0, pos1 = self.embedding_rows(token_id, pos_id)
        token_rows = self.mem0["token_embedding_rows"]
        pos_rows = self.mem0["position_embedding_rows"]
        v_rows = self.mem0["v_rows"]
        staged_rows = self.mem0["staged_attention_rows"]
        logit_rows = self.mem0["logit_rows"]

        body: list[str] = []
        append_row_write(body, MEM0_BASE, int(token_rows[0]), tok0)
        append_row_write(body, MEM0_BASE, int(token_rows[1]), tok1)
        append_row_write(body, MEM0_BASE, int(pos_rows[0]), pos0)
        append_row_write(body, MEM0_BASE, int(pos_rows[1]), pos1)
        append_program(body, self.prefix_program, self.page_size, self.after_ms, "PREFIX")

        # Current-token attention bring-up mode: attention context = FPGA V row.
        append_row_copy(body, MEM0_BASE, int(v_rows[0]), MEM0_BASE, int(staged_rows[0]))
        append_row_copy(body, MEM0_BASE, int(v_rows[1]), MEM0_BASE, int(staged_rows[1]))

        append_program(body, self.suffix_program, self.page_size, self.after_ms, "SUFFIX")
        read_words: list[str] = []
        for row in logit_rows:
            addr = MEM0_BASE + int(row) * 12
            read_words.append(f"set _dummy [master_read_32 $m 0x{addr:04X} 1]")
            read_words.append(f"set part [master_read_32 $m 0x{addr:04X} 3]")
            read_words.append("append logits \" $part\"")
        body.append("set logits \"\"")
        body.extend(read_words)
        body.append("puts \"MICROGPT_LOGITS:$logits\"")
        output = run_system_console(self.system_console, build_session_tcl(body), timeout=240)
        rows = parse_read_rows(output, "MICROGPT_LOGITS")
        return self.decode_logits(rows)

    def decode_logits(self, rows: list[int]) -> list[float]:
        logits: list[float] = []
        for row in rows:
            lanes, scale = unpack_row(row)
            logits.extend(float(lane) * (2.0 ** scale) for lane in lanes)
        return logits[: len(self.chars) + 1]

    def encode_prompt(self, prompt: str) -> list[int]:
        ids = [self.bos_id]
        for char in prompt.lower():
            if char in self.char_to_id:
                ids.append(self.char_to_id[char])
        return ids[-16:]

    def token_to_text(self, token_id: int) -> str:
        if token_id == self.bos_id:
            return ""
        return self.chars[token_id] if 0 <= token_id < len(self.chars) else ""

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        ids = self.encode_prompt(prompt)
        generated = [self.token_to_text(token_id) for token_id in ids if token_id != self.bos_id]
        # Warm the prompt through the FPGA schedule; the final prompt token's
        # logits determine the first generated character.
        logits: list[float] | None = None
        for pos_id, token_id in enumerate(ids):
            logits = self.run_step(token_id, min(pos_id, 15))
        for step in range(max_new_tokens):
            if logits is None:
                logits = self.run_step(self.bos_id, 0)
            next_id = max(range(len(logits)), key=lambda idx: logits[idx])
            if next_id == self.bos_id:
                break
            generated.append(self.token_to_text(next_id))
            print(self.token_to_text(next_id), end="", flush=True)
            logits = self.run_step(next_id, min(len(ids) + step, 15))
        print()
        return "".join(generated)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--imem", type=Path, default=DEFAULT_IMEM)
    parser.add_argument("--mem1", type=Path, default=DEFAULT_MEM1)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--system-console", type=Path, default=DEFAULT_SYSTEM_CONSOLE)
    parser.add_argument("--page-size", type=int, default=900)
    parser.add_argument("--after-ms", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--no-load-weights", action="store_true", help="skip MEM1 load if it is already loaded")
    args = parser.parse_args()

    terminal = MicroGPTTerminal(
        schedule_path=args.schedule,
        imem_path=args.imem,
        mem1_path=args.mem1,
        trace_path=args.trace,
        system_console=args.system_console,
        page_size=args.page_size,
        after_ms=args.after_ms,
    )
    print("MicroGPT JTAG terminal")
    print("Mode: current-token attention staging. Exact for first token; approximate for multi-token prompts.")
    if not args.no_load_weights:
        print("Loading MEM1 model rows over JTAG...")
        terminal.load_weights()
        print("MEM1 load complete.")
    print("Type a-z prompts, or 'exit'.")

    while True:
        prompt = input("Prompt > ").strip()
        if prompt.lower() in {"exit", "quit"}:
            break
        if not prompt:
            continue
        print(prompt, end="", flush=True)
        try:
            terminal.generate(prompt, args.max_new_tokens)
        except MicroGPTJTAGError as exc:
            raise SystemExit(f"\nJTAG error: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
