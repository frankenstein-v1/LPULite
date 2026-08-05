#!/usr/bin/env python3
"""Run a long TinyLPU VLIW image over JTAG without changing LPU RTL.

The synthesized DE1-SoC image has a 1024-word IMEM.  This utility executes a
long, statically compiled program as fixed-size pages: model/scratch SRAM is
loaded once, then each VLIW page replaces IMEM, executes, and leaves its SRAM
results available to the next page.  No tensor arithmetic occurs on the host.

The program must be page-safe: each page has a static cycle count and its
state is passed only through MEM0/MEM1.  The tool clears unused IMEM words for
every page, so an overlong JTAG timing delay can execute only NOPs after the
page body.
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SYSTEM_CONSOLE = Path(r"C:\altera_lite\25.1std\quartus\sopc_builder\bin\system-console.exe")
IMEM_WORDS = 1024
IMEM_BASE = 0x0000
MEM0_BASE = 0x4000
MEM1_BASE = 0x8000
CTRL_RUN = 0xC000
CTRL_PC_LOAD = 0xC004
WORDS_PER_ROW = 3


class JtagPagerError(RuntimeError):
    pass


def read_hex_rows(path: Path, width: int) -> list[int]:
    values: list[int] = []
    allowed = (1 << width) - 1
    for number, raw in enumerate(path.read_text(encoding="ascii").splitlines(), start=1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        try:
            value = int(text.removeprefix("0x"), 16)
        except ValueError as exc:
            raise JtagPagerError(f"{path}:{number}: invalid hexadecimal row") from exc
        if value > allowed:
            raise JtagPagerError(f"{path}:{number}: value is wider than {width} bits")
        values.append(value)
    return values


def words96(value: int) -> tuple[int, int, int]:
    return value & 0xFFFFFFFF, (value >> 32) & 0xFFFFFFFF, (value >> 64) & 0xFFFFFFFF


def append_row_writes(lines: list[str], base: int, row_index: int, value: int) -> None:
    for word_index, word in enumerate(words96(value)):
        lines.append(f"master_write_32 $m 0x{base + row_index * 12 + word_index * 4:04X} 0x{word:08X}")


def append_page(lines: list[str], page: list[int], cycle_padding_ms: int) -> None:
    if len(page) > IMEM_WORDS:
        raise JtagPagerError(f"page has {len(page)} instructions; IMEM holds {IMEM_WORDS}")
    # Erase stale instructions.  A Tcl wait is millisecond-granular, whereas
    # the FPGA runs at 50 MHz; executing past the body must therefore be safe.
    for row in range(IMEM_WORDS):
        append_row_writes(lines, IMEM_BASE, row, page[row] if row < len(page) else 0)
    lines += [
        f"master_write_32 $m 0x{CTRL_PC_LOAD:04X} 0x00000000",
        f"master_write_32 $m 0x{CTRL_RUN:04X} 0x00000001",
        f"after {cycle_padding_ms}",
        f"master_write_32 $m 0x{CTRL_RUN:04X} 0x00000000",
    ]


def pages(values: list[int], page_size: int) -> Iterable[list[int]]:
    if not 1 <= page_size <= IMEM_WORDS:
        raise JtagPagerError(f"page size must be in 1..{IMEM_WORDS}")
    for start in range(0, len(values), page_size):
        yield values[start:start + page_size]


def build_tcl(mem1_rows: list[int], program: list[int], page_size: int, cycle_padding_ms: int) -> str:
    if not program:
        raise JtagPagerError("the VLIW image is empty")
    lines = [
        "refresh_connections",
        "set masters [get_service_paths master]",
        "if {[llength $masters] == 0} { error \"No JTAG-to-Avalon master found\" }",
        "set m [lindex $masters 0]",
        "open_service master $m",
        "# Load persistent model rows once.",
    ]
    for row, value in enumerate(mem1_rows):
        append_row_writes(lines, MEM1_BASE, row, value)
    for page_index, page in enumerate(pages(program, page_size)):
        lines.append(f"puts \"TINY_LPU_PAGE_BEGIN:{page_index}\"")
        append_page(lines, page, cycle_padding_ms)
        lines.append(f"puts \"TINY_LPU_PAGE_DONE:{page_index}\"")
    lines += ["close_service master $m", "puts \"TINY_LPU_PAGED_RUN_DONE\""]
    return "\n".join(lines) + "\n"


def run_system_console(system_console: Path, tcl: str) -> str:
    if not system_console.is_file():
        raise JtagPagerError(f"System Console was not found: {system_console}")
    with tempfile.TemporaryDirectory(prefix="tinylpu-jtag-") as directory:
        script = Path(directory) / "paged_run.tcl"
        script.write_text(tcl, encoding="ascii")
        completed = subprocess.run(
            [str(system_console), f"--script={script}"], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
    if completed.returncode != 0 or "TINY_LPU_PAGED_RUN_DONE" not in completed.stdout:
        raise JtagPagerError(f"System Console paged run failed:\n{completed.stdout}")
    return completed.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imem", required=True, type=Path, help="96-bit VLIW image, one hexadecimal instruction per line")
    parser.add_argument("--mem1", required=True, type=Path, help="72-bit persistent MEM1 image, one hexadecimal row per line")
    parser.add_argument("--page-size", type=int, default=900, help="instructions per page (default: 900)")
    parser.add_argument("--after-ms", type=int, default=1, help="JTAG wait per page; IMEM tail is cleared to NOPs")
    parser.add_argument("--system-console", type=Path, default=DEFAULT_SYSTEM_CONSOLE)
    parser.add_argument("--write-tcl", type=Path, help="write the generated Tcl without running it")
    args = parser.parse_args()
    if args.after_ms < 1:
        parser.error("--after-ms must be at least one millisecond")
    tcl = build_tcl(read_hex_rows(args.mem1, 72), read_hex_rows(args.imem, 96), args.page_size, args.after_ms)
    if args.write_tcl:
        args.write_tcl.write_text(tcl, encoding="ascii")
        print(f"wrote {args.write_tcl}")
        return 0
    print(run_system_console(args.system_console, tcl), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
