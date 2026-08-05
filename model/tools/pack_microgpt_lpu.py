#!/usr/bin/env python3
"""Pack the shipped MicroGPT INT8 checkpoint into TinyLPU 72-bit MEM1 rows.

This tool performs no inference.  It preserves every deployed INT8 lane and
its shared power-of-two scale from ``microgpt_weights_int8.json`` and writes a
simple line-oriented image suitable for the JTAG pager.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "model" / "artifacts" / "microgpt_weights_int8.json"
DEFAULT_OUTPUT = ROOT / "model" / "artifacts" / "fpga_microgpt"
MEM1_ROWS = 16384


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    if checkpoint.get("format") != "tinylpu.microgpt.lpu_int8":
        parser.error(f"not a TinyLPU MicroGPT INT8 checkpoint: {args.checkpoint}")
    numeric = checkpoint.get("numeric_contract", {})
    if numeric.get("packed_row_bits") != 72 or numeric.get("lanes_per_block") != 8:
        parser.error("checkpoint does not use the required 8xINT8/72-bit LPU row format")

    rows: list[int] = []
    symbols: dict[str, dict[str, int]] = {}
    for tensor_name, tensor in checkpoint["state_dict"].items():
        base = len(rows)
        for tensor_row, packed_blocks in enumerate(tensor["packed_72bit_rows"]):
            for block, packed in enumerate(packed_blocks):
                rows.append(int(packed, 16))
        symbols[tensor_name] = {
            "base_row": base,
            "rows": len(rows) - base,
            "shape": tensor["shape"],
            "row_format": "packed_72bit_rows[row][block]",
        }

    if len(rows) > MEM1_ROWS:
        parser.error(f"model needs {len(rows)} MEM1 rows; hardware provides {MEM1_ROWS}")

    args.output.mkdir(parents=True, exist_ok=True)
    image_path = args.output / "microgpt_mem1.hex"
    image_path.write_text("\n".join(f"{row:018X}" for row in rows) + "\n", encoding="ascii")
    manifest = {
        "format": "tinylpu.microgpt.mem1-image",
        "format_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "config": checkpoint["config"],
        "tokenizer": checkpoint["tokenizer"],
        "mem1_rows": len(rows),
        "symbols": symbols,
    }
    manifest_path = args.output / "microgpt_mem1_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} 72-bit rows: {image_path}")
    print(f"wrote layout manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
