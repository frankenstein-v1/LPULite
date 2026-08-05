#!/usr/bin/env python3
"""Export stories10k weights as packed 8-lane FP8 memory rows."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("model/datasets/stories10k/stories10k_weights_export.json")
DEFAULT_OUTPUT = Path("model/datasets/stories10k/stories10k_weights_fp8_packed.json")
LANES = 8
ROW_BITS = 72
LANE_FORMAT = "fp8_e5m2"


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def bits_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFF_FFFF))[0]


def to_f32(value: float) -> float:
    return bits_to_f32(f32_bits(value))


def fp8_e5m2_bits(value: float) -> int:
    bits = f32_bits(value)
    sign = (bits >> 31) & 0x1
    exp = (bits >> 23) & 0xFF
    frac = bits & 0x7FFFFF

    if exp == 0 and frac == 0:
        return sign << 7
    if exp == 0xFF:
        return ((sign << 7) | 0x7D) if frac else ((sign << 7) | 0x7C)
    if exp == 0:
        return sign << 7

    fp8_exp = exp - 127 + 15
    if fp8_exp <= 0:
        return sign << 7

    mantissa_full = (1 << 23) | frac
    mantissa_q = (mantissa_full >> 21) & 0x7
    guard = (mantissa_full >> 20) & 0x1
    sticky = mantissa_full & ((1 << 20) - 1)
    if guard and (sticky or (mantissa_q & 0x1)):
        mantissa_q += 1
    if mantissa_q == 8:
        mantissa_q = 4
        fp8_exp += 1
    if fp8_exp >= 31:
        return (sign << 7) | 0x7C
    return (sign << 7) | ((fp8_exp & 0x1F) << 2) | (mantissa_q & 0x3)


def shape_of(tensor: Any) -> list[int]:
    shape: list[int] = []
    node = tensor
    while isinstance(node, list):
        shape.append(len(node))
        node = node[0] if node else None
    return shape


def pack_row_8(values: list[float]) -> tuple[str, list[int], int]:
    row = list(values[:LANES])
    if len(row) < LANES:
        row.extend([0.0] * (LANES - len(row)))

    absmax = max((abs(float(value)) for value in row), default=0.0)
    scale_exp = 0 if absmax == 0.0 else math.floor(math.log2(absmax))
    scaled = [to_f32(math.ldexp(float(value), -scale_exp)) for value in row]
    row_bits = [fp8_e5m2_bits(value) for value in scaled]

    word = 0
    for idx, bits in enumerate(row_bits):
        word |= (bits & 0xFF) << (8 * idx)
    word |= (scale_exp & 0xFF) << 64

    return f"0x{word:018x}", row_bits, scale_exp


def pack_vector(vector: list[float]) -> dict[str, Any]:
    words: list[str] = []
    lane_bits: list[list[int]] = []
    scale_exps: list[int] = []
    for start in range(0, len(vector), LANES):
        word, bits, scale_exp = pack_row_8(vector[start : start + LANES])
        words.append(word)
        lane_bits.append(bits)
        scale_exps.append(scale_exp)
    return {
        "shape": [len(vector)],
        "words_per_row": 1,
        "rows": words,
        "flat_words": words,
        "lane_bits": lane_bits,
        "scale_exps": scale_exps,
    }


def pack_matrix(matrix: list[list[float]]) -> dict[str, Any]:
    rows: list[list[str]] = []
    lane_bits: list[list[list[int]]] = []
    scale_exps: list[list[int]] = []
    flat_words: list[str] = []

    width = max((len(row) for row in matrix), default=0)
    words_per_row = (width + LANES - 1) // LANES

    for row in matrix:
        row_words: list[str] = []
        row_bits_list: list[list[int]] = []
        row_scales: list[int] = []
        for start in range(0, width, LANES):
            word, bits, scale_exp = pack_row_8(row[start : start + LANES])
            row_words.append(word)
            row_bits_list.append(bits)
            row_scales.append(scale_exp)
            flat_words.append(word)
        rows.append(row_words)
        lane_bits.append(row_bits_list)
        scale_exps.append(row_scales)

    return {
        "shape": [len(matrix), width],
        "words_per_row": words_per_row,
        "rows": rows,
        "flat_words": flat_words,
        "lane_bits": lane_bits,
        "scale_exps": scale_exps,
    }


def pack_tensor(tensor: Any) -> dict[str, Any]:
    shape = shape_of(tensor)
    if len(shape) == 1:
        return pack_vector(tensor)
    if len(shape) == 2:
        return pack_matrix(tensor)
    raise ValueError(f"only 1D/2D tensors are supported, got shape {shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack stories10k FP32 JSON weights into 8-lane FP8 72-bit rows."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open("r", encoding="utf-8") as f:
        export = json.load(f)

    packed_weights = {}
    total_words = 0
    for name, tensor in export["weights"].items():
        packed = pack_tensor(tensor)
        packed_weights[name] = packed
        total_words += len(packed["flat_words"])

    packed_export = {
        "config": export["config"],
        "vocab": export["vocab"],
        "format": {
            "row_bits": ROW_BITS,
            "lanes": LANES,
            "lane_format": LANE_FORMAT,
            "lane_order": "lane0 in bits [7:0], lane7 in bits [63:56]",
            "scale_bits": 8,
            "scale_location": "bits [71:64]",
            "scale_encoding": "signed int8 power-of-two exponent",
            "value": "dequantized_value = fp8_e5m2(lane) * 2**scale_exp",
            "padding": "last chunk of each vector or matrix row is zero padded to 8 lanes",
        },
        "summary": {
            "source": str(args.input),
            "total_72b_words": total_words,
            "total_payload_bits": total_words * ROW_BITS,
        },
        "weights": packed_weights,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(packed_export, f, indent=2)
        f.write("\n")

    print(f"Exported {total_words} packed FP8 rows to {args.output}")


if __name__ == "__main__":
    main()
