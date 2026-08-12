#!/usr/bin/env python3
"""Compile the shipped INT8 MicroGPT checkpoint into a static TinyLPU schedule.

This compiler does not change the LPU RTL.  It targets the current DE1-SoC
image by emitting one long 96-bit VLIW stream that is meant to be executed by
``synthesis/host/lpu_jtag_pager.py`` in <=1024 instruction pages.

The schedule is deliberately static: all matrix dimensions, scratch addresses,
and page boundaries are fixed from the MicroGPT config.  The host may select
which precompiled stage to run and may move rows over JTAG, but it does not
perform tensor arithmetic.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "model" / "artifacts" / "microgpt_weights_int8.json"
DEFAULT_OUTPUT = ROOT / "model" / "artifacts" / "fpga_microgpt"
IMEM_WORDS = 1024
DEFAULT_PAGE_SIZE = 900
LANES = 8
MEM_ROWS = 16384

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lpu_vliw_compiler import (  # noqa: E402
    EB_MEM0,
    EB_MXM,
    EB_VXM,
    EC_MEM0,
    EC_VXM,
    INGRESS_INPUT,
    INGRESS_WGHT,
    WB_MEM0,
    WB_MEM1,
    WB_SXM,
    WB_VXM,
    WC_MEM0,
    WC_MXM,
    WC_SXM,
    WC_VXM,
    build_instruction,
)


VXM_OPERAND_DATA = 0
VXM_OPERAND_BIAS = 1
VXM_OPERAND_GAMMA = 2
VXM_RES_PASS = 0
VXM_RES_CLEAR = 1
VXM_RES_LOAD = 2
VXM_RES_ADD = 3
VXM_RES_EMIT = 4

# The DE1-SoC build uses Intel altsyncram block RAMs with registered outputs.
# Holding memory reads for two VLIW cycles makes the schedule safe for both the
# behavioral simulation RAM and the real FPGA RAM path: the second read keeps
# mem*_valid asserted while the selected row is stable on the bus.
MEM_READ_HOLD_CYCLES = 2

# Intel block RAM read/write behavior around a freshly-written same address is
# not the same as the simple behavioral array.  The schedule stages constants
# such as RMS gamma through MEM0 and reads them immediately afterward, so leave
# a couple of dead cycles after generated MEM0 writes before dependent reads.
MEM_WRITE_SETTLE_CYCLES = 2


def signed8(value: int) -> int:
    value &= 0xFF
    return value - 0x100 if value & 0x80 else value


def pack_row(lanes: Iterable[int], scale: int) -> int:
    padded = list(lanes)[:LANES]
    padded += [0] * (LANES - len(padded))
    word = 0
    for index, lane in enumerate(padded):
        if not -128 <= int(lane) <= 127:
            raise ValueError(f"lane {lane} does not fit in signed int8")
        word |= (int(lane) & 0xFF) << (8 * index)
    word |= (int(scale) & 0xFF) << 64
    return word


def unpack_row(word: int) -> tuple[list[int], int]:
    lanes = [signed8((word >> (8 * lane)) & 0xFF) for lane in range(LANES)]
    return lanes, signed8((word >> 64) & 0xFF)


def quantize_block(values: list[float]) -> tuple[list[int], int]:
    absmax = max((abs(value) for value in values), default=0.0)
    scale = 0 if absmax == 0.0 else math.ceil(math.log2(absmax / 127.0))
    scale = max(-128, min(127, scale))
    inv = math.ldexp(1.0, -scale)
    lanes = [max(-127, min(127, int(round(value * inv)))) for value in values]
    return lanes, scale


def tensor_to_float_rows(tensor: dict) -> list[list[float]]:
    rows: list[list[float]] = []
    packed_rows = tensor["packed_72bit_rows"]
    width = tensor["shape"][1]
    for packed_blocks in packed_rows:
        row: list[float] = []
        for packed in packed_blocks:
            lanes, scale = unpack_row(int(packed, 16))
            row.extend(math.ldexp(float(lane), scale) for lane in lanes)
        rows.append(row[:width])
    return rows


@dataclass
class Mem1Image:
    rows: list[int] = field(default_factory=list)
    symbols: dict[str, dict[str, int | list[int] | str]] = field(default_factory=dict)

    def add_rows(self, name: str, rows: list[int], shape: list[int], layout: str) -> int:
        base = len(self.rows)
        self.rows.extend(rows)
        self.symbols[name] = {
            "base_row": base,
            "rows": len(rows),
            "shape": shape,
            "layout": layout,
        }
        if len(self.rows) > MEM_ROWS:
            raise RuntimeError(f"MEM1 image needs {len(self.rows)} rows; hardware has {MEM_ROWS}")
        return base


def row_major_rows(tensor: dict) -> list[int]:
    rows: list[int] = []
    for packed_blocks in tensor["packed_72bit_rows"]:
        for packed in packed_blocks:
            rows.append(int(packed, 16))
    return rows


def transposed_matvec_rows(matrix: list[list[float]]) -> list[int]:
    """Pack W[out][in] as rows W[input_index][output_block].

    MXM computes an outer product.  For matvec we broadcast x[k] into all MXM
    input rows and load one row containing eight output weights for that same
    k.  MXM row 0 then contains eight accumulated output lanes.
    """
    out_dim = len(matrix)
    in_dim = len(matrix[0]) if matrix else 0
    rows: list[int] = []
    for in_index in range(in_dim):
        for out_block in range(math.ceil(out_dim / LANES)):
            values = []
            for lane in range(LANES):
                out_index = out_block * LANES + lane
                values.append(matrix[out_index][in_index] if out_index < out_dim else 0.0)
            lanes, scale = quantize_block(values)
            rows.append(pack_row(lanes, scale))
    return rows


def build_mem1_image(checkpoint: dict) -> Mem1Image:
    image = Mem1Image()
    for name, tensor in checkpoint["state_dict"].items():
        image.add_rows(
            name,
            row_major_rows(tensor),
            tensor["shape"],
            "checkpoint_row_major_blocks",
        )

    for name in (
        "lm_head",
        "layer0.attn_wq",
        "layer0.attn_wk",
        "layer0.attn_wv",
        "layer0.attn_wo",
        "layer0.mlp_fc1",
        "layer0.mlp_fc2",
    ):
        matrix = tensor_to_float_rows(checkpoint["state_dict"][name])
        rows = transposed_matvec_rows(matrix)
        image.add_rows(
            f"{name}.matvec_t",
            rows,
            [len(matrix), len(matrix[0]) if matrix else 0],
            "input_major_output_blocks_for_mxm_row0",
        )

    image.add_rows(
        "const.ones",
        [pack_row([1] * LANES, 0)],
        [1, LANES],
        "utility_row",
    )
    image.add_rows(
        "const.rms_gamma_identity_q1_7",
        [pack_row([127] * LANES, 0)],
        [1, LANES],
        "vxm_gamma_operand",
    )
    return image


@dataclass
class ScheduleEmitter:
    instructions: list[int] = field(default_factory=list)
    trace: list[dict[str, int | str]] = field(default_factory=list)

    def emit(self, note: str, **fields: int) -> None:
        self.trace.append({"pc": len(self.instructions), "note": note})
        self.instructions.append(build_instruction(**fields))

    def nop(self, cycles: int = 1, note: str = "nop") -> None:
        for _ in range(cycles):
            self.emit(note)

    def mem0_read_hold(self, addr: int, note: str) -> None:
        for cycle in range(MEM_READ_HOLD_CYCLES):
            suffix = "" if cycle == 0 else f" hold {cycle}"
            self.emit(f"{note}: read MEM0[{addr}]{suffix}", mem0_read_en=1, mem0_addr=addr)

    def mem1_read_hold(self, addr: int, note: str) -> None:
        for cycle in range(MEM_READ_HOLD_CYCLES):
            suffix = "" if cycle == 0 else f" hold {cycle}"
            self.emit(f"{note}: read MEM1[{addr}]{suffix}", mem1_read_en=1, mem1_addr=addr)

    def mem0_to_mxm(self, addr: int, mode: int, note: str) -> None:
        self.mem0_read_hold(addr, note)
        self.emit(
            f"{note}: capture MEM0[{addr}]",
            westbound_sel=WB_MEM0,
            westbound_consumer_sel=WC_MXM,
            mxm_ingress_mode=mode,
        )

    def mem1_to_mxm(self, addr: int, mode: int, note: str) -> None:
        self.mem1_read_hold(addr, note)
        self.emit(
            f"{note}: capture MEM1[{addr}]",
            westbound_sel=WB_MEM1,
            westbound_consumer_sel=WC_MXM,
            mxm_ingress_mode=mode,
        )

    def mem1_to_mem0(self, src: int, dst: int, note: str) -> None:
        self.mem1_read_hold(src, note)
        self.emit(
            f"{note}: copy MEM1[{src}] to MEM0[{dst}]",
            westbound_sel=WB_MEM1,
            westbound_consumer_sel=WC_MEM0,
            mem0_write_en=1,
            mem0_addr=dst,
        )
        self.nop(MEM_WRITE_SETTLE_CYCLES, f"{note}: settle MEM0[{dst}] write")

    def mem0_west_to_vxm(self, addr: int, operand: int, note: str) -> None:
        self.mem0_read_hold(addr, f"{note}: west")
        self.emit(
            f"{note}: feed MEM0[{addr}] to VXM west",
            westbound_sel=WB_MEM0,
            westbound_consumer_sel=WC_VXM,
            vxm_operand_sel=operand,
        )

    def mem0_east_to_vxm(
        self,
        addr: int,
        operand: int,
        note: str,
        residual_op: int = VXM_RES_PASS,
        rmsnorm: bool = False,
        vxm_ctrl: int = 0,
        hold_cycles: int = 0,
    ) -> None:
        self.mem0_read_hold(addr, f"{note}: east")
        self.emit(
            f"{note}: feed MEM0[{addr}] to VXM",
            eastbound_sel=EB_MEM0,
            eastbound_consumer_sel=EC_VXM,
            vxm_operand_sel=operand,
            vxm_residual_op=residual_op,
            vxm_layernorm_en=1 if rmsnorm else 0,
            vxm_ctrl=vxm_ctrl,
        )
        for _ in range(hold_cycles):
            self.emit(
                f"{note}: hold VXM control",
                vxm_operand_sel=operand,
                vxm_residual_op=residual_op,
                vxm_layernorm_en=1 if rmsnorm else 0,
                vxm_ctrl=vxm_ctrl,
            )

    def drain_vxm_to_mem0(
        self,
        dst: int,
        note: str,
        wait_cycles: int = 8,
        write_cycles: int = 1,
    ) -> None:
        self.nop(wait_cycles, f"{note}: wait VXM")
        for write_index in range(write_cycles):
            self.emit(
                f"{note}: write VXM to MEM0[{dst}]"
                + (f" drain {write_index}" if write_cycles > 1 else ""),
                westbound_sel=WB_VXM,
                westbound_consumer_sel=WC_MEM0,
                mem0_write_en=1,
                mem0_addr=dst,
            )

    def mxm_row0_to_vxm_to_mem0(
        self,
        dst: int,
        note: str,
        vxm_ctrl: int = 0,
        rmsnorm: bool = False,
        wait_cycles: int = 16,
    ) -> None:
        self.emit(
            f"{note}: feed MXM row0 to VXM",
            eastbound_sel=EB_MXM,
            eastbound_consumer_sel=EC_VXM,
            mxm_e_row_sel=0,
            mxm_e_valid_in=1,
            vxm_operand_sel=VXM_OPERAND_DATA,
            vxm_ctrl=vxm_ctrl,
            vxm_layernorm_en=1 if rmsnorm else 0,
        )
        self.drain_vxm_to_mem0(dst, note, wait_cycles, write_cycles=8)

    def broadcast_mem0_row(self, src: int, dst_base: int, note: str) -> None:
        for load_index in range(LANES):
            self.mem0_read_hold(src, f"{note}: for SXM load {load_index}")
            self.emit(
                f"{note}: SXM capture copy {load_index}",
                westbound_sel=WB_MEM0,
                westbound_consumer_sel=WC_SXM,
                sxm_transpose_load=1 if load_index == 0 else 0,
                sxm_load_from_west=1,
            )
        self.emit(f"{note}: begin SXM emit", sxm_transpose_emit=1)
        for lane in range(LANES):
            self.emit(
                f"{note}: write broadcast lane {lane} to MEM0[{dst_base + lane}]",
                westbound_sel=WB_SXM,
                westbound_consumer_sel=WC_MEM0,
                mem0_write_en=1,
                mem0_addr=dst_base + lane,
            )

    def add_rows_to_mem0(self, a: int, b: int, dst: int, note: str) -> None:
        self.mem0_east_to_vxm(a, VXM_OPERAND_DATA, f"{note}: load add lhs", VXM_RES_LOAD, hold_cycles=8)
        self.mem0_east_to_vxm(b, VXM_OPERAND_DATA, f"{note}: add rhs", VXM_RES_ADD, hold_cycles=8)
        self.emit(f"{note}: emit residual add", vxm_residual_op=VXM_RES_EMIT)
        self.drain_vxm_to_mem0(dst, note)

    def rmsnorm_row_to_mem0(self, src: int, dst: int, gamma_addr: int, note: str) -> None:
        self.mem1_to_mem0(gamma_addr, Scratch.TMP0, f"{note}: stage gamma")
        self.mem0_west_to_vxm(Scratch.TMP0, VXM_OPERAND_GAMMA, f"{note}: load gamma")
        self.mem0_east_to_vxm(src, VXM_OPERAND_DATA, f"{note}: rms row", rmsnorm=True, hold_cycles=10)
        self.drain_vxm_to_mem0(dst, note, wait_cycles=10)

    def rmsnorm_pair_to_mem0(
        self,
        src0: int,
        src1: int,
        dst0: int,
        dst1: int,
        gamma_addr: int,
        note: str,
    ) -> None:
        self.mem1_to_mem0(gamma_addr, Scratch.TMP0, f"{note}: stage gamma")
        self.mem0_west_to_vxm(Scratch.TMP0, VXM_OPERAND_GAMMA, f"{note}: load gamma")
        self.mem0_east_to_vxm(src0, VXM_OPERAND_DATA, f"{note}: rms chunk 0", rmsnorm=True, hold_cycles=10)
        self.mem0_east_to_vxm(src1, VXM_OPERAND_DATA, f"{note}: rms chunk 1", rmsnorm=True, hold_cycles=10)
        self.drain_vxm_to_mem0(dst0, f"{note}: drain chunk 0", wait_cycles=12)
        self.drain_vxm_to_mem0(dst1, f"{note}: drain chunk 1", wait_cycles=8)

    def relu_row_to_mem0(self, src: int, dst: int, note: str) -> None:
        # The row is first queued inside LPU before VXM accepts it.  Keep the
        # ReLU control asserted across that decoupling window so the data and
        # operation cannot separate while the FIFO drains.
        self.mem0_east_to_vxm(
            src,
            VXM_OPERAND_DATA,
            f"{note}: relu",
            vxm_ctrl=0b0010,
            hold_cycles=8,
        )
        # ReLU traverses the complete VXM pipeline, residual pass-through, and
        # output requantizer.  Eight cycles can select WB_VXM before the result
        # reaches the output FIFO, leaving the old FC1 row in place and causing
        # the delayed result to be written into the following row.  Use the same
        # conservative retirement window as MXM-to-VXM requantization.
        self.drain_vxm_to_mem0(dst, note, wait_cycles=16, write_cycles=8)

    def matvec_to_mem0(
        self,
        input_bcast_base: int,
        weight_base: int,
        in_dim: int,
        out_dim: int,
        output_base: int,
        note: str,
    ) -> None:
        output_blocks = math.ceil(out_dim / LANES)
        for out_block in range(output_blocks):
            self.emit(f"{note}: clear MXM output block {out_block}", mxm_clear=1)
            for in_index in range(in_dim):
                weight_row = weight_base + in_index * output_blocks + out_block
                self.mem0_to_mxm(
                    input_bcast_base + in_index,
                    INGRESS_INPUT,
                    f"{note}: x[{in_index}] -> MXM",
                )
                self.mem1_to_mxm(
                    weight_row,
                    INGRESS_WGHT,
                    f"{note}: W[:,{in_index}] block {out_block} -> MXM",
                )
                self.emit(f"{note}: MAC {in_index}/{in_dim} block {out_block}", mxm_start=1)
            self.nop(2, f"{note}: settle MXM block {out_block}")
            self.mxm_row0_to_vxm_to_mem0(output_base + out_block, f"{note}: quant block {out_block}")


class Scratch:
    TOKEN0 = 0
    TOKEN1 = 1
    POS0 = 2
    POS1 = 3
    X0 = 8
    X1 = 9
    XN0 = 10
    XN1 = 11
    X_BCAST = 32
    Q = 80
    K = 84
    V = 88
    ATTN = 96
    ATTN_BCAST = 112
    ATTN_PROJ = 160
    MLP_IN0 = 176
    MLP_IN1 = 177
    MLP_BCAST = 192
    MLP_H = 256
    MLP_H_BCAST = 320
    MLP_OUT = 400
    LOGITS = 448
    TMP0 = 512


def compile_decode_stage(symbols: dict[str, dict[str, int | list[int] | str]]) -> ScheduleEmitter:
    e = ScheduleEmitter()
    gamma = int(symbols["const.rms_gamma_identity_q1_7"]["base_row"])

    e.add_rows_to_mem0(Scratch.TOKEN0, Scratch.POS0, Scratch.X0, "embedding add row0")
    e.add_rows_to_mem0(Scratch.TOKEN1, Scratch.POS1, Scratch.X1, "embedding add row1")
    # Golden MicroGPT first normalizes token+position embeddings, then the
    # transformer block uses that normalized vector as its residual stream.
    # Keep that value in X0/X1 so later residual adds match the CPU model.
    e.rmsnorm_pair_to_mem0(Scratch.X0, Scratch.X1, Scratch.X0, Scratch.X1, gamma, "embedding rms")
    # The attention block applies its own pre-attention RMSNorm to the residual
    # stream before Q/K/V projection.  This is nearly idempotent numerically but
    # keeping the explicit op makes the static schedule match the golden graph.
    e.rmsnorm_pair_to_mem0(Scratch.X0, Scratch.X1, Scratch.XN0, Scratch.XN1, gamma, "attention rms")
    e.broadcast_mem0_row(Scratch.XN0, Scratch.X_BCAST, "broadcast attention input row0")
    e.broadcast_mem0_row(Scratch.XN1, Scratch.X_BCAST + LANES, "broadcast attention input row1")

    for name, dst in (
        ("layer0.attn_wq.matvec_t", Scratch.Q),
        ("layer0.attn_wk.matvec_t", Scratch.K),
        ("layer0.attn_wv.matvec_t", Scratch.V),
    ):
        e.matvec_to_mem0(
            Scratch.X_BCAST,
            int(symbols[name]["base_row"]),
            16,
            16,
            dst,
            name.removesuffix(".matvec_t"),
        )

    # Current RTL has no masked lane insert, so the JTAG side must maintain the
    # transposed KV cache rows for attention.  The static schedule consumes the
    # staged attention result at Scratch.ATTN and continues the remaining model.
    e.broadcast_mem0_row(Scratch.ATTN, Scratch.ATTN_BCAST, "broadcast staged attention row0")
    e.broadcast_mem0_row(Scratch.ATTN + 1, Scratch.ATTN_BCAST + LANES, "broadcast staged attention row1")
    e.matvec_to_mem0(
        Scratch.ATTN_BCAST,
        int(symbols["layer0.attn_wo.matvec_t"]["base_row"]),
        16,
        16,
        Scratch.ATTN_PROJ,
        "attn output projection",
    )
    e.add_rows_to_mem0(Scratch.ATTN_PROJ, Scratch.X0, Scratch.MLP_IN0, "attention residual row0")
    e.add_rows_to_mem0(Scratch.ATTN_PROJ + 1, Scratch.X1, Scratch.MLP_IN1, "attention residual row1")
    e.rmsnorm_pair_to_mem0(Scratch.MLP_IN0, Scratch.MLP_IN1, Scratch.XN0, Scratch.XN1, gamma, "mlp rms")
    e.broadcast_mem0_row(Scratch.XN0, Scratch.MLP_BCAST, "broadcast mlp row0")
    e.broadcast_mem0_row(Scratch.XN1, Scratch.MLP_BCAST + LANES, "broadcast mlp row1")
    e.matvec_to_mem0(
        Scratch.MLP_BCAST,
        int(symbols["layer0.mlp_fc1.matvec_t"]["base_row"]),
        16,
        64,
        Scratch.MLP_H,
        "mlp fc1",
    )
    for block in range(8):
        e.relu_row_to_mem0(Scratch.MLP_H + block, Scratch.MLP_H + block, f"mlp relu block {block}")
        e.broadcast_mem0_row(Scratch.MLP_H + block, Scratch.MLP_H_BCAST + block * LANES, f"broadcast mlp hidden {block}")
    e.matvec_to_mem0(
        Scratch.MLP_H_BCAST,
        int(symbols["layer0.mlp_fc2.matvec_t"]["base_row"]),
        64,
        16,
        Scratch.MLP_OUT,
        "mlp fc2",
    )
    e.add_rows_to_mem0(Scratch.MLP_OUT, Scratch.MLP_IN0, Scratch.X0, "mlp residual row0")
    e.add_rows_to_mem0(Scratch.MLP_OUT + 1, Scratch.MLP_IN1, Scratch.X1, "mlp residual row1")
    e.broadcast_mem0_row(Scratch.X0, Scratch.X_BCAST, "broadcast final row0")
    e.broadcast_mem0_row(Scratch.X1, Scratch.X_BCAST + LANES, "broadcast final row1")
    e.matvec_to_mem0(
        Scratch.X_BCAST,
        int(symbols["lm_head.matvec_t"]["base_row"]),
        16,
        27,
        Scratch.LOGITS,
        "lm head",
    )
    return e


def split_pages(program: list[int], page_size: int) -> list[dict[str, int]]:
    pages: list[dict[str, int]] = []
    for start in range(0, len(program), page_size):
        pages.append({"index": len(pages), "start_pc": start, "instructions": min(page_size, len(program) - start)})
    return pages


def write_hex(path: Path, rows: Iterable[int], width: int) -> None:
    digits = (width + 3) // 4
    path.write_text("".join(f"{row:0{digits}X}\n" for row in rows), encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    args = parser.parse_args()

    if not 1 <= args.page_size <= IMEM_WORDS:
        parser.error(f"--page-size must be in 1..{IMEM_WORDS}")

    checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    if checkpoint.get("format") != "tinylpu.microgpt.lpu_int8":
        parser.error(f"not a TinyLPU MicroGPT INT8 checkpoint: {args.checkpoint}")
    config = checkpoint["config"]
    expected = {"n_layer": 1, "n_embd": 16, "block_size": 16, "n_head": 4, "vocab_size": 27}
    if config != expected:
        parser.error(f"this hardcoded scheduler expects {expected}; checkpoint has {config}")

    mem1 = build_mem1_image(checkpoint)
    schedule = compile_decode_stage(mem1.symbols)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    mem1_path = output / "microgpt_scheduler_mem1.hex"
    imem_path = output / "microgpt_decode_vliw.hex"
    manifest_path = output / "microgpt_decode_schedule.json"
    trace_path = output / "microgpt_decode_trace.json"

    write_hex(mem1_path, mem1.rows, 72)
    write_hex(imem_path, schedule.instructions, 96)
    trace_path.write_text(json.dumps(schedule.trace, indent=2) + "\n", encoding="utf-8")

    limitations = [
        "RMSNorm is compiled as two 8-lane chunks for MicroGPT hidden size 16",
        "current RTL has no masked lane insert for dynamic K/V cache updates; attention rows must be staged over JTAG from FPGA-produced K/V rows",
        "the emitted schedule performs no host tensor arithmetic, but host-side JTAG data movement is part of the runtime ABI",
    ]
    manifest = {
        "format": "tinylpu.microgpt.static-paged-schedule",
        "format_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "config": config,
        "tokenizer": checkpoint["tokenizer"],
        "imem": {
            "path": str(imem_path.resolve()),
            "instructions": len(schedule.instructions),
            "page_size": args.page_size,
            "pages": split_pages(schedule.instructions, args.page_size),
        },
        "mem1": {
            "path": str(mem1_path.resolve()),
            "rows": len(mem1.rows),
            "symbols": mem1.symbols,
        },
        "mem0_abi": {
            "token_embedding_rows": [Scratch.TOKEN0, Scratch.TOKEN1],
            "position_embedding_rows": [Scratch.POS0, Scratch.POS1],
            "staged_attention_rows": [Scratch.ATTN, Scratch.ATTN + 1],
            "q_rows": [Scratch.Q, Scratch.Q + 1],
            "k_rows": [Scratch.K, Scratch.K + 1],
            "v_rows": [Scratch.V, Scratch.V + 1],
            "logit_rows": [Scratch.LOGITS, Scratch.LOGITS + 1, Scratch.LOGITS + 2, Scratch.LOGITS + 3],
        },
        "limitations": limitations,
        "run_hint": (
            "python synthesis/host/lpu_jtag_pager.py "
            f"--mem1 {mem1_path} --imem {imem_path} --page-size {args.page_size}"
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote MEM1 image: {mem1_path} ({len(mem1.rows)} rows)")
    print(f"wrote VLIW image: {imem_path} ({len(schedule.instructions)} instructions)")
    print(f"wrote manifest: {manifest_path}")
    print(f"wrote trace: {trace_path}")
    if len(schedule.instructions) > IMEM_WORDS:
        print(f"paged schedule: {len(split_pages(schedule.instructions, args.page_size))} pages of <= {args.page_size} instructions")
    for item in limitations:
        print(f"warning: {item}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
