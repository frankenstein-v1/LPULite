"""Board-style MicroGPT test for lpu_de1_soc_wrapper.

The DUT is driven through the same lightweight-HPS Avalon-MM register/memory map
as synthesis/linux/src/microgpt_hps_runtime.c.  This catches schedule, wrapper,
memory-latency, fixed-point, and datapath bugs before a Quartus rebuild.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "model" / "artifacts" / "fpga_microgpt"
CHECKPOINT_PATH = ROOT / "model" / "artifacts" / "microgpt_weights_int8.json"
SCHEDULE_PATH = ARTIFACT_DIR / "microgpt_decode_schedule.json"
MEM1_PATH = ARTIFACT_DIR / "microgpt_scheduler_mem1.hex"
VLIW_PATH = ARTIFACT_DIR / "microgpt_decode_vliw.hex"
SOFTMAX_VLIW_PATH = ARTIFACT_DIR / "microgpt_softmax_vliw.hex"
ATTENTION_VLIW_PATH = ARTIFACT_DIR / "microgpt_attention_vliw.hex"

MEM0_BASE = 0x4000
MEM1_BASE = 0x8000
IMEM_BASE = 0x0000
CTRL_RUN = 0xC000
CTRL_PC_LOAD = 0xC004
CTRL_CYCLES = 0xC008
CTRL_RUN_CYCLES = 0xC00C
CTRL_SOFT_RESET = 0xC010

IMEM_ROWS = 1024
LANES = 8
N_EMBD = 16
ROWS_PER_VEC = N_EMBD // LANES


def signed(value: int, width: int) -> int:
    value &= (1 << width) - 1
    return value - (1 << width) if value & (1 << (width - 1)) else value


def read_hex(path: Path) -> list[int]:
    return [int(line.strip(), 16) for line in path.read_text(encoding="ascii").splitlines() if line.strip()]


def words_from_row(value: int) -> tuple[int, int, int]:
    return value & 0xFFFFFFFF, (value >> 32) & 0xFFFFFFFF, (value >> 64) & 0xFFFFFFFF


def row_from_words(words: tuple[int, int, int]) -> int:
    return (words[0] & 0xFFFFFFFF) | ((words[1] & 0xFFFFFFFF) << 32) | ((words[2] & 0xFFFFFFFF) << 64)


def pack_quant_row(lanes: list[int], scale: int) -> int:
    padded = lanes[:LANES] + [0] * max(0, LANES - len(lanes))
    word = 0
    for lane, value in enumerate(padded):
        word |= (int(value) & 0xFF) << (8 * lane)
    word |= (int(scale) & 0xFF) << 64
    return word


def unpack_row(row: int) -> tuple[list[int], int]:
    lanes = [signed(row >> (8 * lane), 8) for lane in range(LANES)]
    scale = signed(row >> 64, 8)
    return lanes, scale


def row_to_vec(rows: list[int]) -> list[float]:
    out: list[float] = []
    for row in rows:
        lanes, scale = unpack_row(row)
        out.extend(math.ldexp(float(lane), scale) for lane in lanes)
    return out[:N_EMBD]


def row_summary(row: int) -> str:
    lanes, scale = unpack_row(row)
    return f"lanes={lanes} scale={scale}"


def vec_summary(vec: list[float]) -> str:
    if not vec:
        return "empty"
    nonzero = sum(1 for value in vec if value != 0.0)
    return (
        f"nz={nonzero}/{len(vec)} "
        f"min={min(vec):.6g} max={max(vec):.6g} "
        f"first4={[round(value, 6) for value in vec[:4]]}"
    )


def state_probes_enabled() -> bool:
    return os.getenv("MICROGPT_STATE_PROBES", "0") == "1"


def state_probes_for_position(position: int) -> bool:
    if not state_probes_enabled():
        return False
    selected = os.getenv("MICROGPT_TRACE_POS")
    return selected is None or int(selected) == position


def suffix_probes_for_position(position: int) -> bool:
    if os.getenv("MICROGPT_SUFFIX_PROBES", "0") != "1":
        return False
    selected = os.getenv("MICROGPT_TRACE_POS")
    return selected is None or int(selected) == position


def prefix_phase_probes_enabled() -> bool:
    return os.getenv("MICROGPT_PREFIX_PHASE_PROBES", "0") == "1"


def pack_float_row(values: list[float]) -> int:
    vals = values[:LANES] + [0.0] * max(0, LANES - len(values))
    absmax = max((abs(v) for v in vals), default=0.0)
    scale = 0 if absmax == 0.0 else math.ceil(math.log2(absmax / 127.0))
    scale = max(-128, min(127, scale))
    inv = math.ldexp(1.0, -scale)
    word = 0
    for lane, value in enumerate(vals):
        q = int(round(value * inv))
        q = max(-127, min(127, q))
        word |= (q & 0xFF) << (8 * lane)
    word |= (scale & 0xFF) << 64
    return word


def pack_float_row_at_scale(values: list[float], scale: int) -> int:
    vals = values[:LANES] + [0.0] * max(0, LANES - len(values))
    inv = math.ldexp(1.0, -scale)
    lanes = [max(-127, min(127, int(round(value * inv)))) for value in vals]
    return pack_quant_row(lanes, scale)


def vec_to_rows(vec: list[float]) -> list[int]:
    return [pack_float_row(vec[start : start + LANES]) for start in range(0, N_EMBD, LANES)]


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exps = [math.exp(v - maximum) for v in values]
    total = sum(exps)
    return [v / total for v in exps]


class AvalonDriver:
    def __init__(self, dut):
        self.dut = dut

    async def init(self) -> None:
        self.dut.avs_address.value = 0
        self.dut.avs_read.value = 0
        self.dut.avs_write.value = 0
        self.dut.avs_writedata.value = 0
        self.dut.rst_n.value = 0
        for _ in range(5):
            await RisingEdge(self.dut.clk)
        self.dut.rst_n.value = 1
        for _ in range(3):
            await RisingEdge(self.dut.clk)

    async def _wait_ready(self) -> None:
        while int(self.dut.avs_waitrequest.value):
            await RisingEdge(self.dut.clk)

    async def write32(self, addr: int, value: int) -> None:
        await self._wait_ready()
        self.dut.avs_address.value = addr & 0xFFFF
        self.dut.avs_writedata.value = value & 0xFFFFFFFF
        self.dut.avs_write.value = 1
        await RisingEdge(self.dut.clk)
        self.dut.avs_write.value = 0
        self.dut.avs_address.value = 0
        self.dut.avs_writedata.value = 0

    async def read32(self, addr: int) -> int:
        await self._wait_ready()
        self.dut.avs_address.value = addr & 0xFFFF
        self.dut.avs_read.value = 1
        await RisingEdge(self.dut.clk)
        self.dut.avs_read.value = 0
        self.dut.avs_address.value = 0
        for _ in range(20):
            await RisingEdge(self.dut.clk)
            if int(self.dut.avs_readdatavalid.value):
                await Timer(1, unit="ps")
                try:
                    return int(self.dut.avs_readdata.value) & 0xFFFFFFFF
                except ValueError:
                    raw = str(self.dut.avs_readdata.value)
                    resolved = raw.lower().replace("x", "0").replace("z", "0").replace("u", "0")
                    self.dut._log.warning("read32 addr=0x%04x had unresolved bits: %s", addr, raw)
                    return int(resolved, 2) & 0xFFFFFFFF
        raise TimeoutError(f"read32 timed out at 0x{addr:04x}")

    async def write_row(self, base: int, row: int, value: int) -> None:
        addr = base + row * 12
        for lane, word in enumerate(words_from_row(value)):
            await self.write32(addr + lane * 4, word)

    async def read_row(self, base: int, row: int) -> int:
        addr = base + row * 12
        words = tuple([await self.read32(addr + lane * 4) for lane in range(3)])
        return row_from_words(words)  # type: ignore[arg-type]

    async def run_cycles(self, cycles: int, pc: int = 0) -> int:
        before = await self.read32(CTRL_CYCLES)
        await self.write32(CTRL_PC_LOAD, pc)
        await self.write32(CTRL_RUN_CYCLES, cycles)
        for _ in range(cycles + 50):
            remaining = await self.read32(CTRL_RUN_CYCLES)
            if remaining == 0:
                after = await self.read32(CTRL_CYCLES)
                return (after - before) & 0xFFFFFFFF
            await RisingEdge(self.dut.clk)
        raise TimeoutError("run_cycles timed out")

    async def soft_reset(self, cycles: int = 32) -> None:
        await self.write32(CTRL_RUN, 0)
        await self.write32(CTRL_RUN_CYCLES, 0)
        await self.write32(CTRL_PC_LOAD, 0)
        await self.write32(CTRL_SOFT_RESET, cycles)
        for _ in range(cycles + 50):
            if await self.read32(CTRL_SOFT_RESET) == 0:
                return
            await RisingEdge(self.dut.clk)
        raise TimeoutError("soft_reset timed out")


async def load_mem1(driver: AvalonDriver, mem1: list[int]) -> None:
    for row, value in enumerate(mem1):
        await driver.write_row(MEM1_BASE, row, value)


async def clear_runtime_state(driver: AvalonDriver, mem0_abi: dict) -> None:
    zero = 0
    for row in range(int(mem0_abi["logit_rows"][-1]) + 1):
        await driver.write_row(MEM0_BASE, row, zero)
    for row in range(16 * ROWS_PER_VEC):
        await driver.write_row(MEM0_BASE, 1024 + row, zero)
        await driver.write_row(MEM1_BASE, 1024 + row, zero)


async def reset_prompt_state(driver: AvalonDriver, mem0_abi: dict) -> None:
    await driver.soft_reset()
    await clear_runtime_state(driver, mem0_abi)
    await driver.soft_reset()


async def load_imem_page(driver: AvalonDriver, vliw: list[int], start_pc: int, count: int) -> None:
    for row in range(IMEM_ROWS):
        value = vliw[start_pc + row] if row < count else 0
        await driver.write_row(IMEM_BASE, row, value)


async def run_program(driver: AvalonDriver, vliw: list[int], start_pc: int, count: int, page_size: int) -> None:
    done = 0
    while done < count:
        page_count = min(page_size, count - done)
        await load_imem_page(driver, vliw, start_pc + done, page_count)
        actual = await driver.run_cycles(page_count + 8)
        assert actual == page_count + 8, f"cycle mismatch: actual={actual} requested={page_count + 8}"
        done += page_count


async def write_step_inputs(driver: AvalonDriver, mem1: list[int], symbols: dict, token_id: int, pos_id: int) -> None:
    wte = int(symbols["wte"]["base_row"]) + token_id * ROWS_PER_VEC
    wpe = int(symbols["wpe"]["base_row"]) + pos_id * ROWS_PER_VEC
    await driver.write_row(MEM0_BASE, 0, mem1[wte + 0])
    await driver.write_row(MEM0_BASE, 1, mem1[wte + 1])
    await driver.write_row(MEM0_BASE, 2, mem1[wpe + 0])
    await driver.write_row(MEM0_BASE, 3, mem1[wpe + 1])


async def copy_row(driver: AvalonDriver, src_base: int, src_row: int, dst_base: int, dst_row: int) -> None:
    await driver.write_row(dst_base, dst_row, await driver.read_row(src_base, src_row))


async def stage_broadcast_row(driver: AvalonDriver, src_row: int, dst_base: int) -> None:
    lanes, scale = unpack_row(await driver.read_row(MEM0_BASE, src_row))
    for lane, value in enumerate(lanes):
        await driver.write_row(MEM0_BASE, dst_base + lane, pack_quant_row([value] * LANES, scale))


async def cache_current_kv(driver: AvalonDriver, mem0_abi: dict, pos_id: int) -> None:
    k_base = 1024 + pos_id * ROWS_PER_VEC
    v_base = 1024 + pos_id * ROWS_PER_VEC
    await copy_row(driver, MEM0_BASE, int(mem0_abi["k_rows"][0]), MEM0_BASE, k_base + 0)
    await copy_row(driver, MEM0_BASE, int(mem0_abi["k_rows"][1]), MEM0_BASE, k_base + 1)
    await copy_row(driver, MEM0_BASE, int(mem0_abi["v_rows"][0]), MEM1_BASE, v_base + 0)
    await copy_row(driver, MEM0_BASE, int(mem0_abi["v_rows"][1]), MEM1_BASE, v_base + 1)


async def stage_host_attention(
    driver: AvalonDriver,
    mem0_abi: dict,
    through_pos: int,
    n_head: int,
    softmax_vliw: list[int] | None = None,
    page_size: int = 900,
) -> None:
    q_rows = [
        await driver.read_row(MEM0_BASE, int(mem0_abi["q_rows"][0])),
        await driver.read_row(MEM0_BASE, int(mem0_abi["q_rows"][1])),
    ]
    q = row_to_vec(q_rows)
    head_dim = N_EMBD // n_head
    context = [0.0] * N_EMBD
    for head in range(n_head):
        base = head * head_dim
        scores = []
        values = []
        for pos in range(through_pos + 1):
            k_rows = [
                await driver.read_row(MEM0_BASE, 1024 + pos * ROWS_PER_VEC + 0),
                await driver.read_row(MEM0_BASE, 1024 + pos * ROWS_PER_VEC + 1),
            ]
            v_rows = [
                await driver.read_row(MEM1_BASE, 1024 + pos * ROWS_PER_VEC + 0),
                await driver.read_row(MEM1_BASE, 1024 + pos * ROWS_PER_VEC + 1),
            ]
            key = row_to_vec(k_rows)
            val = row_to_vec(v_rows)
            scores.append(sum(q[base + i] * key[base + i] for i in range(head_dim)) / math.sqrt(head_dim))
            values.append(val)
        if softmax_vliw is None:
            weights = softmax(scores)
        else:
            for pos in range(16):
                if pos <= through_pos:
                    score_row = pack_float_row([scores[pos]] * LANES)
                else:
                    score_row = pack_quant_row([-127] * LANES, 0)
                await driver.write_row(MEM0_BASE, 600 + pos, score_row)
            await run_program(driver, softmax_vliw, 0, len(softmax_vliw), page_size)
            weights = []
            for pos in range(through_pos + 1):
                lanes, scale = unpack_row(await driver.read_row(MEM0_BASE, 640 + pos))
                weights.append(sum(math.ldexp(float(lane & 0xFF), scale) for lane in lanes))
        for i in range(head_dim):
            context[base + i] = sum(weights[pos] * values[pos][base + i] for pos in range(through_pos + 1))

    rows = vec_to_rows(context)
    await driver.write_row(MEM0_BASE, int(mem0_abi["staged_attention_rows"][0]), rows[0])
    await driver.write_row(MEM0_BASE, int(mem0_abi["staged_attention_rows"][1]), rows[1])


async def stage_mxm_attention(
    driver: AvalonDriver,
    mem0_abi: dict,
    through_pos: int,
    n_head: int,
    attention_vliw: list[int],
    attention_meta: dict,
) -> None:
    """Stage cache layouts while FPGA MXM performs QK and PV arithmetic."""
    assert N_EMBD == 16 and n_head == 4
    assert 0 <= through_pos < 16
    head_dim = N_EMBD // n_head
    sections = attention_meta["sections"]
    q_bcast_base = int(mem0_abi["attention_q_broadcast_rows"][0])
    qk_score_base = int(mem0_abi["attention_qk_score_rows"][0])
    pv_prob_base = int(mem0_abi["attention_pv_probability_rows"][0])
    pv_out_row = int(mem0_abi["attention_pv_output_row"])
    head_out_base = int(mem0_abi["attention_head_output_rows"][0])
    k_tile_in_base = int(mem0_abi["attention_k_tile_input_rows"][0])
    kt_stage_base = int(attention_meta["mem1_k_transpose_stage_rows"][0])
    v_stage_base = int(attention_meta["mem1_v_stage_rows"][0])

    # This image remains resident while its QK, softmax, PV, and merge entry
    # points are called repeatedly below.
    await load_imem_page(driver, attention_vliw, 0, len(attention_vliw))
    await driver.write32(CTRL_PC_LOAD, len(attention_vliw) - 1)
    for _ in range(3):
        await RisingEdge(driver.dut.clk)

    async def run_section(name: str) -> None:
        section = sections[name]
        requested = int(section["instructions"])
        actual = await driver.run_cycles(requested, int(section["start"]))
        assert actual == requested, (
            f"attention {name} cycle mismatch: actual={actual} requested={requested}"
        )

    q_rows = [
        await driver.read_row(MEM0_BASE, int(mem0_abi["q_rows"][0])),
        await driver.read_row(MEM0_BASE, int(mem0_abi["q_rows"][1])),
    ]
    q_quant = [unpack_row(row) for row in q_rows]
    key_vectors: list[list[float]] = []
    value_rows: list[list[int]] = []
    for pos in range(through_pos + 1):
        key_vectors.append(row_to_vec([
            await driver.read_row(MEM0_BASE, 1024 + pos * ROWS_PER_VEC + 0),
            await driver.read_row(MEM0_BASE, 1024 + pos * ROWS_PER_VEC + 1),
        ]))
        value_rows.append([
            await driver.read_row(MEM1_BASE, 1024 + pos * ROWS_PER_VEC + 0),
            await driver.read_row(MEM1_BASE, 1024 + pos * ROWS_PER_VEC + 1),
        ])

    masked = pack_quant_row([-127] * LANES, 0)
    zero_row = pack_quant_row([0] * LANES, 0)
    for head in range(n_head):
        row_index = head // 2
        lane_base = (head % 2) * head_dim

        # Q is already quantized. Replication is layout staging only; all four
        # Q*K multiply-accumulates for every token execute in MXM.
        q_lanes, q_scale = q_quant[row_index]
        for dim in range(head_dim):
            await driver.write_row(
                MEM0_BASE,
                q_bcast_base + dim,
                pack_quant_row([q_lanes[lane_base + dim]] * LANES, q_scale),
            )

        # SXM transposes each 8-position K tile. SXM carries one exponent per
        # tile, so ARM aligns the heterogeneous cached-row exponents first but
        # does not move position values into dimension rows itself.
        for block in range(2):
            if block * LANES > through_pos:
                for dim in range(head_dim):
                    await driver.write_row(
                        MEM1_BASE,
                        kt_stage_base + dim * 2 + block,
                        zero_row,
                    )
                continue
            tile = [[0.0] * LANES for _ in range(LANES)]
            for tile_pos in range(LANES):
                pos = block * LANES + tile_pos
                if pos <= through_pos:
                    for dim in range(head_dim):
                        tile[tile_pos][dim] = key_vectors[pos][
                            head * head_dim + dim
                        ]
            absmax = max(abs(value) for row in tile for value in row)
            tile_scale = 0 if absmax == 0.0 else math.ceil(math.log2(absmax / 127.0))
            tile_scale = max(-128, min(127, tile_scale))
            for tile_pos, row in enumerate(tile):
                await driver.write_row(
                    MEM0_BASE,
                    k_tile_in_base + tile_pos,
                    pack_float_row_at_scale(row, tile_scale),
                )
            await run_section(f"k_transpose_block{block}")

        await run_section("qk")

        # Duplicate each MXM score for the existing 16-chunk FPGA softmax.
        # A 4-wide head requires QK/sqrt(4), exactly an exponent decrement.
        score_rows = [
            unpack_row(await driver.read_row(MEM0_BASE, qk_score_base + 0)),
            unpack_row(await driver.read_row(MEM0_BASE, qk_score_base + 1)),
        ]
        for pos in range(16):
            if pos <= through_pos:
                score_lanes, score_scale = score_rows[pos // LANES]
                scaled = max(-128, score_scale - 1)
                row = pack_quant_row([score_lanes[pos % LANES]] * LANES, scaled)
            else:
                row = masked
            await driver.write_row(MEM0_BASE, 600 + pos, row)
        await run_section("softmax")

        # Softmax emits eight equal lanes per token whose sum is p. Raising
        # the exponent by three changes each lane from p/8 to p for MXM input.
        # V rows are lane-masked to the current head; MXM performs every p*V
        # multiply and every accumulation across the 16 token positions.
        for pos in range(16):
            if pos <= through_pos:
                prob_lanes, prob_scale = unpack_row(
                    await driver.read_row(MEM0_BASE, 640 + pos)
                )
                await driver.write_row(
                    MEM0_BASE,
                    pv_prob_base + pos,
                    pack_quant_row(prob_lanes, min(127, prob_scale + 3)),
                )
                v_lanes, v_scale = unpack_row(value_rows[pos][row_index])
                masked_v = [0] * LANES
                for lane in range(lane_base, lane_base + head_dim):
                    masked_v[lane] = v_lanes[lane]
                await driver.write_row(
                    MEM1_BASE,
                    v_stage_base + pos,
                    pack_quant_row(masked_v, v_scale),
                )
            else:
                await driver.write_row(MEM0_BASE, pv_prob_base + pos, zero_row)
                await driver.write_row(MEM1_BASE, v_stage_base + pos, zero_row)

        await run_section("pv")
        await copy_row(driver, MEM0_BASE, pv_out_row, MEM0_BASE, head_out_base + head)

    # Heads 0/1 occupy disjoint halves of row 0, and heads 2/3 disjoint halves
    # of row 1. Existing VXM residual-add merges each pair on FPGA.
    await run_section("merge")


async def log_decode_state(driver: AvalonDriver, dut, mem0_abi: dict, label: str) -> None:
    named_rows = [
        ("x0", 8),
        ("x1", 9),
        ("xb0", 32),
        ("xb1", 33),
        ("xb8", 40),
        ("xb9", 41),
        ("q0", int(mem0_abi["q_rows"][0])),
        ("q1", int(mem0_abi["q_rows"][1])),
        ("k0", int(mem0_abi["k_rows"][0])),
        ("k1", int(mem0_abi["k_rows"][1])),
        ("v0", int(mem0_abi["v_rows"][0])),
        ("v1", int(mem0_abi["v_rows"][1])),
        ("attn0", int(mem0_abi["staged_attention_rows"][0])),
        ("attn1", int(mem0_abi["staged_attention_rows"][1])),
    ]
    raw = {name: await driver.read_row(MEM0_BASE, row) for name, row in named_rows}
    for group in [
        ("x", ["x0", "x1"]),
        ("xb0", ["xb0"]),
        ("xb1", ["xb1"]),
        ("xb8", ["xb8"]),
        ("xb9", ["xb9"]),
        ("q", ["q0", "q1"]),
        ("k", ["k0", "k1"]),
        ("v", ["v0", "v1"]),
        ("attn", ["attn0", "attn1"]),
    ]:
        values = row_to_vec([raw[name] for name in group[1]])
        dut._log.info("%s %-5s %s", label, group[0], vec_summary(values))
    dut._log.info(
        "%s raw x0=%s q0=%s k0=%s v0=%s attn0=%s",
        label,
        row_summary(raw["x0"]),
        row_summary(raw["q0"]),
        row_summary(raw["k0"]),
        row_summary(raw["v0"]),
        row_summary(raw["attn0"]),
    )


def find_trace_pc(trace: list[dict], startswith: str) -> int:
    for entry in trace:
        note = entry.get("note", "")
        if isinstance(note, str) and note.startswith(startswith):
            return int(entry["pc"])
    raise AssertionError(f"trace note not found: {startswith}")


def choose_next_token(
    logits: list[float],
    generated: str,
    characters: list[str],
    target_names: list[str],
    decode_mode: str,
) -> int:
    if decode_mode != "target":
        return max(range(len(logits)), key=lambda idx: logits[idx])

    allowed: set[int] = set()
    for target in target_names:
        if target.startswith(generated) and len(target) > len(generated):
            try:
                allowed.add(characters.index(target[len(generated)]))
            except ValueError:
                pass

    if not allowed:
        return max(range(len(logits)), key=lambda idx: logits[idx])
    return max(allowed, key=lambda idx: logits[idx])


def safe_int_signal(handle) -> str:
    try:
        return str(int(handle.value))
    except Exception:
        return str(handle.value)


def log_internal_flow(dut, label: str) -> None:
    names = [
        ("vxm_input_overflow", dut.u_lpu.vxm_input_overflow),
        ("vxm_fifo_full", dut.u_lpu.vxm_fifo_full),
        ("vxm_fifo_empty", dut.u_lpu.vxm_fifo_empty),
        ("vxm_result_full", dut.u_lpu.vxm_result_full),
        ("vxm_result_empty", dut.u_lpu.vxm_result_empty),
        ("vxm_out_valid_live", dut.u_lpu.vxm_out_valid_live),
        ("mxm_valid_e", dut.u_lpu.mxm_valid_e),
        ("eastbound_valid", dut.u_lpu.eastbound_valid),
    ]
    dut._log.info("%s internals %s", label, " ".join(f"{name}={safe_int_signal(sig)}" for name, sig in names))


async def decode_logits(driver: AvalonDriver, mem0_abi: dict, vocab_size: int) -> list[float]:
    logits: list[float] = []
    for row in mem0_abi["logit_rows"]:
        lanes, scale = unpack_row(await driver.read_row(MEM0_BASE, int(row)))
        logits.extend(math.ldexp(float(lane), scale) for lane in lanes)
    return logits[:vocab_size]


async def log_mem0_rows(driver: AvalonDriver, dut, label: str, rows: list[tuple[str, int]]) -> None:
    values = []
    for name, row in rows:
        values.append(f"{name}=0x{await driver.read_row(MEM0_BASE, row):018x}")
    dut._log.info("%s %s", label, " ".join(values))


async def log_mem0_vector(driver: AvalonDriver, dut, label: str, rows: list[int], limit: int | None = None) -> None:
    values: list[float] = []
    raw_rows: list[str] = []
    for row in rows:
        raw = await driver.read_row(MEM0_BASE, row)
        lanes, scale = unpack_row(raw)
        values.extend(math.ldexp(float(lane), scale) for lane in lanes)
        raw_rows.append(f"{row}:{row_summary(raw)}")
    if limit is not None:
        values = values[:limit]
    dut._log.info("%s %s", label, vec_summary(values))
    dut._log.info("%s raw %s", label, " | ".join(raw_rows))


@cocotb.test()
async def test_microgpt_wrapper_prompt_next_token(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    driver = AvalonDriver(dut)
    await driver.init()

    schedule = json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))
    mem1 = read_hex(MEM1_PATH)
    vliw = read_hex(VLIW_PATH)
    softmax_vliw = read_hex(SOFTMAX_VLIW_PATH)
    attention_vliw = read_hex(ATTENTION_VLIW_PATH)
    config = schedule["config"]
    tokenizer = schedule["tokenizer"]
    mem0_abi = schedule["mem0_abi"]
    attention_meta = schedule["microkernels"]["attention"]
    symbols = schedule["mem1"]["symbols"]
    page_size = int(schedule["imem"]["page_size"])
    trace = json.loads((ARTIFACT_DIR / "microgpt_decode_trace.json").read_text(encoding="utf-8"))
    prefix_count = find_trace_pc(trace, "broadcast staged attention row0")
    attention_bcast_start_pc = find_trace_pc(trace, "broadcast attention input row0")
    wq_start_pc = find_trace_pc(trace, "layer0.attn_wq: clear MXM output block 0")
    wk_start_pc = find_trace_pc(trace, "layer0.attn_wk: clear MXM output block 0")
    wv_start_pc = find_trace_pc(trace, "layer0.attn_wv: clear MXM output block 0")
    suffix_attention_proj_start_pc = find_trace_pc(trace, "attn output projection: clear MXM output block 0")
    suffix_mlp_bcast_start_pc = find_trace_pc(trace, "broadcast mlp row0")
    suffix_mlp_fc1_start_pc = find_trace_pc(trace, "mlp fc1: clear MXM output block 0")
    hidden_bcast_starts = [find_trace_pc(trace, f"broadcast mlp hidden {idx}") for idx in range(8)]
    hidden_bcast_ends = [find_trace_pc(trace, f"broadcast mlp hidden {idx}: write broadcast lane 7") + 1 for idx in range(8)]
    final_bcast_start_pc = find_trace_pc(trace, "broadcast final row0")
    lm_head_start_pc = find_trace_pc(trace, "lm head: clear MXM output block 0")
    suffix_count = len(vliw) - prefix_count

    await load_mem1(driver, mem1)
    mem1_probe = await driver.read_row(MEM1_BASE, 0)
    assert mem1_probe == mem1[0], (
        f"MEM1 readback mismatch row0 got=0x{mem1_probe:018x} expected=0x{mem1[0]:018x}"
    )

    prompt = os.getenv("MICROGPT_PROMPT", "sat").lower()
    if prompt == "<empty>":
        prompt = ""
    expected_next_char = os.getenv("MICROGPT_EXPECT_NEXT", "v")
    broadcast_mode = os.getenv("MICROGPT_BROADCAST_MODE", "host").lower()
    assert broadcast_mode in {"host", "sxm"}, (
        f"MICROGPT_BROADCAST_MODE must be 'host' or 'sxm', got {broadcast_mode!r}"
    )
    attention_mode = os.getenv("MICROGPT_ATTENTION", "host").lower()
    assert attention_mode in {"host", "fpga-softmax", "fpga-mxm"}, (
        "MICROGPT_ATTENTION must be 'host', 'fpga-softmax', or "
        f"'fpga-mxm', got {attention_mode!r}"
    )
    characters = tokenizer["characters"]
    char_to_id = {character: index for index, character in enumerate(characters)}
    tokens = [int(tokenizer["bos_token_id"])] + [char_to_id[ch] for ch in prompt]

    await reset_prompt_state(driver, mem0_abi)

    async def run_one_token(pos: int, token_id: int) -> list[float]:
        await write_step_inputs(driver, mem1, symbols, token_id, pos)
        await run_program(driver, vliw, 0, attention_bcast_start_pc, page_size)
        if broadcast_mode == "host":
            await stage_broadcast_row(driver, 10, 32)
            await stage_broadcast_row(driver, 11, 40)
            qkv_start_pc = wq_start_pc
        else:
            qkv_start_pc = attention_bcast_start_pc
        if pos == 0 and prefix_phase_probes_enabled():
            await log_decode_state(driver, dut, mem0_abi, f"pos={pos} token={token_id} after host X broadcast")
            await run_program(driver, vliw, qkv_start_pc, wk_start_pc - qkv_start_pc, page_size)
            log_internal_flow(dut, f"pos={pos} token={token_id} after WQ")
            await log_decode_state(driver, dut, mem0_abi, f"pos={pos} token={token_id} after WQ")
            await run_program(driver, vliw, wk_start_pc, wv_start_pc - wk_start_pc, page_size)
            log_internal_flow(dut, f"pos={pos} token={token_id} after WK")
            await log_decode_state(driver, dut, mem0_abi, f"pos={pos} token={token_id} after WK")
            await run_program(driver, vliw, wv_start_pc, prefix_count - wv_start_pc, page_size)
            log_internal_flow(dut, f"pos={pos} token={token_id} after WV")
            await log_decode_state(driver, dut, mem0_abi, f"pos={pos} token={token_id} after WV")
        else:
            await run_program(driver, vliw, qkv_start_pc, prefix_count - qkv_start_pc, page_size)
        if pos == 0 and state_probes_for_position(pos):
            log_internal_flow(dut, f"pos={pos} token={token_id} after prefix")
        if state_probes_for_position(pos):
            await log_decode_state(
                driver,
                dut,
                mem0_abi,
                f"pos={pos} token={token_id} after prefix",
            )
        if pos == 0 and state_probes_for_position(pos):
            imem_probe = await driver.read_row(IMEM_BASE, 0)
            embed_probe = await driver.read_row(MEM0_BASE, 8)
            q_probe = await driver.read_row(MEM0_BASE, int(mem0_abi["q_rows"][0]))
            dut._log.info(
                "probe after first prefix imem0=0x%024x embed0=0x%018x q0=0x%018x",
                imem_probe,
                embed_probe,
                q_probe,
            )
        await cache_current_kv(driver, mem0_abi, pos)
        if state_probes_for_position(pos):
            await log_decode_state(
                driver,
                dut,
                mem0_abi,
                f"pos={pos} token={token_id} after cache",
            )
        if attention_mode == "fpga-mxm":
            await stage_mxm_attention(
                driver,
                mem0_abi,
                pos,
                int(config["n_head"]),
                attention_vliw,
                attention_meta,
            )
        else:
            await stage_host_attention(
                driver,
                mem0_abi,
                pos,
                int(config["n_head"]),
                softmax_vliw if attention_mode == "fpga-softmax" else None,
                page_size,
            )
        if state_probes_for_position(pos):
            await log_decode_state(
                driver,
                dut,
                mem0_abi,
                f"pos={pos} token={token_id} after attention",
            )
        if broadcast_mode == "sxm":
            # The remainder of the static schedule contains every attention,
            # MLP-hidden, and final-residual SXM broadcast.  Running it as one
            # range exercises the same dataflow used by the FPGA runtime.
            await run_program(driver, vliw, prefix_count, len(vliw) - prefix_count, page_size)
            return await decode_logits(driver, mem0_abi, int(config["vocab_size"]))

        await stage_broadcast_row(driver, int(mem0_abi["staged_attention_rows"][0]), 112)
        await stage_broadcast_row(driver, int(mem0_abi["staged_attention_rows"][1]), 120)
        await run_program(
            driver,
            vliw,
            suffix_attention_proj_start_pc,
            suffix_mlp_bcast_start_pc - suffix_attention_proj_start_pc,
            page_size,
        )
        if suffix_probes_for_position(pos):
            await log_mem0_vector(driver, dut, f"pos={pos} after attn projection", [160, 161], N_EMBD)
            await log_mem0_vector(driver, dut, f"pos={pos} after attn residual", [176, 177], N_EMBD)
            await log_mem0_vector(driver, dut, f"pos={pos} after mlp rms", [10, 11], N_EMBD)
        await stage_broadcast_row(driver, 10, 192)
        await stage_broadcast_row(driver, 11, 200)
        await run_program(
            driver,
            vliw,
            suffix_mlp_fc1_start_pc,
            hidden_bcast_starts[0] - suffix_mlp_fc1_start_pc,
            page_size,
        )
        for block in range(8):
            current_pc = hidden_bcast_starts[0] if block == 0 else hidden_bcast_ends[block - 1]
            if hidden_bcast_starts[block] > current_pc:
                await run_program(driver, vliw, current_pc, hidden_bcast_starts[block] - current_pc, page_size)
            await stage_broadcast_row(driver, 256 + block, 320 + block * LANES)
        await run_program(
            driver,
            vliw,
            hidden_bcast_ends[7],
            final_bcast_start_pc - hidden_bcast_ends[7],
            page_size,
        )
        await stage_broadcast_row(driver, 8, 32)
        await stage_broadcast_row(driver, 9, 40)
        await run_program(driver, vliw, lm_head_start_pc, len(vliw) - lm_head_start_pc, page_size)
        if suffix_probes_for_position(pos):
            await log_mem0_vector(driver, dut, f"pos={pos} mlp hidden", list(range(256, 264)), 64)
            await log_mem0_vector(driver, dut, f"pos={pos} mlp output", [400, 401], N_EMBD)
            await log_mem0_vector(driver, dut, f"pos={pos} final residual", [8, 9], N_EMBD)
            await log_mem0_vector(driver, dut, f"pos={pos} final broadcast", list(range(32, 48)), 128)
            await log_mem0_vector(driver, dut, f"pos={pos} logits", [448, 449, 450, 451], int(config["vocab_size"]))
        return await decode_logits(driver, mem0_abi, int(config["vocab_size"]))

    logits: list[float] = []
    warmup_prompts = [
        item.strip().lower()
        for item in os.getenv("MICROGPT_WARMUP_PROMPTS", "").split(",")
        if item.strip()
    ]
    if warmup_prompts:
        checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        target_names = set(checkpoint.get("training", {}).get("target_names", []))
        for warmup_prompt in warmup_prompts:
            await reset_prompt_state(driver, mem0_abi)
            warmup_ids = [int(tokenizer["bos_token_id"])] + [char_to_id[ch] for ch in warmup_prompt]
            warmup_logits: list[float] = []
            for warmup_pos, warmup_id in enumerate(warmup_ids):
                warmup_logits = await run_one_token(warmup_pos, warmup_id)
            warmup_generated = list(warmup_prompt)
            for warmup_step in range(12):
                warmup_next = max(range(len(warmup_logits)), key=lambda idx: warmup_logits[idx])
                if warmup_next == int(tokenizer["bos_token_id"]):
                    break
                warmup_generated.append(characters[warmup_next])
                if ''.join(warmup_generated) in target_names:
                    break
                warmup_logits = await run_one_token(
                    len(warmup_ids) + warmup_step,
                    warmup_next,
                )
            dut._log.info(
                "MicroGPT same-boot warmup prompt=%r generated=%r",
                warmup_prompt,
                ''.join(warmup_generated),
            )
        # Match the Linux runtime: every new terminal prompt performs a soft
        # reset and clears the scratch/KV state without reloading the model.
        await reset_prompt_state(driver, mem0_abi)

    for pos, token_id in enumerate(tokens):
        logits = await run_one_token(pos, token_id)

    ranked = sorted(range(len(logits)), key=lambda idx: logits[idx], reverse=True)[:5]
    next_id = ranked[0]
    next_char = characters[next_id] if next_id < len(characters) else "<bos>"
    dut._log.info(
        "MicroGPT wrapper prompt=%r next=%s top5=%s schedule=%d prefix=%d suffix=%d broadcast=%s attention=%s",
        prompt,
        next_char,
        [(idx, characters[idx] if idx < len(characters) else "<bos>", logits[idx]) for idx in ranked],
        len(vliw),
        prefix_count,
        suffix_count,
        broadcast_mode,
        attention_mode,
    )
    assert next_char == expected_next_char, (
        f"expected next token {expected_next_char!r}, got {next_char!r}; "
        f"top5={[(idx, characters[idx] if idx < len(characters) else '<bos>', logits[idx]) for idx in ranked]}"
    )

    expected_completion = os.getenv("MICROGPT_EXPECT_COMPLETION")
    if expected_completion:
        checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        target_names = list(checkpoint.get("training", {}).get("target_names", []))
        decode_mode = os.getenv("MICROGPT_DECODE", "greedy")
        generated = list(prompt)
        step = 0
        while len(generated) < len(expected_completion):
            next_id = choose_next_token(logits, ''.join(generated), characters, target_names, decode_mode)
            assert next_id != int(tokenizer["bos_token_id"]), (
                f"generated BOS early; generated={''.join(generated)!r} expected={expected_completion!r}"
            )
            next_ch = characters[next_id]
            generated.append(next_ch)
            if ''.join(generated) == expected_completion:
                break
            pos = len(tokens) + step
            logits = await run_one_token(pos, next_id)
            step += 1
        dut._log.info(
            "MicroGPT wrapper completion prompt=%r generated=%r expected=%r decode=%s",
            prompt,
            ''.join(generated),
            expected_completion,
            decode_mode,
        )
        assert ''.join(generated) == expected_completion
        if os.getenv("MICROGPT_EXPECT_STOP", "1") == "1":
            processed_characters = len(prompt) + step
            if len(generated) > processed_characters:
                logits = await run_one_token(len(tokens) + step, next_id)
            stop_id = max(range(len(logits)), key=lambda idx: logits[idx])
            assert stop_id == int(tokenizer["bos_token_id"]), (
                f"expected BOS after {expected_completion!r}, got "
                f"{characters[stop_id] if stop_id < len(characters) else '<bos>'!r}"
            )
            dut._log.info("MicroGPT wrapper stop token after %r is BOS", expected_completion)
