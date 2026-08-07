"""MicroGPT inference with every learned linear transform executed by TinyLPU RTL."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


MODEL_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT = MODEL_DIR / "artifacts" / "microgpt_weights_int8.json"


def signed(value: int, width: int) -> int:
    value &= (1 << width) - 1
    return value - (1 << width) if value & (1 << (width - 1)) else value


def quantize_block(values: list[float]) -> tuple[list[int], int]:
    maximum = max((abs(value) for value in values), default=0.0)
    exponent = 0 if maximum == 0.0 else math.ceil(math.log2(maximum / 127.0))
    exponent = max(-128, min(127, exponent))
    inverse = math.ldexp(1.0, -exponent)
    lanes = [max(-127, min(127, round(value * inverse))) for value in values]
    return lanes, exponent


def pack_lanes(lanes: list[int]) -> int:
    padded = lanes[:8] + [0] * (8 - len(lanes))
    return sum((lane & 0xFF) << (8 * index) for index, lane in enumerate(padded))


def dequantize_matrix(tensor: dict) -> list[list[float]]:
    result = []
    for lanes, exponents in zip(tensor["lanes"], tensor["scale_exponents"]):
        row = []
        for block_index, start in enumerate(range(0, len(lanes), 8)):
            scale = math.ldexp(1.0, exponents[block_index])
            row.extend(lane * scale for lane in lanes[start : start + 8])
        result.append(row)
    return result


def rmsnorm(values: list[float]) -> list[float]:
    scale = (sum(value * value for value in values) / len(values) + 1e-5) ** -0.5
    return [value * scale for value in values]


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


async def run_tile(dut, inputs: list[float], weight_tile: list[list[float]]) -> list[float]:
    input_lanes, input_exp = quantize_block(inputs)
    all_weights = [weight_tile[row][column] for row in range(len(weight_tile)) for column in range(8)]
    _, weight_exp = quantize_block(all_weights)
    inverse_weight_scale = math.ldexp(1.0, -weight_exp)
    quantized_weights = [
        [max(-127, min(127, round(value * inverse_weight_scale))) for value in row]
        for row in weight_tile
    ]

    dut.clear.value = 1
    await RisingEdge(dut.clk)
    dut.clear.value = 0

    async def stage_activation(index: int) -> None:
        dut.payload.value = pack_lanes([input_lanes[index]] * 8)
        dut.input_scale.value = input_exp
        dut.ingress_mode.value = 1
        dut.payload_valid.value = 1
        dut.mxm_enable.value = 1
        await RisingEdge(dut.clk)

    async def stage_weight(index: int) -> None:
        dut.payload.value = pack_lanes([row[index] for row in quantized_weights])
        dut.weight_scale.value = weight_exp
        dut.ingress_mode.value = 2
        dut.payload_valid.value = 1
        dut.mxm_enable.value = 1
        await RisingEdge(dut.clk)

    async def start_staged_row() -> None:
        dut.payload_valid.value = 0
        dut.mxm_enable.value = 0
        dut.ingress_mode.value = 0
        dut.start.value = 1
        await RisingEdge(dut.clk)
        dut.start.value = 0

    # Prime row zero. Thereafter the registered MAC accumulation edge is also
    # used to stage the next activation into the inactive ping-pong bank. The
    # following edge stages its weight, and the third promotes both banks.
    await stage_activation(0)
    await stage_weight(0)
    await start_staged_row()

    for index in range(1, 8):
        await stage_activation(index)  # row index - 1 accumulates on this edge
        await stage_weight(index)
        await start_staged_row()

    # Retire row seven; there is no subsequent activation to preload.
    dut.payload_valid.value = 0
    dut.mxm_enable.value = 0
    dut.ingress_mode.value = 0
    await RisingEdge(dut.clk)

    await Timer(1, unit="ps")
    scale = math.ldexp(1.0, input_exp + weight_exp)
    result = [signed(int(getattr(dut, f"result{index}").value), 32) * scale for index in range(len(weight_tile))]
    return result


async def lpu_matvec(dut, vector: list[float], matrix: list[list[float]]) -> list[float]:
    outputs = [0.0] * len(matrix)
    reference = [0.0] * len(matrix)
    # The trained checkpoint has one scale per eight consecutive values in an
    # output row. Run one logical output row per tile so that its exact scale is
    # retained; unused physical MXM columns are filled with zero.
    for output_start in range(len(matrix)):
        output_count = 1
        for input_start in range(0, len(vector), 8):
            input_block = vector[input_start : input_start + 8]
            if len(input_block) < 8:
                input_block += [0.0] * (8 - len(input_block))
            tile = []
            for output_index in range(8):
                if output_index < output_count:
                    row = matrix[output_start + output_index][input_start : input_start + 8]
                    tile.append(row + [0.0] * (8 - len(row)))
                else:
                    tile.append([0.0] * 8)
            partial = await run_tile(dut, input_block, tile)
            input_lanes, input_exp = quantize_block(input_block)
            row_lanes, row_exp = quantize_block(tile[0])
            expected = sum(left * right for left, right in zip(input_lanes, row_lanes))
            expected *= math.ldexp(1.0, input_exp + row_exp)
            observed = partial[0]
            assert math.isclose(observed, expected, rel_tol=0.0, abs_tol=max(1e-9, abs(expected) * 1e-9)), (
                f"LPU tile mismatch: observed={observed} expected={expected} "
                f"partial={partial[0]} input={input_lanes}@2**{input_exp} "
                f"weight={row_lanes}@2**{row_exp}"
            )
            for index in range(output_count):
                outputs[output_start + index] += partial[index]
            reference[output_start] += expected
    for observed, expected in zip(outputs, reference):
        assert math.isclose(observed, expected, rel_tol=0.0, abs_tol=max(1e-9, abs(expected) * 1e-9))
    return outputs


async def initialize_mxm(dut) -> None:
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst.value = 1
    dut.clear.value = 0
    dut.start.value = 0
    dut.payload.value = 0
    dut.payload_valid.value = 0
    dut.mxm_enable.value = 0
    dut.ingress_mode.value = 0
    dut.input_scale.value = 0
    dut.weight_scale.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


async def capture_row(dut, lanes: list[int], mode: int) -> None:
    dut.payload.value = pack_lanes(lanes)
    dut.ingress_mode.value = mode
    dut.payload_valid.value = 1
    dut.mxm_enable.value = 1
    await RisingEdge(dut.clk)
    dut.payload_valid.value = 0
    dut.mxm_enable.value = 0
    dut.ingress_mode.value = 0


@cocotb.test()
async def test_mxm_double_buffer_overlap(dut):
    """Preload B while A accumulates, then atomically promote B."""
    await initialize_mxm(dut)

    await capture_row(dut, [2] * 8, 1)
    await capture_row(dut, [3] * 8, 2)

    # Promote A. Its registered MAC enable fires on the following edge; use
    # that same edge to capture activation B into the now-inactive bank.
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    dut.payload.value = pack_lanes([4] * 8)
    dut.ingress_mode.value = 1
    dut.payload_valid.value = 1
    dut.mxm_enable.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")
    assert signed(int(dut.result0.value), 32) == 6

    dut.payload_valid.value = 0
    dut.mxm_enable.value = 0
    dut.ingress_mode.value = 0
    await capture_row(dut, [5] * 8, 2)

    # Promote B and accumulate 4*5 without clearing A: 6 + 20 = 26.
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")
    assert signed(int(dut.result0.value), 32) == 26


@cocotb.test()
async def test_microgpt_ken_on_lpu(dut):
    await initialize_mxm(dut)

    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    config = checkpoint["config"]
    characters = checkpoint["tokenizer"]["characters"]
    bos = checkpoint["tokenizer"]["bos_token_id"]
    char_to_id = {character: index for index, character in enumerate(characters)}
    weights = {name: dequantize_matrix(value) for name, value in checkpoint["state_dict"].items()}
    n_layer = config["n_layer"]
    n_head = config["n_head"]
    head_dim = config["n_embd"] // n_head
    linear_calls = 0

    async def linear(vector, matrix):
        nonlocal linear_calls
        linear_calls += 1
        return await lpu_matvec(dut, vector, matrix)

    async def gpt(token_id, position, keys, values):
        vector = rmsnorm([
            token + pos
            for token, pos in zip(weights["wte"][token_id], weights["wpe"][position])
        ])
        for layer in range(n_layer):
            residual = vector
            normalized = rmsnorm(vector)
            query = await linear(normalized, weights[f"layer{layer}.attn_wq"])
            key = await linear(normalized, weights[f"layer{layer}.attn_wk"])
            value = await linear(normalized, weights[f"layer{layer}.attn_wv"])
            keys[layer].append(key)
            values[layer].append(value)
            attention_output = []
            for head in range(n_head):
                start = head * head_dim
                query_head = query[start : start + head_dim]
                key_heads = [item[start : start + head_dim] for item in keys[layer]]
                value_heads = [item[start : start + head_dim] for item in values[layer]]
                probabilities = softmax([
                    sum(query_head[j] * key_head[j] for j in range(head_dim)) / math.sqrt(head_dim)
                    for key_head in key_heads
                ])
                attention_output.extend(
                    sum(probabilities[index] * value_heads[index][j] for index in range(len(value_heads)))
                    for j in range(head_dim)
                )
            vector = await linear(attention_output, weights[f"layer{layer}.attn_wo"])
            vector = [value + skip for value, skip in zip(vector, residual)]
            residual = vector
            vector = await linear(rmsnorm(vector), weights[f"layer{layer}.mlp_fc1"])
            vector = [max(0.0, value) for value in vector]
            vector = await linear(vector, weights[f"layer{layer}.mlp_fc2"])
            vector = [value + skip for value, skip in zip(vector, residual)]
        return await linear(vector, weights["lm_head"])

    prompt = os.getenv("MICROGPT_PROMPT", "ken").lower()
    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]
    prompt_ids = [bos] + [char_to_id[character] for character in prompt]
    for position, token_id in enumerate(prompt_ids):
        logits = await gpt(token_id, position, keys, values)

    generated = list(prompt)
    for position in range(len(prompt_ids), config["block_size"]):
        token_id = max(range(config["vocab_size"]), key=lambda index: logits[index])
        if token_id == bos:
            break
        generated.append(characters[token_id])
        logits = await gpt(token_id, position, keys, values)

    completion = "".join(generated)
    dut._log.info("MicroGPT LPU RTL prompt=%r completion=%r linear_calls=%d", prompt, completion, linear_calls)
    assert linear_calls > 0
    assert completion == "kenny"
