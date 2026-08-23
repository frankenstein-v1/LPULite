#!/usr/bin/env python3
"""Run dependency-free CPU inference from the LPULite MicroGPT INT8 export."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = MODEL_DIR / "artifacts" / "microgpt_weights_int8.json"


def dequantize_matrix(tensor: dict) -> list[list[float]]:
    matrix = []
    for lanes, scale_exponents in zip(tensor["lanes"], tensor["scale_exponents"]):
        row = []
        for block_index, start in enumerate(range(0, len(lanes), 8)):
            scale = math.ldexp(1.0, scale_exponents[block_index])
            row.extend(lane * scale for lane in lanes[start : start + 8])
        matrix.append(row)
    return matrix


def linear(vector: list[float], weights: list[list[float]]) -> list[float]:
    return [sum(weight * value for weight, value in zip(row, vector)) for row in weights]


def rmsnorm(vector: list[float]) -> list[float]:
    mean_square = sum(value * value for value in vector) / len(vector)
    scale = (mean_square + 1e-5) ** -0.5
    return [value * scale for value in vector]


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU simulation of LPULite MicroGPT INT8 inference")
    parser.add_argument("prompt", help="lowercase character prefix, for example: ken")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    if checkpoint.get("format") != "lpulite.microgpt.lpu_int8":
        parser.error("checkpoint is not a LPULite MicroGPT INT8 export")

    config = checkpoint["config"]
    characters = checkpoint["tokenizer"]["characters"]
    bos = checkpoint["tokenizer"]["bos_token_id"]
    char_to_id = {character: index for index, character in enumerate(characters)}
    prompt = args.prompt.lower()
    unknown = sorted(set(prompt) - set(char_to_id))
    if unknown:
        parser.error(f"prompt contains unsupported characters: {''.join(unknown)!r}")

    weights = {name: dequantize_matrix(tensor) for name, tensor in checkpoint["state_dict"].items()}
    n_layer = config["n_layer"]
    n_head = config["n_head"]
    head_dim = config["n_embd"] // n_head

    def gpt(token_id: int, position: int, keys: list, values: list) -> list[float]:
        token_embedding = weights["wte"][token_id]
        position_embedding = weights["wpe"][position]
        vector = rmsnorm([token + pos for token, pos in zip(token_embedding, position_embedding)])

        for layer in range(n_layer):
            residual = vector
            normalized = rmsnorm(vector)
            query = linear(normalized, weights[f"layer{layer}.attn_wq"])
            key = linear(normalized, weights[f"layer{layer}.attn_wk"])
            value = linear(normalized, weights[f"layer{layer}.attn_wv"])
            keys[layer].append(key)
            values[layer].append(value)
            attention_output = []
            for head in range(n_head):
                start = head * head_dim
                query_head = query[start : start + head_dim]
                key_heads = [item[start : start + head_dim] for item in keys[layer]]
                value_heads = [item[start : start + head_dim] for item in values[layer]]
                scores = [
                    sum(query_head[j] * key_head[j] for j in range(head_dim)) / math.sqrt(head_dim)
                    for key_head in key_heads
                ]
                probabilities = softmax(scores)
                attention_output.extend(
                    sum(probabilities[index] * value_heads[index][j] for index in range(len(value_heads)))
                    for j in range(head_dim)
                )
            vector = linear(attention_output, weights[f"layer{layer}.attn_wo"])
            vector = [value + skip for value, skip in zip(vector, residual)]

            residual = vector
            vector = linear(rmsnorm(vector), weights[f"layer{layer}.mlp_fc1"])
            vector = [max(0.0, value) for value in vector]
            vector = linear(vector, weights[f"layer{layer}.mlp_fc2"])
            vector = [value + skip for value, skip in zip(vector, residual)]

        return linear(vector, weights["lm_head"])

    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]
    prompt_ids = [bos] + [char_to_id[character] for character in prompt]
    for position, token_id in enumerate(prompt_ids):
        logits = gpt(token_id, position, keys, values)

    generated = list(prompt)
    for position in range(len(prompt_ids), config["block_size"]):
        token_id = max(range(config["vocab_size"]), key=lambda index: logits[index])
        if token_id == bos:
            break
        generated.append(characters[token_id])
        logits = gpt(token_id, position, keys, values)

    print(f"checkpoint: {args.checkpoint}")
    print("numeric format: 8x signed INT8 lanes + shared power-of-two scale; INT32 MAC contract")
    print(f"prompt: {prompt}")
    print(f"completion: {''.join(generated)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
