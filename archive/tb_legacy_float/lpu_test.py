import json
import math
import os
from pathlib import Path

import cocotb
import lpu_tb as lpu
from cocotb.clock import Clock
from cocotb.handle import Force, Release
from cocotb.triggers import Timer
from cocotb.utils import get_sim_time


import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = ROOT_DIR / "tiny_lm_weights_export.json"
STORIES10K_WEIGHTS_PATH = ROOT_DIR / "model" / "stories10k" / "stories10k_weights_export.json"
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
from tokenizer import Tokenizer

MODEL_CONFIG = {
    "vocab_size": 512,
    "dim": 64,
    "seq_len": 512,
    "layers": 5,
    "heads": 8,
    "kv_heads": 4,
    "ffn_dim": 176,
}

PROMPT_PREFILL = ["he", "is"]
PROMPT_DECODE = ["he", "is", "very"]
PROMPT_PROBE = ["he", "is"]
TILE = 8
DECODE_TOKENS = int(os.getenv("LPU_DECODE_TOKENS", "20"))
SIM_CLK_NS = 10.0
TILE_LOG = os.getenv("LPU_TILE_LOG", "0") == "1"
DECODE_PROGRESS = os.getenv("LPU_DECODE_PROGRESS", "0") == "1"
SOFTMAX_CHUNKS = int(os.getenv("LPU_SOFTMAX_CHUNKS", "64"))
FP_CTRL = dict(mxm_use_fp=1, mxm_input_is_signed=0, mxm_wght_is_signed=0)
KV_CACHE_ROWS_PER_TOKEN = MODEL_CONFIG["kv_heads"]
KV_CACHE_LAYER_ROWS = MODEL_CONFIG["seq_len"] * KV_CACHE_ROWS_PER_TOKEN


def load_tiny_lm_export():
    return load_lm_export(WEIGHTS_PATH)


def load_stories10k_export():
    return load_lm_export(STORIES10K_WEIGHTS_PATH)


def load_lm_export(path):
    with open(path, "r", encoding="utf-8") as f:
        export = json.load(f)

    vocab = export["vocab"]
    id_to_token = {}
    for token_str, val in vocab.items():
        if isinstance(val, dict):
            idx = val["id"]
        else:
            idx = val
        id_to_token[idx] = token_str
    return export["config"], vocab, id_to_token, export["weights"]



def vec_add(a_vec, b_vec):
    return [lpu.to_f32(a + b) for a, b in zip(a_vec, b_vec)]


def matmul(a_matrix, b_matrix):
    out = []
    for row in a_matrix:
        out_row = []
        for col_idx in range(len(b_matrix[0])):
            acc = 0.0
            for k_idx, value in enumerate(row):
                product = lpu.to_f32(value * b_matrix[k_idx][col_idx])
                acc = lpu.to_f32(acc + product)
            out_row.append(acc)
        out.append(out_row)
    return out


def transpose(matrix):
    return [[matrix[row][col] for row in range(len(matrix))] for col in range(len(matrix[0]))]


def linear_no_bias(rows, weight):
    return matmul(rows, transpose(weight))


def add_bias(rows, bias):
    return [[lpu.to_f32(value + bias[col_idx]) for col_idx, value in enumerate(row)] for row in rows]


def linear(rows, weight, bias):
    return add_bias(linear_no_bias(rows, weight), bias)


def layernorm_rows(rows, gamma, beta, eps=1e-5):
    out = []
    for row in rows:
        mean = lpu.to_f32(sum(row) / len(row))
        variance = lpu.to_f32(
            sum(lpu.to_f32((value - mean) * (value - mean)) for value in row) / len(row)
        )
        inv_std = lpu.to_f32(1.0 / math.sqrt(variance + eps))
        out.append([
            lpu.to_f32(lpu.to_f32(lpu.to_f32(value - mean) * inv_std) * gamma[idx] + beta[idx])
            for idx, value in enumerate(row)
        ])
    return out


def rmsnorm_rows(rows, gamma, eps=1e-5):
    out = []
    for row in rows:
        rms_sq = lpu.to_f32(sum(lpu.to_f32(value * value) for value in row) / len(row))
        inv_rms = lpu.to_f32(1.0 / math.sqrt(rms_sq + eps))
        out.append([
            lpu.to_f32(lpu.to_f32(value * inv_rms) * gamma[idx])
            for idx, value in enumerate(row)
        ])
    return out


def softmax_rows(scores):
    out = []
    for row in scores:
        row_max = max(row)
        exp_values = [math.exp(value - row_max) for value in row]
        denom = sum(exp_values)
        out.append([lpu.to_f32(value / denom) for value in exp_values])
    return out


tokenizer = None

def get_tokenizer():
    global tokenizer
    if tokenizer is None:
        tokenizer = Tokenizer(ROOT_DIR / "output/vocab.json")
    return tokenizer

def encode_prompt(tokens, vocab):
    tok = get_tokenizer()
    prompt_str = " ".join(tokens)
    return tok.encode(prompt_str, bos=True, eos=False)


_config_cache = None

def get_model_config():
    global _config_cache
    if _config_cache is None:
        with open(WEIGHTS_PATH, "r", encoding="utf-8") as f:
            export = json.load(f)
        _config_cache = export["config"]
    return _config_cache


def kv_cache_rows_per_token(config):
    return config["kv_heads"]


def kv_cache_layer_rows(config):
    return config["seq_len"] * kv_cache_rows_per_token(config)


def get_rope_cos_sin(pos, head_dim=8):
    inv_freq = [1.0 / (10000.0 ** (2 * i / head_dim)) for i in range(head_dim // 2)]
    angles = [pos * inv_freq[i] for i in range(head_dim // 2)]
    repeated = []
    for a in angles:
        repeated.extend([a, a])
    cos = [math.cos(a) for a in repeated]
    sin = [math.sin(a) for a in repeated]
    return cos, sin


def apply_rope_to_vector(x, cos, sin):
    out = [0.0] * len(x)
    for pair in range(len(x) // 2):
        even = 2 * pair
        odd = even + 1
        x_even = x[even]
        x_odd = x[odd]
        out[even] = x_even * cos[even] - x_odd * sin[even]
        out[odd] = x_even * sin[even] + x_odd * cos[even]
    return out


def tiny_lm_forward(input_ids, weights):
    config = get_model_config()
    dim = config["dim"]
    heads = config["heads"]
    kv_heads = config.get("kv_heads", heads)
    head_dim = dim // heads
    group_size = heads // kv_heads

    token_emb = weights["token_emb.weight"]

    x_prev = [token_emb[token_id] for token_id in input_ids]
    seq_len = len(input_ids)

    for layer_idx in range(config["layers"]):
        block_prefix = f"blocks.{layer_idx}"
        
        ln1_weight = weights[f"{block_prefix}.ln1.weight"]
        ln2_weight = weights[f"{block_prefix}.ln2.weight"]
        q_proj_weight = weights[f"{block_prefix}.attn.q_proj.weight"]
        k_proj_weight = weights[f"{block_prefix}.attn.k_proj.weight"]
        v_proj_weight = weights[f"{block_prefix}.attn.v_proj.weight"]
        out_proj_weight = weights[f"{block_prefix}.attn.out_proj.weight"]
        ffn_gate_weight = weights[f"{block_prefix}.ffn_gate.weight"]
        ffn_down_weight = weights[f"{block_prefix}.ffn_down.weight"]
        
        # 1. RMSNorm
        ln1 = rmsnorm_rows(x_prev, ln1_weight)
        
        # 2. Linear projections
        q = linear_no_bias(ln1, q_proj_weight)
        k = linear_no_bias(ln1, k_proj_weight)
        v = linear_no_bias(ln1, v_proj_weight)
        
        # 3. Apply RoPE to Q and K
        q_rot = []
        k_rot = []
        for t in range(seq_len):
            cos_val, sin_val = get_rope_cos_sin(t, head_dim=8)
            
            q_row = q[t]
            q_rot_row = []
            for h in range(heads):
                q_head = q_row[h*8 : (h+1)*8]
                q_rot_row.extend(apply_rope_to_vector(q_head, cos_val, sin_val))
            q_rot.append(q_rot_row)
            
            k_row = k[t]
            k_rot_row = []
            for kh in range(kv_heads):
                k_head = k_row[kh*8 : (kh+1)*8]
                k_rot_row.extend(apply_rope_to_vector(k_head, cos_val, sin_val))
            k_rot.append(k_rot_row)
            
        # 4. Self-attention
        attn_heads = []
        scale = 1.0 / math.sqrt(head_dim)
        for h in range(heads):
            kh = h // group_size
            q_h = [q_rot[t][h*8 : (h+1)*8] for t in range(seq_len)]
            k_kh = [k_rot[t][kh*8 : (kh+1)*8] for t in range(seq_len)]
            v_kh = [v[t][kh*8 : (kh+1)*8] for t in range(seq_len)]

            s_raw = matmul(q_h, transpose(k_kh))
            scores_h = []
            for row_idx, row in enumerate(s_raw):
                score_row = []
                for col_idx, value in enumerate(row):
                    if col_idx > row_idx:
                        score_row.append(-1.0e30)
                    else:
                        score_row.append(lpu.to_f32(value * scale))
                scores_h.append(score_row)

            probs_h = softmax_rows(scores_h)
            out_h = matmul(probs_h, v_kh)
            attn_heads.append(out_h)

        # Concatenate head outputs
        attn = []
        for t in range(seq_len):
            row = []
            for h in range(heads):
                row.extend(attn_heads[h][t])
            attn.append(row)
            
        # 5. Attention out projection
        attn_out = linear_no_bias(attn, out_proj_weight)
        x_after_attn = [vec_add(row, attn_out[row_idx]) for row_idx, row in enumerate(x_prev)]
        
        # 6. FFN RMSNorm
        ln2 = rmsnorm_rows(x_after_attn, ln2_weight)
        
        # 7. FFN Gate (W1)
        ffn_gate_out = linear_no_bias(ln2, ffn_gate_weight)
        
        # 8. ReLU
        ffn_hidden = [[lpu.to_f32(max(0.0, value)) for value in row] for row in ffn_gate_out]
        
        # 9. FFN Down (W2)
        ffn_down_out = linear_no_bias(ffn_hidden, ffn_down_weight)
        
        # 10. Residual
        x_prev = [vec_add(row, ffn_down_out[row_idx]) for row_idx, row in enumerate(x_after_attn)]
        
    # Final RMSNorm
    final = rmsnorm_rows(x_prev, weights["ln_f.weight"])
    
    # LM Head
    logits = linear_no_bias(final, weights["lm_head.weight"])
    
    return {
        "x0": [token_emb[token_id] for token_id in input_ids],
        "logits": logits
    }



def argmax(values):
    return max(range(len(values)), key=lambda idx: values[idx])


def next_token(tokens, vocab, id_to_token, weights):
    input_ids = encode_prompt(tokens, vocab)
    result = tiny_lm_forward(input_ids, weights)
    token_id = argmax(result["logits"][-1])
    return token_id, id_to_token[token_id], result


def top_tokens(logits, id_to_token, limit=5):
    top_ids = sorted(range(len(logits)), key=lambda idx: logits[idx], reverse=True)[:limit]
    return [(idx, id_to_token[idx], logits[idx]) for idx in top_ids]


def pad_rows(rows, *, row_count=4, width=4):
    padded = []
    for row in rows[:row_count]:
        padded.append([lpu.to_f32(value) for value in row[:width]] + [0.0] * max(0, width - len(row)))
    while len(padded) < row_count:
        padded.append([0.0 for _ in range(width)])
    return padded


def fp8_quantize_matrix(matrix):
    bits = [[lpu.fp8_e5m2_bits(value) for value in row] for row in matrix]
    decoded = [[lpu.fp8_e5m2_to_f32(value) for value in row] for row in bits]
    return bits, decoded


def q1_7_bits(value):
    quantized = int(round(float(value) * 128.0))
    quantized = max(-128, min(127, quantized))
    return quantized & 0xFF


def q1_7_to_f32(byte_value):
    signed_value = int(byte_value) & 0xFF
    if signed_value & 0x80:
        signed_value -= 256
    return signed_value / 128.0


def pack_fp32_row(row):
    word = 0
    for idx, value in enumerate(row):
        word |= lpu.f32_bits(value) << (32 * idx)
    return word


def unpack_fp8_word(word):
    return [(word >> (8 * idx)) & 0xFF for idx in range(4)]


def rope_rows_fp32(data, cos_bits, sin_bits):
    out = [0.0 for _ in range(4)]
    for pair in range(2):
        even = 2 * pair
        odd = even + 1
        x_even = lpu.to_f32(data[even])
        x_odd = lpu.to_f32(data[odd])
        cos_value = q1_7_to_f32(cos_bits[even])
        sin_value = q1_7_to_f32(sin_bits[even])
        out[even] = lpu.to_f32(
            lpu.to_f32(x_even * cos_value) - lpu.to_f32(x_odd * sin_value)
        )
        out[odd] = lpu.to_f32(
            lpu.to_f32(x_even * sin_value) + lpu.to_f32(x_odd * cos_value)
        )
    return out


def rope_rows_fp32_8(data, cos_bits, sin_bits):
    out = [0.0 for _ in range(8)]
    for pair in range(4):
        even = 2 * pair
        odd = even + 1
        x_even = lpu.to_f32(data[even])
        x_odd = lpu.to_f32(data[odd])
        cos_value = q1_7_to_f32(cos_bits[even])
        sin_value = q1_7_to_f32(sin_bits[even])
        out[even] = lpu.to_f32(
            lpu.to_f32(x_even * cos_value) - lpu.to_f32(x_odd * sin_value)
        )
        out[odd] = lpu.to_f32(
            lpu.to_f32(x_even * sin_value) + lpu.to_f32(x_odd * cos_value)
        )
    return out


def rope_rows_fixed32_8(data, cos_bytes, sin_bytes):
    """
    Fixed-point 8-lane RoPE (32-bit signed data, 8-bit signed Q1.7 cos/sin).
    Computes pairwise rotations with arithmetic shift by 7 (>>> 7).
    """
    out = [0 for _ in range(8)]
    for pair in range(4):
        even = 2 * pair
        odd = even + 1
        x_even = int(data[even])
        x_odd = int(data[odd])

        c_val = cos_bytes[even] if isinstance(cos_bytes[even], int) else int(cos_bytes[even])
        if c_val > 127:
            c_val -= 256
        s_val = sin_bytes[even] if isinstance(sin_bytes[even], int) else int(sin_bytes[even])
        if s_val > 127:
            s_val -= 256

        prod0 = x_even * c_val
        prod1 = x_odd * s_val
        prod2 = x_even * s_val
        prod3 = x_odd * c_val

        res_even = (prod0 - prod1) >> 7
        res_odd = (prod2 + prod3) >> 7

        out[even] = res_even
        out[odd] = res_odd
    return out



def matrix_close(actual, expected, *, tol=1e-5):
    assert len(actual) == len(expected)
    assert len(actual[0]) == len(expected[0])
    for row_idx, row in enumerate(actual):
        for col_idx, value in enumerate(row):
            diff = abs(value - expected[row_idx][col_idx])
            assert diff <= tol, (
                f"matrix mismatch ({row_idx}, {col_idx}): "
                f"got {value:.8f}, expected {expected[row_idx][col_idx]:.8f}, diff {diff:.8f}"
            )


def mxm_expected_from_fp8_inputs(left_rows, right_rows):
    left_bits, left_decoded = fp8_quantize_matrix(pad_rows(left_rows))
    right_bits, right_decoded = fp8_quantize_matrix(pad_rows(right_rows))
    expected = lpu.matmul_expected_fp32(left_decoded, lpu.transpose_matrix(right_decoded))
    return left_bits, right_bits, expected


async def run_lpu_mxm_tile(
    dut,
    *,
    left_rows,
    right_rows,
    label,
    mem0_base=0,
    mem1_base=64,
):
    left_bits, right_bits, expected = mxm_expected_from_fp8_inputs(left_rows, right_rows)

    for k_idx in range(4):
        lpu.preload_mem0_word(
            dut,
            addr=mem0_base + k_idx,
            values=[left_bits[row_idx][k_idx] for row_idx in range(4)],
        )
        lpu.preload_mem1_word(
            dut,
            addr=mem1_base + k_idx,
            values=[right_bits[row_idx][k_idx] for row_idx in range(4)],
        )

    program = [lpu.build_instruction(mxm_clear=1, **FP_CTRL)]
    for k_idx in range(4):
        lpu.append_mxm_weight_row_load_from_mem1(program, addr=mem1_base + k_idx)
        lpu.append_mxm_input_column_load_from_mem0(program, addr=mem0_base + k_idx)
        program.append(lpu.build_instruction(mxm_start=1, **FP_CTRL))
        for _ in range(4):
            program.append(lpu.build_instruction(**FP_CTRL))
    program.extend([lpu.build_instruction(**FP_CTRL), lpu.build_instruction(**FP_CTRL)])

    await lpu.run_lpu_program(dut, program, extra_cycles=24)
    observed_bits = lpu.read_mxm_matrix_bits(dut)
    observed = [
        [lpu.bits_to_f32(observed_bits[row_idx][col_idx]) for col_idx in range(4)]
        for row_idx in range(4)
    ]

    for row_idx in range(4):
        for col_idx in range(4):
            expected_bits = lpu.f32_bits(expected[row_idx][col_idx])
            assert observed_bits[row_idx][col_idx] == expected_bits, (
                f"{label} tile ({row_idx}, {col_idx}) mismatch: "
                f"got 0x{observed_bits[row_idx][col_idx]:08x}, "
                f"expected 0x{expected_bits:08x}"
            )

    dut._log.info("%s MXM tile matched FP8->FP32 expected output", label)
    return observed


def pad_rows_width(rows, start_col, *, row_count=4, width=4):
    padded = []
    for row in rows[:row_count]:
        slice_vals = row[start_col : start_col + width]
        padded.append([lpu.to_f32(value) for value in slice_vals] + [0.0] * max(0, width - len(slice_vals)))
    while len(padded) < row_count:
        padded.append([0.0 for _ in range(width)])
    return padded


def mxm_expected_full(left_rows, right_rows):
    left_padded = []
    for r in left_rows[:4]:
        left_padded.append(list(r) + [0.0] * max(0, len(right_rows[0]) - len(r)))
    while len(left_padded) < 4:
        left_padded.append([0.0 for _ in range(len(right_rows[0]))])
        
    right_padded = []
    for r in right_rows[:4]:
        right_padded.append(list(r) + [0.0] * max(0, len(right_rows[0]) - len(r)))
    while len(right_padded) < 4:
        right_padded.append([0.0 for _ in range(len(right_rows[0]))])
        
    K = len(left_padded[0])
    expected = [[0.0 for _ in range(4)] for _ in range(4)]
    
    for start in range(0, K, 4):
        l_chunk = [r[start:start+4] + [0.0] * (4 - len(r[start:start+4])) for r in left_padded]
        r_chunk = [r[start:start+4] + [0.0] * (4 - len(r[start:start+4])) for r in right_padded]
        l_bits, l_dec = fp8_quantize_matrix(l_chunk)
        r_bits, r_dec = fp8_quantize_matrix(r_chunk)
        chunk_expected = lpu.matmul_expected_fp32(l_dec, lpu.transpose_matrix(r_dec))
        for r_idx in range(4):
            for c_idx in range(4):
                expected[r_idx][c_idx] = lpu.to_f32(expected[r_idx][c_idx] + chunk_expected[r_idx][c_idx])
                
    return expected


async def run_lpu_mxm_tile_full(
    dut,
    *,
    left_rows,
    right_rows,
    label,
    mem0_base=0,
    mem1_base=64,
):
    K = max(len(left_rows[0]), len(right_rows[0]))
    num_chunks = (K + 3) // 4
    
    for chunk_idx in range(num_chunks):
        start_col = chunk_idx * 4
        l_pad = pad_rows_width(left_rows, start_col)
        r_pad = pad_rows_width(right_rows, start_col)
        l_bits, _ = fp8_quantize_matrix(l_pad)
        r_bits, _ = fp8_quantize_matrix(r_pad)
        
        mem0_offset = mem0_base + chunk_idx * 4
        mem1_offset = mem1_base + chunk_idx * 4
        
        for k_idx in range(4):
            lpu.preload_mem0_word(
                dut,
                addr=mem0_offset + k_idx,
                values=[l_bits[row_idx][k_idx] for row_idx in range(4)],
            )
            lpu.preload_mem1_word(
                dut,
                addr=mem1_offset + k_idx,
                values=[r_bits[row_idx][k_idx] for row_idx in range(4)],
            )
            
    max_chunks_per_pass = 16
    for pass_idx in range(0, num_chunks, max_chunks_per_pass):
        chunk_start = pass_idx
        chunk_end = min(pass_idx + max_chunks_per_pass, num_chunks)
        
        program = []
        if chunk_start == 0:
            program.append(lpu.build_instruction(mxm_clear=1, **FP_CTRL))
            
        for chunk_idx in range(chunk_start, chunk_end):
            mem0_offset = mem0_base + chunk_idx * 4
            mem1_offset = mem1_base + chunk_idx * 4
            for k_idx in range(4):
                lpu.append_mxm_weight_row_load_from_mem1(program, addr=mem1_offset + k_idx)
                lpu.append_mxm_input_column_load_from_mem0(program, addr=mem0_offset + k_idx)
                program.append(lpu.build_instruction(mxm_start=1, **FP_CTRL))
                for _ in range(4):
                    program.append(lpu.build_instruction(**FP_CTRL))
                    
        program.extend([lpu.build_instruction(**FP_CTRL), lpu.build_instruction(**FP_CTRL)])
        lpu.preload_program(dut, program)
        dut.u_lpu.u_icu.pc.value = 0
        await lpu.tick(dut, len(program) + 24)
        
    observed_bits = lpu.read_mxm_matrix_bits(dut)
    observed = [
        [lpu.bits_to_f32(observed_bits[row_idx][col_idx]) for col_idx in range(4)]
        for row_idx in range(4)
    ]
    
    expected = mxm_expected_full(left_rows, right_rows)
    
    for row_idx in range(4):
        for col_idx in range(4):
            expected_bits = lpu.f32_bits(expected[row_idx][col_idx])
            obs_val = lpu.bits_to_f32(observed_bits[row_idx][col_idx])
            exp_val = expected[row_idx][col_idx]
            float_diff = abs(obs_val - exp_val)
            ulp_diff = abs(observed_bits[row_idx][col_idx] - expected_bits)
            
            assert ulp_diff <= 3 or float_diff < 1e-6, (
                f"{label} tile ({row_idx}, {col_idx}) mismatch: "
                f"got 0x{observed_bits[row_idx][col_idx]:08x} ({obs_val:.6f}), "
                f"expected 0x{expected_bits:08x} ({exp_val:.6f}) (ulp diff: {ulp_diff}, float diff: {float_diff})"
            )
            
    if TILE_LOG:
        dut._log.info("%s MXM full tile matched FP8->FP32 expected output", label)
    return observed


def pad_rows_width_tile(rows, start_col, *, row_count=TILE, width=TILE):
    padded = []
    for row in rows[:row_count]:
        slice_vals = row[start_col : start_col + width]
        padded.append(
            [lpu.to_f32(value) for value in slice_vals]
            + [0.0] * max(0, width - len(slice_vals))
        )
    while len(padded) < row_count:
        padded.append([0.0 for _ in range(width)])
    return padded


def transpose_matrix_tile(matrix):
    return [[matrix[row][col] for row in range(TILE)] for col in range(TILE)]


def matmul_expected_fp32_tile(a_matrix, b_matrix):
    expected = [[0.0 for _ in range(TILE)] for _ in range(TILE)]
    for row in range(TILE):
        for col in range(TILE):
            acc = 0.0
            for k_idx in range(TILE):
                product = lpu.to_f32(
                    lpu.to_f32(a_matrix[row][k_idx]) * lpu.to_f32(b_matrix[k_idx][col])
                )
                acc = lpu.to_f32(acc + product)
            expected[row][col] = acc
    return expected


def mxm_expected_full_8(left_rows, right_rows):
    k_len = max(
        max((len(row) for row in left_rows), default=0),
        max((len(row) for row in right_rows), default=0),
    )
    expected = [[0.0 for _ in range(TILE)] for _ in range(TILE)]

    for start in range(0, k_len, TILE):
        l_chunk = pad_rows_width_tile(left_rows, start)
        r_chunk = pad_rows_width_tile(right_rows, start)
        _, l_decoded = fp8_quantize_matrix(l_chunk)
        _, r_decoded = fp8_quantize_matrix(r_chunk)
        chunk_expected = matmul_expected_fp32_tile(l_decoded, transpose_matrix_tile(r_decoded))
        for row_idx in range(TILE):
            for col_idx in range(TILE):
                expected[row_idx][col_idx] = lpu.to_f32(
                    expected[row_idx][col_idx] + chunk_expected[row_idx][col_idx]
                )

    return expected


def read_mxm_matrix_bits_8(dut):
    return [
        [
            int(getattr(dut, f"mxm_out_{row_idx}{col_idx}_dbg").value) & 0xFFFFFFFF
            for col_idx in range(TILE)
        ]
        for row_idx in range(TILE)
    ]


async def run_lpu_mxm_tile_full_8(
    dut,
    *,
    left_rows,
    right_rows,
    label,
    mem0_base=0,
    mem1_base=512,
):
    k_len = max(
        max((len(row) for row in left_rows), default=0),
        max((len(row) for row in right_rows), default=0),
    )
    num_chunks = (k_len + TILE - 1) // TILE

    for chunk_idx in range(num_chunks):
        start_col = chunk_idx * TILE
        l_pad = pad_rows_width_tile(left_rows, start_col)
        r_pad = pad_rows_width_tile(right_rows, start_col)
        l_bits, _ = fp8_quantize_matrix(l_pad)
        r_bits, _ = fp8_quantize_matrix(r_pad)

        mem0_offset = mem0_base + chunk_idx * TILE
        mem1_offset = mem1_base + chunk_idx * TILE
        for k_idx in range(TILE):
            lpu.preload_mem0_word(
                dut,
                addr=mem0_offset + k_idx,
                values=[l_bits[row_idx][k_idx] for row_idx in range(TILE)],
            )
            lpu.preload_mem1_word(
                dut,
                addr=mem1_offset + k_idx,
                values=[r_bits[row_idx][k_idx] for row_idx in range(TILE)],
            )

    await lpu.reset_dut(dut)

    max_chunks_per_pass = 12
    for pass_idx in range(0, num_chunks, max_chunks_per_pass):
        chunk_start = pass_idx
        chunk_end = min(pass_idx + max_chunks_per_pass, num_chunks)

        program = []
        if chunk_start == 0:
            program.append(lpu.build_instruction(mxm_clear=1, **FP_CTRL))

        for chunk_idx in range(chunk_start, chunk_end):
            mem0_offset = mem0_base + chunk_idx * TILE
            mem1_offset = mem1_base + chunk_idx * TILE
            for k_idx in range(TILE):
                lpu.append_mxm_weight_row_load_from_mem1(program, addr=mem1_offset + k_idx)
                lpu.append_mxm_input_column_load_from_mem0(program, addr=mem0_offset + k_idx)
                program.append(lpu.build_instruction(mxm_start=1, **FP_CTRL))
                for _ in range(4):
                    program.append(lpu.build_instruction(**FP_CTRL))

        program.extend([lpu.build_instruction(**FP_CTRL), lpu.build_instruction(**FP_CTRL)])
        lpu.preload_program(dut, program)
        dut.u_lpu.u_icu.pc.value = 0
        await lpu.tick(dut, len(program) + 24)

    observed_bits = read_mxm_matrix_bits_8(dut)
    observed = [
        [lpu.bits_to_f32(observed_bits[row_idx][col_idx]) for col_idx in range(TILE)]
        for row_idx in range(TILE)
    ]
    expected = mxm_expected_full_8(left_rows, right_rows)

    for row_idx in range(TILE):
        for col_idx in range(TILE):
            expected_bits = lpu.f32_bits(expected[row_idx][col_idx])
            obs_val = lpu.bits_to_f32(observed_bits[row_idx][col_idx])
            exp_val = expected[row_idx][col_idx]
            float_diff = abs(obs_val - exp_val)
            ulp_diff = abs(observed_bits[row_idx][col_idx] - expected_bits)
            assert ulp_diff <= 3 or float_diff < 1e-6, (
                f"{label} 8x8 tile ({row_idx}, {col_idx}) mismatch: "
                f"got 0x{observed_bits[row_idx][col_idx]:08x} ({obs_val:.6f}), "
                f"expected 0x{expected_bits:08x} ({exp_val:.6f}) "
                f"(ulp diff: {ulp_diff}, float diff: {float_diff})"
            )

    if TILE_LOG:
        dut._log.info("%s MXM 8x8 full tile matched FP8->FP32 expected output", label)
    return observed


async def run_forced_vxm_row(
    dut,
    *,
    data,
    vxm_ctrl,
    fp_quant_mode=1,
    bias=None,
    gamma=None,
    beta=None,
    layernorm_en=0,
    rope_en=0,
    rope_cos_bits=None,
    rope_sin_bits=None,
    residual_op=lpu.VXM_RES_PASS,
    reset=True,
):
    if reset:
        await lpu.reset_dut(dut)
    else:
        dut.u_lpu.u_vxm.in_valid.value = Force(0)
        dut.u_lpu.u_vxm.out_ready.value = Force(1)
        for _ in range(8):
            if not int(dut.u_lpu.u_vxm.out_valid.value):
                break
            await lpu.tick(dut, 1)

    if bias is not None:
        dut.u_lpu.vxm_bias_reg.value = pack_fp32_row(bias)
    if gamma is not None:
        dut.u_lpu.vxm_rmsnorm_gamma_reg.value = pack_fp32_row(gamma)
    if beta is not None:
        dut.u_lpu.vxm_rmsnorm_beta_reg.value = pack_fp32_row(beta)
    if rope_cos_bits is not None:
        dut.u_lpu.vxm_rope_cos_q1_7_reg.value = lpu.pack_bytes(rope_cos_bits)
    if rope_sin_bits is not None:
        dut.u_lpu.vxm_rope_sin_q1_7_reg.value = lpu.pack_bytes(rope_sin_bits)

    dut.u_lpu.u_vxm.stream_in_data.value = Force(pack_fp32_row(data))
    dut.u_lpu.u_vxm.vxm_ctrl.value = Force(vxm_ctrl)
    dut.u_lpu.u_vxm.fp_quant_mode.value = Force(fp_quant_mode)
    dut.u_lpu.u_vxm.rope_en.value = Force(rope_en)
    dut.u_lpu.u_vxm.residual_op.value = Force(residual_op)
    dut.u_lpu.u_vxm.rmsnorm_bypass.value = Force(0 if layernorm_en else 1)
    dut.u_lpu.u_vxm.in_valid.value = Force(1)
    dut.u_lpu.u_vxm.out_ready.value = Force(1)

    await lpu.tick(dut, 1)
    dut.u_lpu.u_vxm.in_valid.value = Force(0)

    for _ in range(160):
        await lpu.tick(dut, 1)
        if int(dut.u_lpu.u_vxm.out_valid.value):
            row_word = int(dut.u_lpu.u_vxm.stream_out.value) & 0xFFFFFFFF
            scale_word = int(dut.u_lpu.u_vxm.stream_out_scale.value) & 0xFFFFFFFF
            dut.u_lpu.u_vxm.stream_in_data.value = Release()
            dut.u_lpu.u_vxm.vxm_ctrl.value = Release()
            dut.u_lpu.u_vxm.fp_quant_mode.value = Release()
            dut.u_lpu.u_vxm.rope_en.value = Release()
            dut.u_lpu.u_vxm.residual_op.value = Release()
            dut.u_lpu.u_vxm.rmsnorm_bypass.value = Release()
            dut.u_lpu.u_vxm.in_valid.value = Release()
            dut.u_lpu.u_vxm.out_ready.value = Release()
            return row_word, scale_word

    dut.u_lpu.u_vxm.stream_in_data.value = Release()
    dut.u_lpu.u_vxm.vxm_ctrl.value = Release()
    dut.u_lpu.u_vxm.fp_quant_mode.value = Release()
    dut.u_lpu.u_vxm.rope_en.value = Release()
    dut.u_lpu.u_vxm.residual_op.value = Release()
    dut.u_lpu.u_vxm.rmsnorm_bypass.value = Release()
    dut.u_lpu.u_vxm.in_valid.value = Release()
    dut.u_lpu.u_vxm.out_ready.value = Release()
    raise AssertionError("VXM did not produce an output row")


async def drive_forced_vxm_residual_op(
    dut,
    *,
    data,
    residual_op,
    reset=False,
):
    if reset:
        await lpu.reset_dut(dut)

    dut.u_lpu.u_vxm.stream_in_data.value = Force(pack_fp32_row(data))
    dut.u_lpu.u_vxm.vxm_ctrl.value = Force(0)
    dut.u_lpu.u_vxm.fp_quant_mode.value = Force(1)
    dut.u_lpu.u_vxm.rope_en.value = Force(0)
    dut.u_lpu.u_vxm.residual_op.value = Force(residual_op)
    dut.u_lpu.u_vxm.rmsnorm_bypass.value = Force(1)
    dut.u_lpu.u_vxm.in_valid.value = Force(1)
    dut.u_lpu.u_vxm.out_ready.value = Force(1)

    await lpu.tick(dut, 1)
    dut.u_lpu.u_vxm.in_valid.value = Force(0)

    for _ in range(120):
        await lpu.tick(dut, 1)
        if int(dut.u_lpu.u_vxm.residual_done.value):
            dut.u_lpu.u_vxm.stream_in_data.value = Release()
            dut.u_lpu.u_vxm.vxm_ctrl.value = Release()
            dut.u_lpu.u_vxm.fp_quant_mode.value = Release()
            dut.u_lpu.u_vxm.rope_en.value = Release()
            dut.u_lpu.u_vxm.residual_op.value = Release()
            dut.u_lpu.u_vxm.rmsnorm_bypass.value = Release()
            dut.u_lpu.u_vxm.in_valid.value = Release()
            dut.u_lpu.u_vxm.out_ready.value = Release()
            return

    dut.u_lpu.u_vxm.stream_in_data.value = Release()
    dut.u_lpu.u_vxm.vxm_ctrl.value = Release()
    dut.u_lpu.u_vxm.fp_quant_mode.value = Release()
    dut.u_lpu.u_vxm.rope_en.value = Release()
    dut.u_lpu.u_vxm.residual_op.value = Release()
    dut.u_lpu.u_vxm.rmsnorm_bypass.value = Release()
    dut.u_lpu.u_vxm.in_valid.value = Release()
    dut.u_lpu.u_vxm.out_ready.value = Release()
    raise AssertionError("VXM residual op did not complete")


def token_rows_to_string(tokens):
    return " ".join(tokens)


@cocotb.test()
async def test_tiny_lm_prefill_decode_golden(dut):
    config, vocab, id_to_token, weights = load_tiny_lm_export()
    if config.get("layers", 1) > 1:
        dut._log.info("Skipping 1-layer test case for multi-layer model.")
        return
    assert config == MODEL_CONFIG

    first_id, first_token, first = next_token(PROMPT_PREFILL, vocab, id_to_token, weights)
    second_id, second_token, second = next_token(
        PROMPT_PREFILL + [first_token],
        vocab,
        id_to_token,
        weights,
    )

    assert first_id >= 0 and first_id < config["vocab_size"]
    assert second_id >= 0 and second_id < config["vocab_size"]
    assert first_id == argmax(first["logits"][-1])
    assert second_id == argmax(second["logits"][-1])

    dut._log.info('prefill prompt: "%s"', token_rows_to_string(PROMPT_PREFILL))
    dut._log.info("prefill next-token top5: %s", top_tokens(first["logits"][-1], id_to_token))
    dut._log.info('decode prompt: "%s"', token_rows_to_string(PROMPT_PREFILL + [first_token]))
    dut._log.info("decode next-token top5: %s", top_tokens(second["logits"][-1], id_to_token))
    dut._log.info('golden generation: "%s %s %s"', PROMPT_PREFILL[0], PROMPT_PREFILL[1], first_token)
    dut._log.info('next decoded token: "%s"', second_token)
    await Timer(1, unit="ns")


@cocotb.test()
async def test_lpu_mem_2048_high_address_decode_and_read(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    mem0_addr = 1500
    mem1_addr = 1733
    mem0_bytes = [0xA1, 0xB2, 0xC3, 0xD4]
    mem1_bytes = [0x11, 0x22, 0x33, 0x44]

    lpu.preload_mem0_word(dut, mem0_addr, mem0_bytes)
    lpu.preload_mem1_word(dut, mem1_addr, mem1_bytes)

    program = [
        lpu.build_instruction(mem0_read_en=1, mem0_addr=mem0_addr),
        lpu.build_instruction(
            westbound_sel=lpu.WB_MEM0,
            mem1_read_en=1,
            mem1_addr=mem1_addr,
        ),
        lpu.build_instruction(westbound_sel=lpu.WB_MEM1),
    ]

    lpu.preload_program(dut, program)
    await lpu.reset_dut(dut)

    assert int(dut.u_lpu.mem0_addr.value) == mem0_addr

    await lpu.tick(dut)
    assert int(dut.u_lpu.mem1_addr.value) == mem1_addr
    assert int(dut.u_lpu.westbound_valid.value) == 1
    assert int(dut.u_lpu.westbound_payload.value) == lpu.pack_bytes(mem0_bytes)

    await lpu.tick(dut)
    assert int(dut.u_lpu.westbound_valid.value) == 1
    assert int(dut.u_lpu.westbound_payload.value) == lpu.pack_bytes(mem1_bytes)


@cocotb.test()
async def test_lpu_vxm_hardware_relu_softmax_layernorm_paths(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    relu_data = [-1.35, 0.49, 2.10, -0.75]
    relu_bias = [0.25, -0.90, 0.35, 1.40]
    relu_expected = [
        lpu.to_f32(max(0.0, lpu.to_f32(data + bias)))
        for data, bias in zip(relu_data, relu_bias)
    ]
    relu_expected_bits, relu_expected_scale = lpu.regular_fp8_row_quant_expected(relu_expected)
    relu_word, relu_scale = await run_forced_vxm_row(
        dut,
        data=relu_data,
        bias=relu_bias,
        vxm_ctrl=0b0011,
        fp_quant_mode=1,
    )
    assert unpack_fp8_word(relu_word) == relu_expected_bits
    assert (relu_scale & 0xFF) == (relu_expected_scale & 0xFF)
    dut._log.info("hardware VXM FP bias+ReLU quantized row: %s scale=%d", unpack_fp8_word(relu_word), relu_expected_scale)

    softmax_data = [0.35, -0.49, 1.20, -0.85]
    softmax_expected_bits = lpu.softmax_fp8_quant_expected(softmax_data)
    softmax_word, softmax_scale = await run_forced_vxm_row(
        dut,
        data=softmax_data,
        vxm_ctrl=0b1000,
        fp_quant_mode=1,
    )
    assert unpack_fp8_word(softmax_word) == softmax_expected_bits
    assert softmax_scale == 0
    dut._log.info("hardware VXM FP softmax quantized row: %s", unpack_fp8_word(softmax_word))

    ln_data = [0.35, -0.49, 1.25, -0.75]
    ln_gamma = [1.0, 0.5, 1.25, 0.75]
    ln_beta = [0.10, -0.20, 0.0, 0.35]
    ln_expected = rmsnorm_rows([ln_data], ln_gamma)[0]
    ln_expected_bits, ln_expected_scale = lpu.regular_fp8_row_quant_expected(ln_expected)
    ln_word, ln_scale = await run_forced_vxm_row(
        dut,
        data=ln_data,
        gamma=ln_gamma,
        beta=ln_beta,
        vxm_ctrl=0b0000,
        fp_quant_mode=1,
        layernorm_en=1,
    )
    assert unpack_fp8_word(ln_word) == ln_expected_bits
    assert (ln_scale & 0xFF) == (ln_expected_scale & 0xFF)
    dut._log.info("hardware VXM programmable RMSNorm quantized row: %s scale=%d", unpack_fp8_word(ln_word), ln_expected_scale)

    rope_data = [0.35, -0.49, 1.20, -0.85]
    cos_bits = [
        lpu.fp8_e5m2_bits(0.96),
        lpu.fp8_e5m2_bits(0.96),
        lpu.fp8_e5m2_bits(0.78),
        lpu.fp8_e5m2_bits(0.78),
    ]
    sin_bits = [
        lpu.fp8_e5m2_bits(0.29),
        lpu.fp8_e5m2_bits(0.29),
        lpu.fp8_e5m2_bits(-0.63),
        lpu.fp8_e5m2_bits(-0.63),
    ]

    lpu.preload_mem0_word(dut, addr=12, values=cos_bits)
    lpu.preload_mem1_word(dut, addr=18, values=sin_bits)
    load_rope_operands = [
        lpu.build_instruction(mem0_read_en=1, mem0_addr=12),
        lpu.build_instruction(
            eastbound_sel=lpu.EB_MEM0,
            eastbound_consumer_sel=lpu.EC_VXM,
            vxm_operand_sel=lpu.VXM_OPERAND_ROPE_COS,
        ),
        lpu.build_instruction(mem1_read_en=1, mem1_addr=18),
        lpu.build_instruction(
            westbound_sel=lpu.WB_MEM1,
            westbound_consumer_sel=lpu.WC_VXM,
            vxm_operand_sel=lpu.VXM_OPERAND_ROPE_SIN,
        ),
    ]
    await lpu.run_lpu_program(dut, load_rope_operands, extra_cycles=6)
    assert int(dut.u_lpu.vxm_rope_cos_q1_7_reg.value) == lpu.pack_bytes(cos_bits)
    assert int(dut.u_lpu.vxm_rope_sin_q1_7_reg.value) == lpu.pack_bytes(sin_bits)

    rope_expected = rope_rows_fp32(rope_data, cos_bits, sin_bits)
    rope_expected_bits, rope_expected_scale = lpu.regular_fp8_row_quant_expected(rope_expected)
    rope_word, rope_scale = await run_forced_vxm_row(
        dut,
        data=rope_data,
        vxm_ctrl=0b0000,
        fp_quant_mode=1,
        rope_en=1,
        rope_cos_bits=cos_bits,
        rope_sin_bits=sin_bits,
    )
    assert unpack_fp8_word(rope_word) == rope_expected_bits, (
        f"RoPE output mismatch: got {unpack_fp8_word(rope_word)}, "
        f"expected {rope_expected_bits}, "
        f"rope_out=0x{int(dut.u_lpu.u_vxm.rope_out.value):032x}, "
        f"rope_result=0x{int(dut.u_lpu.u_vxm.rope_result_reg.value):032x}, "
        f"rope_start={int(dut.u_lpu.u_vxm.rope_start.value)}, "
        f"rope_done={int(dut.u_lpu.u_vxm.rope_done.value)}, "
        f"rope_busy={int(dut.u_lpu.u_vxm.rope_busy.value)}, "
        f"rope_state={int(dut.u_lpu.u_vxm.rope_inst.state_q.value)}, "
        f"mux_valid={int(dut.u_lpu.u_vxm.mux_valid.value)}, "
        f"rope_result_valid={int(dut.u_lpu.u_vxm.rope_result_valid.value)}, "
        f"cos=0x{int(dut.u_lpu.vxm_rope_cos_q1_7_reg.value):08x}, "
        f"sin=0x{int(dut.u_lpu.vxm_rope_sin_q1_7_reg.value):08x}"
    )
    assert (rope_scale & 0xFF) == (rope_expected_scale & 0xFF)
    dut._log.info(
        "hardware VXM RoPE quantized row: %s scale=%d",
        unpack_fp8_word(rope_word),
        rope_expected_scale,
    )

    residual_base = [0.37, -0.82, 1.13, -1.41]
    residual_delta = [-0.29, 0.44, -0.61, 0.95]
    residual_expected = [
        lpu.to_f32(base + delta)
        for base, delta in zip(residual_base, residual_delta)
    ]
    residual_expected_bits, residual_expected_scale = lpu.regular_fp8_row_quant_expected(residual_expected)

    await drive_forced_vxm_residual_op(
        dut,
        data=residual_base,
        residual_op=lpu.VXM_RES_LOAD,
        reset=True,
    )
    await drive_forced_vxm_residual_op(
        dut,
        data=residual_delta,
        residual_op=lpu.VXM_RES_ADD,
    )
    residual_word, residual_scale = await run_forced_vxm_row(
        dut,
        data=[0.0, 0.0, 0.0, 0.0],
        vxm_ctrl=0b0000,
        fp_quant_mode=1,
        residual_op=lpu.VXM_RES_EMIT,
        reset=False,
    )
    assert unpack_fp8_word(residual_word) == residual_expected_bits
    assert (residual_scale & 0xFF) == (residual_expected_scale & 0xFF)
    dut._log.info(
        "hardware VXM residual add quantized row: %s scale=%d",
        unpack_fp8_word(residual_word),
        residual_expected_scale,
    )


@cocotb.test()
async def test_tiny_lm_vocab_prompt_probe(dut):
    config, vocab, id_to_token, weights = load_tiny_lm_export()
    if config.get("layers", 1) > 1:
        dut._log.info("Skipping 1-layer test case for multi-layer model.")
        return
    assert config == MODEL_CONFIG

    token_id, token, result = next_token(PROMPT_PROBE, vocab, id_to_token, weights)
    assert token_id == argmax(result["logits"][-1])
    assert not token.startswith("<unused_")

    dut._log.info('probe prompt: "%s"', token_rows_to_string(PROMPT_PROBE))
    dut._log.info("probe next-token top10: %s", top_tokens(result["logits"][-1], id_to_token, limit=10))
    dut._log.info('probe generation: "%s %s"', token_rows_to_string(PROMPT_PROBE), token)
    await Timer(1, unit="ns")


@cocotb.test()
async def test_lpu_vocab_prompt_timing(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    config, vocab, id_to_token, weights = load_tiny_lm_export()
    if config.get("layers", 1) > 1:
        dut._log.info("Skipping 1-layer test case for multi-layer model.")
        return
    assert config == MODEL_CONFIG

    prompt_ids = encode_prompt(PROMPT_PROBE, vocab)
    golden = tiny_lm_forward(prompt_ids, weights)
    golden_token = id_to_token[argmax(golden["logits"][-1])]

    start_ns = get_sim_time(unit="ns")

    for projection, weight_name in [
        ("vocab prompt Q projection", "blocks.0.attn.q_proj.weight"),
        ("vocab prompt K projection", "blocks.0.attn.k_proj.weight"),
        ("vocab prompt V projection", "blocks.0.attn.v_proj.weight"),
    ]:
        await run_lpu_mxm_tile(
            dut,
            left_rows=golden["ln1"],
            right_rows=weights[weight_name],
            label=projection,
        )

    await run_lpu_mxm_tile(
        dut,
        left_rows=golden["q"],
        right_rows=golden["k"],
        label="vocab prompt causal attention Q @ K^T raw scores",
    )

    await run_lpu_mxm_tile(
        dut,
        left_rows=golden["probs"],
        right_rows=transpose(golden["v"]),
        label="vocab prompt attention probabilities @ V",
    )

    await run_lpu_mxm_tile(
        dut,
        left_rows=golden["attn"],
        right_rows=weights["blocks.0.attn.out_proj.weight"],
        label="vocab prompt attention output projection",
    )

    for start in range(0, MODEL_CONFIG["ffn_dim"], 4):
        await run_lpu_mxm_tile(
            dut,
            left_rows=golden["ln2"],
            right_rows=weights["blocks.0.ffn.0.weight"][start:start + 4],
            label=f"vocab prompt FFN W1 tile {start}:{start + 4}",
        )

    for start in range(0, MODEL_CONFIG["ffn_dim"], 4):
        hidden_chunk = [row[start:start + 4] for row in golden["ffn_hidden"]]
        w2_chunk = [row[start:start + 4] for row in weights["blocks.0.ffn.2.weight"]]
        await run_lpu_mxm_tile(
            dut,
            left_rows=hidden_chunk,
            right_rows=w2_chunk,
            label=f"vocab prompt FFN W2 partial tile {start}:{start + 4}",
        )

    last_hidden = golden["final"][-1]
    hw_logits = [0.0 for _ in range(MODEL_CONFIG["vocab_size"])]
    for vocab_start in range(0, MODEL_CONFIG["vocab_size"], 4):
        observed = await run_lpu_mxm_tile(
            dut,
            left_rows=[last_hidden],
            right_rows=weights["lm_head.weight"][vocab_start:vocab_start + 4],
            label=f"vocab prompt LM head tile {vocab_start}:{vocab_start + 4}",
        )
        for lane in range(4):
            token_id = vocab_start + lane
            hw_logits[token_id] = lpu.to_f32(observed[0][lane] + weights["lm_head.bias"][token_id])

    end_ns = get_sim_time(unit="ns")
    elapsed_ns = end_ns - start_ns
    elapsed_cycles = elapsed_ns / 10.0
    hw_next_id = argmax(hw_logits)
    hw_next_token = id_to_token[hw_next_id]

    assert hw_next_token == golden_token

    dut._log.info('LPU-backed timing prompt: "%s"', token_rows_to_string(PROMPT_PROBE))
    dut._log.info("golden vocab prompt top5: %s", top_tokens(golden["logits"][-1], id_to_token))
    dut._log.info("LPU-backed quantized vocab prompt top5: %s", top_tokens(hw_logits, id_to_token))
    dut._log.info('LPU-backed next token for "%s" is "%s"', token_rows_to_string(PROMPT_PROBE), hw_next_token)
    dut._log.info("LPU-backed validation latency: %.0f ns = %.1f cycles at 10 ns clock", elapsed_ns, elapsed_cycles)


@cocotb.test()
async def test_lpu_tiny_lm_prefill_decode_tiles_and_lm_head(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    config, vocab, id_to_token, weights = load_tiny_lm_export()
    if config.get("layers", 1) > 1:
        dut._log.info("Skipping 1-layer test case for multi-layer model.")
        return
    assert config == MODEL_CONFIG

    prefill_ids = encode_prompt(PROMPT_PREFILL, vocab)
    prefill = tiny_lm_forward(prefill_ids, weights)
    prefill_token = id_to_token[argmax(prefill["logits"][-1])]
    assert prefill_token == "king"

    decode_ids = encode_prompt(PROMPT_DECODE, vocab)
    decode = tiny_lm_forward(decode_ids, weights)
    decode_token = id_to_token[argmax(decode["logits"][-1])]
    assert decode_token == "."

    dut._log.info("LPU test uses prompt tokens: %s", PROMPT_DECODE)
    dut._log.info("Residual adds and causal mask are runtime/TB steps in this test.")
    dut._log.info("LPU MXM tiles execute the trained FP datapath matrix products.")

    for projection, weight_name in [
        ("Q projection", "blocks.0.attn.q_proj.weight"),
        ("K projection", "blocks.0.attn.k_proj.weight"),
        ("V projection", "blocks.0.attn.v_proj.weight"),
    ]:
        observed = await run_lpu_mxm_tile(
            dut,
            left_rows=decode["ln1"],
            right_rows=weights[weight_name],
            label=projection,
        )
        _, _, expected = mxm_expected_from_fp8_inputs(decode["ln1"], weights[weight_name])
        matrix_close(observed, expected)
        dut._log.info("%s hardware-compatible output rows: %s", projection, observed[:3])

    scores_raw = await run_lpu_mxm_tile(
        dut,
        left_rows=decode["q"],
        right_rows=decode["k"],
        label="causal attention Q @ K^T raw scores",
    )
    dut._log.info("raw QK scores before TB causal mask/scale: %s", [row[:3] for row in scores_raw[:3]])

    v_by_hidden = transpose(decode["v"])
    attn = await run_lpu_mxm_tile(
        dut,
        left_rows=decode["probs"],
        right_rows=v_by_hidden,
        label="attention probabilities @ V",
    )
    dut._log.info("attention @ V hardware-compatible rows: %s", attn[:3])

    observed = await run_lpu_mxm_tile(
        dut,
        left_rows=decode["attn"],
        right_rows=weights["blocks.0.attn.out_proj.weight"],
        label="attention output projection",
    )
    dut._log.info("attention output projection no-bias rows: %s", observed[:3])

    for start in range(0, MODEL_CONFIG["ffn_dim"], 4):
        observed = await run_lpu_mxm_tile(
            dut,
            left_rows=decode["ln2"],
            right_rows=weights["blocks.0.ffn.0.weight"][start:start + 4],
            label=f"FFN W1 tile {start}:{start + 4}",
        )
        dut._log.info("FFN W1 tile %d:%d no-bias rows: %s", start, start + 4, observed[:3])

    w2 = weights["blocks.0.ffn.2.weight"]
    ffn_w2_partials = []
    for start in range(0, MODEL_CONFIG["ffn_dim"], 4):
        hidden_chunk = [row[start:start + 4] for row in decode["ffn_hidden"]]
        w2_chunk = [row[start:start + 4] for row in w2]
        observed = await run_lpu_mxm_tile(
            dut,
            left_rows=hidden_chunk,
            right_rows=w2_chunk,
            label=f"FFN W2 partial tile {start}:{start + 4}",
        )
        ffn_w2_partials.append(observed)
        dut._log.info("FFN W2 partial tile %d:%d rows: %s", start, start + 4, observed[:3])

    ffn_w2_sum = [[0.0 for _ in range(4)] for _ in range(3)]
    for partial in ffn_w2_partials:
        for row_idx in range(3):
            for col_idx in range(4):
                ffn_w2_sum[row_idx][col_idx] = lpu.to_f32(ffn_w2_sum[row_idx][col_idx] + partial[row_idx][col_idx])
    dut._log.info("summed FFN W2 no-bias hardware-compatible rows: %s", ffn_w2_sum)

    last_hidden = decode["final"][-1]
    hw_logits = [0.0 for _ in range(MODEL_CONFIG["vocab_size"])]
    for vocab_start in range(0, MODEL_CONFIG["vocab_size"], 4):
        observed = await run_lpu_mxm_tile(
            dut,
            left_rows=[last_hidden],
            right_rows=weights["lm_head.weight"][vocab_start:vocab_start + 4],
            label=f"LM head tile {vocab_start}:{vocab_start + 4}",
        )
        for lane in range(4):
            token_id = vocab_start + lane
            hw_logits[token_id] = lpu.to_f32(observed[0][lane] + weights["lm_head.bias"][token_id])

    hw_next_id = argmax(hw_logits)
    hw_next_token = id_to_token[hw_next_id]
    assert hw_next_token == "."

    dut._log.info("golden decode next-token top5: %s", top_tokens(decode["logits"][-1], id_to_token))
    dut._log.info("LPU-backed quantized LM-head top5: %s", top_tokens(hw_logits, id_to_token))
    dut._log.info('LPU-backed next token for "%s" is "%s"', token_rows_to_string(PROMPT_DECODE), hw_next_token)


@cocotb.task.bridge
def get_user_input(prompt: str) -> str:
    import sys
    try:
        sys.stdout.flush()
        return input(prompt)
    except (KeyboardInterrupt, EOFError):
        return "exit"
    except Exception:
        return "exit"


async def drive_rope_registers(dut, cos_bits, sin_bits):
    dut.u_lpu.vxm_rope_cos_q1_7_reg.value = pack_bytes_8(cos_bits)
    dut.u_lpu.vxm_rope_sin_q1_7_reg.value = pack_bytes_8(sin_bits)


def pack_fp32_row_8(row):
    padded = list(row) + [0.0] * (8 - len(row))
    word = 0
    for idx, value in enumerate(padded):
        word |= lpu.f32_bits(value) << (32 * idx)
    return word


def unpack_fp32_row_8(word):
    return [lpu.bits_to_f32((word >> (32 * idx)) & 0xFFFF_FFFF) for idx in range(8)]


def pack_bytes_8(values):
    padded = list(values) + [0] * (8 - len(values))
    word = 0
    for idx, value in enumerate(padded):
        word |= (value & 0xFF) << (8 * idx)
    return word


async def run_forced_vxm_rmsnorm(dut, data, gamma):
    # Set default values to prevent X-propagation
    dut.u_lpu.u_vxm.in_valid.value = Force(0)
    dut.u_lpu.u_vxm.stream_in_data.value = Force(0)
    dut.u_lpu.u_vxm.stream_in_bias.value = Force(0)
    dut.u_lpu.vxm_rmsnorm_gamma_reg.value = Force(0)
    dut.u_lpu.vxm_rmsnorm_beta_reg.value = Force(0)
    dut.u_lpu.u_vxm.rmsnorm_bypass.value = Force(0)
    dut.u_lpu.u_vxm.vxm_ctrl.value = Force(0b0000)
    dut.u_lpu.u_vxm.fp_quant_mode.value = Force(1)
    dut.u_lpu.u_vxm.rope_en.value = Force(0)
    dut.u_lpu.u_vxm.residual_op.value = Force(lpu.VXM_RES_PASS)
    dut.u_lpu.u_vxm.out_ready.value = Force(1)
    
    # Tick to propagate clean reset state
    await lpu.tick(dut, 2)
    
    chunks_in = [data[i*8 : (i+1)*8] for i in range(8)]
    gamma_chunks = [gamma[i*8 : (i+1)*8] for i in range(8)]
    
    # Feed chunks
    for c_idx in range(8):
        dut.u_lpu.u_vxm.stream_in_data.value = Force(pack_fp32_row_8(chunks_in[c_idx]))
        dut.u_lpu.vxm_rmsnorm_gamma_reg.value = Force(pack_fp32_row_8(gamma_chunks[c_idx]))
        dut.u_lpu.vxm_rmsnorm_beta_reg.value = Force(0)
        dut.u_lpu.u_vxm.in_valid.value = Force(1)
        
        ready = False
        for _ in range(100):
            val = dut.u_lpu.u_vxm.in_ready.value
            if str(val) not in ('U', 'X', 'Z') and int(val):
                ready = True
                await lpu.tick(dut, 1)
                break
            await lpu.tick(dut, 1)
        dut.u_lpu.u_vxm.in_valid.value = Force(0)
        if not ready:
            raise AssertionError(f"VXM RMSNorm ready timeout on input chunk {c_idx}")
        
    out_row = []
    for c_idx in range(8):
        found = False
        for _ in range(200):
            val = dut.u_lpu.u_vxm.out_valid.value
            if str(val) not in ('U', 'X', 'Z') and int(val):
                row_word = int(dut.u_lpu.u_vxm.stream_out.value)
                scale_word = int(dut.u_lpu.u_vxm.stream_out_scale.value)
                row_bits = [(row_word >> (8 * idx)) & 0xFF for idx in range(8)]
                scale_exp = scale_word & 0xFF
                if scale_exp & 0x80:
                    scale_exp -= 256
                chunk_floats = lpu.dequantize_regular_fp8_row(row_bits, scale_exp)
                out_row.extend(chunk_floats)
                found = True
                await lpu.tick(dut, 1)
                break
            await lpu.tick(dut, 1)
        if not found:
            raise AssertionError(f"RMSNorm did not produce output chunk {c_idx}")
            
    # Release forces
    dut.u_lpu.u_vxm.in_valid.value = Release()
    dut.u_lpu.u_vxm.stream_in_data.value = Release()
    dut.u_lpu.u_vxm.stream_in_bias.value = Release()
    dut.u_lpu.vxm_rmsnorm_gamma_reg.value = Release()
    dut.u_lpu.vxm_rmsnorm_beta_reg.value = Release()
    dut.u_lpu.u_vxm.rmsnorm_bypass.value = Release()
    dut.u_lpu.u_vxm.vxm_ctrl.value = Release()
    dut.u_lpu.u_vxm.fp_quant_mode.value = Release()
    dut.u_lpu.u_vxm.rope_en.value = Release()
    dut.u_lpu.u_vxm.residual_op.value = Release()
    dut.u_lpu.u_vxm.out_ready.value = Release()
    
    return out_row


async def run_forced_vxm_rope_chunk(dut, data, cos_bits, sin_bits):
    dut.u_lpu.u_vxm.in_valid.value = Force(0)
    dut.u_lpu.u_vxm.stream_in_data.value = Force(0)
    dut.u_lpu.u_vxm.stream_in_bias.value = Force(0)
    dut.u_lpu.vxm_rmsnorm_gamma_reg.value = Force(0)
    dut.u_lpu.vxm_rmsnorm_beta_reg.value = Force(0)
    dut.u_lpu.u_vxm.rmsnorm_bypass.value = Force(1)
    dut.u_lpu.u_vxm.vxm_ctrl.value = Force(0b0000)
    dut.u_lpu.u_vxm.fp_quant_mode.value = Force(1)
    dut.u_lpu.u_vxm.rope_en.value = Force(1)
    dut.u_lpu.u_vxm.residual_op.value = Force(lpu.VXM_RES_PASS)
    dut.u_lpu.u_vxm.out_ready.value = Force(1)
    
    await drive_rope_registers(dut, cos_bits, sin_bits)
    await lpu.tick(dut, 2)
    
    dut.u_lpu.u_vxm.stream_in_data.value = Force(pack_fp32_row_8(data))
    dut.u_lpu.u_vxm.in_valid.value = Force(1)
    
    await lpu.tick(dut, 1)
    dut.u_lpu.u_vxm.in_valid.value = Force(0)
    
    rotated = None
    for _ in range(160):
        val = dut.u_lpu.u_vxm.out_valid.value
        if str(val) not in ('U', 'X', 'Z') and int(val):
            row_word = int(dut.u_lpu.u_vxm.stream_out.value)
            scale_word = int(dut.u_lpu.u_vxm.stream_out_scale.value)
            row_bits = [(row_word >> (8 * idx)) & 0xFF for idx in range(8)]
            scale_exp = scale_word & 0xFF
            if scale_exp & 0x80:
                scale_exp -= 256
            rotated = lpu.dequantize_regular_fp8_row(row_bits, scale_exp)
            await lpu.tick(dut, 1)
            break
        await lpu.tick(dut, 1)
        
    dut.u_lpu.u_vxm.in_valid.value = Release()
    dut.u_lpu.u_vxm.stream_in_data.value = Release()
    dut.u_lpu.u_vxm.stream_in_bias.value = Release()
    dut.u_lpu.vxm_rmsnorm_gamma_reg.value = Release()
    dut.u_lpu.vxm_rmsnorm_beta_reg.value = Release()
    dut.u_lpu.u_vxm.rmsnorm_bypass.value = Release()
    dut.u_lpu.u_vxm.vxm_ctrl.value = Release()
    dut.u_lpu.u_vxm.fp_quant_mode.value = Release()
    dut.u_lpu.u_vxm.rope_en.value = Release()
    dut.u_lpu.u_vxm.residual_op.value = Release()
    dut.u_lpu.u_vxm.out_ready.value = Release()
    
    if rotated is None:
        raise AssertionError("RoPE chunk did not produce an output")
    return rotated


async def run_forced_vxm_relu_chunk(dut, data):
    dut.u_lpu.u_vxm.in_valid.value = Force(0)
    dut.u_lpu.u_vxm.stream_in_data.value = Force(0)
    dut.u_lpu.u_vxm.stream_in_bias.value = Force(0)
    dut.u_lpu.vxm_rmsnorm_gamma_reg.value = Force(0)
    dut.u_lpu.vxm_rmsnorm_beta_reg.value = Force(0)
    dut.u_lpu.u_vxm.rmsnorm_bypass.value = Force(1)
    dut.u_lpu.u_vxm.vxm_ctrl.value = Force(0b0011)
    dut.u_lpu.u_vxm.fp_quant_mode.value = Force(1)
    dut.u_lpu.u_vxm.rope_en.value = Force(0)
    dut.u_lpu.u_vxm.residual_op.value = Force(lpu.VXM_RES_PASS)
    dut.u_lpu.u_vxm.out_ready.value = Force(1)
    
    await lpu.tick(dut, 2)
    
    dut.u_lpu.u_vxm.stream_in_data.value = Force(pack_fp32_row_8(data))
    dut.u_lpu.u_vxm.in_valid.value = Force(1)
    
    await lpu.tick(dut, 1)
    dut.u_lpu.u_vxm.in_valid.value = Force(0)
    
    relu_out = None
    for _ in range(160):
        val = dut.u_lpu.u_vxm.out_valid.value
        if str(val) not in ('U', 'X', 'Z') and int(val):
            row_word = int(dut.u_lpu.u_vxm.stream_out.value)
            scale_word = int(dut.u_lpu.u_vxm.stream_out_scale.value)
            row_bits = [(row_word >> (8 * idx)) & 0xFF for idx in range(8)]
            scale_exp = scale_word & 0xFF
            if scale_exp & 0x80:
                scale_exp -= 256
            relu_out = lpu.dequantize_regular_fp8_row(row_bits, scale_exp)
            await lpu.tick(dut, 1)
            break
        await lpu.tick(dut, 1)
        
    dut.u_lpu.u_vxm.in_valid.value = Release()
    dut.u_lpu.u_vxm.stream_in_data.value = Release()
    dut.u_lpu.u_vxm.stream_in_bias.value = Release()
    dut.u_lpu.vxm_rmsnorm_gamma_reg.value = Release()
    dut.u_lpu.vxm_rmsnorm_beta_reg.value = Release()
    dut.u_lpu.u_vxm.rmsnorm_bypass.value = Release()
    dut.u_lpu.u_vxm.vxm_ctrl.value = Release()
    dut.u_lpu.u_vxm.fp_quant_mode.value = Release()
    dut.u_lpu.u_vxm.rope_en.value = Release()
    dut.u_lpu.u_vxm.residual_op.value = Release()
    dut.u_lpu.u_vxm.out_ready.value = Release()
    
    if relu_out is None:
        raise AssertionError("ReLU chunk did not produce an output")
    return relu_out


async def run_forced_vxm_softmax(dut, data):
    data_512 = list(data) + [-1e30] * (512 - len(data))
    chunks = [data_512[i*8 : (i+1)*8] for i in range(64)]
    
    dut.u_lpu.u_vxm.in_valid.value = Force(0)
    dut.u_lpu.u_vxm.stream_in_data.value = Force(0)
    dut.u_lpu.u_vxm.stream_in_bias.value = Force(0)
    dut.u_lpu.vxm_rmsnorm_gamma_reg.value = Force(0)
    dut.u_lpu.vxm_rmsnorm_beta_reg.value = Force(0)
    dut.u_lpu.u_vxm.rmsnorm_bypass.value = Force(1)
    dut.u_lpu.u_vxm.vxm_ctrl.value = Force(0b1000)
    dut.u_lpu.u_vxm.fp_quant_mode.value = Force(1)
    dut.u_lpu.u_vxm.rope_en.value = Force(0)
    dut.u_lpu.u_vxm.residual_op.value = Force(lpu.VXM_RES_PASS)
    dut.u_lpu.u_vxm.out_ready.value = Force(1)
    
    await lpu.tick(dut, 2)
    
    for c_idx in range(64):
        dut.u_lpu.u_vxm.stream_in_data.value = Force(pack_fp32_row_8(chunks[c_idx]))
        dut.u_lpu.u_vxm.in_valid.value = Force(1)
        ready = False
        for _ in range(100):
            val = dut.u_lpu.u_vxm.in_ready.value
            if str(val) not in ('U', 'X', 'Z') and int(val):
                ready = True
                await lpu.tick(dut, 1)
                break
            await lpu.tick(dut, 1)
        dut.u_lpu.u_vxm.in_valid.value = Force(0)
        if not ready:
            raise AssertionError(f"Softmax in_ready timeout on chunk {c_idx}")
            
    out_elements = []
    for c_idx in range(64):
        found = False
        for _ in range(200):
            val = dut.u_lpu.u_vxm.out_valid.value
            if str(val) not in ('U', 'X', 'Z') and int(val):
                row_word = int(dut.u_lpu.u_vxm.stream_out.value)
                scale_word = int(dut.u_lpu.u_vxm.stream_out_scale.value)
                row_bits = [(row_word >> (8 * idx)) & 0xFF for idx in range(8)]
                scale_exp = scale_word & 0xFF
                if scale_exp & 0x80:
                    scale_exp -= 256
                chunk_floats = lpu.dequantize_regular_fp8_row(row_bits, scale_exp)
                out_elements.extend(chunk_floats)
                found = True
                await lpu.tick(dut, 1)
                break
            await lpu.tick(dut, 1)
        if not found:
            raise AssertionError(f"Softmax out_valid timeout on chunk {c_idx}")
            
    dut.u_lpu.u_vxm.in_valid.value = Release()
    dut.u_lpu.u_vxm.stream_in_data.value = Release()
    dut.u_lpu.u_vxm.stream_in_bias.value = Release()
    dut.u_lpu.vxm_rmsnorm_gamma_reg.value = Release()
    dut.u_lpu.vxm_rmsnorm_beta_reg.value = Release()
    dut.u_lpu.u_vxm.rmsnorm_bypass.value = Release()
    dut.u_lpu.u_vxm.vxm_ctrl.value = Release()
    dut.u_lpu.u_vxm.fp_quant_mode.value = Release()
    dut.u_lpu.u_vxm.rope_en.value = Release()
    dut.u_lpu.u_vxm.residual_op.value = Release()
    dut.u_lpu.u_vxm.out_ready.value = Release()
    
    return out_elements[:len(data)]


async def run_forced_vxm_residual_chunk(dut, base_chunk, delta_chunk):
    await drive_forced_vxm_residual_op_8(dut, data=base_chunk, residual_op=lpu.VXM_RES_LOAD, reset=True)
    await drive_forced_vxm_residual_op_8(dut, data=delta_chunk, residual_op=lpu.VXM_RES_ADD)
    
    dut.u_lpu.u_vxm.in_valid.value = Force(0)
    dut.u_lpu.u_vxm.stream_in_data.value = Force(0)
    dut.u_lpu.u_vxm.stream_in_bias.value = Force(0)
    dut.u_lpu.vxm_rmsnorm_gamma_reg.value = Force(0)
    dut.u_lpu.vxm_rmsnorm_beta_reg.value = Force(0)
    dut.u_lpu.u_vxm.rmsnorm_bypass.value = Force(1)
    dut.u_lpu.u_vxm.vxm_ctrl.value = Force(0b0000)
    dut.u_lpu.u_vxm.fp_quant_mode.value = Force(1)
    dut.u_lpu.u_vxm.rope_en.value = Force(0)
    dut.u_lpu.u_vxm.residual_op.value = Force(lpu.VXM_RES_EMIT)
    dut.u_lpu.u_vxm.out_ready.value = Force(1)
    
    await lpu.tick(dut, 2)
    
    dut.u_lpu.u_vxm.stream_in_data.value = Force(0)
    dut.u_lpu.u_vxm.in_valid.value = Force(1)
    
    await lpu.tick(dut, 1)
    dut.u_lpu.u_vxm.in_valid.value = Force(0)
    
    out_chunk = None
    for _ in range(160):
        val = dut.u_lpu.u_vxm.out_valid.value
        if str(val) not in ('U', 'X', 'Z') and int(val):
            row_word = int(dut.u_lpu.u_vxm.stream_out.value)
            scale_word = int(dut.u_lpu.u_vxm.stream_out_scale.value)
            row_bits = [(row_word >> (8 * idx)) & 0xFF for idx in range(8)]
            scale_exp = scale_word & 0xFF
            if scale_exp & 0x80:
                scale_exp -= 256
            out_chunk = lpu.dequantize_regular_fp8_row(row_bits, scale_exp)
            await lpu.tick(dut, 1)
            break
        await lpu.tick(dut, 1)
        
    dut.u_lpu.u_vxm.in_valid.value = Release()
    dut.u_lpu.u_vxm.stream_in_data.value = Release()
    dut.u_lpu.u_vxm.stream_in_bias.value = Release()
    dut.u_lpu.vxm_rmsnorm_gamma_reg.value = Release()
    dut.u_lpu.vxm_rmsnorm_beta_reg.value = Release()
    dut.u_lpu.u_vxm.rmsnorm_bypass.value = Release()
    dut.u_lpu.u_vxm.vxm_ctrl.value = Release()
    dut.u_lpu.u_vxm.fp_quant_mode.value = Release()
    dut.u_lpu.u_vxm.rope_en.value = Release()
    dut.u_lpu.u_vxm.residual_op.value = Release()
    dut.u_lpu.u_vxm.out_ready.value = Release()
    
    if out_chunk is None:
        raise AssertionError("Residual emit did not produce an output")
    return out_chunk


async def drive_forced_vxm_residual_op_8(dut, data, residual_op, reset=False):
    if reset:
        await lpu.reset_dut(dut)
        
    dut.u_lpu.u_vxm.in_valid.value = Force(0)
    dut.u_lpu.u_vxm.stream_in_data.value = Force(0)
    dut.u_lpu.u_vxm.stream_in_bias.value = Force(0)
    dut.u_lpu.vxm_rmsnorm_gamma_reg.value = Force(0)
    dut.u_lpu.vxm_rmsnorm_beta_reg.value = Force(0)
    dut.u_lpu.u_vxm.rmsnorm_bypass.value = Force(1)
    dut.u_lpu.u_vxm.vxm_ctrl.value = Force(0)
    dut.u_lpu.u_vxm.fp_quant_mode.value = Force(1)
    dut.u_lpu.u_vxm.rope_en.value = Force(0)
    dut.u_lpu.u_vxm.residual_op.value = Force(residual_op)
    dut.u_lpu.u_vxm.out_ready.value = Force(1)
    
    await lpu.tick(dut, 2)
    
    dut.u_lpu.u_vxm.stream_in_data.value = Force(pack_fp32_row_8(data))
    dut.u_lpu.u_vxm.in_valid.value = Force(1)
    
    await lpu.tick(dut, 1)
    dut.u_lpu.u_vxm.in_valid.value = Force(0)
    
    for _ in range(120):
        await lpu.tick(dut, 1)
        val = dut.u_lpu.u_vxm.residual_done.value
        if str(val) not in ('U', 'X', 'Z') and int(val):
            dut.u_lpu.u_vxm.in_valid.value = Release()
            dut.u_lpu.u_vxm.stream_in_data.value = Release()
            dut.u_lpu.u_vxm.stream_in_bias.value = Release()
            dut.u_lpu.vxm_rmsnorm_gamma_reg.value = Release()
            dut.u_lpu.vxm_rmsnorm_beta_reg.value = Release()
            dut.u_lpu.u_vxm.rmsnorm_bypass.value = Release()
            dut.u_lpu.u_vxm.vxm_ctrl.value = Release()
            dut.u_lpu.u_vxm.fp_quant_mode.value = Release()
            dut.u_lpu.u_vxm.rope_en.value = Release()
            dut.u_lpu.u_vxm.residual_op.value = Release()
            dut.u_lpu.u_vxm.out_ready.value = Release()
            return
            
    dut.u_lpu.u_vxm.in_valid.value = Release()
    dut.u_lpu.u_vxm.stream_in_data.value = Release()
    dut.u_lpu.u_vxm.stream_in_bias.value = Release()
    dut.u_lpu.vxm_rmsnorm_gamma_reg.value = Release()
    dut.u_lpu.vxm_rmsnorm_beta_reg.value = Release()
    dut.u_lpu.u_vxm.rmsnorm_bypass.value = Release()
    dut.u_lpu.u_vxm.vxm_ctrl.value = Release()
    dut.u_lpu.u_vxm.fp_quant_mode.value = Release()
    dut.u_lpu.u_vxm.rope_en.value = Release()
    dut.u_lpu.u_vxm.residual_op.value = Release()
    dut.u_lpu.u_vxm.out_ready.value = Release()
    raise AssertionError("VXM residual op did not complete")


def validate_direct_prompt_tokens(vocab, words):
    missing = [word for word in words if word not in vocab]
    assert not missing, f"prompt words are not direct vocab entries: {missing}"


def direct_prompt_ids(vocab, words):
    validate_direct_prompt_tokens(vocab, words)
    ids = []
    for word in words:
        entry = vocab[word]
        ids.append(entry["id"] if isinstance(entry, dict) else entry)
    return ids


async def run_lpu_model_forward_8(dut, input_ids, id_to_token, weights, *, label):
    config = get_model_config()
    dim = config["dim"]
    heads = config["heads"]
    kv_heads = config["kv_heads"]
    head_dim = dim // heads
    group_size = heads // kv_heads
    seq_len = len(input_ids)

    x = [weights["token_emb.weight"][token_id] for token_id in input_ids]

    for layer_idx in range(config["layers"]):
        block_prefix = f"blocks.{layer_idx}"

        ln1_hw = []
        for row in x:
            ln1_hw.append(
                await run_forced_vxm_rmsnorm(
                    dut,
                    data=row,
                    gamma=weights[f"{block_prefix}.ln1.weight"],
                )
            )

        q_hw = [[0.0 for _ in range(dim)] for _ in range(seq_len)]
        k_width = kv_heads * head_dim
        k_hw = [[0.0 for _ in range(k_width)] for _ in range(seq_len)]
        v_hw = [[0.0 for _ in range(k_width)] for _ in range(seq_len)]

        for r_start in range(0, seq_len, TILE):
            r_end = min(r_start + TILE, seq_len)
            left_chunk = ln1_hw[r_start:r_end]
            for start in range(0, dim, TILE):
                observed = await run_lpu_mxm_tile_full_8(
                    dut,
                    left_rows=left_chunk,
                    right_rows=weights[f"{block_prefix}.attn.q_proj.weight"][start:start + TILE],
                    label=f"{label} L{layer_idx} Q {r_start}:{r_end} {start}:{start + TILE}",
                )
                for r_idx in range(r_start, r_end):
                    for c_idx in range(TILE):
                        q_hw[r_idx][start + c_idx] = observed[r_idx - r_start][c_idx]

            for start in range(0, k_width, TILE):
                observed = await run_lpu_mxm_tile_full_8(
                    dut,
                    left_rows=left_chunk,
                    right_rows=weights[f"{block_prefix}.attn.k_proj.weight"][start:start + TILE],
                    label=f"{label} L{layer_idx} K {r_start}:{r_end} {start}:{start + TILE}",
                )
                for r_idx in range(r_start, r_end):
                    for c_idx in range(TILE):
                        k_hw[r_idx][start + c_idx] = observed[r_idx - r_start][c_idx]

                observed = await run_lpu_mxm_tile_full_8(
                    dut,
                    left_rows=left_chunk,
                    right_rows=weights[f"{block_prefix}.attn.v_proj.weight"][start:start + TILE],
                    label=f"{label} L{layer_idx} V {r_start}:{r_end} {start}:{start + TILE}",
                )
                for r_idx in range(r_start, r_end):
                    for c_idx in range(TILE):
                        v_hw[r_idx][start + c_idx] = observed[r_idx - r_start][c_idx]

        q_rot_hw = [[0.0 for _ in range(dim)] for _ in range(seq_len)]
        k_rot_hw = [[0.0 for _ in range(k_width)] for _ in range(seq_len)]
        for t in range(seq_len):
            cos_val, sin_val = get_rope_cos_sin(t, head_dim=head_dim)
            cos_bits = [q1_7_bits(c) for c in cos_val]
            sin_bits = [q1_7_bits(s) for s in sin_val]
            for h in range(heads):
                start = h * head_dim
                rotated_chunk = await run_forced_vxm_rope_chunk(
                    dut,
                    data=q_hw[t][start:start + TILE],
                    cos_bits=cos_bits,
                    sin_bits=sin_bits,
                )
                q_rot_hw[t][start:start + TILE] = rotated_chunk[:TILE]
            for kh in range(kv_heads):
                start = kh * head_dim
                rotated_chunk = await run_forced_vxm_rope_chunk(
                    dut,
                    data=k_hw[t][start:start + TILE],
                    cos_bits=cos_bits,
                    sin_bits=sin_bits,
                )
                k_rot_hw[t][start:start + TILE] = rotated_chunk[:TILE]

        attn_out_hw = [[0.0 for _ in range(dim)] for _ in range(seq_len)]
        for h in range(heads):
            kh = h // group_size
            q_h = [q_rot_hw[t][h * head_dim:(h + 1) * head_dim] for t in range(seq_len)]
            k_kh = [k_rot_hw[t][kh * head_dim:(kh + 1) * head_dim] for t in range(seq_len)]
            v_kh = [v_hw[t][kh * head_dim:(kh + 1) * head_dim] for t in range(seq_len)]

            scores_h = [[0.0 for _ in range(seq_len)] for _ in range(seq_len)]
            for q_start in range(0, seq_len, TILE):
                q_end = min(q_start + TILE, seq_len)
                for k_start in range(0, seq_len, TILE):
                    k_end = min(k_start + TILE, seq_len)
                    observed = await run_lpu_mxm_tile_full_8(
                        dut,
                        left_rows=q_h[q_start:q_end],
                        right_rows=k_kh[k_start:k_end],
                        label=f"{label} L{layer_idx} H{h} QK {q_start}:{q_end} {k_start}:{k_end}",
                    )
                    for r_idx in range(q_start, q_end):
                        for c_idx in range(k_start, k_end):
                            scores_h[r_idx][c_idx] = observed[r_idx - q_start][c_idx - k_start]

            probs_h = []
            scale = math.sqrt(head_dim)
            for row_idx, row in enumerate(scores_h):
                scaled_row = [lpu.to_f32(val / scale) for val in row]
                masked_row = [
                    val if col_idx <= row_idx else -1.0e30
                    for col_idx, val in enumerate(scaled_row)
                ]
                probs_h.append(await run_forced_vxm_softmax(dut, masked_row))

            v_by_hidden = transpose(v_kh)
            out_h = [[0.0 for _ in range(head_dim)] for _ in range(seq_len)]
            for r_start in range(0, seq_len, TILE):
                r_end = min(r_start + TILE, seq_len)
                observed = await run_lpu_mxm_tile_full_8(
                    dut,
                    left_rows=probs_h[r_start:r_end],
                    right_rows=v_by_hidden,
                    label=f"{label} L{layer_idx} H{h} PV {r_start}:{r_end}",
                )
                for r_idx in range(r_start, r_end):
                    out_h[r_idx] = observed[r_idx - r_start][:head_dim]

            for r_idx in range(seq_len):
                attn_out_hw[r_idx][h * head_dim:(h + 1) * head_dim] = out_h[r_idx]

        attn_proj_hw = [[0.0 for _ in range(dim)] for _ in range(seq_len)]
        for r_start in range(0, seq_len, TILE):
            r_end = min(r_start + TILE, seq_len)
            for start in range(0, dim, TILE):
                observed = await run_lpu_mxm_tile_full_8(
                    dut,
                    left_rows=attn_out_hw[r_start:r_end],
                    right_rows=weights[f"{block_prefix}.attn.out_proj.weight"][start:start + TILE],
                    label=f"{label} L{layer_idx} O {r_start}:{r_end} {start}:{start + TILE}",
                )
                for r_idx in range(r_start, r_end):
                    for c_idx in range(TILE):
                        attn_proj_hw[r_idx][start + c_idx] = observed[r_idx - r_start][c_idx]

        x_after_attn_hw = []
        for row_idx in range(seq_len):
            row_res = []
            for start in range(0, dim, TILE):
                row_res.extend(
                    await run_forced_vxm_residual_chunk(
                        dut,
                        base_chunk=x[row_idx][start:start + TILE],
                        delta_chunk=attn_proj_hw[row_idx][start:start + TILE],
                    )
                )
            x_after_attn_hw.append(row_res[:dim])

        ln2_hw = []
        for row in x_after_attn_hw:
            ln2_hw.append(
                await run_forced_vxm_rmsnorm(
                    dut,
                    data=row,
                    gamma=weights[f"{block_prefix}.ln2.weight"],
                )
            )

        ffn_dim = config["ffn_dim"]
        ffn_gate_hw = [[0.0 for _ in range(ffn_dim)] for _ in range(seq_len)]
        for r_start in range(0, seq_len, TILE):
            r_end = min(r_start + TILE, seq_len)
            for start in range(0, ffn_dim, TILE):
                observed = await run_lpu_mxm_tile_full_8(
                    dut,
                    left_rows=ln2_hw[r_start:r_end],
                    right_rows=weights[f"{block_prefix}.ffn_gate.weight"][start:start + TILE],
                    label=f"{label} L{layer_idx} FFN gate {r_start}:{r_end} {start}:{start + TILE}",
                )
                for r_idx in range(r_start, r_end):
                    for c_idx in range(min(TILE, ffn_dim - start)):
                        ffn_gate_hw[r_idx][start + c_idx] = observed[r_idx - r_start][c_idx]

        ffn_hidden_hw = []
        for row in ffn_gate_hw:
            row_relu = []
            for start in range(0, ffn_dim, TILE):
                row_relu.extend(await run_forced_vxm_relu_chunk(dut, row[start:start + TILE]))
            ffn_hidden_hw.append(row_relu[:ffn_dim])

        ffn_down_hw = [[0.0 for _ in range(dim)] for _ in range(seq_len)]
        for r_start in range(0, seq_len, TILE):
            r_end = min(r_start + TILE, seq_len)
            for start in range(0, dim, TILE):
                observed = await run_lpu_mxm_tile_full_8(
                    dut,
                    left_rows=ffn_hidden_hw[r_start:r_end],
                    right_rows=weights[f"{block_prefix}.ffn_down.weight"][start:start + TILE],
                    label=f"{label} L{layer_idx} FFN down {r_start}:{r_end} {start}:{start + TILE}",
                )
                for r_idx in range(r_start, r_end):
                    for c_idx in range(TILE):
                        ffn_down_hw[r_idx][start + c_idx] = observed[r_idx - r_start][c_idx]

        x_next_hw = []
        for row_idx in range(seq_len):
            row_res = []
            for start in range(0, dim, TILE):
                row_res.extend(
                    await run_forced_vxm_residual_chunk(
                        dut,
                        base_chunk=x_after_attn_hw[row_idx][start:start + TILE],
                        delta_chunk=ffn_down_hw[row_idx][start:start + TILE],
                    )
                )
            x_next_hw.append(row_res[:dim])
        x = x_next_hw

    final_hw = []
    for row in x:
        final_hw.append(await run_forced_vxm_rmsnorm(dut, data=row, gamma=weights["ln_f.weight"]))

    last_hidden_hw = final_hw[-1]
    hw_logits = [0.0 for _ in range(config["vocab_size"])]
    for vocab_start in range(0, config["vocab_size"], TILE):
        observed = await run_lpu_mxm_tile_full_8(
            dut,
            left_rows=[last_hidden_hw],
            right_rows=weights["lm_head.weight"][vocab_start:vocab_start + TILE],
            label=f"{label} LM head {vocab_start}:{vocab_start + TILE}",
        )
        for lane in range(min(TILE, config["vocab_size"] - vocab_start)):
            hw_logits[vocab_start + lane] = observed[0][lane]

    pred_id = argmax(hw_logits)
    return {
        "pred_id": pred_id,
        "pred_token": id_to_token[pred_id],
        "logits": hw_logits,
        "top5": top_tokens(hw_logits, id_to_token),
    }


def kv_cache_addr(layer_idx, token_pos, kv_head, config=None):
    if config is None:
        config = MODEL_CONFIG
    return (
        layer_idx * kv_cache_layer_rows(config)
        + token_pos * kv_cache_rows_per_token(config)
        + kv_head
    )


def pack_mem_row_8(row_bits, scale_exp):
    return pack_bytes_8(row_bits) | ((scale_exp & 0xFF) << 64)


def read_mem_raw_word(dut, *, addr, is_key):
    if is_key:
        return int(dut.u_lpu.u_mem0.sram_array[addr].value) & ((1 << 72) - 1)
    return int(dut.u_lpu.u_mem1.sram_array[addr].value) & ((1 << 72) - 1)


def unpack_mem_row_8(word):
    row_bits = [(word >> (8 * idx)) & 0xFF for idx in range(8)]
    scale_exp = (word >> 64) & 0xFF
    if scale_exp & 0x80:
        scale_exp -= 1 << 8
    return row_bits, scale_exp


def preload_mem0_quantized_row(dut, addr, row):
    row_bits, scale_exp = lpu.regular_fp8_row_quant_expected(row)
    dut.u_lpu.u_mem0.sram_array[addr].value = pack_mem_row_8(row_bits, scale_exp)
    return lpu.dequantize_regular_fp8_row(row_bits, scale_exp)


def read_mem0_dequantized_row(dut, addr):
    raw_word = int(dut.u_lpu.u_mem0.sram_array[addr].value) & ((1 << 72) - 1)
    row_bits, scale_exp = unpack_mem_row_8(raw_word)
    return lpu.dequantize_regular_fp8_row(row_bits, scale_exp), row_bits, scale_exp


async def run_icu_fragment(dut, program, *, cycles_extra=2):
    lpu.preload_program(dut, program)
    dut.u_lpu.u_icu.pc.value = 0
    await lpu.tick(dut, len(program) + cycles_extra)


def append_mem0_to_vxm_operand(
    program,
    *,
    addr,
    operand_sel,
    vxm_ctrl=0,
    rmsnorm_en=0,
    rope_en=0,
    residual_op=lpu.VXM_RES_PASS,
    hold_cycles=12,
):
    program.append(lpu.build_instruction(mem0_read_en=1, mem0_addr=addr, fp_quant_mode=1))
    program.append(
        lpu.build_instruction(
            eastbound_sel=lpu.EB_MEM0,
            eastbound_consumer_sel=lpu.EC_VXM,
            vxm_operand_sel=operand_sel,
            vxm_ctrl=vxm_ctrl,
            vxm_rope_en=rope_en,
            vxm_residual_op=residual_op,
            vxm_layernorm_en=rmsnorm_en,
            fp_quant_mode=1,
        )
    )
    if operand_sel == lpu.VXM_OPERAND_DATA:
        for _ in range(hold_cycles):
            program.append(
                lpu.build_instruction(
                    vxm_ctrl=vxm_ctrl,
                    vxm_rope_en=rope_en,
                    vxm_residual_op=residual_op,
                    vxm_layernorm_en=rmsnorm_en,
                    fp_quant_mode=1,
                )
            )


async def wait_vxm_input_fifo_empty(dut, *, timeout=240):
    for _ in range(timeout):
        empty = dut.u_lpu.u_vxm_input_fifo.empty.value
        if str(empty) not in ("U", "X", "Z") and int(empty):
            return
        await lpu.tick(dut, 1)
    raise AssertionError("VXM input FIFO did not drain")


async def wait_vxm_output_fifo_nonempty(dut, *, timeout=400):
    for _ in range(timeout):
        empty = dut.u_lpu.u_vxm_output_fifo.empty.value
        if str(empty) not in ("U", "X", "Z") and not int(empty):
            return
        await lpu.tick(dut, 1)
    raise AssertionError(
        "VXM output FIFO did not receive a row: "
        f"softmax_state={dut.u_lpu.u_vxm.softmax_inst.state_q.value}, "
        f"softmax_write_idx={dut.u_lpu.u_vxm.softmax_inst.write_idx.value}, "
        f"softmax_read_idx={dut.u_lpu.u_vxm.softmax_inst.read_idx.value}, "
        f"softmax_out_valid={dut.u_lpu.u_vxm.chunked_softmax_out_valid.value}, "
        f"softmax_out_ready={dut.u_lpu.u_vxm.chunked_softmax_out_ready.value}, "
        f"softmax_skid_valid={dut.u_lpu.u_vxm.softmax_result_valid.value}, "
        f"residual_start={dut.u_lpu.u_vxm.residual_start.value}, "
        f"residual_result_valid={dut.u_lpu.u_vxm.residual_result_valid.value}, "
        f"quant_issue={dut.u_lpu.u_vxm.quant_issue.value}, "
        f"quantize_valid={dut.u_lpu.u_vxm.quantize_valid.value}, "
        f"stream_out_valid={dut.u_lpu.u_vxm.stream_out_valid_reg.value}"
    )


async def wait_vxm_input_fifo_empty_with_hold(dut, hold_instruction, *, timeout=400):
    program_len = min(timeout, 960)
    lpu.preload_program(dut, [hold_instruction for _ in range(program_len)])
    dut.u_lpu.u_icu.pc.value = 0
    for _ in range(timeout):
        empty = dut.u_lpu.u_vxm_input_fifo.empty.value
        if str(empty) not in ("U", "X", "Z") and int(empty):
            return
        await lpu.tick(dut, 1)
    raise AssertionError("VXM input FIFO did not drain")


async def wait_vxm_output_fifo_nonempty_with_hold(dut, hold_instruction, *, timeout=1200):
    program_len = min(timeout, 960)
    lpu.preload_program(dut, [hold_instruction for _ in range(program_len)])
    dut.u_lpu.u_icu.pc.value = 0
    for _ in range(timeout):
        empty = dut.u_lpu.u_vxm_output_fifo.empty.value
        if str(empty) not in ("U", "X", "Z") and not int(empty):
            return
        await lpu.tick(dut, 1)
    await wait_vxm_output_fifo_nonempty(dut, timeout=0)


def vxm_softmax_path_idle(dut):
    state = dut.u_lpu.u_vxm.softmax_inst.state_q.value
    out_empty = dut.u_lpu.u_vxm_output_fifo.empty.value
    scale_empty = dut.u_lpu.u_vxm_output_scale_fifo.empty.value
    softmax_valid = dut.u_lpu.u_vxm.chunked_softmax_out_valid.value
    skid_valid = dut.u_lpu.u_vxm.softmax_result_valid.value
    stream_valid = dut.u_lpu.u_vxm.stream_out_valid_reg.value
    values = (state, out_empty, scale_empty, softmax_valid, skid_valid, stream_valid)
    if any(str(value) in ("U", "X", "Z") for value in values):
        return False
    return (
        int(state) == 0
        and int(out_empty)
        and int(scale_empty)
        and not int(softmax_valid)
        and not int(skid_valid)
        and not int(stream_valid)
    )


async def drain_vxm_softmax_outputs(dut, hold_instruction, *, drain_addr, timeout=1600):
    lpu.preload_program(dut, [hold_instruction for _ in range(min(timeout, 960))])
    dut.u_lpu.u_icu.pc.value = 0
    drained = 0
    for _ in range(timeout):
        out_empty = dut.u_lpu.u_vxm_output_fifo.empty.value
        if str(out_empty) not in ("U", "X", "Z") and not int(out_empty):
            await store_next_vxm_result_to_mem0(
                dut,
                drain_addr + drained,
                vxm_ctrl=0b1000,
            )
            lpu.preload_program(dut, [hold_instruction for _ in range(min(timeout, 960))])
            dut.u_lpu.u_icu.pc.value = 0
            drained += 1
            continue

        if vxm_softmax_path_idle(dut):
            return drained

        await lpu.tick(dut, 1)

    raise AssertionError("VXM softmax path did not drain")


async def wait_vxm_residual_done(dut, *, timeout=240):
    for _ in range(timeout):
        done = dut.u_lpu.u_vxm.residual_done.value
        if str(done) not in ("U", "X", "Z") and not int(done):
            break
        await lpu.tick(dut, 1)
    else:
        raise AssertionError("VXM residual done did not deassert")

    for _ in range(timeout):
        done = dut.u_lpu.u_vxm.residual_done.value
        if str(done) not in ("U", "X", "Z") and int(done):
            await lpu.tick(dut, 1)
            return
        await lpu.tick(dut, 1)
    raise AssertionError("VXM residual operation did not complete")


async def run_mem0_vxm_residual_op(dut, *, addr, residual_op, timeout=320):
    program = []
    append_mem0_to_vxm_operand(
        program,
        addr=addr,
        operand_sel=lpu.VXM_OPERAND_DATA,
        residual_op=residual_op,
    )
    lpu.preload_program(dut, program)
    dut.u_lpu.u_icu.pc.value = 0
    for _ in range(timeout):
        await lpu.tick(dut, 1)
        done = dut.u_lpu.u_vxm.residual_done.value
        if str(done) not in ("U", "X", "Z") and int(done):
            await lpu.tick(dut, 1)
            return
    raise AssertionError("VXM residual operation did not complete")


async def store_next_vxm_result_to_mem0(dut, target_addr, *, rmsnorm_en=0, vxm_ctrl=0):
    await wait_vxm_output_fifo_nonempty(dut)
    await run_icu_fragment(
        dut,
        [
            lpu.build_instruction(
                eastbound_sel=lpu.EB_VXM,
                eastbound_consumer_sel=lpu.EC_MEM0,
                mem0_write_en=1,
                mem0_addr=target_addr,
                vxm_ctrl=vxm_ctrl,
                vxm_layernorm_en=rmsnorm_en,
                fp_quant_mode=1,
            ),
            lpu.build_instruction(vxm_ctrl=vxm_ctrl, vxm_layernorm_en=rmsnorm_en, fp_quant_mode=1),
        ],
        cycles_extra=2,
    )
    row, _, _ = read_mem0_dequantized_row(dut, target_addr)
    return row


async def run_lpu_vxm_relu_chunk(dut, data, *, base_addr=24000, out_addr=24100):
    await lpu.reset_dut(dut)
    expected_input = preload_mem0_quantized_row(dut, base_addr, data)
    program = []
    append_mem0_to_vxm_operand(
        program,
        addr=base_addr,
        operand_sel=lpu.VXM_OPERAND_DATA,
        vxm_ctrl=0b0010,
    )
    lpu.preload_program(dut, program)
    dut.u_lpu.u_icu.pc.value = 0
    await lpu.tick(dut, 1)
    observed_payload = unpack_fp32_row_8(int(dut.u_lpu.vxm_operand_payload.value))
    assert observed_payload == expected_input, (
        f"Instruction ReLU VXM input mismatch: got {observed_payload}, expected {expected_input}"
    )
    await lpu.tick(dut, len(program) + 4)
    return await store_next_vxm_result_to_mem0(dut, out_addr)


async def run_lpu_vxm_rope_chunk(
    dut,
    data,
    cos_bits,
    sin_bits,
    *,
    base_addr=24000,
    cos_addr=24001,
    sin_addr=24002,
    out_addr=24100,
):
    await lpu.reset_dut(dut)
    preload_mem0_quantized_row(dut, base_addr, data)
    dut.u_lpu.u_mem0.sram_array[cos_addr].value = pack_mem_row_8(cos_bits, 0)
    dut.u_lpu.u_mem0.sram_array[sin_addr].value = pack_mem_row_8(sin_bits, 0)
    program = []
    append_mem0_to_vxm_operand(program, addr=cos_addr, operand_sel=lpu.VXM_OPERAND_ROPE_COS)
    append_mem0_to_vxm_operand(program, addr=sin_addr, operand_sel=lpu.VXM_OPERAND_ROPE_SIN)
    append_mem0_to_vxm_operand(
        program,
        addr=base_addr,
        operand_sel=lpu.VXM_OPERAND_DATA,
        rope_en=1,
        hold_cycles=64,
    )
    await run_icu_fragment(dut, program, cycles_extra=4)
    return await store_next_vxm_result_to_mem0(dut, out_addr)


async def run_lpu_vxm_residual_chunk(
    dut,
    base_chunk,
    delta_chunk,
    *,
    base_addr=24000,
    delta_addr=24001,
    out_addr=24100,
):
    await lpu.reset_dut(dut)
    preload_mem0_quantized_row(dut, base_addr, base_chunk)
    preload_mem0_quantized_row(dut, delta_addr, delta_chunk)

    await run_mem0_vxm_residual_op(dut, addr=base_addr, residual_op=lpu.VXM_RES_LOAD)
    await run_mem0_vxm_residual_op(dut, addr=delta_addr, residual_op=lpu.VXM_RES_ADD)

    await run_icu_fragment(
        dut,
        [
            lpu.build_instruction(
                vxm_residual_op=lpu.VXM_RES_EMIT,
                fp_quant_mode=1,
            ),
            lpu.build_instruction(fp_quant_mode=1),
        ],
        cycles_extra=4,
    )
    return await store_next_vxm_result_to_mem0(dut, out_addr)


async def run_lpu_vxm_rmsnorm(dut, data, gamma, *, base_addr=24000, gamma_addr=24016, out_addr=24100):
    await lpu.reset_dut(dut)
    assert len(data) == len(gamma)
    assert len(data) % TILE == 0
    num_chunks = len(data) // TILE
    chunks_in = [data[i*TILE : (i+1)*TILE] for i in range(num_chunks)]
    gamma_chunks = [gamma[i*TILE : (i+1)*TILE] for i in range(num_chunks)]

    for c_idx, (chunk, gamma_chunk) in enumerate(zip(chunks_in, gamma_chunks)):
        preload_mem0_quantized_row(dut, base_addr + c_idx, chunk)
        preload_mem0_quantized_row(dut, gamma_addr + c_idx, gamma_chunk)

    for c_idx in range(num_chunks):
        program = []
        append_mem0_to_vxm_operand(
            program,
            addr=gamma_addr + c_idx,
            operand_sel=lpu.VXM_OPERAND_GAMMA,
            rmsnorm_en=1,
        )
        append_mem0_to_vxm_operand(
            program,
            addr=base_addr + c_idx,
            operand_sel=lpu.VXM_OPERAND_DATA,
            rmsnorm_en=1,
            hold_cycles=140,
        )
        await run_icu_fragment(dut, program, cycles_extra=0)

    out_row = []
    for c_idx in range(num_chunks):
        hold_cycles = 320 if c_idx == 0 else 120
        await run_icu_fragment(
            dut,
            [
                lpu.build_instruction(vxm_layernorm_en=1, fp_quant_mode=1)
                for _ in range(hold_cycles)
            ],
            cycles_extra=0,
        )
        out_row.extend(
            await store_next_vxm_result_to_mem0(
                dut,
                out_addr + c_idx,
                rmsnorm_en=1,
            )
        )
    return out_row[:len(data)]


async def run_lpu_vxm_softmax(
    dut,
    data,
    *,
    base_addr=24000,
    out_addr=24100,
    drain_addr=24200,
    reset=True,
):
    if reset:
        await lpu.reset_dut(dut)
    softmax_len = SOFTMAX_CHUNKS * TILE
    assert len(data) <= softmax_len
    data_padded = list(data) + [-1e30] * (softmax_len - len(data))
    chunks = [data_padded[i*TILE : (i+1)*TILE] for i in range(SOFTMAX_CHUNKS)]
    softmax_hold = lpu.build_instruction(vxm_ctrl=0b1000, fp_quant_mode=1)

    for c_idx, chunk in enumerate(chunks):
        preload_mem0_quantized_row(dut, base_addr + c_idx, chunk)

    for c_idx in range(SOFTMAX_CHUNKS):
        program = []
        append_mem0_to_vxm_operand(
            program,
            addr=base_addr + c_idx,
            operand_sel=lpu.VXM_OPERAND_DATA,
            vxm_ctrl=0b1000,
            hold_cycles=20,
        )
        await run_icu_fragment(dut, program, cycles_extra=0)
        await wait_vxm_input_fifo_empty_with_hold(dut, softmax_hold, timeout=400)

    out_row = []
    needed_chunks = max(1, (len(data) + TILE - 1) // TILE)
    for c_idx in range(needed_chunks):
        await wait_vxm_output_fifo_nonempty_with_hold(dut, softmax_hold, timeout=1000)
        out_row.extend(await store_next_vxm_result_to_mem0(dut, out_addr + c_idx, vxm_ctrl=0b1000))

    await drain_vxm_softmax_outputs(dut, softmax_hold, drain_addr=drain_addr)
    return out_row[:len(data)]


def stage_vxm_result_fifo_for_store(dut, *, packed_row, scale_exp):
    dut.u_lpu.u_vxm_output_fifo.mem[0].value = packed_row
    dut.u_lpu.u_vxm_output_fifo.rd_ptr.value = 0
    dut.u_lpu.u_vxm_output_fifo.wr_ptr.value = 1
    dut.u_lpu.u_vxm_output_fifo.count.value = 1
    dut.u_lpu.u_vxm_output_scale_fifo.mem[0].value = scale_exp & 0xFF
    dut.u_lpu.u_vxm_output_scale_fifo.rd_ptr.value = 0
    dut.u_lpu.u_vxm_output_scale_fifo.wr_ptr.value = 1
    dut.u_lpu.u_vxm_output_scale_fifo.count.value = 1


async def store_staged_vxm_result_to_memory(dut, *, addr, is_key):
    if is_key:
        store_instr = lpu.build_instruction(
            eastbound_sel=lpu.EB_VXM,
            eastbound_consumer_sel=lpu.EC_MEM0,
            mem0_write_en=1,
            mem0_addr=addr,
            fp_quant_mode=1,
        )
    else:
        store_instr = lpu.build_instruction(
            eastbound_sel=lpu.EB_VXM,
            eastbound_consumer_sel=lpu.EC_MEM1,
            mem1_write_en=1,
            mem1_addr=addr,
            fp_quant_mode=1,
        )

    program = [
        store_instr,
        lpu.build_instruction(fp_quant_mode=1),
        lpu.build_instruction(fp_quant_mode=1),
    ]
    lpu.preload_program(dut, program)
    dut.u_lpu.u_icu.pc.value = 0
    await lpu.tick(dut, len(program) + 2)


async def store_kv_cache_row(dut, *, layer_idx, token_pos, kv_head, row, is_key, config=None):
    row_bits, scale_exp = lpu.regular_fp8_row_quant_expected(row)
    packed = pack_mem_row_8(row_bits, scale_exp)
    addr = kv_cache_addr(layer_idx, token_pos, kv_head, config)
    stage_vxm_result_fifo_for_store(
        dut,
        packed_row=pack_bytes_8(row_bits),
        scale_exp=scale_exp,
    )
    await store_staged_vxm_result_to_memory(dut, addr=addr, is_key=is_key)
    observed = read_mem_raw_word(dut, addr=addr, is_key=is_key)
    assert observed == packed, (
        f"KV cache store mismatch at addr {addr}: got 0x{observed:018x}, "
        f"expected 0x{packed:018x}"
    )
    return lpu.dequantize_regular_fp8_row(row_bits, scale_exp)


async def linear_current_row_mxm_8(dut, row, weight, out_dim, *, label):
    out = [0.0 for _ in range(out_dim)]
    for start in range(0, out_dim, TILE):
        observed = await run_lpu_mxm_tile_full_8(
            dut,
            left_rows=[row],
            right_rows=weight[start:start + TILE],
            label=f"{label} {start}:{start + TILE}",
        )
        for lane in range(min(TILE, out_dim - start)):
            out[start + lane] = observed[0][lane]
    return out


async def run_lpu_decode_token_kv_8(
    dut,
    *,
    token_id,
    token_pos,
    kv_cache,
    id_to_token,
    weights,
    label,
    emit_logits=True,
    config=None,
):
    if config is None:
        config = get_model_config()
    dim = config["dim"]
    heads = config["heads"]
    kv_heads = config["kv_heads"]
    head_dim = dim // heads
    group_size = heads // kv_heads
    kv_width = kv_heads * head_dim

    x = list(weights["token_emb.weight"][token_id])

    for layer_idx in range(config["layers"]):
        if DECODE_PROGRESS:
            dut._log.info("%s token_pos=%d layer=%d start", label, token_pos, layer_idx)
        block_prefix = f"blocks.{layer_idx}"

        ln1 = await run_lpu_vxm_rmsnorm(
            dut,
            data=x,
            gamma=weights[f"{block_prefix}.ln1.weight"],
        )

        q = await linear_current_row_mxm_8(
            dut,
            ln1,
            weights[f"{block_prefix}.attn.q_proj.weight"],
            dim,
            label=f"{label} L{layer_idx} Q",
        )
        k = await linear_current_row_mxm_8(
            dut,
            ln1,
            weights[f"{block_prefix}.attn.k_proj.weight"],
            kv_width,
            label=f"{label} L{layer_idx} K",
        )
        v = await linear_current_row_mxm_8(
            dut,
            ln1,
            weights[f"{block_prefix}.attn.v_proj.weight"],
            kv_width,
            label=f"{label} L{layer_idx} V",
        )

        cos_val, sin_val = get_rope_cos_sin(token_pos, head_dim=head_dim)
        cos_bits = [q1_7_bits(c) for c in cos_val]
        sin_bits = [q1_7_bits(s) for s in sin_val]

        q_rot = [0.0 for _ in range(dim)]
        for h in range(heads):
            start = h * head_dim
            q_rot[start:start + head_dim] = (
                await run_lpu_vxm_rope_chunk(
                    dut,
                    data=q[start:start + head_dim],
                    cos_bits=cos_bits,
                    sin_bits=sin_bits,
                )
            )[:head_dim]

        k_rot = [0.0 for _ in range(kv_width)]
        for kh in range(kv_heads):
            start = kh * head_dim
            k_chunk = (
                await run_lpu_vxm_rope_chunk(
                    dut,
                    data=k[start:start + head_dim],
                    cos_bits=cos_bits,
                    sin_bits=sin_bits,
                )
            )[:head_dim]
            k_rot[start:start + head_dim] = await store_kv_cache_row(
                dut,
                layer_idx=layer_idx,
                token_pos=token_pos,
                kv_head=kh,
                row=k_chunk,
                is_key=True,
                config=config,
            )

            v_chunk = v[start:start + head_dim]
            kv_cache["v"][layer_idx][kh].append(
                await store_kv_cache_row(
                    dut,
                    layer_idx=layer_idx,
                    token_pos=token_pos,
                    kv_head=kh,
                    row=v_chunk,
                    is_key=False,
                    config=config,
                )
            )
            kv_cache["k"][layer_idx][kh].append(k_rot[start:start + head_dim])

        attn_out = [0.0 for _ in range(dim)]
        cache_len = token_pos + 1
        for h in range(heads):
            if DECODE_PROGRESS:
                dut._log.info(
                    "%s token_pos=%d layer=%d head=%d attention start cache_len=%d",
                    label,
                    token_pos,
                    layer_idx,
                    h,
                    cache_len,
                )
            kh = h // group_size
            q_h = q_rot[h * head_dim:(h + 1) * head_dim]
            k_rows = kv_cache["k"][layer_idx][kh]
            v_rows = kv_cache["v"][layer_idx][kh]
            scores = [0.0 for _ in range(cache_len)]

            for k_start in range(0, cache_len, TILE):
                k_end = min(k_start + TILE, cache_len)
                observed = await run_lpu_mxm_tile_full_8(
                    dut,
                    left_rows=[q_h],
                    right_rows=k_rows[k_start:k_end],
                    label=f"{label} L{layer_idx} H{h} QK {k_start}:{k_end}",
                )
                for lane in range(k_end - k_start):
                    scores[k_start + lane] = observed[0][lane]

            scaled_scores = [lpu.to_f32(score / math.sqrt(head_dim)) for score in scores]
            probs = await run_lpu_vxm_softmax(
                dut,
                scaled_scores,
                base_addr=24000,
                out_addr=24100,
                drain_addr=24200,
                reset=False,
            )
            if DECODE_PROGRESS:
                dut._log.info(
                    "%s token_pos=%d layer=%d head=%d softmax done",
                    label,
                    token_pos,
                    layer_idx,
                    h,
                )
            out_h = [0.0 for _ in range(head_dim)]
            for v_start in range(0, cache_len, TILE):
                v_end = min(v_start + TILE, cache_len)
                probs_chunk = probs[v_start:v_end]
                v_chunk = v_rows[v_start:v_end]
                v_by_hidden = [
                    [v_chunk[row_idx][lane] for row_idx in range(v_end - v_start)]
                    for lane in range(head_dim)
                ]
                observed = await run_lpu_mxm_tile_full_8(
                    dut,
                    left_rows=[probs_chunk],
                    right_rows=v_by_hidden,
                    label=f"{label} L{layer_idx} H{h} PV {v_start}:{v_end}",
                )
                for lane in range(head_dim):
                    out_h[lane] = lpu.to_f32(out_h[lane] + observed[0][lane])

            attn_out[h * head_dim:(h + 1) * head_dim] = out_h

        attn_proj = await linear_current_row_mxm_8(
            dut,
            attn_out,
            weights[f"{block_prefix}.attn.out_proj.weight"],
            dim,
            label=f"{label} L{layer_idx} O",
        )

        x_after_attn = []
        for start in range(0, dim, TILE):
            x_after_attn.extend(
                await run_lpu_vxm_residual_chunk(
                    dut,
                    base_chunk=x[start:start + TILE],
                    delta_chunk=attn_proj[start:start + TILE],
                )
            )
        x_after_attn = x_after_attn[:dim]

        ln2 = await run_lpu_vxm_rmsnorm(
            dut,
            data=x_after_attn,
            gamma=weights[f"{block_prefix}.ln2.weight"],
        )
        ffn_gate = await linear_current_row_mxm_8(
            dut,
            ln2,
            weights[f"{block_prefix}.ffn_gate.weight"],
            config["ffn_dim"],
            label=f"{label} L{layer_idx} FFN gate",
        )

        ffn_hidden = []
        for start in range(0, config["ffn_dim"], TILE):
            ffn_hidden.extend(await run_lpu_vxm_relu_chunk(dut, ffn_gate[start:start + TILE]))
        ffn_hidden = ffn_hidden[:config["ffn_dim"]]

        ffn_down = await linear_current_row_mxm_8(
            dut,
            ffn_hidden,
            weights[f"{block_prefix}.ffn_down.weight"],
            dim,
            label=f"{label} L{layer_idx} FFN down",
        )

        x_next = []
        for start in range(0, dim, TILE):
            x_next.extend(
                await run_lpu_vxm_residual_chunk(
                    dut,
                    base_chunk=x_after_attn[start:start + TILE],
                    delta_chunk=ffn_down[start:start + TILE],
                )
            )
        x = x_next[:dim]
        if DECODE_PROGRESS:
            dut._log.info("%s token_pos=%d layer=%d done", label, token_pos, layer_idx)

    if not emit_logits:
        return {
            "pred_id": None,
            "pred_token": None,
            "logits": None,
            "top5": None,
        }

    final = await run_lpu_vxm_rmsnorm(dut, data=x, gamma=weights["ln_f.weight"])
    logits = await linear_current_row_mxm_8(
        dut,
        final,
        weights["lm_head.weight"],
        config["vocab_size"],
        label=f"{label} LM head",
    )
    pred_id = argmax(logits)
    return {
        "pred_id": pred_id,
        "pred_token": id_to_token[pred_id],
        "logits": logits,
        "top5": top_tokens(logits, id_to_token),
    }


def init_kv_cache(config=None):
    if config is None:
        config = MODEL_CONFIG
    return {
        "k": [[[] for _ in range(config["kv_heads"])] for _ in range(config["layers"])],
        "v": [[[] for _ in range(config["kv_heads"])] for _ in range(config["layers"])],
    }


@cocotb.test()
async def test_lpu_kv_cache_store_instructions(dut):
    cocotb.start_soon(Clock(dut.clk, SIM_CLK_NS, unit="ns").start())
    await lpu.reset_dut(dut)

    k_row = [0.125, -0.25, 0.375, -0.5, 0.625, -0.75, 0.875, -1.0]
    v_row = [-1.0, 0.875, -0.75, 0.625, -0.5, 0.375, -0.25, 0.125]
    k_addr = kv_cache_addr(layer_idx=2, token_pos=17, kv_head=3)
    v_addr = kv_cache_addr(layer_idx=4, token_pos=31, kv_head=1)

    k_decoded = await store_kv_cache_row(
        dut,
        layer_idx=2,
        token_pos=17,
        kv_head=3,
        row=k_row,
        is_key=True,
    )
    v_decoded = await store_kv_cache_row(
        dut,
        layer_idx=4,
        token_pos=31,
        kv_head=1,
        row=v_row,
        is_key=False,
    )

    assert read_mem_raw_word(dut, addr=k_addr, is_key=True) != 0
    assert read_mem_raw_word(dut, addr=v_addr, is_key=False) != 0
    dut._log.info("KV cache instruction store K MEM0[%d] decoded=%s", k_addr, k_decoded)
    dut._log.info("KV cache instruction store V MEM1[%d] decoded=%s", v_addr, v_decoded)


@cocotb.test()
async def test_lpu_mem0_to_vxm_dequant_operand(dut):
    cocotb.start_soon(Clock(dut.clk, SIM_CLK_NS, unit="ns").start())
    await lpu.reset_dut(dut)

    src_addr = 1234
    source_row = [2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -1.5]
    row_bits, scale_exp = lpu.regular_fp8_row_quant_expected(source_row)
    decoded_row = lpu.dequantize_regular_fp8_row(row_bits, scale_exp)
    raw_mem_row = pack_mem_row_8(row_bits, scale_exp)
    expected_payload = pack_fp32_row_8(decoded_row)

    dut.u_lpu.u_mem0.sram_array[src_addr].value = raw_mem_row
    program = [
        lpu.build_instruction(mem0_read_en=1, mem0_addr=src_addr, fp_quant_mode=1),
        lpu.build_instruction(
            eastbound_sel=lpu.EB_MEM0,
            eastbound_consumer_sel=lpu.EC_VXM,
            vxm_operand_sel=lpu.VXM_OPERAND_DATA,
            fp_quant_mode=1,
        ),
        lpu.build_instruction(fp_quant_mode=1),
    ]
    lpu.preload_program(dut, program)
    dut.u_lpu.u_icu.pc.value = 0

    await lpu.tick(dut, 1)

    observed_payload = int(dut.u_lpu.vxm_operand_payload.value)
    assert observed_payload == expected_payload, (
        "MEM0->VXM dequant payload mismatch: "
        f"got {unpack_fp32_row_8(observed_payload)}, expected {decoded_row}"
    )
    dut._log.info(
        "MEM0->VXM DATA dequantized 0x%018x scale=%d to %s",
        raw_mem_row,
        scale_exp,
        decoded_row,
    )


@cocotb.test()
async def test_lpu_instruction_driven_vxm_chunks(dut):
    cocotb.start_soon(Clock(dut.clk, SIM_CLK_NS, unit="ns").start())

    relu_in = [-2.0, -0.25, 0.0, 0.5, 1.25, -3.5, 4.0, 8.0]
    relu_observed = await run_lpu_vxm_relu_chunk(dut, relu_in)
    relu_expected = [lpu.to_f32(max(0.0, value)) for value in relu_in]
    relu_bits, relu_scale = lpu.regular_fp8_row_quant_expected(relu_expected)
    relu_expected = lpu.dequantize_regular_fp8_row(relu_bits, relu_scale)
    assert relu_observed == relu_expected

    residual_base = [0.5, -1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0]
    residual_delta = [0.25, 0.5, -1.0, 1.5, -2.0, 2.5, -3.0, 3.5]
    residual_observed = await run_lpu_vxm_residual_chunk(dut, residual_base, residual_delta)
    residual_expected = [
        lpu.to_f32(base + delta)
        for base, delta in zip(residual_base, residual_delta)
    ]
    residual_bits, residual_scale = lpu.regular_fp8_row_quant_expected(residual_expected)
    residual_expected = lpu.dequantize_regular_fp8_row(residual_bits, residual_scale)
    assert residual_observed == residual_expected

    rope_in = [0.5, 1.0, -1.5, 2.0, -2.5, 3.0, -3.5, 4.0]
    cos_bits = [q1_7_bits(v) for v in [1.0, 0.875, 0.75, 0.5, 1.0, 0.875, 0.75, 0.5]]
    sin_bits = [q1_7_bits(v) for v in [0.0, 0.5, 0.75, 0.875, 0.0, 0.5, 0.75, 0.875]]
    rope_observed = await run_lpu_vxm_rope_chunk(dut, rope_in, cos_bits, sin_bits)
    rope_expected = rope_rows_fp32_8(rope_in, cos_bits, sin_bits)
    rope_bits, rope_scale = lpu.regular_fp8_row_quant_expected(rope_expected)
    rope_expected = lpu.dequantize_regular_fp8_row(rope_bits, rope_scale)
    assert rope_observed == rope_expected

    softmax_in = [0.0 for _ in range(TILE)]
    softmax_observed = await run_lpu_vxm_softmax(dut, softmax_in)
    softmax_expected = softmax_rows([softmax_in])[0]
    softmax_bits, softmax_scale = lpu.regular_fp8_row_quant_expected(softmax_expected)
    softmax_expected = lpu.dequantize_regular_fp8_row(softmax_bits, softmax_scale)
    assert softmax_observed == softmax_expected

    rms_in = [
        0.125, -0.25, 0.375, -0.5, 0.625, -0.75, 0.875, -1.0,
        1.125, -1.25, 1.375, -1.5, 1.625, -1.75, 1.875, -2.0,
        2.125, -2.25, 2.375, -2.5, 2.625, -2.75, 2.875, -3.0,
        3.125, -3.25, 3.375, -3.5, 3.625, -3.75, 3.875, -4.0,
        4.125, -4.25, 4.375, -4.5, 4.625, -4.75, 4.875, -5.0,
        5.125, -5.25, 5.375, -5.5, 5.625, -5.75, 5.875, -6.0,
        6.125, -6.25, 6.375, -6.5, 6.625, -6.75, 6.875, -7.0,
        7.125, -7.25, 7.375, -7.5, 7.625, -7.75, 7.875, -8.0,
    ]
    rms_gamma = [1.0 + 0.0078125 * idx for idx in range(64)]
    rms_observed = await run_lpu_vxm_rmsnorm(dut, rms_in, rms_gamma)
    rms_in_dequantized = []
    rms_gamma_dequantized = []
    for start in range(0, 64, TILE):
        row_bits, scale_exp = lpu.regular_fp8_row_quant_expected(rms_in[start:start + TILE])
        rms_in_dequantized.extend(lpu.dequantize_regular_fp8_row(row_bits, scale_exp))
        gamma_bits, gamma_scale = lpu.regular_fp8_row_quant_expected(rms_gamma[start:start + TILE])
        rms_gamma_dequantized.extend(lpu.dequantize_regular_fp8_row(gamma_bits, gamma_scale))

    rms_expected = rmsnorm_rows([rms_in_dequantized], rms_gamma_dequantized)[0]
    rms_expected_quantized = []
    for start in range(0, 64, TILE):
        row_bits, scale_exp = lpu.regular_fp8_row_quant_expected(rms_expected[start:start + TILE])
        rms_expected_quantized.extend(lpu.dequantize_regular_fp8_row(row_bits, scale_exp))
    assert len(rms_observed) == len(rms_expected_quantized)
    for idx, (observed, expected) in enumerate(zip(rms_observed, rms_expected_quantized)):
        assert observed == expected, (
            f"RMSNorm lane {idx} mismatch: got {observed}, expected {expected}; "
            f"observed={rms_observed}; expected={rms_expected_quantized}"
        )

    dut._log.info("Instruction-driven VXM ReLU/RoPE/softmax/residual/RMSNorm chunks matched expected output")


@cocotb.test()
async def test_lpu_stories260k_decode_throughput(dut):
    cocotb.start_soon(Clock(dut.clk, SIM_CLK_NS, unit="ns").start())

    config, vocab, id_to_token, weights = load_tiny_lm_export()
    assert config == MODEL_CONFIG
    assert KV_CACHE_LAYER_ROWS * config["layers"] <= 32768

    await lpu.reset_dut(dut)
    input_ids = direct_prompt_ids(vocab, PROMPT_PREFILL)
    kv_cache = init_kv_cache()
    generated_ids = list(input_ids)
    generated_tokens = []
    decode_cycles = []

    dut._log.info(
        'stories260k LPU decode throughput prompt words: "%s"',
        token_rows_to_string(PROMPT_PREFILL),
    )
    dut._log.info("stories260k LPU decode will generate %d tokens", DECODE_TOKENS)

    prefill_start_ns = get_sim_time(unit="ns")
    prefill_result = None
    for pos, token_id in enumerate(input_ids):
        prefill_result = await run_lpu_decode_token_kv_8(
            dut,
            token_id=token_id,
            token_pos=pos,
            kv_cache=kv_cache,
            id_to_token=id_to_token,
            weights=weights,
            label=f"prefill{pos}",
            emit_logits=(pos == len(input_ids) - 1),
        )
    prefill_cycles = (get_sim_time(unit="ns") - prefill_start_ns) / SIM_CLK_NS
    dut._log.info(
        "stories260k KV prefill cached %d tokens in %.1f cycles",
        len(input_ids),
        prefill_cycles,
    )

    if DECODE_TOKENS > 0:
        generated_ids.append(prefill_result["pred_id"])
        generated_tokens.append(prefill_result["pred_token"])
        dut._log.info(
            'decode[0] LPU token="%s" id=%d produced by final prefill logits',
            prefill_result["pred_token"],
            prefill_result["pred_id"],
        )
        dut._log.info("decode[0] LPU top5: %s", prefill_result["top5"])

    while len(generated_tokens) < DECODE_TOKENS:
        decode_idx = len(generated_tokens)
        start_ns = get_sim_time(unit="ns")
        result = await run_lpu_decode_token_kv_8(
            dut,
            token_id=generated_ids[-1],
            token_pos=len(generated_ids) - 1,
            kv_cache=kv_cache,
            id_to_token=id_to_token,
            weights=weights,
            label=f"decode{decode_idx}",
        )
        end_ns = get_sim_time(unit="ns")

        elapsed_cycles = (end_ns - start_ns) / SIM_CLK_NS
        decode_cycles.append(elapsed_cycles)
        generated_ids.append(result["pred_id"])
        generated_tokens.append(result["pred_token"])

        dut._log.info(
            'decode[%d] LPU token="%s" id=%d cycles=%.1f',
            decode_idx,
            result["pred_token"],
            result["pred_id"],
            elapsed_cycles,
        )
        dut._log.info("decode[%d] LPU top5: %s", decode_idx, result["top5"])

        assert 0 <= result["pred_id"] < config["vocab_size"]
        assert not result["pred_token"].startswith("<unused_")

    total_cycles = sum(decode_cycles)
    avg_cycles = total_cycles / len(decode_cycles) if decode_cycles else 0.0
    tokens_per_cycle = len(decode_cycles) / total_cycles if total_cycles else 0.0
    tokens_per_second_100mhz = tokens_per_cycle * 100_000_000.0

    dut._log.info(
        'stories260k generated continuation: "%s"',
        token_rows_to_string(generated_tokens),
    )
    dut._log.info(
        "stories260k decode throughput: generated_tokens=%d measured_incremental_decode_tokens=%d "
        "total_cycles=%.1f avg_cycles_per_token=%.1f "
        "tokens_per_cycle=%.8f tokens_per_second_at_100MHz=%.2f",
        len(generated_tokens),
        len(decode_cycles),
        total_cycles,
        avg_cycles,
        tokens_per_cycle,
        tokens_per_second_100mhz,
    )


@cocotb.test()
async def test_lpu_stories10k_decode_smoke(dut):
    cocotb.start_soon(Clock(dut.clk, SIM_CLK_NS, unit="ns").start())

    config, vocab, id_to_token, weights = load_stories10k_export()
    expected_config = {
        "vocab_size": 128,
        "dim": 32,
        "seq_len": 64,
        "layers": 1,
        "heads": 4,
        "kv_heads": 2,
        "ffn_dim": 48,
    }
    assert config == expected_config
    assert kv_cache_layer_rows(config) * config["layers"] <= 32768

    await lpu.reset_dut(dut)
    prompt = os.getenv("LPU_STORIES10K_PROMPT", "one day , lily").split()
    new_tokens = int(os.getenv("LPU_DECODE_TOKENS", "3"))
    input_ids = direct_prompt_ids(vocab, prompt)
    kv_cache = init_kv_cache(config)
    generated_ids = list(input_ids)
    generated_tokens = []
    decode_cycles = []

    dut._log.info('stories10k LPU prompt: "%s"', token_rows_to_string(prompt))
    dut._log.info("stories10k LPU decode will generate %d tokens", new_tokens)

    prefill_start_ns = get_sim_time(unit="ns")
    prefill_result = None
    for pos, token_id in enumerate(input_ids):
        prefill_result = await run_lpu_decode_token_kv_8(
            dut,
            token_id=token_id,
            token_pos=pos,
            kv_cache=kv_cache,
            id_to_token=id_to_token,
            weights=weights,
            label=f"stories10k_prefill{pos}",
            emit_logits=(pos == len(input_ids) - 1),
            config=config,
        )
    prefill_cycles = (get_sim_time(unit="ns") - prefill_start_ns) / SIM_CLK_NS
    dut._log.info(
        "stories10k KV prefill cached %d tokens in %.1f cycles",
        len(input_ids),
        prefill_cycles,
    )

    if new_tokens > 0:
        generated_ids.append(prefill_result["pred_id"])
        generated_tokens.append(prefill_result["pred_token"])
        dut._log.info(
            'stories10k decode[0] token="%s" id=%d produced by final prefill logits',
            prefill_result["pred_token"],
            prefill_result["pred_id"],
        )
        dut._log.info("stories10k decode[0] top5: %s", prefill_result["top5"])

    while len(generated_tokens) < new_tokens:
        decode_idx = len(generated_tokens)
        start_ns = get_sim_time(unit="ns")
        result = await run_lpu_decode_token_kv_8(
            dut,
            token_id=generated_ids[-1],
            token_pos=len(generated_ids) - 1,
            kv_cache=kv_cache,
            id_to_token=id_to_token,
            weights=weights,
            label=f"stories10k_decode{decode_idx}",
            config=config,
        )
        end_ns = get_sim_time(unit="ns")

        elapsed_cycles = (end_ns - start_ns) / SIM_CLK_NS
        decode_cycles.append(elapsed_cycles)
        generated_ids.append(result["pred_id"])
        generated_tokens.append(result["pred_token"])

        dut._log.info(
            'stories10k decode[%d] token="%s" id=%d cycles=%.1f',
            decode_idx,
            result["pred_token"],
            result["pred_id"],
            elapsed_cycles,
        )
        dut._log.info("stories10k decode[%d] top5: %s", decode_idx, result["top5"])

        assert 0 <= result["pred_id"] < config["vocab_size"]
        assert not result["pred_token"].startswith("<unused_")

    total_cycles = sum(decode_cycles)
    avg_cycles = total_cycles / len(decode_cycles) if decode_cycles else 0.0
    tokens_per_cycle = len(decode_cycles) / total_cycles if total_cycles else 0.0
    tokens_per_second_100mhz = tokens_per_cycle * 100_000_000.0

    dut._log.info(
        'stories10k generated continuation: "%s"',
        token_rows_to_string(generated_tokens),
    )
    dut._log.info(
        "stories10k decode throughput: generated_tokens=%d measured_incremental_decode_tokens=%d "
        "total_cycles=%.1f avg_cycles_per_token=%.1f "
        "tokens_per_cycle=%.8f tokens_per_second_at_100MHz=%.2f",
        len(generated_tokens),
        len(decode_cycles),
        total_cycles,
        avg_cycles,
        tokens_per_cycle,
        tokens_per_second_100mhz,
    )


@cocotb.test()
async def test_interactive_prompt(dut):
    import os
    import sys
    
    if os.getenv("INTERACTIVE") != "1":
        dut._log.info("Skipping interactive prompt test. Set INTERACTIVE=1 to run.")
        await Timer(1, unit="ns")
        return

    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    config, vocab, id_to_token, weights = load_tiny_lm_export()
    assert config == MODEL_CONFIG

    print("\n" + "="*60)
    print("Welcome to the Continuous Interactive LPU Inference Console!")
    print("Type your prompt (up to 3 words) and press Enter.")
    print("Type 'exit' or 'quit' to stop.")
    print("="*60 + "\n")

    while True:
        prompt_str = await get_user_input("LPU-Prompt> ")
        prompt_str = prompt_str.strip()
        if not prompt_str:
            continue
        if prompt_str.lower() in ["exit", "quit"]:
            break

        words = prompt_str.split()
        if not words:
            continue

        # Truncate prompt if it exceeds the maximum supported sequence length (4 tokens total, so 3 prompt tokens max)
        if len(words) > 3:
            words = words[-3:]
            print(f"[*] Note: Prompt truncated to last 3 tokens: {words}")

        # Run forward pass on the simulated LPU chip
        try:
            await lpu.reset_dut(dut)
            prompt_ids = encode_prompt(words, vocab)
            golden = tiny_lm_forward(prompt_ids, weights)
            seq_len = len(prompt_ids)

            # Start from token embeddings
            x = golden["x0"]

            for layer_idx in range(MODEL_CONFIG["layers"]):
                block_prefix = f"blocks.{layer_idx}"
                
                # 1. RMSNorm on VXM
                ln1_hw = []
                for row_idx, row in enumerate(x):
                    normalized_row = await run_forced_vxm_rmsnorm(
                        dut,
                        data=row,
                        gamma=weights[f"{block_prefix}.ln1.weight"],
                    )
                    ln1_hw.append(normalized_row)
                
                # 2. Q, K, V Projections on MXM
                q_hw = [[0.0 for _ in range(MODEL_CONFIG["dim"])] for _ in range(seq_len)]
                k_hw = [[0.0 for _ in range(MODEL_CONFIG["kv_heads"] * 8)] for _ in range(seq_len)]
                v_hw = [[0.0 for _ in range(MODEL_CONFIG["kv_heads"] * 8)] for _ in range(seq_len)]
                
                # Q projection
                for r_start in range(0, seq_len, 4):
                    r_end = min(r_start + 4, seq_len)
                    left_chunk = ln1_hw[r_start:r_end]
                    for start in range(0, MODEL_CONFIG["dim"], 4):
                        observed = await run_lpu_mxm_tile_full(
                            dut,
                            left_rows=left_chunk,
                            right_rows=weights[f"{block_prefix}.attn.q_proj.weight"][start:start+4],
                            label=f"L{layer_idx} Q proj tile {start}",
                        )
                        for r_idx in range(r_start, r_end):
                            for c_idx in range(4):
                                q_hw[r_idx][start + c_idx] = observed[r_idx - r_start][c_idx]
                            
                # K projection
                for r_start in range(0, seq_len, 4):
                    r_end = min(r_start + 4, seq_len)
                    left_chunk = ln1_hw[r_start:r_end]
                    for start in range(0, MODEL_CONFIG["kv_heads"] * 8, 4):
                        observed = await run_lpu_mxm_tile_full(
                            dut,
                            left_rows=left_chunk,
                            right_rows=weights[f"{block_prefix}.attn.k_proj.weight"][start:start+4],
                            label=f"L{layer_idx} K proj tile {start}",
                        )
                        for r_idx in range(r_start, r_end):
                            for c_idx in range(4):
                                k_hw[r_idx][start + c_idx] = observed[r_idx - r_start][c_idx]
                            
                # V projection
                for r_start in range(0, seq_len, 4):
                    r_end = min(r_start + 4, seq_len)
                    left_chunk = ln1_hw[r_start:r_end]
                    for start in range(0, MODEL_CONFIG["kv_heads"] * 8, 4):
                        observed = await run_lpu_mxm_tile_full(
                            dut,
                            left_rows=left_chunk,
                            right_rows=weights[f"{block_prefix}.attn.v_proj.weight"][start:start+4],
                            label=f"L{layer_idx} V proj tile {start}",
                        )
                        for r_idx in range(r_start, r_end):
                            for c_idx in range(4):
                                v_hw[r_idx][start + c_idx] = observed[r_idx - r_start][c_idx]
                            
                # 3. RoPE on VXM
                q_rot_hw = [[0.0 for _ in range(MODEL_CONFIG["dim"])] for _ in range(seq_len)]
                k_rot_hw = [[0.0 for _ in range(MODEL_CONFIG["kv_heads"] * 8)] for _ in range(seq_len)]
                
                # Apply RoPE for Q
                for t in range(seq_len):
                    cos_val, sin_val = get_rope_cos_sin(t, head_dim=8)
                    cos_bits = [q1_7_bits(c) for c in cos_val]
                    sin_bits = [q1_7_bits(s) for s in sin_val]
                    for h in range(8):
                        start = h * 8
                        rotated_chunk = await run_forced_vxm_rope_chunk(
                            dut,
                            data=q_hw[t][start:start+8],
                            cos_bits=cos_bits,
                            sin_bits=sin_bits,
                        )
                        for c_idx in range(8):
                            q_rot_hw[t][start + c_idx] = rotated_chunk[c_idx]
                            
                # Apply RoPE for K
                for t in range(seq_len):
                    cos_val, sin_val = get_rope_cos_sin(t, head_dim=8)
                    cos_bits = [q1_7_bits(c) for c in cos_val]
                    sin_bits = [q1_7_bits(s) for s in sin_val]
                    for kh in range(4):
                        start = kh * 8
                        rotated_chunk = await run_forced_vxm_rope_chunk(
                            dut,
                            data=k_hw[t][start:start+8],
                            cos_bits=cos_bits,
                            sin_bits=sin_bits,
                        )
                        for c_idx in range(8):
                            k_rot_hw[t][start + c_idx] = rotated_chunk[c_idx]

                # 4. Self-Attention Dot Product (Q @ K^T) and Softmax
                attn_out_hw = [[0.0 for _ in range(MODEL_CONFIG["dim"])] for _ in range(seq_len)]
                for h in range(8):
                    kh = h // 2
                    q_h = [q_rot_hw[t][h*8 : (h+1)*8] for t in range(seq_len)]
                    k_kh = [k_rot_hw[t][kh*8 : (kh+1)*8] for t in range(seq_len)]
                    v_kh = [v_hw[t][kh*8 : (kh+1)*8] for t in range(seq_len)]
                    
                    scores_h = [[0.0 for _ in range(seq_len)] for _ in range(seq_len)]
                    for q_start in range(0, seq_len, 4):
                        q_end = min(q_start + 4, seq_len)
                        q_left_chunk = q_h[q_start:q_end]
                        for k_start in range(0, seq_len, 4):
                            k_end = min(k_start + 4, seq_len)
                            k_right_chunk = k_kh[k_start:k_end]
                            
                            observed = await run_lpu_mxm_tile_full(
                                dut,
                                left_rows=q_left_chunk,
                                right_rows=k_right_chunk,
                                label=f"L{layer_idx} H{h} Q@K tile",
                            )
                            for r_idx in range(q_start, q_end):
                                for c_idx in range(k_start, k_end):
                                    scores_h[r_idx][c_idx] = observed[r_idx - q_start][c_idx - k_start]
                                
                    # Scale, mask and Softmax on VXM
                    probs_h = []
                    for row_idx, row in enumerate(scores_h):
                        scaled_row = [lpu.to_f32(val / math.sqrt(8)) for val in row]
                        masked_row = [val if col_idx <= row_idx else -1e30 for col_idx, val in enumerate(scaled_row)]
                        softmax_out = await run_forced_vxm_softmax(dut, masked_row)
                        probs_h.append(softmax_out)
                        
                    # Probs @ V on MXM
                    v_by_hidden = transpose(v_kh)
                    out_h = [[0.0 for _ in range(8)] for _ in range(seq_len)]
                    for r_start in range(0, seq_len, 4):
                        r_end = min(r_start + 4, seq_len)
                        probs_chunk = probs_h[r_start:r_end]
                        for start in range(0, 8, 4):
                            observed = await run_lpu_mxm_tile_full(
                                dut,
                                left_rows=probs_chunk,
                                right_rows=v_by_hidden[start:start+4],
                                label=f"L{layer_idx} H{h} Probs@V tile {start}",
                            )
                            for r_idx in range(r_start, r_end):
                                for c_idx in range(4):
                                    out_h[r_idx][start + c_idx] = observed[r_idx - r_start][c_idx]
                                
                    # Store head output
                    for r_idx in range(seq_len):
                        for c_idx in range(8):
                            attn_out_hw[r_idx][h*8 + c_idx] = out_h[r_idx][c_idx]
                            
                # 5. Attention Output Projection on MXM
                attn_proj_hw = [[0.0 for _ in range(MODEL_CONFIG["dim"])] for _ in range(seq_len)]
                for r_start in range(0, seq_len, 4):
                    r_end = min(r_start + 4, seq_len)
                    attn_out_chunk = attn_out_hw[r_start:r_end]
                    for start in range(0, MODEL_CONFIG["dim"], 4):
                        observed = await run_lpu_mxm_tile_full(
                            dut,
                            left_rows=attn_out_chunk,
                            right_rows=weights[f"{block_prefix}.attn.out_proj.weight"][start:start+4],
                            label=f"L{layer_idx} OutProj tile {start}",
                        )
                        for r_idx in range(r_start, r_end):
                            for c_idx in range(4):
                                attn_proj_hw[r_idx][start + c_idx] = observed[r_idx - r_start][c_idx]
                            
                # 6. Residual Addition (x = x + attn_proj_hw) on VXM
                x_after_attn_hw = []
                for row_idx in range(seq_len):
                    row_res = []
                    for chunk_idx in range(8):
                        start = chunk_idx * 8
                        res_chunk = await run_forced_vxm_residual_chunk(
                            dut,
                            base_chunk=x[row_idx][start:start+8],
                            delta_chunk=attn_proj_hw[row_idx][start:start+8],
                        )
                        row_res.extend(res_chunk)
                    x_after_attn_hw.append(row_res)
                    
                # 7. FFN block
                # RMSNorm
                ln2_hw = []
                for row_idx, row in enumerate(x_after_attn_hw):
                    normalized_row = await run_forced_vxm_rmsnorm(
                        dut,
                        data=row,
                        gamma=weights[f"{block_prefix}.ln2.weight"],
                    )
                    ln2_hw.append(normalized_row)
                    
                # FFN Gate Projection (W1) on MXM
                ffn_gate_hw = [[0.0 for _ in range(MODEL_CONFIG["ffn_dim"])] for _ in range(seq_len)]
                for r_start in range(0, seq_len, 4):
                    r_end = min(r_start + 4, seq_len)
                    ln2_chunk = ln2_hw[r_start:r_end]
                    for start in range(0, MODEL_CONFIG["ffn_dim"], 4):
                        observed = await run_lpu_mxm_tile_full(
                            dut,
                            left_rows=ln2_chunk,
                            right_rows=weights[f"{block_prefix}.ffn_gate.weight"][start:start+4],
                            label=f"L{layer_idx} FFN W1 tile {start}",
                        )
                        for r_idx in range(r_start, r_end):
                            for c_idx in range(4):
                                ffn_gate_hw[r_idx][start + c_idx] = observed[r_idx - r_start][c_idx]
                            
                # ReLU Activation on VXM
                ffn_hidden_hw = []
                for row_idx, row in enumerate(ffn_gate_hw):
                    row_relu = []
                    for chunk_idx in range(22):
                        start = chunk_idx * 8
                        relu_chunk = await run_forced_vxm_relu_chunk(
                            dut,
                            data=row[start:start+8],
                        )
                        row_relu.extend(relu_chunk)
                    ffn_hidden_hw.append(row_relu)
                    
                # FFN Down Projection (W2) on MXM
                ffn_down_hw = [[0.0 for _ in range(MODEL_CONFIG["dim"])] for _ in range(seq_len)]
                for r_start in range(0, seq_len, 4):
                    r_end = min(r_start + 4, seq_len)
                    hidden_r_chunk = ffn_hidden_hw[r_start:r_end]
                    for w_start in range(0, MODEL_CONFIG["dim"], 4):
                        w2_rows = weights[f"{block_prefix}.ffn_down.weight"][w_start : w_start + 4]
                        observed = await run_lpu_mxm_tile_full(
                            dut,
                            left_rows=hidden_r_chunk,
                            right_rows=w2_rows,
                            label=f"L{layer_idx} FFN W2 tile r{r_start} w{w_start}",
                        )
                        for r_idx in range(r_start, r_end):
                            for c_idx in range(4):
                                ffn_down_hw[r_idx][w_start + c_idx] = observed[r_idx - r_start][c_idx]
                            
                # Residual Addition (x = x_after_attn_hw + ffn_down_hw) on VXM
                x_next_hw = []
                for row_idx in range(seq_len):
                    row_res = []
                    for chunk_idx in range(8):
                        start = chunk_idx * 8
                        res_chunk = await run_forced_vxm_residual_chunk(
                            dut,
                            base_chunk=x_after_attn_hw[row_idx][start:start+8],
                            delta_chunk=ffn_down_hw[row_idx][start:start+8],
                        )
                        row_res.extend(res_chunk)
                    x_next_hw.append(row_res)
                
                # Output of this layer becomes input to the next layer
                x = x_next_hw
                
            # End of layers. Final RMSNorm!
            final_hw = []
            for row_idx, row in enumerate(x):
                normalized_row = await run_forced_vxm_rmsnorm(
                    dut,
                    data=row,
                    gamma=weights["ln_f.weight"],
                )
                final_hw.append(normalized_row)
                
            # LM Head on LPU!
            last_hidden_hw = final_hw[-1]
            hw_logits = [0.0 for _ in range(MODEL_CONFIG["vocab_size"])]
            for vocab_start in range(0, MODEL_CONFIG["vocab_size"], 4):
                observed = await run_lpu_mxm_tile_full(
                    dut,
                    left_rows=[last_hidden_hw],
                    right_rows=weights["lm_head.weight"][vocab_start:vocab_start + 4],
                    label=f"LM head tile {vocab_start}",
                )
                for lane in range(4):
                    token_id = vocab_start + lane
                    hw_logits[token_id] = observed[0][lane]

            # Process outputs
            pred_id = argmax(hw_logits)
            pred_token = id_to_token[pred_id]
            probs = softmax_rows([hw_logits])[0]

            top5_indices = sorted(range(len(hw_logits)), key=lambda idx: hw_logits[idx], reverse=True)[:5]

            # Print required format:
            # "the only thing I want in the response is the prompt + prediction and the top 5 highest predictions with scores"
            print("\n" + "="*50)
            print(f"Prompt: {' '.join(words)}")
            print(f"Prediction: {pred_token}")
            print("\nTop 5 predictions:")
            for rank, idx in enumerate(top5_indices):
                token_str = id_to_token[idx]
                score = hw_logits[idx]
                prob = probs[idx]
                print(f"  {rank+1}. {token_str:<15} (score: {score:8.3f}, prob: {prob*100:5.1f}%)")
            print("="*50 + "\n")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n[!] Error during LPU inference: {e}\n", file=sys.stderr)

    print("\nExiting interactive mode.\n")
