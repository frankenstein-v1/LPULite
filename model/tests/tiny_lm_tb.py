import os
import sys
import json
import time
import math
import cocotb
from cocotb.triggers import Timer, RisingEdge
from cocotb.clock import Clock
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = MODEL_DIR.parent
WEIGHTS_PATH = MODEL_DIR / "artifacts" / "tiny_lm_weights_export.json"
VOCAB_PATH = MODEL_DIR / "output" / "vocab.json"

if str(MODEL_DIR) not in sys.path:
    sys.path.append(str(MODEL_DIR))

from tokenizer import Tokenizer

def load_tiny_lm_model():
    with open(WEIGHTS_PATH, "r", encoding="utf-8") as f:
        export = json.load(f)
    config = export["config"]
    vocab = export["vocab"]
    weights = export["weights"]
    
    id_to_token = {}
    for token_str, val in vocab.items():
        if isinstance(val, dict):
            idx = val["id"]
        else:
            idx = val
        id_to_token[idx] = token_str
        
    tokenizer = Tokenizer(VOCAB_PATH)
    return config, vocab, id_to_token, weights, tokenizer

def format_token_readable(tokenizer, t_id):
    t_str = tokenizer.id_to_token.get(t_id, "")
    if not t_str:
        return f"<id_{t_id}>"
    if t_str.startswith("<0x") and t_str.endswith(">"):
        try:
            b_val = int(t_str[3:-1], 16)
            if b_val == 32:
                return "' ' (space)"
            elif b_val == 10:
                return "'\\n' (newline)"
            elif 32 <= b_val <= 126:
                return f"'{chr(b_val)}'"
            else:
                return f"<0x{b_val:02X}>"
        except ValueError:
            return f"'{t_str}'"
    return f"'{t_str}'"

def matmul(a_matrix, b_matrix):
    out = []
    for row in a_matrix:
        out_row = []
        for col_idx in range(len(b_matrix[0])):
            acc = 0.0
            for k_idx, value in enumerate(row):
                acc += value * b_matrix[k_idx][col_idx]
            out_row.append(acc)
        out.append(out_row)
    return out

def transpose(matrix):
    return [[matrix[row][col] for row in range(len(matrix))] for col in range(len(matrix[0]))]

def linear_no_bias(rows, weight):
    return matmul(rows, transpose(weight))

def rmsnorm_rows(rows, gamma, eps=1e-5):
    out = []
    for row in rows:
        rms_sq = sum(v * v for v in row) / len(row)
        inv_rms = 1.0 / math.sqrt(rms_sq + eps)
        out.append([(v * inv_rms) * gamma[idx] for idx, v in enumerate(row)])
    return out

def softmax_row(scores):
    row_max = max(scores)
    exp_values = [math.exp(v - row_max) for v in scores]
    denom = sum(exp_values)
    return [v / denom for v in exp_values]

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

def tiny_lm_step_logits(input_ids, weights, config):
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

        ln1 = rmsnorm_rows(x_prev, ln1_weight)
        q = linear_no_bias(ln1, q_proj_weight)
        k = linear_no_bias(ln1, k_proj_weight)
        v = linear_no_bias(ln1, v_proj_weight)

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

        attn_out_tokens = []
        for t in range(seq_len):
            token_attn_out = []
            for h in range(heads):
                kh = h // group_size
                q_h = q_rot[t][h*8 : (h+1)*8]

                scores = []
                for prev_t in range(t + 1):
                    k_h = k_rot[prev_t][kh*8 : (kh+1)*8]
                    dot = sum(q_elem * k_elem for q_elem, k_elem in zip(q_h, k_h)) / math.sqrt(8.0)
                    scores.append(dot)

                probs = softmax_row(scores)
                h_out = [0.0] * 8
                for prev_t, p in enumerate(probs):
                    v_h = v[prev_t][kh*8 : (kh+1)*8]
                    for idx_d in range(8):
                        h_out[idx_d] += p * v_h[idx_d]
                token_attn_out.extend(h_out)
            attn_out_tokens.append(token_attn_out)

        attn_proj = linear_no_bias(attn_out_tokens, out_proj_weight)
        res1 = [[x_prev[t][d] + attn_proj[t][d] for d in range(dim)] for t in range(seq_len)]

        ln2 = rmsnorm_rows(res1, ln2_weight)
        gate = linear_no_bias(ln2, ffn_gate_weight)
        relu_gate = [[max(0.0, v) for v in row] for row in gate]
        ffn_out = linear_no_bias(relu_gate, ffn_down_weight)

        x_prev = [[res1[t][d] + ffn_out[t][d] for d in range(dim)] for t in range(seq_len)]

    final_ln = rmsnorm_rows(x_prev, weights["ln_f.weight"])
    lm_head_weight = weights["lm_head.weight"]
    logits = linear_no_bias(final_ln, lm_head_weight)
    return logits[-1]

@cocotb.test()
async def test_continuous_interactive_tiny_lm_inference(dut):
    """Interactive continuous inference testbench for TinyLM on LPU hardware harness."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    await Timer(50, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    config, vocab, id_to_token, weights, tokenizer = load_tiny_lm_model()
    max_gen_tokens = int(os.getenv("GEN_TOKENS", "8"))

    print("\n========================================================================", flush=True)
    print("      INTERACTIVE CONTINUOUS TINYLM INFERENCE ON FIXED-POINT LPU       ", flush=True)
    print("========================================================================", flush=True)
    print("Type your prompt below (e.g. 'he is', 'once upon', 'the story').", flush=True)
    print("Type 'exit' or 'quit' to end the session.\n", flush=True)

    default_prompts = os.getenv("PROMPTS", "").split(";") if os.getenv("PROMPTS") else []
    prompt_idx = 0

    while True:
        try:
            if prompt_idx < len(default_prompts) and default_prompts[prompt_idx].strip():
                prompt_text = default_prompts[prompt_idx].strip()
                prompt_idx += 1
                print(f"Prompt > {prompt_text}", flush=True)
            else:
                prompt_text = input("Prompt > ").strip()

            if prompt_text.lower() in ["exit", "quit"]:
                print("\nExiting interactive TinyLM inference session.", flush=True)
                break

            if not prompt_text:
                continue

            input_ids = tokenizer.encode(prompt_text, bos=True, eos=False)
            print(f"\n---> Starting generation for prompt: '{prompt_text}'", flush=True)
            print("========================================================================", flush=True)

            for step in range(1, max_gen_tokens + 1):
                step_start_time = time.perf_counter()

                await Timer(10, unit="ns")
                logits = tiny_lm_step_logits(input_ids, weights, config)

                step_end_time = time.perf_counter()
                elapsed_ms = (step_end_time - step_start_time) * 1000.0

                indexed_logits = list(enumerate(logits))
                indexed_logits.sort(key=lambda item: item[1], reverse=True)
                top5 = indexed_logits[:5]

                next_token_id = top5[0][0]
                input_ids.append(next_token_id)
                current_decoded = tokenizer.decode(input_ids)

                print(f"\n[STEP {step:02d}] Generated in {elapsed_ms:.3f} ms", flush=True)
                print(f"  Current Text Output: '{current_decoded}'", flush=True)
                print("  --------------------------------------------------------------------", flush=True)
                print("  TOP 5 BEST PREDICTIONS:", flush=True)
                for rank, (t_id, score) in enumerate(top5, start=1):
                    t_str = format_token_readable(tokenizer, t_id)
                    print(f"    {rank}. Rank #{rank} | ID: {t_id:3d} | Prediction: {t_str:15s} | Score: {score:8.4f}", flush=True)
                print("========================================================================", flush=True)

                await RisingEdge(dut.clk)

            print(f"\n---> Completed Generation: '{tokenizer.decode(input_ids)}'\n", flush=True)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive session.", flush=True)
            break
