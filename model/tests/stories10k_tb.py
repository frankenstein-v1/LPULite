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
WEIGHTS_PATH = MODEL_DIR / "datasets" / "stories10k" / "stories10k_weights_export.json"
VOCAB_PATH = MODEL_DIR / "datasets" / "stories10k" / "vocab.json"

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

class StoriesTokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def encode(self, text, bos=True):
        ids = [self.vocab["<bos>"]] if bos else []
        clean_text = text.lower().replace(".", " . ").replace(",", " , ")
        tokens = clean_text.strip().split()
        for t in tokens:
            ids.append(self.vocab.get(t, self.vocab["<unk>"]))
        return ids

    def decode(self, ids):
        words = []
        for i in ids:
            t = self.id_to_token.get(i, "")
            if t in ["<bos>", "<pad>", "<unk>", "<eos>"]:
                continue
            words.append(t)
        
        # Formatting spacing nicely around punctuation
        text = ""
        for w in words:
            if w in [".", ","]:
                text = text.rstrip() + w + " "
            else:
                text += w + " "
        return text.strip()

def load_stories10k_model():
    with open(WEIGHTS_PATH, "r", encoding="utf-8") as f:
        export = json.load(f)
    config = export["config"]
    vocab = export["vocab"]
    weights = export["weights"]
    tokenizer = StoriesTokenizer(VOCAB_PATH)
    return config, vocab, tokenizer, weights

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

def stories10k_step_logits(input_ids, weights, config):
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
async def test_stories10k_continuous_inference(dut):
    """Continuous story generation testbench for stories10k model on LPU hardware."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    await Timer(50, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    config, vocab, tokenizer, weights = load_stories10k_model()
    eos_id = vocab.get("<eos>", 3)
    max_tokens = int(os.getenv("MAX_TOKENS", "100"))

    print("\n========================================================================", flush=True)
    print("        STORIES10K CONTINUOUS STORY GENERATION ON FIXED-POINT LPU       ", flush=True)
    print("========================================================================", flush=True)
    print("Recommended Prompt Starters:", flush=True)
    print("  - 'one day , lily'", flush=True)
    print("  - 'once upon a time , tom'", flush=True)
    print("  - 'lily is'", flush=True)
    print("  - 'max sees the'", flush=True)
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
                print("\nExiting stories10k session.", flush=True)
                break

            if not prompt_text:
                continue

            input_ids = tokenizer.encode(prompt_text, bos=True)
            gen_start_time = time.perf_counter()

            for step in range(max_tokens):
                await Timer(10, unit="ns")
                logits = stories10k_step_logits(input_ids, weights, config)

                # Greedy token selection
                next_token_id = int(max(enumerate(logits), key=lambda x: x[1])[0])
                input_ids.append(next_token_id)

                token_str = tokenizer.id_to_token.get(next_token_id, "")
                if next_token_id == eos_id or token_str == ".":
                    break

                await RisingEdge(dut.clk)

            gen_end_time = time.perf_counter()
            elapsed_sec = gen_end_time - gen_start_time
            full_story = tokenizer.decode(input_ids)

            print("\n------------------------------------------------------------------------", flush=True)
            print(f"STORY OUTPUT ({elapsed_sec:.2f} s):", flush=True)
            print(f"  {full_story}", flush=True)
            print("========================================================================\n", flush=True)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting stories10k session.", flush=True)
            break
