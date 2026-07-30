import json
import math
import random
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Tiny LPU LM Training Script
# ============================================================

# Default settings / configuration constants
SEED = 42
MODEL_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = MODEL_DIR / "output"

TRAIN_PATH = DATA_DIR / "dataset_train.txt"
VAL_PATH = DATA_DIR / "dataset_val.txt"
VOCAB_PATH = DATA_DIR / "vocab.json"

MODEL_PATH = MODEL_DIR / "artifacts" / "tiny_lm_model.pt"
EXPORT_PATH = MODEL_DIR / "artifacts" / "tiny_lm_weights_export.json"

# Global variables that can be overridden dynamically at runtime
VOCAB_SIZE = 256
DIM = 4
SEQ_LEN = 4
LAYERS = 1
HEADS = 1
KV_HEADS = 1
FFN_DIM = 16

BATCH_SIZE = 64
EPOCHS = 80
LR = 3e-3
PATIENCE = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
from tokenizer import Tokenizer
tokenizer = None
vocab = {}
id_to_token = {}
PAD_ID = 0
UNK_ID = 1
model = None


# ============================================================
# Tokenization and Dataset Helpers
# ============================================================

def encode_line(line, seq_len=None):
    """Converts a sentence into token IDs and pads to seq_len."""
    global tokenizer, vocab, PAD_ID, UNK_ID
    if seq_len is None:
        seq_len = SEQ_LEN
    uses_byte_vocab = any(token.startswith("<0x") and token.endswith(">") for token in vocab)
    if uses_byte_vocab:
        ids = tokenizer.encode(line, bos=True, eos=False)
    else:
        bos_entry = vocab.get("<bos>", {"id": 2})
        bos_id = bos_entry["id"] if isinstance(bos_entry, dict) else bos_entry
        ids = [bos_id]
        for token in line.lower().split():
            entry = vocab.get(token)
            ids.append(entry["id"] if isinstance(entry, dict) else (entry if entry is not None else UNK_ID))

    if len(ids) > seq_len:
        ids = ids[:seq_len]

    while len(ids) < seq_len:
        ids.append(PAD_ID)

    return ids


def encode_prompt_text(prompt, bos=True, eos=False):
    global tokenizer, vocab, UNK_ID
    uses_byte_vocab = any(token.startswith("<0x") and token.endswith(">") for token in vocab)
    if uses_byte_vocab:
        return tokenizer.encode(prompt, bos=bos, eos=eos)

    ids = []
    if bos:
        bos_entry = vocab.get("<bos>", {"id": 2})
        ids.append(bos_entry["id"] if isinstance(bos_entry, dict) else bos_entry)
    for token in prompt.lower().split():
        entry = vocab.get(token)
        ids.append(entry["id"] if isinstance(entry, dict) else (entry if entry is not None else UNK_ID))
    if eos:
        eos_entry = vocab.get("<eos>", {"id": 3})
        ids.append(eos_entry["id"] if isinstance(eos_entry, dict) else eos_entry)
    return ids


def load_dataset(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            ids = encode_line(line)

            # Causal LM shift:
            # input:  token 0 to token SEQ_LEN - 2
            # target: token 1 to token SEQ_LEN - 1
            x = ids[:-1]
            y = ids[1:]
            examples.append((x, y))
    return examples


def get_batches(data, batch_size, shuffle=True):
    if shuffle:
        random.shuffle(data)

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        x = torch.tensor([item[0] for item in batch], dtype=torch.long)
        y = torch.tensor([item[1] for item in batch], dtype=torch.long)
        yield x.to(DEVICE), y.to(DEVICE)


def quantize_to_fp8_e5m2(x):
    # Quantize tensor x to FP8 E5M2
    x_fp32 = x.float()
    abs_x = torch.abs(x_fp32)
    # Clamp to avoid log of 0
    exp = torch.floor(torch.log2(torch.clamp(abs_x, min=1e-12)))
    exp_clamped = torch.clamp(exp, min=-14, max=15)
    # Scale to normalize the mantissa
    scaled = x_fp32 * torch.pow(2.0, -exp_clamped)
    # Round mantissa to 2 fractional bits
    rounded = torch.round(scaled * 4.0) / 4.0
    # Reconstruct
    q_x = rounded * torch.pow(2.0, exp_clamped)
    # Handle zero and underflow
    q_x = torch.where(abs_x < 7.629e-6, torch.zeros_like(q_x), q_x)
    # Clip to max representable value in FP8 E5M2 (57344.0)
    q_x = torch.clamp(q_x, min=-57344.0, max=57344.0)
    return q_x.type_as(x)


def fake_quantize_fp8_e5m2_row(x):
    # x shape: [..., D]
    if x.numel() == 0:
        return x
    with torch.no_grad():
        absmax = torch.max(torch.abs(x), dim=-1, keepdim=True).values
        scale_exp = torch.where(absmax > 0, torch.floor(torch.log2(torch.clamp(absmax, min=1e-12))), torch.zeros_like(absmax))
        scaled = x * torch.pow(2.0, -scale_exp)
        q_scaled = quantize_to_fp8_e5m2(scaled)
        dequant = q_scaled * torch.pow(2.0, scale_exp)
    return x + (dequant - x).detach()


class FakeQuantLinear(nn.Linear):
    def forward(self, x):
        # Fake-quantize input activations per row (along last dimension)
        x_q = fake_quantize_fp8_e5m2_row(x)
        # Fake-quantize weights per row
        w_q = fake_quantize_fp8_e5m2_row(self.weight)
        return F.linear(x_q, w_q, self.bias)


class FakeQuantEmbedding(nn.Embedding):
    def forward(self, x):
        # Fake-quantize embedding weights per row
        w_q = fake_quantize_fp8_e5m2_row(self.weight)
        return F.embedding(x, w_q, self.padding_idx, self.max_norm,
                           self.norm_type, self.scale_grad_by_freq, self.sparse)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_norm = x_fp32 * torch.rsqrt(variance + self.eps)
        return (x_norm * self.weight).type_as(x)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.repeat_interleave(freqs, 2, dim=-1)
        
        # Pre-quantize the cos and sin caches to FP8 E5M2
        cos_fp8 = quantize_to_fp8_e5m2(emb.cos())
        sin_fp8 = quantize_to_fp8_e5m2(emb.sin())
        
        self.register_buffer("cos_cached", cos_fp8)
        self.register_buffer("sin_cached", sin_fp8)

    def forward(self, x):
        t = x.shape[2]
        return self.cos_cached[:t, :], self.sin_cached[:t, :]


def apply_rope(x, cos, sin):
    # x: [B, H, T, D]
    # cos, sin: [T, D]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    
    cos_even = cos[..., 0::2]
    sin_even = sin[..., 0::2]
    
    y_even = x_even * cos_even - x_odd * sin_even
    y_odd = x_even * sin_even + x_odd * cos_even
    
    out = torch.empty_like(x)
    out[..., 0::2] = y_even
    out[..., 1::2] = y_odd
    return out


# ============================================================
# Tiny causal Transformer model with GQA & ReLU
# ============================================================

class TinyCausalSelfAttention(nn.Module):
    def __init__(self, dim, heads, kv_heads, seq_len):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.kv_heads = kv_heads if kv_heads is not None else heads
        self.seq_len = seq_len
        self.head_dim = dim // heads

        assert dim % heads == 0, f"dim {dim} must be divisible by heads {heads}"
        assert heads % self.kv_heads == 0, f"heads {heads} must be divisible by kv_heads {self.kv_heads}"

        self.q_proj = FakeQuantLinear(dim, dim, bias=False)
        self.k_proj = FakeQuantLinear(dim, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = FakeQuantLinear(dim, self.kv_heads * self.head_dim, bias=False)
        self.out_proj = FakeQuantLinear(dim, dim, bias=False)

        self.rope = RotaryEmbedding(self.head_dim, seq_len)

        mask = torch.tril(torch.ones(seq_len - 1, seq_len - 1))
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        cos, sin = self.rope(q)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.kv_heads != self.heads:
            num_queries_per_kv = self.heads // self.kv_heads
            k = k.repeat_interleave(num_queries_per_kv, dim=1)
            v = v.repeat_interleave(num_queries_per_kv, dim=1)

        # Fake-quantize Q, K, V activations before matmul
        q_q = fake_quantize_fp8_e5m2_row(q)
        k_q = fake_quantize_fp8_e5m2_row(k)
        v_q = fake_quantize_fp8_e5m2_row(v)

        # Attention scores: QK^T / sqrt(head_dim)
        scores = q_q @ k_q.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        mask = self.causal_mask[:T, :T]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        
        # Fake-quantize attention probs before multiplying with V
        attn_q = fake_quantize_fp8_e5m2_row(attn)

        out = attn_q @ v_q
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        return out


class TinyTransformerBlock(nn.Module):
    def __init__(self, dim, heads, kv_heads, ffn_dim, seq_len):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = TinyCausalSelfAttention(dim, heads, kv_heads, seq_len)
        self.ln2 = RMSNorm(dim)
        self.ffn_gate = FakeQuantLinear(dim, ffn_dim, bias=False)
        self.ffn_down = FakeQuantLinear(ffn_dim, dim, bias=False)

    def forward(self, x):
        # Attention block
        norm1 = self.ln1(x)
        x = x + self.attn(norm1)
        
        # FFN block
        norm2 = self.ln2(x)
        ffn1 = self.ffn_gate(norm2)
        
        # Activation function: ReLU
        act = F.relu(ffn1)
        
        # Fake-quantize activations after ReLU before down projection
        act_q = fake_quantize_fp8_e5m2_row(act)
        
        ffn2 = self.ffn_down(act_q)
        x = x + ffn2
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab_size=256, dim=4, seq_len=4, layers=1, heads=1, kv_heads=1, ffn_dim=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.seq_len = seq_len
        self.layers = layers
        self.heads = heads
        self.kv_heads = kv_heads
        self.ffn_dim = ffn_dim

        self.token_emb = FakeQuantEmbedding(self.vocab_size, self.dim)

        self.blocks = nn.ModuleList([
            TinyTransformerBlock(self.dim, self.heads, self.kv_heads, self.ffn_dim, self.seq_len)
            for _ in range(self.layers)
        ])

        self.ln_f = RMSNorm(self.dim)
        self.lm_head = FakeQuantLinear(self.dim, self.vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        x = self.token_emb(idx)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


# ============================================================
# Evaluation Helper
# ============================================================

def compute_loss_and_accuracy(data):
    global model, BATCH_SIZE, VOCAB_SIZE, PAD_ID, DEVICE
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for x, y in get_batches(data, BATCH_SIZE, shuffle=False):
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                y.reshape(-1),
                ignore_index=PAD_ID,
            )
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            mask = y != PAD_ID
            correct = (preds == y) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

    avg_loss = total_loss / max(1, math.ceil(len(data) / BATCH_SIZE))
    accuracy = total_correct / max(1, total_tokens)
    return avg_loss, accuracy


# ============================================================
# Decode / Generation Helpers
# ============================================================

def decode_ids(ids):
    global tokenizer
    return tokenizer.decode(ids)


def generate(prompt, max_new_tokens=5):
    global model, tokenizer, SEQ_LEN, PAD_ID, DEVICE
    model.eval()

    ids = encode_prompt_text(prompt, bos=True, eos=False)

    for _ in range(max_new_tokens):
        context = ids[-(SEQ_LEN - 1):]
        while len(context) < SEQ_LEN - 1:
            context.append(PAD_ID)

        x = torch.tensor([context], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            logits = model(x)

        real_len = min(len(ids), SEQ_LEN - 1)
        next_logits = logits[0, real_len - 1]
        next_id = int(torch.argmax(next_logits).item())
        ids.append(next_id)
        next_token = tokenizer.id_to_token.get(next_id, "")
        if next_token in ("<eos>", "\n</s>\n") or next_id in (2, 3):
            break

    if len(ids) > 0 and ids[0] == 1:
        return tokenizer.decode(ids[1:])
    return tokenizer.decode(ids)


def tensor_to_list(t):
    return t.detach().cpu().tolist()


# ============================================================
# Main Script Execution
# ============================================================

if __name__ == "__main__":
    random.seed(SEED)
    torch.manual_seed(SEED)

    import argparse
    parser = argparse.ArgumentParser(description="Train Tiny LPU LM")
    parser.add_argument("--preset", type=str, default="tiny", choices=["tiny", "stories10k", "stories260k", "stories288k_lpu", "qa476k_lpu"], help="Preset configuration")
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--ffn-dim", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--kv-heads", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--export-path", type=Path, default=None)
    args = parser.parse_args()

    # Apply preset defaults
    if args.preset == "qa476k_lpu":
        preset_config = {
            "dim": 64,
            "ffn_dim": 224,
            "layers": 10,
            "heads": 4,
            "kv_heads": 2,
            "seq_len": 512,
            "vocab_size": 512,
            "epochs": 50,
            "batch_size": 32,
            "lr": 2e-3,
            "patience": 10,
        }
    elif args.preset == "stories288k_lpu":
        preset_config = {
            "dim": 64,
            "ffn_dim": 192,
            "layers": 6,
            "heads": 4,
            "kv_heads": 2,
            "seq_len": 512,
            "vocab_size": 512,
            "epochs": 80,
            "batch_size": 32,
            "lr": 1.5e-3,
            "patience": 10,
        }
    elif args.preset == "stories260k":
        preset_config = {
            "dim": 64,
            "ffn_dim": 176,
            "layers": 5,
            "heads": 8,
            "kv_heads": 4,
            "seq_len": 512,
            "vocab_size": 512,
            "epochs": 80,
            "batch_size": 32,
            "lr": 1e-3,
            "patience": 10,
        }
    elif args.preset == "stories10k":
        preset_config = {
            "dim": 32,
            "ffn_dim": 48,
            "layers": 1,
            "heads": 4,
            "kv_heads": 2,
            "seq_len": 64,
            "vocab_size": 128,
            "epochs": 40,
            "batch_size": 64,
            "lr": 2e-3,
            "patience": 10,
        }
    else:  # tiny
        preset_config = {
            "dim": 4,
            "ffn_dim": 16,
            "layers": 1,
            "heads": 1,
            "kv_heads": 1,
            "seq_len": 4,
            "vocab_size": 256,
            "epochs": 80,
            "batch_size": 64,
            "lr": 3e-3,
            "patience": 10,
        }

    # Override with CLI arguments if provided
    DIM = args.dim if args.dim is not None else preset_config["dim"]
    FFN_DIM = args.ffn_dim if args.ffn_dim is not None else preset_config["ffn_dim"]
    LAYERS = args.layers if args.layers is not None else preset_config["layers"]
    HEADS = args.heads if args.heads is not None else preset_config["heads"]
    KV_HEADS = args.kv_heads if args.kv_heads is not None else preset_config.get("kv_heads", HEADS)
    SEQ_LEN = args.seq_len if args.seq_len is not None else preset_config["seq_len"]
    VOCAB_SIZE = args.vocab_size if args.vocab_size is not None else preset_config["vocab_size"]
    EPOCHS = args.epochs if args.epochs is not None else preset_config["epochs"]
    BATCH_SIZE = args.batch_size if args.batch_size is not None else preset_config["batch_size"]
    LR = args.lr if args.lr is not None else preset_config["lr"]
    PATIENCE = args.patience if args.patience is not None else preset_config["patience"]
    if args.preset == "qa476k_lpu":
        default_data_dir = Path("model") / "qa_facts"
        default_model_path = Path("model") / "qa_facts" / "qa476k_model.pt"
        default_export_path = Path("model") / "qa_facts" / "qa476k_weights_export.json"
    elif args.preset == "stories288k_lpu":
        default_data_dir = Path("model") / "stories288k"
        default_model_path = Path("model") / "stories288k" / "stories288k_model.pt"
        default_export_path = Path("model") / "stories288k" / "stories288k_weights_export.json"
    elif args.preset == "stories10k":
        default_data_dir = Path("model") / "stories10k"
        default_model_path = Path("model") / "stories10k" / "stories10k_model.pt"
        default_export_path = Path("model") / "stories10k" / "stories10k_weights_export.json"
    else:
        default_data_dir = DATA_DIR
        default_model_path = MODEL_PATH
        default_export_path = EXPORT_PATH

    data_dir = args.data_dir if args.data_dir is not None else default_data_dir
    model_path = args.model_path if args.model_path is not None else default_model_path
    export_path = args.export_path if args.export_path is not None else default_export_path
    train_path = data_dir / "dataset_train.txt"
    val_path = data_dir / "dataset_val.txt"
    vocab_path = data_dir / "vocab.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"Configuration: dim={DIM}, ffn_dim={FFN_DIM}, layers={LAYERS}, heads={HEADS}, kv_heads={KV_HEADS}, seq_len={SEQ_LEN}, vocab_size={VOCAB_SIZE}")
    print(f"Data directory: {data_dir}")
    print(f"Model path:     {model_path}")
    print(f"Export path:    {export_path}")

    # Load vocab and initialize Tokenizer
    tokenizer = Tokenizer(vocab_path)
    vocab = tokenizer.vocab
    id_to_token = tokenizer.id_to_token

    assert len(vocab) == VOCAB_SIZE, f"Expected vocab size {VOCAB_SIZE}, got {len(vocab)}"
    PAD_ID = 0  # <unk> / byte fallback base
    UNK_ID = vocab.get("<unk>", {}).get("id", 0)

    # Load dataset
    train_data = load_dataset(train_path)
    val_data = load_dataset(val_path)

    print(f"Train examples: {len(train_data)}")
    print(f"Val examples:   {len(val_data)}")

    # Instantiate model
    model = TinyLM(
        vocab_size=VOCAB_SIZE,
        dim=DIM,
        seq_len=SEQ_LEN,
        layers=LAYERS,
        heads=HEADS,
        kv_heads=KV_HEADS,
        ffn_dim=FFN_DIM
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    patience_counter = 0

    print()
    print("Starting training...")
    print()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        num_batches = 0

        for x, y in get_batches(train_data, BATCH_SIZE, shuffle=True):
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                y.reshape(-1),
                ignore_index=PAD_ID,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            num_batches += 1

        train_loss = total_train_loss / max(1, num_batches)
        val_loss, val_acc = compute_loss_and_accuracy(val_data)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss: {train_loss:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"val acc: {val_acc * 100:.2f}%"
        )

        # Save best model only
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                    "config": {
                        "vocab_size": VOCAB_SIZE,
                        "dim": DIM,
                        "seq_len": SEQ_LEN,
                        "layers": LAYERS,
                        "heads": HEADS,
                        "kv_heads": KV_HEADS,
                        "ffn_dim": FFN_DIM,
                    },
                },
                model_path,
            )
            print("  saved best model")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print()
            print("Validation loss stopped improving. Early stopping.")
            break

    # Load best model
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print()
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {model_path}")

    # Generation tests
    print()
    print("Generation tests:")
    print()
    prompts = [
        "one day , lily",
        "tom has a",
        "mia finds a",
        "max sees the",
        "lily and tom",
        "once upon a time",
        "anna sees mia",
        "the dog is",
        "sam went to the",
    ]
    for prompt in prompts:
        print(f"{prompt!r} -> {generate(prompt)}")

    # Export weights
    export = {
        "config": {
            "vocab_size": VOCAB_SIZE,
            "dim": DIM,
            "seq_len": SEQ_LEN,
            "layers": LAYERS,
            "heads": HEADS,
            "kv_heads": KV_HEADS,
            "ffn_dim": FFN_DIM,
        },
        "vocab": vocab,
        "weights": {},
    }
    state = model.state_dict()
    for name, tensor in state.items():
        export["weights"][name] = tensor_to_list(tensor)

    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(export, f)

    print()
    print(f"Exported weights to: {export_path}")
