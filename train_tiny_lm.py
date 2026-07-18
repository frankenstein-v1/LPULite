import json
import math
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Tiny LPU LM Training Script
# ============================================================

# Default settings / configuration constants
SEED = 42
DATA_DIR = Path("output")

TRAIN_PATH = DATA_DIR / "dataset_train.txt"
VAL_PATH = DATA_DIR / "dataset_val.txt"
VOCAB_PATH = DATA_DIR / "vocab.json"

MODEL_PATH = Path("tiny_lm_model.pt")
EXPORT_PATH = Path("tiny_lm_weights_export.json")

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
    global tokenizer, PAD_ID
    if seq_len is None:
        seq_len = SEQ_LEN
    ids = tokenizer.encode(line, bos=True, eos=False)

    if len(ids) > seq_len:
        ids = ids[:seq_len]

    while len(ids) < seq_len:
        ids.append(PAD_ID)

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

        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, self.kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(dim, self.kv_heads * self.head_dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

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

        if self.kv_heads != self.heads:
            num_queries_per_kv = self.heads // self.kv_heads
            k = k.repeat_interleave(num_queries_per_kv, dim=1)
            v = v.repeat_interleave(num_queries_per_kv, dim=1)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        mask = self.causal_mask[:T, :T]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        return out


class TinyTransformerBlock(nn.Module):
    def __init__(self, dim, heads, kv_heads, ffn_dim, seq_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = TinyCausalSelfAttention(dim, heads, kv_heads, seq_len)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
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

        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_emb = nn.Embedding(self.seq_len - 1, self.dim)

        self.blocks = nn.ModuleList([
            TinyTransformerBlock(self.dim, self.heads, self.kv_heads, self.ffn_dim, self.seq_len)
            for _ in range(self.layers)
        ])

        self.ln_f = nn.LayerNorm(self.dim)
        self.lm_head = nn.Linear(self.dim, self.vocab_size, bias=True)

    def forward(self, idx):
        B, T = idx.shape
        token_embeddings = self.token_emb(idx)
        positions = torch.arange(T, device=idx.device)
        position_embeddings = self.pos_emb(positions)
        x = token_embeddings + position_embeddings

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

    ids = tokenizer.encode(prompt, bos=True, eos=False)

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
        if next_token == "." or next_id == 2 or next_token == "\n</s>\n":
            break

    if len(ids) > 0 and ids[0] == 1:
        return tokenizer.decode(ids[1:])
    return tokenizer.decode(ids)


def tensor_to_list(t):
    return t.detach().cpu().numpy().tolist()


# ============================================================
# Main Script Execution
# ============================================================

if __name__ == "__main__":
    random.seed(SEED)
    torch.manual_seed(SEED)

    import argparse
    parser = argparse.ArgumentParser(description="Train Tiny LPU LM")
    parser.add_argument("--preset", type=str, default="tiny", choices=["tiny", "stories260k"], help="Preset configuration")
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
    args = parser.parse_args()

    # Apply preset defaults
    if args.preset == "stories260k":
        preset_config = {
            "dim": 64,
            "ffn_dim": 172,
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

    print(f"Using device: {DEVICE}")
    print(f"Configuration: dim={DIM}, ffn_dim={FFN_DIM}, layers={LAYERS}, heads={HEADS}, kv_heads={KV_HEADS}, seq_len={SEQ_LEN}, vocab_size={VOCAB_SIZE}")

    # Load vocab and initialize Tokenizer
    tokenizer = Tokenizer(VOCAB_PATH)
    vocab = tokenizer.vocab
    id_to_token = tokenizer.id_to_token

    assert len(vocab) == VOCAB_SIZE, f"Expected vocab size {VOCAB_SIZE}, got {len(vocab)}"
    PAD_ID = 0  # <unk> / byte fallback base
    UNK_ID = vocab.get("<unk>", {}).get("id", 0)

    # Load dataset
    train_data = load_dataset(TRAIN_PATH)
    val_data = load_dataset(VAL_PATH)

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
                MODEL_PATH,
            )
            print("  saved best model")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print()
            print("Validation loss stopped improving. Early stopping.")
            break

    # Load best model
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print()
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {MODEL_PATH}")

    # Generation tests
    print()
    print("Generation tests:")
    print()
    prompts = [
        "lebron is",
        "messi is",
        "ronaldo is",
        "ranvijay is",
        "satvik is",
        "cat wears",
        "lily likes",
        "sky is",
        "dog sees",
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

    with open(EXPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(export, f)

    print()
    print(f"Exported weights to: {EXPORT_PATH}")
