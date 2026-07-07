import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Tiny LPU LM Training Script
#
# Uses:
#   output/dataset_train.txt
#   output/dataset_val.txt
#   output/vocab.json
#
# Produces:
#   tiny_lm_model.pt
#   tiny_lm_weights_export.json
# ============================================================

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = Path("output")

TRAIN_PATH = DATA_DIR / "dataset_train.txt"
VAL_PATH = DATA_DIR / "dataset_val.txt"
VOCAB_PATH = DATA_DIR / "vocab.json"

MODEL_PATH = Path("tiny_lm_model.pt")
EXPORT_PATH = Path("tiny_lm_weights_export.json")

# Model target from your spec
VOCAB_SIZE = 256
DIM = 8
SEQ_LEN = 4
LAYERS = 1
HEADS = 1
FFN_DIM = 16

BATCH_SIZE = 64
EPOCHS = 80
LR = 3e-3
PATIENCE = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")


# ============================================================
# 1. Load vocab
# ============================================================

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)

assert len(vocab) == VOCAB_SIZE, f"Expected vocab size 256, got {len(vocab)}"

id_to_token = {idx: tok for tok, idx in vocab.items()}

PAD_ID = vocab["<pad>"]
UNK_ID = vocab["<unk>"]


# ============================================================
# 2. Tokenization
# ============================================================

def encode_line(line, seq_len=SEQ_LEN):
    """
    Converts a sentence into token IDs and pads to SEQ_LEN.
    Example:
      "ranvijay is alpha ."
    becomes:
      [ranvijay_id, is_id, alpha_id, dot_id]
    """
    tokens = line.lower().strip().split()
    ids = [vocab.get(tok, UNK_ID) for tok in tokens]

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


train_data = load_dataset(TRAIN_PATH)
val_data = load_dataset(VAL_PATH)

print(f"Train examples: {len(train_data)}")
print(f"Val examples:   {len(val_data)}")


# ============================================================
# 3. Batch loader
# ============================================================

def get_batches(data, batch_size, shuffle=True):
    if shuffle:
        random.shuffle(data)

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]

        x = torch.tensor([item[0] for item in batch], dtype=torch.long)
        y = torch.tensor([item[1] for item in batch], dtype=torch.long)

        yield x.to(DEVICE), y.to(DEVICE)


# ============================================================
# 4. Tiny causal Transformer model
# ============================================================

class TinyCausalSelfAttention(nn.Module):
    def __init__(self, dim, heads, seq_len):
        super().__init__()

        assert heads == 1, "This tiny version assumes 1 head"

        self.dim = dim
        self.heads = heads
        self.seq_len = seq_len

        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

        # causal mask prevents token from seeing future tokens
        mask = torch.tril(torch.ones(seq_len - 1, seq_len - 1))
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        # x shape: [batch, time, dim]
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # attention scores: QK^T / sqrt(dim)
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(C)

        # mask future positions
        mask = self.causal_mask[:T, :T]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)

        out = attn @ v
        out = self.out_proj(out)

        return out


class TinyTransformerBlock(nn.Module):
    def __init__(self, dim, heads, ffn_dim, seq_len):
        super().__init__()

        self.ln1 = nn.LayerNorm(dim)
        self.attn = TinyCausalSelfAttention(dim, heads, seq_len)
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
    def __init__(self):
        super().__init__()

        self.token_emb = nn.Embedding(VOCAB_SIZE, DIM)
        self.pos_emb = nn.Embedding(SEQ_LEN - 1, DIM)

        self.blocks = nn.ModuleList([
            TinyTransformerBlock(DIM, HEADS, FFN_DIM, SEQ_LEN)
            for _ in range(LAYERS)
        ])

        self.ln_f = nn.LayerNorm(DIM)
        self.lm_head = nn.Linear(DIM, VOCAB_SIZE, bias=True)

    def forward(self, idx):
        # idx shape: [batch, time]
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


model = TinyLM().to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


# ============================================================
# 5. Loss and accuracy
# ============================================================

def compute_loss_and_accuracy(data):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for x, y in get_batches(data, BATCH_SIZE, shuffle=False):
            logits = model(x)

            # logits: [B, T, vocab]
            # y:      [B, T]
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
# 6. Training loop
# ============================================================

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


# ============================================================
# 7. Load best model
# ============================================================

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print()
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Best model saved to: {MODEL_PATH}")


# ============================================================
# 8. Decode/generate function
# ============================================================

def decode_ids(ids):
    tokens = []

    for idx in ids:
        tok = id_to_token[int(idx)]

        if tok == "<pad>":
            continue

        tokens.append(tok)

    return " ".join(tokens)


def generate(prompt, max_new_tokens=5):
    model.eval()

    tokens = prompt.lower().strip().split()
    ids = [vocab.get(tok, UNK_ID) for tok in tokens]

    for _ in range(max_new_tokens):
        # Keep only last SEQ_LEN - 1 tokens because that is the model input length.
        context = ids[-(SEQ_LEN - 1):]

        while len(context) < SEQ_LEN - 1:
            context.append(PAD_ID)

        x = torch.tensor([context], dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            logits = model(x)

        # Get prediction at last real position
        real_len = min(len(ids), SEQ_LEN - 1)
        next_logits = logits[0, real_len - 1]

        next_id = int(torch.argmax(next_logits).item())
        ids.append(next_id)

        next_token = id_to_token[next_id]

        if next_token == "." or next_token == "<pad>":
            break

    return decode_ids(ids)


# ============================================================
# 9. Test prompts
# ============================================================

print()
print("Generation tests:")
print()

prompts = [
    "lebron is",
    "messi is",
    "ronaldo is",
    "ranvijay is",
    "cat wears",
    "lily likes",
    "sky is",
    "dog sees",
]

for prompt in prompts:
    print(f"{prompt!r} -> {generate(prompt)}")


# ============================================================
# 10. Export weights to JSON
# ============================================================

def tensor_to_list(t):
    return t.detach().cpu().numpy().tolist()


export = {
    "config": {
        "vocab_size": VOCAB_SIZE,
        "dim": DIM,
        "seq_len": SEQ_LEN,
        "layers": LAYERS,
        "heads": HEADS,
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
