import sys
import torch
import json
from pathlib import Path
from train_tiny_lm import TinyLM

# Load mapping
VOCAB_PATH = Path("output/vocab.json")
MODEL_PATH = Path("tiny_lm_model.pt")

if not MODEL_PATH.exists():
    print(f"Pretrained model path {MODEL_PATH} not found. Please run: python train_tiny_lm.py to train it first!")
    sys.exit(1)

from tokenizer import Tokenizer
tokenizer = Tokenizer(VOCAB_PATH)

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load(MODEL_PATH, map_location=device)
config = checkpoint["config"]

VOCAB_SIZE = config.get("vocab_size", 256)
DIM = config.get("dim", 4)
SEQ_LEN = config.get("seq_len", 4)
LAYERS = config.get("layers", 1)
HEADS = config.get("heads", 1)
KV_HEADS = config.get("kv_heads", HEADS)
FFN_DIM = config.get("ffn_dim", 16)

PAD_ID = 0  # <unk> / byte fallback base
UNK_ID = tokenizer.vocab.get("<unk>", {}).get("id", 0)

# Load model architecture
model = TinyLM(
    vocab_size=VOCAB_SIZE,
    dim=DIM,
    seq_len=SEQ_LEN,
    layers=LAYERS,
    heads=HEADS,
    kv_heads=KV_HEADS,
    ffn_dim=FFN_DIM
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

def generate(prompt, max_new_tokens=5):
    ids = tokenizer.encode(prompt, bos=True, eos=False)

    for _ in range(max_new_tokens):
        # Only take the last SEQ_LEN - 1 tokens
        context = ids[-(SEQ_LEN - 1):]

        while len(context) < SEQ_LEN - 1:
            context.append(PAD_ID)

        x = torch.tensor([context], dtype=torch.long).to(device)

        with torch.no_grad():
            logits = model(x)

        # Get logits of the last real token
        real_len = min(len(ids), SEQ_LEN - 1)
        next_logits = logits[0, real_len - 1]

        next_id = int(torch.argmax(next_logits).item())
        ids.append(next_id)

        next_token = tokenizer.id_to_token.get(next_id, "")
        if next_token == "." or next_id == 2 or next_token == "\n</s>\n":
            break

    # Decode to text
    if len(ids) > 0 and ids[0] == 1:
        return tokenizer.decode(ids[1:])
    return tokenizer.decode(ids)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Prompt passed from CLI argument
        prompt = " ".join(sys.argv[1:])
        completed = generate(prompt)
        print(f"\nPrompt:      '{prompt}'")
        print(f"Completion:  '{completed}'")
    else:
        # Interactive shell
        print("\n=== TinyLM Interactive Generator ===")
        print("Type a prompt (e.g. 'messi is', 'cat wears', 'sky is') or type 'exit' to quit.\n")
        while True:
            try:
                prompt = input("Prompt > ")
                if prompt.strip().lower() == "exit":
                    break
                if not prompt.strip():
                    continue
                completed = generate(prompt)
                print(f"Gen    > {completed}\n")
            except (KeyboardInterrupt, EOFError):
                break
