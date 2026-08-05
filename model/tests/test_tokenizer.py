import json
from pathlib import Path

with open(Path(__file__).resolve().parents[1] / "datasets" / "tiny_lm" / "vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

def token_id(name, fallback=0):
    value = vocab.get(name, fallback)
    return value.get("id", fallback) if isinstance(value, dict) else value

unk_id = token_id("<unk>")
pad_id = token_id("<pad>", unk_id)

def encode(sentence, seq_len=4):
    tokens = sentence.lower().strip().split()
    ids = [token_id(tok, unk_id) for tok in tokens]

    # Pad to seq_len
    while len(ids) < seq_len:
        ids.append(pad_id)

    # Cut off if too long
    ids = ids[:seq_len]

    return tokens, ids

sentence = "ranvijay is alpha ."

tokens, ids = encode(sentence)

print("Sentence:")
print(sentence)
print()
print("Tokens:")
print(tokens)
print()
print("Token IDs:")
print(ids)
