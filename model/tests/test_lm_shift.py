import json
from pathlib import Path

with open(Path(__file__).resolve().parents[1] / "output" / "vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

def token_id(name, fallback=0):
    value = vocab.get(name, fallback)
    return value.get("id", fallback) if isinstance(value, dict) else value

id_to_token = {token_id(tok): tok for tok in vocab}
unk_id = token_id("<unk>")
pad_id = token_id("<pad>", unk_id)

def encode(sentence, seq_len=4):
    tokens = sentence.lower().strip().split()
    ids = [token_id(tok, unk_id) for tok in tokens]

    while len(ids) < seq_len:
        ids.append(pad_id)

    ids = ids[:seq_len]
    return ids

sentence = "ranvijay is alpha ."

ids = encode(sentence, seq_len=4)

input_ids = ids[:-1]
target_ids = ids[1:]

print("Sentence:")
print(sentence)
print()

print("Full IDs:")
print(ids)
print()

print("Input IDs:")
print(input_ids)
print()

print("Target IDs:")
print(target_ids)
print()

print("Input tokens:")
print([id_to_token[i] for i in input_ids])
print()

print("Target tokens:")
print([id_to_token[i] for i in target_ids])
