import json

with open("output/vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

pad_id = vocab["<pad>"]
unk_id = vocab["<unk>"]

def encode(sentence, seq_len=4):
    tokens = sentence.lower().strip().split()
    ids = [vocab.get(tok, unk_id) for tok in tokens]

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
