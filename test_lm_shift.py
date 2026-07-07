import json

with open("output/vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

id_to_token = {idx: tok for tok, idx in vocab.items()}

pad_id = vocab["<pad>"]
unk_id = vocab["<unk>"]

def encode(sentence, seq_len=4):
    tokens = sentence.lower().strip().split()
    ids = [vocab.get(tok, unk_id) for tok in tokens]

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
