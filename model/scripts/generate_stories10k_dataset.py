import argparse
import json
import random
from pathlib import Path


parser = argparse.ArgumentParser(description="Generate controlled stories10k dataset")
parser.add_argument("--out-dir", type=Path, default=Path("model") / "stories10k")
parser.add_argument("--vocab-size", type=int, default=128)
args = parser.parse_args()

random.seed(42)
out_dir = args.out_dir
out_dir.mkdir(parents=True, exist_ok=True)

special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
names = ["lily", "tom", "sam", "mia", "max", "anna", "tim", "zoe"]
animals = ["cat", "dog", "bird", "fish", "duck", "frog"]
objects = ["ball", "toy", "book", "cake", "hat", "box", "kite", "train", "apple", "cookie"]
places = ["park", "house", "room", "yard", "school", "garden"]
adjectives = ["big", "small", "red", "blue", "happy", "sad", "fast", "slow", "kind", "funny"]
verbs = ["is", "sees", "likes", "has", "finds", "helps", "wears", "eats", "runs", "plays"]
connectors = ["and", "then", "so", "because", "with", "in", "the", "a", "to"]
story_words = ["one", "day", "once", "upon", "time", "there", "was", "went", "home", "friend", "thanks", "smiles"]
punctuation = [".", ","]


def add(sentence_set, text):
    sentence_set.add(" ".join(text.lower().strip().split()))


sentences = set()

# Short next-token patterns inherited from the older toy dataset.
for name in names:
    for adj in adjectives:
        add(sentences, f"{name} is {adj} .")
    for obj in objects:
        add(sentences, f"{name} has a {obj} .")
        add(sentences, f"{name} finds a {obj} .")
        add(sentences, f"{name} likes the {obj} .")
    for place in places:
        add(sentences, f"{name} went to the {place} .")

for animal in animals:
    for obj in objects:
        add(sentences, f"{animal} sees the {obj} .")
        add(sentences, f"{animal} likes the {obj} .")
    for adj in adjectives:
        add(sentences, f"{animal} is {adj} .")

# Longer controlled stories so greedy generation can continue for multiple tokens.
story_templates = []
for name in names:
    for friend in names:
        if friend == name:
            continue
        for obj in objects[:6]:
            for place in places[:4]:
                story_templates.extend([
                    f"one day , {name} went to the {place} . {name} found a {obj} . {friend} saw the {obj} . {name} and {friend} played with the {obj} .",
                    f"once upon a time , {name} had a {obj} . the {obj} was {random.choice(adjectives)} . {friend} helped {name} . they were happy .",
                    f"{name} is {random.choice(adjectives)} . {name} sees {friend} in the {place} . {friend} has a {obj} . they play and smile .",
                ])

for story in story_templates:
    add(sentences, story)

# Explicit demo continuations with repeated weight.
demo_bias = [
    "one day , lily went to the park . lily found a ball . tom saw the ball . lily and tom played with the ball .",
    "tom has a red kite . tom went to the yard . the kite was big . tom plays with the kite .",
    "mia finds a book . mia likes the book . anna sees the book . mia helps anna read the book .",
    "max sees the dog . the dog is happy . max has a cookie . max gives the cookie to the dog .",
]
for _ in range(128):
    for story in demo_bias:
        add(sentences, story)

all_sentences = sorted(sentences)
random.shuffle(all_sentences)
val_size = max(250, len(all_sentences) // 10)
val_sentences = all_sentences[:val_size]
train_sentences = all_sentences[val_size:]
train_sentences.extend(demo_bias * 128)
random.shuffle(train_sentences)

used_tokens = []
for group in [
    special_tokens,
    names,
    animals,
    objects,
    places,
    adjectives,
    verbs,
    connectors,
    story_words,
    punctuation,
]:
    for token in group:
        if token not in used_tokens:
            used_tokens.append(token)

for line in train_sentences + val_sentences:
    for token in line.split():
        if token not in used_tokens:
            used_tokens.append(token)

while len(used_tokens) < args.vocab_size:
    used_tokens.append(f"<unused_{len(used_tokens):03d}>")
assert len(used_tokens) <= args.vocab_size, f"vocab has {len(used_tokens)} tokens, target is {args.vocab_size}"
used_tokens = used_tokens[:args.vocab_size]

vocab = {token: idx for idx, token in enumerate(used_tokens)}
(out_dir / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")
(out_dir / "dataset_train.txt").write_text("\n".join(train_sentences) + "\n", encoding="utf-8")
(out_dir / "dataset_val.txt").write_text("\n".join(val_sentences) + "\n", encoding="utf-8")

print("Done.")
print(f"Train sentences: {len(train_sentences)}")
print(f"Val sentences:   {len(val_sentences)}")
print(f"Vocab size:      {len(vocab)}")
print(f"Output dir:      {out_dir}")
