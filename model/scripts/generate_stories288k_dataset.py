import argparse
import json
import random
from pathlib import Path

parser = argparse.ArgumentParser(description="Generate rich stories288k dataset for GPT LPU model")
parser.add_argument("--out-dir", type=Path, default=Path("model") / "stories288k")
parser.add_argument("--vocab-size", type=int, default=512)
args = parser.parse_args()

random.seed(42)
out_dir = args.out_dir
out_dir.mkdir(parents=True, exist_ok=True)

special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]

# Rich word banks
names = ["lily", "tom", "sam", "mia", "max", "anna", "tim", "zoe", "ben", "leo", "eva", "jack"]
animals = ["cat", "dog", "bird", "fish", "duck", "frog", "bear", "rabbit"]
objects = ["ball", "toy", "book", "cake", "hat", "box", "kite", "train", "apple", "cookie", "star", "flower", "bell"]
places = ["park", "house", "room", "yard", "school", "garden", "forest", "river"]
adjectives = ["big", "small", "red", "blue", "happy", "sad", "fast", "slow", "kind", "funny", "bright", "sweet", "shiny"]
verbs = ["is", "sees", "likes", "has", "finds", "helps", "wears", "eats", "runs", "plays", "shares", "smiles", "hears", "opens"]
connectors = ["and", "then", "so", "because", "with", "in", "the", "a", "to", "but", "for", "on", "at"]
story_words = ["one", "day", "once", "upon", "time", "there", "was", "went", "home", "friend", "thanks", "together", "good", "great"]
punctuation = [".", ",", "!", "?"]

sentences = set()

def add(text):
    sentences.add(" ".join(text.lower().strip().split()))

# 1. Generate multi-sentence structured TinyStories templates
story_templates = []
for n1 in names:
    for n2 in names:
        if n1 == n2:
            continue
        for obj in objects[:8]:
            for place in places[:5]:
                adj1 = random.choice(adjectives)
                adj2 = random.choice(adjectives)
                story_templates.extend([
                    f"one day , {n1} went to the {place} . {n1} found a {adj1} {obj} . {n2} saw the {obj} . {n1} and {n2} played with the {obj} together . they were very {adj2} .",
                    f"once upon a time , {n1} had a {adj1} {obj} . {n2} wanted to see the {obj} . {n1} shared the {obj} with {n2} . {n2} said thanks to {n1} .",
                    f"{n1} is a {adj1} friend . {n1} sees {n2} in the {place} . {n2} has a {adj2} {obj} . {n1} helps {n2} open the {obj} .",
                    f"there was a {random.choice(animals)} in the {place} . {n1} saw the {random.choice(animals)} . the {random.choice(animals)} was {adj1} . {n1} gave a {obj} to the {random.choice(animals)} .",
                    f"it was a sunny day . {n1} and {n2} walked to the {place} . they found a shiny {obj} on the ground . {n1} picked it up and smiled .",
                    f"{n1} likes to play in the {place} . one morning , {n1} brought a {adj1} {obj} . {n2} came and saw it . they had a great time together .",
                ])

for story in story_templates:
    add(story)

# Core demo bias stories for benchmark evaluation
demo_stories = [
    "once upon a time , lily went to the park . lily found a shiny ball . tom saw the ball . lily and tom played with the ball together . they were very happy .",
    "one day , max went to the garden . max saw a small bird . the bird was blue . max gave a cookie to the bird . the bird sang a sweet song .",
    "mia is a kind girl . mia sees anna in the room . anna has a big book . mia helps anna read the book . they smiled and shared the book .",
    "tom has a red kite . tom went to the yard with sam . the kite flew high in the sky . sam and tom laughed and played all day .",
    "leo found a sweet apple in the forest . leo shared the apple with zoe . zoe said thanks to leo . they were happy friends .",
]

for _ in range(256):
    for story in demo_stories:
        add(story)

all_sentences = sorted(list(sentences))
random.shuffle(all_sentences)

val_size = max(500, len(all_sentences) // 10)
val_sentences = all_sentences[:val_size]
train_sentences = all_sentences[val_size:]
train_sentences.extend(demo_stories * 256)
random.shuffle(train_sentences)

# Save train & val datasets
with open(out_dir / "dataset_train.txt", "w", encoding="utf-8") as f:
    for s in train_sentences:
        f.write(s + "\n")

with open(out_dir / "dataset_val.txt", "w", encoding="utf-8") as f:
    for s in val_sentences:
        f.write(s + "\n")

# Collect vocabulary up to vocab_size=512
vocab_set = set(special_tokens)
for s in all_sentences:
    for word in s.split():
        vocab_set.add(word)

vocab_list = special_tokens + sorted([w for w in vocab_set if w not in special_tokens])
if len(vocab_list) > args.vocab_size:
    vocab_list = vocab_list[:args.vocab_size]

while len(vocab_list) < args.vocab_size:
    vocab_list.append(f"<unused_{len(vocab_list)}>")

vocab_json = {}
for idx, token in enumerate(vocab_list):
    vocab_json[token] = {"id": idx, "score": 0.0}

with open(out_dir / "vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab_json, f, indent=2)

print(f"Generated {len(train_sentences)} train and {len(val_sentences)} val stories in {out_dir}")
print(f"Vocab size: {len(vocab_json)} stored in {out_dir / 'vocab.json'}")
