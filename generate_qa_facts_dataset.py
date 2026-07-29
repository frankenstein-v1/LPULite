import argparse
import json
import random
from pathlib import Path

parser = argparse.ArgumentParser(description="Generate QA & Fact dataset for 476k GPT LPU model")
parser.add_argument("--out-dir", type=Path, default=Path("model") / "qa_facts")
parser.add_argument("--vocab-size", type=int, default=512)
args = parser.parse_args()

random.seed(42)
out_dir = args.out_dir
out_dir.mkdir(parents=True, exist_ok=True)

special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]

# Comprehensive QA & Fact Dataset Pairs
qa_pairs = [
    # Skyscrapers & Buildings
    ("what is the tallest building in the world", "burj khalifa in dubai"),
    ("where is the burj khalifa located", "dubai in united arab emirates"),
    ("what is the tallest tower in japan", "tokyo skytree"),
    ("what is the famous clock tower in london", "big ben"),
    ("where is the eiffel tower located", "paris in france"),
    ("where is the statue of liberty", "new york city"),
    ("where is the taj mahal located", "india"),
    ("where are the Great Pyramids located", "egypt"),

    # World Capitals & Geography
    ("what is the capital of france", "paris"),
    ("what is the capital of japan", "tokyo"),
    ("what is the capital of england", "london"),
    ("what is the capital of united states", "washington dc"),
    ("what is the capital of canada", "ottawa"),
    ("what is the capital of germany", "berlin"),
    ("what is the capital of italy", "rome"),
    ("what is the capital of spain", "madrid"),
    ("what is the capital of china", "beijing"),
    ("what is the capital of india", "new delhi"),
    ("what is the capital of australia", "canberra"),
    ("what is the capital of brazil", "brasilia"),
    ("what is the largest ocean in the world", "pacific ocean"),
    ("what is the smallest continent", "australia"),
    ("what is the largest continent", "asia"),
    ("what is the longest river in the world", "nile river"),
    ("what is the largest desert in the world", "antarctica"),
    ("what is the highest mountain in the world", "mount everest"),

    # Science & Space
    ("what planet is known as the red planet", "mars"),
    ("what is the largest planet in our solar system", "jupiter"),
    ("what is the closest planet to the sun", "mercury"),
    ("what star is at the center of our solar system", "the sun"),
    ("what is the satellite of earth", "the moon"),
    ("what gas do humans breathe to live", "oxygen"),
    ("what gas do plants absorb from air", "carbon dioxide"),
    ("what chemical element has symbol H", "hydrogen"),
    ("what chemical element has symbol O", "oxygen"),
    ("what chemical element has symbol Au", "gold"),
    ("what chemical element has symbol Ag", "silver"),
    ("what state of matter is water ice", "solid"),
    ("what state of matter is water vapor", "gas"),

    # Animals
    ("what is the fastest land animal", "cheetah"),
    ("what is the largest animal in the world", "blue whale"),
    ("what is the tallest animal in the world", "giraffe"),
    ("what animal says meow", "a cat"),
    ("what animal says woof", "a dog"),
    ("what animal says oink", "a pig"),
    ("what animal says moo", "a cow"),
    ("what animal says quack", "a duck"),
    ("what bird cannot fly and lives in cold ice", "penguin"),

    # Colors & Nature
    ("what color is the sky on a clear day", "blue"),
    ("what color are grass and leaves", "green"),
    ("what color is fresh snow", "white"),
    ("what color is a ripe banana", "yellow"),
    ("what color is a strawberry", "red"),
    ("what color do you get when you mix red and yellow", "orange"),
    ("what color do you get when you mix blue and yellow", "green"),

    # Simple Math & Numbers
    ("what is 1 plus 1", "2"),
    ("what is 2 plus 2", "4"),
    ("what is 3 plus 3", "6"),
    ("what is 4 plus 4", "8"),
    ("what is 5 plus 5", "10"),
    ("what is 10 plus 10", "20"),
    ("what is 5 times 5", "25"),
    ("what is 10 times 10", "100"),
    ("how many hours are in one day", "24 hours"),
    ("how many days are in one week", "7 days"),
    ("how many days are in one year", "365 days"),
    ("how many minutes are in one hour", "60 minutes"),
    ("how many seconds are in one minute", "60 seconds"),
    ("how many sides does a triangle have", "3 sides"),
    ("how many sides does a square have", "4 sides"),
    ("how many sides does a hexagon have", "6 sides"),

    # Literature & History
    ("who wrote romeo and juliet", "william shakespeare"),
    ("who painted the mona lisa", "leonardo da vinci"),
    ("who was the first president of the united states", "george washington"),
    ("who developed the theory of relativity", "albert einstein"),
    ("who invented the light bulb", "thomas edison"),

    # Tech & Computing
    ("what does CPU stand for in computers", "central processing unit"),
    ("what does LPU stand for in hardware", "language processing unit"),
    ("what FPGA chip is on the DE1-SoC board", "intel cyclone v"),
    ("what is the main operating system for PC", "windows"),
]

# Formatted sentence generator
formatted_sentences = set()

def add(q, a):
    # Lowercase & clean string
    q_str = " ".join(q.lower().strip().split())
    a_str = " ".join(a.lower().strip().split())
    formatted_sentences.add(f"q: {q_str} ? a: {a_str} .")
    formatted_sentences.add(f"question: {q_str} ? answer: {a_str} .")

for q, a in qa_pairs:
    add(q, a)

# Multiply examples for training stability
all_sentences = sorted(list(formatted_sentences))
train_sentences = []
for _ in range(300):
    train_sentences.extend(all_sentences)

random.shuffle(train_sentences)

val_sentences = all_sentences.copy()
random.shuffle(val_sentences)

# Save datasets
with open(out_dir / "dataset_train.txt", "w", encoding="utf-8") as f:
    for s in train_sentences:
        f.write(s + "\n")

with open(out_dir / "dataset_val.txt", "w", encoding="utf-8") as f:
    for s in val_sentences:
        f.write(s + "\n")

# Build vocab
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

print(f"Generated {len(train_sentences)} train QA examples in {out_dir}")
print(f"Vocab size: {len(vocab_json)} stored in {out_dir / 'vocab.json'}")
