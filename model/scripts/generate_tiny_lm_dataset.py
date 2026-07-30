import json
import random
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Tiny LPU LM Dataset Generator")
parser.add_argument("--vocab-size", type=int, default=256, help="Target vocabulary size (default: 256)")
parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "output", help="Output directory for dataset and vocab")
args = parser.parse_args()

TARGET_VOCAB_SIZE = args.vocab_size

# ============================================================
# Tiny LPU LM Dataset Generator
#
# Creates:
#   output/dataset_train.txt
#   output/dataset_val.txt
#   model/output/vocab.json
#
# Goal:
#   Generate a small text dataset for a tiny causal LM.
#   The model should learn simple sentence patterns instead
#   of only memorizing one fixed phrase.
# ============================================================

SEED = 42
random.seed(SEED)

OUT_DIR = args.out_dir
OUT_DIR.mkdir(exist_ok=True)

# ============================================================
# 1. Define token groups
# ============================================================

special_tokens = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
]

names = [
    "lebron",
    "messi",
    "ronaldo",
    "ranvijay",
    "lily",
    "tim",
    "anna",
    "max",
    "bob",
    "mia",
    "sam",
    "zoe",
    "kai",
    "nina",
    "omar",
    "ivy",
    "leo",
]

verbs = [
    "is",
    "likes",
    "sees",
    "wears",
    "eats",
    "has",
    "finds",
    "helps",
]

articles = [
    "the",
    "a",
    "an",
]

adjectives = [
    "big",
    "small",
    "red",
    "blue",
    "green",
    "happy",
    "sad",
    "fast",
    "slow",
    "yellow",
    "tiny",
    "round",
    "soft",
    "loud",
    "quiet",
    "bright",
]

roles = [
    "king",
    "goat",
    "hero",
    "legend",
    "rookie",
    "friend",
    "winner",
    "runner",
    "player",
    "leader",
    "champion",
    "student",
    "teacher",
    "pilot",
    "artist",
    "merchant",
]

animals = [
    "cat",
    "dog",
    "bird",
    "fish",
    "cow",
    "duck",
    "frog",
    "lion",
    "bear",
    "horse",
    "goat",
    "mouse",
]

objects = [
    "hat",
    "ball",
    "food",
    "park",
    "toy",
    "house",
    "book",
    "car",
    "bag",
    "cup",
    "box",
    "bed",
    "chair",
    "shoe",
    "shirt",
    "game",
    "phone",
    "bike",
    "cake",
    "map",
    "pen",
    "desk",
    "door",
    "tree",
    "apple",
    "pizza",
    "cookie",
    "robot",
    "kite",
    "boat",
    "train",
    "plane",
    "coin",
    "key",
    "flower",
    "stone",
]
wear_items = [
    "hat",
    "shirt",
    "shoe",
    "bag",
]

things = [
    "sky",
    "grass",
    "sun",
    "moon",
    "water",
    "fire",
    "tree",
    "car",
    "house",
    "ball",
    "hat",
    "food",
    "park",
    "toy",
    "book",
    "robot",
    "apple",
    "pizza",
    "cloud",
    "river",
    "road",
    "door",
    "chair",
    "box",
]

extra_words = [
    "washed",
]

punctuation = [
    ".",
]

# ============================================================
# 2. Helper function to add sentences
# ============================================================

def add(sentence_set, sentence):
    """
    Lowercase and clean each sentence before adding it.
    This keeps tokenization simple and consistent.
    """
    sentence = sentence.lower().strip()
    sentence_set.add(sentence)


all_sentences = set()

# ============================================================
# 3. Generate sentence combinations from templates
# ============================================================

# Template:
#   {name} is {role_or_adjective} .
#
# Every generated sentence is capped at 4 tokens so SEQ_LEN=4 training never
# depends on truncation.
name_states = roles + adjectives + [
    "alpha",
    "washed",
]

for name in names:
    for state in name_states:
        if state == "washed" and name != "ronaldo":
            continue
        add(all_sentences, f"{name} is {state} .")

# Related examples to help the model generalize.
special_related = [
    # Ranvijay-related examples
    "lebron is alpha .",
    "messi is alpha .",
    "tim is alpha .",
    "anna is alpha .",
    "max is alpha .",
    "bob is alpha .",

    # Color/object related examples
    "cloud is blue .",
    "water is blue .",
    "river is blue .",
    "grass is green .",
    "tree is green .",
]

for sentence in special_related:
    add(all_sentences, sentence)

# Template:
#   {subject} likes {object} .
like_subjects = names + animals + things

for subject in like_subjects:
    for obj in objects:
        add(all_sentences, f"{subject} likes {obj} .")

# Template:
#   {subject} sees {animal} .
see_subjects = names + animals + things

for subject in see_subjects:
    for animal in animals:
        add(all_sentences, f"{subject} sees {animal} .")

# Template:
#   {animal} wears {item} .
for animal in animals:
    for item in wear_items:
        add(all_sentences, f"{animal} wears {item} .")

# Template:
#   {animal} eats {object} .
for animal in animals:
    for obj in objects:
        add(all_sentences, f"{animal} eats {obj} .")

# Template:
#   {thing} is {adjective} .
for thing in things:
    for adjective in adjectives:
            add(all_sentences, f"{thing} is {adjective} .")

# Template:
#   {name} has {object} .
for name in names:
    for obj in objects:
        add(all_sentences, f"{name} has {obj} .")

# Template:
#   {name} helps {name2} .
for name in names:
    for name2 in names:
        if name != name2:
            add(all_sentences, f"{name} helps {name2} .")

# Template:
#   {name} finds {object} .
for name in names:
    for obj in objects:
        add(all_sentences, f"{name} finds {obj} .")

# Exact outputs used by the generation demo. Since all_sentences is a set,
# adding these guarantees their presence but does not by itself weight them.
demo_bias_sentences = [
    "lebron is king .",
    "messi is goat .",
    "ronaldo is washed .",
    "ranvijay is alpha .",
    "cat wears hat .",
    "sky is blue .",
    "dog sees cat .",
    "lily likes ball .",
]

for s in demo_bias_sentences:
    add(all_sentences, s)

all_sentences = sorted(all_sentences)

# ============================================================
# 4. Force specific validation holdouts
# ============================================================

forced_val = {
    "messi is king .",
    "grass is green .",
    "tim helps anna .",
}

# Make sure every forced validation sentence actually exists
for sentence in forced_val:
    assert sentence in all_sentences, f"Missing forced validation sentence: {sentence}"

# Keep demo targets out of validation so they can be deliberately weighted in
# training. Repetition matters here because the model sees one example per line.
demo_bias_set = set(demo_bias_sentences)
remaining = [
    s for s in all_sentences
    if s not in forced_val and s not in demo_bias_set
]

# Shuffle deterministically
random.shuffle(remaining)

# Validation should be at least 500 sentences
val_size = max(500, int(0.10 * len(all_sentences)))

extra_val_needed = val_size - len(forced_val)

val_sentences = list(forced_val) + remaining[:extra_val_needed]
train_sentences = remaining[extra_val_needed:]

# The model is intentionally tiny (8-dimensional), so give the exact demo
# mappings enough weight to beat the many uniformly generated alternatives.
DEMO_BIAS_REPEATS = 128
train_sentences.extend(demo_bias_sentences * DEMO_BIAS_REPEATS)

# Shuffle final files
random.shuffle(train_sentences)
random.shuffle(val_sentences)

# ============================================================
# 5. Safety checks
# ============================================================

assert len(train_sentences) >= 5000, f"Only {len(train_sentences)} training sentences generated"
assert len(val_sentences) >= 500, f"Only {len(val_sentences)} validation sentences generated"

# Make sure train and validation do not overlap
overlap = set(train_sentences) & set(val_sentences)
assert len(overlap) == 0, f"Train/val overlap found: {overlap}"

# Make sure every sentence is max 4 tokens
for sentence in train_sentences + val_sentences:
    tokens = sentence.split()
    assert len(tokens) <= 4, f"Sentence too long: {sentence} has {len(tokens)} tokens"

# Make sure demo targets are training-only and receive the intended weight.
for sentence in demo_bias_sentences:
    assert train_sentences.count(sentence) == DEMO_BIAS_REPEATS
    assert sentence not in val_sentences

# "washed" must describe only Ronaldo.
washed_sentences = [s for s in all_sentences if "washed" in s.split()]
assert washed_sentences == ["ronaldo is washed ."]

# ============================================================
# 6. Build vocab.json with exactly 256 entries
# ============================================================

used_tokens = set()

for sentence in train_sentences + val_sentences:
    for token in sentence.split():
        used_tokens.add(token)

vocab_tokens = []

# Add special tokens first so their IDs are stable
for token in special_tokens:
    if token not in vocab_tokens:
        vocab_tokens.append(token)

# Add normal vocabulary groups
vocab_groups = [
    names,
    verbs,
    articles,
    adjectives,
    roles,
    animals,
    objects,
    things,
    extra_words,
    punctuation,
]

for group in vocab_groups:
    for token in group:
        if token not in vocab_tokens:
            vocab_tokens.append(token)

# Add anything that appeared in sentences but was missed
for token in sorted(used_tokens):
    if token not in vocab_tokens:
        vocab_tokens.append(token)

# Fill unused slots until vocab has exactly TARGET_VOCAB_SIZE entries
unused_index = 0

while len(vocab_tokens) < TARGET_VOCAB_SIZE:
    filler_token = f"<unused_{unused_index:03d}>"
    vocab_tokens.append(filler_token)
    unused_index += 1

# If this fails, we accidentally added too many real words
assert len(vocab_tokens) == TARGET_VOCAB_SIZE, f"Vocab size is {len(vocab_tokens)}, expected {TARGET_VOCAB_SIZE}"

vocab = {token: idx for idx, token in enumerate(vocab_tokens)}

# ============================================================
# 7. Write output files
# ============================================================

train_path = OUT_DIR / "dataset_train.txt"
val_path = OUT_DIR / "dataset_val.txt"
vocab_path = OUT_DIR / "vocab.json"

with open(train_path, "w", encoding="utf-8") as f:
    for sentence in train_sentences:
        f.write(sentence + "\n")

with open(val_path, "w", encoding="utf-8") as f:
    for sentence in val_sentences:
        f.write(sentence + "\n")

with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump(vocab, f, indent=2)

# ============================================================
# 8. Print summary
# ============================================================

print("Done.")
print(f"Total sentences:       {len(all_sentences)}")
print(f"Training sentences:    {len(train_sentences)}")
print(f"Validation sentences:  {len(val_sentences)}")
print(f"Vocab size:            {len(vocab)}")
print()
print("Files created:")
print(f"  {train_path}")
print(f"  {val_path}")
print(f"  {vocab_path}")
print()
print("Biased training demos included:")
print("  lebron is king .")
print("  messi is goat .")
print("  ronaldo is washed .")
print("  ranvijay is alpha .")
print("  cat wears hat .")
print("  sky is blue .")
print("  dog sees cat .")
print("  lily likes ball .")
