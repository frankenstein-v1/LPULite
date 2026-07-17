import sys
import torch
import json
from pathlib import Path
from train_tiny_lm import TinyLM, UNK_ID, PAD_ID, SEQ_LEN, VOCAB_SIZE

# Load mapping
VOCAB_PATH = Path("output/vocab.json")
MODEL_PATH = Path("tiny_lm_model.pt")

if not MODEL_PATH.exists():
    print(f"Pretrained model path {MODEL_PATH} not found. Please run: python train_tiny_lm.py to train it first!")
    sys.exit(1)

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)

id_to_token = {idx: tok for tok, idx in vocab.items()}

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load(MODEL_PATH, map_location=device)

# Load model architecture
model = TinyLM().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

def generate(prompt, max_new_tokens=5):
    tokens = prompt.lower().strip().split()
    ids = [vocab.get(tok, UNK_ID) for tok in tokens]

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
        
        next_token = id_to_token[next_id]
        if next_token == "." or next_token == "<pad>":
            break
            
    # Decode to text
    decoded_tokens = [id_to_token[idx] for idx in ids if id_to_token[idx] != "<pad>"]
    return " ".join(decoded_tokens)

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
