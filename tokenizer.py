import json

class Tokenizer:
    def __init__(self, vocab_path="output/vocab.json"):
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)  # token string -> {"id": id, "score": score}
        self.id_to_token = {v["id"]: k for k, v in self.vocab.items()}
        
        # Populate byte-to-id mapping for initialization
        self.byte_to_id = {}
        for b in range(256):
            token_str = f"<0x{b:02X}>"
            if token_str in self.vocab:
                self.byte_to_id[b] = self.vocab[token_str]["id"]
            elif chr(b) in self.vocab:
                self.byte_to_id[b] = self.vocab[chr(b)]["id"]

    def encode(self, text, bos=True, eos=False):
        if not text:
            res = []
            if bos:
                bos_id = self.vocab.get("\n<s>\n", {}).get("id", 1)
                res.append(bos_id)
            return res
            
        # SentencePiece BPE prepends a dummy prefix space if the first char is not space
        if len(text) > 0 and not text.startswith(" "):
            text = " " + text
            
        # 1. Convert text to bytes
        byte_data = text.encode("utf-8")
        
        # 2. Map each byte to token ID
        ids = []
        unk_id = self.vocab.get("<unk>", {"id": 0})["id"]
        for b in byte_data:
            ids.append(self.byte_to_id.get(b, unk_id))
            
        # 3. Greedy BPE merge loop
        while True:
            best_pair_idx = -1
            best_pair_id = -1
            best_score = -9999999.0
            
            # Find the best mergeable adjacent pair in ids
            for i in range(len(ids) - 1):
                t1 = self.id_to_token.get(ids[i], "")
                t2 = self.id_to_token.get(ids[i+1], "")
                
                # Check if concatenated string is in vocab
                concat_str = t1 + t2
                if concat_str in self.vocab:
                    score = self.vocab[concat_str]["score"]
                    if score > best_score:
                        best_score = score
                        best_pair_idx = i
                        best_pair_id = self.vocab[concat_str]["id"]
                        
            # If no merge found, stop
            if best_pair_idx == -1:
                break
                
            # Perform the merge at best_pair_idx
            ids[best_pair_idx] = best_pair_id
            del ids[best_pair_idx + 1]
            
        # Add special tokens
        res = []
        if bos:
            bos_id = self.vocab.get("\n<s>\n", {}).get("id", 1)
            res.append(bos_id)
        res.extend(ids)
        if eos:
            eos_id = self.vocab.get("\n</s>\n", {}).get("id", 2)
            res.append(eos_id)
        return res

    def decode(self, ids):
        tokens = []
        for token_id in ids:
            t = self.id_to_token.get(token_id, "")
            # Skip special tokens
            if t in ["\n<s>\n", "\n</s>\n", "<s>", "</s>", "<unk>"]:
                continue
            if t.startswith("<0x") and t.endswith(">"):
                # Byte fallback token
                try:
                    b_val = int(t[3:-1], 16)
                    tokens.append(bytes([b_val]))
                except ValueError:
                    pass
            else:
                tokens.append(t.encode("utf-8"))
        
        # Join bytes and decode
        raw_decoded = b"".join(tokens).decode("utf-8", errors="replace")
        # Remove dummy prefix space if it exists
        if raw_decoded.startswith(" "):
            raw_decoded = raw_decoded[1:]
        return raw_decoded
