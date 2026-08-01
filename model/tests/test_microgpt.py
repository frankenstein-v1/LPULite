"""
Fine-tuning microGPT to accurately generate exact target names:
satvik, surya, saksham, yash, evan, xander, aymaan, michael, arjun
"""

import math
import random
from pathlib import Path

random.seed(42)

# Load dataset
input_path = Path(__file__).resolve().parents[1] / 'output' / 'microgpt_names.txt'
if not input_path.exists():
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    input_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(names_url, input_path)

general_docs = [line.strip().lower() for line in input_path.open() if line.strip()]
random.shuffle(general_docs)

# Target names requested by user
target_names = ["satvik", "surya", "saksham", "yash", "evan", "xander", "aymaan", "michael", "arjun"]

name_prefixes = {
    "satvik": "sat",
    "surya": "sur",
    "saksham": "sak",
    "yash": "ya",
    "evan": "ev",
    "xander": "xan",
    "aymaan": "ay",
    "michael": "mic",
    "arjun": "arj"
}

# Tokenizer built over all text
all_text = ''.join(general_docs) + ''.join(target_names)
uchars = sorted(set(all_text))
BOS = len(uchars)
vocab_size = len(uchars) + 1
char_to_id = {ch: i for i, ch in enumerate(uchars)}

class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# Model Architecture
n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

state_dict = {
    'wte': matrix(vocab_size, n_embd),
    'wpe': matrix(block_size, n_embd),
    'lm_head': matrix(vocab_size, n_embd)
}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

params = [p for mat in state_dict.values() for row in mat for p in row]

def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(n_layer):
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]

        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits

# Fast training (400 steps with 70% target name sampling focus)
num_steps = 400
print(f"Training microGPT for {num_steps} steps...", flush=True)
learning_rate, beta1, beta2, eps_adam = 0.02, 0.85, 0.99, 1e-8
m = [0.0] * len(params)
v = [0.0] * len(params)

for step in range(num_steps):
    if random.random() < 0.7:
        doc = random.choice(target_names)
    else:
        doc = general_docs[step % len(general_docs)]

    tokens = [BOS] + [char_to_id[ch] for ch in doc if ch in char_to_id] + [BOS]
    n = min(block_size, len(tokens) - 1)

    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)

    loss.backward()

    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    if (step + 1) % 100 == 0 or step == 0 or step == num_steps - 1:
        print(f"  Step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", flush=True)

print("\n=======================================================", flush=True)
print("EVALUATION: Exact Name Prediction Results", flush=True)
print("=======================================================", flush=True)

exact_matches = 0
for target_name in target_names:
    prefix = name_prefixes[target_name]
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    prefix_ids = [BOS] + [char_to_id[ch] for ch in prefix if ch in char_to_id]

    for pos_id, tid in enumerate(prefix_ids):
        logits = gpt(tid, pos_id, keys, values)

    gen_chars = list(prefix)
    curr_pos = len(prefix_ids) - 1

    for pos_id in range(curr_pos + 1, block_size):
        # Greedy decoding (argmax) for exact prediction
        next_tid = max(range(vocab_size), key=lambda idx: logits[idx].data)
        if next_tid == BOS:
            break
        gen_chars.append(uchars[next_tid])
        logits = gpt(next_tid, pos_id, keys, values)

    generated_name = "".join(gen_chars)
    is_exact = (generated_name == target_name)
    if is_exact:
        exact_matches += 1
    match_status = "[EXACT MATCH]" if is_exact else f"[CLOSE MATCH: '{generated_name}']"
    print(f"Target: '{target_name:8s}' | Prompt Prefix: '{prefix:3s}' -> Predicted: '{generated_name:10s}' | {match_status}", flush=True)

print(f"\nOverall Score: {exact_matches}/{len(target_names)} target names predicted perfectly!", flush=True)
