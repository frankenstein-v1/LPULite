"""
The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.

@karpathy
"""

import argparse
import json
import math     # math.log, math.exp
import random   # random.seed, random.choices, random.gauss, random.shuffle
from datetime import datetime, timezone
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = MODEL_DIR / 'artifacts' / 'microgpt_weights.json'
DEFAULT_INT8_CHECKPOINT = MODEL_DIR / 'artifacts' / 'microgpt_weights_int8.json'
DEFAULT_TARGET_NAMES = 'saksham,satvik,surya,michael,evan,xander,aymaan,yash,kenny'

parser = argparse.ArgumentParser(description='Train the dependency-free MicroGPT name model.')
parser.add_argument('--steps', type=int, default=1000, help='number of Adam training steps (default: 1000)')
parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT, help='output JSON checkpoint path')
parser.add_argument(
    '--int8-checkpoint',
    type=Path,
    default=DEFAULT_INT8_CHECKPOINT,
    help='output LPU block-scaled INT8 checkpoint path',
)
parser.add_argument('--samples', type=int, default=20, help='number of names to sample after training')
parser.add_argument(
    '--target-names',
    default=DEFAULT_TARGET_NAMES,
    help='comma-separated names to emphasize during training',
)
parser.add_argument(
    '--target-sampling-rate',
    type=float,
    default=0.6,
    help='probability of selecting an emphasized name each step (default: 0.6)',
)
args = parser.parse_args()
if args.steps < 1:
    parser.error('--steps must be at least 1')
if args.samples < 0:
    parser.error('--samples cannot be negative')
if not 0.0 <= args.target_sampling_rate <= 1.0:
    parser.error('--target-sampling-rate must be between 0 and 1')

target_names = [name.strip().lower() for name in args.target_names.split(',') if name.strip()]
if args.target_sampling_rate > 0.0 and not target_names:
    parser.error('--target-names must contain at least one name when target sampling is enabled')

seed = 42
random.seed(seed) # Let there be order among chaos

# Let there be a Dataset `docs`: list[str] of documents (e.g. a list of names)
input_path = MODEL_DIR / 'output' / 'microgpt_names.txt'
if not input_path.exists():
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    input_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(names_url, input_path)
docs = [line.strip() for line in input_path.open() if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
uchars = sorted(set(''.join(docs + target_names))) # unique characters in the dataset become token ids 0..n-1
BOS = len(uchars) # token id for a special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1 # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")

# Let there be Autograd to recursively apply the chain rule through a computation graph
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads') # Python optimization for memory usage

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # scalar value of this node calculated during forward pass
        self.grad = 0                   # derivative of the loss w.r.t. this node, calculated in backward pass
        self._children = children       # children of this node in the computation graph
        self._local_grads = local_grads # local derivative of this node w.r.t. its children

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

# Initialize the parameters, to store the knowledge of the model
n_layer = 1     # depth of the transformer neural network (number of layers)
n_embd = 16     # width of the network (embedding dimension)
block_size = 16 # maximum context length of the attention window (note: the longest name is 15 characters)
n_head = 4      # number of attention heads
head_dim = n_embd // n_head # derived dimension of each head
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = [p for mat in state_dict.values() for row in mat for p in row] # flatten params into a single list[Value]
print(f"num params: {len(params)}")

# Define the model architecture: a function mapping tokens and parameters to logits over what comes next
# Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
def linear(x, w):
    quantized_x = fake_quantize_lpu_row(x)
    return [sum(wi * xi for wi, xi in zip(fake_quantize_lpu_row(wo), quantized_x)) for wo in w]

def quantize_lpu_block(values):
    """Quantize up to eight floats as LPU int8 lanes plus a shared 2**scale."""
    absmax = max((abs(value.data if isinstance(value, Value) else value) for value in values), default=0.0)
    scale_exp = 0 if absmax == 0.0 else math.ceil(math.log2(absmax / 127.0))
    scale_exp = max(-128, min(127, scale_exp))
    inverse_scale = math.ldexp(1.0, -scale_exp)
    lanes = []
    for value in values:
        raw = value.data if isinstance(value, Value) else value
        lanes.append(max(-127, min(127, round(raw * inverse_scale))))
    return lanes, scale_exp

def fake_quantize_lpu_row(row):
    """Use LPU values in the forward pass while passing gradients to shadow weights."""
    result = []
    for start in range(0, len(row), 8):
        block = row[start:start + 8]
        lanes, scale_exp = quantize_lpu_block(block)
        scale = math.ldexp(1.0, scale_exp)
        result.extend(value + (lane * scale - value.data) for value, lane in zip(block, lanes))
    return result

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
    tok_emb = fake_quantize_lpu_row(state_dict['wte'][token_id]) # token embedding
    pos_emb = fake_quantize_lpu_row(state_dict['wpe'][pos_id]) # position embedding
    x = [t + p for t, p in zip(tok_emb, pos_emb)] # joint token and position embedding
    x = rmsnorm(x) # note: not redundant due to backward pass via the residual connection

    for li in range(n_layer):
        # 1) Multi-head Attention block
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
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits

# Let there be Adam, the blessed optimizer and its buffers
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params) # first moment buffer
v = [0.0] * len(params) # second moment buffer

# Repeat in sequence
num_steps = args.steps
for step in range(num_steps):

    # Take single document, tokenize it, surround it with BOS special token on both sides
    if target_names and random.random() < args.target_sampling_rate:
        doc = random.choice(target_names)
    else:
        doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # Forward the token sequence through the model, building up the computation graph all the way to the loss
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses) # final average loss over the document sequence. May yours be low.

    # Backward the loss, calculating the gradients with respect to all model parameters
    loss.backward()

    # Adam optimizer update: update the model parameters based on the corresponding gradients
    lr_t = learning_rate * (1 - step / num_steps) # linear learning rate decay
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    if (step + 1) % 50 == 0 or step == 0 or step == num_steps - 1:
        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", flush=True)

# Save plain numeric matrices plus enough metadata to validate or load them in
# another implementation. JSON keeps this tiny reference model dependency-free.
checkpoint = {
    'format': 'lpulite.microgpt.weights',
    'format_version': 1,
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'config': {
        'n_layer': n_layer,
        'n_embd': n_embd,
        'block_size': block_size,
        'n_head': n_head,
        'vocab_size': vocab_size,
    },
    'tokenizer': {
        'characters': uchars,
        'bos_token_id': BOS,
    },
    'training': {
        'seed': seed,
        'steps': num_steps,
        'final_loss': loss.data,
        'learning_rate': learning_rate,
        'beta1': beta1,
        'beta2': beta2,
        'dataset': str(input_path.relative_to(MODEL_DIR.parent)),
        'num_documents': len(docs),
        'target_names': target_names,
        'target_sampling_rate': args.target_sampling_rate,
        'quantization_aware': True,
        'forward_weight_format': 'lpu_block_scaled_int8',
        'forward_activation_format': 'lpu_block_scaled_int8_at_linear_inputs',
    },
    'state_dict': {
        name: [[value.data for value in row] for row in matrix]
        for name, matrix in state_dict.items()
    },
}
checkpoint_path = args.checkpoint.expanduser().resolve()
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + '.tmp')
temporary_path.write_text(json.dumps(checkpoint, indent=2) + '\n', encoding='utf-8')
temporary_path.replace(checkpoint_path)
print(f"saved checkpoint: {checkpoint_path}", flush=True)

def export_lpu_matrix(matrix):
    lane_rows = []
    scale_rows = []
    packed_rows = []
    for row in matrix:
        row_lanes = []
        row_scales = []
        row_words = []
        for start in range(0, len(row), 8):
            lanes, scale_exp = quantize_lpu_block(row[start:start + 8])
            padded_lanes = lanes + [0] * (8 - len(lanes))
            packed = sum((lane & 0xff) << (8 * idx) for idx, lane in enumerate(padded_lanes))
            packed |= (scale_exp & 0xff) << 64
            row_lanes.extend(lanes)
            row_scales.append(scale_exp)
            row_words.append(f'0x{packed:018x}')
        lane_rows.append(row_lanes)
        scale_rows.append(row_scales)
        packed_rows.append(row_words)
    return {
        'shape': [len(matrix), len(matrix[0]) if matrix else 0],
        'lanes': lane_rows,
        'scale_exponents': scale_rows,
        'packed_72bit_rows': packed_rows,
    }

lpu_checkpoint = {
    'format': 'lpulite.microgpt.lpu_int8',
    'format_version': 1,
    'created_utc': checkpoint['created_utc'],
    'numeric_contract': {
        'lane_format': 'signed_int8',
        'lane_range': [-127, 127],
        'lanes_per_block': 8,
        'scale_format': 'signed_int8_power_of_two_exponent',
        'dequantization': 'real_value = lane * 2**scale_exponent',
        'packed_row_bits': 72,
        'packed_layout': 'lane0 bits [7:0] through lane7 bits [63:56], scale bits [71:64]',
        'accumulator_format': 'signed_int32',
    },
    'config': checkpoint['config'],
    'tokenizer': checkpoint['tokenizer'],
    'training': checkpoint['training'],
    'state_dict': {name: export_lpu_matrix(matrix) for name, matrix in state_dict.items()},
}
int8_checkpoint_path = args.int8_checkpoint.expanduser().resolve()
int8_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
int8_temporary_path = int8_checkpoint_path.with_suffix(int8_checkpoint_path.suffix + '.tmp')
int8_temporary_path.write_text(json.dumps(lpu_checkpoint, indent=2) + '\n', encoding='utf-8')
int8_temporary_path.replace(int8_checkpoint_path)
print(f"saved LPU INT8 checkpoint: {int8_checkpoint_path}", flush=True)

# Inference: may the model babble back to us
temperature = 0.5 # in (0, 1], control the "creativity" of generated text, low to high
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(args.samples):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")

if target_names:
    print("\n--- emphasized-name prefix evaluation ---")
    char_to_id = {char: idx for idx, char in enumerate(uchars)}
    for target_name in target_names:
        prefix = target_name[:min(3, max(1, len(target_name) - 1))]
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        prefix_ids = [BOS] + [char_to_id[char] for char in prefix]
        for pos_id, token_id in enumerate(prefix_ids):
            logits = gpt(token_id, pos_id, keys, values)
        generated = list(prefix)
        for pos_id in range(len(prefix_ids), block_size):
            token_id = max(range(vocab_size), key=lambda idx: logits[idx].data)
            if token_id == BOS:
                break
            generated.append(uchars[token_id])
            logits = gpt(token_id, pos_id, keys, values)
        prediction = ''.join(generated)
        status = 'exact' if prediction == target_name else 'learned target, non-exact completion'
        print(f"{prefix!r} -> {prediction!r} ({status}; target={target_name!r})")
