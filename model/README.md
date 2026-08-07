# Model and inference

This directory contains all model-facing code and data.

## Layout

- `microgpt.py` — dependency-free MicroGPT reference implementation
- `scripts/` — dataset generation, training, and text generation
- `tools/` — VLIW compilation, weight packing, and hardware export
- `tests/` — tokenizer, inference, and RTL-backed model tests
- `artifacts/` — compact reference checkpoints/exports
- `datasets/` — model datasets (`tiny_lm/`, `stories10k/`, `stories288k/`, `qa_facts/`)

## Basic workflow

```bash
python model/scripts/generate_tiny_lm_dataset.py
python model/scripts/train_tiny_lm.py
python model/scripts/generate.py
```

Train the dependency-free MicroGPT reference and save its portable JSON weights:

```bash
python model/microgpt.py
```

Training uses LPU-aware fake quantization on every forward pass. It saves the
float shadow checkpoint to `model/artifacts/microgpt_weights.json` and the
deployable block-scaled INT8 checkpoint to
`model/artifacts/microgpt_weights_int8.json`. Use `--steps`, `--checkpoint`, or
`--int8-checkpoint` to override the training length or output paths.

Run CPU inference directly from the LPU INT8 checkpoint:

```bash
python model/scripts/run_microgpt_int8.py ken
```

Run the MicroGPT learned linear layers through the real LPU RTL with cocotb:

```bash
python model/tests/run_microgpt_lpu_cocotb.py
```

This RTL test uses the LPU's real 8×8 MXM datapath for every learned
matrix-vector operation; cocotb performs operand sequencing, RMSNorm, softmax,
and token selection.
The default training mix emphasizes Saksham, Satvik, Surya, Michael, Evan,
Xander, Aymaan, Yash, and Kenny; customize it with `--target-names` and
`--target-sampling-rate`.

Model tools resolve paths from the repository root, so commands may be run
from any working directory.
