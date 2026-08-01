# Model and inference

This directory contains all model-facing code and data.

## Layout

- `microgpt.py` — dependency-free MicroGPT reference implementation
- `scripts/` — dataset generation, training, and text generation
- `tools/` — VLIW compilation, weight packing, and hardware export
- `tests/` — tokenizer, inference, and RTL-backed model tests
- `artifacts/` — compact reference checkpoints/exports
- `output/` — default tiny-model dataset and vocabulary
- `stories10k/`, `stories288k/`, `qa_facts/` — named model datasets

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
The default training mix emphasizes Saksham, Satvik, Surya, Michael, Evan,
Xander, Aymaan, Yash, and Kenny; customize it with `--target-names` and
`--target-sampling-rate`.

Model tools resolve paths from the repository root, so commands may be run
from any working directory.
