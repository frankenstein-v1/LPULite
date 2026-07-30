# Model and inference

This directory contains all model-facing code and data.

## Layout

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

Model tools resolve paths from the repository root, so commands may be run
from any working directory.
