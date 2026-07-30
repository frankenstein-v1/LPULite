# TinyLPU

TinyLPU is an 8-lane language-processing accelerator built around an 8×8
matrix engine, vector post-processing pipeline, banked memories, and a 96-bit
VLIW control path.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/` | Core synthesizable SystemVerilog |
| `tb/` | Core RTL testbench |
| `model/` | Training, datasets, checkpoints, inference, and model exporters |
| `synthesis/` | Quartus/DE1-SoC project, wrappers, drivers, and FPGA tests |
| `asic/` | SKY130/OpenLane experiments and GDS3D visualizations |
| `archive/` | Historical implementations and legacy tests |
| `misc/` | General utilities and retained miscellaneous artifacts |

Each major directory has its own README with focused commands and details.

## Quick checks

```bash
make asic-audit
python -m pytest model/tests/test_tokenizer.py model/tests/test_lm_shift.py
```

Generate or train the reference model:

```bash
python model/scripts/generate_tiny_lm_dataset.py
python model/scripts/train_tiny_lm.py
python model/scripts/generate.py
```

Build the DE1-SoC project:

```bash
python synthesis/scripts/build_de1_soc.py
```

Generate the logic-level GDS3D showcase:

```bash
klayout -b -r asic/make_logic_showcase_gds.py
```
