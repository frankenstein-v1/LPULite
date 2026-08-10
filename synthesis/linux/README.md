# TinyLPU MicroGPT on DE1-SoC HPS Linux

This directory contains the ARM/Linux userspace runtime for driving the TinyLPU
through the DE1-SoC HPS-to-FPGA lightweight bridge instead of USB-JTAG/System
Console.

The runtime uses the same generated model artifacts as the JTAG testbench:

- `model/artifacts/fpga_microgpt/microgpt_scheduler_mem1.hex`
- `model/artifacts/fpga_microgpt/microgpt_decode_vliw.hex`
- `model/artifacts/fpga_microgpt/microgpt_decode_schedule.json`
- `model/artifacts/fpga_microgpt/microgpt_decode_trace.json`

## Files

- `src/tinylpu_hps_mmio.c` / `include/tinylpu_hps_mmio.h`:
  `/dev/mem` MMIO driver for the lightweight bridge.
- `src/microgpt_hps_runtime.c`:
  interactive char-level MicroGPT terminal.
- `include/microgpt_hps_image.h`:
  generated C image containing the VLIW schedule and MEM1 model rows.
- `Makefile`:
  regenerates the image header and builds the runtime.

## Build on the DE1-SoC ARM Linux shell

Copy the repo, or at least `synthesis/linux`, `synthesis/scripts`, and
`model/artifacts/fpga_microgpt`, onto the ARM Linux filesystem.

```sh
cd tinyLPU/synthesis/linux
make
```

Then run:

```sh
sudo ./microgpt_hps_runtime --attention host
```

Useful options:

```sh
sudo ./microgpt_hps_runtime --attention current
sudo ./microgpt_hps_runtime --no-load-weights
sudo ./microgpt_hps_runtime --base 0xff200000 --span 0x10000
```

`--attention host` means the ARM-side testbench computes the small causal
attention context from FPGA-produced Q/K/V rows cached in FPGA MEM0. The LPU
still runs the VLIW pages for the model stages. `--attention current` performs
no ARM attention math and stages the current FPGA-produced V row as the
attention context, which is useful for hardware bring-up but is not exact
multi-token attention.

## Required FPGA address map

The Linux runtime expects the TinyLPU Avalon wrapper to appear in the
lightweight bridge window with this internal map:

| Offset | Region |
| --- | --- |
| `0x0000` | IMEM, 1024 packed 96-bit VLIW rows |
| `0x4000` | MEM0 activations/scratch/KV cache |
| `0x8000` | MEM1 model weights |
| `0xC000` | `CTRL_RUN` |
| `0xC004` | `CTRL_PC_LOAD` |
| `0xC008` | `CTRL_CYCLES` |

On the DE1-SoC Linux side, the lightweight HPS-to-FPGA bridge is normally
mapped at physical `0xFF200000`. If Platform Designer assigns the LPU slave an
additional offset under that bridge, pass the adjusted base with `--base`.

Example: if the LPU slave is at lightweight bridge offset `0x00001000`, run:

```sh
sudo ./microgpt_hps_runtime --base 0xff201000
```

## Important bitstream note

The existing JTAG build exports a JTAG-to-Avalon master. That is perfect for the
Windows/System Console testbench, but ARM Linux cannot use it as `/dev/mem`.

For this runtime, the Platform Designer system must expose the HPS lightweight
AXI/Avalon master to the same `lpu_de1_soc_wrapper` Avalon slave. This changes
the board integration around the LPU, not the TinyLPU compute core.

