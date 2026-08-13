# TinyLPU MicroGPT on DE1-SoC HPS Linux

This directory contains the ARM/Linux userspace runtime for driving the TinyLPU
through the DE1-SoC HPS-to-FPGA lightweight bridge instead of USB-JTAG/System
Console.

The runtime uses the same generated model artifacts as the JTAG testbench:

- `model/artifacts/fpga_microgpt/microgpt_scheduler_mem1.hex`
- `model/artifacts/fpga_microgpt/microgpt_decode_vliw.hex`
- `model/artifacts/fpga_microgpt/microgpt_decode_schedule.json`
- `model/artifacts/fpga_microgpt/microgpt_decode_trace.json`
- `model/artifacts/fpga_microgpt/microgpt_softmax_vliw.hex`
- `model/artifacts/fpga_microgpt/microgpt_attention_vliw.hex`

## Files

- `src/tinylpu_hps_mmio.c` / `include/tinylpu_hps_mmio.h`:
  `/dev/mem` MMIO driver for the lightweight bridge.
- `src/microgpt_hps_runtime.c`:
  interactive char-level MicroGPT terminal.
- `include/microgpt_hps_image.h`:
  generated C image containing the VLIW schedule and MEM1 model rows.
- `Makefile`:
  regenerates the image header and builds the runtime.

## Cross-compile on the laptop

The Terasic Linux image does not include GCC. Build a static ARMv7 hard-float
binary from the WSL terminal, then copy it to the board:

```sh
cd "/mnt/c/Users/micha/Documents(Local)/Projects/tinyLPU/synthesis/linux"
make clean
make CC=arm-linux-gnueabihf-gcc LDFLAGS=-static
```

From Windows PowerShell:

```powershell
scp "C:\Users\micha\Documents(Local)\Projects\tinyLPU\synthesis\linux\microgpt_hps_runtime" root@192.168.1.101:/home/root/linux/
```

Then run on the board:

```sh
cd /home/root/linux
chmod +x microgpt_hps_runtime
./microgpt_hps_runtime --attention fpga-mxm --broadcast host --benchmark
```

Useful options:

```sh
./microgpt_hps_runtime --attention host --broadcast host --benchmark
./microgpt_hps_runtime --attention fpga-softmax --broadcast host --benchmark
./microgpt_hps_runtime --attention current
./microgpt_hps_runtime --no-load-weights
./microgpt_hps_runtime --base 0xff200000 --span 0x10000
```

`--attention fpga-mxm` is the default. It time-multiplexes the existing MXM for
both QK dot products and PV weighted sums, and executes exp, reciprocal, and
normalization in the existing 16-chunk VXM softmax. Before QK, the existing SXM
transposes each aligned 8x8 K tile from positions-by-dimensions into
dimensions-by-positions. The ARM aligns the block-scaled K rows to the SXM's
single tile exponent and writes the causal/dynamic-length mask. SXM sends its
transposed dimensions through the existing VXM packing/store path into the
MEM1 layout consumed by MXM, so ARM does not copy or transpose the emitted
rows. The 1020-row resident attention
image contains separately callable block-0 K-transpose, block-1 K-transpose,
QK, softmax, PV, and head-merge entry points.

`--attention fpga-softmax` is retained as a comparison mode: VXM computes
softmax while ARM computes QK/PV. `--attention host` performs all causal
attention arithmetic on ARM.
`--attention current` stages only the current V row and is a bring-up mode, not
exact multi-token attention.

`--broadcast host` is the board-safe default while the synthesized SXM path is
being diagnosed. `--broadcast sxm` executes every compiled model broadcast in
the FPGA SXM and is retained as an explicit diagnostic mode. This option only
selects the full-model broadcast path: `--attention fpga-mxm` uses SXM for the
K transpose regardless of the selected broadcast mode.

`--benchmark` reports model-load time and per-prompt ARM+FPGA wall-clock
measurements. Its end-to-end output tokens/s includes reset, MMIO transfers,
IMEM paging, FPGA execution/polling, the selected attention assistance, and
decode logic. It excludes time spent typing and the one-time model load, which
is reported separately. It also reports TTFT, prefill LPU steps/s, and decode
LPU steps/s.

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
