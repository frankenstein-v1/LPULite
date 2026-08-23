# LPULite MicroGPT on DE1-SoC HPS Linux

This directory contains the ARM/Linux userspace runtime for driving the LPULite
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

- `src/lpulite_hps_mmio.c` / `include/lpulite_hps_mmio.h`:
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
cd "/mnt/c/Users/micha/Documents(Local)/Projects/LPULite/synthesis/linux"
make clean
make CC=arm-linux-gnueabihf-gcc LDFLAGS=-static
```

From Windows PowerShell:

```powershell
scp "C:\Users\micha\Documents(Local)\Projects\LPULite\synthesis\linux\microgpt_hps_runtime" root@192.168.1.101:/home/root/linux/
```

Then run on the board:

```sh
cd /home/root/linux
chmod +x microgpt_hps_runtime
./microgpt_hps_runtime --attention fpga-mxm --broadcast host --benchmark
```

These changes only rebuild the ARM/Linux executable. If the currently loaded
`.sof` already runs `--attention fpga-mxm` correctly, **do not rerun Quartus or
reprogram the FPGA** for this software optimization update.

Useful options:

```sh
./microgpt_hps_runtime --attention host --broadcast host --benchmark
./microgpt_hps_runtime --attention fpga-softmax --broadcast host --benchmark
./microgpt_hps_runtime --attention current
./microgpt_hps_runtime --no-load-weights
./microgpt_hps_runtime --base 0xff200000 --span 0x10000
./microgpt_hps_runtime --attention fpga-mxm --broadcast host --benchmark --prompt sat --repeat 10
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

`--broadcast host` is the default because full-chip simulation found it
slightly faster for this schedule. `--broadcast sxm` is verified and executes
every compiled model broadcast in the FPGA SXM, so it remains available for
measurement on the real board. This option only selects the full-model
broadcast path: `--attention fpga-mxm` uses SXM for the K transpose regardless
of the selected broadcast mode.

`--benchmark` reports model-load time and per-prompt ARM+FPGA wall-clock
measurements. Its end-to-end output tokens/s includes reset, MMIO transfers,
IMEM paging, FPGA execution/polling, the selected attention assistance, and
decode logic. It excludes time spent typing and the one-time model load, which
is reported separately. It also reports TTFT, prefill LPU steps/s, and decode
LPU steps/s. It now also reports MMIO row/word counts and how many physical
IMEM rows were skipped by the software shadow. Use `--prompt` with `--repeat`
to avoid human typing delays and obtain an aggregate rate over multiple runs:

```sh
./microgpt_hps_runtime \
  --attention fpga-mxm --broadcast host --decode greedy \
  --benchmark --prompt sat --repeat 10
```

The aggregate line is the most stable tokens/second number. Output-token rate
still depends on the prompt because different names generate different numbers
of output tokens and require different numbers of autoregressive LPU steps.

## Software-only performance changes

The optimized runtime keeps the existing FPGA RTL, Avalon register map, and
4,920-instruction model schedule unchanged. It reduces ARM/Linux overhead by:

- polling exact-cycle completion immediately by default (`--settle-us 0`),
  instead of sleeping between every poll;
- batching consecutive row transfers so one ARM memory barrier covers a batch,
  while preserving word 2 as the hardware row-commit write;
- shadowing the 1,024-row physical IMEM and transmitting only changed ranges;
- loading only the active page, eight retirement NOPs, and the required
  stopped-PC guard row, instead of clearing all unused IMEM rows;
- retaining packed K/V rows in an ARM mirror after their first FPGA readback,
  avoiding repeated historical-cache reads while arranging the next FPGA
  attention invocation;
- staging only active sequence positions after causal masking and preserving
  reset-established masked/zero rows for inactive positions;
- resetting only state that must be initialized between prompts rather than
  clearing the entire scratch and KV address ranges; and
- compiling the runtime at `-O3` for Cortex-A9 hard-float/NEON.

The ARM mirror does not perform QK, softmax, or PV in the default
`--attention fpga-mxm` mode. It holds the already quantized K/V representation
needed to lay out active operands. SXM still performs `K^T`, MXM performs QK
and PV, and VXM performs softmax.

The software keeps the tested 900-instruction paging boundary. Experiments
with a 1,016-row page and with a compacted FC1/ReLU schedule produced incorrect
logits in full-chip simulation, so those unsafe changes were rejected.

Two otherwise attractive optimizations are outside the software-only scope:

- HPS DMA is disabled in the current Platform Designer system, so adding DMA
  transfers requires a hardware integration change.
- The Platform Designer/LPU clock is currently derived from the 50 MHz board
  clock with `qsys_clk_div[2]` (6.25 MHz). Raising it requires changing the FPGA
  top-level clock integration and rebuilding the bitstream.

## Required FPGA address map

The Linux runtime expects the LPULite Avalon wrapper to appear in the
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
the board integration around the LPU, not the LPULite compute core.
