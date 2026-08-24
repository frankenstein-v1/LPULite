# LPULite

**A learn-by-building guide to a language-model accelerator, from MicroGPT to
SystemVerilog to a working FPGA.**

LPULite is an 8-lane INT8 language-processing accelerator built from scratch
for the Intel/Altera DE1-SoC. This repository is both the implementation and a
guide to the question behind it:

> What has to happen between “run this transformer” and actual data moving
> through memory, matrix units, vector units, and control logic?

The finished system runs a compiled character-level MicroGPT forward pass on
the LPU, including RMSNorm, Q/K/V projections, causal attention, softmax,
residual connections, the MLP, and the language-model head.

Read the illustrated project story at **[lpulite.com](https://www.lpulite.com/)**.

Created by **Michael Trbovic**, **Saksham Batra**, and **Arjun Harinath**.

## What this repository contains

This is the complete LPULite project, not only the accelerator RTL. It contains
the model used to exercise the machine, the compiler that turns that model into
LPU instructions, the synthesizable hardware, the verification environment,
the DE1-SoC deployment stack, and an experimental ASIC visualization flow.

The repository is organized as one end-to-end pipeline:

```text
model/          define, train, quantize, and run MicroGPT in Python
   |
   v
model/tools/    pack weights and compile a static LPU memory/schedule image
   |
   v
src/            execute that image in the synthesizable LPU core
   |
   +--> tb/ and model/tests/       unit and model-level RTL verification
   |
   +--> synthesis/                 DE1-SoC wrapper, ARM runtime, and Quartus flow
   |
   +--> asic/                      SKY130/OpenLane physical-design showcase
```

The major directories are:

```text
LPULite/
├── model/
│   ├── microgpt.py       dependency-free reference model and training entry point
│   ├── datasets/         names, stories, and other training datasets
│   ├── artifacts/        checkpoints, packed weights, and compiled model images
│   ├── scripts/          training, generation, and CPU inference commands
│   ├── tools/            quantization, memory packing, VLIW compilation, and export
│   └── tests/            tokenizer, model, compiler, and RTL-backed inference tests
├── src/                  synthesizable SystemVerilog LPU core
├── tb/                   focused RTL unit testbenches
├── synthesis/
│   ├── rtl/              HPS/Avalon and DE1-SoC wrapper RTL
│   ├── linux/            ARM Cortex-A9 runtime and MMIO support
│   ├── project/          Quartus project sources
│   ├── build/            verified release project and bitstream
│   ├── scripts/          model, software, and FPGA build automation
│   ├── tests/            wrapper-level cocotb and physical-board tests
│   └── docs/             board setup, architecture audits, and runbooks
├── asic/                 SKY130 SRAM, synthesis, OpenLane, and GDS3D experiment
├── archive/              older implementations kept for historical reference
└── misc/                 supporting project material that is not in the main flow
```

## Where to find the important parts

| If you want to find... | Open this first | Then look at... |
| --- | --- | --- |
| The reference transformer | [`model/microgpt.py`](model/microgpt.py) | `model/scripts/` and [`model/README.md`](model/README.md) |
| Training data and saved weights | `model/datasets/` | `model/artifacts/` |
| CPU INT8 inference | [`model/scripts/run_microgpt_int8.py`](model/scripts/run_microgpt_int8.py) | `model/tests/` |
| The full-model LPU compiler | [`model/tools/compile_microgpt_lpu.py`](model/tools/compile_microgpt_lpu.py) | [`model/tools/lpu_vliw_compiler.py`](model/tools/lpu_vliw_compiler.py) |
| The LPU top level | [`src/lpu.sv`](src/lpu.sv) | [`src/lpu_pkg.sv`](src/lpu_pkg.sv) |
| Instruction decoding and scheduling | [`src/icu.sv`](src/icu.sv) | the VLIW compiler above |
| Matrix multiplication | [`src/mxm.sv`](src/mxm.sv) | [`src/mac.sv`](src/mac.sv) and [`src/acc.sv`](src/acc.sv) |
| Softmax, RMSNorm, quantization, residuals, or RoPE | [`src/vxm.sv`](src/vxm.sv) | `src/softmax.sv`, `src/rmsnorm.sv`, `src/quant.sv`, `src/residual_add.sv`, and `src/vxm_rope.sv` |
| K transpose and scalar broadcast | [`src/sxm.sv`](src/sxm.sv) | `tb/sxm_scaled_broadcast_tb.sv` |
| Data memory behavior | [`src/mem.sv`](src/mem.sv) | [`src/lpu_pkg.sv`](src/lpu_pkg.sv) |
| Complete wrapper simulation | [`synthesis/tests/run_microgpt_wrapper_cocotb.py`](synthesis/tests/run_microgpt_wrapper_cocotb.py) | `synthesis/tests/` |
| The ARM-side FPGA runtime | [`synthesis/linux/src/microgpt_hps_runtime.c`](synthesis/linux/src/microgpt_hps_runtime.c) | [`synthesis/linux/README.md`](synthesis/linux/README.md) |
| The verified FPGA setup | [`synthesis/docs/microgpt_fpga_runbook.md`](synthesis/docs/microgpt_fpga_runbook.md) | `synthesis/rtl/` and `synthesis/project/` |
| The physical-design experiment | [`asic/README.md`](asic/README.md) | `asic/config.json` and `asic/src/` |

If you are new to the project, read **The machine in one picture**, then
**Guides 1–6**. If you already know which layer you want to change, the table
above is the fastest entry point.

## What you will learn

Following this guide, you will see how to:

1. Train and inspect a small transformer in Python.
2. Convert its learned tensors into block-scaled INT8 data.
3. Lay weights, activations, constants, and scratch space out in LPU memory.
4. Compile the model into a static 96-bit VLIW instruction stream.
5. Execute matrix work in the MXM and vector/reduction work in the VXM.
6. Build causal attention from K transpose, `QK^T`, softmax, and `PV`.
7. Verify the complete forward pass in RTL simulation.
8. Run the same compiled model through the ARM+FPGA system.
9. Explore the core as a SKY130 physical-design showcase.

You can follow the whole path or stop at the layer you care about. Python-only
model work does not require FPGA tools, and RTL simulation does not require a
DE1-SoC board.

## The result first

The completed FPGA demonstration uses:

| Item | Result |
| --- | --- |
| FPGA | Cyclone V `5CSEMA5F31C6` on the Terasic DE1-SoC |
| Datapath | 8-lane block-scaled INT8 with INT32 accumulation |
| Model | 1 transformer layer, `d_model=16`, 4 heads, 16-token context, 27-character vocabulary |
| Compiled image | 4,920 96-bit VLIW instructions and 974 MEM1 rows |
| Attention | FPGA SXM K transpose, MXM `QK^T`/`PV`, and VXM LUT softmax |
| Throughput | Approximately 41.3 complete LPU forward passes per second |
| Demo | `sat` to `satvik` at approximately 20.7 generated characters/s and 145 ms prompt-to-completion |

The prediction is learned from the checkpoint; it is not hardcoded in the RTL
or host runtime.

## The machine in one picture

```text
INT8 checkpoint
      |
      v
Python compiler ---- packed MEM1 image + static VLIW schedule
      |                                  |
      v                                  v
ARM Cortex-A9 host ----------------> ICU control
                                         |
                        +----------------+----------------+
                        |                |                |
                        v                v                v
                    MEM0/MEM1          MXM              VXM
                    data storage    matrix math     vector/reductions
                                         ^
                                         |
                                        SXM
                                  transpose/broadcast
                                         |
                                         v
                                      logits
                                         |
                                         v
                               ARM argmax + terminal text
```

The core is statically scheduled. The compiler decides where values live and
when each unit runs; the ICU issues those decisions in parallel. Supporting a
compatible new model means generating a new memory image and schedule, not
adding model-specific lookup logic to the hardware.

## Guide 1: start with the model

Before thinking about RTL, establish the computation that the hardware must
preserve. The dependency-free reference is
[`model/microgpt.py`](model/microgpt.py).

Train it and save both the floating-point shadow weights and the deployable
INT8 checkpoint:

```sh
python model/microgpt.py
```

The important outputs are:

```text
model/artifacts/microgpt_weights.json       float shadow checkpoint
model/artifacts/microgpt_weights_int8.json  block-scaled INT8 checkpoint
```

Run the quantized checkpoint on the CPU first:

```sh
python model/scripts/run_microgpt_int8.py ken
```

At this point, focus on the transformer dataflow rather than the hardware:

```text
token embedding
  -> RMSNorm
  -> Q, K, V projections
  -> causal self-attention
  -> output projection + residual
  -> RMSNorm
  -> MLP + residual
  -> LM head
  -> next-token logits
```

**Checkpoint:** the INT8 reference produces sensible logits before the model
is lowered to hardware. See [the model guide](model/README.md) for training,
datasets, checkpoints, and CPU/RTL-backed inference.

## Guide 2: turn tensors into rows

The LPU does not consume Python tensors. It consumes rows with eight signed
INT8 lanes plus block-scale metadata. Learned matrices are therefore packed
into 72-bit memory rows and reorganized for the way the MXM reuses operands.

The key transition is:

```text
named model tensor
  -> blocks of 8 values
  -> signed INT8 lanes + shared power-of-two scale
  -> packed 72-bit rows
  -> fixed MEM1 addresses
```

Inspect these files in order:

1. [`model/tools/pack_microgpt_lpu.py`](model/tools/pack_microgpt_lpu.py) —
   checkpoint packing.
2. [`model/tools/compile_microgpt_lpu.py`](model/tools/compile_microgpt_lpu.py)
   — memory layout and full-model schedule construction.
3. [`src/lpu_pkg.sv`](src/lpu_pkg.sv) — the RTL-visible widths and types.

The general arithmetic contract is INT8 operands, wider accumulation, then an
explicit requantization before a result returns to the narrow datapath.

**Checkpoint:** every tensor needed by the forward pass has an address,
layout, shape, and scale interpretation that both the compiler and RTL agree
on.

## Guide 3: compile operations into time

A model graph says *what* to calculate. LPULite's compiler must also decide
*where the operands are*, *which unit performs the work*, and *which cycle
transfers the result*.

Generate the compiled model artifacts with:

```sh
python model/tools/compile_microgpt_lpu.py
```

The compiler emits a packed MEM1 image and a static instruction program. Each
96-bit VLIW word can coordinate independent actions such as:

- reading MEM0 or MEM1;
- selecting westbound and eastbound bus sources;
- loading MXM activation or weight ingress;
- starting SXM or VXM work;
- writing a result back to memory.

That is the central idea of this accelerator: the compiler exposes overlap,
while the ICU keeps the hardware simple and deterministic.

Read [`model/tools/lpu_vliw_compiler.py`](model/tools/lpu_vliw_compiler.py)
beside [`src/icu.sv`](src/icu.sv) to see the software and hardware definitions
of the same instruction.

**Checkpoint:** a forward pass has become a reproducible sequence of memory
movement and unit-level operations.

## Guide 4: understand the four hardware jobs

Do not begin by reading the entire top level. Follow one value through these
units:

| Unit | Question it answers | Main implementation |
| --- | --- | --- |
| MEM0/MEM1 | Where is the current value stored? | [`src/mem.sv`](src/mem.sv) |
| MXM | How are learned linear layers and attention products calculated? | [`src/mxm.sv`](src/mxm.sv) |
| SXM | How is data rearranged before matrix work? | [`src/sxm.sv`](src/sxm.sv) |
| VXM | How are lane-wise and reduction operations calculated? | [`src/vxm.sv`](src/vxm.sv) |
| ICU | What runs and moves on this cycle? | [`src/icu.sv`](src/icu.sv) |

The MXM is an 8x8 INT8 MAC array. It handles the model's learned projections,
attention score products, weighted-value products, and language-model head.

The VXM handles the operations that are awkward for a MAC array: chunked
RMSNorm, LUT-based chunked softmax, ReLU, residual operations, quantization,
and attention-head merging.

The SXM does data movement, not nonlinear math. In the attention path it
transposes aligned K tiles so the MXM can calculate `QK^T`; it can also
broadcast scalar lanes when requested.

**Checkpoint:** you can point to a transformer operation and identify exactly
which unit owns it.

## Guide 5: assemble causal self-attention

For one layer, attention is built from ordinary steps rather than one opaque
“attention block”:

```text
normalized X
   |\
   | +--> MXM: K = X Wk --> SXM: transpose K --+
   |                                             |
   +----> MXM: Q = X Wq -------------------------+--> MXM: QK^T
   |                                                       |
   |                                             causal limits + softmax
   |                                                       |
   +----> MXM: V = X Wv -----------------------------------+--> MXM: P V
                                                                  |
                                                        output projection Wo
                                                                  |
                                                            residual add
```

During prefill, K and V for each prompt position are also written into the KV
cache. During decode, the new query attends over the cached positions plus the
current K/V entry. Causality is enforced by limiting which score elements may
contribute before softmax normalization.

The production path keeps K transpose, `QK^T`, softmax, and `PV` on the FPGA.
The ARM manages positions, valid lengths, memory layouts, and execution; it
does not calculate those attention values in the `fpga-mxm` mode.

For the exact ownership and measured behavior, read the
[attention/runtime audit](synthesis/docs/lpu_attention_runtime_and_performance_audit.md).

**Checkpoint:** the attention output is produced by `softmax(QK^T) V`, with
future positions excluded and K/V retained for subsequent decode steps.

## Guide 6: finish the transformer layer

Attention is only half of the layer. The complete pre-norm flow is:

```text
X
 -> RMSNorm
 -> attention
 -> output projection
 -> add X residual
 -> RMSNorm
 -> FC1
 -> ReLU
 -> FC2
 -> add attention residual
 -> LM head
 -> logits
```

This stage is useful because it shows the accelerator alternating between
matrix-heavy and vector-heavy work. Intermediate values return to memory so a
later instruction can reuse them, and the compiler schedules independent
ingress work whenever the datapath permits overlap.

**Checkpoint:** the final logits agree with the quantized software reference,
not merely with an isolated matmul test.

## Guide 7: prove it in simulation

Run the HPS/LPU wrapper integration test:

```sh
python synthesis/tests/run_microgpt_wrapper_cocotb.py
```

This drives the same Avalon-MM interface used by the ARM runtime, loads the
compiled model and microcode, executes the forward pass, and checks the next
token.

Useful model-level checks are:

```sh
python -m pytest model/tests/test_tokenizer.py model/tests/test_lm_shift.py
```

When debugging, move down one level at a time:

```text
model mismatch
  -> compiled memory/schedule mismatch
  -> wrapper/MMIO mismatch
  -> unit handshake mismatch
  -> arithmetic mismatch
```

The [debugging and generality notes](synthesis/docs/lpu_debugging_and_generality_status.md)
record which paths are production-ready and which constraints are deliberate.

## Guide 8: run it on the DE1-SoC

The canonical board procedure is the
**[MicroGPT FPGA runbook](synthesis/docs/microgpt_fpga_runbook.md)**. It covers
the Linux image, UART, Ethernet, JTAG, FPGA programming, runtime transfer,
bridge probing, inference, and benchmarking.

Once the board is configured and the runtime is in `/home/root/linux`:

```sh
cd /home/root/linux
./microgpt_hps_runtime \
  --attention fpga-mxm \
  --broadcast host \
  --decode greedy \
  --benchmark
```

Example output:

```text
Prompt > sat
satvik
```

To rebuild the release from Windows PowerShell with Quartus 25.1 and WSL ARM
cross-compilation tools installed:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\synthesis\scripts\build_microgpt_fpga_release.ps1
```

For a software-only rebuild:

```powershell
.\synthesis\scripts\build_microgpt_fpga_release.ps1 -SkipQuartus
```

The verified bitstream is:

```text
synthesis/build/lpu_lite_de1_soc_hps/lpu_lite_de1_soc_hps.sof
SHA-256: 8102D2F0DFEE19AC8F49AC1335D4AB93A9914107E071829572B7188DE692D02B
```

Quartus Prime 25.1 Standard Lite reported:

| Resource | Utilization |
| --- | ---: |
| Logic | 32,070 / 32,070 ALMs (100%) |
| Registers | 10,031 |
| Block-memory bits | 2,462,976 / 4,065,280 (61%) |
| RAM blocks | 314 / 397 (79%) |
| DSP blocks | 87 / 87 (100%) |

Static timing analysis reports positive setup and hold slack for the LPU and
HPS bridge clocks.

## Guide 9: explore the physical core

The [`asic/`](asic/) flow is a visual and physical-design showcase of the LPU
core using SKY130 and OpenLane 2. It preserves the 8x8 MXM and 8-lane VXM while
using public SRAM macros and a smaller memory configuration suitable for the
experiment.

Run the local elaboration audit with:

```sh
make asic-audit
```

With OpenLane and the SKY130 PDK installed:

```sh
make asic-gds
```

This flow is useful for seeing what the project becomes after synthesis and
placement. It is an architectural showcase, not a claim that the FPGA release
is a tapeout-ready ASIC. See [the ASIC guide](asic/README.md) for memory macros,
the 100 MHz target, GDS generation, and GDS3D rendering.

## Reference documentation

- [MicroGPT FPGA setup and demo](synthesis/docs/microgpt_fpga_runbook.md)
- [Attention, runtime, and performance audit](synthesis/docs/lpu_attention_runtime_and_performance_audit.md)
- [LPU debugging and generality status](synthesis/docs/lpu_debugging_and_generality_status.md)
- [HPS Linux integration](synthesis/docs/de1_soc_hps_linux.md)
- [Model and compiler workflow](model/README.md)
- [FPGA synthesis layout](synthesis/README.md)
- [ASIC showcase](asic/README.md)

## License

LPULite is available under the [Apache License 2.0](LICENSE).
