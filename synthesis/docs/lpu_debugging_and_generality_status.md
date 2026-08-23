# LPULite debugging changes and general-purpose status

Date: 2026-08-12

## User prompt

> ok I dont really care if its a model issue rn, what I want to know is throughout all of the debugging that we have done over the last couple of days what was it that changed and why? why did it fix the output and is our LPU general pourpose now? like if I give it a new model and a new compiler designd for that model will it get the same outputs?

## Answer

The output was fixed because we repaired real data movement, timing, scaling, and scheduling errors—not because the nine names were hardcoded. However, the current system is only partially general-purpose, and the complete forward pass is not yet 100% on the FPGA.

## What changed and why it mattered

| Change | Original problem | Result |
|---|---|---|
| M10K/IMEM/MEM read timing | FPGA block RAM has registered latency, but the scheduler and HPS bridge sampled rows too early or gated them back to zero | Weights, operands, and logits became readable instead of zero |
| Exact-cycle execution and soft reset | Pages could overrun and old FIFO/scratch state survived between prompts | Repeated prompts became deterministic |
| Clock reduced to 6.25 MHz | The LPU could not meet timing at the original 50 MHz fabric clock | Physical FPGA execution became timing-safe |
| Q8.8 scale correction | Quantizer reported the shift count as the exponent, even though inputs were Q8.8 | Values acquired the correct real magnitude |
| Signed rounding correction | Negative values were rounded incorrectly | Quantized activations became symmetric and consistent |
| Chunked RMSNorm | MicroGPT’s 16-element embedding was being normalized incorrectly or in isolated 8-element rows | Both rows now share one RMS value |
| RMSNorm Q-format correction | Reciprocal RMS and gamma fractional bits were not removed correctly | Normalized values stopped being incorrectly scaled |
| Mixed-scale MAC alignment | Products with different block exponents were added as raw integers | Matrix-vector products now represent the correct real sum |
| Schedule graph correction | Residual values and pre-attention RMSNorm did not exactly follow the model graph | The scheduled transformer matches the reference graph |
| ReLU pipeline scheduling | ReLU control disappeared before queued data reached VXM, and results were read too early | Negative MLP values were removed from the correct rows |
| FPGA SXM broadcasts | SXM captured stale rows, ignored valid gaps, dropped the row exponent, and emitted lanes 1..7 instead of 0..7 | RTL/wrapper simulation passes, but physical-board inference still corrupts the SXM path; ARM staging is the board-safe default while the focused hardware probe identifies the synthesized failure |
| VXM softmax backpressure | The softmax core produced 16 rows but the residual/quant handoff overwrote five before they reached the output FIFO | All 16 chunks now drain through VXM, and the compiled 544-instruction FPGA softmax microkernel passes wrapper simulation |
| Resident MXM attention microkernel | QK dot products and PV weighted sums were still ARM arithmetic | A 896-instruction resident image time-multiplexes the existing MXM for QK/PV and VXM for softmax; full `sat -> satvik -> BOS` wrapper simulation passes |
| Greedy/BOS decoding | Earlier runners could force names or ignore the end token | Default output now comes from raw logits and stops on BOS |

### Why it originally printed `a`

When all logits were zero, greedy decoding selected token index zero every time. Token zero is `a`, producing:

```text
sat -> sata
```

That was not a model prediction. It was the default result of broken, all-zero logits.

### Why it later became random

Once data was nonzero, stale FIFO rows, page overruns, missing resets, ReLU latency, and malformed broadcasts caused different scratch-memory values between prompts. Therefore, the same prompt could produce different answers.

### Why it then became consistently wrong

After reset and execution control were fixed, it became deterministic, but the arithmetic was still wrong. In particular, the MAC effectively did this:

```text
raw_product_0 + raw_product_1
```

even if those products represented:

```text
raw_product_0 × 2^-5
raw_product_1 × 2^-8
```

Those integers cannot be directly added. `src/mac.sv` and `src/mxm.sv` now align them to a common exponent first.

That mixed-scale correction was one of the most important genuine LPU fixes.

## What currently runs where

With the newly verified full-chip simulation path (physical-board validation is
the next deployment step):

```sh
--attention fpga-mxm --broadcast host --decode greedy
```

the FPGA LPU performs:

- INT8 matrix-vector multiplication
- Token and position residual addition
- RMSNorm
- Q, K, and V projections
- Attention output projection
- MLP FC1
- ReLU
- MLP FC2
- Residual additions
- LM-head projection
- Final logits
- FPGA storage of K/V rows
- Causal QK dot products through MXM
- Attention exponential, reciprocal, and probability normalization through VXM softmax
- PV weighted sums through MXM
- Attention-head row merging through VXM residual-add

The ARM performs:

- Loading weights and VLIW pages
- Selecting token and position embeddings
- Managing the KV-cache addresses
- Aligning block-scaled K tiles to the SXM's single tile exponent
- Causal/dynamic-length mask staging
- Matrix-input row replication/broadcast staging while the synthesized SXM path is diagnosed
- Greedy argmax and BOS handling
- Sequencing the next token

The FPGA attention scheduler is in `synthesis/linux/src/microgpt_hps_runtime.c`;
the resident microcode is generated by `model/tools/compile_microgpt_lpu.py`.

`--broadcast host` is currently required for correct physical-board output.
`--broadcast sxm` remains available for the focused probe and RTL debugging.

K transpose, QK, and PV no longer run on ARM in `fpga-mxm` mode: SXM performs
the position-by-dimension transpose and MXM performs the attention multiply-
accumulates. The platform is still not an autonomous FPGA-only inference
engine: ARM software aligns the K block scale required by the current SXM ABI,
manages cache layout and causal length, invokes microcode entry points, selects
tokens, and handles terminal I/O.

The reciprocal-LUT hardware softmax is used by both `--attention fpga-mxm` and
`--attention fpga-softmax`. The latter and `--attention host` remain diagnostic
comparison modes. The new default `fpga-mxm` mode performs all attention
multiply-accumulate arithmetic in the existing LPU datapaths.

## Is the LPU general-purpose now?

It is better described as a programmable tensor accelerator.

It has reusable hardware for:

- 8×8 tiled matrix multiplication
- Block-scaled INT8 arithmetic
- Memory movement
- Quantization
- Residual operations
- ReLU
- RMSNorm
- Softmax
- Programmable 96-bit VLIW scheduling

Those datapath fixes are model-independent. Mixed-scale MAC alignment, quantization correction, memory timing, and reset behavior should benefit any model.

But the current software stack remains MicroGPT-specific:

- The compiler explicitly accepts only:

```text
n_layer    = 1
n_embd     = 16
block_size = 16
n_head     = 4
vocab_size = 27
```

- Scratch addresses are hardcoded.
- RMSNorm is configured for exactly two 8-lane chunks.
- Attention layout/orchestration is MicroGPT-specific, although QK/PV arithmetic is on MXM.
- Broadcast split locations are exported specifically for MicroGPT.
- The runtime knows the exact MicroGPT layer order.
- The FPGA is currently using 100% of its ALMs and DSP blocks.

So the hardware has general-purpose building blocks, but the compiler/runtime are not yet model-agnostic.

## What happens with a new model?

### New weights, identical architecture

This is the easiest case.

A new model with the same architecture and numeric format should require:

1. Exporting the new INT8 checkpoint.
2. Regenerating MEM1 and VLIW artifacts.
3. Regenerating the C header.
4. Rebuilding the ARM runtime.
5. Loading the new weights.

It should **not** require another Quartus compile because the RTL has not changed.

It should match a bit-accurate INT8 reference, assuming the compiler and runtime use the same scaling, rounding, saturation, memory latency, and attention implementation.

### Different architecture using supported operations

It may work with a new compiler, but that compiler must generate more than instructions. It must also generate:

- Weight layout
- Scratch-memory allocation
- Instruction pages
- Pipeline waits
- Broadcast operations
- KV-cache layout
- Attention orchestration
- Output-decoding metadata

Different matrix sizes can potentially be tiled through the 8×8 MXM. However, different RMSNorm widths, softmax lengths, unsupported operations, or larger memories may require RTL changes and another FPGA compile.

### Exact equality with a floating-point model

No. The LPU uses:

- Block-scaled INT8
- INT32 accumulation
- LUT approximations
- Fixed-point rounding
- Saturation

The correct comparison target is a bit-accurate quantized reference, not ordinary PyTorch FP32 output.

## What is still needed before claiming general-purpose correctness?

The biggest remaining tasks are:

1. Validate the new SXM K-transpose path in `--attention fpga-mxm` on the physical DE1-SoC board.
2. Move K-tile exponent alignment, cache layout, and causal-mask staging out of ARM if autonomous FPGA scheduling is required.
3. Replace the hardcoded MicroGPT compiler with a shape-driven compiler.
4. Define a precise LPU numeric/latency contract.
5. Create a bit-accurate instruction-set emulator.
6. Differentially test every intermediate tensor, not only final tokens.
7. Test multiple random checkpoints and at least one unrelated model.

We have proven:

```text
current checkpoint + current scheduler + current host runtime
```

for several completions.

We have not yet proven:

```text
arbitrary supported model + arbitrary new compiler
```

The hardware now executes QK, softmax, and PV. The remaining boundary is
control/layout work on ARM and the model-specific compiler/runtime, not missing
attention arithmetic hardware.
