# TinyLPU debugging changes and general-purpose status

Date: 2026-08-11

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
| Host-staged broadcasts | The SXM broadcast path produced malformed replicated matrix input rows | The ARM now reliably replicates rows before each matrix operation |
| Host causal attention | Complete KV attention was not yet scheduled correctly on the LPU | ARM computes causal attention using FPGA-generated Q/K/V |
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

With:

```sh
--attention host --decode greedy
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

The ARM performs:

- Loading weights and VLIW pages
- Selecting token and position embeddings
- Managing the KV-cache addresses
- Causal `QKᵀ` attention scores
- Attention softmax
- Weighted sum of V
- Replicating rows for matrix input
- Greedy argmax and BOS handling
- Sequencing the next token

The host attention implementation is in `synthesis/linux/src/microgpt_hps_runtime.c`.

The ARM also replaces several failing SXM broadcast operations by staging replicated rows through HPS MMIO.

Therefore, the current system does **not** meet the original “entire forward pass 100% on the FPGA LPU” requirement. Most learned computation is on the FPGA, but causal attention is currently calculated by the ARM.

The reciprocal-LUT hardware softmax exists and passes its focused test, but it is not being used for causal attention in the current `--attention host` execution path.

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
- Attention is host-specific.
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

1. Fix the SXM broadcast path in RTL instead of staging broadcasts on ARM.
2. Compile causal attention and KV-cache access onto the LPU.
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

The hardware is now much closer to being a reusable programmable accelerator, but the host attention and broadcast workarounds are the main reasons the complete platform should not yet be described as fully general-purpose.
