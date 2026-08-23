# LPULite Attention, Runtime, and Performance Audit

Date: August 12, 2026

This document records the question and answer from the final LPULite debugging
discussion. It explains how to verify that attention executes on the FPGA,
which parts of the LPU are active, which work remains on the ARM, what the
runtime C functions do, and how to interpret the measured tokens/second.

## Original question

> OK, so it is working. How do I know you are not just computing the attention
> of the model on the ARM CPU? I want to make sure you are using the LPU. Also,
> list what each component in the LPU is doing, what features of those
> components are being used or not used, and explain the tokens/second because
> they are different every run. Also explain what each function in the runtime
> C code does because some of it looks like it is running attention there.

The observed results were:

```text
Prompt > sat
satvik
[perf] prompt_tokens=3 output_tokens=3 lpu_steps=6 (prefill=4 decode=2)
[perf] request=0.217 s  TTFT=0.145 s  end_to_end=13.828 output tokens/s
[perf] prefill=0.143 s (28.061 LPU steps/s)  decode_steps=0.072 s (27.961 LPU steps/s)

Prompt > su
surya
[perf] prompt_tokens=2 output_tokens=3 lpu_steps=5 (prefill=3 decode=2)
[perf] request=0.181 s  TTFT=0.110 s  end_to_end=16.573 output tokens/s
[perf] prefill=0.107 s (28.145 LPU steps/s)  decode_steps=0.071 s (28.001 LPU steps/s)

Prompt > m
michael
[perf] prompt_tokens=1 output_tokens=6 lpu_steps=7 (prefill=2 decode=5)
[perf] request=0.253 s  TTFT=0.074 s  end_to_end=23.749 output tokens/s
[perf] prefill=0.071 s (28.172 LPU steps/s)  decode_steps=0.179 s (27.997 LPU steps/s)

Prompt > y
yash
[perf] prompt_tokens=1 output_tokens=3 lpu_steps=4 (prefill=2 decode=2)
[perf] request=0.145 s  TTFT=0.074 s  end_to_end=20.653 output tokens/s
[perf] prefill=0.071 s (28.174 LPU steps/s)  decode_steps=0.071 s (28.064 LPU steps/s)

Prompt > s
saksham
[perf] prompt_tokens=1 output_tokens=6 lpu_steps=7 (prefill=2 decode=5)
[perf] request=0.253 s  TTFT=0.074 s  end_to_end=23.745 output tokens/s
[perf] prefill=0.071 s (28.180 LPU steps/s)  decode_steps=0.179 s (27.986 LPU steps/s)
```

## Short answer

With the following mode, the ARM does **not** calculate the attention QK dot
products, softmax, or PV weighted sums:

```sh
./microgpt_hps_runtime \
  --attention fpga-mxm \
  --broadcast host \
  --decode greedy \
  --benchmark \
  --settle-us 100
```

In this mode the attention data path is:

```text
ARM orchestration and MMIO staging
    |
    v
FPGA SXM: transpose K tiles
    |
    v
FPGA MXM: QK^T dot products
    |
    v
FPGA VXM: causal softmax using exp and reciprocal LUTs
    |
    v
FPGA MXM: PV weighted sums
    |
    v
FPGA VXM: merge attention heads
```

The ARM still schedules this work, manages cache addresses and the causal
length, aligns block-floating-point representations, transfers rows through
MMIO, reads logits, chooses the next token, and prints text. Therefore this is
an ARM-controlled FPGA accelerator, not a completely autonomous FPGA text
generator.

## How to prove that attention is executing on the FPGA

### 1. Select the correct runtime path

`run_token()` selects one of the following attention implementations:

| Runtime option | QK | Softmax | PV |
| --- | --- | --- | --- |
| `--attention host` | ARM | ARM | ARM |
| `--attention fpga-softmax` | ARM | FPGA VXM | ARM |
| `--attention fpga-mxm` | FPGA MXM | FPGA VXM | FPGA MXM |
| `--attention current` | Not full causal attention | Not full causal attention | Current V only |

The actual ARM QK and PV arithmetic loops are isolated in
`stage_host_attention()`. That function contains the software dot-product and
weighted-sum operations. `stage_mxm_attention()` does not contain those
multiply-accumulate loops and does not calculate `exp()` in software. It
stages representations, invokes FPGA microcode sections, and moves the results
between the LPU memories.

### 2. Run with verbose hardware-section reporting

Use one generated token to keep the log short:

```sh
./microgpt_hps_runtime \
  --verbose \
  --attention fpga-mxm \
  --broadcast host \
  --decode greedy \
  --max-new-tokens 1 \
  --settle-us 100
```

The output should report hardware execution for:

```text
K[0:8] transpose on SXM -> MEM1
K[8:16] transpose on SXM -> MEM1
QK
softmax
PV
head merge
```

`run_attention_section()` starts each resident attention microcode section at
its assigned program counter and asks the LPU to execute an exact number of
cycles. `lpulite_run_cycles_from()` then polls the FPGA run-control registers
and reads the FPGA cycle counter. A cycle mismatch produces a warning.

The cycle counter is part of the FPGA RTL and increments only while the LPU's
`run_en` is active. Seeing the expected section cycle counts without mismatch
warnings is direct evidence that the FPGA executed those microcode regions; it
is not a timer synthesized by the ARM runtime.

### 3. Confirm that the deployed binary is the one that was built

Run `sha256sum` on the laptop build and the board copy and compare the values:

```sh
sha256sum microgpt_hps_runtime
sha256sum /home/root/linux/microgpt_hps_runtime
```

This prevents an older runtime from silently selecting a different attention
path.

### 4. Simulation evidence

The focused tests verify the hardware calculations independently of the ARM:

- Both SXM K-transpose blocks produce the expected rows in MEM1.
- The MXM QK test produces the exact expected score vector.
- The full wrapper simulation completes `sat` to `satvik` and then predicts the
  stop/BOS token.
- The RMSNorm, quantizer, mixed-scale MAC, SXM, softmax LUT, and chunked
  softmax tests pass.

## What runs on the ARM

The ARM performs control and data-layout work in `--attention fpga-mxm` mode:

- Reads terminal input and performs character tokenization.
- Loads the model rows and microcode through MMIO.
- Starts and polls FPGA microcode pages and resident attention sections.
- Maintains K/V cache positions and copies cached rows.
- Supplies the dynamic causal length and masked score rows.
- Aligns K cache rows with different block exponents to the single exponent
  accepted by one SXM tile.
- Replicates Q scalars into the row layout consumed by the MXM.
- Selects the lanes for each V attention head and stages probabilities/values.
- Stages ordinary model broadcast rows when `--broadcast host` is selected.
- Reads logits and performs greedy argmax.
- Applies the current name-recognition stopping policy and prints characters.

These loops can resemble attention code because they iterate over heads,
positions, dimensions, and cache rows. In `stage_mxm_attention()`, however,
those loops arrange operands and exponents; the FPGA performs the QK and PV
multiply-accumulate arithmetic.

The offline Python compiler is not part of inference. It generates the model
memory image and static microcode before the runtime is built.

## Important stopping-policy detail

`--decode greedy` uses raw model argmax for each generated character, but the
runtime currently calls `is_target_name()` and stops as soon as the emitted
text exactly matches a name exported with the model. Thus the board can stop at
`satvik` because the ARM recognizes that complete name rather than waiting for
the model to emit BOS.

This does not change the generated characters, but it does affect when
generation stops. `--decode target` goes further and constrains each next
character to an exported target name; it is not the mode to use when measuring
unconstrained model predictions. The full-chip simulation separately verified
that the model predicts BOS after `satvik` in the tested case.

## LPU component audit

### ICU and IMEM

Used:

- Executes the 96-bit VLIW microcode schedule.
- Supports page loading for the 4,920-instruction model schedule.
- Supports direct PC loading and exact-cycle execution for resident attention
  sections.
- Issues simultaneous memory, bus, MXM, SXM, and VXM controls.

Not used by this model:

- Data-dependent branches.
- Hardware loops.
- Dynamic instruction scheduling.

The schedule is intentionally static and generated offline.

### MEM0

Used for activations and runtime scratch space, including:

- Token and positional inputs.
- Q and K activations.
- K cache.
- RMSNorm, residual, MLP, attention, and logit scratch rows.
- QK score rows, softmax input/output rows, and PV output rows.

### MEM1

Used for:

- Quantized model weights and constant rows.
- V cache.
- SXM-produced transposed K staging consumed by the MXM.
- V rows staged for PV.

### Internal buses and writeback network

Used to route rows between MEM0, MEM1, SXM, MXM, VXM, and the memory
writeback ports. The K transpose specifically uses the existing
SXM-to-VXM-to-MEM1 path; the ARM does not read back and manually transpose the
SXM results.

### SXM

Used:

- Transposes each aligned 8x8 K tile from positions-by-dimensions to
  dimensions-by-positions for `QK^T`.
- Preserves the selected block exponent while emitting transposed rows.

Not currently used in the recommended board command:

- Ordinary full-model activation broadcasting, because `--broadcast host` is
  selected as the board-safe path.
- The SXM `opcode_weight` input; it is currently unused by the RTL.

`--attention fpga-mxm` uses the SXM for K transpose regardless of the
`--broadcast` choice.

### MXM

Used for:

- All main model matrix multiplications: Q, K, V, attention output projection,
  FC1, FC2, and LM head.
- QK attention dot products.
- PV attention weighted sums.
- Signed INT8 operands, 32-bit accumulation, and block-scale alignment.

Not used by this schedule:

- Floating-point arithmetic.
- Unsigned matrix operands.
- Every possible output-row/column selection mode; the compiler consumes the
  row layout needed by this model.
- An independent runtime weight-loading opcode path; weights are staged in
  MEM1 using the compiled layout.

### VXM

Used:

- ReLU for the MLP.
- Chunked RMSNorm for the 16-element hidden vector.
- Chunked softmax for the 16-token attention window.
- Residual additions and attention-head merge.
- Signed INT8 requantization for ordinary activations.
- Unsigned probability quantization for softmax output.
- Identity/quantize/store routing for SXM transpose output.

Not used by this model:

- Bias-add stage.
- Fixed divide-by-two scale stage.
- RoPE, because this MicroGPT uses learned positional embeddings.
- RMSNorm beta; the model uses an identity gamma and no beta.
- Residual saturation mode.

### Softmax implementation

The FPGA softmax does not evaluate the exponential or division on the ARM. It
uses:

- Maximum subtraction for numerical stability.
- Constant range reduction.
- An exponential lookup table.
- A 256-entry reciprocal lookup table.
- Integer multiply/shift normalization.
- Chunking across the 16-token sequence.

The ARM still supplies causal/dynamic-length masks because the active sequence
length changes each step.

### RMSNorm implementation

RMSNorm processes the 16-element hidden vector as two 8-lane chunks. It
accumulates the squared magnitude across the chunks, uses the inverse-square-
root LUT, and applies the result to both chunks. This is FPGA arithmetic, not
ARM RMSNorm.

## Runtime C function guide

The main runtime is `synthesis/linux/src/microgpt_hps_runtime.c`.

### Timing and row conversion

| Function | Purpose |
| --- | --- |
| `monotonic_seconds()` | Reads a monotonic ARM/Linux wall clock for performance measurements. |
| `print_benchmark()` | Prints per-request TTFT, prefill, decode, and tokens/second measurements. |
| `as_mmio_row()` | Converts a generated 96-bit row into the MMIO-driver row type. |
| `s8()` | Interprets the low byte of a word as signed INT8. |
| `unpack_row()` | Extracts eight INT8 lanes and the block exponent from one packed row. |
| `row_to_vector()` | Converts packed block-floating rows to ARM `double` values; used for representation handling and host comparison paths. |
| `pack_float_row()` | Quantizes floating-point values into a packed INT8 block row with a selected exponent. |
| `pack_float_row_at_scale()` | Quantizes values using an explicitly supplied exponent, used for SXM tile alignment. |
| `pack_quant_row()` | Packs already-quantized INT8 lanes and an exponent into an MMIO row. |
| `vector_to_rows()` | Packs a 16-element host vector into two LPU rows. |
| `softmax()` | ARM reference softmax used only by `--attention host`. |
| `clamp_scale()` | Clamps a computed exponent to the signed 8-bit field. |

The presence of conversion helpers does not prove that model arithmetic runs
on ARM. They are also needed to translate between MMIO rows and the block-
floating representation expected by existing FPGA units.

### Model state and debug helpers

| Function | Purpose |
| --- | --- |
| `load_mem1()` | Loads the generated model weights/constants into FPGA MEM1. |
| `clear_runtime_state()` | Clears activation, scratch, and cache regions. |
| `reset_prompt_state()` | Resets FPGA/runtime state before a new prompt. |
| `write_step_inputs()` | Writes token and positional inputs for one forward pass. |
| `cache_current_kv()` | Copies the newly produced K/V rows into their cache locations. |
| `stage_broadcast_row()` | Replicates one source row into the MXM input layout on ARM. |
| `stage_broadcast_pair()` | Performs the same staging for a two-row hidden vector. |
| `debug_dump_row()` | Prints one packed hardware row. |
| `debug_dump_runtime_rows()` | Prints selected intermediate rows for diagnosis. |

### FPGA program execution

| Function | Purpose |
| --- | --- |
| `run_image()` | Loads and runs pages from a generated VLIW image. |
| `run_program()` | Runs a selected range of the main model schedule. |
| `run_softmax_program()` | Runs the standalone VXM softmax image used by comparison mode. |
| `load_attention_image()` | Loads the 1,020-row resident SXM/MXM/VXM attention image and parks the ICU safely. |
| `run_attention_section()` | Starts one resident attention section at its assigned PC and verifies its FPGA cycle count. |

### Attention implementations

| Function | Purpose |
| --- | --- |
| `stage_current_attention()` | Bring-up path that stages only the current V; it is not full causal attention. |
| `fpga_softmax()` | Stages scores around the standalone FPGA softmax path. |
| `stage_host_attention()` | Reference/comparison path where ARM computes QK and PV; optional FPGA softmax. |
| `stage_mxm_attention()` | Production FPGA-attention path: ARM stages layout/scales/masks, SXM transposes K, MXM computes QK and PV, VXM computes softmax and merges heads. |

### Token execution and decoding

| Function | Purpose |
| --- | --- |
| `decode_logits()` | Reads and dequantizes FPGA logit rows. |
| `run_token()` | Executes one complete next-token forward pass: input, prefix, cache update, selected attention path, suffix, and logits. |
| `encode_prompt()` | Converts prompt characters into vocabulary IDs. |
| `greedy_next()` | Selects the highest-logit token. |
| `token_char()` | Maps a token ID back to a printable character. |
| `is_target_name()` | Checks whether emitted text exactly equals an exported model target. |
| `target_has_prefix()` | Checks whether emitted text remains a prefix of an exported target. |
| `constrained_next()` | Implements target-constrained decode mode; it is not used by raw greedy decoding. |
| `generate()` | Performs BOS/prompt prefill, autoregressive decoding, stopping, output, and benchmark accounting. |

### Command-line and diagnostics

| Function | Purpose |
| --- | --- |
| `usage()` | Prints command-line help. |
| `parse_u32()` | Parses numeric address/span options. |
| `parse_args()` | Selects attention, broadcast, decode, benchmark, and MMIO options. |
| `probe_bridge()` | Verifies control registers, exact-cycle execution, and MEM0/MEM1 read/write access. |
| `probe_sxm()` | Runs a known-data SXM test and verifies its output. |
| `main()` | Opens MMIO, loads the model unless disabled, and runs the interactive prompt loop. |

## MMIO driver function guide

The low-level driver is `synthesis/linux/src/lpulite_hps_mmio.c`.

| Function | Purpose |
| --- | --- |
| `word_ptr()` | Computes a volatile register pointer inside the mapped bridge. |
| `io_barrier()` | Prevents unsafe compiler/CPU reordering around MMIO. |
| `lpulite_mmio_open()` | Opens `/dev/mem` and maps the lightweight HPS-to-FPGA bridge. |
| `lpulite_mmio_close()` | Unmaps the bridge and closes `/dev/mem`. |
| `lpulite_write32()` / `lpulite_read32()` | Access 32-bit LPU control registers. |
| `lpulite_soft_reset()` | Pulses and polls the LPU soft-reset control. |
| `lpulite_write_row()` / `lpulite_read_row()` | Transfer one packed 96-bit row through three MMIO words. |
| `lpulite_copy_row()` | Copies one row through ARM-visible MMIO. |
| `lpulite_load_imem_page()` | Loads one VLIW page into FPGA IMEM. |
| `lpulite_run_page()` | Starts a loaded page and waits for completion. |
| `lpulite_run_cycles_from()` | Loads a PC, requests an exact FPGA cycle count, polls completion, and returns the hardware cycle delta. |
| `lpulite_run_cycles()` | Runs an exact number of cycles from the current PC. |

## Tokens/second interpretation

There are two useful rates, and they answer different questions.

### Complete next-token forward-pass rate

Across the five captured prompts:

- Total LPU steps: 29.
- Total request time: 1.049 seconds.
- Full request step rate: `29 / 1.049 = 27.65` LPU steps/s.
- Active prefill/decode time: approximately 1.035 seconds.
- Active step rate: approximately `28.02` LPU steps/s.

The stable hardware-plus-runtime result is therefore approximately:

> **28 complete next-token forward passes per second**, or about **35.7 ms per
> forward pass**.

This rate includes ARM MMIO staging, FPGA execution and polling, cache
management, attention orchestration, and decode work. It excludes the one-time
model load and time spent typing.

After prefill, one additional output character normally requires one forward
pass, so steady-state decode approaches 28 generated tokens/s.

### Prompt-to-completion output rate

The sample produced 21 output characters in 1.049 seconds:

```text
21 / 1.049 = 20.02 output tokens/s
```

Therefore the aggregate end-to-end rate for that mixed prompt workload was
approximately **20.0 output tokens/s**.

### Why each prompt prints a different tokens/second value

The printed end-to-end metric is:

```text
number of output characters / total request time
```

Total request time includes BOS and prompt prefill. Short outputs pay that
fixed prefill cost but have fewer output characters in the numerator. Longer
outputs amortize the same cost across more characters. For example, `michael`
reports a higher output-token rate than `satvik` even though both execute each
forward pass at almost the same 28 steps/s.

The variation is therefore mostly a property of the metric and prompt/output
length, not nondeterministic FPGA arithmetic. Smaller variations come from
Linux scheduling, MMIO polling, terminal I/O, and the chosen `--settle-us`
value.

For repeatable comparisons:

- Use the same prompt and `--max-new-tokens` value.
- Keep `--settle-us` unchanged.
- Disable `--verbose` while benchmarking.
- Run several repetitions and divide total generated tokens by total request
  time.
- Report both output tokens/s and complete LPU steps/s.

## Current architectural conclusion

The LPU is performing the model's expensive tensor arithmetic, including full
causal attention QK, softmax, and PV. The ARM is the runtime controller and
data-layout engine. The measured system rate is approximately 28 complete
forward passes/s and about 20 output tokens/s for the captured mixed workload.

This remains a general-purpose statically scheduled accelerator within the
operations and memory sizes supported by its compiler and hardware. A new
compatible quantized model can use the same LPU if its compiler generates the
correct memory layout, VLIW schedule, scales, and supported operation sequence;
it does not require hardcoding name completions into the LPU RTL.

