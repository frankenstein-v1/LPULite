# TinyLPU

TinyLPU is an 8-lane INT8 language-processing accelerator designed from
scratch and demonstrated on the Intel/Altera DE1-SoC FPGA. The completed
system runs a compiled character-level MicroGPT forward pass through the LPU,
including RMSNorm, transformer matrix multiplications, causal attention,
softmax, residual connections, the MLP, and the language-model head.

Read the illustrated project article at **[LPU Lite](https://www.lpulite.com/)**.

Created by **Michael Trbovic**, **Saksham Batra**, and **Arjun Harinath**.

## Project status

The FPGA demonstration is complete and working:

- Target board: Terasic DE1-SoC with a Cyclone V `5CSEMA5F31C6` FPGA.
- Model: block-scaled INT8 MicroGPT with 1 transformer layer, a 16-element
  embedding, 4 attention heads, a 16-token context, and a 27-token character
  vocabulary.
- Compiled image: 4,920 96-bit VLIW instructions and 974 MEM1 rows.
- Production attention path: FPGA SXM K transpose, FPGA MXM QK/PV, and FPGA
  VXM LUT softmax.
- Verified aggregate performance: approximately 41.3 complete LPU forward
  passes per second.
- Demo performance for `sat` to `satvik`: approximately 20.7 generated
  characters per second and 145 ms prompt-to-completion latency.

The committed release bitstream is:

```text
synthesis/build/tiny_lpu_de1_soc_hps/tiny_lpu_de1_soc_hps.sof
SHA-256: 12AF29D5F557E933FC112FF874403131F23D14AD048A284ACFFA284869333A4A
```

The demo checkpoint is trained for character-level name completion. Its
predictions reflect its training data and are not hardcoded into the RTL or
runtime.

## Architecture

| Component | Role in the completed MicroGPT data path |
| --- | --- |
| ICU | Executes the compiler-generated static 96-bit VLIW schedule and issues memory, bus, MXM, SXM, and VXM controls in parallel. |
| MXM | Uses an 8x8 INT8 MAC array for the model's learned linear layers, `QK^T` attention scores, and `PV` weighted sums. |
| VXM | Executes chunked RMSNorm, LUT-based chunked softmax, ReLU, residual operations, quantization, and attention-head merging. |
| SXM | Broadcasts scalar lanes when requested and transposes aligned 8x8 K tiles for `QK^T`. |
| MEM0/MEM1 | Store activations, quantized weights, constants, scratch rows, and the K/V cache. |
| HPS bridge wrapper | Exposes LPU control, memories, IMEM paging, cycle counters, and exact-cycle execution to the ARM through lightweight Avalon-MM. |

The LPU is statically scheduled. It does not contain a dynamic instruction
scheduler or model-specific name lookup logic. Supporting another compatible
model requires a new quantized memory layout and compiler-generated schedule,
not a hardcoded output path.

## Compilation and inference flow

```text
INT8 checkpoint
      |
      v
Offline Python compiler -- VLIW schedule + packed MEM1 image
      |
      v
ARM Cortex-A9 runtime -- lightweight HPS-to-FPGA MMIO
      |
      v
ICU + MEM0/MEM1 + SXM + MXM + VXM on the FPGA
      |
      v
Logits -- ARM greedy argmax -- terminal output
```

The work is divided as follows:

| FPGA LPU | ARM HPS runtime | Offline Python compiler |
| --- | --- | --- |
| Embeddings and transformer arithmetic | Terminal input and character tokenization | Reads and validates the INT8 checkpoint |
| Q, K, V, output projection, FC1, FC2, and LM-head matmuls | Loads model rows and microcode through MMIO | Packs weights, constants, and memory rows |
| K transpose, QK dot products, causal softmax, and PV weighted sums | Starts and polls deterministic microcode sections | Generates the static VLIW schedule and attention sections |
| RMSNorm, ReLU, residuals, head merge, and requantization | Manages K/V cache positions, causal length, layouts, and block exponents | Exports the generated images into the ARM runtime header |
| Produces the final logits | Reads logits, performs greedy argmax/stopping, and prints text | Does not run during inference |

In the production `--attention fpga-mxm` mode, the ARM does not calculate the
QK dot products, softmax exponentials/reciprocals, or PV weighted sums. It is
the control processor for the FPGA accelerator.

## FPGA result

Quartus Prime 25.1 Standard Lite successfully fitted the completed HPS design:

| Resource | Utilization |
| --- | ---: |
| Logic | 32,070 / 32,070 ALMs (100%) |
| Registers | 10,031 |
| Block-memory bits | 2,462,976 / 4,065,280 (61%) |
| RAM blocks | 314 / 397 (79%) |
| DSP blocks | 87 / 87 (100%) |

Static timing analysis reports positive setup and hold slack for both the LPU
and HPS bridge clocks.

## Run the FPGA demonstration

The canonical, step-by-step procedure is in the
**[MicroGPT FPGA runbook](synthesis/docs/microgpt_fpga_runbook.md)**. It covers
the Linux SD image, UART, Ethernet, JTAG chain position, FPGA programming,
runtime transfer, bridge probing, inference, and benchmarking.

After the board is booted, networked, programmed, and the runtime has been
copied to `/home/root/linux`, run:

```sh
cd /home/root/linux
./microgpt_hps_runtime \
  --attention fpga-mxm \
  --broadcast host \
  --decode greedy \
  --benchmark
```

Example:

```text
Prompt > sat
satvik
```

Output characters per second varies with prompt and completion length. The
aggregate LPU forward-pass rate is the more stable hardware/software system
measurement.

## Build the release

From Windows PowerShell with Quartus 25.1 and WSL ARM cross-compilation tools
installed:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\synthesis\scripts\build_microgpt_fpga_release.ps1
```

The script:

1. Compiles the INT8 model into memory and VLIW artifacts.
2. Exports the generated ARM runtime header.
3. Cross-compiles the static ARM Cortex-A9 runtime in WSL.
4. Compiles the existing HPS Quartus project.
5. Reports the resulting `.sof` SHA-256.

For a software-only rebuild without waiting for Quartus:

```powershell
.\synthesis\scripts\build_microgpt_fpga_release.ps1 -SkipQuartus
```

## Verify in simulation

Run the HPS/LPU wrapper integration test before a Quartus compile:

```sh
python synthesis/tests/run_microgpt_wrapper_cocotb.py
```

This drives the same Avalon-MM interface used by the ARM runtime, loads the
compiled model and microcode, executes the forward pass, and checks the next
token. Additional repository checks include:

```sh
make sim
make lint
python -m pytest model/tests/test_tokenizer.py model/tests/test_lm_shift.py
```

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/` | Synthesizable TinyLPU SystemVerilog: ICU, memories, buses, MXM, SXM, and VXM. |
| `tb/` | Unit and core RTL testbenches. |
| `model/` | MicroGPT training, datasets, checkpoints, INT8 inference, compiler, and generated model artifacts. |
| `synthesis/` | DE1-SoC wrapper RTL, Quartus project, Linux runtime, drivers, FPGA tests, scripts, and documentation. |
| `synthesis/build/tiny_lpu_de1_soc_hps/` | Completed HPS Quartus project and verified release bitstream. |
| `asic/` | SKY130/OpenLane synthesis and physical-design experiments. |
| `archive/` | Historical implementations retained for reference. |
| `misc/` | Supporting utilities and project notes. |

## Further documentation

- [FPGA setup and demo runbook](synthesis/docs/microgpt_fpga_runbook.md)
- [Attention, runtime, and performance audit](synthesis/docs/lpu_attention_runtime_and_performance_audit.md)
- [LPU debugging and generality status](synthesis/docs/lpu_debugging_and_generality_status.md)
- [HPS Linux integration](synthesis/docs/de1_soc_hps_linux.md)
- [Model and compiler workflow](model/README.md)
- [FPGA synthesis layout](synthesis/README.md)
- [ASIC experiments](asic/README.md)

## License

TinyLPU is available under the [Apache License 2.0](LICENSE).
