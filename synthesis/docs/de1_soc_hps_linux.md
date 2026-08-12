# DE1-SoC HPS/Linux TinyLPU path

This is the faster path than USB-JTAG for interactive inference:

```text
ARM Linux process
  -> /dev/mem mmap
  -> lightweight HPS-to-FPGA bridge
  -> TinyLPU Avalon wrapper
  -> IMEM / MEM0 / MEM1 / control regs
```

The LPU RTL does not need to become a CPU peripheral internally. It already has
an Avalon-style wrapper in `synthesis/rtl/lpu_de1_soc_wrapper.sv`. The hardware
integration job is to let the HPS lightweight bridge drive that wrapper instead
of the JTAG-to-Avalon bridge.

## Repo storage layout

- HPS/Linux runtime source:
  `synthesis/linux/`
- Generated HPS C model image:
  `synthesis/linux/include/microgpt_hps_image.h`
- Image exporter:
  `synthesis/scripts/export_microgpt_hps_headers.py`
- Source model/schedule artifacts:
  `model/artifacts/fpga_microgpt/`

This keeps Linux runtime code under synthesis/platform bring-up, while the model
compiler output stays in the model artifact area.

## Current status

Done:

1. Added a `/dev/mem` MMIO driver for ARM Linux.
2. Added an interactive MicroGPT HPS runtime.
3. Added a C header exporter for the current MEM1/VLIW images.
4. Generated `synthesis/linux/include/microgpt_hps_image.h` from the current
   schedule.

Still required before the ARM runtime can hit real hardware:

1. Build an HPS-enabled FPGA image where the HPS lightweight bridge controls the
   LPU wrapper.
2. Program that image, or convert it to an `.rbf` and load it from Linux.
3. Enable the Linux FPGA bridges if your image/kernel leaves them disabled.

## Platform Designer wiring target

Keep the existing LPU wrapper address contract:

| LPU wrapper offset | Meaning |
| --- | --- |
| `0x0000` | IMEM |
| `0x4000` | MEM0 |
| `0x8000` | MEM1 |
| `0xC000` | run control |
| `0xC004` | PC load |
| `0xC008` | cycle counter |

In Platform Designer, connect the HPS lightweight AXI/Avalon master to the LPU
wrapper slave. The cleanest address map is to place the LPU slave at offset
`0x0000` in the lightweight bridge window. Then Linux uses:

```sh
sudo ./microgpt_hps_runtime --base 0xff200000
```

If you give the LPU wrapper a non-zero slave offset, add that offset to the
base passed to the runtime.

## Running on the board

On the ARM Linux shell:

```sh
cd tinyLPU/synthesis/linux
make
sudo ./microgpt_hps_runtime --attention host
```

Then type a prompt like:

```text
Prompt > sat
```

For a quick bring-up mode that does not compute attention on the ARM:

```sh
sudo ./microgpt_hps_runtime --attention current
```

For repeat runs after MEM1 is already loaded:

```sh
sudo ./microgpt_hps_runtime --no-load-weights --attention host
```

## Bridge enable notes

Some Linux images expose FPGA bridge controls under `/sys/class/fpga-bridge`.
If reads/writes hang or return bus errors, check bridge state:

```sh
ls /sys/class/fpga-bridge
```

If your image has named bridge entries, enable the lightweight bridge before
running the runtime. Exact names vary by kernel/device tree, but the intent is:

```sh
echo 1 | sudo tee /sys/class/fpga-bridge/lwhps2fpga/enable
```

If your image does not expose those sysfs bridge controls, bridge enablement is
usually handled by the bootloader/device tree.

## Bitstream loading options

During bring-up it is fine to configure the FPGA over Quartus/JTAG and use the
ARM only for inference traffic. That already removes JTAG from the slow runtime
data path.

Later, convert the `.sof` to an `.rbf` and load it from Linux using the FPGA
manager flow supported by your SD-card image.

## Accuracy caveat

The HPS runtime mirrors the current host-side testbench split:

- FPGA computes the scheduled LPU stages.
- FPGA MEM0 stores the K/V cache rows.
- In `--attention host` mode, ARM Linux computes the small causal attention
  context from those cached FPGA rows and stages the result.

That is much faster than laptop JTAG and keeps the model data path on the board,
but it is not yet the final “all attention math in RTL/VLIW” version. To make
attention 100% FPGA-side, the remaining work is a VLIW attention stage that
iterates over cached K/V rows and writes the context rows to MEM0.

