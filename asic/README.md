# LPULite ASIC showcase

This directory turns the LPULite compute core into a SKY130 physical-design
showcase targeting 100 MHz. It intentionally excludes the FPGA-specific
DE1-SoC wrapper and preserves the architectural 8x8 MXM and 8-lane VXM.

## Memory plan

The normal RTL configuration retains the project's full 32K-row data
memories. The physical showcase deliberately uses a 1K-row configuration that
maps to the available public 8x1024 OpenRAM macro:

| Memory | Showcase organization | SRAM macros | Capacity |
|---|---:|---:|---:|
| MEM0 | 1,024 x 72 | 9 | 9 KiB |
| MEM1 | 1,024 x 72 | 9 | 9 KiB |
| IMEM | 1,024 x 96 | 12 | 12 KiB |

That produces 30 visually distinct SRAM blocks while preserving the requested
8x8 MXM, 8-lane VXM, 64-bit westbound data, and 256-bit eastbound data.
The checked-in macro views come from `VLSIDA/sky130_sram_macros`.

## Local RTL synthesis audit

From the repository root:

```sh
yosys -s asic/yosys_synth.ys
```

This audit stops before technology mapping and confirms that the complete core
hierarchy elaborates with all 30 SRAM macro instances preserved as black boxes.

## OpenLane 2

OpenLane 2 with Nix is recommended on Apple Silicon macOS. Install Nix from a
normal Terminal window so macOS can request administrator authorization:

```sh
curl --proto '=https' --tlsv1.2 -sSf -L \
  https://install.determinate.systems/nix | \
  sh -s -- install --no-confirm \
  --extra-conf "extra-substituters = https://openlane.cachix.org" \
  --extra-conf "extra-trusted-public-keys = openlane.cachix.org-1:qqdwh+QMNGZpAuyeQJTH9ErW57OWSvdtuwfBKdS254E="
```

Then clone OpenLane 2 and enter its environment:

```sh
git clone https://github.com/efabless/openlane2.git ../openlane2
nix-shell ../openlane2/shell.nix
```

Once `openlane` and the SKY130 PDK are available:

```sh
openlane asic/config.json
```

The final GDS is written under the run directory created by OpenLane. The
configuration uses the `sky130_fd_sc_hd` library, a 10 ns clock, a square
relative floorplan, and 45% initial core utilization.

## Render with GDS3D

Clone and build GDS3D, then pass the produced GDS to the launch helper:

```sh
git clone https://github.com/trilomix/GDS3D.git ../GDS3D
asic/render-gds3d.sh ../GDS3D /absolute/path/to/lpulite_asic_top.gds
```

The repository includes a legacy x86_64 macOS executable. Apple Silicon needs
Rosetta 2 to run it. GDS3D's included `techfiles/sky130.txt` supplies the layer
heights, colors, and thicknesses used for the 3D view.

## Reporting

The requested target is 100 MHz. Final reporting should retain the requested
10 ns period and separately report achieved worst slack and the corresponding
post-route frequency estimate. Area should be split into standard-cell area,
hard-SRAM area, and total core/die area.
