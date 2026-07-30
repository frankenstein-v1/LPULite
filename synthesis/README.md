# FPGA synthesis

Everything specific to Quartus, Platform Designer, the DE1-SoC board, and its
host/JTAG interface lives here.

## Layout

- `project/` — Quartus project and generated Platform Designer system
- `rtl/` — DE1-SoC top-level and Avalon wrapper RTL
- `scripts/` — project and C-driver build tools
- `driver/` — native/ARM host driver and generated headers
- `tests/` — physical-board JTAG tests
- `docs/` — FPGA inference workflow notes

## Build

```bash
python synthesis/scripts/build_de1_soc.py
```

Generate embedded model headers and build the host driver:

```bash
python model/tools/export_c_headers.py
python synthesis/scripts/build_c_driver.py
```

Quartus build products are written below `synthesis/build/`.
