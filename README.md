# TinyLPU Verilog Workspace

This project is set up as a minimal Language Processing Unit (LPU) Verilog workspace with a local toolchain and helper scripts.

## Included

- `Makefile` targets for:
  - `make sim` – compile + run the default testbench with iverilog
  - `make wave` – compile and run simulation with VCD output
  - `make lint` – run Verilator linting
  - `make synth` – run Yosys synthesis sanity check
  - `make clean` – remove build artifacts
- `scripts/setup-verilog-toolchain.sh` – install required tools
- `src/lpu_top.sv` – minimal LPU datapath module
- `tb/lpu_top_tb.sv` – starter testbench
- `src/half_adder.sv` – sample 1-bit half-adder module
- `tb/half_adder_tb.sv` – half-adder testbench with optional waveform dump

## Prerequisites

- `iverilog`
- `verilator`
- `yosys`
- `gtkwave` (optional for waveform viewing)

## Toolchain install

```bash
./scripts/setup-verilog-toolchain.sh
```

### macOS (Homebrew)

```bash
brew install icarus-verilog verilator yosys gtkwave
```

If `gtkwave` on macOS is blocked or shows the "not compatible with macOS 14+" message, use a local build:

```bash
# build gtkwave from source into ~/opt/gtkwave
git clone --depth 1 https://github.com/gtkwave/gtkwave.git /tmp/gtkwave-src
cd /tmp/gtkwave-src
brew install meson ninja gtk+3 gtk-mac-integration gobject-introspection shared-mime-info desktop-file-utils gtk4 json-glib
meson setup build --prefix=$HOME/opt/gtkwave
meson compile -C build
meson install -C build
```

### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install -y iverilog verilator yosys gtkwave
```

### Fedora/Rocky/Alma (dnf)

```bash
sudo dnf install -y iverilog verilator yosys gtkwave
```

## Quick start

```bash
# Build and run the default testbench
make sim

# Generate a waveform for half_adder
make wave TOP=half_adder
# open waveform (fixes missing Switch.pm if installed in ~/perl5 and auto-loads signals)
./scripts/open-wave.sh build/half_adder.vcd

# or manually
open build/half_adder.vcd  # macOS
~/opt/gtkwave/bin/gtkwave build/half_adder.vcd

# If macOS blocks GTKWave as unverified software
xattr -dr com.apple.quarantine /Applications/gtkwave.app

# Run Verilator lint
make lint TOP=half_adder

# Run a synthesis check
make synth TOP=half_adder
```
