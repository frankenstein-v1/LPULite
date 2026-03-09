#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/run_nripple_cocotb.sh

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source .venv_cocotb13/bin/activate

export PYTHONPATH="$PWD/tb:$PWD"
export TOPLEVEL=nripple_wave_tb
export TOPLEVEL_LANG=verilog
export MODULE=nripple_cocotb
export SIM=icarus
export SIM_BUILD=build/cocotb_sim_nripple
export VERILOG_SOURCES="$PWD/src/nripple.sv $PWD/tb/nripple_wave_tb.sv"

# Waveform dump is handled by tb/nripple_wave_tb.sv via $dumpfile/$dumpvars.
export COMPILE_ARGS='-g2012'

rm -rf "$SIM_BUILD"
mkdir -p "$SIM_BUILD"

make -C "$SIM_BUILD" -f "$(cocotb-config --makefiles)/Makefile.sim" sim

VCD_PATH="$SIM_BUILD/nripple_cocotb.vcd"
if [[ -f "$VCD_PATH" ]]; then
  echo "Generated waveform: $VCD_PATH"
else
  echo "Waveform not found at $VCD_PATH (if you still see flat/no waveforms, ensure a dump is forced in source tb)"
fi
