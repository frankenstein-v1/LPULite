# LPULite Native C Hardware Driver Makefile
# Supports both native x86 testing and ARM GCC cross-compilation for DE1-SoC

CC ?= gcc
CFLAGS ?= -O3 -Wall -Wextra -Isynthesis/driver
IVERILOG ?= iverilog
VVP ?= vvp
VERILATOR ?= verilator
YOSYS ?= yosys

SRC_DIR ?= src
TB_DIR ?= tb
BUILD_DIR ?= build
TOP ?= lpu_top
SIM_TOP ?= $(TOP)_tb
SOURCES = $(wildcard $(SRC_DIR)/*.sv) $(wildcard $(SRC_DIR)/*.v)
TESTBENCH = $(TB_DIR)/$(TOP)_tb.sv

TARGET = synthesis/driver/lpu_driver
SRC = synthesis/driver/lpu_driver.c

all: headers $(TARGET)

headers:
	python model/tools/export_c_headers.py

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET)

VFLAGS ?= -g2012 -Wall -Wimplicit
VLTFLAGS ?= --lint-only --language 1800-2017 -Wall

.PHONY: help check-tools sim wave lint synth asic-audit asic-gds clean

help:
	@printf "Targets:\n"
	@printf "  make sim         - compile + run testbench with iverilog\n"
	@printf "  make wave        - run simulation and dump waveform (gtkwave-ready)\n"
	@printf "  make lint        - run Verilator lint\n"
	@printf "  make synth       - run Yosys synthesis check\n"
	@printf "  make asic-audit  - elaborate the ASIC core and preserve inferred SRAMs\n"
	@printf "  make asic-gds    - run the SKY130 OpenLane 2 physical-design flow\n"
	@printf "  make check-tools - verify required binaries exist\n"
	@printf "  make clean       - remove generated artifacts\n"

check-tools:
	@command -v $(IVERILOG) >/dev/null 2>&1 || (echo "Missing: $(IVERILOG)" && exit 1)
	@command -v $(VVP) >/dev/null 2>&1 || (echo "Missing: $(VVP)" && exit 1)
	@command -v $(VERILATOR) >/dev/null 2>&1 || (echo "Missing: $(VERILATOR)" && exit 1)
	@command -v $(YOSYS) >/dev/null 2>&1 || (echo "Missing: $(YOSYS)" && exit 1)
	@echo "All required toolchain commands found."

sim: check-tools
	@mkdir -p $(BUILD_DIR)
	$(IVERILOG) $(VFLAGS) -s $(SIM_TOP) -o $(BUILD_DIR)/$(TOP).out $(SOURCES) $(TESTBENCH)
	$(VVP) $(BUILD_DIR)/$(TOP).out

wave: check-tools
	@mkdir -p $(BUILD_DIR)
	$(IVERILOG) $(VFLAGS) -DWAVEFORM -s $(SIM_TOP) -o $(BUILD_DIR)/$(TOP).out $(SOURCES) $(TESTBENCH)
	$(VVP) $(BUILD_DIR)/$(TOP).out

lint: check-tools
	$(VERILATOR) $(VLTFLAGS) --top-module $(TOP) $(SOURCES) $(TESTBENCH)

synth: check-tools
	@mkdir -p $(BUILD_DIR)
	$(YOSYS) -p 'read_verilog -sv $(SOURCES); synth -top $(TOP); stat; write_json $(BUILD_DIR)/$(TOP).json;'

asic-audit:
	$(YOSYS) -s asic/yosys_synth.ys

asic-gds:
	@command -v openlane >/dev/null 2>&1 || (echo "Missing: openlane (see asic/README.md)" && exit 1)
	openlane asic/config.json

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(TARGET) synthesis/driver/lpu_jtag_runner synthesis/driver/include/lpu_vliw.h synthesis/driver/include/lpu_weights.h

.PHONY: all clean headers
