#!/usr/bin/env python3
"""Create and compile the LPULite DE1-SoC JTAG-programmable design.

Run from any directory with:
    py synthesis/scripts/build_de1_soc.py

Use --generate-only to stop before Quartus compilation.  The script deliberately
does not run on import and never removes existing Quartus output directories.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SYNTHESIS_DIR = ROOT / "synthesis"
SRC = ROOT / "src"
SYNTHESIS_RTL = SYNTHESIS_DIR / "rtl"
PROJECT_FILES = SYNTHESIS_DIR / "project"
PROJECT = "lpu_lite_de1_soc"
# Quartus/Qsys use the orderable device name without the trailing temperature
# suffix; the DE1-SoC package is commonly labelled 5CSEMA5F31C6N.
DEVICE = "5CSEMA5F31C6"
PROJECT_DIR = SYNTHESIS_DIR / "build" / PROJECT


def write(path: Path, contents: str) -> None:
    """Write only when the generated contents differ."""
    contents = contents.replace("\r\n", "\n")
    if not path.exists() or path.read_text(encoding="utf-8").replace("\r\n", "\n") != contents:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8", newline="\n")
        print(f"wrote {path.relative_to(ROOT)}")


def replace_once(text: str, pattern: str, replacement: str, label: str) -> str:
    result, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Could not apply the expected {label} edit; baseline RTL differs.")
    return result


def patch_rtl() -> None:
    """Apply the narrow RTL additions required by the host interface.

    Each edit is skipped when its distinctive external port already exists, which
    makes a second invocation safe and preserves user changes outside these areas.
    """
    mem = SRC / "mem.sv"
    text = mem.read_text(encoding="utf-8")
    if "ext_write_en" not in text and "ext_imem_en" not in text:
        text = replace_once(text,
            r"(input\s+logic\s+\[ADDR_W-1:0\]\s+addr\s*)(,?\s*\n\s*\);)",
            r"\1,\n\n    // External Host/JTAG interface (bypass ports)\n    input  logic              ext_write_en,\n    input  logic              ext_read_en,\n    input  logic [ADDR_W-1:0] ext_addr,\n    input  logic [DATA_W-1:0] ext_data_in,\n    output logic [DATA_W-1:0] ext_data_out\2", "mem ports")
        text = replace_once(text, r"(stream_out\s*<=\s*'0;)", r"\1\n            ext_data_out <= '0;", "mem reset")
        text = replace_once(text, r"(if\s*\(write_en\)\s*begin\s*\n\s*sram_array\[addr\]\s*<=\s*stream_in;\s*\n\s*end)",
            r"\1 else if (ext_write_en) begin\n                sram_array[ext_addr] <= ext_data_in;\n            end", "mem write bypass")
        text = replace_once(text, r"(end\s*\n\s*end\s*\n\s*end\s*\n\s*endmodule)",
            "end\n\n            if (ext_read_en) ext_data_out <= sram_array[ext_addr];\n            else             ext_data_out <= '0;\n        end\n    end\nendmodule", "mem read bypass")
        write(mem, text)
        text = mem.read_text(encoding="utf-8")
    # The wrapper holds the LPU reset while programming.  ICU's programming
    # port is independently clocked already; retain the same capability for
    # the SRAMs so that MEM0/MEM1 writes are not discarded during that reset.
    if "External writes remain available while reset is asserted." not in text and "ext_write_en" not in text:
        text = replace_once(text,
            r"(ext_data_out\s*<=\s*'0;)",
            r"\1\n            // External writes remain available while reset is asserted.\n            if (ext_write_en) sram_array[ext_addr] <= ext_data_in;", "mem reset programming")
        write(mem, text)

    icu = SRC / "icu.sv"
    text = icu.read_text(encoding="utf-8")
    if "ext_write_en" not in text and "ext_imem_en" not in text:
        text = replace_once(text, r"(output\s+logic\s+\[2:0\]\s+vxm_residual_op)(\s*\n\s*\);)",
            r"\1,\n\n    // External JTAG write interface for programming\n    input  logic                      ext_write_en,\n    input  logic [$clog2(INSTRUCTION_COUNT)-1:0] ext_addr,\n    input  logic [95:0]               ext_data_in\2", "ICU ports")
        text = replace_once(text, r"(logic\s+\[95:0\]\s+imem_array\s+\[0:INSTRUCTION_COUNT-1\];)",
            r"\1\n\n    always_ff @(posedge clk) begin\n        if (ext_write_en) imem_array[ext_addr] <= ext_data_in;\n    end", "ICU programming write")
        write(icu, text)

    lpu = SRC / "lpu.sv"
    text = lpu.read_text(encoding="utf-8")
    if "ext_imem_write_en" not in text and "ext_en" not in text:
        text = replace_once(text, r"(input\s+logic\s+rst_n)(\s*\n\s*\);)",
            r"\1,\n\n    input logic ext_imem_write_en,\n    input logic [9:0] ext_imem_addr,\n    input logic [95:0] ext_imem_data_in,\n    input logic ext_mem0_write_en, ext_mem0_read_en,\n    input logic [MEM_ADDR_W-1:0] ext_mem0_addr,\n    input logic [71:0] ext_mem0_data_in,\n    output logic [71:0] ext_mem0_data_out,\n    input logic ext_mem1_write_en, ext_mem1_read_en,\n    input logic [MEM_ADDR_W-1:0] ext_mem1_addr,\n    input logic [71:0] ext_mem1_data_in,\n    output logic [71:0] ext_mem1_data_out\2", "LPU ports")
        text = replace_once(text, r"(\.vxm_residual_op\(vxm_residual_op\))(\s*\n\s*\);)",
            r"\1,\n    .ext_write_en(ext_imem_write_en),\n    .ext_addr(ext_imem_addr),\n    .ext_data_in(ext_imem_data_in)\2", "ICU hookup")
        for name in ("0", "1"):
            text = replace_once(text, rf"(\.addr\(mem{name}_addr\))(\s*\n\s*\);)",
                rf"\1,\n    .ext_write_en(ext_mem{name}_write_en),\n    .ext_read_en(ext_mem{name}_read_en),\n    .ext_addr(ext_mem{name}_addr),\n    .ext_data_in(ext_mem{name}_data_in),\n    .ext_data_out(ext_mem{name}_data_out)\2", f"MEM{name} hookup")
        write(lpu, text)

    # Quartus enforces enum typing at module ports. The ICU exposes its bus
    # selects as packed logic vectors, so make typed views and cast at the
    # boundary before passing them to the bus/decode modules.
    text = lpu.read_text(encoding="utf-8")
    old_bus_views = """// Encoded bus select views. Keep these as plain vectors for Icarus compatibility.
logic [2:0] westbound_consumer_sel_t;
logic [2:0] eastbound_consumer_sel_t;

logic [2:0] westbound_sel_t;
logic [2:0] eastbound_sel_t;"""
    new_bus_views = """// Enum-typed views required at the strongly typed bus module boundaries.
westbound_consumer_e westbound_consumer_sel_t;
eastbound_consumer_e eastbound_consumer_sel_t;

westbound_producer_e westbound_sel_t;
eastbound_producer_e eastbound_sel_t;"""
    if old_bus_views in text:
        text = text.replace(old_bus_views, new_bus_views)
        text = text.replace("assign westbound_consumer_sel_t = westbound_consumer_sel;", "assign westbound_consumer_sel_t = westbound_consumer_e'(westbound_consumer_sel);")
        text = text.replace("assign eastbound_consumer_sel_t = eastbound_consumer_sel;", "assign eastbound_consumer_sel_t = eastbound_consumer_e'(eastbound_consumer_sel);")
        text = text.replace("assign westbound_sel_t = westbound_sel;", "assign westbound_sel_t = westbound_producer_e'(westbound_sel);")
        text = text.replace("assign eastbound_sel_t = eastbound_sel;", "assign eastbound_sel_t = eastbound_producer_e'(eastbound_sel);")
        write(lpu, text)

    # Quartus 25.1 Lite accepts SystemVerilog generate loops, but not an
    # inline ``for (genvar i = ...)`` declaration.  Keep the baseline RTL
    # portable by declaring each genvar before its generate loop.
    generate_fixes = {
        "lpu.sv": [("for (genvar i = 0; i < MXM_SIZE; i++) begin : g_mxm_feed", "genvar i;\n    for (i = 0; i < MXM_SIZE; i++) begin : g_mxm_feed")],
        "residual_add.sv": [("for (genvar lane = 0; lane < LANES; lane++) begin : g_residual_lanes", "genvar lane;\n        for (lane = 0; lane < LANES; lane++) begin : g_residual_lanes")],
        "rmsnorm.sv": [
            ("for (genvar i = 0; i < LANES; i++) begin : g_rms_lanes", "genvar i;\n        genvar j;\n        genvar m;\n        for (i = 0; i < LANES; i++) begin : g_rms_lanes"),
            ("for (genvar j = 0; j < 4; j++) begin : g_sum1", "for (j = 0; j < 4; j++) begin : g_sum1"),
            ("for (genvar m = 0; m < 2; m++) begin : g_sum2", "for (m = 0; m < 2; m++) begin : g_sum2"),
        ],
        "softmax.sv": [("for (genvar lane = 0; lane < LANES; lane++) begin : gen_exp", "genvar lane;\n        for (lane = 0; lane < LANES; lane++) begin : gen_exp")],
        "vxm.sv": [("for (genvar i = 0; i < LANES; i++) begin : g_vxm_lanes", "genvar i;\n        for (i = 0; i < LANES; i++) begin : g_vxm_lanes")],
        "vxm_rope.sv": [("for (genvar pair = 0; pair < PAIRS; pair++) begin : g_rope_pairs", "genvar pair;\n        for (pair = 0; pair < PAIRS; pair++) begin : g_rope_pairs")],
        "mem_row_dequant.sv": [("for (genvar lane = 0; lane < MXM_SIZE; lane++) begin : g_dequant_lane", "genvar lane;\n        for (lane = 0; lane < MXM_SIZE; lane++) begin : g_dequant_lane")],
        "eastbound_bus/mxm_eastbound_adapter.sv": [
            ("for (genvar row = 0; row < MXM_SIZE; row++) begin : g_row\n            for (genvar col = 0; col < MXM_SIZE; col++) begin : g_col", "genvar row;\n        genvar col;\n        for (row = 0; row < MXM_SIZE; row++) begin : g_row\n            for (col = 0; col < MXM_SIZE; col++) begin : g_col"),
        ],
    }
    for relative_path, fixes in generate_fixes.items():
        path = SRC / relative_path
        text = path.read_text(encoding="utf-8")
        changed = False
        for old, new in fixes:
            if old in text:
                text = text.replace(old, new)
                changed = True
        if changed:
            write(path, text)

    # An asynchronous-reset event control must test only that reset signal in
    # its first branch. Quartus rejects ``if (rst || mxm_clear)`` under
    # ``@(posedge clk or posedge rst)``; mxm_clear is synchronous.
    mxm = SRC / "mxm.sv"
    text = mxm.read_text(encoding="utf-8")
    old_reset = """    //if reset or mxm_clear, we know the registers hjave nothing in them/ we reset them to 0
    if (rst || mxm_clear) begin 
        mxm_input_ingress_loaded <= 1'b0;
        mxm_wght_ingress_loaded <= 1'b0;


        //
        for (int idx = 0; idx < mxm_size; idx++) begin 
            mxm_input_ingress_reg[idx] <= '0;
            mxm_wght_ingress_reg[idx] <= '0;
        end 
    end """
    new_reset = """    // rst is asynchronous; mxm_clear is a synchronous command.
    if (rst) begin 
        mxm_input_ingress_loaded <= 1'b0;
        mxm_wght_ingress_loaded <= 1'b0;
        for (int idx = 0; idx < mxm_size; idx++) begin
            mxm_input_ingress_reg[idx] <= '0;
            mxm_wght_ingress_reg[idx] <= '0;
        end
    end else if (mxm_clear) begin
        mxm_input_ingress_loaded <= 1'b0;
        mxm_wght_ingress_loaded <= 1'b0;
        for (int idx = 0; idx < mxm_size; idx++) begin
            mxm_input_ingress_reg[idx] <= '0;
            mxm_wght_ingress_reg[idx] <= '0;
        end
    end """
    if old_reset in text:
        write(mxm, text.replace(old_reset, new_reset))


TOP = """module de1_soc_top (
    input  logic        CLOCK_50,  // PIN_AF14 (50MHz clock input)
    input  logic [0:0]  KEY,       // PIN_AJ4 (Active-low pushbutton as Reset)
    output logic [0:0]  LEDR       // PIN_V16 (Diagnostic LED)
);
    logic [31:0] jtag_address, jtag_writedata, jtag_readdata;
    logic [3:0]  jtag_byteenable;
    logic        jtag_read, jtag_write, jtag_waitrequest, jtag_readdatavalid;

    platform_designer_system u_qsys (
        .clk_clk       (CLOCK_50),
        .reset_reset_n (KEY[0]),
        .lpu_avalon_address       (jtag_address),
        .lpu_avalon_read          (jtag_read),
        .lpu_avalon_write         (jtag_write),
        .lpu_avalon_writedata     (jtag_writedata),
        .lpu_avalon_byteenable    (jtag_byteenable),
        .lpu_avalon_readdata      (jtag_readdata),
        .lpu_avalon_waitrequest   (jtag_waitrequest),
        .lpu_avalon_readdatavalid (jtag_readdatavalid)
    );
    lpu_de1_soc_wrapper u_lpu_avalon (
        .clk             (CLOCK_50),
        .rst_n           (KEY[0]),
        .avs_address     (jtag_address[15:0]),
        .avs_read        (jtag_read),
        .avs_write       (jtag_write),
        .avs_writedata   (jtag_writedata),
        .avs_readdata      (jtag_readdata),
        .avs_waitrequest   (jtag_waitrequest),
        .avs_readdatavalid (jtag_readdatavalid)
    );
    assign LEDR[0] = KEY[0];
endmodule
"""


WRAPPER = r'''`timescale 1ns/1ps
module lpu_de1_soc_wrapper (
    input logic clk, input logic rst_n,
    input logic [15:0] avs_address, input logic avs_read, input logic avs_write,
    input logic [31:0] avs_writedata,
    output logic [31:0] avs_readdata, output logic avs_waitrequest,
    output logic avs_readdatavalid
);
    localparam logic [1:0] MEM0 = 2'd0, MEM1 = 2'd1, IMEM = 2'd2;
    localparam logic [15:0] CTRL_RUN = 16'hc000;
    localparam logic [15:0] CTRL_PC_LOAD = 16'hc004;
    localparam logic [15:0] CTRL_CYCLES = 16'hc008;
    logic run_enable, pc_load_en, ext_en, ext_write;
    logic [31:0] pc_load_value, cycle_counter;
    logic [1:0] ext_target;
    logic [31:0] ext_addr;
    logic [95:0] ext_wdata, ext_rdata, assembly;
    integer word_index, row_index, lane_index;

    assign avs_waitrequest = 1'b0;
    assign avs_readdatavalid = avs_read;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            run_enable <= 1'b0; pc_load_en <= 1'b0; pc_load_value <= '0; ext_en <= 1'b0; ext_write <= 1'b0;
            ext_target <= '0; ext_addr <= '0; ext_wdata <= '0; assembly <= '0;
            avs_readdata <= '0;
        end else begin
            ext_en <= 1'b0;
            pc_load_en <= 1'b0;
            if (avs_write) begin
                if (avs_address == CTRL_RUN) run_enable <= avs_writedata[0];
                else if (avs_address == CTRL_PC_LOAD) begin
                    pc_load_value <= avs_writedata;
                    pc_load_en <= 1'b1;
                end
                else begin
                    if (avs_address < 16'h4000) begin word_index = avs_address[13:2]; ext_target <= IMEM; end
                    else if (avs_address < 16'h8000) begin word_index = (avs_address - 16'h4000) >> 2; ext_target <= MEM0; end
                    else begin word_index = (avs_address - 16'h8000) >> 2; ext_target <= MEM1; end
                    row_index = word_index / 3; lane_index = word_index % 3;
                    case (lane_index)
                        0: assembly[31:0] <= avs_writedata;
                        1: assembly[63:32] <= avs_writedata;
                        2: begin
                            ext_en <= 1'b1; ext_write <= 1'b1; ext_addr <= row_index;
                            ext_wdata <= {avs_writedata, assembly[63:0]};
                        end
                    endcase
                end
            end
            if (avs_read) begin
                if (avs_address == CTRL_RUN) avs_readdata <= {31'b0, run_enable};
                else if (avs_address == CTRL_CYCLES) avs_readdata <= cycle_counter;
                else begin
                    if (avs_address < 16'h4000) begin word_index = avs_address[13:2]; ext_target <= IMEM; end
                    else if (avs_address < 16'h8000) begin word_index = (avs_address - 16'h4000) >> 2; ext_target <= MEM0; end
                    else begin word_index = (avs_address - 16'h8000) >> 2; ext_target <= MEM1; end
                    row_index = word_index / 3; lane_index = word_index % 3;
                    ext_en <= 1'b1; ext_write <= 1'b0; ext_addr <= row_index;
                    case (lane_index) 0: avs_readdata <= ext_rdata[31:0];
                      1: avs_readdata <= ext_rdata[63:32];
                      default: avs_readdata <= ext_rdata[95:64]; endcase
                end
            end
        end
    end
    lpu #(
        .RMSNORM_CHUNKS(2),
        .SOFTMAX_CHUNKS(16)
    ) u_lpu (
        .clk(clk), .rst_n(rst_n), .run_en(run_enable),
        .pc_load_en(pc_load_en), .pc_load_value(pc_load_value),
        .ext_en(ext_en), .ext_write(ext_write), .ext_target(ext_target),
        .ext_addr(ext_addr), .ext_wdata(ext_wdata), .ext_rdata(ext_rdata), .cycle_counter(cycle_counter)
    );
endmodule
'''


HW_TCL = r'''set_module_property NAME lpu_de1_soc
set_module_property VERSION 1.0
set_module_property GROUP "LPULite"
set_module_property DISPLAY_NAME "LPULite DE1-SoC Avalon wrapper"
set_module_property TOP_LEVEL_HDL_MODULE lpu_de1_soc_wrapper
set_module_property INSTANTIATE_IN_SYSTEM_MODULE true
add_interface clk clock end
add_interface_port clk clk clk Input 1
add_interface rst_n reset end
set_interface_property rst_n associatedClock clk
set_interface_property rst_n synchronousEdges DEASSERT
add_interface_port rst_n rst_n reset_n Input 1
add_interface avs avalon end
set_interface_property avs addressUnits SYMBOLS
set_interface_property avs associatedClock clk
set_interface_property avs associatedReset rst_n
set_interface_property avs readWaitTime 0
set_interface_property avs writeWaitTime 0
set_interface_property avs maximumPendingReadTransactions 0
add_interface_port avs avs_address address Input 16
add_interface_port avs avs_read read Input 1
add_interface_port avs avs_write write Input 1
add_interface_port avs avs_writedata writedata Input 32
add_interface_port avs avs_readdata readdata Output 32
add_interface_port avs avs_waitrequest waitrequest Output 1
'''


SYSTEM_TCL = r'''package require -exact qsys @QSYS_API_VERSION@
create_system platform_designer_system
set_project_property DEVICE_FAMILY "Cyclone V"
set_project_property DEVICE 5CSEMA5F31C6
add_instance clk_0 clock_source
set_instance_parameter_value clk_0 clockFrequency {50000000.0}
add_instance jtag_master altera_jtag_avalon_master
add_connection clk_0.clk jtag_master.clk
add_connection clk_0.clk_reset jtag_master.clk_reset
add_interface clk clock sink
set_interface_property clk EXPORT_OF clk_0.clk_in
add_interface reset reset sink
set_interface_property reset EXPORT_OF clk_0.clk_in_reset
add_interface lpu_avalon avalon master
set_interface_property lpu_avalon EXPORT_OF jtag_master.master
save_system platform_designer_system.qsys
'''


INIT_TCL = r'''set project_root [file normalize [file dirname [info script]]]
set synthesis_root [file dirname $project_root]
set root [file dirname $synthesis_root]
set project_dir [file join $synthesis_root build lpu_lite_de1_soc]
file mkdir $project_dir
cd $project_dir
# project_new -overwrite leaves an existing QSF in place.  This project file is
# generated by this script, so remove it to prevent stale device assignments.
file delete -force lpu_lite_de1_soc.qsf
project_new -overwrite lpu_lite_de1_soc
set_global_assignment -name FAMILY "Cyclone V"
set_global_assignment -name DEVICE 5CSEMA5F31C6
set_global_assignment -name TOP_LEVEL_ENTITY de1_soc_top
# Exclude simulation-only real-number conversion helpers guarded with
# `ifndef SYNTHESIS in the model RTL. Quartus does not define this macro by
# default, and real data types are not synthesizable.
set_global_assignment -name VERILOG_MACRO SYNTHESIS
set_global_assignment -name SYSTEMVERILOG_FILE [file join $synthesis_root rtl de1_soc_top.sv]
set source_dirs [concat [list [file join $root src]] [glob -nocomplain -types d [file join $root src *]]]
foreach dir $source_dirs {
    # One level is sufficient for LPULite's bus modules; archive is deliberately
    # excluded because it contains alternative implementations.
    if {[file tail $dir] ne "archive"} {
        foreach f [glob -nocomplain [file join $dir *.sv]] {
            if {$f ne [file join $root src lpu_pkg.sv]} {
                set_global_assignment -name SYSTEMVERILOG_FILE $f
            }
        }
    }
}
foreach f [glob -nocomplain [file join $synthesis_root rtl *.sv]] {
    if {$f ne [file join $synthesis_root rtl de1_soc_top.sv]} {
        set_global_assignment -name SYSTEMVERILOG_FILE $f
    }
}
set_global_assignment -name SEARCH_PATH [file join $root src]
if {0} {
# CVFPU is kept here as a reference source list, but is intentionally disabled:
# the pinned upstream sources use SystemVerilog constructs Quartus Lite 25.1
# does not support. A Quartus-compatible FPU implementation is required before
# this list can be enabled.
set cvfpu [file join $root third_party cvfpu]
set_global_assignment -name SEARCH_PATH [file join $cvfpu src common_cells include]
# The model's FP32 wrappers bind to the repository-pinned CVFPU implementation
# when HAVE_CVFPU is defined. Keep this curated list in the same order as the
# existing simulation makefiles; do not glob the CVFPU tree because it contains
# mutually exclusive test and vendor implementations.
foreach rel {
    src/common_cells/src/cf_math_pkg.sv
    src/common_cells/src/lzc.sv
    src/common_cells/src/rr_arb_tree.sv
    src/fpu_div_sqrt_mvp/hdl/defs_div_sqrt_mvp.sv
    src/fpu_div_sqrt_mvp/hdl/iteration_div_sqrt_mvp.sv
    src/fpu_div_sqrt_mvp/hdl/control_mvp.sv
    src/fpu_div_sqrt_mvp/hdl/norm_div_sqrt_mvp.sv
    src/fpu_div_sqrt_mvp/hdl/preprocess_mvp.sv
    src/fpu_div_sqrt_mvp/hdl/nrbd_nrsc_mvp.sv
    src/fpu_div_sqrt_mvp/hdl/div_sqrt_top_mvp.sv
    src/fpu_div_sqrt_mvp/hdl/div_sqrt_mvp_wrapper.sv
    vendor/opene906/E906_RTL_FACTORY/gen_rtl/clk/rtl/gated_clk_cell.v
    vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_ctrl.v
    vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_ff1.v
    vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_pack_single.v
    vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_prepare.v
    vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_round_single.v
    vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_special.v
    vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_srt_single.v
    vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_top.v
    vendor/opene906/E906_RTL_FACTORY/gen_rtl/fpu/rtl/pa_fpu_dp.v
    vendor/opene906/E906_RTL_FACTORY/gen_rtl/fpu/rtl/pa_fpu_frbus.v
    vendor/opene906/E906_RTL_FACTORY/gen_rtl/fpu/rtl/pa_fpu_src_type.v
    vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_ctrl.v
    vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_double.v
    vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_ff1.v
    vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_pack.v
    vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_prepare.v
    vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_round.v
    vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_scalar_dp.v
    vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_srt_radix16_bound_table.v
    vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_srt_radix16_with_sqrt.v
    vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_srt.v
    vendor/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ct_vfdsu_top.v
    vendor/cvw/fma/fmalza.sv
    src/fpnew_pkg.sv
    src/fpnew_cast_multi.sv
    src/fpnew_classifier.sv
    src/fpnew_divsqrt_th_32.sv
    src/fpnew_divsqrt_th_64_multi.sv
    src/fpnew_divsqrt_multi.sv
    src/fpnew_fma.sv
    src/fpnew_fma_multi.sv
    src/fpnew_noncomp.sv
    src/fpnew_opgroup_block.sv
    src/fpnew_opgroup_fmt_slice.sv
    src/fpnew_rounding.sv
    src/fpnew_top.sv
} {
    set f [file join $cvfpu $rel]
    if {![file exists $f]} { error "Required CVFPU source is missing: $f. Run: git submodule update --init --recursive" }
    if {[file extension $f] eq ".v"} { set_global_assignment -name VERILOG_FILE $f } else { set_global_assignment -name SYSTEMVERILOG_FILE $f }
}
}
set_location_assignment PIN_AF14 -to CLOCK_50
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to CLOCK_50
set_location_assignment PIN_AJ4 -to KEY[0]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to KEY[0]
set_location_assignment PIN_V16 -to LEDR[0]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to LEDR[0]
project_close
'''

ADD_QIP_TCL = r'''set project_root [file normalize [file dirname [info script]]]
set synthesis_root [file dirname $project_root]
set project_dir [file join $synthesis_root build lpu_lite_de1_soc]
cd $project_dir
project_open lpu_lite_de1_soc
set_global_assignment -name QIP_FILE [file join $project_root platform_designer_system synthesis platform_designer_system.qip]
project_close
'''


def find_quartus() -> Path:
    requested = os.environ.get("QUARTUS_BIN64")
    candidates = [Path(requested)] if requested else []
    candidates += [
        Path(r"C:\intelFPGA_lite\18.1\quartus\bin64"),
        Path(r"C:\intelFPGA\18.1\quartus\bin64"),
        Path(r"C:\altera_lite\25.1std\quartus\bin64"),
    ]
    for base in (Path(r"C:\intelFPGA_lite"), Path(r"C:\intelFPGA"), Path(r"C:\altera_lite"), Path(r"C:\altera")):
        if base.exists(): candidates += list(base.glob("*\\quartus\\bin64"))
    for candidate in candidates:
        if (candidate / "quartus_sh.exe").is_file(): return candidate
    found = shutil.which("quartus_sh")
    if found: return Path(found).parent
    raise RuntimeError("Quartus was not found. Set QUARTUS_BIN64 to its bin64 directory.")


def qsys_api_version(qsys_bin: Path) -> str:
    """Return the major.minor API version advertised by Platform Designer."""
    # qsys_bin is <quartus>/sopc_builder/bin; the full numeric version is in
    # <quartus>/version.txt, while the SOPC Builder file only has a label.
    for version_file in (qsys_bin.parent.parent / "version.txt", qsys_bin.parent / "version.txt"):
        if version_file.is_file():
            match = re.search(r"Version=(\d+\.\d+)", version_file.read_text(encoding="utf-8", errors="replace"))
            if match:
                return match.group(1)
    # Both the requested Quartus 18.1 and current 25.1 use these APIs.
    return "18.1"


def run(command: list[str], cwd: Path = ROOT) -> None:
    print("+", subprocess.list2cmdline(command))
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-only", action="store_true", help="create sources/Qsys/project but do not compile")
    args = parser.parse_args()
    patch_rtl()
    write(SYNTHESIS_RTL / "de1_soc_top.sv", TOP)
    write(SYNTHESIS_RTL / "lpu_de1_soc_wrapper.sv", WRAPPER)
    init = PROJECT_FILES / "quartus_init_de1_soc.tcl"
    write(init, INIT_TCL)
    add_qip = PROJECT_FILES / "quartus_add_qip_de1_soc.tcl"
    write(add_qip, ADD_QIP_TCL)
    quartus = find_quartus()
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    qsys_bin = quartus.parent / "sopc_builder" / "bin"
    if (not (qsys_bin / "qsys-script.exe").is_file() or
            not (qsys_bin / "qsys-generate.exe").is_file()):
        raise RuntimeError(f"Platform Designer tools were not found in {qsys_bin}.")
    api_version = qsys_api_version(qsys_bin)
    qsys = PROJECT_FILES / "platform_designer_system.qsys.tcl"
    write(qsys, SYSTEM_TCL.replace("@QSYS_API_VERSION@", api_version))
    # Initialize the Quartus project first, then let Platform Designer add the
    # generated QIP referenced by that project.
    run([str(quartus / "quartus_sh.exe"), "-t", str(init)], PROJECT_DIR)
    # The JTAG master is exported from Platform Designer and connected to the
    # repository HDL wrapper at the Quartus top level.  This avoids fragile
    # version-specific custom-IP cataloguing entirely.
    qsys_script_command = [
        str(qsys_bin / "qsys-script.exe"),
        f"--script={qsys}",
    ]
    print("+", subprocess.list2cmdline(qsys_script_command))
    subprocess.run(qsys_script_command, cwd=PROJECT_FILES, check=True)
    run(
        [
            str(qsys_bin / "qsys-generate.exe"),
            str(PROJECT_FILES / "platform_designer_system.qsys"),
            "--synthesis=VERILOG",
        ],
        PROJECT_FILES,
    )
    run([str(quartus / "quartus_sh.exe"), "-t", str(add_qip)], PROJECT_DIR)
    if not args.generate_only:
        project_file = PROJECT_DIR / PROJECT
        run([str(quartus / "quartus_sh.exe"), "--flow", "compile", str(project_file)], PROJECT_DIR)
        print(f"Success: {PROJECT_DIR / 'output_files' / (PROJECT + '.sof')}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
