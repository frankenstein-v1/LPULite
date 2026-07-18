#!/usr/bin/env python3
"""Create and compile the TinyLPU DE1-SoC JTAG-programmable design.

Run from any directory with:
    py scripts/build_de1_soc.py

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


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PROJECT = "tiny_lpu_de1_soc"
DEVICE = "5CSEMA5F31C6N"
PROJECT_DIR = ROOT / "build" / PROJECT


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
    if "ext_write_en" not in text:
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
    if "External writes remain available while reset is asserted." not in text:
        text = replace_once(text,
            r"(ext_data_out\s*<=\s*'0;)",
            r"\1\n            // External writes remain available while reset is asserted.\n            if (ext_write_en) sram_array[ext_addr] <= ext_data_in;", "mem reset programming")
        write(mem, text)

    icu = SRC / "icu.sv"
    text = icu.read_text(encoding="utf-8")
    if "ext_write_en" not in text:
        text = replace_once(text, r"(output\s+logic\s+\[2:0\]\s+vxm_residual_op)(\s*\n\s*\);)",
            r"\1,\n\n    // External JTAG write interface for programming\n    input  logic                      ext_write_en,\n    input  logic [$clog2(INSTRUCTION_COUNT)-1:0] ext_addr,\n    input  logic [95:0]               ext_data_in\2", "ICU ports")
        text = replace_once(text, r"(logic\s+\[95:0\]\s+imem_array\s+\[0:INSTRUCTION_COUNT-1\];)",
            r"\1\n\n    always_ff @(posedge clk) begin\n        if (ext_write_en) imem_array[ext_addr] <= ext_data_in;\n    end", "ICU programming write")
        write(icu, text)

    lpu = SRC / "lpu.sv"
    text = lpu.read_text(encoding="utf-8")
    if "ext_imem_write_en" not in text:
        text = replace_once(text, r"(input\s+logic\s+rst_n)(\s*\n\s*\);)",
            r"\1,\n\n    input logic ext_imem_write_en,\n    input logic [9:0] ext_imem_addr,\n    input logic [95:0] ext_imem_data_in,\n    input logic ext_mem0_write_en, ext_mem0_read_en,\n    input logic [MEM_ADDR_W-1:0] ext_mem0_addr,\n    input logic [71:0] ext_mem0_data_in,\n    output logic [71:0] ext_mem0_data_out,\n    input logic ext_mem1_write_en, ext_mem1_read_en,\n    input logic [MEM_ADDR_W-1:0] ext_mem1_addr,\n    input logic [71:0] ext_mem1_data_in,\n    output logic [71:0] ext_mem1_data_out\2", "LPU ports")
        text = replace_once(text, r"(\.vxm_residual_op\(vxm_residual_op\))(\s*\n\s*\);)",
            r"\1,\n    .ext_write_en(ext_imem_write_en),\n    .ext_addr(ext_imem_addr),\n    .ext_data_in(ext_imem_data_in)\2", "ICU hookup")
        for name in ("0", "1"):
            text = replace_once(text, rf"(\.addr\(mem{name}_addr\))(\s*\n\s*\);)",
                rf"\1,\n    .ext_write_en(ext_mem{name}_write_en),\n    .ext_read_en(ext_mem{name}_read_en),\n    .ext_addr(ext_mem{name}_addr),\n    .ext_data_in(ext_mem{name}_data_in),\n    .ext_data_out(ext_mem{name}_data_out)\2", f"MEM{name} hookup")
        write(lpu, text)


TOP = """module de1_soc_top (
    input  logic        CLOCK_50,  // PIN_AF14 (50MHz clock input)
    input  logic [0:0]  KEY,       // PIN_AJ4 (Active-low pushbutton as Reset)
    output logic [0:0]  LEDR       // PIN_V16 (Diagnostic LED)
);
    platform_designer_system u_qsys (
        .clk_clk       (CLOCK_50),
        .reset_reset_n (KEY[0])
    );
    assign LEDR[0] = KEY[0];
endmodule
"""


WRAPPER = r'''`timescale 1ns/1ps
`include "lpu_pkg.sv"

module lpu_de1_soc_wrapper (
    input logic clk, input logic rst_n,
    input logic [15:0] avs_address, input logic avs_read, input logic avs_write,
    input logic [31:0] avs_writedata,
    output logic [31:0] avs_readdata, output logic avs_waitrequest
);
    logic run_enable;
    logic [95:0] imem_assembly;
    logic [71:0] mem0_assembly, mem1_assembly;
    logic ext_imem_write_en;
    logic [9:0] ext_imem_addr;
    logic [95:0] ext_imem_data_in;
    logic ext_mem0_write_en, ext_mem0_read_en, ext_mem1_write_en, ext_mem1_read_en;
    logic [14:0] ext_mem0_addr, ext_mem1_addr;
    logic [71:0] ext_mem0_data_in, ext_mem1_data_in, ext_mem0_data_out, ext_mem1_data_out;
    logic [71:0] mem0_read_latched, mem1_read_latched;

    wire [13:0] imem_word = avs_address[13:2];
    wire [13:0] mem0_word = (avs_address - 16'h4000) >> 2;
    wire [13:0] mem1_word = (avs_address - 16'h8000) >> 2;
    /* Avalon addresses are byte addresses.  Rows use consecutive 32-bit words:
       row = word-address / 3, lane = word-address % 3. */
    integer row_index;
    integer lane_index;

    assign avs_waitrequest = 1'b0;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            run_enable <= 1'b0; imem_assembly <= '0; mem0_assembly <= '0; mem1_assembly <= '0;
            ext_imem_write_en <= 1'b0; ext_mem0_write_en <= 1'b0; ext_mem1_write_en <= 1'b0;
            ext_mem0_read_en <= 1'b0; ext_mem1_read_en <= 1'b0; avs_readdata <= '0;
            mem0_read_latched <= '0; mem1_read_latched <= '0;
        end else begin
            ext_imem_write_en <= 1'b0; ext_mem0_write_en <= 1'b0; ext_mem1_write_en <= 1'b0;
            ext_mem0_read_en <= 1'b0; ext_mem1_read_en <= 1'b0;
            /* The SRAM read is synchronous.  Capture its result for the read
               transaction launched on the preceding cycle. */
            mem0_read_latched <= ext_mem0_data_out;
            mem1_read_latched <= ext_mem1_data_out;
            if (avs_write) begin
                if (avs_address == 16'hc000) run_enable <= avs_writedata[0];
                else if (avs_address < 16'h4000) begin
                    row_index = imem_word / 3; lane_index = imem_word % 3;
                    case (lane_index)
                      0: imem_assembly[31:0] <= avs_writedata;
                      1: imem_assembly[63:32] <= avs_writedata;
                      2: begin ext_imem_write_en <= 1'b1; ext_imem_addr <= row_index[9:0];
                               ext_imem_data_in <= {avs_writedata, imem_assembly[63:0]}; end
                    endcase
                end else if (avs_address < 16'h8000) begin
                    row_index = mem0_word / 3; lane_index = mem0_word % 3;
                    case (lane_index)
                      0: mem0_assembly[31:0] <= avs_writedata;
                      1: mem0_assembly[63:32] <= avs_writedata;
                      2: begin ext_mem0_write_en <= 1'b1; ext_mem0_addr <= row_index[14:0];
                               ext_mem0_data_in <= {avs_writedata[7:0], mem0_assembly[63:0]}; end
                    endcase
                end else if (avs_address < 16'hc000) begin
                    row_index = mem1_word / 3; lane_index = mem1_word % 3;
                    case (lane_index)
                      0: mem1_assembly[31:0] <= avs_writedata;
                      1: mem1_assembly[63:32] <= avs_writedata;
                      2: begin ext_mem1_write_en <= 1'b1; ext_mem1_addr <= row_index[14:0];
                               ext_mem1_data_in <= {avs_writedata[7:0], mem1_assembly[63:0]}; end
                    endcase
                end
            end
            if (avs_read) begin
                if (avs_address == 16'hc000) avs_readdata <= {31'b0, run_enable};
                else if (avs_address >= 16'h4000 && avs_address < 16'h8000) begin
                    row_index = mem0_word / 3; lane_index = mem0_word % 3;
                    ext_mem0_read_en <= 1'b1; ext_mem0_addr <= row_index[14:0];
                    case (lane_index) 0: avs_readdata <= mem0_read_latched[31:0];
                      1: avs_readdata <= mem0_read_latched[63:32]; default: avs_readdata <= {24'b0,mem0_read_latched[71:64]}; endcase
                end else if (avs_address >= 16'h8000 && avs_address < 16'hc000) begin
                    row_index = mem1_word / 3; lane_index = mem1_word % 3;
                    ext_mem1_read_en <= 1'b1; ext_mem1_addr <= row_index[14:0];
                    case (lane_index) 0: avs_readdata <= mem1_read_latched[31:0];
                      1: avs_readdata <= mem1_read_latched[63:32]; default: avs_readdata <= {24'b0,mem1_read_latched[71:64]}; endcase
                end else avs_readdata <= '0; // IMEM has no external read port.
            end
        end
    end
    lpu u_lpu (.clk(clk), .rst_n(rst_n & run_enable),
        .ext_imem_write_en(ext_imem_write_en), .ext_imem_addr(ext_imem_addr), .ext_imem_data_in(ext_imem_data_in),
        .ext_mem0_write_en(ext_mem0_write_en), .ext_mem0_read_en(ext_mem0_read_en), .ext_mem0_addr(ext_mem0_addr), .ext_mem0_data_in(ext_mem0_data_in), .ext_mem0_data_out(ext_mem0_data_out),
        .ext_mem1_write_en(ext_mem1_write_en), .ext_mem1_read_en(ext_mem1_read_en), .ext_mem1_addr(ext_mem1_addr), .ext_mem1_data_in(ext_mem1_data_in), .ext_mem1_data_out(ext_mem1_data_out));
endmodule
'''


HW_TCL = r'''package require -exact qsys @QSYS_API_VERSION@
set_module_property NAME lpu_de1_soc
set_module_property VERSION 1.0
set_module_property GROUP "TinyLPU"
set_module_property DISPLAY_NAME "TinyLPU DE1-SoC Avalon wrapper"
set_module_property TOP_LEVEL_HDL_FILE lpu_de1_soc_wrapper.sv
set_module_property TOP_LEVEL_HDL_MODULE lpu_de1_soc_wrapper
set_module_property INSTANTIATE_IN_SYSTEM_MODULE true
add_fileset QUARTUS_SYNTH QUARTUS_SYNTH ""
set_fileset_property QUARTUS_SYNTH TOP_LEVEL lpu_de1_soc_wrapper
add_fileset_file lpu_de1_soc_wrapper.sv SYSTEM_VERILOG PATH lpu_de1_soc_wrapper.sv TOP_LEVEL_FILE
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
set_interface_property avs maximumPendingReadTransactions 1
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
set_project_property DEVICE 5CSEMA5F31C6N
add_instance clk_0 clock_source
set_instance_parameter_value clk_0 clockFrequency {50000000.0}
add_instance reset_bridge_0 altera_reset_bridge
set_instance_parameter_value reset_bridge_0 ACTIVE_LOW_RESET {1}
set_instance_parameter_value reset_bridge_0 SYNCHRONOUS_EDGES {none}
add_instance jtag_master altera_jtag_avalon_master
add_instance lpu_de1_soc_0 lpu_de1_soc
add_connection clk_0.clk reset_bridge_0.clk
add_connection clk_0.clk jtag_master.clk
add_connection clk_0.clk lpu_de1_soc_0.clk
add_connection reset_bridge_0.out_reset jtag_master.reset
add_connection reset_bridge_0.out_reset lpu_de1_soc_0.rst_n
add_connection jtag_master.master lpu_de1_soc_0.avs
set_connection_parameter_value jtag_master.master/lpu_de1_soc_0.avs baseAddress {0x00000000}
add_interface clk clock sink
set_interface_property clk EXPORT_OF clk_0.clk
add_interface reset reset sink
set_interface_property reset EXPORT_OF reset_bridge_0.in_reset
save_system platform_designer_system.qsys
'''


INIT_TCL = r'''set root [file normalize [file dirname [info script]]]
set project_dir [file join $root build tiny_lpu_de1_soc]
file mkdir $project_dir
cd $project_dir
project_new -overwrite tiny_lpu_de1_soc
set_global_assignment -name FAMILY "Cyclone V"
set_global_assignment -name DEVICE 5CSEMA5F31C6N
set_global_assignment -name TOP_LEVEL_ENTITY de1_soc_top
set_global_assignment -name SYSTEMVERILOG_FILE [file join $root src de1_soc_top.sv]
foreach f [glob -nocomplain [file join $root src *.sv]] {
    if {$f ne [file join $root src de1_soc_top.sv] && $f ne [file join $root src lpu_de1_soc_wrapper.sv]} { set_global_assignment -name SYSTEMVERILOG_FILE $f }
}
set_global_assignment -name SEARCH_PATH [file join $root src]
set_location_assignment PIN_AF14 -to CLOCK_50
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to CLOCK_50
set_location_assignment PIN_AJ4 -to KEY[0]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to KEY[0]
set_location_assignment PIN_V16 -to LEDR[0]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to LEDR[0]
project_close
'''

ADD_QIP_TCL = r'''set root [file normalize [file dirname [info script]]]
set project_dir [file join $root build tiny_lpu_de1_soc]
cd $project_dir
project_open tiny_lpu_de1_soc
set_global_assignment -name QIP_FILE [file join $root platform_designer_system synthesis platform_designer_system.qip]
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
    version_file = qsys_bin.parent / "version.txt"
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
    write(SRC / "de1_soc_top.sv", TOP)
    write(SRC / "lpu_de1_soc_wrapper.sv", WRAPPER)
    ip = ROOT / "ip" / "lpu_de1_soc"
    write(ip / "lpu_de1_soc_wrapper.sv", WRAPPER)
    init = ROOT / "quartus_init_de1_soc.tcl"
    write(init, INIT_TCL)
    add_qip = ROOT / "quartus_add_qip_de1_soc.tcl"
    write(add_qip, ADD_QIP_TCL)
    quartus = find_quartus()
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    qsys_bin = quartus.parent / "sopc_builder" / "bin"
    if not (qsys_bin / "qsys-script.exe").is_file() or not (qsys_bin / "qsys-generate.exe").is_file():
        raise RuntimeError(f"Platform Designer tools were not found in {qsys_bin}.")
    api_version = qsys_api_version(qsys_bin)
    write(ip / "lpu_de1_soc_hw.tcl", HW_TCL.replace("@QSYS_API_VERSION@", api_version))
    qsys = ROOT / "platform_designer_system.qsys.tcl"
    write(qsys, SYSTEM_TCL.replace("@QSYS_API_VERSION@", api_version))
    # Initialize the Quartus project first, then let Platform Designer add the
    # generated QIP referenced by that project.
    run([str(quartus / "quartus_sh.exe"), "-t", str(init)], PROJECT_DIR)
    # Qsys finds the packaged component through this repository-local IP path.
    env = os.environ.copy()
    env["QSYS_COMPONENT_DIR"] = str(ROOT / "ip")
    qsys_script_command = [str(qsys_bin / "qsys-script.exe"), "--search-path", str(ip), "--script", str(qsys)]
    print("+", subprocess.list2cmdline(qsys_script_command))
    subprocess.run(qsys_script_command, cwd=ROOT, env=env, check=True)
    run([str(qsys_bin / "qsys-generate.exe"), "platform_designer_system.qsys", "--search-path", str(ip), "--synthesis=VERILOG"])
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
