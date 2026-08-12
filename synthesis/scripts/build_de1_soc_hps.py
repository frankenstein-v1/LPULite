#!/usr/bin/env python3
"""Create/compile a TinyLPU DE1-SoC design driven by the HPS LW bridge.

This is a separate target from build_de1_soc.py.  It keeps the existing
JTAG-to-Avalon build intact and creates:

    synthesis/build/tiny_lpu_de1_soc_hps/

The intended runtime path is:

    ARM Linux /dev/mem @ 0xFF200000
      -> HPS lightweight HPS-to-FPGA bridge
      -> Platform Designer AXI/Avalon interconnect
      -> lpu_de1_soc_wrapper Avalon slave
      -> TinyLPU

The TinyLPU compute RTL is not changed by this script.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(os.environ.get("TINYLPU_REPO_ROOT", Path(__file__).resolve().parents[2])).absolute()
SYNTHESIS_DIR = ROOT / "synthesis"
SRC = ROOT / "src"
SYNTHESIS_RTL = SYNTHESIS_DIR / "rtl"
PROJECT_FILES = SYNTHESIS_DIR / "project"
IP_DIR = PROJECT_FILES / "ip"
LPU_IP_DIR = IP_DIR / "lpu_de1_soc"
PROJECT = "tiny_lpu_de1_soc_hps"
PROJECT_DIR = SYNTHESIS_DIR / "build" / PROJECT
DEVICE = "5CSEMA5F31C6"


def write(path: Path, contents: str) -> None:
    contents = contents.replace("\r\n", "\n")
    if not path.exists() or path.read_text(encoding="utf-8").replace("\r\n", "\n") != contents:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8", newline="\n")
        print(f"wrote {path.relative_to(ROOT)}")


def find_quartus() -> Path:
    requested = os.environ.get("QUARTUS_BIN64")
    candidates = [Path(requested)] if requested else []
    candidates += [
        Path(r"C:\altera_lite\25.1std\quartus\bin64"),
        Path(r"C:\intelFPGA_lite\18.1\quartus\bin64"),
        Path(r"C:\intelFPGA\18.1\quartus\bin64"),
    ]
    for base in (Path(r"C:\altera_lite"), Path(r"C:\altera"), Path(r"C:\intelFPGA_lite"), Path(r"C:\intelFPGA")):
        if base.exists():
            candidates += list(base.glob("*\\quartus\\bin64"))
    for candidate in candidates:
        if candidate and (candidate / "quartus_sh.exe").is_file():
            return candidate
    found = shutil.which("quartus_sh")
    if found:
        return Path(found).parent
    raise RuntimeError("Quartus was not found. Set QUARTUS_BIN64 to its bin64 directory.")


def qsys_api_version(qsys_bin: Path) -> str:
    for version_file in (qsys_bin.parent.parent / "version.txt", qsys_bin.parent / "version.txt"):
        if version_file.is_file():
            text = version_file.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                if line.startswith("Version="):
                    return line.split("=", 1)[1].strip().split(".")[0] + "." + line.split("=", 1)[1].strip().split(".")[1]
    return "25.1"


def run(command: list[str], cwd: Path = ROOT, *, allow_fail: bool = False) -> subprocess.CompletedProcess[str]:
    print("+", subprocess.list2cmdline(command), flush=True)
    completed = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.returncode != 0 and not allow_fail:
        raise subprocess.CalledProcessError(completed.returncode, command, output=completed.stdout)
    return completed


TOP = """module de1_soc_hps_top (
    input  logic        CLOCK_50,
    input  logic [0:0]  KEY,
    output logic [0:0]  LEDR
);
    // TinyLPU currently does not close timing at the board 50 MHz clock.
    // TimeQuest reports an LPU fabric Fmax of ~11 MHz, so run the entire
    // lightweight-HPS/LPU Platform Designer fabric at 50/8 = 6.25 MHz.
    logic [2:0] qsys_clk_div;
    logic       qsys_clk;

    always_ff @(posedge CLOCK_50 or negedge KEY[0]) begin
        if (!KEY[0]) begin
            qsys_clk_div <= 3'd0;
        end else begin
            qsys_clk_div <= qsys_clk_div + 3'd1;
        end
    end

    assign qsys_clk = qsys_clk_div[2];

    platform_designer_hps_system u_qsys (
        .clk_clk       (qsys_clk),
        .reset_reset_n (KEY[0])
    );

    assign LEDR[0] = KEY[0];
endmodule
"""


TOP_SDC = """create_clock -name CLOCK_50 -period 20.000 [get_ports {CLOCK_50}]

set qsys_clk_pins [get_pins -nowarn {*qsys_clk_div[2]*|q}]
if {[llength $qsys_clk_pins] > 0} {
    create_generated_clock -name QSYS_CLK -source [get_ports {CLOCK_50}] -divide_by 8 $qsys_clk_pins
}
"""


LPU_HW_TCL = r"""set_module_property NAME lpu_de1_soc
set_module_property VERSION 1.0
set_module_property GROUP "TinyLPU"
set_module_property DISPLAY_NAME "TinyLPU DE1-SoC Avalon wrapper"
set_module_property TOP_LEVEL_HDL_MODULE lpu_de1_soc_wrapper
set_module_property INSTANTIATE_IN_SYSTEM_MODULE true

add_fileset synth QUARTUS_SYNTH generate_synth
set_fileset_property synth TOP_LEVEL lpu_de1_soc_wrapper
proc generate_synth {top_level} {
    add_fileset_file lpu_de1_soc_wrapper.sv SYSTEM_VERILOG PATH lpu_de1_soc_wrapper.sv
}

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
add_interface_port avs avs_readdatavalid readdatavalid Output 1
"""


SYSTEM_TCL = r"""package require -exact qsys @QSYS_API_VERSION@
create_system platform_designer_hps_system
set_project_property DEVICE_FAMILY "Cyclone V"
set_project_property DEVICE 5CSEMA5F31C6

add_instance clk_0 clock_source
set_instance_parameter_value clk_0 clockFrequency {50000000.0}

add_instance hps_0 altera_hps
set_instance_parameter_value hps_0 S2F_Width 0
set_instance_parameter_value hps_0 F2S_Width 0
set_instance_parameter_value hps_0 LWH2F_Enable true
set_instance_parameter_value hps_0 F2SDRAM_Width {}
set_instance_parameter_value hps_0 F2SDRAM_Type {}
set_instance_parameter_value hps_0 MPU_EVENTS_Enable false
# This is intended to avoid regenerating the SDRAM sequencer for an FPGA-only
# bridge image.  Quartus 25.1 may still invoke the HPS SDRAM generator; if it
# does, install a WSL distro or use a board reference HPS handoff project.
set_instance_parameter_value hps_0 quartus_ini_hps_ip_suppress_sdram_synth true

add_instance lpu_0 lpu_de1_soc

add_connection clk_0.clk hps_0.h2f_lw_axi_clock
add_connection clk_0.clk lpu_0.clk
add_connection clk_0.clk_reset lpu_0.rst_n
add_connection hps_0.h2f_lw_axi_master lpu_0.avs
set_connection_parameter_value hps_0.h2f_lw_axi_master/lpu_0.avs baseAddress 0x00000000

add_interface clk clock sink
set_interface_property clk EXPORT_OF clk_0.clk_in
add_interface reset reset sink
set_interface_property reset EXPORT_OF clk_0.clk_in_reset

save_system platform_designer_hps_system.qsys
"""


def prepare_sources() -> None:
    # Reuse the current wrapper as the custom IP HDL so Platform Designer sees
    # the same ABI as the JTAG/runtime code.
    write(SYNTHESIS_RTL / "de1_soc_hps_top.sv", TOP)
    write(SYNTHESIS_RTL / "de1_soc_hps_top.sdc", TOP_SDC)
    write(LPU_IP_DIR / "lpu_de1_soc_hw.tcl", LPU_HW_TCL)
    wrapper_text = (SYNTHESIS_RTL / "lpu_de1_soc_wrapper.sv").read_text(encoding="utf-8")
    write(LPU_IP_DIR / "lpu_de1_soc_wrapper.sv", wrapper_text)
    write(PROJECT_FILES / "platform_designer_hps_system.qsys.tcl", SYSTEM_TCL.replace("@QSYS_API_VERSION@", "25.1"))


def qpath(path: Path) -> str:
    return str(path.absolute()).replace("\\", "/")


def source_files() -> list[Path]:
    files = [SYNTHESIS_RTL / "de1_soc_hps_top.sv"]
    for directory in [SRC, *[p for p in SRC.iterdir() if p.is_dir()]]:
        if directory.name == "archive":
            continue
        for path in sorted(directory.glob("*.sv")):
            if path.name != "lpu_pkg.sv":
                files.append(path)
    return files


def write_project_files(include_qip: bool) -> None:
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    write(
        PROJECT_DIR / f"{PROJECT}.qpf",
        f"""QUARTUS_VERSION = "25.1"
DATE = "August 9, 2026"

PROJECT_REVISION = "{PROJECT}"
""",
    )
    lines = [
        'set_global_assignment -name FAMILY "Cyclone V"',
        f"set_global_assignment -name DEVICE {DEVICE}",
        "set_global_assignment -name TOP_LEVEL_ENTITY de1_soc_hps_top",
        "set_global_assignment -name VERILOG_MACRO SYNTHESIS",
        "set_global_assignment -name VERILOG_MACRO TINYLPU_FPGA_ALTSYNCRAM",
        "set_global_assignment -name VERILOG_MACRO TINYLPU_MXM_MAC_LOGIC_MULT",
        "set_global_assignment -name VERILOG_MACRO TINYLPU_VXM_LOGIC_MULT",
    ]
    lines.append(f'set_global_assignment -name SDC_FILE "{qpath(SYNTHESIS_RTL / "de1_soc_hps_top.sdc")}"')
    for path in source_files():
        lines.append(f'set_global_assignment -name SYSTEMVERILOG_FILE "{qpath(path)}"')
    lines += [
        f'set_global_assignment -name SEARCH_PATH "{qpath(SRC)}"',
        f'set_global_assignment -name SEARCH_PATH "{qpath(LPU_IP_DIR)}"',
    ]
    qip = PROJECT_FILES / "platform_designer_hps_system" / "synthesis" / "platform_designer_hps_system.qip"
    if include_qip:
        lines.append(f'set_global_assignment -name QIP_FILE "{qpath(qip)}"')
    lines += [
        "set_location_assignment PIN_AF14 -to CLOCK_50",
        'set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to CLOCK_50',
        "set_location_assignment PIN_AJ4 -to KEY[0]",
        'set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to KEY[0]',
        "set_location_assignment PIN_V16 -to LEDR[0]",
        'set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to LEDR[0]',
        "set_global_assignment -name PARTITION_NETLIST_TYPE SOURCE -section_id Top",
        "set_global_assignment -name PARTITION_FITTER_PRESERVATION_LEVEL PLACEMENT_AND_ROUTING -section_id Top",
        "set_global_assignment -name PARTITION_COLOR 16764057 -section_id Top",
        "set_instance_assignment -name PARTITION_HIERARCHY root_partition -to | -section_id Top",
        "set_global_assignment -name MAX_BALANCING_DSP_BLOCKS 87",
        "set_global_assignment -name NUM_PARALLEL_PROCESSORS 2",
    ]
    write(PROJECT_DIR / f"{PROJECT}.qsf", "\n".join(lines) + "\n")


def remove_inapplicable_hps_sdram_sdc() -> None:
    """Drop the generated HPS SDRAM SDC for this lightweight-bridge-only top.

    Platform Designer emits DDR timing collateral for the HPS block even though
    this target does not export the HPS SDRAM interface to the FPGA top.  When
    Quartus reads that SDC it cannot find the expected DDR core instance and
    aborts fitter preparation.  The Linux image still uses the HPS hard DDR
    path; this FPGA persona only needs the HPS lightweight bridge.
    """
    qip = PROJECT_FILES / "platform_designer_hps_system" / "synthesis" / "platform_designer_hps_system.qip"
    if not qip.is_file():
        return
    original = qip.read_text(encoding="utf-8", errors="replace")
    filtered_lines = [
        line for line in original.splitlines()
        if "hps_sdram_p0.sdc" not in line
    ]
    filtered = "\n".join(filtered_lines) + "\n"
    if filtered != original.replace("\r\n", "\n"):
        qip.write_text(filtered, encoding="utf-8", newline="\n")
        print(f"filtered generated SDC assignment from {qip.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--skip-qsys-generate", action="store_true", help="stop after writing Tcl/Qsys scripts")
    args = parser.parse_args()

    prepare_sources()

    quartus = find_quartus()
    qsys_bin = quartus.parent / "sopc_builder" / "bin"
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)

    qsys = PROJECT_FILES / "platform_designer_hps_system.qsys.tcl"
    write_project_files(include_qip=False)
    run([
        str(qsys_bin / "qsys-script.exe"),
        f"--search-path={IP_DIR},$",
        f"--script={qsys}",
    ], PROJECT_FILES)

    if args.skip_qsys_generate:
        return 0

    generated = run([
        str(qsys_bin / "qsys-generate.exe"),
        str(PROJECT_FILES / "platform_designer_hps_system.qsys"),
        "--synthesis=VERILOG",
    ], PROJECT_FILES, allow_fail=True)
    if generated.returncode != 0:
        print("\nHPS Platform Designer generation failed.", file=sys.stderr)
        print("On Quartus 25.1 this commonly means the Nios II command shell needs a real WSL distro", file=sys.stderr)
        print("to generate HPS SDRAM sequencer files, even for this lightweight-bridge target.", file=sys.stderr)
        print("Install a WSL distro, or seed this target from a Terasic/DE1-SoC HPS reference project.", file=sys.stderr)
        return generated.returncode

    remove_inapplicable_hps_sdram_sdc()
    write_project_files(include_qip=True)
    if not args.generate_only:
        run([str(quartus / "quartus_sh.exe"), "--flow", "compile", str(PROJECT_DIR / PROJECT)], PROJECT_DIR)
        print(f"Success: {PROJECT_DIR / (PROJECT + '.sof')}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
