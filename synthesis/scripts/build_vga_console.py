#!/usr/bin/env python3
"""Build the standalone DE1-SoC VGA text-console bring-up image.

This target is intentionally separate from the JTAG/LPU design. It verifies the
monitor, cable, VGA DAC pins and text renderer before a MicroGPT decode engine
is connected to it.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SYNTHESIS = ROOT / "synthesis"
RTL = SYNTHESIS / "rtl"
CONSTRAINTS = SYNTHESIS / "constraints"
BUILD = SYNTHESIS / "build" / "vga_console"
PROJECT = "vga_console"

SOURCES = [
    RTL / "vga_text_console.sv",
    RTL / "vga_boot_banner.sv",
    RTL / "de1_soc_vga_console_top.sv",
]

# Pin assignments from the DE1-SoC User Manual, VGA table. The standalone
# target uses separate H/V sync, so VGA_SYNC_N is not required.
QSF = r'''set_global_assignment -name FAMILY "Cyclone V"
set_global_assignment -name DEVICE 5CSEMA5F31C6
set_global_assignment -name TOP_LEVEL_ENTITY de1_soc_vga_console_top
set_global_assignment -name NUM_PARALLEL_PROCESSORS 4
set_global_assignment -name SYSTEMVERILOG_FILE "{vga_text_console}"
set_global_assignment -name SYSTEMVERILOG_FILE "{vga_boot_banner}"
set_global_assignment -name SYSTEMVERILOG_FILE "{top}"
set_global_assignment -name SDC_FILE "{sdc}"

set_location_assignment PIN_AF14 -to CLOCK_50
set_location_assignment PIN_AA14 -to KEY[0]
set_location_assignment PIN_V16  -to LEDR[0]

set_location_assignment PIN_A13 -to VGA_R[0]
set_location_assignment PIN_C13 -to VGA_R[1]
set_location_assignment PIN_E13 -to VGA_R[2]
set_location_assignment PIN_B12 -to VGA_R[3]
set_location_assignment PIN_C12 -to VGA_R[4]
set_location_assignment PIN_D12 -to VGA_R[5]
set_location_assignment PIN_E12 -to VGA_R[6]
set_location_assignment PIN_F13 -to VGA_R[7]

set_location_assignment PIN_J9  -to VGA_G[0]
set_location_assignment PIN_J10 -to VGA_G[1]
set_location_assignment PIN_H12 -to VGA_G[2]
set_location_assignment PIN_G10 -to VGA_G[3]
set_location_assignment PIN_G11 -to VGA_G[4]
set_location_assignment PIN_G12 -to VGA_G[5]
set_location_assignment PIN_F11 -to VGA_G[6]
set_location_assignment PIN_E11 -to VGA_G[7]

set_location_assignment PIN_B13 -to VGA_B[0]
set_location_assignment PIN_G13 -to VGA_B[1]
set_location_assignment PIN_H13 -to VGA_B[2]
set_location_assignment PIN_F14 -to VGA_B[3]
set_location_assignment PIN_H14 -to VGA_B[4]
set_location_assignment PIN_F15 -to VGA_B[5]
set_location_assignment PIN_G15 -to VGA_B[6]
set_location_assignment PIN_J14 -to VGA_B[7]

set_location_assignment PIN_A11 -to VGA_CLK
set_location_assignment PIN_F10 -to VGA_BLANK_N
set_location_assignment PIN_B11 -to VGA_HS
set_location_assignment PIN_D11 -to VGA_VS
set_location_assignment PIN_C10 -to VGA_SYNC_N

set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to CLOCK_50
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to KEY[0]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to LEDR[0]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to VGA_R[*]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to VGA_G[*]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to VGA_B[*]
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to VGA_CLK
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to VGA_BLANK_N
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to VGA_HS
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to VGA_VS
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to VGA_SYNC_N
'''


def write_if_changed(path: Path, content: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def find_quartus() -> Path:
    for base in (
        Path(r"C:\altera_lite\25.1std\quartus\bin64"),
        Path(r"C:\intelFPGA_lite\18.1\quartus\bin64"),
        Path(r"C:\intelFPGA\18.1\quartus\bin64"),
    ):
        if (base / "quartus_sh.exe").is_file():
            return base
    found = shutil.which("quartus_sh")
    if found:
        return Path(found).parent
    raise RuntimeError("Quartus was not found. Set QUARTUS_BIN64 or install Quartus Prime.")


def quartus_safe_paths() -> tuple[Path, callable, callable]:
    """Return a root whose path Quartus can parse and a matching cleanup.

    Quartus 25.1 treats parentheses in a project path as Tcl syntax. The shared
    workspace is named ``Documents(Local)``, so map the repository to a spare
    temporary drive on Windows rather than copying the project elsewhere.
    """
    root_text = str(ROOT)
    if os.name != "nt" or ("(" not in root_text and ")" not in root_text):
        return ROOT, lambda path: path, lambda: None

    mapped = subprocess.run(["subst"], capture_output=True, text=True, check=True).stdout.upper()
    drive = next((candidate for candidate in ("T", "U", "V", "W", "X", "Y", "Z")
                  if f"{candidate}:" not in mapped), None)
    if drive is None:
        raise RuntimeError("No spare drive letter is available for the Quartus path workaround.")
    subprocess.run(["subst", f"{drive}:", root_text], check=True)
    safe_root = Path(f"{drive}:/")

    def to_safe(path: Path) -> Path:
        return safe_root / path.relative_to(ROOT)

    def cleanup() -> None:
        subprocess.run(["subst", f"{drive}:", "/D"], check=False)

    return safe_root, to_safe, cleanup


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()

    missing = [str(path) for path in SOURCES if not path.is_file()]
    sdc = CONSTRAINTS / "vga_console.sdc"
    if not sdc.is_file():
        missing.append(str(sdc))
    if missing:
        raise RuntimeError("Missing VGA source files:\n" + "\n".join(missing))

    safe_root, to_safe, cleanup = quartus_safe_paths()
    try:
        qsf = QSF.format(
            vga_text_console=to_safe(SOURCES[0]).as_posix(),
            vga_boot_banner=to_safe(SOURCES[1]).as_posix(),
            top=to_safe(SOURCES[2]).as_posix(),
            sdc=to_safe(sdc).as_posix(),
        )
        write_if_changed(BUILD / f"{PROJECT}.qsf", qsf)
        write_if_changed(BUILD / f"{PROJECT}.qpf", 'QUARTUS_VERSION = "25.1"\nPROJECT_REVISION = "vga_console"\n')
        print(f"Generated {BUILD.relative_to(ROOT)}")
        if args.generate_only:
            return 0

        quartus = find_quartus()
        command = [str(quartus / "quartus_sh.exe"), "--flow", "compile", PROJECT]
        print("+", subprocess.list2cmdline(command))
        subprocess.run(command, cwd=to_safe(BUILD), check=True)
        print(f"SOF: {BUILD / (PROJECT + '.sof')}")
        return 0
    finally:
        cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
