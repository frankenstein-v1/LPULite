#!/usr/bin/env python3
"""DE1-SoC FPGA Hardware JTAG Driver.

Communicates over Intel System Console CLI to read/write Avalon-MM slaves
(IMEM, MEM0, MEM1, and CTRL) directly on the physical Cyclone V FPGA hardware.
"""
import os
import sys
import subprocess
from pathlib import Path

SYSTEM_CONSOLE_PATH = Path("C:/altera_lite/25.1std/quartus/sopc_builder/bin/system-console.exe")

class DE1SoCJTAGDriver:
    def __init__(self, system_console_path=SYSTEM_CONSOLE_PATH):
        self.system_console_path = Path(system_console_path)
        if not self.system_console_path.is_file():
            raise FileNotFoundError(f"System Console not found at {self.system_console_path}")
        self.proc = None
        self._start_persistent_console()
        self.verify_fpga_hardware_connection()

    def _start_persistent_console(self):
        """Start a single background System Console process to eliminate Java VM startup overhead."""
        cmd = [str(self.system_console_path), "--cli"]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

    def run_tcl_script(self, tcl_code: str) -> str:
        """Run Tcl commands through the persistent System Console background process."""
        if self.proc is None or self.proc.poll() is not None:
            self._start_persistent_console()

        marker = "__CMD_END_MARKER__"
        clean_tcl = tcl_code.strip() + f'\nputs "{marker}"\n'
        
        try:
            self.proc.stdin.write(clean_tcl)
            self.proc.stdin.flush()
        except OSError:
            self._start_persistent_console()
            self.proc.stdin.write(clean_tcl)
            self.proc.stdin.flush()

        output_lines = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                break
            if marker in line:
                break
            output_lines.append(line)

        return "".join(output_lines)

    def verify_fpga_hardware_connection(self):
        """Instant non-blocking hardware connection check using Quartus jtagconfig utility."""
        jtagconfig = Path("C:/altera_lite/25.1std/quartus/bin64/jtagconfig.exe")
        if not jtagconfig.is_file():
            return
        res = subprocess.run([str(jtagconfig)], capture_output=True, text=True, timeout=5)
        if "02D120DD" not in res.stdout:
            raise RuntimeError("DE1-SoC Cyclone V FPGA chip (0x02D120DD) not detected. Ensure USB-Blaster is connected and powered ON!")

    def read_ctrl_register(self) -> int:
        """Read the LPU Control Register (0xC000)."""
        tcl = """
set master [lindex [get_service_paths master] 0]
open_service master $master
set val [master_read_32 $master 0xC000 1]
close_service master $master
puts "CTRL_VAL:$val"
"""
        out = self.run_tcl_script(tcl)
        for line in out.splitlines():
            if line.startswith("CTRL_VAL:"):
                raw = line.split(":", 1)[1].strip()
                return int(raw, 0)
        return 0

    def write_avalon_words(self, base_address: int, words_32: list[int]):
        """Write a sequence of 32-bit words starting at base_address to the FPGA."""
        tcl_lines = [
            "set master [lindex [get_service_paths master] 0]",
            "open_service master $master"
        ]
        addr = base_address
        for w in words_32:
            tcl_lines.append(f"master_write_32 $master 0x{addr:X} 0x{w & 0xFFFFFFFF:X}")
            addr += 4
        tcl_lines.append("close_service master $master")
        tcl_lines.append('puts "WRITE_SUCCESS"')
        
        self.run_tcl_script("\n".join(tcl_lines))

    def read_avalon_words(self, base_address: int, count: int) -> list[int]:
        """Read count 32-bit words starting at base_address from the FPGA."""
        tcl = f"""
set master [lindex [get_service_paths master] 0]
open_service master $master
set val [master_read_32 $master 0x{base_address:X} {count}]
close_service master $master
puts "READ_VALS:$val"
"""
        out = self.run_tcl_script(tcl)
        for line in out.splitlines():
            if line.startswith("READ_VALS:"):
                raw_list = line.split(":", 1)[1].strip()
                items = raw_list.replace("{", "").replace("}", "").split()
                return [int(item, 0) for item in items]
        return []

    def set_run_enable(self, enable: bool):
        """Set run_enable bit on control register 0xC000."""
        val = 1 if enable else 0
        self.write_avalon_words(0xC000, [val])
