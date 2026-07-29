#!/usr/bin/env python3
"""Cross-platform build script for TinyLPU C hardware driver.

Automatically exports C headers (export_c_headers.py) and compiles src/lpu_driver.c
using the best available C compiler on Windows/Linux (gcc, clang, cl, or wsl gcc).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
INCLUDE_DIR = ROOT_DIR / "include"
OUTPUT_EXE = ROOT_DIR / ("lpu_driver.exe" if sys.platform == "win32" else "lpu_driver")

def generate_headers():
    print("  [1/2] Exporting C header files...", flush=True)
    cmd = [sys.executable, str(ROOT_DIR / "scripts" / "export_c_headers.py")]
    res = subprocess.run(cmd, cwd=ROOT_DIR, text=True)
    if res.returncode != 0:
        raise RuntimeError("Failed to export C header files.")

def find_compiler():
    # 1. Check GCC / MinGW / Clang on system PATH
    for comp in ("gcc", "clang", "cl"):
        path = shutil.which(comp)
        if path:
            return comp, path

    # 2. Check Quartus or common GCC Windows paths
    candidates = [
        Path(r"C:\mingw64\bin\gcc.exe"),
        Path(r"C:\msys64\ucrt64\bin\gcc.exe"),
        Path(r"C:\msys64\mingw64\bin\gcc.exe"),
        Path(r"C:\TDM-GCC-64\bin\gcc.exe"),
    ]
    for c in candidates:
        if c.is_file():
            return "gcc", str(c)

    return None, None

def compile_c_driver():
    comp_type, comp_path = find_compiler()
    print("  [2/2] Compiling src/lpu_driver.c...", flush=True)

    if not comp_path:
        print("\n[NOTE]: No local C compiler (gcc/clang/cl) was found on Windows PATH.", flush=True)
        print("  - To compile on Windows: Install MinGW-w64 or Visual Studio C++ Build Tools.", flush=True)
        print("  - To compile on DE1-SoC ARM Linux: Transfer the repo and run 'gcc -O3 -Iinclude -Isrc src/lpu_driver.c -o lpu_driver'\n", flush=True)
        return False

    print(f"  [COMPILER]: Using {comp_type} ({comp_path})", flush=True)
    if comp_type in ("gcc", "clang"):
        cmd = [comp_path, "-O3", "-Wall", "-Iinclude", "-Isrc", str(SRC_DIR / "lpu_driver.c"), "-o", str(OUTPUT_EXE)]
    else: # MSVC cl
        cmd = [comp_path, "/O2", "/Iinclude", "/Isrc", str(SRC_DIR / "lpu_driver.c"), f"/Fe:{OUTPUT_EXE}"]

    print("+", " ".join(cmd), flush=True)
    res = subprocess.run(cmd, cwd=ROOT_DIR, text=True)
    if res.returncode == 0:
        print(f"  [SUCCESS]: Compiled binary -> {OUTPUT_EXE.relative_to(ROOT_DIR)}", flush=True)
        return True
    else:
        print("  [ERROR]: Compilation failed.", flush=True)
        return False

def main():
    print("\n========================================================================", flush=True)
    print("        TINYLPU C HARDWARE DRIVER CROSS-PLATFORM BUILDER                ", flush=True)
    print("========================================================================\n", flush=True)
    generate_headers()
    compile_c_driver()

if __name__ == "__main__":
    main()
