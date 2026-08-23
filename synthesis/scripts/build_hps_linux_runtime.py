#!/usr/bin/env python3
"""Build helper for the DE1-SoC HPS/Linux LPULite runtime."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LINUX_DIR = ROOT / "synthesis" / "linux"


def run(cmd: list[str], cwd: Path) -> None:
    print("+", subprocess.list2cmdline(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate-only", action="store_true", help="only regenerate include/microgpt_hps_image.h")
    args = parser.parse_args()

    run([sys.executable, str(ROOT / "synthesis" / "scripts" / "export_microgpt_hps_headers.py")], ROOT)

    if args.generate_only:
        return 0

    make = shutil.which("make")
    if not make:
        print("NOTE: make was not found. Header generation succeeded; compile on the DE1-SoC Linux shell with:", flush=True)
        print("  cd synthesis/linux && make", flush=True)
        return 0

    run([make], LINUX_DIR)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: command failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode)
