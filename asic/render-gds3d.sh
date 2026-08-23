#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "Usage: $0 /path/to/GDS3D /path/to/layout.gds [techfile] [topcell]"
    exit 2
fi

gds3d_root=$1
gds_file=$2
tech_file=${3:-"$PWD/asic/gds3d_sky130_showcase.txt"}
top_cell=${4:-lpulite_showcase}
gds3d_bin="$gds3d_root/mac/GDS3D.app/Contents/MacOS/GDS3D"

if [[ ! -x "$gds3d_bin" ]]; then
    echo "GDS3D executable not found: $gds3d_bin"
    echo "Build it with: make -C \"$gds3d_root/mac\""
    exit 1
fi

if [[ ! -f "$gds_file" ]]; then
    echo "GDS file not found: $gds_file"
    exit 1
fi

if [[ ! -f "$tech_file" ]]; then
    echo "SKY130 GDS3D process file not found: $tech_file"
    exit 1
fi

exec "$gds3d_bin" \
    -p "$tech_file" \
    -i "$gds_file" \
    -t "$top_cell" \
    -v
