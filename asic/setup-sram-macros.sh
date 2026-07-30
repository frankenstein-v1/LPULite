#!/usr/bin/env bash
set -euo pipefail

repo_url=https://github.com/VLSIDA/sky130_sram_macros.git
macro_name=sky130_sram_1kbyte_1rw1r_8x1024_8
script_dir=$(cd "$(dirname "$0")" && pwd)
dest_dir="$script_dir/macros"
temp_dir=$(mktemp -d)

cleanup() {
    rm -rf "$temp_dir"
}
trap cleanup EXIT

git clone --depth 1 "$repo_url" "$temp_dir/sky130_sram_macros"
mkdir -p "$dest_dir"

cp "$temp_dir/sky130_sram_macros/$macro_name/${macro_name}.gds" "$dest_dir/"
cp "$temp_dir/sky130_sram_macros/$macro_name/${macro_name}.lef" "$dest_dir/"
cp "$temp_dir/sky130_sram_macros/$macro_name/${macro_name}.v" "$dest_dir/"
cp "$temp_dir/sky130_sram_macros/$macro_name/${macro_name}_TT_1p8V_25C.lib" "$dest_dir/"

echo "Installed $macro_name physical and timing views in $dest_dir"
