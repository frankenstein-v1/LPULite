#!/usr/bin/env bash
set -euo pipefail

VCD_FILE=${1-}
if [ -z "$VCD_FILE" ]; then
  echo "Usage: $0 <path-to-vcd>"
  exit 1
fi

if [ ! -f "$VCD_FILE" ]; then
  echo "VCD file not found: $VCD_FILE"
  exit 1
fi

VIEWER="${WAVE_VIEWER:-gtkwave}"
GTKWAVE_BIN="${GTKWAVE_BIN:-}"
SURFER_BIN="${SURFER_BIN:-}"

if [ "$VIEWER" = "gtkwave" ]; then
  if ! perl -MSwitch -e '1' >/dev/null 2>&1; then
    # Common Homebrew-local Perl install location used by CPAN
    LOCAL_SWITCH="$HOME/perl5/lib/perl5"
    if [ -f "$LOCAL_SWITCH/Switch.pm" ]; then
      export PERL5LIB="$LOCAL_SWITCH${PERL5LIB:+:${PERL5LIB}}"
    fi
  fi
fi

VIEWER_BIN=""
SURFER_ARGS=()
GTKWAVE_SAVE_ARGS=()

build_gtkwave_savefile() {
  local vcd_file=$1
  local save_file=$2
  local dump_abs
  dump_abs="$(cd "$(dirname "$vcd_file")" && pwd)/$(basename "$vcd_file")"
  {
    echo "[*]"
    echo "[*] GTKWave Analyzer"
    echo "[dumpfile] \"$dump_abs\""
    echo "[dumpfile_size] $(wc -c < "$vcd_file")"
    echo "[dumpfile_mtime] \"$(stat -f '%Sm' "$vcd_file")\""
    echo "[savefile] \"$save_file\""
    echo "[timestart] 0"
    echo "[size] 1024 768"
    echo "[pos] -1 -1"
    echo "[sst_width] 200"
    echo "[signals_width] 120"
    echo "[sst_expanded] 1"
    awk '
BEGIN {
  scope = ""
}

$1 == "$scope" && $2 == "module" {
  if (scope == "") {
    scope = $3
  } else if (scope != "") {
    scope = scope "." $3
  }
  next
}

$1 == "$upscope" {
  sub(/\.[^.]+$/, "", scope)
  if (scope == ".") {
    scope = ""
  }
  next
}

$1 == "$var" {
  name = $5
  if (name ~ /^\"/ && name ~ /\"$/) {
    gsub(/^\"|\"$/, "", name)
  }

  if (name != "") {
    if (scope == "") {
      print name
    } else {
      print scope "." name
    }
  }
  next
}

$1 == "$enddefinitions" {
  exit 0
}' "$vcd_file" | awk 'NF' | sort -u > "$save_file.signals"

    while IFS= read -r signal; do
      if [ -n "$signal" ]; then
        echo "@22"
        echo "$signal"
      fi
    done < "$save_file.signals"

    rm -f "$save_file.signals"

    echo "[pattern_trace] 1"
    echo "[pattern_trace] 0"
  } > "$save_file"
}

if [ "$VIEWER" = "surfer" ]; then
  top_scope=$(awk '$1=="$scope" && $2=="module" { print $3; exit }' "$VCD_FILE" || true)
  if [ -n "$top_scope" ]; then
    SURFER_CMD_FILE="$(mktemp /tmp/surfer-open-cmd.XXXXXX)"
    {
      echo "scope_add_recursive ${top_scope}"
      echo "zoom_fit"
    } > "$SURFER_CMD_FILE"
    SURFER_ARGS=(--command-file "$SURFER_CMD_FILE")
    trap 'rm -f "$SURFER_CMD_FILE"' EXIT
  fi

  if [ -n "${SURFER_BIN}" ] && [ -x "$SURFER_BIN" ]; then
    VIEWER_BIN="$SURFER_BIN"
  else
    candidates=(
      "${HOME}/.cargo/bin/surfer"
      "${HOME}/bin/surfer"
      "/opt/homebrew/bin/surfer"
      "/usr/local/bin/surfer"
      "$(command -v surfer 2>/dev/null || true)"
    )
    for candidate in "${candidates[@]}"; do
      if [ -n "$candidate" ] && [ -x "$candidate" ]; then
        VIEWER_BIN="$candidate"
        break
      fi
    done
  fi
else
  if [ -n "${GTKWAVE_BIN}" ] && [ -x "$GTKWAVE_BIN" ]; then
    VIEWER_BIN="$GTKWAVE_BIN"
  else
    candidates=(
      "${HOME}/opt/gtkwave/bin/gtkwave"
      "${HOME}/.local/bin/gtkwave"
      "/Applications/gtkwave.app/Contents/Resources/bin/gtkwave"
      "$(command -v gtkwave 2>/dev/null || true)"
    )
    for candidate in "${candidates[@]}"; do
      if [ -n "$candidate" ] && [ -x "$candidate" ]; then
        VIEWER_BIN="$candidate"
        break
      fi
    done
  fi

  GTKWAVE_SAVE_FILE="$(mktemp /tmp/gtkwave-open-gtkw.XXXXXX).gtkw"
  build_gtkwave_savefile "$VCD_FILE" "$GTKWAVE_SAVE_FILE"
  GTKWAVE_SAVE_ARGS=("$GTKWAVE_SAVE_FILE")
  trap 'rm -f "$GTKWAVE_SAVE_FILE"' EXIT
fi

if [ -z "$VIEWER_BIN" ]; then
  if [ "$VIEWER" = "surfer" ]; then
    echo "Could not find Surfer executable."
    echo "Common install paths:"
    echo "  cargo install --git https://gitlab.com/surfer-project/surfer.git surfer"
  else
    echo "Could not find gtkwave executable."
    echo "Options on macOS:"
    echo "  - Use the app: /Applications/gtkwave.app (may be deprecated/blocked)"
    echo "  - Use Homebrew: brew install --cask gtkwave"
    echo "  - Use local build:  ~/opt/gtkwave/bin/gtkwave"
  fi
  exit 1
fi

if [ "$VIEWER" = "surfer" ] && [ "${#SURFER_ARGS[@]}" -gt 0 ]; then
  exec "$VIEWER_BIN" "${SURFER_ARGS[@]}" "$VCD_FILE"
fi

if [ "$VIEWER" = "gtkwave" ] && [ "${#GTKWAVE_SAVE_ARGS[@]}" -gt 0 ]; then
  exec "$VIEWER_BIN" "$VCD_FILE" "${GTKWAVE_SAVE_ARGS[@]}"
fi

exec "$VIEWER_BIN" "$VCD_FILE"
