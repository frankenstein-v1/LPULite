#!/usr/bin/env bash
set -euo pipefail

printf "LPULite Verilog toolchain setup\n"

has_required_tools() {
  command -v iverilog >/dev/null 2>&1 && \
  command -v verilator >/dev/null 2>&1 && \
  command -v yosys >/dev/null 2>&1
}

ensure_switch_module() {
  if ! command -v gtkwave >/dev/null 2>&1; then
    return
  fi

  if perl -MSwitch -e '1' >/dev/null 2>&1; then
    return
  fi

  local_perl="${HOME}/perl5/lib/perl5"
  export PERL5LIB="${local_perl}${PERL5LIB:+:${PERL5LIB}}"

  if perl -MSwitch -e '1' >/dev/null 2>&1; then
    return
  fi

  if command -v cpan >/dev/null 2>&1; then
    echo "Installing Perl Switch module required by GTKWave launcher"
    PERL_MM_OPT="INSTALL_BASE=${HOME}/perl5" \
    PERL_MB_OPT="--install_base ${HOME}/perl5" \
    cpan -T -i Switch
  fi

  if ! perl -MSwitch -e '1' >/dev/null 2>&1; then
    echo "Could not install Switch.pm automatically."
    echo "Set PERL5LIB manually if needed:"
    echo "  export PERL5LIB=\"${HOME}/perl5/lib/perl5:\${PERL5LIB:-}\""
  fi
}

if has_required_tools; then
  echo "iverilog, verilator, and yosys are already installed."
  ensure_switch_module
  exit 0
fi

OS="$(uname -s)"

case "$OS" in
  Darwin)
    if ! command -v brew >/dev/null 2>&1; then
      echo "Homebrew not found. Install Homebrew first: https://brew.sh/"
      exit 1
    fi

    echo "Installing with Homebrew..."
    brew update

    set +e
    brew install icarus-verilog verilator yosys
    result=$?
    set -e

    if [ "$result" -ne 0 ]; then
      echo "Installing fallback package names..."
      brew install iverilog verilator yosys
    fi

    # cask install may be quarantined from source; disable quarantine after install
    brew install --cask --no-quarantine gtkwave || brew install gtkwave
    if [ -d "/Applications/gtkwave.app" ]; then
      xattr -dr com.apple.quarantine /Applications/gtkwave.app 2>/dev/null || true
    fi
    ;;
  Linux)
    if command -v apt-get >/dev/null 2>&1; then
      echo "Installing with apt..."
      sudo apt-get update
      sudo apt-get install -y iverilog verilator yosys gtkwave
    elif command -v dnf >/dev/null 2>&1; then
      echo "Installing with dnf..."
      sudo dnf install -y iverilog verilator yosys gtkwave
    elif command -v pacman >/dev/null 2>&1; then
      echo "Installing with pacman..."
      sudo pacman -Syu --noconfirm iverilog verilator yosys gtkwave
    else
      echo "No supported package manager detected. Install the tools manually:"
      echo "  - iverilog"
      echo "  - verilator"
      echo "  - yosys"
      echo "  - gtkwave"
      exit 1
    fi
    ;;
  *)
    echo "Unsupported OS: $OS"
    echo "Please install these tools manually:"
    echo "  - iverilog"
    echo "  - verilator"
    echo "  - yosys"
    echo "  - gtkwave"
    exit 1
    ;;
esac

ensure_switch_module

echo "Verifying installation..."
for bin in iverilog verilator yosys gtkwave vvp; do
  if command -v "$bin" >/dev/null 2>&1; then
    echo "  ok: $bin"
  else
    echo "  missing: $bin"
  fi
done

echo "Done."
