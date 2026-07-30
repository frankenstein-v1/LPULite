#!/usr/bin/env bash
set -euo pipefail

# Edit these if your repo default branch differs
REMOTE_URL="https://github.com/sakshambatra1/tinyLPU.git"
BRANCH="main"

cd "$(cd "$(dirname "$0")/.." && pwd)"

# Optional cleanup before commit
rm -f results.xml

if [ ! -d .git ]; then
  git init
fi

# Add remote if missing
if ! git remote get-url origin >/dev/null 2>&1; then
  git remote add origin "$REMOTE_URL"
else
  git remote set-url origin "$REMOTE_URL"
fi

git add .gitignore Makefile README.md scripts src

git commit -m "Initial project snapshot"

git branch -M "$BRANCH"

git push -u origin "$BRANCH"
