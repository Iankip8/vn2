#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
  echo "Error: uv is not installed. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

# Sync environment with uv (creates venv automatically if needed)
cd "$ROOT"
uv sync --quiet

# Activate the uv-managed virtual environment
source "$ROOT/.venv/bin/activate"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
echo "✓ Activated $(python -V) with uv"
echo "  PYTHONPATH=$PYTHONPATH"

