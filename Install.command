#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
export PYTHONUTF8=1
export PYTHONWARNINGS="ignore"

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is not installed or not in PATH. Install it from https://docs.astral.sh/uv/"
    exit 1
fi

if [ -x "$HOME/.venvs/videolingo/bin/python" ]; then
    VENV_DIR="$HOME/.venvs/videolingo"
else
    VENV_DIR="$PWD/.venv"
fi
PY_CMD="$VENV_DIR/bin/python"

if [ ! -x "$PY_CMD" ]; then
    echo "[1/2] Creating Python 3.12 virtual environment at $VENV_DIR ..."
    uv venv "$VENV_DIR" --python 3.12
else
    echo "[1/2] Using existing virtual environment at $VENV_DIR ..."
fi

echo "[2/2] Installing dependencies via uv ..."
uv pip install --python "$PY_CMD" -r requirements.txt

echo ""
echo "Environment installation complete! Double-click Start.command to launch VideoLingo."
