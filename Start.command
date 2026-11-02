#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
export PYTHONUTF8=1
export PYTHONWARNINGS="ignore"
PORT="${VIDEOLINGO_API_PORT:-8501}"
URL="http://127.0.0.1:${PORT}"

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
fi

if ! "$PY_CMD" -c "import fastapi, multipart, uvicorn, openai, ruamel.yaml, demucs, rapidfuzz; import mlx_audio" >/dev/null 2>&1 && \
   ! "$PY_CMD" -c "import fastapi, multipart, uvicorn, openai, ruamel.yaml, demucs, rapidfuzz; import qwen_asr" >/dev/null 2>&1; then
    echo "[2/2] Installing dependencies via uv ..."
    uv pip install --python "$PY_CMD" -r requirements.txt
fi

echo "Using Python: $PY_CMD"
echo "Starting VideoLingo WebUI at $URL ..."

(
    for _ in $(seq 1 50); do
        if nc -z 127.0.0.1 "$PORT" >/dev/null 2>&1; then
            open "$URL" >/dev/null 2>&1 || true
            exit 0
        fi
        sleep 0.2
    done
) &

exec "$PY_CMD" api.py
