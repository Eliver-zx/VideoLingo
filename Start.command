#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
export PYTHONUTF8=1
export PYTHONWARNINGS="ignore"
PORT="${VIDEOLINGO_API_PORT:-8501}"
URL="http://127.0.0.1:${PORT}"

if [ -x "$HOME/.venvs/videolingo/bin/python" ]; then
    PY_CMD="$HOME/.venvs/videolingo/bin/python"
elif [ -x "$PWD/.venv/bin/python" ]; then
    PY_CMD="$PWD/.venv/bin/python"
else
    echo "ERROR: Python virtual environment not found (.venv/bin/python)."
    echo "Please run ./Install.command first to set up dependencies."
    exit 1
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
