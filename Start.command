#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONUTF8=1
export PYTHONWARNINGS="ignore"
PORT="${VIDEOLINGO_API_PORT:-8501}"
URL="http://127.0.0.1:${PORT}"

PY_CMD=""
for candidate in \
    "$HOME/.venvs/videolingo/bin/python" \
    "$PWD/.venv/bin/python" \
    "$HOME/miniconda3/bin/python3" \
    "$HOME/anaconda3/bin/python3" \
    "/opt/homebrew/anaconda3/bin/python3" \
    "/opt/homebrew/Caskroom/miniconda/base/bin/python3" \
    "$(command -v python3 2>/dev/null || true)"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
        if "$candidate" -c "import fastapi, uvicorn, openai, ruamel.yaml" >/dev/null 2>&1; then
            PY_CMD="$candidate"
            break
        fi
    fi
done

if [ -z "$PY_CMD" ]; then
    echo "Installing VideoLingo environment..."
    python3 setup_env.py --yes
    PY_CMD="$PWD/.venv/bin/python"
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
