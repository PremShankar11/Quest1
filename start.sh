#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python; [ -x "$PY" ] || PY=.venv/Scripts/python
[ -x "$PY" ] || { echo "Create the venv first: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"; exit 1; }
( sleep 2; xdg-open http://127.0.0.1:8000 2>/dev/null || open http://127.0.0.1:8000 2>/dev/null || true ) &
exec "$PY" -m uvicorn api.main:app --app-dir backend --host 127.0.0.1 --port 8000
