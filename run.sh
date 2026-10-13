#!/usr/bin/env bash
# Start the AI Content Factory:  ./run.sh   then open http://127.0.0.1:8000
cd "$(dirname "$0")"
PY=python3; [ -x .venv/bin/python3 ] && PY=.venv/bin/python3; [ -x .venv/bin/python ] && PY=.venv/bin/python
exec "$PY" -m uvicorn factory.app:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
