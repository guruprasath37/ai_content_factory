#!/usr/bin/env bash
# Start the AI Content Factory:  ./run.sh   then open http://127.0.0.1:8000
# To reach it from other devices (e.g. over Tailscale) set HOST=0.0.0.0 and APP_PASSWORD in .env.
cd "$(dirname "$0")"
PY=python3; [ -x .venv/bin/python3 ] && PY=.venv/bin/python3; [ -x .venv/bin/python ] && PY=.venv/bin/python
cfg() { "$PY" -c "import os; from dotenv import dotenv_values as d; print(os.environ.get('$1') or d('.env').get('$1') or '$2')"; }
HOST=$(cfg HOST 127.0.0.1); PORT=$(cfg PORT 8000); PW=$(cfg APP_PASSWORD "")
case "$HOST" in 127.0.0.1|localhost|::1) ;; *) [ -z "$PW" ] && { echo "Refusing to listen on $HOST without APP_PASSWORD set in .env" >&2; exit 1; };; esac
exec "$PY" -m uvicorn factory.app:app --host "$HOST" --port "$PORT"
