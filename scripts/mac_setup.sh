#!/usr/bin/env bash
# One-time setup of the AI Content Factory as an always-on host on macOS, reachable over Tailscale.
# Run from the project folder:  bash scripts/mac_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
command -v brew >/dev/null || { echo "Install Homebrew first: https://brew.sh" >&2; exit 1; }

brew list ffmpeg-full >/dev/null 2>&1 || { brew install ffmpeg-full && brew link --force ffmpeg-full; }
brew list yt-dlp >/dev/null 2>&1 || brew install yt-dlp
brew list --cask tailscale >/dev/null 2>&1 || brew install --cask tailscale

[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt

[ -f .env ] || cp .env.example .env
if ! grep -q '^APP_PASSWORD=.' .env; then
  PW="$(python3 -c 'import secrets; print(secrets.token_urlsafe(14))')"
  printf '\nHOST=0.0.0.0\nAPP_PASSWORD=%s\n' "$PW" >> .env
  echo "Generated a login password (stored in .env as APP_PASSWORD): $PW"
fi

# start at login, restart if it crashes, and keep the Mac awake while it runs
PLIST="$HOME/Library/LaunchAgents/com.contentfactory.app.plist"
mkdir -p "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
 <key>Label</key><string>com.contentfactory.app</string>
 <key>ProgramArguments</key><array><string>/usr/bin/caffeinate</string><string>-s</string><string>$ROOT/run.sh</string></array>
 <key>WorkingDirectory</key><string>$ROOT</string>
 <key>EnvironmentVariables</key><dict><key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string></dict>
 <key>RunAtLoad</key><true/><key>KeepAlive</key><true/>
 <key>StandardOutPath</key><string>$ROOT/content/server.log</string><key>StandardErrorPath</key><string>$ROOT/content/server.log</string>
</dict></plist>
PL
mkdir -p content
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"

echo
echo "Done. Next: open the Tailscale app and sign in (same account as your other devices)."
echo "Then browse to  http://<this-mac's-tailscale-name>:8000  from any device (any username, password from .env)."
