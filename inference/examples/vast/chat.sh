#!/usr/bin/env bash
# Send a chat completion through `soma inference proxy`. Starts the proxy
# lazily if it isn't already listening on 127.0.0.1:11434.
# Usage: chat.sh "your prompt"          (default: "hello")
# Env:   MAX_TOKENS=1024
set -euo pipefail

ENV_FILE="$(dirname "$0")/.env"
if [ -f "$ENV_FILE" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|\#*) continue;; esac
    key="${line%%=*}"; val="${line#*=}"
    val="${val#\"}"; val="${val%\"}"; val="${val#\'}"; val="${val%\'}"
    [ -z "${!key:-}" ] && export "$key=$val"
  done < "$ENV_FILE"
fi

STATE="$HOME/.soma/inference-vast-demo.state"
[ -f "$STATE" ] || { echo "no demo running. Run examples/vast/up.sh first." >&2; exit 1; }
# shellcheck disable=SC1090
. "$STATE"

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CLI_PID_FILE="$HOME/.soma/inference-vast-demo.proxy.pid"

# Ensure the proxy is up.
if ! curl -sS --max-time 1 http://127.0.0.1:11434/v1/models >/dev/null 2>&1; then
  [ -x "$ROOT/target/release/soma" ] || \
    (cd "$ROOT" && PYO3_PYTHON=python3 cargo build --release -p cli)

  PROXY_TOML=/tmp/soma-inference-vast-demo-proxy.toml
  cat >"$PROXY_TOML" <<'EOF'
[listen]
addr = "127.0.0.1:11434"

[chain]
mode      = "file"
soma_home = "~/.soma"

[wallet]
default_deposit_micros = 5_000_000
channel_expires_secs   = 86_400

[discovery]
provider_cache_ttl_secs = 30
EOF

  ADDR_FLAG=()
  [ -n "${SOMA_ADDRESS:-}" ] && ADDR_FLAG=(--address "$SOMA_ADDRESS")

  "$ROOT/target/release/soma" inference proxy --config "$PROXY_TOML" "${ADDR_FLAG[@]}" \
    >/tmp/soma-inference-proxy.log 2>&1 &
  echo $! >"$CLI_PID_FILE"
  for i in $(seq 1 15); do
    curl -sS --max-time 1 http://127.0.0.1:11434/v1/models >/dev/null 2>&1 && break
    sleep 1
  done
fi

PROMPT="${*:-hello}"
MAX_TOKENS="${MAX_TOKENS:-1024}"

PAYLOAD=$(MODEL="$MODEL" PROMPT="$PROMPT" MAX_TOKENS="$MAX_TOKENS" python3 -c '
import os, json
print(json.dumps({
  "model": os.environ["MODEL"],
  "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
  "max_tokens": int(os.environ["MAX_TOKENS"]),
  "stream": True,
}))')

if [ "${RAW:-}" = "1" ]; then
  curl -sS -N -X POST http://127.0.0.1:11434/v1/chat/completions \
    -H 'content-type: application/json' -d "$PAYLOAD"
else
  curl -sS -N -X POST http://127.0.0.1:11434/v1/chat/completions \
    -H 'content-type: application/json' -d "$PAYLOAD" \
  | python3 -u -c '
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("data:"): continue
    payload = line[5:].lstrip()
    if payload == "[DONE]":
        sys.stdout.write("\n")
        break
    try:
        d = json.loads(payload)
    except Exception:
        continue
    for c in d.get("choices", []):
        chunk = (c.get("delta") or {}).get("content") or ""
        if chunk:
            sys.stdout.write(chunk)
            sys.stdout.flush()
'
fi
