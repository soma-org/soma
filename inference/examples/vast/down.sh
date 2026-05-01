#!/usr/bin/env bash
# Stop `soma inference {serve,proxy}` (the proxy may have been started
# lazily by chat.sh) and destroy the rented Vast.ai instance.
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

: "${VAST_API_KEY:?VAST_API_KEY must be set (export it or put it in examples/vast/.env)}"
STATE="$HOME/.soma/inference-vast-demo.state"
CLI_PID_FILE="$HOME/.soma/inference-vast-demo.proxy.pid"

[ -f "$STATE" ] || { echo "no demo state at $STATE; nothing to tear down."; exit 0; }
# shellcheck disable=SC1090
. "$STATE"

echo "→ stopping soma inference processes..."
[ -n "${PROV_PID:-}" ] && kill "$PROV_PID" 2>/dev/null && echo "  killed serve pid=$PROV_PID" || true
if [ -f "$CLI_PID_FILE" ]; then
  CLI_PID=$(cat "$CLI_PID_FILE")
  kill "$CLI_PID" 2>/dev/null && echo "  killed proxy pid=$CLI_PID" || true
  rm -f "$CLI_PID_FILE"
fi

echo "→ destroying Vast instance ${INSTANCE_ID}..."
curl -sS -X DELETE -H "Authorization: Bearer $VAST_API_KEY" \
  "https://console.vast.ai/api/v0/instances/${INSTANCE_ID}/" \
  -w "\n  HTTP %{http_code}\n"

rm -f "$STATE"
echo "down."
