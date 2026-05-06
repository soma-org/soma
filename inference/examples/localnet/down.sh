#!/usr/bin/env bash
# Tear down: SIGTERM the provider (triggers on-chain Settle for every
# open channel), then the proxy, then the localnet.
set -euo pipefail
. "$(dirname "$0")/config.sh"

[ -f "$STATE_FILE" ] || { echo "no demo running." ; exit 0; }
# shellcheck disable=SC1090
. "$STATE_FILE"

stop() {
  local name="$1" pid="$2"
  if [ -z "${pid:-}" ]; then return; fi
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    # Provider takes longer — it submits Settle txs.
    for _ in $(seq 1 20); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.5
    done
    if kill -0 "$pid" 2>/dev/null; then
      echo "  $name pid=$pid still alive after 10s — SIGKILL"
      kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
  echo "  stopped $name (pid=$pid)"
}

echo "→ stopping provider (settles each channel on the way out)..."
stop provider "${PROV_PID:-}"
echo "→ stopping proxy..."
stop proxy "${PROXY_PID:-}"
echo "→ stopping localnet..."
stop localnet "${LOCALNET_PID:-}"

rm -f "$STATE_FILE"
echo "down."
