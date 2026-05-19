#!/usr/bin/env bash
# Stop the inference proxy and provider. The provider settles every
# open channel on SIGTERM before it exits, so the payee claims its
# earned share. The on-chain channel itself stays open — reuse it on
# the next up.sh, or close it with `soma channel request-close` +
# `soma channel withdraw` to reclaim the payer's unused deposit.
set -euo pipefail
. "$(dirname "$0")/config.sh"

[ -f "$STATE_FILE" ] || { echo "no demo state at $STATE_FILE; nothing to tear down."; exit 0; }
# shellcheck disable=SC1090
. "$STATE_FILE"

echo "→ stopping soma inference proxy (pid $PROXY_PID)..."
kill "$PROXY_PID" 2>/dev/null || true

echo "→ stopping soma inference serve (pid $PROV_PID) — settles open channels..."
kill "$PROV_PID" 2>/dev/null || true
for _ in $(seq 1 20); do
  kill -0 "$PROV_PID" 2>/dev/null || break
  sleep 0.5
done

echo
echo "=== channels for payer $PAYER ==="
"$SOMA_BIN" channel list --role payer --address "$PAYER" --indexer-url "$INDEXER_URL" || true

rm -f "$STATE_FILE"
echo "down."
