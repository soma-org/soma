#!/usr/bin/env bash
# Check things are working: network, balances, processes, channels.
set -euo pipefail
. "$(dirname "$0")/config.sh"

PAYER="$(payer_address)"
PROVIDER="$(provider_address)"

echo "── network ──────────────────────────────────────────"
"$SOMA" status 2>/dev/null | grep -E "Network|Current Epoch" || true

echo
echo "── payer $PAYER ──"
"$SOMA" balance "$PAYER" 2>/dev/null | grep -E "USDC|SOMA" || true

if [ -n "$PROVIDER" ]; then
  echo
  echo "── provider $PROVIDER ──"
  "$SOMA" balance "$PROVIDER" 2>/dev/null | grep -E "USDC|SOMA" || true
fi

echo
echo "── processes ────────────────────────────────────────"
if curl -sS --max-time 2 "http://127.0.0.1:${PROVIDER_PORT}/health" >/dev/null 2>&1; then
  echo "provider : up   (http://127.0.0.1:${PROVIDER_PORT})"
else
  echo "provider : down"
fi
if curl -sS --max-time 2 "http://127.0.0.1:${PROXY_PORT}/v1/models" >/dev/null 2>&1; then
  echo "proxy    : up   (http://127.0.0.1:${PROXY_PORT})"
else
  echo "proxy    : down"
fi

echo
echo "── your payment channels ────────────────────────────"
"$SOMA" channel list --role payer --address "$PAYER" --indexer-url "$INDEXER_URL" 2>/dev/null \
  | python3 -c 'import sys, json
try: edges = json.load(sys.stdin)["data"]["channels"]["edges"]
except Exception: edges = []
if not edges:
    print("  (none)")
for e in edges:
    n = e["node"]
    print("  %s  %-10s  deposit=%s  settled=%s" % (
        n["id"], n["status"], n["deposit"], n["settledAmount"]))'
