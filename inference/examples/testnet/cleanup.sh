#!/usr/bin/env bash
# Tear down: stop the provider + proxy, then close your payment
# channels and reclaim their deposits.
#
# Channels have a ~10-minute close grace period, so this is two-phase
# and converges — safe to re-run:
#   1st run : stops the processes, begins the close timer on each channel
#   2nd run : (after ~10 min) withdraws each channel's remaining deposit
set -euo pipefail
. "$(dirname "$0")/config.sh"
PAYER="$(payer_address)"

echo "── stopping processes ──"
pkill -f 'soma start proxy' 2>/dev/null && echo "proxy stopped"            || echo "proxy not running"
pkill -f 'soma start provider' 2>/dev/null && echo "provider stopping (settling open channels)…" || echo "provider not running"
for _ in $(seq 1 10); do pgrep -f 'soma start provider' >/dev/null 2>&1 || break; sleep 1; done

echo
echo "── channels ──"
CHANNELS="$("$SOMA" channel list --role payer --address "$PAYER" --indexer-url "$INDEXER_URL" 2>/dev/null \
  | python3 -c 'import sys, json
try: edges = json.load(sys.stdin)["data"]["channels"]["edges"]
except Exception: edges = []
for e in edges:
    n = e["node"]
    print(n["id"], n["status"], "closing" if n.get("closeRequestedAtMs") else "open")')"

if [ -z "$CHANNELS" ]; then
  echo "(no channels)"
else
  PENDING=0
  while read -r ID STATUS PHASE; do
    [ -z "$ID" ] && continue
    case "$STATUS:$PHASE" in
      WITHDRAWN:*)
        echo "  ${ID}  already closed" ;;
      *:open)
        if "$SOMA" channel request-close --channel-id "$ID" >/dev/null 2>&1; then
          echo "  ${ID}  close timer started"
        else
          echo "  ${ID}  request-close failed"
        fi
        PENDING=1 ;;
      *:closing)
        if "$SOMA" channel withdraw --channel-id "$ID" >/dev/null 2>&1; then
          echo "  ${ID}  withdrawn — deposit reclaimed"
        else
          echo "  ${ID}  still in grace period"
          PENDING=1
        fi ;;
    esac
  done <<< "$CHANNELS"
  echo
  if [ "$PENDING" = 1 ]; then
    echo "↻ channels still closing — wait ~10 min, then run ./cleanup.sh again"
  else
    echo "✓ all channels closed, deposits reclaimed"
  fi
fi

rm -f "$PROVIDER_TOML"
echo "done."
