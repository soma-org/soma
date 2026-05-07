#!/usr/bin/env bash
# Show on-chain channel state via the indexer (proxy is stateless on
# disk now — channel enumeration is GraphQL-driven).
set -euo pipefail
. "$(dirname "$0")/config.sh"

[ -f "$STATE_FILE" ] || { echo "no demo running." >&2; exit 1; }
# shellcheck disable=SC1090
. "$STATE_FILE"

GQL="${GRAPHQL_URL:-http://127.0.0.1:7000/graphql}"

echo "=== channels for payer $PAYER ==="
"$SOMA_BIN" channel list --role payer --address "$PAYER" --indexer-url "$GQL"
echo
echo "=== channels for payee $PROVIDER ==="
"$SOMA_BIN" channel list --role payee --address "$PROVIDER" --indexer-url "$GQL"
