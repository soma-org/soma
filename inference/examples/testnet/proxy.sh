#!/usr/bin/env bash
# TERMINAL 2 — run the local inference proxy.
#
# Point any OpenAI-compatible client at http://127.0.0.1:11434. The
# proxy discovers providers through the testnet indexer, signs each
# request, and opens a USDC payment channel on the first call.
# Ctrl-C to stop.
set -euo pipefail
. "$(dirname "$0")/config.sh"
ensure_soma

PAYER="$(payer_address)"
echo "→ payer $PAYER"
echo "→ proxy  http://127.0.0.1:${PROXY_PORT}"
echo "  Ctrl-C to stop."
echo
EXTRA_ARGS=()
if [ "${INCLUDE_UNTRUSTED:-0}" = "1" ]; then
  EXTRA_ARGS+=(--include-untrusted)
fi

exec env RUST_LOG="${RUST_LOG:-inference=info,sdk=info}" \
  "$SOMA" proxy \
    --address "$PAYER" \
    --listen "127.0.0.1:${PROXY_PORT}" \
    --indexer-url "$INDEXER_URL" \
    --soma-home "$PROXY_HOME" \
    --prewarm-model "$MODEL" \
    "${EXTRA_ARGS[@]}"
