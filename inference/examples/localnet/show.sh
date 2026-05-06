#!/usr/bin/env bash
# Show on-chain state of the channel the proxy opened with the
# provider. Reads the proxy's pointer file (channels-by-provider) to
# find the ObjectID, then `soma channel show` it.
set -euo pipefail
. "$(dirname "$0")/config.sh"

[ -f "$STATE_FILE" ] || { echo "no demo running." >&2; exit 1; }
# shellcheck disable=SC1090
. "$STATE_FILE"

# SomaAddress display strips the leading "0x" — the pointer file name
# matches the raw hex.
PROVIDER_HEX="${PROVIDER#0x}"
POINTER="$SOMA_HOME/client/channels-by-provider/${PROVIDER_HEX}.txt"
if [ ! -f "$POINTER" ]; then
  echo "no channel opened yet — run examples/localnet/chat.sh first." >&2
  exit 1
fi
CHANNEL_ID=$(cat "$POINTER")
echo "channel = $CHANNEL_ID"
"$SOMA_BIN" channel show --channel-id "$CHANNEL_ID"
