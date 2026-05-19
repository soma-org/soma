#!/usr/bin/env bash
# Sourced by provider.sh / proxy.sh / chat.sh / status.sh / cleanup.sh.
# Shared paths, ports, the OpenRouter key, and a few wallet helpers.
#
# Everything is overridable from the environment, e.g.:
#   MODEL=openai/gpt-5.5 ./provider.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SOMA="${SOMA:-$ROOT/target/release/soma}"

INDEXER_URL="${INDEXER_URL:-https://graphql.testnet.soma.org/graphql}"
PROVIDER_PORT="${PROVIDER_PORT:-8444}"
PROXY_PORT="${PROXY_PORT:-11434}"
PROVIDER_HOME="${PROVIDER_HOME:-$HOME/.soma/inference-testnet-provider}"
PROXY_HOME="${PROXY_HOME:-$HOME/.soma/inference-testnet-proxy}"
PROVIDER_ALIAS="${PROVIDER_ALIAS:-testnet-inference-provider}"
MODEL="${MODEL:-anthropic/claude-haiku-4.5}"
PROVIDER_TOML="${PROVIDER_TOML:-/tmp/soma-testnet-provider.toml}"

# OpenRouter key — auto-loaded from ~/autodebate/.env if not already set.
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f "$HOME/autodebate/.env" ]; then
  OPENROUTER_API_KEY="$(grep -E '^OPENROUTER_API_KEY=' "$HOME/autodebate/.env" | head -1 \
    | cut -d= -f2- | tr -d "\"'")"
  export OPENROUTER_API_KEY
fi

# Build the soma CLI once if it isn't there yet.
ensure_soma() {
  [ -x "$SOMA" ] && return
  echo "building soma CLI (one-time, ~2 min)…" >&2
  ( cd "$ROOT" && PYO3_PYTHON=python3 cargo build --release -p cli )
}

# Print an address from a `soma wallet --json` payload, 0x-prefixed.
_norm_addr='import sys,json
v=json.load(sys.stdin)
a=v if isinstance(v,str) else v["address"]
print(a if a.startswith("0x") else "0x"+a)'

# The payer = the wallet's active address.
payer_address() { "$SOMA" wallet --json active | python3 -c "$_norm_addr"; }

# The provider address for $PROVIDER_ALIAS — empty if it doesn't exist yet.
provider_address() {
  "$SOMA" wallet --json list | PROVIDER_ALIAS="$PROVIDER_ALIAS" python3 -c 'import sys,json,os
alias=os.environ["PROVIDER_ALIAS"]
for name,addr in json.load(sys.stdin).get("addresses",[]):
    if name==alias:
        print(addr if addr.startswith("0x") else "0x"+addr); break'
}
