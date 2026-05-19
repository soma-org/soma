#!/usr/bin/env bash
# Sourced by up.sh / chat.sh / down.sh — paths, ports, env-loading.
#
# Runs an inference provider + proxy against the public testnet. Unlike
# the localnet example, nothing chain-side is started locally: the RPC
# fullnode and the GraphQL indexer are already hosted. Your wallet's
# active env must be `testnet` (`soma env switch testnet`).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SOMA_BIN="${SOMA_BIN:-$ROOT/target/release/soma}"
STATE_FILE="${STATE_FILE:-$HOME/.soma/inference-testnet-demo.state}"
LOG_DIR="${LOG_DIR:-$HOME/.soma/inference-testnet-demo-logs}"
mkdir -p "$LOG_DIR"

# Public testnet GraphQL indexer — provider discovery + channel
# enumeration. Override INDEXER_URL to point at a private indexer.
INDEXER_URL="${INDEXER_URL:-https://graphql.testnet.soma.org/graphql}"

# Local ports for the two inference services.
PROVIDER_PORT="${PROVIDER_PORT:-8444}"
PROXY_PORT="${PROXY_PORT:-11434}"

# Per-service soma-home (ledger + client state) — kept apart from the
# wallet config dir so a stale ledger never bleeds across runs.
PROVIDER_HOME="${PROVIDER_HOME:-$HOME/.soma/inference-testnet-provider}"
PROXY_HOME="${PROXY_HOME:-$HOME/.soma/inference-testnet-proxy}"

# Alias of the dedicated provider wallet; up.sh creates it on first run.
PROVIDER_ALIAS="${PROVIDER_ALIAS:-testnet-inference-provider}"

# OpenRouter backend — key is sourced from ~/autodebate/.env if it
# isn't already exported. up.sh hard-requires it; chat.sh/down.sh
# don't, so the assertion lives there, not here.
load_env_file() {
  local file="$1"
  [ -f "$file" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|\#*) continue;; esac
    key="${line%%=*}"; val="${line#*=}"
    val="${val#\"}"; val="${val%\"}"; val="${val#\'}"; val="${val%\'}"
    [ -z "${!key:-}" ] && export "$key=$val"
  done < "$file"
}
load_env_file "$HOME/autodebate/.env"

DEFAULT_MODEL="${MODEL:-anthropic/claude-haiku-4.5}"

# Demo offering pricing — USD per token, the provider's choice. `serve`
# converts these to on-chain micros and registers the offering itself.
PRICE_PROMPT="${PRICE_PROMPT:-0.000001}"
PRICE_COMPLETION="${PRICE_COMPLETION:-0.000005}"

# USDC sent to a freshly-created provider wallet for gas + Settle txs.
PROVIDER_FUND_USDC="${PROVIDER_FUND_USDC:-50}"

ensure_soma_built() {
  if [ -x "$SOMA_BIN" ]; then return; fi
  echo "→ building soma CLI (release)..." >&2
  (cd "$ROOT" && PYO3_PYTHON=python3 cargo build --release -p cli)
}

# Read a wallet address from stdin JSON, normalized to a 0x prefix.
# Accepts either a bare string (`wallet --json active`) or an object
# with an `address` field (`wallet new --json`).
addr_field() {
  python3 -c 'import sys,json
v=json.load(sys.stdin)
a=v if isinstance(v,str) else v["address"]
print(a if a.startswith("0x") else "0x"+a)'
}
