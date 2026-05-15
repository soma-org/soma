#!/usr/bin/env bash
# Sourced by up.sh / chat.sh / down.sh — paths, ports, env-loading.
#
# Substitution to testnet: set SOMA_ENV=testnet (and provide a
# client.yaml whose `localnet` env is replaced with `testnet`). The
# inference binaries don't care which env is active — they reuse the
# wallet's active env via WalletContext.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SOMA_BIN="${SOMA_BIN:-$ROOT/target/release/soma}"
# `soma_config_dir()` returns ~/.soma/soma_config — that's where the
# wallet's client.yaml + soma.keystore live, and where we'll keep
# the inference registry + per-channel state too.
SOMA_HOME="${SOMA_HOME:-$HOME/.soma/soma_config}"
STATE_FILE="${STATE_FILE:-$HOME/.soma/inference-localnet-demo.state}"
LOG_DIR="${LOG_DIR:-$HOME/.soma/inference-localnet-demo-logs}"
mkdir -p "$LOG_DIR"

# Backend selection — `BACKEND=openrouter` (default) or `BACKEND=vast`.
BACKEND="${BACKEND:-openrouter}"

# Auto-source whichever .env carries the API key for the chosen backend.
load_env_file() {
  local file="$1"
  [ -f "$file" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|\#*) continue;; esac
    key="${line%%=*}"; val="${line#*=}"
    val="${val#\"}"; val="${val%\"}"; val="${val#\'}"; val="${val%\'}"
    # Only set if not already in env (caller can override).
    [ -z "${!key:-}" ] && export "$key=$val"
  done < "$file"
}

case "$BACKEND" in
  openrouter)
    # The user keeps their OpenRouter key in ~/autodebate/.env.
    load_env_file "$HOME/autodebate/.env"
    : "${OPENROUTER_API_KEY:?BACKEND=openrouter requires OPENROUTER_API_KEY (looked in ~/autodebate/.env)}"
    BACKEND_API_KEY_ENV="OPENROUTER_API_KEY"
    BACKEND_UPSTREAM_URL="${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"
    DEFAULT_MODEL="${MODEL:-google/gemma-4-26b-a4b-it:free}"
    ;;
  vast)
    load_env_file "$ROOT/inference/examples/vast/.env"
    : "${VAST_API_KEY:?BACKEND=vast requires VAST_API_KEY (looked in inference/examples/vast/.env)}"
    : "${VAST_UPSTREAM_URL:?BACKEND=vast requires VAST_UPSTREAM_URL (run examples/vast/up.sh first or set explicitly)}"
    BACKEND_API_KEY_ENV="VAST_API_KEY"
    BACKEND_UPSTREAM_URL="$VAST_UPSTREAM_URL"
    DEFAULT_MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
    ;;
  *)
    echo "unknown BACKEND=$BACKEND (expected 'openrouter' or 'vast')" >&2
    exit 1
    ;;
esac

# Pricing for the demo offering (USDC-microdollar per token).
# Cheap & uniform — exact value doesn't matter; we only verify
# accounting works.
DEMO_PRICE_PROMPT="${DEMO_PRICE_PROMPT:-0.0000002}"
DEMO_PRICE_COMPLETION="${DEMO_PRICE_COMPLETION:-0.0000004}"

# Fixed ports.
LOCALNET_RPC_PORT=9000
PROVIDER_PORT=8444
PROXY_PORT=11434

# Build the soma CLI if needed. Honour the existing release vs debug
# env (set SOMA_BIN explicitly to skip).
ensure_soma_built() {
  if [ -x "$SOMA_BIN" ]; then return; fi
  echo "→ building soma CLI (release)..." >&2
  (cd "$ROOT" && PYO3_PYTHON=python3 cargo build --release -p cli)
}
