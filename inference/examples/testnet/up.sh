#!/usr/bin/env bash
# Bring up an inference provider + proxy against the public testnet.
#
# The provider auto-registers itself AND its offering on-chain when it
# boots (`soma inference serve` reads the `[[offerings]]` block below
# and submits RegisterProvider + RegisterOffering). The proxy then
# discovers it through the testnet GraphQL indexer. A USDC payment
# channel opens lazily on the first chat request; down.sh settles it.
#
# Roles: the wallet's active address is the payer (consumer); a
# dedicated `testnet-inference-provider` wallet is the payee.
set -euo pipefail
. "$(dirname "$0")/config.sh"

if [ -f "$STATE_FILE" ]; then
  echo "demo already running — see $STATE_FILE; tear down with down.sh first." >&2
  exit 1
fi
ensure_soma_built

: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY (export it or put it in ~/autodebate/.env)}"

if ! "$SOMA_BIN" env list 2>/dev/null | grep -E '●' | grep -q testnet; then
  echo "active env is not testnet — run: soma env switch testnet" >&2
  exit 1
fi

# --- 1. Payer = the wallet's active address ----------------------------------
PAYER=$("$SOMA_BIN" wallet --json active | addr_field)
echo "→ payer (consumer) = $PAYER"

# --- 2. Dedicated provider wallet (created once, reused after) ----------------
PROVIDER=$("$SOMA_BIN" wallet --json list 2>/dev/null | python3 -c '
import sys, json
alias = sys.argv[1]
for name, addr in json.load(sys.stdin).get("addresses", []):
    if name == alias:
        print(addr if addr.startswith("0x") else "0x"+addr)
        break
' "$PROVIDER_ALIAS")

if [ -z "$PROVIDER" ]; then
  echo "→ creating provider wallet (alias $PROVIDER_ALIAS)..."
  PROVIDER=$("$SOMA_BIN" wallet new --alias "$PROVIDER_ALIAS" --json | addr_field)
  echo "→ funding it with $PROVIDER_FUND_USDC USDC (gas + settlement)..."
  "$SOMA_BIN" transfer "$PROVIDER_FUND_USDC" "$PROVIDER" --usdc >/dev/null
fi
echo "→ provider (payee) = $PROVIDER"
"$SOMA_BIN" balance "$PROVIDER" || true

# --- 3. Provider TOML --------------------------------------------------------
PROV_TOML="/tmp/soma-inference-testnet-provider.toml"
cat >"$PROV_TOML" <<EOF
[server]
listen          = "127.0.0.1:${PROVIDER_PORT}"
public_endpoint = "http://127.0.0.1:${PROVIDER_PORT}"

[backend]
kind         = "openrouter"
api_key_env  = "OPENROUTER_API_KEY"
upstream_url = "https://openrouter.ai/api/v1"

[auth]
clock_skew_tolerance_secs = 60

# serve registers this offering on-chain at boot — the pricing block
# below is the single source of truth (no separate offering register).
[[offerings]]
id              = "${DEFAULT_MODEL}"
name            = "${DEFAULT_MODEL} (OpenRouter)"
hugging_face_id = "${DEFAULT_MODEL}"
context_length  = 200000
architecture    = { input_modalities = ["text"], output_modalities = ["text"], tokenizer = "Claude" }
top_provider    = { context_length = 200000, max_completion_tokens = 4096, is_moderated = false }
supported_parameters = ["max_tokens", "temperature", "top_p", "stop", "seed"]
ttft_bound_ms   = 60000
ttot_bound_ms   = 10000
pricing = { prompt = "${PRICE_PROMPT}", completion = "${PRICE_COMPLETION}", request = "0", image = "0", input_cache_read = "0", input_cache_write = "0" }
EOF
echo "→ provider config: $PROV_TOML"

# --- 4. Provider server (registers Provider + Offering on-chain) -------------
echo "→ starting soma inference serve..."
export OPENROUTER_API_KEY
RUST_LOG="${RUST_LOG:-inference=info,sdk=info}" \
  "$SOMA_BIN" inference serve \
    --config "$PROV_TOML" \
    --address "$PROVIDER" \
    --soma-home "$PROVIDER_HOME" \
    >"$LOG_DIR/provider.log" 2>&1 &
PROV_PID=$!
for _ in $(seq 1 30); do
  curl -sS --max-time 1 "http://127.0.0.1:${PROVIDER_PORT}/health" >/dev/null 2>&1 && break
  sleep 0.5
done

# The proxy can only route once providers() returns this provider.
echo -n "  waiting for provider to land in the testnet indexer"
for _ in $(seq 1 60); do
  N=$(curl -sS --max-time 5 -X POST "$INDEXER_URL" \
        -H 'content-type: application/json' \
        -d '{"query":"{providers(first:50){edges{node{address}}}}"}' 2>/dev/null \
      | PROVIDER="$PROVIDER" python3 -c '
import sys, json, os
def norm(s): return (s[2:] if s.startswith("0x") else s).lower()
want = norm(os.environ["PROVIDER"])
try:
    edges = (json.load(sys.stdin).get("data") or {}).get("providers",{}).get("edges") or []
    print(sum(1 for e in edges if norm(e["node"]["address"]) == want))
except Exception:
    print(0)
' 2>/dev/null)
  if [ "${N:-0}" -gt 0 ]; then echo " ready"; break; fi
  sleep 2
  echo -n "."
done
echo

# --- 5. Proxy ----------------------------------------------------------------
echo "→ starting soma inference proxy..."
RUST_LOG="${RUST_LOG:-inference=info,sdk=info}" \
  "$SOMA_BIN" inference proxy \
    --address "$PAYER" \
    --listen "127.0.0.1:${PROXY_PORT}" \
    --indexer-url "$INDEXER_URL" \
    --soma-home "$PROXY_HOME" \
    >"$LOG_DIR/proxy.log" 2>&1 &
PROXY_PID=$!
for _ in $(seq 1 30); do
  curl -sS --max-time 1 "http://127.0.0.1:${PROXY_PORT}/v1/models" >/dev/null 2>&1 && break
  sleep 0.5
done

# --- 6. Persist state --------------------------------------------------------
cat >"$STATE_FILE" <<EOF
PROV_PID=${PROV_PID}
PROXY_PID=${PROXY_PID}
PAYER=${PAYER}
PROVIDER=${PROVIDER}
MODEL=${DEFAULT_MODEL}
EOF

echo
echo "READY"
echo "  payer     = $PAYER"
echo "  provider  = $PROVIDER  (http://127.0.0.1:${PROVIDER_PORT})"
echo "  proxy     = http://127.0.0.1:${PROXY_PORT}"
echo "  model     = $DEFAULT_MODEL"
echo "  indexer   = $INDEXER_URL"
echo "  logs      = $LOG_DIR"
echo
echo "send a request:  examples/testnet/chat.sh \"hello\""
echo "tear down:       examples/testnet/down.sh   # provider settles on SIGTERM"
