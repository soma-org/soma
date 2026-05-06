#!/usr/bin/env bash
# Bring up: localnet + provider + proxy. Lazily opens an on-chain
# channel on first request. Settles on `down.sh`.
#
# Variants:
#   BACKEND=openrouter  ./up.sh   (default — needs OPENROUTER_API_KEY in ~/autodebate/.env)
#   BACKEND=vast        ./up.sh   (needs VAST_API_KEY in inference/examples/vast/.env
#                                  AND VAST_UPSTREAM_URL exported by an active
#                                  `inference/examples/vast/up.sh` rental)
set -euo pipefail
. "$(dirname "$0")/config.sh"

if [ -f "$STATE_FILE" ]; then
  echo "demo already running — see $STATE_FILE; tear down with down.sh first." >&2
  exit 1
fi

ensure_soma_built

# --- 1. Start localnet --------------------------------------------------------
# We don't use --force-regenesis because the tempdir keystore wouldn't
# be reachable from `soma keytool list`. Instead we wipe the soma
# config dir first so the next boot triggers a fresh genesis that
# persists at $SOMA_HOME (= ~/.soma/soma_config).
if [ -e "$SOMA_HOME/network.yaml" ] && [ "${REUSE:-0}" != "1" ]; then
  echo "  wiping $SOMA_HOME for fresh genesis (set REUSE=1 to keep)..."
  rm -rf "$SOMA_HOME"
fi

echo "→ booting localnet..."
"$SOMA_BIN" start localnet --epoch-duration-ms 60000 \
    >"$LOG_DIR/localnet.log" 2>&1 &
LOCALNET_PID=$!

echo -n "  waiting for RPC at 127.0.0.1:$LOCALNET_RPC_PORT"
for _ in $(seq 1 200); do
  if curl -sS --max-time 1 "http://127.0.0.1:${LOCALNET_RPC_PORT}/" >/dev/null 2>&1; then
    # /metrics responds even if the gRPC layer isn't fully up yet —
    # also check that we can list keys (which needs client.yaml).
    if [ -f "$SOMA_HOME/client.yaml" ] && [ -f "$SOMA_HOME/soma.keystore" ]; then
      echo " ready"
      break
    fi
  fi
  sleep 0.5
  echo -n "."
done

if [ ! -f "$SOMA_HOME/soma.keystore" ]; then
  echo
  echo "  localnet didn't write $SOMA_HOME/soma.keystore — see $LOG_DIR/localnet.log"
  exit 1
fi

# Localnet seeds ~/.soma/soma_config/soma.keystore with funded
# accounts at boot. Read them out via JSON.
list_addrs() {
  "$SOMA_BIN" keytool list 2>/dev/null \
    | python3 -c 'import sys, json
data = json.load(sys.stdin)
for e in data:
    print("0x" + e["somaAddress"])'
}
ADDRS=( $(list_addrs) )
if [ "${#ADDRS[@]}" -lt 2 ]; then
  echo "  keystore has ${#ADDRS[@]} addresses; need ≥2 — generating..." >&2
  while [ "${#ADDRS[@]}" -lt 2 ]; do
    "$SOMA_BIN" keytool generate ed25519 >/dev/null
    ADDRS=( $(list_addrs) )
  done
fi
PAYER="${ADDRS[0]}"
PROVIDER="${ADDRS[1]}"

echo "  payer    = $PAYER"
echo "  provider = $PROVIDER"

# Fund the provider with USDC so it has gas (channel ops are payer-
# debited but the provider needs gas for Settle).
echo "→ funding provider with 100 USDC for settlement gas..."
"$SOMA_BIN" transfer 100 "$PROVIDER" --usdc --gas-budget 10000000 >/dev/null 2>&1 || \
  echo "  (transfer skipped — provider may already be funded)"

# --- 2. Write provider TOML --------------------------------------------------
PROV_TOML="/tmp/soma-inference-localnet-provider.toml"
cat >"$PROV_TOML" <<EOF
[server]
listen          = "127.0.0.1:${PROVIDER_PORT}"
public_endpoint = "http://127.0.0.1:${PROVIDER_PORT}"

[backend]
kind         = "${BACKEND}"
api_key_env  = "${BACKEND_API_KEY_ENV}"
upstream_url = "${BACKEND_UPSTREAM_URL}"

[auth]
clock_skew_tolerance_secs = 60

[[offerings]]
id              = "${DEFAULT_MODEL}"
name            = "${DEFAULT_MODEL} (${BACKEND})"
hugging_face_id = "${DEFAULT_MODEL}"
context_length  = 8192
architecture    = { input_modalities = ["text"], output_modalities = ["text"], tokenizer = "Qwen", instruct_type = "chatml" }
top_provider    = { context_length = 8192, max_completion_tokens = 4096, is_moderated = false }
supported_parameters = ["max_tokens","temperature","top_p","stop","seed"]
pricing = { prompt = "${DEMO_PRICE_PROMPT}", completion = "${DEMO_PRICE_COMPLETION}", request = "0", image = "0", input_cache_read = "0", input_cache_write = "0" }
EOF
echo "→ provider config: $PROV_TOML"

# --- 3. Start the provider server --------------------------------------------
echo "→ starting soma inference serve --address $PROVIDER..."
"$SOMA_BIN" inference serve \
    --config "$PROV_TOML" \
    --address "$PROVIDER" \
    >"$LOG_DIR/provider.log" 2>&1 &
PROV_PID=$!
for _ in $(seq 1 30); do
  if curl -sS --max-time 1 "http://127.0.0.1:${PROVIDER_PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.3
done

# Manually register the provider into the local registry so the
# proxy's discovery can find it. (One PR away from being on-chain.)
PROVIDER_REG="$SOMA_HOME/registry/providers/${PROVIDER}.json"
mkdir -p "$(dirname "$PROVIDER_REG")"
cat >"$PROVIDER_REG" <<EOF
{
  "address": "${PROVIDER}",
  "pubkey_hex": "",
  "endpoint": "http://127.0.0.1:${PROVIDER_PORT}",
  "last_heartbeat_ms": $(date +%s)000
}
EOF
echo "  registry entry: $PROVIDER_REG"

# --- 4. Start the proxy ------------------------------------------------------
echo "→ starting soma inference proxy --address $PAYER..."
"$SOMA_BIN" inference proxy \
    --address "$PAYER" \
    --listen "127.0.0.1:${PROXY_PORT}" \
    >"$LOG_DIR/proxy.log" 2>&1 &
PROXY_PID=$!
for _ in $(seq 1 30); do
  if curl -sS --max-time 1 "http://127.0.0.1:${PROXY_PORT}/v1/models" >/dev/null 2>&1; then
    break
  fi
  sleep 0.3
done

# --- 5. Persist state --------------------------------------------------------
cat >"$STATE_FILE" <<EOF
LOCALNET_PID=${LOCALNET_PID}
PROV_PID=${PROV_PID}
PROXY_PID=${PROXY_PID}
PAYER=${PAYER}
PROVIDER=${PROVIDER}
MODEL=${DEFAULT_MODEL}
BACKEND=${BACKEND}
EOF

echo
echo "READY ($BACKEND)"
echo "  payer        = $PAYER"
echo "  provider     = $PROVIDER  (http://127.0.0.1:${PROVIDER_PORT})"
echo "  proxy        = http://127.0.0.1:${PROXY_PORT}"
echo "  model        = $DEFAULT_MODEL"
echo "  state        = $STATE_FILE"
echo "  logs         = $LOG_DIR"
echo
echo "send a request:  examples/localnet/chat.sh \"hello\""
echo "show channel:    examples/localnet/show.sh"
echo "tear down:       examples/localnet/down.sh   # provider settles on SIGTERM"
