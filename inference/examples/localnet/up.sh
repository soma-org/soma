#!/usr/bin/env bash
# Bring up: localnet + indexer (Postgres + soma-indexer-alt + soma-graphql) +
# provider + proxy. Lazily opens an on-chain channel on first request.
# Settles on `down.sh`.
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

# --- 0. Build indexer + graphql binaries (cached after first run) -------------
ensure_built() {
  local pkg="$1"; local bin="$2"
  if [ -x "$ROOT/target/release/$bin" ]; then return; fi
  echo "→ building $pkg (release)..." >&2
  (cd "$ROOT" && PYO3_PYTHON=python3 cargo build --release -p "$pkg")
}
ensure_built indexer-alt indexer-alt
ensure_built soma-graphql soma-graphql

# --- 1. Postgres --------------------------------------------------------------
PG_DATA="${PG_DATA:-$HOME/.soma/inference-localnet-pg}"
PG_PORT="${PG_PORT:-5447}"
PG_DB="${PG_DB:-soma_localnet}"
PG_URL="postgres://${USER}@127.0.0.1:${PG_PORT}/${PG_DB}"

if [ ! -d "$PG_DATA/global" ]; then
  echo "→ initializing Postgres data dir at $PG_DATA..."
  initdb -D "$PG_DATA" --auth-local=trust --auth-host=trust --username="$USER" >/dev/null
fi

if pg_isready -p "$PG_PORT" -h 127.0.0.1 >/dev/null 2>&1; then
  echo "→ Postgres on :$PG_PORT already up; reusing."
else
  echo "→ starting Postgres on :$PG_PORT..."
  pg_ctl -D "$PG_DATA" -l "$LOG_DIR/postgres.log" \
    -o "-p $PG_PORT -c unix_socket_directories=$PG_DATA -c listen_addresses=127.0.0.1" \
    start
fi
createdb -p "$PG_PORT" -h 127.0.0.1 "$PG_DB" 2>/dev/null || true

# --- 2. Start localnet --------------------------------------------------------
if [ -e "$SOMA_HOME/network.yaml" ] && [ "${REUSE:-0}" != "1" ]; then
  echo "  wiping $SOMA_HOME for fresh genesis (set REUSE=1 to keep)..."
  rm -rf "$SOMA_HOME"
fi

INGEST_DIR="$HOME/.soma/inference-localnet-ingestion"
rm -rf "$INGEST_DIR" && mkdir -p "$INGEST_DIR"

echo "→ booting localnet (with data ingestion, 5min epochs)..."
# 5-minute epochs keep the localnet from reconfiguring mid-test —
# user txs submitted right around an epoch boundary can otherwise
# stall on committee handoff (see down.sh for the symptom).
"$SOMA_BIN" start localnet --epoch-duration-ms 300000 \
    --data-ingestion-dir "$INGEST_DIR" \
    >"$LOG_DIR/localnet.log" 2>&1 &
LOCALNET_PID=$!

echo -n "  waiting for RPC at 127.0.0.1:$LOCALNET_RPC_PORT"
for _ in $(seq 1 200); do
  if curl -sS --max-time 1 "http://127.0.0.1:${LOCALNET_RPC_PORT}/" >/dev/null 2>&1; then
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

# Fund the provider with USDC for settlement gas.
echo "→ funding provider with 100 USDC for settlement gas..."
"$SOMA_BIN" transfer 100 "$PROVIDER" --usdc --gas-budget 10000000 >/dev/null 2>&1 || \
  echo "  (transfer skipped — provider may already be funded)"

# --- 3. Start indexer-alt -----------------------------------------------------
echo "→ starting soma indexer..."
"$ROOT/target/release/indexer-alt" \
    --database-url "$PG_URL" \
    --local-ingestion-path "$INGEST_DIR" \
    >"$LOG_DIR/indexer.log" 2>&1 &
INDEXER_PID=$!

# --- 4. Start soma-graphql ----------------------------------------------------
GRAPHQL_PORT="${GRAPHQL_PORT:-7000}"
GRAPHQL_URL="http://127.0.0.1:${GRAPHQL_PORT}/graphql"
echo "→ starting soma-graphql on :$GRAPHQL_PORT..."
"$ROOT/target/release/soma-graphql" \
    --database-url "$PG_URL" \
    --listen-address "127.0.0.1:${GRAPHQL_PORT}" \
    >"$LOG_DIR/graphql.log" 2>&1 &
GRAPHQL_PID=$!

echo -n "  waiting for GraphQL"
for _ in $(seq 1 30); do
  if curl -sS --max-time 1 "http://127.0.0.1:${GRAPHQL_PORT}/health" >/dev/null 2>&1 \
     || curl -sS --max-time 1 "$GRAPHQL_URL" -X POST -H 'content-type: application/json' \
            -d '{"query":"{__typename}"}' >/dev/null 2>&1; then
    echo " ready"
    break
  fi
  sleep 0.5
  echo -n "."
done
echo

# --- 5. Write provider TOML --------------------------------------------------
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

# --- 6. Start the provider server (registers on-chain) ------------------------
echo "→ starting soma inference serve --address $PROVIDER..."
"$SOMA_BIN" inference serve \
    --config "$PROV_TOML" \
    --address "$PROVIDER" \
    --heartbeat-interval-secs 30 \
    >"$LOG_DIR/provider.log" 2>&1 &
PROV_PID=$!
for _ in $(seq 1 30); do
  if curl -sS --max-time 1 "http://127.0.0.1:${PROVIDER_PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.3
done

# Wait for the indexer to surface the provider via GraphQL — the
# proxy needs `providers()` to return at least one row before
# routing.
echo -n "  waiting for provider to land in indexer"
for _ in $(seq 1 60); do
  N=$(curl -sS --max-time 2 -X POST "$GRAPHQL_URL" \
        -H 'content-type: application/json' \
        -d '{"query":"{providers(first:10){edges{node{address}}}}"}' \
        2>/dev/null | python3 -c 'import sys, json
try:
    d = json.load(sys.stdin)
    print(len((d.get("data") or {}).get("providers",{}).get("edges") or []))
except Exception:
    print(0)' 2>/dev/null)
  if [ "${N:-0}" -gt 0 ]; then
    echo " ready ($N)"
    break
  fi
  sleep 1
  echo -n "."
done

# --- 7. Start the proxy (indexer-backed registry) ----------------------------
echo "→ starting soma inference proxy --address $PAYER..."
"$SOMA_BIN" inference proxy \
    --address "$PAYER" \
    --listen "127.0.0.1:${PROXY_PORT}" \
    --indexer-url "$GRAPHQL_URL" \
    >"$LOG_DIR/proxy.log" 2>&1 &
PROXY_PID=$!
for _ in $(seq 1 30); do
  if curl -sS --max-time 1 "http://127.0.0.1:${PROXY_PORT}/v1/models" >/dev/null 2>&1; then
    break
  fi
  sleep 0.3
done

# --- 8. Persist state ---------------------------------------------------------
cat >"$STATE_FILE" <<EOF
LOCALNET_PID=${LOCALNET_PID}
INDEXER_PID=${INDEXER_PID}
GRAPHQL_PID=${GRAPHQL_PID}
PROV_PID=${PROV_PID}
PROXY_PID=${PROXY_PID}
PAYER=${PAYER}
PROVIDER=${PROVIDER}
MODEL=${DEFAULT_MODEL}
BACKEND=${BACKEND}
GRAPHQL_URL=${GRAPHQL_URL}
INGEST_DIR=${INGEST_DIR}
EOF

echo
echo "READY ($BACKEND)"
echo "  payer        = $PAYER"
echo "  provider     = $PROVIDER  (http://127.0.0.1:${PROVIDER_PORT})"
echo "  proxy        = http://127.0.0.1:${PROXY_PORT}"
echo "  graphql      = $GRAPHQL_URL"
echo "  model        = $DEFAULT_MODEL"
echo "  state        = $STATE_FILE"
echo "  logs         = $LOG_DIR"
echo
echo "send a request:  examples/localnet/chat.sh \"hello\""
echo "list channels:   examples/localnet/show.sh"
echo "tear down:       examples/localnet/down.sh   # provider settles on SIGTERM"
