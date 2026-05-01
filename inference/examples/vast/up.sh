#!/usr/bin/env bash
# Rent a Vast.ai GPU, run vLLM with an HF model, then boot
# `soma inference serve`. (`soma inference proxy` is started lazily by
# chat.sh on the consumer side.)
# Required: VAST_API_KEY in env.
# Optional: HF_TOKEN, MODEL (default Qwen/Qwen2.5-1.5B-Instruct), MAX_DPH (default 0.4),
#           SOMA_ADDRESS (defaults to keystore active address).
set -euo pipefail

ENV_FILE="$(dirname "$0")/.env"
if [ -f "$ENV_FILE" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|\#*) continue;; esac
    key="${line%%=*}"; val="${line#*=}"
    val="${val#\"}"; val="${val%\"}"; val="${val#\'}"; val="${val%\'}"
    [ -z "${!key:-}" ] && export "$key=$val"
  done < "$ENV_FILE"
fi

: "${VAST_API_KEY:?VAST_API_KEY must be set (export it or put it in examples/vast/.env)}"
HF_TOKEN="${HF_TOKEN:-}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
MAX_DPH="${MAX_DPH:-0.4}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
STATE="$HOME/.soma/inference-vast-demo.state"

if [ -f "$STATE" ]; then
  echo "demo already running; run examples/vast/down.sh first." >&2
  exit 1
fi
mkdir -p "$HOME/.soma"

# JSON-safe parser for Vast responses (status_msg sometimes has raw \r\n).
PARSER="$(mktemp)"
cat >"$PARSER" <<'PY'
import sys, json, re
raw = sys.stdin.buffer.read().decode("utf-8", "replace")
clean = re.sub(r"[\x00-\x1f]", lambda m: "\\u%04x" % ord(m.group()), raw)
print(json.dumps(json.loads(clean)))
PY
parse() { python3 "$PARSER"; }

cleanup_on_fail() {
  rc=$?
  if [ $rc -ne 0 ] && [ -n "${INST:-}" ] && [ ! -f "$STATE" ]; then
    echo "→ aborting: destroying instance ${INST}..." >&2
    curl -sS -X DELETE -H "Authorization: Bearer $VAST_API_KEY" \
      "https://console.vast.ai/api/v0/instances/${INST}/" >/dev/null || true
  fi
  rm -f "$PARSER"
  exit $rc
}
trap cleanup_on_fail EXIT

# 1. Pick the cheapest 1×24GB+ offer in a non-blocked region.
echo "→ searching offers (model=$MODEL, max=\$$MAX_DPH/hr)..."
Q=$(cat <<EOF
{"verified":{"eq":true},"rentable":{"eq":true},"rented":{"eq":false},
 "num_gpus":{"eq":"1"},"gpu_ram":{"gte":24000},"cuda_max_good":{"gte":12.9},
 "compute_cap":{"gte":750},"dph_total":{"lt":${MAX_DPH}},
 "reliability":{"gte":0.98},"inet_down":{"gte":500},
 "geolocation":{"in":["US","CA","DE","FR","NL","GB","SE","FI","IE"]},
 "order":[["dph_total","asc"]]}
EOF
)
OFFERS=$(curl -sS -G -H "Authorization: Bearer $VAST_API_KEY" \
  --data-urlencode "q=${Q}" \
  "https://console.vast.ai/api/v0/search/asks/" | parse)
read -r ASK_ID GPU DPH GEO < <(echo "$OFFERS" | python3 -c '
import sys, json
o = json.load(sys.stdin)["offers"][0]
print(o["id"], o["gpu_name"].replace(" ", "_"), o["dph_total"], o["geolocation"].split(",")[0])')
echo "  ask=$ASK_ID  ${GPU//_/ }  \$$DPH/hr  $GEO"

# 2. Rent it.
echo "→ renting..."
BODY=$(MODEL="$MODEL" HF_TOKEN="$HF_TOKEN" python3 -c '
import os, json
model = os.environ["MODEL"]
print(json.dumps({
  "image": "vllm/vllm-openai:latest",
  "label": "soma-inference-vast-demo",
  "disk": 32,
  "runtype": "args",
  "target_state": "running",
  "env": {"HF_TOKEN": os.environ["HF_TOKEN"], "-p 8000:8000": ""},
  "args_str": (
    "--model " + model +
    " --max-model-len 8192 --enforce-eager"
    " --host 0.0.0.0 --port 8000"
    " --download-dir /workspace/models"
  ),
  "cancel_unavail": False,
}))')
INST=$(curl -sS -X PUT -H "Authorization: Bearer $VAST_API_KEY" \
  -H "Content-Type: application/json" \
  --data "$BODY" "https://console.vast.ai/api/v0/asks/${ASK_ID}/" \
  | parse | python3 -c 'import sys,json; print(json.load(sys.stdin)["new_contract"])')
echo "  instance=$INST"

# 3. Wait for the host to map ports and vLLM to answer /v1/models.
echo "→ waiting for vLLM (cold start: docker pull + HF download + load — usually 5-10 min)"
URL=""
for i in $(seq 1 80); do
  R=$(curl -sS -H "Authorization: Bearer $VAST_API_KEY" \
        "https://console.vast.ai/api/v0/instances/${INST}/" | parse)
  IP=$(echo "$R" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("instances",d).get("public_ipaddr") or "")')
  PORT=$(echo "$R" | python3 -c 'import sys,json; d=json.load(sys.stdin); i=d.get("instances",d); p=(i.get("ports") or {}).get("8000/tcp") or []; print(p[0].get("HostPort") if p else "")')
  if [ -n "$IP" ] && [ -n "$PORT" ]; then
    if curl -sS --connect-timeout 4 --max-time 8 "http://${IP}:${PORT}/v1/models" 2>/dev/null | grep -q '"data"'; then
      URL="http://${IP}:${PORT}"
      echo "  vLLM ready at $URL"
      break
    fi
  fi
  printf "."
  sleep 15
done
echo
[ -n "$URL" ] || { echo "vLLM never came up — bad host?" >&2; exit 1; }

# 4. Build if needed; write provider config; boot soma inference serve.
echo "→ booting soma inference serve"
[ -x "$ROOT/target/release/soma" ] || \
  (cd "$ROOT" && PYO3_PYTHON=python3 cargo build --release -p cli)

PROV_TOML=/tmp/soma-inference-vast-demo-provider.toml
cat >"$PROV_TOML" <<EOF
[server]
listen          = "127.0.0.1:8444"
public_endpoint = "http://127.0.0.1:8444"
[chain]
mode = "file"
soma_home = "~/.soma"
heartbeat_interval_secs = 600
[backend]
kind         = "vast"
api_key_env  = "VAST_API_KEY"
upstream_url = "${URL}"
[auth]
clock_skew_tolerance_secs = 60
[[offerings]]
id              = "${MODEL}"
name            = "${MODEL} (vLLM on Vast.ai)"
hugging_face_id = "${MODEL}"
context_length  = 8192
architecture    = { input_modalities = ["text"], output_modalities = ["text"], tokenizer = "Qwen", instruct_type = "chatml" }
top_provider    = { context_length = 8192, max_completion_tokens = 4096, is_moderated = false }
supported_parameters = ["max_tokens","temperature","top_p","stop","seed"]
pricing = { prompt = "0.0000002", completion = "0.0000004", request = "0", image = "0", input_cache_read = "0", input_cache_write = "0" }
EOF

ADDR_FLAG=()
[ -n "${SOMA_ADDRESS:-}" ] && ADDR_FLAG=(--address "$SOMA_ADDRESS")

"$ROOT/target/release/soma" inference serve --config "$PROV_TOML" "${ADDR_FLAG[@]}" \
  >/tmp/soma-inference-prov.log 2>&1 &
PROV_PID=$!

for i in $(seq 1 15); do
  curl -sS --max-time 2 http://127.0.0.1:8444/health >/dev/null 2>&1 && break
  sleep 1
done

cat >"$STATE" <<EOF
INSTANCE_ID=${INST}
UPSTREAM_URL=${URL}
PROV_PID=${PROV_PID}
MODEL=${MODEL}
EOF

echo
echo "READY"
echo "  instance:  ${INST}  (${GPU//_/ } @ \$$DPH/hr in $GEO)"
echo "  vLLM:      ${URL}"
echo "  provider:  http://127.0.0.1:8444  (logs: /tmp/soma-inference-prov.log)"
echo "  state:     ${STATE}"
echo
echo "send a request:  examples/vast/chat.sh \"hello\""
echo "tear down:       examples/vast/down.sh"
