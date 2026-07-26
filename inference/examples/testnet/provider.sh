#!/usr/bin/env bash
# TERMINAL 1 — run a soma inference provider against testnet.
#
# Writes the provider TOML for you, creates + funds a provider wallet
# on first run, then runs `soma start provider` in the foreground.
# `serve` registers the provider AND its offering on-chain at boot and
# heartbeats while it runs. Ctrl-C to stop (it settles open channels).
set -euo pipefail
. "$(dirname "$0")/config.sh"
ensure_soma
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY (or put it in ~/autodebate/.env)}"

# Provider wallet — created + funded once, reused after.
PROVIDER="$(provider_address)"
if [ -z "$PROVIDER" ]; then
  echo "→ creating provider wallet '$PROVIDER_ALIAS'…"
  PROVIDER="$("$SOMA" wallet new --alias "$PROVIDER_ALIAS" --json | python3 -c "$_norm_addr")"
  echo "→ funding it with 50 USDC for gas…"
  "$SOMA" transfer 50 "$PROVIDER" --usdc >/dev/null
fi

# Provider TOML — generated for you; edit pricing/model here if you like.
cat > "$PROVIDER_TOML" <<EOF
[server]
listen          = "127.0.0.1:${PROVIDER_PORT}"
public_endpoint = "http://127.0.0.1:${PROVIDER_PORT}"
[backend]
kind         = "openrouter"
api_key_env  = "OPENROUTER_API_KEY"
upstream_url = "https://openrouter.ai/api/v1"
[[offerings]]
id              = "${MODEL}"
name            = "${MODEL}"
hugging_face_id = "${MODEL}"
context_length  = 200000
architecture = { input_modalities = ["text"], output_modalities = ["text"], tokenizer = "Claude" }
top_provider = { context_length = 200000, max_completion_tokens = 4096, is_moderated = false }
supported_parameters = ["max_tokens", "temperature"]
ttft_bound_ms = 60000
ttot_bound_ms = 10000
pricing = { prompt = "0.000001", completion = "0.000005", request = "0", image = "0", input_cache_read = "0", input_cache_write = "0" }
EOF

echo "→ provider $PROVIDER"
echo "→ serving http://127.0.0.1:${PROVIDER_PORT}  ·  model ${MODEL}  ·  config ${PROVIDER_TOML}"
echo "  Ctrl-C to stop (settles open channels)."
echo
exec env RUST_LOG="${RUST_LOG:-inference=info,sdk=info}" \
  "$SOMA" start provider \
    --config "$PROVIDER_TOML" --address "$PROVIDER" --soma-home "$PROVIDER_HOME"
