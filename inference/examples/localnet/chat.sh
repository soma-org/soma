#!/usr/bin/env bash
# Send a chat completion through the running proxy.
#
# Usage: chat.sh "your prompt"     (default: "hello")
# Env:   MAX_TOKENS=64
set -euo pipefail
. "$(dirname "$0")/config.sh"

[ -f "$STATE_FILE" ] || { echo "no demo running. examples/localnet/up.sh first." >&2; exit 1; }
# shellcheck disable=SC1090
. "$STATE_FILE"

PROMPT="${*:-hello}"
MAX_TOKENS="${MAX_TOKENS:-64}"

PAYLOAD=$(MODEL="$MODEL" PROMPT="$PROMPT" MAX_TOKENS="$MAX_TOKENS" python3 -c '
import os, json
print(json.dumps({
  "model": os.environ["MODEL"],
  "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
  "max_tokens": int(os.environ["MAX_TOKENS"]),
  "stream": True,
}))')

if [ "${RAW:-}" = "1" ]; then
  curl -sS -N -X POST "http://127.0.0.1:${PROXY_PORT}/v1/chat/completions" \
    -H 'content-type: application/json' -d "$PAYLOAD"
else
  curl -sS -N -X POST "http://127.0.0.1:${PROXY_PORT}/v1/chat/completions" \
    -H 'content-type: application/json' -d "$PAYLOAD" \
  | python3 -u -c '
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("data:"): continue
    payload = line[5:].lstrip()
    if payload == "[DONE]":
        sys.stdout.write("\n")
        break
    try:
        d = json.loads(payload)
    except Exception:
        continue
    for c in d.get("choices", []):
        chunk = (c.get("delta") or {}).get("content") or ""
        if chunk:
            sys.stdout.write(chunk)
            sys.stdout.flush()
'
fi
