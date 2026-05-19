#!/usr/bin/env bash
# Send a chat through the proxy.   Usage: ./chat.sh "your message"
# The first call opens a payment channel, so it's a bit slower.
set -euo pipefail
. "$(dirname "$0")/config.sh"

PROMPT="${*:-hello}"
PAYLOAD="$(MODEL="$MODEL" PROMPT="$PROMPT" python3 -c 'import os,json
print(json.dumps({
  "model": os.environ["MODEL"],
  "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
  "max_tokens": 256,
  "stream": True,
}))')"

curl -sN -X POST "http://127.0.0.1:${PROXY_PORT}/v1/chat/completions" \
  -H 'content-type: application/json' -d "$PAYLOAD" \
| python3 -u -c 'import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("data:"): continue
    p = line[5:].lstrip()
    if p == "[DONE]":
        print(); break
    try: d = json.loads(p)
    except Exception: continue
    for c in d.get("choices", []):
        t = (c.get("delta") or {}).get("content") or ""
        if t:
            sys.stdout.write(t); sys.stdout.flush()'
