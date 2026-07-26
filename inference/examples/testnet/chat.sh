#!/usr/bin/env bash
# Send a chat through the proxy.   Usage: ./chat.sh "your message"
# The first call opens a payment channel, so it's a bit slower.
set -euo pipefail
. "$(dirname "$0")/config.sh"

PROMPT="${*:-hello}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
PAYLOAD="$(MODEL="$MODEL" PROMPT="$PROMPT" MAX_TOKENS="$MAX_TOKENS" python3 -c 'import os,json
print(json.dumps({
  "model": os.environ["MODEL"],
  "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
  "max_tokens": int(os.environ["MAX_TOKENS"]),
  "stream": True,
}))')"

curl -sN -X POST "http://127.0.0.1:${PROXY_PORT}/v1/chat/completions" \
  -H 'content-type: application/json' -d "$PAYLOAD" \
| python3 -u -c 'import sys, json
got = False
raw = []
for line in sys.stdin:
    raw.append(line)
    s = line.strip()
    if not s.startswith("data:"): continue
    p = s[5:].lstrip()
    if p == "[DONE]":
        sys.stdout.write("\n"); break
    try: d = json.loads(p)
    except Exception: continue
    for c in d.get("choices", []):
        t = (c.get("delta") or {}).get("content") or ""
        if t:
            got = True
            sys.stdout.write(t); sys.stdout.flush()
if not got:
    body = "".join(raw).rstrip()
    sys.stderr.write("chat.sh: proxy returned no SSE content — raw response below.\n")
    sys.stderr.write("  (first request opens a channel; if it failed, the tx usually still landed — try again.)\n")
    if body:
        sys.stderr.write(body + "\n")
    sys.exit(1)'
