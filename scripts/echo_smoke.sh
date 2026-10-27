#!/usr/bin/env bash
# Serve the echo scheduler and check a completion round-trips through the
# vLLM frontend.
set -euo pipefail

port="${PORT:-18731}"
cargo build --locked --example echo-server
./target/debug/examples/echo-server --port "$port" &
server=$!
trap 'kill "$server" 2>/dev/null || true' EXIT

for _ in $(seq 120); do
  curl -sf "localhost:$port/v1/models" >/dev/null && break
  sleep 1
done

text=$(curl -sf "localhost:$port/v1/completions" \
  -H 'content-type: application/json' \
  -d '{"model": "echo", "prompt": "hello, infront", "max_tokens": 64}' |
  python3 -c 'import json, sys; print(json.load(sys.stdin)["choices"][0]["text"])')
[[ "$text" == "hello, infront" ]] || { echo "unexpected completion: $text" >&2; exit 1; }
echo "echo smoke ok"
