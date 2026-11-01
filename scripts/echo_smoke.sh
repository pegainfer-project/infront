#!/usr/bin/env bash
# Serve the echo scheduler and check completions and chat, plain and
# streamed, round-trip through the vLLM frontend.
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

post() {
  curl -sfN "localhost:$port/v1/$1" -H 'content-type: application/json' -d "$2"
}

expect() {
  [[ "$2" == "$3" ]] || { echo "$1: expected '$3', got '$2'" >&2; exit 1; }
}

text=$(post completions '{"model": "echo", "prompt": "hello, infront", "max_tokens": 64}' |
  python3 -c 'import json, sys; print(json.load(sys.stdin)["choices"][0]["text"])')
expect completion "$text" "hello, infront"

conversation='"model": "echo", "max_tokens": 64, "messages": [
  {"role": "user", "content": "hello, "},
  {"role": "assistant", "content": "in"},
  {"role": "user", "content": "front"}]'

reply=$(post chat/completions "{$conversation}" | python3 -c '
import json, sys
choice = json.load(sys.stdin)["choices"][0]
print(choice["message"]["content"], choice["finish_reason"])')
expect chat "$reply" "hello, infront stop"

streamed=$(post chat/completions "{$conversation, \"stream\": true}" | python3 -c '
import json, sys
chunks = [json.loads(line[6:]) for line in sys.stdin if line.startswith("data: {")]
choices = [choice for chunk in chunks for choice in chunk["choices"]]
content = "".join(choice["delta"].get("content") or "" for choice in choices)
print(content, choices[-1]["finish_reason"])')
expect "streamed chat" "$streamed" "hello, infront stop"

# A stop token drawn before min_tokens is an ordinary token: the echo runs
# past its EOS until the script runs out.
reply=$(post completions '{"model": "echo", "prompt": "hello, infront", "max_tokens": 64, "min_tokens": 32}' |
  python3 -c '
import json, sys
choice = json.load(sys.stdin)["choices"][0]
print(choice["text"], choice["finish_reason"])')
expect "completion under min_tokens" "$reply" "hello, infront length"

one_token=$(post chat/completions '{"model": "echo", "stream": true, "max_tokens": 1, "min_tokens": 1,
  "messages": [{"role": "user", "content": "hello"}]}' | python3 -c '
import json, sys
chunks = [json.loads(line[6:]) for line in sys.stdin if line.startswith("data: {")]
print([choice for chunk in chunks for choice in chunk["choices"]][-1]["finish_reason"])')
expect "streamed chat at min_tokens = max_tokens" "$one_token" "length"

status() {
  curl -s -o /dev/null -w '%{http_code}' "localhost:$port/v1/$1" -H 'content-type: application/json' -d "$2"
}

expect "a penalty" "$(status completions '{"model": "echo", "prompt": "hi", "frequency_penalty": 0.5}')" 400
expect "min_tokens over max_tokens" "$(status chat/completions '{"model": "echo", "max_tokens": 1, "min_tokens": 2,
  "messages": [{"role": "user", "content": "hi"}]}')" 400

echo "echo smoke ok"
