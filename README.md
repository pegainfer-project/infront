# infront

Implement one `Scheduler`, get an OpenAI-compatible inference server.

infront is the serving frontend extracted from [pegainfer](https://github.com/pegainfer-project/pegainfer).
An engine implements a three-method scheduler contract; infront wires it behind
[vLLM's Rust frontend](https://github.com/vllm-project/vllm/tree/main/rust)
(HTTP, chat templates, tokenization, reasoning / tool-call parsing, `vllm:*`
Prometheus metrics) by speaking vLLM's EngineCore protocol in-process. Routers
and tooling that talk to vLLM talk to an infront engine unchanged.

```text
 HTTP (OpenAI / chat / completions)
        │
 vllm-server  ── vLLM Rust frontend, unmodified
        │  EngineCore protocol (ZMQ + msgpack, local IPC)
 infront-vllm ── bridge: EngineCoreRequest ⇄ Request, StepOutputs ⇄ EngineCoreOutputs
        │
 infront-engine ── Scheduler trait, RequestLedger, drive loop
        │
 your scheduler (GPU work, batching, KV) ── one OS thread per scheduler
```

| Crate | What it does |
| -- | -- |
| [`infront-engine`](./infront-engine) | The contract. `Scheduler`, `RequestLedger`, `Request` / `StepOutputs`, `SchedulerMetrics`, `Engine`. No HTTP, no CUDA, no wire format. |
| [`infront-vllm`](./infront-vllm) | Serves an `Engine` behind vLLM's Rust frontend. |

## The contract

```rust
pub trait Scheduler: Send {
    /// Take ownership of one submitted request.
    fn submit(&mut self, request: QueuedRequest);
    /// Advance one step: admission, GPU work, per-request ledger writes.
    fn step(&mut self, ledger: &mut RequestLedger) -> anyhow::Result<()>;
    /// Snapshot published once per iteration.
    fn metrics(&self) -> SchedulerMetrics;
}
```

Every per-request verdict goes through the `RequestLedger` from `step`:
`admit`, `push_tokens`, `finish`, `fail`, `reject`, `retire`. The ledger
enforces exactly one terminal per request and panics on writes after it, so a
scheduler cannot leak or double-answer a request. The driver loop, metrics
publishing and shutdown/fatal handling are shared code, not per-engine
discipline.

## Serving

```rust
let engine = Engine {
    schedulers: vec![spawn_scheduler("my-engine", MyScheduler::new(...))],
    info: EngineInfo { kv_capacity: None, servable_len: None },
};
let options = ServeOptions {
    served_model_names: vec!["my-model".into()],
    ..ServeOptions::new(model_path) // config.json + tokenizer + chat template
};
infront_vllm::serve(
    std::future::ready(Ok(engine)),
    options,
    infront_vllm::shutdown_token_from_ctrl_c(),
)
.await
```

`serve` takes the engine as a future: HTTP starts loading the tokenizer and
templates while weights load, and the port only accepts requests once the
engine has registered.

`ServeOptions` also declares what the engine serves. An engine without
logits sets `capabilities.logprobs = false`, one that only decodes greedily
sets `capabilities.sampling = false`, and a request asking for either is
refused with a 400 before it reaches the engine. An engine whose state cannot
roll back sets `history_boundary` to receive, on each chat request, where the
conversation ends and the generation prompt begins
(`Request::history_tokens`).

A complete, runnable engine is [`examples/echo-server.rs`](./infront-vllm/examples/echo-server.rs):
a scheduler that echoes the prompt back, no GPU or weights.

```bash
cargo run -p infront-vllm --example echo-server
curl localhost:8000/v1/completions -H 'content-type: application/json' \
  -d '{"model": "echo", "prompt": "hello, infront", "max_tokens": 64}'
```

## Building

The toolchain is pinned in [`rust-toolchain.toml`](./rust-toolchain.toml); the
vLLM crates need `protoc`, and linting needs
[`cargo-deny`](https://github.com/EmbarkStudios/cargo-deny).

```bash
cargo x lint    # fmt, clippy, cargo-deny; --fix applies fixes
cargo x test    # workspace tests and the echo-server smoke test
```

See [`AGENTS.md`](./AGENTS.md) for contribution conventions.

## Status

Early. The vLLM crates are pinned to a git revision and move fast; the
`infront-engine` contract is meant to stay stable across those bumps. More
protocol adapters (e.g. SGLang's gRPC runtime service) can sit beside
`infront-vllm` on the same contract.

## License

Apache-2.0.
