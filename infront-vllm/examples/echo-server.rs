//! Echo server: the full infront serving stack (OpenAI-compatible HTTP ->
//! vLLM engine protocol -> the `Scheduler` contract) over a scheduler that
//! echoes the prompt back instead of running a model. No GPU, no weights.
//!
//! It doubles as the reference for plugging an engine in: everything an
//! engine implements is the `EchoScheduler` below; `main` hands it to the
//! vLLM frontend.
//!
//!   cargo run -p infront-vllm --example echo-server
//!
//!   curl http://127.0.0.1:8000/v1/completions \
//!     -H 'content-type: application/json' \
//!     -d '{"model": "echo", "prompt": "hello, infront", "max_tokens": 64}'
//!
//!   # the completion text is the prompt back: "hello, infront"
//!
//!   curl http://127.0.0.1:8000/v1/chat/completions \
//!     -H 'content-type: application/json' \
//!     -d '{"model": "echo", "messages": [{"role": "user", "content": "hello"}]}'
//!
//!   # the fixture's chat template concatenates the messages, so the reply is "hello"

use std::path::PathBuf;

use clap::Parser;
use infront_engine::Engine;
use infront_engine::EngineInfo;
use infront_engine::EosPolicy;
use infront_engine::FinishReason;
use infront_engine::QueuedRequest;
use infront_engine::RequestId;
use infront_engine::RequestLedger;
use infront_engine::Scheduler;
use infront_engine::SchedulerMetrics;
use infront_engine::StopPolicy;
use infront_engine::spawn_scheduler;
use infront_vllm::Capabilities;
use infront_vllm::ServeOptions;

/// A `config.json`, a byte-level tokenizer and a chat template: what the
/// frontend needs to render and detokenize.
const FIXTURE_MODEL_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/echo-model");

#[derive(Parser)]
#[command(about = "infront echo server (no GPU, no weights)")]
struct Args {
    #[arg(long, default_value = FIXTURE_MODEL_DIR)]
    model_path: PathBuf,
    #[arg(long, default_value_t = 8000)]
    port: u16,
    #[arg(long, default_value = "echo")]
    served_model_name: String,
}

/// Answers each request with its own prompt tokens, one per step, then the
/// request's EOS: a model whose output is its input. Every token goes through
/// the request's [`StopPolicy`] as a real model's would, so a stop token in
/// the prompt ends the echo there; `max_tokens` cuts it short with
/// `FinishReason::Length`, as does running out of script with EOS ignored.
/// One token per step keeps the per-token streaming path warm instead of
/// collapsing the response into one batch.
#[derive(Default)]
struct EchoScheduler {
    queued: Vec<QueuedRequest>,
    running: Vec<RunningRequest>,
}

struct RunningRequest {
    id: RequestId,
    /// Tokens still to emit, reversed for cheap pops.
    script: Vec<u32>,
    stop_policy: StopPolicy,
    max_tokens: usize,
}

impl Scheduler for EchoScheduler {
    fn submit(&mut self, request: QueuedRequest) {
        self.queued.push(request);
    }

    fn step(&mut self, ledger: &mut RequestLedger) -> anyhow::Result<()> {
        for QueuedRequest { id, request } in self.queued.drain(..) {
            if ledger.is_aborted(id) {
                ledger.retire(id);
                continue;
            }
            let eos = match request.stop_policy.eos() {
                EosPolicy::Token(eos) => Some(eos),
                EosPolicy::Ignore | EosPolicy::ModelDefault => None,
            };
            ledger.admit(id);
            self.running.push(RunningRequest {
                id,
                script: request.prompt_tokens.into_iter().chain(eos).rev().collect(),
                stop_policy: request.stop_policy,
                max_tokens: request.max_tokens,
            });
        }
        let mut still_running = Vec::new();
        for mut running in self.running.drain(..) {
            let id = running.id;
            if ledger.is_aborted(id) {
                ledger.retire(id);
                continue;
            }
            let Some(token) = running.script.pop() else {
                ledger.finish(id, FinishReason::Length);
                continue;
            };
            ledger.push_tokens(id, &[token], &[]);
            if let Some(cause) = running.stop_policy.classify(token, |_| false) {
                ledger.finish(id, FinishReason::Stop(cause));
            } else if ledger.completion_tokens(id) >= running.max_tokens {
                ledger.finish(id, FinishReason::Length);
            } else {
                still_running.push(running);
            }
        }
        self.running = still_running;
        Ok(())
    }

    fn metrics(&self) -> SchedulerMetrics {
        SchedulerMetrics {
            num_running_reqs: self.running.len() as u64,
            num_waiting_reqs: self.queued.len() as u64,
            ..SchedulerMetrics::default()
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let engine = Engine {
        schedulers: vec![spawn_scheduler("echo", EchoScheduler::default())],
        info: EngineInfo {
            kv_capacity: None,
            servable_len: None,
        },
    };
    eprintln!(
        "serving the echo model on port {} (model={:?}); Ctrl-C to stop",
        args.port, args.served_model_name,
    );
    let options = ServeOptions {
        served_model_names: vec![args.served_model_name],
        port: args.port,
        // An echo has no logits to report and nothing to draw.
        capabilities: Capabilities {
            logprobs: false,
            sampling: false,
        },
        ..ServeOptions::new(args.model_path.display().to_string())
    };
    infront_vllm::serve(
        std::future::ready(Ok(engine)),
        options,
        infront_vllm::shutdown_token_from_ctrl_c(),
    )
    .await
}
