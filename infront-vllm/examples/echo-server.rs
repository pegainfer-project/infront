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

use std::path::PathBuf;

use clap::Parser;
use infront_engine::Engine;
use infront_engine::EngineInfo;
use infront_engine::FinishReason;
use infront_engine::QueuedRequest;
use infront_engine::RequestId;
use infront_engine::RequestLedger;
use infront_engine::Scheduler;
use infront_engine::SchedulerMetrics;
use infront_engine::spawn_scheduler;

/// A `config.json` plus a byte-level tokenizer: what the frontend needs to
/// render and detokenize.
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

/// Answers each request with its own prompt tokens: echo
/// `prompt[..min(len, max_tokens)]` back, one token per step, then finish —
/// `FinishReason::Stop` for a full echo, `FinishReason::Length` when
/// `max_tokens` cut it short. One token per step keeps the per-token
/// streaming path warm instead of collapsing the response into one batch.
#[derive(Default)]
struct EchoScheduler {
    queued: Vec<QueuedRequest>,
    running: Vec<RunningRequest>,
}

/// One admission onward: the request's ledger id plus its echo cursor —
/// prompt tokens not yet echoed and whether `max_tokens` trimmed the echo
/// short.
struct RunningRequest {
    id: RequestId,
    /// Prompt tokens not yet echoed, stored reversed for cheap pops.
    pending: Vec<u32>,
    /// Whether `max_tokens` trimmed the prompt (decides the finish reason).
    truncated: bool,
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
            let echo_len = request.prompt_tokens.len().min(request.max_tokens);
            let truncated = echo_len < request.prompt_tokens.len();
            let mut pending = Vec::from(&request.prompt_tokens[..echo_len]);
            pending.reverse();
            ledger.admit(id);
            self.running.push(RunningRequest {
                id,
                pending,
                truncated,
            });
        }
        let mut still_running = Vec::new();
        for mut running in self.running.drain(..) {
            if ledger.is_aborted(running.id) {
                ledger.retire(running.id);
                continue;
            }
            if let Some(token) = running.pending.pop() {
                ledger.push_tokens(running.id, &[token], &[]);
                still_running.push(running);
            } else {
                let reason = if running.truncated {
                    FinishReason::Length
                } else {
                    FinishReason::Stop
                };
                ledger.finish(running.id, reason);
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
    infront_vllm::serve(
        std::future::ready(Ok(engine)),
        &args.model_path,
        vec![args.served_model_name],
        infront_vllm::ParserSelection::Auto,
        args.port,
        None,
        infront_vllm::shutdown_token_from_ctrl_c(),
    )
    .await
}
