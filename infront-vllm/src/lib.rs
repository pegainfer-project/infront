use std::collections::HashMap;
use std::future::Future;
use std::num::NonZeroU32;
use std::num::NonZeroUsize;
use std::os::fd::IntoRawFd;
use std::os::unix::net::UnixListener;
use std::path::Path;
use std::time::Duration;

use anyhow::Result;
use infront_engine::Engine;
use log::warn;
use serde::Deserialize;
use tokio_util::sync::CancellationToken;
use vllm_chat::LoadModelBackendsOptions;
use vllm_chat::load_model_backends;
use vllm_engine_core_client::TransportMode;
use vllm_engine_core_client::protocol::structured_outputs::StructuredOutputBackend;
use vllm_server::ApiServerOptions;
use vllm_server::ChatTemplateContentFormatOption;
use vllm_server::Config;
use vllm_server::CoordinatorMode;
use vllm_server::CorsConfig;
use vllm_server::CustomHistogramBuckets;
use vllm_server::DEFAULT_KEEP_ALIVE_TIMEOUT;
use vllm_server::DEFAULT_LOG_STATS_INTERVAL;
use vllm_server::GenerationConfigMode;
use vllm_server::HttpListenerMode;
pub use vllm_server::ParserSelection;
use vllm_server::RendererSelection;
use vllm_text::backend::hf::HfOverrides;

mod bridge;
mod history;
mod shaping;
mod wire;

use bridge::SteppedEngineBridge;
use bridge::ipc_endpoint;
use bridge::local_ipc_namespace;
pub use shaping::Capabilities;

#[derive(Debug, Deserialize)]
struct ModelLenConfig {
    max_position_embeddings: Option<u32>,
    text_config: Option<Box<ModelLenConfig>>,
}

impl ModelLenConfig {
    fn max_model_len(&self) -> Option<u32> {
        self.max_position_embeddings
            .or_else(|| self.text_config.as_ref()?.max_model_len())
    }
}

/// How [`serve`] exposes an engine. [`ServeOptions::new`] fills in the
/// defaults: one scheduler, every capability served, no history boundary,
/// all interfaces on port 8000.
#[derive(Clone, Debug)]
pub struct ServeOptions {
    /// A local checkpoint directory (config.json, tokenizer, chat template)
    /// or a Hugging Face model id.
    pub model: String,
    /// Names the endpoint answers to; empty serves under `model`.
    pub served_model_names: Vec<String>,
    pub tool_call_parser: ParserSelection,
    pub host: String,
    pub port: u16,
    /// Longest request in tokens; `None` reads `max_position_embeddings`
    /// from `model`'s config.json. The engine's `servable_len` caps either.
    pub max_model_len: Option<u32>,
    /// Schedulers the engine exposes. The frontend sizes its routing before
    /// the engine resolves, so the count is declared here and checked against
    /// [`Engine::schedulers`].
    pub engine_count: NonZeroUsize,
    pub capabilities: Capabilities,
    /// Measure each chat prompt's history boundary into
    /// [`infront_engine::Request::history_tokens`], at one render and
    /// tokenization of the last message per chat request.
    pub history_boundary: bool,
}

impl ServeOptions {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            served_model_names: Vec::new(),
            tool_call_parser: ParserSelection::Auto,
            host: "0.0.0.0".to_string(),
            port: 8000,
            max_model_len: None,
            engine_count: NonZeroUsize::MIN,
            capabilities: Capabilities::default(),
            history_boundary: false,
        }
    }
}

/// Serve `engine` behind vLLM's OpenAI-compatible frontend until `shutdown`.
/// The HTTP frontend (tokenizer, chat templates) starts loading immediately
/// and the engine bridge attaches once `engine` resolves; HTTP binds only
/// after the bridge registers, so a reachable port means the engine is ready.
pub async fn serve(
    engine: impl Future<Output = Result<Engine>> + Send + 'static,
    options: ServeOptions,
    shutdown: CancellationToken,
) -> Result<()> {
    let ServeOptions {
        model: model_id,
        served_model_names,
        tool_call_parser,
        host,
        port,
        max_model_len,
        engine_count,
        capabilities,
        history_boundary,
    } = options;
    let max_model_len = resolve_max_model_len(Path::new(&model_id), max_model_len);
    let engine_count = engine_count.get();
    let data_parallel_size = u32::try_from(engine_count)
        .map_err(|_| anyhow::anyhow!("frontend engine_count {engine_count} exceeds u32"))?;
    let namespace = local_ipc_namespace()?;
    let input_address = ipc_endpoint(&namespace, "input.sock");
    let output_address = ipc_endpoint(&namespace, "output.sock");
    // vllm-server adopts already-bound listeners (the supervisor's role in
    // upstream's Python deployment); binding them here means the bridge can
    // connect at any time, the server owns and closes the descriptors.
    let input_listener_fd = UnixListener::bind(namespace.join("input.sock"))?.into_raw_fd();
    let output_listener_fd = UnixListener::bind(namespace.join("output.sock"))?.into_raw_fd();

    // The HTTP server runs concurrently with the engine load: vllm-server
    // spends ~1s loading the tokenizer and chat templates before it waits for
    // an engine to register, so neither waits on the other. This task attaches
    // the bridge once the engine resolves and runs it to completion; on engine
    // failure it cancels the server so the error surfaces instead of hanging
    // in the registration wait.
    let server_shutdown = shutdown.child_token();
    let bridge_shutdown = shutdown.child_token();
    let engine_task = tokio::spawn({
        let server_shutdown = server_shutdown.clone();
        let bridge_shutdown = bridge_shutdown.clone();
        let input_address = input_address.clone();
        let output_address = output_address.clone();
        async move {
            let engine = match engine.await {
                Ok(engine) => engine,
                Err(error) => {
                    server_shutdown.cancel();
                    return Err(error);
                }
            };
            let mut bridges = tokio::task::JoinSet::new();
            // The engine owns its scheduler threads; reaped below after the
            // bridges (and with them the scheduler handles) are gone.
            let mut scheduler_joins: Vec<std::thread::JoinHandle<()>> = Vec::new();
            let actual_schedulers = engine.schedulers.len();
            if actual_schedulers != engine_count {
                server_shutdown.cancel();
                anyhow::bail!(
                    "frontend declared {engine_count} engines but the launched engine \
                     exposes {actual_schedulers} schedulers"
                );
            }
            let info = engine.info;
            let servable_limit = info.servable_len.map(|cap| max_model_len.min(cap));
            let max_model_len = servable_limit.unwrap_or(max_model_len);
            for (engine_index, scheduler) in engine.schedulers.into_iter().enumerate() {
                scheduler_joins.push(scheduler.join);
                let bridge = SteppedEngineBridge {
                    input_address: input_address.clone(),
                    output_address: output_address.clone(),
                    scheduler: scheduler.handle,
                    kv_capacity: info.kv_capacity,
                    max_model_len,
                    engine_index: engine_index as u32,
                    data_parallel_size,
                };
                let shutdown = bridge_shutdown.clone();
                bridges.spawn(async move { (engine_index, bridge.run(shutdown).await) });
            }

            let mut bridge_error = None;
            while let Some(joined) = bridges.join_next().await {
                match joined {
                    Ok((_, Ok(()))) if bridge_shutdown.is_cancelled() => {}
                    Ok((engine_index, Ok(()))) => {
                        bridge_error = Some(anyhow::anyhow!(
                            "local vLLM engine {engine_index} bridge exited unexpectedly"
                        ));
                        break;
                    }
                    Ok((engine_index, Err(error))) => {
                        bridge_error = Some(
                            error
                                .context(format!("local vLLM engine {engine_index} bridge failed")),
                        );
                        break;
                    }
                    Err(error) => {
                        bridge_error = Some(anyhow::anyhow!(
                            "local vLLM engine bridge task panicked: {error}"
                        ));
                        break;
                    }
                }
            }
            if bridge_error.is_some() {
                server_shutdown.cancel();
                bridge_shutdown.cancel();
                // Let the other bridges abort their requests and flush their output.
                while bridges.join_next().await.is_some() {}
            }
            // The bridges are gone, and with them the partition handles:
            // submission channels are disconnected, so the drivers drain and
            // exit. Reap their threads to surface scheduler panics.
            if !scheduler_joins.is_empty() {
                let _ = tokio::task::spawn_blocking(move || {
                    for join in scheduler_joins {
                        if join.join().is_err() {
                            warn!("scheduler thread panicked during shutdown");
                        }
                    }
                })
                .await;
            }
            match bridge_error {
                Some(error) => Err(error),
                None => Ok(()),
            }
        }
    });

    // The history measure reads the served checkpoint's template and
    // tokenizer, loaded once here; the server loads its own afterwards and
    // reports its own error if the model is genuinely broken.
    let boundary = if history_boundary {
        match load_model_backends(
            &model_id,
            LoadModelBackendsOptions {
                language_model_only: true,
                ..Default::default()
            },
        )
        .await
        {
            Ok(backends) => Some(history::Boundary::new(&backends)),
            Err(error) => {
                warn!("history boundary disabled: backends for {model_id} did not load: {error}");
                None
            }
        }
    } else {
        None
    };

    let config = Config {
        transport_mode: TransportMode::Bootstrapped {
            input_listener_fd,
            output_listener_fd,
            engine_start_index: 0,
            engine_count,
            data_parallel_size: engine_count,
            // The in-process bridge registers once the engine future resolves,
            // so this bounds the whole engine load (multi-GPU MoE models take
            // minutes, cold starts longer). Load *failure* already cancels the
            // server via the engine task; this only catches a truly hung load.
            ready_timeout: Duration::from_mins(30),
        },
        coordinator_mode: CoordinatorMode::None,
        model: model_id,
        revision: None,
        tokenizer: None,
        hf_config: None,
        hf_overrides: HfOverrides::default(),
        generation_config: GenerationConfigMode::Auto,
        served_model_name: served_model_names,
        listener_mode: HttpListenerMode::BindTcp { host, port },
        tool_call_parser,
        reasoning_parser: ParserSelection::default(),
        tool_strict_level: None,
        structured_outputs_backend: StructuredOutputBackend::default(),
        renderer: RendererSelection::default(),
        chat_template: None,
        default_chat_template_kwargs: None,
        trust_request_chat_template: false,
        exclude_tools_when_tool_choice_none: false,
        limit_mm_per_prompt: HashMap::new(),
        lora_modules: Vec::new(),
        chat_template_content_format: ChatTemplateContentFormatOption::default(),
        max_logprobs: None,
        language_model_only: true,
        cors: CorsConfig::default(),
        api_keys: Vec::new(),
        api_server_options: ApiServerOptions {
            enable_log_requests: true,
            enable_prompt_tokens_details: true,
            ..ApiServerOptions::default()
        },
        // Upstream ties the scheduler-stats and request-event metric recorders
        // to this flag, and the bridge's stats are what `/metrics` exports; the
        // periodic stats log line comes with it.
        disable_log_stats: false,
        custom_histogram_buckets: CustomHistogramBuckets::default(),
        log_stats_interval: Some(DEFAULT_LOG_STATS_INTERVAL),
        disable_request_id_randomization: false,
        grpc_listener_mode: None,
        manages_engine: false,
        shutdown_timeout: Duration::from_secs(10),
        stream_interval: NonZeroU32::MIN,
        keep_alive_timeout: DEFAULT_KEEP_ALIVE_TIMEOUT,
        profiler: None,
        tls: None,
    };

    let result = vllm_server::serve_with_router_extension(config, server_shutdown, move |router| {
        shaping::shape(router, capabilities, boundary)
    })
    .await;
    // Stop the bridge (no-op if the caller's shutdown already cancelled it),
    // then collect the engine task. If the server failed while the engine is
    // still loading, the uncancellable blocking load must finish first.
    bridge_shutdown.cancel();
    if result.is_err() && !engine_task.is_finished() {
        warn!("HTTP server failed; waiting for the in-flight engine load to finish before exit");
    }
    let result = match engine_task.await {
        Ok(Ok(())) => result,
        // Engine failed: the server saw a cancel and returned Ok — the engine
        // error is the one worth reporting.
        Ok(Err(engine_error)) => result.and(Err(engine_error)),
        Err(join_error) => result.and(Err(anyhow::anyhow!(
            "engine startup task panicked: {join_error}"
        ))),
    };
    let _ = std::fs::remove_dir_all(namespace);
    result
}

fn load_max_model_len(model_path: &Path) -> Option<u32> {
    let content = std::fs::read_to_string(model_path.join("config.json")).ok()?;
    serde_json::from_str::<ModelLenConfig>(&content)
        .ok()?
        .max_model_len()
}

fn resolve_max_model_len(model_path: &Path, max_model_len: Option<u32>) -> u32 {
    max_model_len.unwrap_or_else(|| {
        load_max_model_len(model_path).unwrap_or_else(|| {
            const FALLBACK_MAX_MODEL_LEN: u32 = 4096;
            warn!(
                "max_position_embeddings not found in {}/config.json; capping max_model_len at {FALLBACK_MAX_MODEL_LEN}. \
                 Requests are limited to this length — set max_position_embeddings in the model config if it supports more.",
                model_path.display()
            );
            FALLBACK_MAX_MODEL_LEN
        })
    })
}

/// Cancel `token` on the first CTRL+C. Installing the handler replaces the
/// default SIGINT kill behavior — only call this once whatever the token
/// guards can actually wind down (e.g. after an uncancellable blocking engine
/// load has finished), otherwise CTRL+C turns into a no-op wait.
pub fn cancel_token_on_ctrl_c(token: &CancellationToken) {
    let shutdown = token.clone();
    tokio::spawn(async move {
        if let Err(error) = tokio::signal::ctrl_c().await {
            warn!("failed to install CTRL+C handler: {error}");
        }
        shutdown.cancel();
    });
}

pub fn shutdown_token_from_ctrl_c() -> CancellationToken {
    let token = CancellationToken::new();
    cancel_token_on_ctrl_c(&token);
    token
}
