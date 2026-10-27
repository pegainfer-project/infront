use std::collections::BTreeSet;
use std::path::Path;
use std::path::PathBuf;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use anyhow::Context;
use anyhow::Result;
use infront_engine::PrefixCacheCounters;
use infront_engine::SchedulerMetrics;
use infront_engine::SpecDecodeCounters;
use log::info;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use vllm_engine_core_client::ENGINE_CORE_DEAD_SENTINEL;
use vllm_engine_core_client::EngineId;
use vllm_engine_core_client::protocol::dtype::ModelDtype;
use vllm_engine_core_client::protocol::encode_msgpack;
use vllm_engine_core_client::protocol::handshake::EngineCoreReadyResponse;
use vllm_engine_core_client::protocol::logprobs::MaybeWireLogprobs;
use vllm_engine_core_client::protocol::output::EngineCoreEvent;
use vllm_engine_core_client::protocol::output::EngineCoreFinishReason;
use vllm_engine_core_client::protocol::output::EngineCoreOutput;
use vllm_engine_core_client::protocol::output::EngineCoreOutputs;
use vllm_engine_core_client::protocol::output::RequestBatchOutputs;
use vllm_engine_core_client::protocol::output::StopReason;
use vllm_engine_core_client::protocol::output::UtilityCallOutput;
use vllm_engine_core_client::protocol::stats::BaseCacheStats;
use vllm_engine_core_client::protocol::stats::CachedTokensBySource;
use vllm_engine_core_client::protocol::stats::PrefillStats;
use vllm_engine_core_client::protocol::stats::PrefixCacheStats;
use vllm_engine_core_client::protocol::stats::SchedulerStats;
use vllm_engine_core_client::protocol::stats::SpecDecodingStats;
use vllm_engine_core_client::protocol::utility::UtilityCallId;
use vllm_engine_core_client::protocol::utility::UtilityOutput;
use vllm_engine_core_client::protocol::utility::UtilityResultEnvelope;
use zeromq::DealerSocket;
use zeromq::PushSocket;
use zeromq::SocketOptions;
use zeromq::ZmqMessage;
use zeromq::prelude::Socket;
use zeromq::prelude::SocketSend;
use zeromq::util::PeerIdentity;

/// vLLM `SchedulerStats` view of a load snapshot — what the frontend's
/// Prometheus gauges (`scheduler_running`, `scheduler_waiting`,
/// `kv_cache_usage`) and DP load balancer consume.
pub(crate) fn scheduler_stats_from(snapshot: &SchedulerMetrics) -> SchedulerStats {
    SchedulerStats {
        num_running_reqs: snapshot.num_running_reqs,
        num_waiting_reqs: snapshot.num_waiting_reqs,
        kv_cache_usage: if snapshot.kv_total_blocks == 0 {
            0.0
        } else {
            snapshot.kv_used_blocks as f64 / snapshot.kv_total_blocks as f64
        },
        ..SchedulerStats::default()
    }
}

/// Per-interval spec-decode delta from two cumulative snapshots, in the wire
/// shape the frontend increments its `vllm:spec_decode_*_total` counters by (see
/// [`SpecDecodeCounters`] for why the transport carries totals and the wire
/// carries deltas).
fn spec_decode_delta(last: &SpecDecodeCounters, cur: &SpecDecodeCounters) -> SpecDecodingStats {
    let num_accepted_tokens_per_pos = cur
        .num_accepted_tokens_per_pos
        .iter()
        .zip(&last.num_accepted_tokens_per_pos)
        .map(|(cur_pos, last_pos)| cur_pos.saturating_sub(*last_pos))
        .take(cur.num_spec_tokens as usize)
        .collect();
    SpecDecodingStats {
        num_spec_tokens: cur.num_spec_tokens,
        num_drafts: cur.num_drafts.saturating_sub(last.num_drafts),
        num_draft_tokens: cur.num_draft_tokens.saturating_sub(last.num_draft_tokens),
        num_accepted_tokens: cur
            .num_accepted_tokens
            .saturating_sub(last.num_accepted_tokens),
        num_accepted_tokens_per_pos,
    }
}

/// Per-interval prefix-cache deltas from two cumulative snapshots, in the wire
/// shape the frontend increments its prefix-cache counters by. The frontend
/// credits every local hit to `source="device"` and reads a tier split only off
/// the connector stats, so host-tier hits travel there, the way vLLM's own
/// offload connector reports them: local stats carry the remaining hits into
/// `vllm:prefix_cache_*` and `source="device"`, connector stats carry the host
/// lookups and hits into `vllm:external_prefix_cache_*` and `source="host"`.
/// The connector side is `None` for an interval without host activity. A total
/// below its baseline means the engine restarted, so the interval diffs from
/// zero rather than going negative or dropping the new counts.
fn prefix_cache_delta(
    last: &PrefixCacheCounters,
    cur: &PrefixCacheCounters,
) -> (PrefixCacheStats, Option<PrefixCacheStats>) {
    let restarted = cur.queries < last.queries
        || cur.hits < last.hits
        || cur.host_queries < last.host_queries
        || cur.host_hits < last.host_hits;
    let base = if restarted {
        PrefixCacheCounters::default()
    } else {
        *last
    };
    let hits = cur.hits - base.hits;
    let host_queries = cur.host_queries - base.host_queries;
    let host_hits = cur.host_hits - base.host_hits;
    let local = PrefixCacheStats {
        base: BaseCacheStats {
            queries: cur.queries - base.queries,
            hits: hits.saturating_sub(host_hits),
            ..BaseCacheStats::default()
        },
        ..PrefixCacheStats::default()
    };
    let connector = (host_queries > 0 || host_hits > 0).then(|| PrefixCacheStats {
        base: BaseCacheStats {
            queries: host_queries,
            hits: host_hits,
            ..BaseCacheStats::default()
        },
        hits_by_source: CachedTokensBySource {
            host: host_hits,
            ..CachedTokensBySource::default()
        },
        ..PrefixCacheStats::default()
    });
    (local, connector)
}

/// Turns the scheduler's cumulative counters into the per-interval deltas
/// stamped on each outgoing batch.
#[derive(Default)]
pub(crate) struct StatsTracker {
    spec_decode: SpecDecodeCounters,
    prefix_cache: PrefixCacheCounters,
}

impl StatsTracker {
    /// Stats for the next outgoing batch. Advances the baselines, so a caller
    /// that declines to send after calling this drops only that interval's
    /// counter deltas.
    pub(crate) fn stats(&mut self, snapshot: &SchedulerMetrics) -> SchedulerStats {
        let mut stats = scheduler_stats_from(snapshot);
        stats.spec_decoding_stats = self.spec_decode_interval(snapshot);
        if let Some(cur) = &snapshot.prefix_cache {
            (stats.prefix_cache_stats, stats.connector_prefix_cache_stats) =
                prefix_cache_delta(&self.prefix_cache, cur);
            self.prefix_cache = *cur;
        } else {
            // Reset so a prefix cache that shows up later diffs from zero.
            self.prefix_cache = PrefixCacheCounters::default();
        }
        stats
    }

    /// The spec-decode delta, or `None` when nothing drafted.
    fn spec_decode_interval(&mut self, snapshot: &SchedulerMetrics) -> Option<SpecDecodingStats> {
        let Some(cur) = &snapshot.spec_decode else {
            // Reset so a drafter loaded later diffs from zero.
            self.spec_decode = SpecDecodeCounters::default();
            return None;
        };
        let delta = spec_decode_delta(&self.spec_decode, cur);
        self.spec_decode = *cur;
        // A zero-draft interval would divide by zero in the frontend's
        // acceptance-rate log, and every counter moves only inside
        // `observe_draft`, so dropping it loses nothing.
        (delta.num_drafts > 0).then_some(delta)
    }
}

/// The connected transport of one engine bridge: the vLLM EngineCore
/// handshake done, the output pump running.
struct BridgeLink {
    input: DealerSocket,
    output_tx: mpsc::UnboundedSender<EngineCoreOutputs>,
    child_tasks: tokio::task::JoinSet<(&'static str, Result<()>)>,
}

#[allow(clippy::too_many_arguments)]
async fn connect_link(
    input_address: &str,
    output_address: &str,
    engine_index: u32,
    data_parallel_size: u32,
    max_model_len: u32,
    kv_capacity: Option<infront_engine::KvCapacity>,
    engine_dead: Option<CancellationToken>,
    shutdown: &CancellationToken,
) -> Result<Option<BridgeLink>> {
    let engine_id = EngineId::from_engine_index(
        u16::try_from(engine_index).context("engine_index exceeds u16 EngineId")?,
    );
    let mut socket_options = SocketOptions::default();
    socket_options.peer_identity(PeerIdentity::try_from(engine_id)?);

    let (num_gpu_blocks, block_size, kv_cache_size_tokens, kv_cache_max_concurrency) =
        match kv_capacity {
            Some(c) => {
                // vLLM single-group concurrency: blocks / ceil(max_len / block_size).
                let blocks_per_req = u64::from(max_model_len).div_ceil(c.block_size as u64);
                (
                    c.total_blocks as u64,
                    c.block_size as u64,
                    Some(c.total_tokens() as u64),
                    Some(c.total_blocks as f64 / blocks_per_req as f64),
                )
            }
            None => (0, 16, None, None),
        };
    let ready = EngineCoreReadyResponse {
        max_model_len: u64::from(max_model_len),
        num_gpu_blocks,
        block_size,
        dp_stats_address: None,
        dtype: ModelDtype::BFloat16,
        vllm_version: "infront-local-bridge".to_string(),
        world_size: 1,
        effective_data_parallel_size: u64::from(data_parallel_size),
        tensor_parallel_size: 1,
        pipeline_parallel_size: 1,
        decode_context_parallel_size: 1,
        data_parallel_rank: engine_index,
        // Discovery metadata for gRPC control; HTTP completions do not admit
        // against these caps. Keep them above any sim concurrency we bench.
        max_num_seqs: 8192,
        max_num_batched_tokens: u64::from(max_model_len),
        instance_id: format!("infront-local-{engine_index}"),
        supports_lora: false,
        max_loras: 0,
        kv_cache_size_tokens,
        kv_cache_max_concurrency,
        effective_attention_block_size: kv_capacity.map(|c| c.block_size as u64),
        kv_events_config: None,
        weight_transfer_backend: None,
        enable_sleep_mode: false,
        supports_draft_weight_updates: false,
    };
    info!(
        "local engine {engine_index} KV capacity: {kv_capacity:?} -> \
         kv_cache_size_tokens={kv_cache_size_tokens:?} \
         kv_cache_max_concurrency={kv_cache_max_concurrency:?}"
    );
    let mut input = DealerSocket::with_options(socket_options);
    let mut output = PushSocket::new();
    let handshake = async {
        input
            .connect(input_address)
            .await
            .with_context(|| format!("failed to connect local engine input {input_address}"))?;
        input
            .send(ZmqMessage::from(encode_msgpack(&ready)?))
            .await
            .context("failed to send local engine ready response")?;
        output
            .connect(output_address)
            .await
            .with_context(|| format!("failed to connect local engine output {output_address}"))
    };
    // zeromq retries a refused connect for 30 s, which outlives a server that
    // failed its own startup and closed the listeners.
    tokio::select! {
        biased;
        () = shutdown.cancelled() => return Ok(None),
        handshake = handshake => handshake?,
    }

    let (output_tx, output_rx) = mpsc::unbounded_channel();
    let mut child_tasks = tokio::task::JoinSet::new();
    child_tasks.spawn(async move {
        (
            "output sender",
            output_loop(output, output_rx, engine_dead).await,
        )
    });

    Ok(Some(BridgeLink {
        input,
        output_tx,
        child_tasks,
    }))
}

async fn output_loop(
    mut output: PushSocket,
    mut output_rx: mpsc::UnboundedReceiver<EngineCoreOutputs>,
    engine_dead: Option<CancellationToken>,
) -> Result<()> {
    while let Some(outputs) = output_rx.recv().await {
        output
            .send(ZmqMessage::from(encode_msgpack(&outputs)?))
            .await
            .context("failed to send local engine output")?;
    }
    // Sent after the queued outputs: the client then fails every request it
    // still tracks, including ones the bridge never read.
    if engine_dead
        .as_ref()
        .is_some_and(CancellationToken::is_cancelled)
    {
        output
            .send(ZmqMessage::from(ENGINE_CORE_DEAD_SENTINEL.to_vec()))
            .await
            .context("failed to send the engine dead sentinel")?;
    }
    Ok(())
}

fn send_terminal_output(
    engine_index: u32,
    output_tx: &mpsc::UnboundedSender<EngineCoreOutputs>,
    request_id: String,
    finish_reason: EngineCoreFinishReason,
    stop_reason: Option<StopReason>,
    events: Option<Vec<EngineCoreEvent>>,
    prefill_stats: Option<PrefillStats>,
) -> Result<()> {
    send_outputs(
        output_tx,
        RequestBatchOutputs {
            engine_index,
            outputs: vec![engine_output(
                request_id.clone(),
                Vec::new(),
                None,
                Some(finish_reason),
                stop_reason,
                events,
                prefill_stats,
            )],
            finished_requests: Some(BTreeSet::from([request_id])),
            timestamp: now_secs_f64(),
            ..Default::default()
        }
        .into(),
    )
}

fn send_utility_response(
    engine_index: u32,
    output_tx: &mpsc::UnboundedSender<EngineCoreOutputs>,
    call_id: UtilityCallId,
    method_name: &str,
) -> Result<()> {
    let result = match method_name {
        "is_sleeping" | "is_paused" | "reset_prefix_cache" => rmpv::ext::to_value(false)?,
        "sleep" | "wake_up" | "reset_mm_cache" | "reset_encoder_cache" | "collective_rpc" => {
            rmpv::Value::Nil
        }
        _ => rmpv::Value::Nil,
    };

    send_outputs(
        output_tx,
        UtilityCallOutput {
            engine_index,
            timestamp: now_secs_f64(),
            output: UtilityOutput {
                call_id,
                failure_message: None,
                result: Some(UtilityResultEnvelope::without_type_info(result)),
            },
        }
        .into(),
    )
}

fn send_outputs(
    output_tx: &mpsc::UnboundedSender<EngineCoreOutputs>,
    outputs: EngineCoreOutputs,
) -> Result<()> {
    output_tx
        .send(outputs)
        .map_err(|_| anyhow::anyhow!("local engine output channel closed"))
}

fn engine_output(
    request_id: String,
    new_token_ids: Vec<u32>,
    new_logprobs: Option<MaybeWireLogprobs>,
    finish_reason: Option<EngineCoreFinishReason>,
    stop_reason: Option<StopReason>,
    events: Option<Vec<EngineCoreEvent>>,
    prefill_stats: Option<PrefillStats>,
) -> EngineCoreOutput {
    EngineCoreOutput {
        request_id,
        new_token_ids,
        new_logprobs,
        new_prompt_logprobs_tensors: None,
        pooling_output: None,
        finish_reason,
        stop_reason,
        events,
        kv_transfer_params: None,
        ec_transfer_params: None,
        trace_headers: None,
        prefill_stats: prefill_stats.map(Box::new),
        routed_experts: None,
        num_nans_in_logits: 0,
        mm_cache_miss_hashes: None,
        new_sampling_mask: None,
        spec_decode_metrics: None,
        prompt_token_id_logprobs: None,
        error: None,
    }
}

fn now_secs_f64() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_secs_f64()
}

pub(crate) fn local_ipc_namespace() -> Result<PathBuf> {
    let base_dir =
        std::env::var_os("INFRONT_IPC_DIR").map_or_else(|| PathBuf::from("/tmp"), PathBuf::from);
    let uuid = uuid::Uuid::new_v4().to_string();
    let path = base_dir.join(format!("infront-{}-{}", std::process::id(), &uuid[..8]));
    std::fs::create_dir_all(&path)
        .with_context(|| format!("failed to create IPC namespace {}", path.display()))?;
    Ok(path)
}

pub(crate) fn ipc_endpoint(namespace: &Path, name: &str) -> String {
    format!("ipc://{}", namespace.join(name).to_string_lossy())
}

mod stepped;

pub(crate) use stepped::SteppedEngineBridge;

#[cfg(test)]
mod tests;
