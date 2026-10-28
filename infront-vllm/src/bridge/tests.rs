use infront_engine::PrefixCacheCounters;
use infront_engine::SchedulerMetrics;
use infront_engine::SpecDecodeCounters;
use vllm_engine_core_client::protocol::stats::PrefixCacheStats;

use super::*;

/// The queue gauges and the fractional KV usage the frontend records into
/// Prometheus come straight off the snapshot.
#[test]
fn snapshots_become_scheduler_gauges() {
    let mut tracker = StatsTracker::default();
    let stats = tracker.stats(&SchedulerMetrics {
        kv_used_blocks: 25,
        kv_total_blocks: 100,
        num_running_reqs: 2,
        num_waiting_reqs: 1,
        spec_decode: None,
        prefix_cache: None,
    });
    assert_eq!(stats.num_running_reqs, 2);
    assert_eq!(stats.num_waiting_reqs, 1);
    assert!((stats.kv_cache_usage - 0.25).abs() < 1e-9);

    let stats = tracker.stats(&SchedulerMetrics::default());
    assert_eq!(stats.num_running_reqs, 0);
    assert_eq!(stats.kv_cache_usage.to_bits(), 0.0_f64.to_bits());
}

/// The tracker diffs the scheduler's cumulative totals into the per-interval
/// deltas the frontend `inc_by`s, and attaches `spec_decoding_stats` only on
/// intervals that actually ran a draft step — an idle interval would divide by
/// a zero `num_drafts` in the frontend's acceptance-rate log.
#[test]
fn spec_stats_are_per_interval_deltas_that_skip_idle_intervals() {
    let mut totals = SpecDecodeCounters::new(2).expect("K within bounds");
    totals.observe_draft(2, 1);
    let initial = SchedulerMetrics {
        spec_decode: Some(totals),
        ..SchedulerMetrics::default()
    };
    let mut tracker = StatsTracker::default();

    // First publish diffs against zero, so it carries the whole cumulative.
    let first = tracker.stats(&initial);
    let spec = first
        .spec_decoding_stats
        .expect("first interval has a draft");
    assert_eq!(spec.num_spec_tokens, 2);
    assert_eq!(spec.num_drafts, 1);
    assert_eq!(spec.num_accepted_tokens, 1);
    // Widened to the drafter's K, not the array's.
    assert_eq!(spec.num_accepted_tokens_per_pos, vec![1, 0]);

    // A no-op republish (same totals) is an idle interval: no spec stats.
    let idle = tracker.stats(&SchedulerMetrics {
        spec_decode: Some(totals),
        num_running_reqs: 1,
        ..SchedulerMetrics::default()
    });
    assert!(idle.spec_decoding_stats.is_none());

    // Two verify steps land in one snapshot — the watch coalesced them. The
    // delta has to cover the whole gap, per position as well as in total,
    // which is the reason the transport carries totals instead of deltas.
    totals.observe_draft(2, 2);
    totals.observe_draft(2, 2);
    let resumed = tracker.stats(&SchedulerMetrics {
        spec_decode: Some(totals),
        ..SchedulerMetrics::default()
    });
    let spec = resumed.spec_decoding_stats.expect("second draft interval");
    assert_eq!(spec.num_drafts, 2);
    assert_eq!(spec.num_draft_tokens, 4);
    assert_eq!(spec.num_accepted_tokens, 4);
    assert_eq!(spec.num_accepted_tokens_per_pos, vec![2, 2]);
}

/// Prefix-cache totals become per-interval deltas that the frontend `inc_by`s
/// into `vllm:prefix_cache_{queries,hits}_total` for the device tier and, via
/// the connector stats, `vllm:external_prefix_cache_*` and
/// `prompt_tokens_cached_by_source{source="host"}` for the host tier, or
/// `source="external_unspecified"` when the engine cannot attribute them. An
/// engine restart (totals going backwards) diffs from zero instead of going
/// negative, and an engine without a prefix cache exports zeros as before.
#[test]
fn prefix_cache_stats_are_per_interval_deltas_across_a_restart() {
    let mut totals = PrefixCacheCounters {
        queries: 4096,
        ..PrefixCacheCounters::default()
    };
    let initial = SchedulerMetrics {
        prefix_cache: Some(totals),
        ..SchedulerMetrics::default()
    };
    let mut tracker = StatsTracker::default();

    // First publish diffs against zero, so it carries the whole cumulative.
    let first = tracker.stats(&initial);
    assert_eq!(first.prefix_cache_stats.base.queries, 4096);
    assert_eq!(first.prefix_cache_stats.base.hits, 0);
    // No host lookup yet: no external stats.
    assert_eq!(first.connector_prefix_cache_stats, None);

    // A later turn: 4352 tokens looked up, 4096 of them served from cache;
    // the 1280 that missed device were looked up in host, which had 1024.
    totals.queries += 4352;
    totals.hits += 4096;
    totals.host_queries += 1280;
    totals.host_hits += 1024;
    let turn = tracker.stats(&SchedulerMetrics {
        prefix_cache: Some(totals),
        ..SchedulerMetrics::default()
    });
    let local = &turn.prefix_cache_stats;
    assert_eq!(local.base.queries, 4352);
    assert_eq!(local.base.hits, 3072);
    let host = turn
        .connector_prefix_cache_stats
        .as_ref()
        .expect("host hits ride the connector stats");
    assert_eq!(host.base.queries, 1280);
    assert_eq!(host.base.hits, 1024);
    assert_eq!(host.hits_by_source.host, 1024);
    // What the frontend adds to `prompt_tokens_cached_by_source`: local hits
    // as `device`, connector hits by their source. Together, every hit once.
    let mut by_source = host.hits_by_source.counts();
    by_source[0] += local.base.hits;
    assert_eq!(by_source, [3072, 1024, 0, 0, 0]);
    assert_eq!(by_source.iter().sum::<u64>(), 4096);

    // Another turn whose connector restores 512 tokens from a tier it cannot
    // name: the hits are external, labeled `external_unspecified`, and stay
    // out of both the local hits and the `host` label.
    totals.queries += 2048;
    totals.hits += 1536;
    totals.host_queries += 1024;
    totals.external_unspecified_hits += 512;
    let unspecified = tracker.stats(&SchedulerMetrics {
        prefix_cache: Some(totals),
        ..SchedulerMetrics::default()
    });
    assert_eq!(unspecified.prefix_cache_stats.base.queries, 2048);
    assert_eq!(unspecified.prefix_cache_stats.base.hits, 1024);
    let external = unspecified
        .connector_prefix_cache_stats
        .as_ref()
        .expect("unattributed hits ride the connector stats");
    assert_eq!(external.base.queries, 1024);
    assert_eq!(external.base.hits, 512);
    assert_eq!(external.hits_by_source.external_unspecified, 512);
    assert_eq!(external.hits_by_source.host, 0);
    let mut by_source = external.hits_by_source.counts();
    by_source[0] += unspecified.prefix_cache_stats.base.hits;
    assert_eq!(by_source, [1024, 0, 0, 0, 512]);
    assert_eq!(by_source.iter().sum::<u64>(), 1536);

    // The engine restarts: totals fall below the baseline. The interval
    // counts from zero rather than underflowing or dropping the new work.
    let restarted = tracker.stats(&SchedulerMetrics {
        prefix_cache: Some(PrefixCacheCounters {
            queries: 300,
            hits: 200,
            ..PrefixCacheCounters::default()
        }),
        ..SchedulerMetrics::default()
    });
    assert_eq!(restarted.prefix_cache_stats.base.queries, 300);
    assert_eq!(restarted.prefix_cache_stats.base.hits, 200);
    assert_eq!(restarted.connector_prefix_cache_stats, None);

    // No prefix cache: the default stats, exactly what was exported before.
    let none = tracker.stats(&SchedulerMetrics::default());
    assert_eq!(none.prefix_cache_stats, PrefixCacheStats::default());
    assert_eq!(none.connector_prefix_cache_stats, None);
}

#[test]
fn unsupported_utility_calls_fail_instead_of_faking_success() {
    let (tx, mut rx) = mpsc::unbounded_channel();
    let mut answer = |method: &str| {
        send_utility_response(0, &tx, UtilityCallId::default(), method).unwrap();
        let EngineCoreOutputs::Utility(call) = rx.try_recv().expect("an answer") else {
            panic!("expected a utility answer");
        };
        call.output
    };
    for method in [
        "sleep",
        "wake_up",
        "pause_generation",
        "collective_rpc",
        "add_lora",
    ] {
        let output = answer(method);
        assert!(output.result.is_none(), "{method}");
        assert!(
            output
                .failure_message
                .is_some_and(|message| message.contains(method)),
            "{method}"
        );
    }
    for method in ["is_sleeping", "is_scheduler_paused", "reset_mm_cache"] {
        let output = answer(method);
        assert!(output.failure_message.is_none(), "{method}");
        assert!(output.result.is_some(), "{method}");
    }
}
