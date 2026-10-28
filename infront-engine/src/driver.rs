//! The model-side runtime trait and the polling loop that drives it.
//!
//! The loop lives here — once, for every engine — so its conventions
//! (drain order, one metrics publish and one commit per iteration, shutdown
//! and fatal handling) are code, not per-crate discipline. An engine's
//! runtime obligation is exactly the three [`Scheduler`] methods. Anything
//! beyond them is not the contract's business: a scheduler that serves an
//! extra capability captures its own channel before spawn and drains it
//! inside `step`.

use std::time::Duration;

use super::ledger::RequestLedger;
use super::metrics::SchedulerMetrics;
use super::request_lifecycle::RequestEnvelope;
use super::step::QueuedRequest;
use super::wiring::LiveScheduler;
use super::wiring::MetricsPublisher;
use super::wiring::SchedulerBackend;
use super::wiring::scheduler_pair;

/// One scheduler's runtime behavior. Runs on a dedicated OS thread spawned
/// by [`spawn_scheduler`]; `Send` suffices.
pub trait Scheduler: Send {
    /// Take ownership of one submitted request. Ownership transfer only —
    /// every verdict (admit/reject/retire) is written to the ledger from
    /// [`Self::step`], the single emission site. Nothing is lost by
    /// deferring: the driver commits once per iteration, so a verdict
    /// recorded in `step` lands in the same commit a submit-time verdict
    /// would have.
    fn submit(&mut self, request: QueuedRequest);

    /// Advance one step: admission, GPU work, and per-request ledger writes.
    /// While [`Self::metrics`] reports running or waiting requests the driver
    /// loops over this without pause, so in-flight completions from the
    /// scheduler's own worker threads are picked up within a step. An idle
    /// scheduler is stepped again on its next submission or after
    /// [`IDLE_PARK`], whichever comes first. Recoverable
    /// execution failures are the scheduler's to absorb (fail the touched
    /// requests, keep serving); `Err` means the engine is beyond use. The
    /// driver then writes off every open account with the error and winds
    /// down.
    fn step(&mut self, ledger: &mut RequestLedger) -> anyhow::Result<()>;

    /// Metrics snapshot; the driver publishes it once per iteration. Work
    /// that only [`Self::step`] can collect counts as running or waiting, or
    /// the driver parks over it.
    fn metrics(&self) -> SchedulerMetrics;
}

/// Longest an idle driver waits for a submission before stepping again. Bounds
/// how late an idle scheduler sees work that does not arrive as a submission
/// (e.g. its own control channel).
pub const IDLE_PARK: Duration = Duration::from_millis(1);

/// Spawn one scheduler: mint the wiring, start the driver thread, return the
/// frontend end. `name` names the OS thread (shows in `top`/gdb).
pub fn spawn_scheduler<S: Scheduler + 'static>(name: &str, scheduler: S) -> LiveScheduler {
    let (handle, backend) = scheduler_pair();
    let join = std::thread::Builder::new()
        .name(name.to_string())
        .spawn(move || drive(scheduler, backend))
        .expect("failed to spawn scheduler thread");
    LiveScheduler { handle, join }
}

/// The polling loop. Exits when the frontend is gone (submission channel
/// disconnected) and the scheduler reports itself drained, or when `step`
/// reports a fatal error. After an idle iteration (nothing running or
/// waiting) the driver parks on the submission channel for up to
/// [`IDLE_PARK`]; busy iterations never pause.
pub fn drive<S: Scheduler>(scheduler: S, backend: SchedulerBackend) {
    let SchedulerBackend {
        submissions,
        ledger,
        metrics,
        exit_guard,
    } = backend;
    run(scheduler, submissions, ledger, &metrics);
    // Everything that can still answer a request (the ledger's write-off, the
    // envelopes left in the channel) has gone with `run`, returned or
    // unwound, so `exited` resolves after the last of them was sent.
    drop(exit_guard);
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "envelopes still queued are answered when the receiver drops, which must happen inside `run`"
)]
fn run<S: Scheduler>(
    mut scheduler: S,
    submissions: crossbeam_channel::Receiver<RequestEnvelope>,
    mut ledger: RequestLedger,
    metrics: &MetricsPublisher,
) {
    let mut submissions_open = true;
    let mut idle = false;
    loop {
        if idle {
            match submissions.recv_timeout(IDLE_PARK) {
                Ok(envelope) => {
                    let queued = ledger.register(envelope);
                    scheduler.submit(queued);
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    submissions_open = false;
                }
            }
        }
        loop {
            match submissions.try_recv() {
                Ok(envelope) => {
                    let queued = ledger.register(envelope);
                    scheduler.submit(queued);
                }
                Err(crossbeam_channel::TryRecvError::Empty) => break,
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    submissions_open = false;
                    break;
                }
            }
        }
        let step = scheduler.step(&mut ledger);
        let snapshot = scheduler.metrics();
        let metrics_changed = metrics.publish(&snapshot);
        ledger.commit_step(metrics_changed);
        if let Err(error) = step {
            // The ledger holds an account for every unanswered request, so
            // the write-off reaches them all — including any the scheduler
            // lost track of — with the real error attached.
            log::error!("scheduler fatal, engine winding down: {error:#}");
            ledger.fail_all(&format!("engine fatal: {error:#}"));
            ledger.commit_step(false);
            return;
        }
        idle = snapshot.num_running_reqs == 0 && snapshot.num_waiting_reqs == 0;
        if idle && !submissions_open {
            log::info!("scheduler drained after frontend shutdown, exiting");
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use super::super::step::Request;
    use super::super::step::RequestId;
    use super::super::step::Terminal;
    use super::super::stop::StopPolicy;
    use super::*;
    use crate::FinishReason;

    /// Emits one token per step per request and finishes at `max_tokens`.
    #[derive(Default)]
    struct EchoScheduler {
        queued: Vec<(RequestId, usize)>,
        running: Vec<(RequestId, usize)>,
        fatal_next_step: bool,
    }

    impl Scheduler for EchoScheduler {
        fn submit(&mut self, request: QueuedRequest) {
            self.queued.push((request.id, request.request.max_tokens));
        }

        fn step(&mut self, ledger: &mut RequestLedger) -> anyhow::Result<()> {
            if self.fatal_next_step && !(self.queued.is_empty() && self.running.is_empty()) {
                anyhow::bail!("injected fatal");
            }
            for (id, max_tokens) in self.queued.drain(..) {
                ledger.admit(id);
                self.running.push((id, max_tokens));
            }
            let mut still_running = Vec::new();
            for (id, max_tokens) in self.running.drain(..) {
                if ledger.is_aborted(id) {
                    ledger.retire(id);
                    continue;
                }
                let next = ledger.completion_tokens(id) as u32;
                ledger.push_tokens(id, &[next], &[]);
                if ledger.completion_tokens(id) >= max_tokens {
                    ledger.finish(id, FinishReason::Length);
                } else {
                    still_running.push((id, max_tokens));
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

    fn request(max_tokens: usize) -> Request {
        Request {
            prompt_tokens: vec![1, 2],
            params: crate::sampler::SamplingParams::default(),
            history_tokens: None,
            history_marks: Vec::new(),
            stop_policy: StopPolicy::default(),
            max_tokens,
            logprobs: None,
            prompt_logprobs: None,
        }
    }

    #[test]
    fn driven_engine_streams_and_drains_on_shutdown() {
        let partition = spawn_scheduler("test-echo", EchoScheduler::default());
        let mut handle = partition.handle;
        let mut steps = handle.take_steps().expect("step stream");

        let _control = handle.submit(request(3));
        let mut tokens = Vec::new();
        let mut terminal = None;
        while terminal.is_none() {
            let step = steps.blocking_recv().expect("step message");
            for update in step.updates {
                assert!(update.scheduled.is_some() || !tokens.is_empty());
                tokens.extend(update.tokens);
                if let Some(t) = update.terminal {
                    terminal = Some(t);
                }
            }
        }
        assert_eq!(tokens, vec![0, 1, 2]);
        assert!(matches!(
            terminal,
            Some(Terminal::Finished {
                reason: FinishReason::Length,
                prompt_tokens: 2,
                completion_tokens: 3,
            })
        ));

        // Dropping the handle disconnects the submission channel; a drained
        // scheduler exits.
        drop(handle);
        partition.join.join().expect("driver thread exits cleanly");
    }

    #[test]
    fn cancellation_notifies_metrics_once_without_request_output() {
        struct PausedScheduler {
            queued: Vec<RequestId>,
            running: Vec<RequestId>,
            entered: std::sync::mpsc::Sender<()>,
            resume: std::sync::mpsc::Receiver<()>,
        }

        impl Scheduler for PausedScheduler {
            fn submit(&mut self, request: QueuedRequest) {
                self.queued.push(request.id);
            }

            fn step(&mut self, ledger: &mut RequestLedger) -> anyhow::Result<()> {
                let _ = self.entered.send(());
                let _ = self.resume.recv();
                for id in self.queued.drain(..) {
                    ledger.admit(id);
                    self.running.push(id);
                }
                self.running.retain(|&id| {
                    if ledger.is_aborted(id) {
                        ledger.retire(id);
                        false
                    } else {
                        true
                    }
                });
                Ok(())
            }

            fn metrics(&self) -> SchedulerMetrics {
                SchedulerMetrics {
                    num_running_reqs: self.running.len() as u64,
                    ..SchedulerMetrics::default()
                }
            }
        }

        let (mut handle, backend) = scheduler_pair();
        let mut steps = handle.take_steps().unwrap();
        let control = handle.submit(request(8));
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (resume_tx, resume_rx) = std::sync::mpsc::channel();
        let join = std::thread::spawn(move || {
            drive(
                PausedScheduler {
                    queued: Vec::new(),
                    running: Vec::new(),
                    entered: entered_tx,
                    resume: resume_rx,
                },
                backend,
            );
        });
        let next_step = || {
            entered_rx
                .recv_timeout(std::time::Duration::from_secs(5))
                .expect("driver advanced to its next step");
        };

        next_step();
        resume_tx.send(()).unwrap();
        next_step();
        steps.try_recv().expect("admission step");

        control.abort();
        resume_tx.send(()).unwrap();
        next_step();
        assert!(
            steps
                .try_recv()
                .expect("cancellation load change")
                .updates
                .is_empty()
        );
        assert_eq!(handle.metrics().num_running_reqs, 0);

        resume_tx.send(()).unwrap();
        next_step();
        assert!(matches!(
            steps.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
        drop(handle);
        drop(resume_tx);
        join.join().expect("driver drains after cancellation");
    }

    #[test]
    fn idle_driver_parks_between_steps() {
        struct IdleScheduler(Arc<AtomicUsize>);

        impl Scheduler for IdleScheduler {
            fn submit(&mut self, _request: QueuedRequest) {}

            fn step(&mut self, _ledger: &mut RequestLedger) -> anyhow::Result<()> {
                self.0.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }

            fn metrics(&self) -> SchedulerMetrics {
                SchedulerMetrics::default()
            }
        }

        let steps = Arc::new(AtomicUsize::new(0));
        let window = Duration::from_millis(200);
        let partition = spawn_scheduler("test-idle", IdleScheduler(Arc::clone(&steps)));
        std::thread::sleep(window);
        let observed = steps.load(Ordering::Relaxed);
        drop(partition.handle);
        partition
            .join
            .join()
            .expect("idle driver exits on shutdown");

        let parks = (window.as_micros() / IDLE_PARK.as_micros()) as usize;
        assert!(
            observed <= 2 * parks,
            "an idle driver must park between steps: {observed} steps in {window:?}"
        );
    }

    #[test]
    fn exited_resolves_only_after_the_scheduler_is_gone() {
        struct SlowToDrop(Arc<AtomicBool>);

        impl Drop for SlowToDrop {
            fn drop(&mut self) {
                std::thread::sleep(Duration::from_millis(100));
                self.0.store(true, Ordering::Release);
            }
        }

        impl Scheduler for SlowToDrop {
            fn submit(&mut self, _request: QueuedRequest) {}

            fn step(&mut self, _ledger: &mut RequestLedger) -> anyhow::Result<()> {
                anyhow::bail!("injected fatal")
            }

            fn metrics(&self) -> SchedulerMetrics {
                SchedulerMetrics::default()
            }
        }

        let dropped = Arc::new(AtomicBool::new(false));
        let partition = spawn_scheduler("test-exit-order", SlowToDrop(Arc::clone(&dropped)));
        tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("runtime")
            .block_on(partition.handle.exited());
        assert!(
            dropped.load(Ordering::Acquire),
            "exited resolved while the scheduler could still answer requests"
        );
        partition.join.join().expect("driver exits after fatal");
    }

    #[test]
    fn a_panicking_step_still_ships_what_it_answered() {
        struct PanicsAfterFinishing;

        impl Scheduler for PanicsAfterFinishing {
            fn submit(&mut self, _request: QueuedRequest) {}

            fn step(&mut self, ledger: &mut RequestLedger) -> anyhow::Result<()> {
                let (finished, open) = (RequestId::new(0), RequestId::new(1));
                ledger.admit(finished);
                ledger.admit(open);
                ledger.push_tokens(finished, &[7], &[]);
                ledger.finish(finished, FinishReason::Length);
                panic!("injected panic");
            }

            fn metrics(&self) -> SchedulerMetrics {
                SchedulerMetrics::default()
            }
        }

        let (mut handle, backend) = scheduler_pair();
        let mut steps = handle.take_steps().expect("step stream");
        let _finished = handle.submit(request(1));
        let _open = handle.submit(request(8));
        let join = std::thread::spawn(move || drive(PanicsAfterFinishing, backend));
        assert!(join.join().is_err(), "the step panics");

        let step = steps.try_recv().expect("the teardown sweep");
        let terminal = |id: u64| {
            let update = step
                .updates
                .iter()
                .find(|update| update.id == RequestId::new(id))
                .expect("an update per request");
            (update.tokens.clone(), update.terminal.as_ref())
        };
        assert!(matches!(
            terminal(0),
            (tokens, Some(Terminal::Finished { reason: FinishReason::Length, .. })) if tokens == [7]
        ));
        assert!(matches!(terminal(1), (_, Some(Terminal::Failed { .. }))));
    }

    #[test]
    fn fatal_step_fails_in_flight_requests_with_the_error() {
        let partition = spawn_scheduler(
            "test-fatal",
            EchoScheduler {
                fatal_next_step: true,
                ..EchoScheduler::default()
            },
        );
        let mut handle = partition.handle;
        let mut steps = handle.take_steps().expect("step stream");

        let _control = handle.submit(request(100));
        let mut failure_message = None;
        while let Some(step) = steps.blocking_recv() {
            for update in step.updates {
                if let Some(Terminal::Failed { message, .. }) = update.terminal {
                    failure_message = Some(message);
                }
            }
            if failure_message.is_some() {
                break;
            }
        }
        let message = failure_message.expect("in-flight request must be answered on fatal");
        assert!(
            message.contains("injected fatal"),
            "the write-off must carry the real error: {message}"
        );
        partition
            .join
            .join()
            .expect("driver thread exits after fatal");
    }
}
