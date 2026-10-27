//! Submission envelope and abort handle for one request's lifetime.
//!
//! The scheduler side of the contract deals in plain [`RequestId`]s against
//! the [`super::RequestLedger`] — no per-request typestate handles cross the
//! `Scheduler` trait. What lives here are the pieces that carry a request
//! *outside* the ledger's reach and therefore need their own answer-on-drop
//! guarantee:
//!
//! - [`RequestEnvelope`] — the channel envelope between
//!   [`super::SchedulerHandle::submit`] and the driver's
//!   [`super::RequestLedger::register`]. It covers the window where the
//!   request exists but no ledger account does yet: sitting in the submit
//!   channel, or sent to a scheduler thread that already exited.
//! - [`RequestControl`] — the frontend's per-request abort flag.
//!
//! The envelope holds `Option<inner>` for one reason only: Rust's `Drop`
//! cannot move fields out, so a consuming transition `take`s the inner and
//! the drop bomb fires on `Some`. An unanswered request must surface as a
//! `Failed` terminal, never a client hang.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::time::Instant;

use super::step::Request;
use super::step::RequestId;
use super::step::RequestUpdate;
use super::step::StepOutputs;
use super::step::Terminal;

/// The scheduler-to-frontend stream: one message per step. The ledger holds
/// the sender; drop bombs hold clones for solo sends.
pub(crate) type StepSender = tokio::sync::mpsc::UnboundedSender<StepOutputs>;
pub type StepReceiver = tokio::sync::mpsc::UnboundedReceiver<StepOutputs>;

/// One submitted request in flight between the frontend and the driver.
/// Minted by [`super::SchedulerHandle::submit`]; consumed (and thereby
/// disarmed) by [`super::RequestLedger::register`], which opens the ledger
/// account that takes over the answer-on-drop duty.
pub(crate) struct RequestEnvelope {
    /// `Some` until [`Self::consume`]; `Drop` fires on `Some`.
    inner: Option<EnvelopeInner>,
}

pub(crate) struct EnvelopeInner {
    pub(crate) id: RequestId,
    pub(crate) abort: Arc<AtomicBool>,
    pub(crate) tx: StepSender,
    pub(crate) request: Request,
    pub(crate) queued_at: Instant,
}

impl RequestEnvelope {
    pub(crate) fn new(
        id: RequestId,
        abort: Arc<AtomicBool>,
        tx: StepSender,
        request: Request,
        queued_at: Instant,
    ) -> Self {
        Self {
            inner: Some(EnvelopeInner {
                id,
                abort,
                tx,
                request,
                queued_at,
            }),
        }
    }

    pub(crate) fn consume(mut self) -> EnvelopeInner {
        self.inner.take().expect("submission consumed twice")
    }
}

/// A submission dropped unconsumed never reached a ledger account: the
/// scheduler is gone (send failed, or the channel fell with the driver).
/// Answer the request so the client observes a terminal instead of a hang.
impl Drop for RequestEnvelope {
    fn drop(&mut self) {
        let Some(inner) = self.inner.take() else {
            return;
        };
        let mut update = RequestUpdate::empty(inner.id);
        update.terminal = Some(Terminal::Failed {
            message: "request dropped by the engine before it was answered".to_string(),
            prompt_tokens: inner.request.prompt_tokens.len(),
            completion_tokens: 0,
        });
        let _ = inner.tx.send(StepOutputs {
            updates: vec![update],
        });
    }
}

/// The frontend's per-request cancel handle, returned by
/// [`super::SchedulerHandle::submit`]. Flipping the flag is the only abort
/// mechanism: the scheduler observes it through
/// [`super::RequestLedger::is_aborted`] and retires the request reactively;
/// no channel is closed.
pub struct RequestControl {
    id: RequestId,
    abort: Arc<AtomicBool>,
}

impl RequestControl {
    pub(crate) fn new(id: RequestId, abort: Arc<AtomicBool>) -> Self {
        Self { id, abort }
    }

    pub fn id(&self) -> RequestId {
        self.id
    }

    /// Mark the request aborted. `Release` orders the store after the
    /// caller's own teardown of per-request state, mirroring the scheduler's
    /// `Acquire` load.
    pub fn abort(&self) {
        self.abort.store(true, Ordering::Release);
    }
}
