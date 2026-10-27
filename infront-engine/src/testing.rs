//! Hooks for protocol-stack tests: play the driver's part on a
//! [`SchedulerBackend`] without running a scheduler thread.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use crate::QueuedRequest;
use crate::RequestControl;
use crate::RequestId;
use crate::RequestLedger;
use crate::RequestUpdate;
use crate::SchedulerBackend;
use crate::SchedulerMetrics;

pub fn empty_update(id: RequestId) -> RequestUpdate {
    RequestUpdate::empty(id)
}

pub fn request_control(id: RequestId, abort: Arc<AtomicBool>) -> RequestControl {
    RequestControl::new(id, abort)
}

impl SchedulerBackend {
    pub fn publish_metrics(&self, snapshot: &SchedulerMetrics) {
        self.metrics.publish(snapshot);
    }

    /// Open a ledger account for the next pending submission, as the driver
    /// does before handing it to the scheduler.
    pub fn register_next(&mut self) -> Option<QueuedRequest> {
        let envelope = self.submissions.try_recv().ok()?;
        Some(self.ledger.register(envelope))
    }

    pub fn has_pending_submission(&self) -> bool {
        !self.submissions.is_empty()
    }

    pub fn ledger(&self) -> &RequestLedger {
        &self.ledger
    }
}
