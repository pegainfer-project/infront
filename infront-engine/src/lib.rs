//! The engine contract: what a scheduler implements and what a protocol stack
//! consumes. No CUDA, no HTTP, no wire format.
//!
//! - `step` — the wire types: `Request`, `RequestId`, `StepOutputs` with one
//!   flat `RequestUpdate` per touched request per step.
//! - `ledger` — `RequestLedger`, the account book of live requests:
//!   schedulers write verdicts and tokens against it by `RequestId`, and it
//!   enforces terminal-exactly-once at the call site.
//! - `request_lifecycle` — the pieces that carry a request outside the
//!   ledger's reach: the submission envelope, deferred finishes, and the abort
//!   control.
//! - `metrics` — `SchedulerMetrics`, the per-iteration snapshot a scheduler
//!   republishes about itself.
//! - `wiring` — `scheduler_pair` wiring, `SchedulerHandle`, and the
//!   `Engine`/`EngineInfo` bundle an engine hands to the protocol stack.
//! - `driver` — the `Scheduler` trait and the polling `drive` loop.
//! - `control` — the LoRA adapter capability: vocabulary + client, outside
//!   the scheduler contract (engines mint their own channel before spawn).
//!
//! Everything is re-exported flat at the crate root.

mod control;
mod driver;
mod kv;
mod ledger;
mod metrics;
mod request_lifecycle;
mod sampler;
mod step;
mod stop;
#[cfg(feature = "testing")]
pub mod testing;
pub mod tracing_state;
mod wiring;

pub use control::*;
pub use driver::*;
pub use kv::*;
pub use ledger::*;
pub use metrics::*;
pub use request_lifecycle::*;
pub use sampler::*;
pub use step::*;
pub use stop::*;
pub use wiring::*;
