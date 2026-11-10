//! The order execution kernel.
//!
//! This is the semantic heart of the engine and the densest code in it. Every
//! simulation loop routes through the same [`execute_order`], calling it in a
//! tight loop over a column. The signature stays callable for a single bar in
//! isolation, with no dependence on the rest of the column being known, so any
//! caller can drive it one order at a time.
//!
//! # Two kinds of failure
//!
//! A *rejected* order is a normal outcome: it returns an [`OrderResult`] with
//! [`OrderStatus::Rejected`] and a [`StatusInfo`] explaining why, leaving the
//! state untouched. An *invariant violation* — negative cash, a non-finite
//! position, a long-only column holding a short — is an `Err`. Conflating the
//! two is how a simulation silently produces a plausible, wrong equity curve.
//!
//! # Determinism
//!
//! [`execute_order`] is pure. Random rejection takes its draw as an argument
//! rather than reaching for a global RNG, so a parameter sweep can seed each
//! column reproducibly and still run in parallel.
//!
//! # Verification
//!
//! Correctness here means matching recorded expected outputs, not merely
//! looking right. Two committed fixture sets do that at different
//! granularities:
//!
//! - `tests/golden.rs` replays single calls — every reachable rejection
//!   reason, every size type, every direction, the invariant violations and
//!   the float tolerance boundary. It localizes a divergence to a stage of the
//!   size cascade.
//! - `tests/golden_sequence.rs` replays chains, threading its own outputs
//!   forward. It catches state-threading bugs and residues that only become
//!   visible a step later.
//!
//! Both fixture sets were shaped by mutation testing rather than by guessing
//! at coverage. The tolerance-boundary cases exist because replacing the
//! tolerant affordability check with a plain `<=` originally passed
//! everything; the sequence amplifiers exist because the zero-snapping
//! machinery was otherwise unverified.

mod kernel;
mod types;

pub use kernel::{ValidatedOrder, execute_order, execute_validated};
pub use types::{ExecuteOrderState, Order, OrderResult, ProcessOrderState};

// Re-exported so callers matching on a result do not need `alphax-model` too.
pub use alphax_model::{
    AccountType, Direction, LeverageMode, MarginBasis, MarginMode, OrderSide, OrderStatus,
    SizeType, StatusInfo,
};
