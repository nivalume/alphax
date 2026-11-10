//! Generating, cleaning and ranking entry and exit signals.
//!
//! The engine *consumes* entries and exits. This crate is where they come
//! from — which until now was the caller's problem, and is the difference
//! between an engine that runs a backtest and one you can write a strategy
//! with.
//!
//! Four things, and they are four different jobs:
//!
//! - [`clean`] turns a raw condition array into an alternating sequence. A
//!   thresholded factor fires on every bar the condition holds, not on the
//!   bar it starts holding; feeding that to a simulation is a column of
//!   redundant orders.
//! - [`generate`] builds exits from entries, or both from scratch, through a
//!   choice function. In Rust the callback monomorphizes away and costs
//!   nothing.
//! - [`rank`] says where a signal sits within its run, which is what "enter
//!   on the third consecutive signal" needs.
//! - [`ranges`] reads a signal array as spans rather than bars, so "how long
//!   does this rule stay on" has an answer in the same record shape the
//!   order and trade layers use.
//!
//! # Randomized signals: seeded
//!
//! [`random`] generates random entries and exits for null-model comparison.
//! Each draw is a function of `(seed, row, col)` alone, so a run is
//! reproducible and a column does not change when others are added beside it.
//! No external random stream is involved, so there is no golden file to
//! replay; they are tested by their properties — exact counts, strict
//! alternation, the rate a probability asks for — instead.
//!
//! What a careful null model needs is a *matched* one — the same number of
//! signals, in the same runs, reshuffled — which is still a design question.

pub mod clean;
pub mod generate;
pub mod labels;
pub mod random;
pub mod ranges;
pub mod rank;

pub use clean::{FirstSignal, clean_enex, clean_enex_1d};
pub use generate::{ExitWindow, exit_after, exit_on, generate_alternating, generate_exits};
pub use labels::{fixed_horizon_returns, pivot_labels, trend_labels, triple_barrier};
pub use random::{random_by_prob, random_entries_exits, random_n};
pub use ranges::{
    RangeRecords, RangeStatus, between_partition_ranges, between_ranges, partition_ranges,
};
pub use rank::{
    Gaps, LeadingRun, normalized_average_index, nth_index, partition_rank, signal_rank,
};
