//! `alphax` — a parallel, factor-driven vectorized backtesting engine.
//!
//! This crate is the façade: it re-exports the workspace's crates and
//! provides a [`prelude`]. The short version is:
//!
//! - Factor expressions produce a `(time x symbol)` panel.
//! - Signals derived from it fan out across a column axis where parameters
//!   and symbols are the same dimension.
//! - Account groups are independent; symbols within a cash-sharing group
//!   execute sequentially, and parameter groups run in parallel.
//! - Every simulation emits the same record schemas, so one analysis layer
//!   serves all of them and their results can be compared directly.
//!
//! # A whole sweep, end to end
//!
//! `examples/sweep.py` is the shortest honest answer to "what does using this
//! look like": prices in, a parameter grid, one parallel run, and an answer
//! attributed back to the parameter that produced it. It is also an
//! integration test, so it cannot rot.
//!
//! # Contents
//!
//! The execution kernel, both simulation loops, the record and trade layers,
//! the analysis layer, the array layer and the indicators, each pinned by
//! committed golden outputs. The factor DSL computes the full Alpha158 and
//! Alpha360 expression sets and feeds the signal simulator.

pub use alphax_analysis as analysis;
pub use alphax_core as core;
pub use alphax_exec as exec;
pub use alphax_expr as expr;
pub use alphax_generic as generic;
pub use alphax_indicators as indicators;
pub use alphax_model as model;
pub use alphax_records as records;
pub use alphax_signals as signals;
pub use alphax_sim as sim;

pub mod sweep;

/// The common imports.
pub mod prelude {
    pub use alphax_core::columns::{Columns, Label, Level};
    pub use alphax_core::flex::FlexArray;
    pub use alphax_core::{Error, Result};
    pub use alphax_exec::{ExecuteOrderState, Order, OrderResult, ProcessOrderState};
    pub use alphax_model::enums::*;
    pub use alphax_model::ids::{InstrumentId, Symbol, Venue};
    pub use alphax_model::types::{Price, Quantity};

    pub use crate::sweep::SignalSweep;
    pub use alphax_analysis::metrics::{DAILY, sharpe_ratio, total_return};
    pub use alphax_analysis::stats::{Select, Tag, Value};
    pub use alphax_analysis::{Portfolio, Reduce, Scoring, best_column, group_by};
    pub use alphax_core::matrix::Matrix;
    #[cfg(feature = "polars")]
    pub use alphax_expr::PolarsBackend;
    pub use alphax_expr::{
        ComputeBackend, FactorInputs, FactorSet, FactorValues, NativeBackend, alpha158, alpha360,
    };
    pub use alphax_indicators::{MaParams, combinations, ma};
    pub use alphax_records::{OrderRecords, exit_trades};
    pub use alphax_signals::{ExitWindow, FirstSignal, clean_enex, exit_after, partition_ranges};
    pub use alphax_sim::groups::{CallSeq, Groups};
    pub use alphax_sim::{FromSignals, simulate_from_signals, simulate_from_signals_parallel};
}
