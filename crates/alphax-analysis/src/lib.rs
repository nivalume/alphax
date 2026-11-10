//! Metrics, and aggregation across a sweep.
//!
//! Two things distinguish this layer from a conventional metrics library.
//!
//! **It is engine-agnostic.** It consumes records, not simulation state, so it
//! works identically on the output of any engine that emits the record
//! schemas.
//!
//! **Aggregation by column level is a primary feature, not a convenience.**
//! When a run produces thousands of columns, a per-column metric vector is
//! not an answer by itself — "mean Sharpe per factor", "best quantile per symbol"
//! is. Those reductions go through
//! [`alphax_core::columns::Columns::group_indices`], which is why the label
//! machinery lives in the foundation crate rather than here.
//!
//! # Lazy reconstruction
//!
//! A portfolio stores only its order records, its logs, the close prices and
//! the initial cash. Everything else — asset flow, cash flow, value,
//! returns, exposure, drawdowns — is derived on demand. For a sweep this is
//! the difference between holding one array and holding twenty, and most
//! columns in a sweep are never asked for more than a single scalar.
//!
//! # Contents
//!
//! The rebuild chain, return metrics in both scalar and rolling form,
//! benchmark-relative metrics, trade statistics, drawdown records, by-level
//! aggregation and the tag-filtered stats registry, for single-column
//! accounts and for groups sharing cash.

pub mod aggregate;
pub mod benchmark;
pub mod events;
pub mod metrics;
pub mod percentile;
pub mod portfolio;
pub mod robust;
pub mod rolling;
pub mod series;
pub mod stats;
pub mod trade_stats;

pub use aggregate::{Reduce, argmax, argmin, best_column, group_by};
pub use percentile::percentile;
pub use portfolio::{Cache, Portfolio, Scoring};
pub use series::returns_from_orders;
pub use stats::{Metric, Select, StatsInput, Tag, Value, stats};
pub use trade_stats::TradeStats;
