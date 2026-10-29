//! Factor evaluation: how well a factor ranks what comes next, without a
//! simulation.
//!
//! The rest of the engine judges a *strategy*: signals go into the
//! simulation, records come out, and metrics are rebuilt from the records.
//! This crate judges the *factor* that a strategy would be built on, directly
//! against the returns that follow it:
//!
//! - [`forward::forward_returns`]: the return from `lag` bars after each bar
//!   over the next `horizon` bars, the target a factor is scored against.
//! - [`ic::cs_ic`]: the information coefficient across symbols on each row
//!   (Pearson and Spearman), for many factors and horizons in one pass.
//! - [`ic::ts_ic`], [`ic::ts_rolling_ic`], [`ic::ts_block_ic`]: the same
//!   along time for each symbol: over the whole sample, over a rolling
//!   window, and within blocks such as calendar months.
//! - [`summary::summarize`]: mean, ICIR, the plain and the Newey–West
//!   t-statistic and its p-value, the hit rate, of any IC series.
//! - [`quantile::quantile_labels`]: each name's quantile on its own bar.
//!   This does not read the future. [`quantile::quantile_returns`] and
//!   [`quantile::long_short_returns`] score those groups against a forward
//!   return, and [`quantile::monotonicity`] asks whether that earning rises
//!   with the group.
//! - [`turnover::quantile_turnover`] and [`turnover::rank_autocorr`]: how
//!   fast the groups and the ranking change. Neither reads the future.
//! - [`regress::neutralize`]: residuals after group means and exposures.
//!   [`regress::fama_macbeth`] is the cross-sectional regression of a
//!   forward return on those factors, one bar at a time.
//! - [`correlation::mean_cs_corr`] and [`correlation::cross_corr`]: mean
//!   cross-sectional Spearman correlation, for redundancy between factors.
//! - [`placebo::cs_permutation`] and [`placebo::ts_circular_shift`]: rank-IC
//!   placebos. One shuffles each bar's forward returns, the other rotates
//!   the forward-return panel in time. The p-value is how often that null
//!   is at least as large as the factor that was actually observed.
//!
//! # Reading the future
//!
//! Forward returns, and anything scored against them, read bars after the
//! one they are stamped on, on purpose: that is what a factor is judged by.
//! Nothing computed from a forward return may be fed back into a simulation
//! as a signal. Quantile labels, turnover, rank autocorrelation,
//! neutralization and the correlation matrices use only the factor's own bar.
//! Fama–MacBeth regresses the panels it is given; when the left-hand side is
//! a forward return, that return is what reads the future. A placebo scores
//! the same forward return after the pairing has been destroyed.
//!
//! # Layout and threads
//!
//! Every panel is `(n_rows, n_cols)` = (bars, symbols), column-major, as
//! everywhere in phasor, and is borrowed rather than copied ([`Panel`]).
//! Work is split into contiguous ranges of rows (cross-sectional) or columns
//! (time series) that depend only on the shape and the thread count, and each
//! range writes its own part of the output. A mean over rows (the correlation
//! matrices) adds those per-row values in row order on one thread afterwards.
//! A run on one thread and on many is bit-identical. Placebo draws are pure
//! functions of `(seed, draw, row)`, so the thread count is not part of the key.

mod corr;
pub mod correlation;
pub mod forward;
pub mod ic;
pub mod panel;
pub mod placebo;
pub mod quantile;
mod rank;
pub mod regress;
pub mod summary;
pub mod turnover;

pub use correlation::{cross_corr, mean_cs_corr};
pub use forward::forward_returns;
pub use ic::{CsIc, Method, cs_ic, ts_block_ic, ts_ic, ts_rolling_ic};
pub use panel::Panel;
pub use placebo::{PlaceboResult, cs_permutation, ts_circular_shift};
pub use quantile::{long_short_returns, monotonicity, quantile_labels, quantile_returns};
pub use regress::{FmResult, fama_macbeth, neutralize};
pub use summary::{Summary, auto_lags, summarize};
pub use turnover::{quantile_turnover, rank_autocorr};
