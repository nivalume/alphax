//! Rolling windows, shifts and fills: the array layer under everything else.
//!
//! Indicators are rolling windows with names. Factor expressions are rolling
//! windows with a parser. Several analysis metrics are rolling windows with a
//! ratio at the end. Putting them in one place means the semantics are
//! decided once — and the semantics are where this layer earns its keep,
//! because the arithmetic is trivial and the edge cases are not.
//!
//! Three of those decisions run through everything here:
//!
//! - **`min_periods`**: how many observations a window needs before it
//!   reports a number rather than `NaN`. Get it wrong and a 20-bar average
//!   silently starts on bar 0, computed from one observation.
//! - **`NaN` shrinks a window, it does not poison it.** A missing bar is
//!   missing information, not a failed computation.
//! - **Lookahead is labelled, not prevented.** [`elementwise::bshift`] and
//!   [`elementwise::bfill`] read the future by definition; both say so, and
//!   neither belongs between a price and a signal.
//!
//! # Verification
//!
//! Every function here is pinned by committed golden outputs in
//! `tests/golden_generic.rs`, over series built to hit the awkward cases:
//! leading and trailing gaps, all-`NaN` windows, constant series where the
//! variance cancels, and windows longer than the series.

pub mod elementwise;
pub mod rolling;
pub mod synthetic;

pub use elementwise::{
    bfill, bshift, crossed_above, crossed_above_1d, crossed_below_1d, diff, ffill, fillna, fshift,
    pct_change,
};
pub use rolling::{
    ewm_mean, ewm_mean_1d, ewm_std, ewm_std_1d, expanding_max_1d, expanding_mean_1d,
    expanding_min_1d, expanding_std_1d, rolling_max, rolling_max_1d, rolling_mean, rolling_mean_1d,
    rolling_min, rolling_min_1d, rolling_std, rolling_std_1d,
};
pub use synthetic::{gbm, ohlc_from_close, random_walk};
