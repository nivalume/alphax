//! Every return metric again, as a series instead of a number.
//!
//! A scalar Sharpe ratio over a five-year backtest says what happened on
//! average and hides when. The rolling form says whether the edge was there
//! throughout or came from one quarter — which is the difference between a
//! strategy and a coincidence, and the single most useful thing to look at
//! before believing a sweep's winner.
//!
//! Each function here is its scalar counterpart applied to every trailing
//! window, through [`rolling_apply_1d`]. Nothing is computed incrementally:
//! these recompute from the window, because most of them cannot be updated in
//! O(1) — a percentile, a drawdown and a regression all need the window in
//! hand. On a sweep this is the expensive part of the analysis layer, and it
//! is only paid when asked for.
//!
//! # `min_periods` counts measurable bars; the reduction sees all of them
//!
//! Passing `None` for `min_periods` requires a full window. Whatever it is,
//! it is compared against the count of non-`NaN` bars, and the reduction is
//! then handed the window *including* them. Each metric's own `NaN` handling
//! is what applies from there — which is why the rolling and scalar forms of
//! a metric always agree on a full window.
//!
//! # The benchmark forms take the same window of the benchmark
//!
//! [`rolling_beta`] and its neighbours slice the benchmark to the same bars,
//! so window `i` regresses the column's last `n` returns against the
//! benchmark's last `n`. Both series must be the same length.

use phasor_core::error::{Error, Result};
use phasor_generic::rolling::rolling_apply_1d;

use crate::benchmark;
use crate::metrics;

/// The rolling form of a metric that reads only its own returns.
macro_rules! rolling_solo {
    ($(#[$meta:meta])* $name:ident, $scalar:path $(, $arg:ident : $ty:ty)*) => {
        $(#[$meta])*
        ///
        /// # Errors
        ///
        /// Returns [`Error::Invariant`] if `min_periods` exceeds `window` or
        /// `window` is zero.
        pub fn $name(
            returns: &[f64],
            window: usize,
            min_periods: Option<usize>
            $(, $arg: $ty)*
        ) -> Result<Vec<f64>> {
            rolling_apply_1d(returns, window, min_periods, |_, w| $scalar(w $(, $arg)*))
        }
    };
}

rolling_solo!(
    /// Total return over each trailing window.
    rolling_total_return,
    metrics::total_return
);
rolling_solo!(
    /// Annualized return over each trailing window.
    rolling_annualized_return,
    metrics::annualized_return,
    ann_factor: f64
);
rolling_solo!(
    /// Annualized volatility over each trailing window.
    rolling_annualized_volatility,
    metrics::annualized_volatility,
    ann_factor: f64
);
rolling_solo!(
    /// Worst drawdown within each trailing window.
    ///
    /// Each window is measured from its own peak, so this is not the running
    /// drawdown of the whole series — a recovery that outlives the window
    /// resets it.
    rolling_max_drawdown,
    metrics::max_drawdown
);
rolling_solo!(
    /// Calmar ratio over each trailing window.
    rolling_calmar_ratio,
    metrics::calmar_ratio,
    ann_factor: f64
);
rolling_solo!(
    /// Omega ratio over each trailing window.
    rolling_omega_ratio,
    metrics::omega_ratio,
    ann_factor: f64,
    risk_free: f64,
    required_return: f64
);
rolling_solo!(
    /// Sharpe ratio over each trailing window.
    ///
    /// The one worth plotting: a strategy whose edge came from one stretch
    /// shows a spike here and a respectable-looking scalar Sharpe.
    rolling_sharpe_ratio,
    metrics::sharpe_ratio,
    ann_factor: f64,
    risk_free: f64
);
rolling_solo!(
    /// Downside risk over each trailing window.
    rolling_downside_risk,
    metrics::downside_risk,
    ann_factor: f64,
    required_return: f64
);
rolling_solo!(
    /// Sortino ratio over each trailing window.
    rolling_sortino_ratio,
    metrics::sortino_ratio,
    ann_factor: f64,
    required_return: f64
);
rolling_solo!(
    /// Tail ratio over each trailing window.
    rolling_tail_ratio,
    metrics::tail_ratio
);
rolling_solo!(
    /// Hit ratio over each trailing window.
    rolling_hit_ratio,
    metrics::hit_ratio
);
rolling_solo!(
    /// Return concentration over each trailing window.
    rolling_return_concentration,
    metrics::return_concentration,
    cutoff: f64
);
rolling_solo!(
    /// Value at risk over each trailing window.
    rolling_value_at_risk,
    metrics::value_at_risk,
    cutoff: f64
);
rolling_solo!(
    /// Conditional value at risk over each trailing window.
    ///
    /// Unlike its neighbours this one does not drop missing bars, so a window
    /// containing one answers `NaN` even when `min_periods` let it through.
    /// That asymmetry is deliberate; see [`metrics::cond_value_at_risk`].
    rolling_cond_value_at_risk,
    metrics::cond_value_at_risk,
    cutoff: f64
);

/// The rolling form of a metric that reads a benchmark too.
macro_rules! rolling_paired {
    ($(#[$meta:meta])* $name:ident, $scalar:path $(, $arg:ident : $ty:ty)*) => {
        $(#[$meta])*
        ///
        /// # Errors
        ///
        /// Returns [`Error::Invariant`] if the two series are different
        /// lengths, if `min_periods` exceeds `window`, or if `window` is zero.
        pub fn $name(
            returns: &[f64],
            benchmark: &[f64],
            window: usize,
            min_periods: Option<usize>
            $(, $arg: $ty)*
        ) -> Result<Vec<f64>> {
            same_length(returns, benchmark)?;
            rolling_apply_1d(returns, window, min_periods, |i, w| {
                // The benchmark's matching bars: the window is right-anchored
                // at `i`, and short at the start of the series.
                let start = i + 1 - w.len();
                $scalar(w, &benchmark[start..=i] $(, $arg)*)
            })
        }
    };
}

rolling_paired!(
    /// Information ratio over each trailing window.
    rolling_information_ratio,
    benchmark::information_ratio,
    ddof: usize
);
rolling_paired!(
    /// Beta over each trailing window.
    ///
    /// Where a scalar beta says how the column tracked the benchmark overall,
    /// this shows the tracking changing — which is what a regime shift looks
    /// like from the outside.
    rolling_beta,
    benchmark::beta
);
rolling_paired!(
    /// Alpha over each trailing window.
    rolling_alpha,
    benchmark::alpha,
    ann_factor: f64,
    risk_free: f64
);
rolling_paired!(
    /// Capture ratio over each trailing window.
    rolling_capture,
    benchmark::capture,
    ann_factor: f64
);
rolling_paired!(
    /// Upside capture over each trailing window.
    rolling_up_capture,
    benchmark::up_capture,
    ann_factor: f64
);
rolling_paired!(
    /// Downside capture over each trailing window.
    ///
    /// Below 1 is the good direction, as for [`benchmark::down_capture`].
    rolling_down_capture,
    benchmark::down_capture,
    ann_factor: f64
);

fn same_length(returns: &[f64], benchmark: &[f64]) -> Result<()> {
    if returns.len() == benchmark.len() {
        Ok(())
    } else {
        Err(Error::Invariant(format!(
            "a benchmark of {} bars against {} of returns",
            benchmark.len(),
            returns.len()
        )))
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "the full-window agreement with the scalar form must be exact"
)]
mod tests {
    use super::*;

    fn series() -> Vec<f64> {
        vec![0.01, -0.02, 0.03, -0.01, 0.02, 0.005, -0.015, 0.025]
    }

    fn bench() -> Vec<f64> {
        vec![0.005, -0.01, 0.02, -0.02, 0.01, 0.01, -0.005, 0.015]
    }

    #[test]
    fn the_last_full_window_agrees_with_the_scalar_form() {
        // The strongest property this layer has: a rolling metric over the
        // whole series must land on exactly the scalar one, bit for bit. Any
        // divergence means the window slice is off or the metric was
        // reimplemented rather than reused.
        let r = series();
        let n = r.len();
        let got = rolling_sharpe_ratio(&r, n, None, 365.0, 0.0).unwrap();
        assert_eq!(got[n - 1], metrics::sharpe_ratio(&r, 365.0, 0.0));
        assert!(
            got[..n - 1].iter().all(|v| v.is_nan()),
            "a full window is only available on the last bar"
        );

        let b = bench();
        let beta = rolling_beta(&r, &b, n, None).unwrap();
        assert_eq!(beta[n - 1], benchmark::beta(&r, &b));
    }

    #[test]
    fn a_short_window_is_right_anchored() {
        // With min_periods below the window, the early bars use a shorter
        // window ending on that bar — not a window centred on it, and not a
        // window borrowed from the future.
        let r = series();
        let got = rolling_total_return(&r, 4, Some(2)).unwrap();
        assert!(got[0].is_nan(), "one bar is below min_periods");
        assert_eq!(got[1], metrics::total_return(&r[..2]));
        assert_eq!(got[2], metrics::total_return(&r[..3]));
        // From bar 3 on, the window stops growing.
        assert_eq!(got[4], metrics::total_return(&r[1..5]));
    }

    #[test]
    fn the_benchmark_window_tracks_the_returns_window() {
        // If the benchmark slice drifted by even one bar, a column compared
        // against itself would stop showing a beta of 1 — which is why this
        // is asserted that way rather than against a hand-computed number.
        let r = series();
        let got = rolling_beta(&r, &r, 4, None).unwrap();
        for v in &got[3..] {
            assert!((v - 1.0).abs() < 1e-12, "got {v}");
        }
    }

    #[test]
    fn min_periods_counts_measurable_bars() {
        // Three bars in the window but only two with data, so a min_periods
        // of 3 is not met even though the window is full. It stays unmet at
        // bar 3, whose window still reaches back over the gap, and is met
        // only at bar 4 once the gap has rolled out.
        let r = [0.01, f64::NAN, 0.02, 0.03, 0.04];
        let strict = rolling_total_return(&r, 3, Some(3)).unwrap();
        assert!(strict[2].is_nan(), "two measurable bars is not three");
        assert!(strict[3].is_nan(), "bar 3's window still spans the gap");
        assert!(!strict[4].is_nan(), "bars 2..4 are all measurable");
        let loose = rolling_total_return(&r, 3, Some(2)).unwrap();
        assert!(!loose[2].is_nan());
    }

    #[test]
    fn the_reduction_still_sees_the_missing_bars() {
        // min_periods decides whether the metric is called; it does not clean
        // the window first. `cond_value_at_risk` is the one metric that does
        // not filter, so the gap still counts toward its length — which moves
        // the cutoff index and changes the answer, without the gap itself
        // ever entering the tail.
        let r = [0.01, f64::NAN, 0.02, 0.03];
        let with_gap = rolling_cond_value_at_risk(&r, 3, Some(2), 0.5).unwrap();
        // Bar 2's window is [0.01, NaN, 0.02]: three values, so the tail is
        // the two smallest and averages them.
        assert_eq!(with_gap[2], (0.01 + 0.02) / 2.0);
        // The same two measurable bars with no gap between them is a window
        // of two, whose tail is the single smallest.
        let without = [0.01, 0.02];
        assert_eq!(metrics::cond_value_at_risk(&without, 0.5), 0.01);

        // A wide enough cutoff does pull the gap in, and then there is no
        // answer at all.
        let wide = rolling_cond_value_at_risk(&r, 3, Some(2), 1.0).unwrap();
        assert!(wide[2].is_nan());
    }

    #[test]
    fn mismatched_lengths_are_rejected() {
        assert!(rolling_beta(&series(), &[0.1], 3, None).is_err());
        assert!(rolling_sharpe_ratio(&series(), 3, Some(4), 365.0, 0.0).is_err());
        assert!(rolling_sharpe_ratio(&series(), 0, None, 365.0, 0.0).is_err());
    }

    #[test]
    fn every_rolling_form_produces_one_value_per_bar() {
        let r = series();
        let b = bench();
        let n = r.len();
        let solo: Vec<Vec<f64>> = vec![
            rolling_total_return(&r, 3, None).unwrap(),
            rolling_annualized_return(&r, 3, None, 365.0).unwrap(),
            rolling_annualized_volatility(&r, 3, None, 365.0).unwrap(),
            rolling_max_drawdown(&r, 3, None).unwrap(),
            rolling_calmar_ratio(&r, 3, None, 365.0).unwrap(),
            rolling_omega_ratio(&r, 3, None, 365.0, 0.0, 0.0).unwrap(),
            rolling_sharpe_ratio(&r, 3, None, 365.0, 0.0).unwrap(),
            rolling_downside_risk(&r, 3, None, 365.0, 0.0).unwrap(),
            rolling_sortino_ratio(&r, 3, None, 365.0, 0.0).unwrap(),
            rolling_tail_ratio(&r, 3, None).unwrap(),
            rolling_value_at_risk(&r, 3, None, 0.05).unwrap(),
            rolling_cond_value_at_risk(&r, 3, None, 0.05).unwrap(),
        ];
        let paired: Vec<Vec<f64>> = vec![
            rolling_information_ratio(&r, &b, 3, None, 1).unwrap(),
            rolling_beta(&r, &b, 3, None).unwrap(),
            rolling_alpha(&r, &b, 3, None, 365.0, 0.0).unwrap(),
            rolling_capture(&r, &b, 3, None, 365.0).unwrap(),
            rolling_up_capture(&r, &b, 3, None, 365.0).unwrap(),
            rolling_down_capture(&r, &b, 3, None, 365.0).unwrap(),
        ];
        assert_eq!(solo.len() + paired.len(), 18, "one per scalar metric");
        for out in solo.iter().chain(&paired) {
            assert_eq!(out.len(), n);
            assert!(out[0].is_nan() && out[1].is_nan(), "warm-up is NaN");
        }
    }
}
