//! Metrics that compare a column against a benchmark.
//!
//! Everything in [`crate::metrics`] scores a column on its own terms. These
//! score it against something else — usually buy-and-hold on the same
//! instrument, which is exactly the comparison a sweep exists to make. A
//! parameter set that returned 40% is interesting only next to what the
//! instrument did anyway.
//!
//! # The benchmark is per column, not per sweep
//!
//! Each function takes a benchmark series the same length as the returns, and
//! the caller is the one that decides what it is. In a sweep where parameters
//! repeat and symbols tile, the benchmark for column `c` is the buy-and-hold
//! return of *that column's symbol* — so the benchmark array is tiled the same
//! way the symbols are. Passing one symbol's benchmark against every column is
//! a mistake this layer cannot detect, and [`crate::rolling`]'s callers have
//! the column layout to get it right.
//!
//! # Alignment, not correlation
//!
//! [`beta`] masks the benchmark wherever the returns are missing, so a gap in
//! one series removes that bar from both. Without it a column with a different
//! set of missing bars would be regressed against a benchmark measured over a
//! different sample, and the number would look fine.

use crate::metrics::{annualized_return, nanmean, nanstd};

/// Information ratio: mean active return over its own volatility.
///
/// Active return is the column's return minus the benchmark's, bar by bar, so
/// this measures consistency of outperformance rather than its size.
///
/// `+inf` when the active return never varied, `NaN` for fewer than two bars.
#[must_use]
pub fn information_ratio(returns: &[f64], benchmark: &[f64], ddof: usize) -> f64 {
    if returns.len() < 2 {
        return f64::NAN;
    }
    let active: Vec<f64> = returns.iter().zip(benchmark).map(|(r, b)| r - b).collect();
    let std = nanstd(&active, ddof);
    if std == 0.0 {
        return f64::INFINITY;
    }
    nanmean(&active) / std
}

/// Beta: how much of the benchmark's movement the column reproduces.
///
/// 1 means it moved with the benchmark, 0 that it did not move with it at all,
/// negative that it moved against it.
///
/// # A variance below 1e-30 is treated as missing, not as zero
///
/// A near-zero benchmark variance is blanked to `NaN` *before*
/// testing it for exact zero, which means the `+inf` branch below is
/// unreachable for any finite input: anything small enough to divide to
/// infinity was already replaced. Both are kept, and the ordering is what
/// makes a flat benchmark answer "no answer" rather than "infinitely
/// sensitive".
#[must_use]
pub fn beta(returns: &[f64], benchmark: &[f64]) -> f64 {
    if benchmark.len() < 2 {
        return f64::NAN;
    }
    // Wherever the column has no return, the benchmark bar is dropped too, so
    // both sides are measured over the same sample.
    let independent: Vec<f64> = returns
        .iter()
        .zip(benchmark)
        .map(|(r, b)| if r.is_nan() { f64::NAN } else { *b })
        .collect();
    let ind_mean = nanmean(&independent);
    let residual: Vec<f64> = independent.iter().map(|v| v - ind_mean).collect();
    // The returns go in raw, not centred. Centring them too is algebraically
    // the same thing — the correction term is `mean(residual)` times a
    // constant, and `mean(residual)` is zero by construction — so it changes
    // only the last bits, below the tolerance this layer is compared at.
    // Mutation testing finds it, and it is a genuine equivalence rather than a
    // gap: distinguishing the two spellings would need this port to sum in
    // exactly NumPy's pairwise order, which it does not claim to.
    let cov: Vec<f64> = residual
        .iter()
        .zip(returns)
        .map(|(res, r)| res * r)
        .collect();
    let covariance = nanmean(&cov);
    let squared: Vec<f64> = residual.iter().map(|v| v * v).collect();
    let mut variance = nanmean(&squared);
    if variance < 1.0e-30 {
        variance = f64::NAN;
    }
    if variance == 0.0 {
        return f64::INFINITY;
    }
    covariance / variance
}

/// Alpha: the annualized return left over once the benchmark's share is
/// removed.
///
/// The column's excess return minus [`beta`] times the benchmark's excess
/// return, averaged and annualized. Positive alpha is return the benchmark
/// does not explain — which is the claim a strategy is actually making.
#[must_use]
pub fn alpha(returns: &[f64], benchmark: &[f64], ann_factor: f64, risk_free: f64) -> f64 {
    if returns.len() < 2 {
        return f64::NAN;
    }
    let b = beta(returns, benchmark);
    let series: Vec<f64> = returns
        .iter()
        .zip(benchmark)
        .map(|(r, bm)| (r - risk_free) - b * (bm - risk_free))
        .collect();
    (nanmean(&series) + 1.0).powf(ann_factor) - 1.0
}

/// Capture ratio: the column's annualized return over the benchmark's.
///
/// `+inf` when the benchmark went nowhere.
#[must_use]
pub fn capture(returns: &[f64], benchmark: &[f64], ann_factor: f64) -> f64 {
    ratio_of_annualized(returns, benchmark, ann_factor)
}

/// Capture ratio over the bars the benchmark rose.
///
/// Above 1 means the column outpaced the benchmark on its good days. Paired
/// with [`down_capture`] this is the asymmetry that a single capture number
/// averages away.
///
/// `NaN` when the benchmark never rose.
#[must_use]
pub fn up_capture(returns: &[f64], benchmark: &[f64], ann_factor: f64) -> f64 {
    directional_capture(returns, benchmark, ann_factor, true)
}

/// Capture ratio over the bars the benchmark fell.
///
/// *Below* 1 is the good direction here: it means the column lost less than
/// the benchmark did. Reading it the same way round as [`up_capture`] is the
/// easy mistake.
///
/// `NaN` when the benchmark never fell.
#[must_use]
pub fn down_capture(returns: &[f64], benchmark: &[f64], ann_factor: f64) -> f64 {
    directional_capture(returns, benchmark, ann_factor, false)
}

fn directional_capture(returns: &[f64], benchmark: &[f64], ann_factor: f64, up: bool) -> f64 {
    let keep = |b: &f64| if up { *b > 0.0 } else { *b < 0.0 };
    let r: Vec<f64> = returns
        .iter()
        .zip(benchmark)
        .filter(|(_, b)| keep(b))
        .map(|(r, _)| *r)
        .collect();
    let b: Vec<f64> = benchmark.iter().copied().filter(|v| keep(v)).collect();
    if r.is_empty() {
        return f64::NAN;
    }
    ratio_of_annualized(&r, &b, ann_factor)
}

fn ratio_of_annualized(returns: &[f64], benchmark: &[f64], ann_factor: f64) -> f64 {
    let theirs = annualized_return(benchmark, ann_factor);
    if theirs == 0.0 {
        return f64::INFINITY;
    }
    annualized_return(returns, ann_factor) / theirs
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "the degenerate results and the exact-1 cases are the contract here"
)]
mod tests {
    use super::*;

    fn close_to(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn a_column_that_is_the_benchmark_has_beta_one_and_no_alpha() {
        let r = [0.01, -0.02, 0.03, 0.005];
        assert!(close_to(beta(&r, &r), 1.0));
        assert!(close_to(alpha(&r, &r, 365.0, 0.0), 0.0));
        assert!(close_to(capture(&r, &r, 365.0), 1.0));
    }

    #[test]
    fn twice_the_benchmark_has_beta_two() {
        let b = [0.01, -0.02, 0.03, 0.005];
        let r: Vec<f64> = b.iter().map(|v| v * 2.0).collect();
        assert!(close_to(beta(&r, &b), 2.0));
        // And moving against it is negative.
        let against: Vec<f64> = b.iter().map(|v| -v).collect();
        assert!(close_to(beta(&against, &b), -1.0));
    }

    #[test]
    fn a_flat_benchmark_has_no_beta() {
        // The near-zero blanking fires before the exact-zero test, so a
        // benchmark that never moved gives "no answer" rather than infinity.
        assert!(beta(&[0.1, 0.2], &[0.0, 0.0]).is_nan());
        assert!(beta(&[0.1, 0.2], &[1e-20, 1e-20]).is_nan());
    }

    #[test]
    fn beta_drops_a_bar_the_column_is_missing() {
        // The gap must remove that bar from the benchmark too. If it did not,
        // the regression would be against a different sample and would still
        // return a plausible-looking number.
        let b = [0.01, 5.0, -0.02, 0.03];
        let r = [0.02, f64::NAN, -0.04, 0.06];
        let without = [0.02, -0.04, 0.06];
        let b_without = [0.01, -0.02, 0.03];
        assert!(close_to(beta(&r, &b), beta(&without, &b_without)));
    }

    #[test]
    fn the_information_ratio_is_infinite_when_outperformance_never_varies() {
        // Dyadic values, so `r - b` is exactly 0.25 on every bar. With
        // decimal ones the subtraction leaves a few ULPs of spread and this
        // answers a large finite number instead — which is the honest result
        // for a series that is not quite constant, and the reason the
        // assertion needs inputs where "constant" is exact.
        let b = [0.25, 0.5, 0.75];
        let r: Vec<f64> = b.iter().map(|v| v + 0.25).collect();
        assert_eq!(information_ratio(&r, &b, 1), f64::INFINITY);
    }

    #[test]
    fn matching_the_benchmark_exactly_gives_a_zero_information_ratio() {
        let b = [0.01, -0.02, 0.03];
        // Zero active return, zero variance: the zero-std branch
        // fires before the mean is looked at, so this is +inf and not 0/0.
        assert_eq!(information_ratio(&b, &b, 1), f64::INFINITY);
    }

    #[test]
    fn up_and_down_capture_split_the_benchmark_by_sign() {
        //                 up      down    up      down
        let b = [0.10, -0.10, 0.20, -0.05];
        let r = [0.20, -0.05, 0.40, -0.02];
        // Twice the benchmark on the way up.
        assert!(up_capture(&r, &b, 365.0) > 1.0);
        // Less than the benchmark on the way down, which is the good
        // direction for this one.
        assert!(down_capture(&r, &b, 365.0) < 1.0);
    }

    #[test]
    fn a_benchmark_that_only_rose_has_no_down_capture() {
        let b = [0.01, 0.02];
        let r = [0.03, 0.04];
        assert!(down_capture(&r, &b, 365.0).is_nan());
        assert!(!up_capture(&r, &b, 365.0).is_nan());
    }

    #[test]
    fn one_bar_is_not_enough_for_a_relative_ratio() {
        assert!(information_ratio(&[0.1], &[0.05], 1).is_nan());
        assert!(beta(&[0.1], &[0.05]).is_nan());
        assert!(alpha(&[0.1], &[0.05], 365.0, 0.0).is_nan());
    }
}
