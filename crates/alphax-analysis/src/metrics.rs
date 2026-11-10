//! Return metrics, one column at a time.
//!
//! Each takes a column of per-bar returns and produces one number, so a whole
//! sweep reduces to a vector that [`crate::aggregate`] can rank and group.
//!
//! # `NaN` is a missing bar, not an error
//!
//! Every reduction here skips `NaN`s rather than propagating them. A bar with
//! no price is a bar with no information, not a poisoned column — propagating
//! would turn one gap into a column of nothing.
//!
//! # Degenerate cases are answers, not accidents
//!
//! A zero standard deviation gives an infinite Sharpe, a zero drawdown gives
//! a `NaN` Calmar, and fewer than two bars gives `NaN` throughout. These are
//! deliberate choices and they are load-bearing for ranking: a column
//! that never moved must not sort as the best one.
//!
//! # Only one of the three `len() < 2` guards does anything
//!
//! [`sharpe_ratio`] and [`annualized_volatility`] both check for a short
//! series and both would behave identically without the check: they route
//! through [`nanstd`], whose own `ddof` arithmetic already yields `NaN` when
//! there are fewer values than degrees of freedom. Deleting either guard is
//! undetectable, and mutation testing confirms it.
//!
//! [`sortino_ratio`]'s guard is not redundant. It divides by
//! [`downside_risk`], which has no `ddof` and happily returns a finite number
//! from a single bar — so without the guard a one-bar column would report a
//! real-looking Sortino ratio and outrank columns that actually traded.
//!
//! All three are kept, because a
//! reader cannot tell which one matters by looking. This note is how they
//! tell.

use crate::percentile::{percentile, sorted_nan_last};

/// Annualization factor for daily bars, the default.
pub const DAILY: f64 = 365.0;

/// Default cutoff for the value-at-risk metrics: the worst 5% of bars.
pub const VAR_CUTOFF: f64 = 0.05;

/// Mean of the non-`NaN` values, or `NaN` if there are none.
#[must_use]
pub fn nanmean(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for &v in values {
        if !v.is_nan() {
            sum += v;
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

/// Standard deviation of the non-`NaN` values, with `ddof` degrees of freedom.
///
/// Computed as the population variance over the non-`NaN`
/// values, rescaled by `count / (count - ddof)`. Rescaling rather than
/// dividing directly is not equivalent in floating point, and the difference
/// lands in the last bits of every Sharpe ratio.
#[must_use]
pub fn nanstd(values: &[f64], ddof: usize) -> f64 {
    let count = values.iter().filter(|v| !v.is_nan()).count();
    let rcount = count.saturating_sub(ddof);
    if rcount == 0 {
        return f64::NAN;
    }
    let mean = nanmean(values);
    let var = values
        .iter()
        .filter(|v| !v.is_nan())
        .map(|v| (v - mean) * (v - mean))
        .sum::<f64>()
        / count as f64;
    (var * count as f64 / rcount as f64).sqrt()
}

/// Mean and standard deviation of `values` shifted by `shift`, in one go.
///
/// Exists because the ratio metrics need both of a series they do not have:
/// `r - risk_free`. Materializing that series costs an allocation per call,
/// and computing the two separately walks it twice more than necessary —
/// [`nanstd`] derives the mean internally and then the caller derives it
/// again.
///
/// **Every arithmetic operation is the same, in the same order, as the
/// materialized version.** The subtraction is recomputed per pass rather than
/// stored, which is deterministic; the mean is computed once rather than
/// twice, which is the same value. An algebraic shortcut — `mean(r) - shift`
/// instead of `mean(r - shift)` — would be a rounding change, and this layer's
/// whole job is to reproduce its recorded values bit for bit.
fn nanmean_nanstd_shifted(values: &[f64], shift: f64, ddof: usize) -> (f64, f64) {
    let mut count = 0_usize;
    let mut sum = 0.0;
    for &v in values {
        let x = v - shift;
        if !x.is_nan() {
            sum += x;
            count += 1;
        }
    }
    if count == 0 {
        return (f64::NAN, f64::NAN);
    }
    let mean = sum / count as f64;
    if count.saturating_sub(ddof) == 0 {
        return (mean, f64::NAN);
    }
    let mut ss = 0.0;
    for &v in values {
        let x = v - shift;
        if !x.is_nan() {
            ss += (x - mean) * (x - mean);
        }
    }
    let var = ss / count as f64;
    let rcount = count - ddof;
    (mean, (var * count as f64 / rcount as f64).sqrt())
}

/// Cumulative growth factor: the product of `1 + r` over non-`NaN` bars.
#[must_use]
pub fn growth_factor(returns: &[f64]) -> f64 {
    returns
        .iter()
        .filter(|r| !r.is_nan())
        .fold(1.0, |acc, r| acc * (1.0 + r))
}

/// Total return over the whole period.
#[must_use]
pub fn total_return(returns: &[f64]) -> f64 {
    growth_factor(returns) - 1.0
}

/// Compound annual growth rate.
#[must_use]
pub fn annualized_return(returns: &[f64], ann_factor: f64) -> f64 {
    if returns.is_empty() {
        return f64::NAN;
    }
    growth_factor(returns).powf(ann_factor / returns.len() as f64) - 1.0
}

/// Annualized standard deviation of returns.
#[must_use]
pub fn annualized_volatility(returns: &[f64], ann_factor: f64) -> f64 {
    if returns.len() < 2 {
        return f64::NAN;
    }
    nanstd(returns, 1) * ann_factor.sqrt()
}

/// The equity curve as a drawdown from its running peak, per bar.
///
/// Started at 100 rather than 1. The ratio is
/// scale-free, so the choice does not change the answer — but it does change
/// the last bits, and the golden outputs pin them.
#[must_use]
pub fn drawdown(returns: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(returns.len());
    let mut cum = 1.0_f64;
    let mut peak = f64::NAN;
    for &r in returns {
        if !r.is_nan() {
            cum *= 1.0 + r;
        }
        let level = cum * 100.0;
        if peak.is_nan() || level > peak {
            peak = level;
        }
        out.push(level / peak - 1.0);
    }
    out
}

/// The worst drawdown reached, as a negative fraction.
///
/// # This is the one metric that propagates rather than skipping
///
/// This takes a plain minimum of the drawdown series, not a
/// `NaN`-skipping one, so a single unmeasurable bar in the *drawdown* makes
/// the whole answer `NaN`. It is reachable only through an infinite return,
/// which drives both the level and the peak to infinity and leaves their ratio
/// undefined — and an infinite return is what a position closing against a
/// price of zero produces.
///
/// A missing *return* is still skipped: those never reach the drawdown series,
/// because [`drawdown`] carries the previous level through a gap.
///
/// Skipping here instead answers `0.0`, which reads as "this column never drew
/// down" — the most flattering possible reading of a column whose equity went
/// to infinity. Mine did that until a fixture with an infinite return caught
/// it.
#[must_use]
pub fn max_drawdown(returns: &[f64]) -> f64 {
    let series = drawdown(returns);
    if series.is_empty() {
        // An empty series has nothing to propagate, so the missing answer
        // says so.
        return f64::NAN;
    }
    series.into_iter().fold(f64::INFINITY, |acc, v| {
        if acc.is_nan() || v.is_nan() {
            f64::NAN
        } else {
            acc.min(v)
        }
    })
}

/// Sharpe ratio: excess return per unit of volatility, annualized.
///
/// A column that never moved has zero volatility and returns `+inf`. That is
/// deliberate: it makes an unmoving column unmistakable in a ranking rather
/// than quietly average.
#[must_use]
pub fn sharpe_ratio(returns: &[f64], ann_factor: f64, risk_free: f64) -> f64 {
    if returns.len() < 2 {
        return f64::NAN;
    }
    let (mean, std) = nanmean_nanstd_shifted(returns, risk_free, 1);
    if std == 0.0 {
        return f64::INFINITY;
    }
    mean / std * ann_factor.sqrt()
}

/// Deviation of returns below `required_return`, annualized.
#[must_use]
pub fn downside_risk(returns: &[f64], ann_factor: f64, required_return: f64) -> f64 {
    // The clipped squares are summed as they are computed rather than
    // materialized twice over. Same operations, same order, no allocation.
    let mut count = 0_usize;
    let mut sum = 0.0;
    for r in returns {
        let adj = r - required_return;
        // A `NaN` fails this test and stays `NaN`, so it is dropped here for
        // the same reason `nanmean` would have dropped it.
        let below = if adj > 0.0 { 0.0 } else { adj };
        let sq = below * below;
        if !sq.is_nan() {
            sum += sq;
            count += 1;
        }
    }
    if count == 0 {
        return f64::NAN;
    }
    (sum / count as f64).sqrt() * ann_factor.sqrt()
}

/// Sortino ratio: return per unit of *downside* volatility.
#[must_use]
pub fn sortino_ratio(returns: &[f64], ann_factor: f64, required_return: f64) -> f64 {
    if returns.len() < 2 {
        return f64::NAN;
    }
    let risk = downside_risk(returns, ann_factor, required_return);
    if risk == 0.0 {
        return f64::INFINITY;
    }
    let (mean, _) = nanmean_nanstd_shifted(returns, required_return, 0);
    mean * ann_factor / risk
}

/// Calmar ratio: annualized return over the worst drawdown.
///
/// `NaN` when there was no drawdown, because the ratio is then unbounded and
/// says nothing about the strategy.
#[must_use]
pub fn calmar_ratio(returns: &[f64], ann_factor: f64) -> f64 {
    let mdd = max_drawdown(returns);
    if mdd == 0.0 {
        return f64::NAN;
    }
    annualized_return(returns, ann_factor) / mdd.abs()
}

/// Omega ratio: gains above a threshold over losses below it.
///
/// The threshold is `required_return` de-annualized to the bar's own
/// frequency, then shifted by `risk_free`. An `ann_factor` of exactly 1 says
/// the returns are already at the required return's frequency and skips the
/// conversion; one at or below -1 has no real root and gives `NaN`.
///
/// `+inf` when nothing fell below the threshold.
#[must_use]
pub fn omega_ratio(returns: &[f64], ann_factor: f64, risk_free: f64, required_return: f64) -> f64 {
    #[allow(
        clippy::float_cmp,
        reason = "the rule dispatches on exactly 1, not on a neighbourhood"
    )]
    let threshold = if ann_factor == 1.0 {
        required_return
    } else if ann_factor <= -1.0 {
        return f64::NAN;
    } else {
        (1.0 + required_return).powf(1.0 / ann_factor) - 1.0
    };
    let excess: Vec<f64> = returns.iter().map(|r| r - risk_free - threshold).collect();
    // A NaN fails both comparisons, so it contributes to neither side.
    //
    // The strictness of these two tests is *not* load-bearing: a bar exactly
    // at the threshold contributes zero to whichever sum it joins. Mutation
    // testing confirms it, and the tests are kept strict as the definition
    // writes them.
    let numer: f64 = excess.iter().filter(|v| **v > 0.0).sum();
    let denom: f64 = -excess.iter().filter(|v| **v < 0.0).sum::<f64>();
    if denom == 0.0 {
        return f64::INFINITY;
    }
    numer / denom
}

/// Tail ratio: the size of the right tail over the size of the left.
///
/// The 95th percentile over the 5th, both by magnitude. Above 1 means the
/// best days are bigger than the worst; below 1 means the reverse.
///
/// Missing bars are dropped first, and a column with none left is `NaN`.
#[must_use]
pub fn tail_ratio(returns: &[f64]) -> f64 {
    let live: Vec<f64> = returns.iter().copied().filter(|r| !r.is_nan()).collect();
    if live.is_empty() {
        return f64::NAN;
    }
    let right = percentile(&live, 95.0).abs();
    let left = percentile(&live, 5.0).abs();
    if left == 0.0 {
        return f64::INFINITY;
    }
    right / left
}

/// Value at risk: the return at the `cutoff` quantile.
///
/// With the default cutoff of 0.05, the return that only 5% of bars were worse
/// than. Negative for any column with a losing tail, which is the usual case
/// and not a sign error.
///
/// Missing bars are dropped first.
#[must_use]
pub fn value_at_risk(returns: &[f64], cutoff: f64) -> f64 {
    let live: Vec<f64> = returns.iter().copied().filter(|r| !r.is_nan()).collect();
    if live.is_empty() {
        return f64::NAN;
    }
    percentile(&live, 100.0 * cutoff)
}

/// Conditional value at risk: the mean of the worst `cutoff` of bars.
///
/// How bad it gets *when* it goes bad, where [`value_at_risk`] only says where
/// the boundary is.
///
/// # This one does not drop missing bars, by design
///
/// Its two neighbours filter `NaN` before doing anything; this one does not.
/// The effect is not that a gap poisons the answer — missing bars sort to the
/// end, so the tail usually never reaches them — it is that **a gap still
/// counts toward the length**, which moves the cutoff index. Over
/// `[NaN, 0.02, 0.03]` at a cutoff of 0.5 this averages the two smallest and
/// answers 0.025, where the same series with the gap removed takes only the
/// smallest and answers 0.02.
///
/// A wide enough cutoff does pull a missing bar into the tail, and then the
/// mean really is `NaN`.
///
/// The inconsistency with [`value_at_risk`] is kept rather than tidied: a
/// caller comparing the two on a series with a gap would otherwise see this
/// metric quietly change.
#[must_use]
pub fn cond_value_at_risk(returns: &[f64], cutoff: f64) -> f64 {
    if returns.is_empty() {
        return f64::NAN;
    }
    // The count of bars in the tail, floored: a cutoff of 0.05 over 10 bars
    // takes one bar, not half of one.
    let cutoff_index = ((returns.len() - 1) as f64 * cutoff) as usize;
    let sorted = sorted_nan_last(returns);
    let tail = &sorted[..=cutoff_index];
    // A plain mean, not `nanmean`: see the note above.
    tail.iter().sum::<f64>() / tail.len() as f64
}

/// Hit ratio: the share of bars with a gain among the bars that moved.
///
/// A bar with a return of exactly zero -- no position, or a flat bar -- is
/// counted on neither side: a strategy that is out of the market most of the
/// time is judged on the bars it took a risk on. `NaN` bars are skipped.
/// `NaN` when no bar moved.
#[must_use]
pub fn hit_ratio(returns: &[f64]) -> f64 {
    let mut gains = 0usize;
    let mut moved = 0usize;
    for &r in returns {
        if r > 0.0 {
            gains += 1;
            moved += 1;
        } else if r < 0.0 {
            moved += 1;
        }
    }
    if moved == 0 {
        return f64::NAN;
    }
    gains as f64 / moved as f64
}

/// Return concentration: the share of all the gains made on the best
/// `cutoff` of bars.
///
/// The bars with a return are sorted best first; the best
/// `floor((n - 1) * cutoff) + 1` of them -- the count [`cond_value_at_risk`]
/// takes from the other end -- contribute their gains, over the sum of every
/// positive return. Near `cutoff` when the gains are spread evenly, near 1
/// when a handful of bars made the whole result. `NaN` when nothing gained.
#[must_use]
pub fn return_concentration(returns: &[f64], cutoff: f64) -> f64 {
    let mut live: Vec<f64> = returns.iter().copied().filter(|r| !r.is_nan()).collect();
    let total: f64 = live.iter().filter(|r| **r > 0.0).sum();
    if live.is_empty() || total <= 0.0 {
        return f64::NAN;
    }
    live.sort_by(|a, b| b.total_cmp(a));
    let count = ((live.len() - 1) as f64 * cutoff.clamp(0.0, 1.0)) as usize + 1;
    let best: f64 = live[..count].iter().filter(|r| **r > 0.0).sum();
    best / total
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "degenerate results (inf, NaN, exact zero) are the contract here \
              and must be asserted exactly"
)]
mod tests {
    use super::*;

    fn close_to(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn total_return_compounds() {
        // 1.1 * 1.1 - 1.
        assert!(close_to(total_return(&[0.1, 0.1]), 0.21));
    }

    #[test]
    fn nan_bars_are_skipped_not_propagated() {
        assert!(close_to(total_return(&[0.1, f64::NAN, 0.1]), 0.21));
        assert!(close_to(nanmean(&[1.0, f64::NAN, 3.0]), 2.0));
    }

    #[test]
    fn an_all_nan_column_has_no_mean() {
        assert!(nanmean(&[f64::NAN, f64::NAN]).is_nan());
    }

    #[test]
    fn nanstd_rescales_the_population_variance() {
        // Sample standard deviation of [1,2,3,4] is sqrt(5/3).
        assert!(close_to(
            nanstd(&[1.0, 2.0, 3.0, 4.0], 1),
            (5.0f64 / 3.0).sqrt()
        ));
        // Population, ddof 0, is sqrt(5/4).
        assert!(close_to(
            nanstd(&[1.0, 2.0, 3.0, 4.0], 0),
            (5.0f64 / 4.0).sqrt()
        ));
    }

    #[test]
    fn one_bar_is_not_enough_for_a_ratio() {
        assert!(sharpe_ratio(&[0.1], 365.0, 0.0).is_nan());
        assert!(annualized_volatility(&[0.1], 365.0).is_nan());
        assert!(sortino_ratio(&[0.1], 365.0, 0.0).is_nan());
    }

    #[test]
    fn a_column_that_never_moved_has_an_infinite_sharpe() {
        assert_eq!(sharpe_ratio(&[0.0, 0.0, 0.0], 365.0, 0.0), f64::INFINITY);
    }

    #[test]
    fn drawdown_is_measured_from_the_running_peak() {
        // +10%, then -50%: the peak is at bar 0, the trough 50% below it.
        let d = drawdown(&[0.1, -0.5]);
        assert!(close_to(d[0], 0.0));
        assert!(close_to(d[1], -0.5));
        assert!(close_to(max_drawdown(&[0.1, -0.5]), -0.5));
    }

    #[test]
    fn an_infinite_return_leaves_no_drawdown_to_report() {
        // The level and the peak both go to infinity, and their ratio is
        // undefined. This must propagate: skipping would answer 0.0, which
        // reads as "never drew down" — the most flattering possible reading
        // of a column whose equity blew up. This port did exactly that until
        // a fixture with an infinite return caught it.
        assert!(max_drawdown(&[0.01, f64::INFINITY, 0.03]).is_nan());
        assert!(calmar_ratio(&[0.01, f64::INFINITY, 0.03], 365.0).is_nan());
        // A missing *return* is still skipped, because it never reaches the
        // drawdown series — the level carries through the gap.
        assert!(close_to(max_drawdown(&[0.1, f64::NAN, -0.5]), -0.5));
    }

    #[test]
    fn the_new_metrics_answer_on_the_degenerate_columns() {
        // A column that never moved has no losing side, no tail, and nothing
        // at risk. Each of these is a branch rather than a limit.
        let flat = [0.0, 0.0, 0.0];
        assert_eq!(omega_ratio(&flat, 365.0, 0.0, 0.0), f64::INFINITY);
        assert_eq!(tail_ratio(&flat), f64::INFINITY);
        assert_eq!(value_at_risk(&flat, VAR_CUTOFF), 0.0);
        assert_eq!(cond_value_at_risk(&flat, VAR_CUTOFF), 0.0);
        // And a column with no bars at all has no answer.
        assert!(tail_ratio(&[]).is_nan());
        assert!(value_at_risk(&[], VAR_CUTOFF).is_nan());
        assert!(cond_value_at_risk(&[], VAR_CUTOFF).is_nan());
        assert!(tail_ratio(&[f64::NAN, f64::NAN]).is_nan());
    }

    #[test]
    fn omega_converts_the_threshold_unless_the_factor_is_exactly_one() {
        // At ann_factor 1 the required return is used as given; otherwise it
        // is de-annualized first, so the same required return means a much
        // smaller per-bar threshold. The two answers must differ.
        let r = [0.01, -0.02, 0.03, -0.01];
        let direct = omega_ratio(&r, 1.0, 0.0, 0.1);
        let converted = omega_ratio(&r, 365.0, 0.0, 0.1);
        assert!(direct < converted, "{direct} should be below {converted}");
        // A factor at or below -1 has no real root.
        assert!(omega_ratio(&r, -1.0, 0.0, 0.1).is_nan());
        assert!(omega_ratio(&r, -2.0, 0.0, 0.1).is_nan());
    }

    #[test]
    fn value_at_risk_and_its_conditional_form_answer_different_questions() {
        // VaR is where the boundary is; CVaR is how bad it gets past it. Over
        // a series with one clear outlier, CVaR is the outlier and VaR is
        // nearly the next value up.
        let r = [-0.50, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09];
        let var = value_at_risk(&r, 0.05);
        let cvar = cond_value_at_risk(&r, 0.05);
        assert_eq!(cvar, -0.50, "the worst bar, on its own");
        assert!(var > cvar, "{var} should sit above {cvar}");
    }

    #[test]
    fn a_curve_that_only_rises_has_no_drawdown() {
        assert!(close_to(max_drawdown(&[0.1, 0.1, 0.1]), 0.0));
        // And therefore no Calmar ratio.
        assert!(calmar_ratio(&[0.1, 0.1, 0.1], 365.0).is_nan());
    }

    #[test]
    fn downside_risk_ignores_the_upside() {
        // Only the -0.1 contributes; the +0.5 is clipped to zero.
        let a = downside_risk(&[-0.1, 0.5], 1.0, 0.0);
        let b = downside_risk(&[-0.1, 0.0], 1.0, 0.0);
        assert!(close_to(a, b));
    }

    #[test]
    fn sortino_is_infinite_when_nothing_went_down() {
        assert_eq!(sortino_ratio(&[0.1, 0.2], 365.0, 0.0), f64::INFINITY);
    }

    #[test]
    fn hit_ratio_counts_only_the_bars_that_moved() {
        assert!(close_to(
            hit_ratio(&[0.1, -0.1, 0.0, f64::NAN, 0.2]),
            2.0 / 3.0
        ));
        assert!(hit_ratio(&[0.0, f64::NAN]).is_nan());
        assert_eq!(hit_ratio(&[0.01, 0.02]), 1.0);
    }

    #[test]
    fn return_concentration_is_the_best_bars_share_of_the_gains() {
        // 21 bars: the best one (cutoff 5% of 20 is 1, plus 1 = 2 bars) ...
        let mut r = vec![0.01; 19];
        r.push(0.5);
        r.push(-0.2);
        // ... the two best are 0.5 and 0.01, of 0.5 + 19 * 0.01 of gains.
        assert!(close_to(return_concentration(&r, 0.05), 0.51 / 0.69));
        // Gains spread evenly: the share is the share of bars.
        let even = vec![0.01; 101];
        assert!(close_to(return_concentration(&even, 0.05), 6.0 / 101.0));
        assert!(return_concentration(&[-0.1, 0.0], 0.05).is_nan());
        assert!(return_concentration(&[], 0.05).is_nan());
    }
}
