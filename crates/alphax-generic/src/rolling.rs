//! Rolling and expanding windows.
//!
//! # `min_periods` decides what a partial window is worth
//!
//! Every function here takes `minp`: how many non-`NaN` observations a window
//! needs before it produces a number instead of `NaN`. It is not a detail. A
//! 20-bar moving average with `minp = 1` starts producing values on bar 0 —
//! values computed from one observation and presented in the same array as
//! the ones computed from twenty. A strategy that enters on those is trading
//! a different signal than it thinks. The default, `minp = window`, is the
//! conservative one.
//!
//! # Why the mean and the standard deviation accumulate
//!
//! [`rolling_mean`] and [`rolling_std`] keep running sums and subtract the
//! value that fell out of the window, rather than re-summing it. That is
//! `O(1)` per bar instead of `O(window)` — and it is *not* the same answer in
//! floating point. Re-summing would give slightly different last bits, which
//! is enough to move a fill across a threshold. The accumulating form is the
//! one used here.
//!
//! The minimum and maximum keep a monotonic queue of candidate bars instead:
//! there is no subtraction that removes a value from a maximum, but a bar that
//! is dominated by a later one can never be the maximum again, so each bar is
//! queued and dropped once. The result is a comparison, not an accumulation,
//! so it is bit-identical to re-scanning the window.

use std::collections::VecDeque;

use alphax_core::error::{Error, Result};
use alphax_core::matrix::Matrix;

/// Resolves the `minp` default and rejects an impossible one.
fn resolve_minp(window: usize, minp: Option<usize>) -> Result<usize> {
    let minp = minp.unwrap_or(window);
    if minp > window {
        return Err(Error::Invariant(format!(
            "min_periods ({minp}) cannot exceed the window ({window})"
        )));
    }
    if window == 0 {
        return Err(Error::Invariant("the window must be at least 1".into()));
    }
    Ok(minp)
}

/// Applies a reduction to every trailing window of one series.
///
/// The escape hatch for anything that cannot be computed incrementally — a
/// percentile, a drawdown, a regression against a second series. Where
/// [`rolling_mean_1d`] and its neighbours carry running sums and cost O(1) per
/// bar, this recomputes from the whole window, so it costs whatever `f` costs
/// times the number of bars. That is the price of generality and it is why the
/// specific kernels exist alongside it.
///
/// # What `minp` counts, and what `f` receives, are not the same thing
///
/// `minp` is compared against the count of **non-`NaN`** values in the window,
/// but `f` is then handed the **whole window including them**. So a reduction
/// here has to do its own `NaN` handling; `minp` only decides whether it is
/// called at all. Both halves are deliberate, and getting the second half
/// wrong — filtering before the call — would silently change every metric that
/// treats a gap as a zero.
///
/// `f` receives the window and the index of its last bar.
///
/// # Errors
///
/// Returns [`Error::Invariant`] if `minp` exceeds `window` or `window` is
/// zero.
pub fn rolling_apply_1d(
    a: &[f64],
    window: usize,
    minp: Option<usize>,
    mut f: impl FnMut(usize, &[f64]) -> f64,
) -> Result<Vec<f64>> {
    let minp = resolve_minp(window, minp)?;
    let n = a.len();
    let mut out = vec![f64::NAN; n];
    let mut nancnt_arr = vec![0_usize; n];
    let mut nancnt = 0_usize;

    for i in 0..n {
        if a[i].is_nan() {
            nancnt += 1;
        }
        nancnt_arr[i] = nancnt;
        let valid = if i < window {
            i + 1 - nancnt
        } else {
            window - (nancnt - nancnt_arr[i - window])
        };
        if valid >= minp {
            let start = (i + 1).saturating_sub(window);
            out[i] = f(i, &a[start..=i]);
        }
    }
    Ok(out)
}

/// Rolling mean of one series.
///
/// # Errors
///
/// Returns [`Error::Invariant`] if `minp` exceeds `window` or `window` is
/// zero.
pub fn rolling_mean_1d(a: &[f64], window: usize, minp: Option<usize>) -> Result<Vec<f64>> {
    let minp = resolve_minp(window, minp)?;
    let n = a.len();
    let mut out = vec![f64::NAN; n];
    // Prefix sums and prefix `NaN` counts, so a window is one subtraction.
    let mut cumsum_arr = vec![0.0; n];
    let mut nancnt_arr = vec![0_usize; n];
    let mut cumsum = 0.0;
    let mut nancnt = 0_usize;

    for i in 0..n {
        if a[i].is_nan() {
            nancnt += 1;
        } else {
            cumsum += a[i];
        }
        nancnt_arr[i] = nancnt;
        cumsum_arr[i] = cumsum;

        let (window_len, window_cumsum) = if i < window {
            (i + 1 - nancnt, cumsum)
        } else {
            (
                window - (nancnt - nancnt_arr[i - window]),
                cumsum - cumsum_arr[i - window],
            )
        };
        out[i] = if window_len < minp {
            f64::NAN
        } else {
            window_cumsum / window_len as f64
        };
    }
    Ok(out)
}

/// Rolling standard deviation of one series.
///
/// `ddof` is the divisor's correction: 0 for the population figure, 1 for the
/// sample one.
///
/// # Errors
///
/// As [`rolling_mean_1d`].
pub fn rolling_std_1d(
    a: &[f64],
    window: usize,
    minp: Option<usize>,
    ddof: usize,
) -> Result<Vec<f64>> {
    let minp = resolve_minp(window, minp)?;
    let n = a.len();
    let mut out = vec![f64::NAN; n];
    let mut cumsum_arr = vec![0.0; n];
    let mut cumsum_sq_arr = vec![0.0; n];
    let mut nancnt_arr = vec![0_usize; n];
    let (mut cumsum, mut cumsum_sq, mut nancnt) = (0.0, 0.0, 0_usize);

    for i in 0..n {
        if a[i].is_nan() {
            nancnt += 1;
        } else {
            cumsum += a[i];
            cumsum_sq += a[i] * a[i];
        }
        nancnt_arr[i] = nancnt;
        cumsum_arr[i] = cumsum;
        cumsum_sq_arr[i] = cumsum_sq;

        let (window_len, window_cumsum, window_cumsum_sq) = if i < window {
            (i + 1 - nancnt, cumsum, cumsum_sq)
        } else {
            (
                window - (nancnt - nancnt_arr[i - window]),
                cumsum - cumsum_arr[i - window],
                cumsum_sq - cumsum_sq_arr[i - window],
            )
        };

        out[i] = if window_len < minp || window_len == ddof {
            f64::NAN
        } else {
            let len = window_len as f64;
            let mean = window_cumsum / len;
            // Expanded rather than computed from deviations, because the
            // prefix sums are what is available. The absolute value guards
            // the cancellation this form is prone to: on a constant series
            // the three terms cancel to a tiny negative number, and the
            // square root of that would be `NaN` instead of zero.
            let var = (window_cumsum_sq - 2.0 * window_cumsum * mean + len * (mean * mean)).abs();
            (var / (len - ddof as f64)).sqrt()
        };
    }
    Ok(out)
}

/// Rolling minimum of one series.
///
/// # Errors
///
/// As [`rolling_mean_1d`].
pub fn rolling_min_1d(a: &[f64], window: usize, minp: Option<usize>) -> Result<Vec<f64>> {
    rolling_extremum_1d(a, window, minp, true)
}

/// Rolling maximum of one series.
///
/// # Errors
///
/// As [`rolling_mean_1d`].
pub fn rolling_max_1d(a: &[f64], window: usize, minp: Option<usize>) -> Result<Vec<f64>> {
    rolling_extremum_1d(a, window, minp, false)
}

fn rolling_extremum_1d(
    a: &[f64],
    window: usize,
    minp: Option<usize>,
    want_min: bool,
) -> Result<Vec<f64>> {
    let minp = resolve_minp(window, minp)?;
    let mut out = vec![f64::NAN; a.len()];
    // Indices of the window's non-`NaN` values whose value is still a
    // candidate: each is strictly better than every later one, so the front is
    // the extremum and every index is pushed and popped once.
    let mut candidates = VecDeque::new();
    let mut cnt = 0_usize;
    for i in 0..a.len() {
        if i >= window && !a[i - window].is_nan() {
            cnt -= 1;
            if candidates.front() == Some(&(i - window)) {
                candidates.pop_front();
            }
        }
        let v = a[i];
        if !v.is_nan() {
            cnt += 1;
            while candidates
                .back()
                .is_some_and(|&j: &usize| if want_min { a[j] >= v } else { a[j] <= v })
            {
                candidates.pop_back();
            }
            candidates.push_back(i);
        }
        // An all-`NaN` window carries the current bar's `NaN`, as the
        // seed does, rather than an infinity.
        let best = candidates.front().map_or(v, |&j| a[j]);
        out[i] = if cnt < minp { f64::NAN } else { best };
    }
    Ok(out)
}

/// Expanding mean: every bar's window starts at the first.
///
/// # Errors
///
/// Returns [`Error::Invariant`] if `minp` exceeds the series length. That is
/// a deliberate choice and it is not obviously the right one — asking
/// for two observations from a one-element series could reasonably give
/// `NaN`. Clamping instead would be a silent divergence, so the error stands.
pub fn expanding_mean_1d(a: &[f64], minp: usize) -> Result<Vec<f64>> {
    rolling_mean_1d(a, a.len().max(1), Some(minp))
}

/// Expanding standard deviation.
///
/// # Errors
///
/// As [`expanding_mean_1d`].
pub fn expanding_std_1d(a: &[f64], minp: usize, ddof: usize) -> Result<Vec<f64>> {
    rolling_std_1d(a, a.len().max(1), Some(minp), ddof)
}

/// Expanding minimum.
#[must_use]
pub fn expanding_min_1d(a: &[f64], minp: usize) -> Vec<f64> {
    expanding_extremum_1d(a, minp, true)
}

/// Expanding maximum.
#[must_use]
pub fn expanding_max_1d(a: &[f64], minp: usize) -> Vec<f64> {
    expanding_extremum_1d(a, minp, false)
}

fn expanding_extremum_1d(a: &[f64], minp: usize, want_min: bool) -> Vec<f64> {
    let mut out = vec![f64::NAN; a.len()];
    if a.is_empty() {
        return out;
    }
    let mut best = a[0];
    let mut cnt = 0_usize;
    for i in 0..a.len() {
        if best.is_nan() || (if want_min { a[i] < best } else { a[i] > best }) {
            best = a[i];
        }
        if !a[i].is_nan() {
            cnt += 1;
        }
        out[i] = if cnt < minp { f64::NAN } else { best };
    }
    out
}

/// Exponentially weighted mean over a span.
///
/// `span` maps to a decay of `alpha = 2 / (span + 1)`. With `adjust`, early
/// bars are rescaled so the weights sum to one; without it, the series is a
/// plain recursion seeded at the first observation. The default is
/// `adjust = false`, and the two differ for the whole length of the series,
/// not just its start.
///
/// # Errors
///
/// Returns [`Error::Invariant`] if `minp` exceeds `span`.
#[allow(
    clippy::float_cmp,
    reason = "the guard is *about* exact equality: it skips recomputing a \
              value that has not moved, which is what keeps a constant series \
              constant. A tolerance here would skip real updates"
)]
pub fn ewm_mean_1d(a: &[f64], span: usize, minp: usize, adjust: bool) -> Result<Vec<f64>> {
    if minp > span {
        return Err(Error::Invariant(format!(
            "min_periods ({minp}) cannot exceed the span ({span})"
        )));
    }
    let n = a.len();
    let mut out = vec![f64::NAN; n];
    if n == 0 {
        return Ok(out);
    }

    let com = (span as f64 - 1.0) / 2.0;
    let alpha = 1.0 / (1.0 + com);
    let old_wt_factor = 1.0 - alpha;
    let new_wt = if adjust { 1.0 } else { alpha };

    let mut weighted_avg = a[0];
    let mut nobs = usize::from(!weighted_avg.is_nan());
    out[0] = if nobs >= minp { weighted_avg } else { f64::NAN };
    let mut old_wt = 1.0;

    for i in 1..n {
        let cur = a[i];
        let is_observation = !cur.is_nan();
        nobs += usize::from(is_observation);
        if !weighted_avg.is_nan() {
            old_wt *= old_wt_factor;
            if is_observation {
                // Skipped when the value has not moved. Not an optimization:
                // recomputing the same number through the weighted form
                // introduces drift, and a constant series would stop being
                // constant.
                if weighted_avg != cur {
                    weighted_avg = (old_wt * weighted_avg + new_wt * cur) / (old_wt + new_wt);
                }
                if adjust {
                    old_wt += new_wt;
                } else {
                    old_wt = 1.0;
                }
            }
        } else if is_observation {
            weighted_avg = cur;
        }
        out[i] = if nobs >= minp { weighted_avg } else { f64::NAN };
    }
    Ok(out)
}

/// Exponentially weighted standard deviation over a span.
///
/// Always bias-corrected, and the first bar is always `NaN` — a single
/// observation has no deviation to report. There is no `ddof`
/// argument: this signature leaves it out rather than carrying a parameter
/// that does nothing.
///
/// # Errors
///
/// As [`ewm_mean_1d`].
#[allow(
    clippy::float_cmp,
    reason = "the guard is *about* exact equality: it skips recomputing a \
              value that has not moved, which is what keeps a constant series \
              constant. A tolerance here would skip real updates"
)]
pub fn ewm_std_1d(a: &[f64], span: usize, minp: usize, adjust: bool) -> Result<Vec<f64>> {
    if minp > span {
        return Err(Error::Invariant(format!(
            "min_periods ({minp}) cannot exceed the span ({span})"
        )));
    }
    let n = a.len();
    let mut out = vec![f64::NAN; n];
    if n == 0 {
        return Ok(out);
    }

    let com = (span as f64 - 1.0) / 2.0;
    let alpha = 1.0 / (1.0 + com);
    let old_wt_factor = 1.0 - alpha;
    let new_wt = if adjust { 1.0 } else { alpha };

    let mut mean = a[0];
    let mut nobs = usize::from(!mean.is_nan());
    if mean.is_nan() {
        mean = f64::NAN;
    }
    let mut cov = 0.0;
    let mut sum_wt = 1.0;
    let mut sum_wt2 = 1.0;
    let mut old_wt = 1.0;

    for i in 1..n {
        let cur = a[i];
        let is_observation = !cur.is_nan();
        nobs += usize::from(is_observation);
        if !mean.is_nan() {
            sum_wt *= old_wt_factor;
            sum_wt2 *= old_wt_factor * old_wt_factor;
            old_wt *= old_wt_factor;
            if is_observation {
                let old_mean = mean;
                if mean != cur {
                    mean = (old_wt * old_mean + new_wt * cur) / (old_wt + new_wt);
                }
                cov = (old_wt * (cov + (old_mean - mean) * (old_mean - mean))
                    + new_wt * ((cur - mean) * (cur - mean)))
                    / (old_wt + new_wt);
                sum_wt += new_wt;
                sum_wt2 += new_wt * new_wt;
                old_wt += new_wt;
                if !adjust {
                    sum_wt /= old_wt;
                    sum_wt2 /= old_wt * old_wt;
                    old_wt = 1.0;
                }
            }
        } else if is_observation {
            mean = cur;
        }

        out[i] = if nobs >= minp {
            let numerator = sum_wt * sum_wt;
            let denominator = numerator - sum_wt2;
            if denominator > 0.0 {
                ((numerator / denominator) * cov).sqrt()
            } else {
                f64::NAN
            }
        } else {
            f64::NAN
        };
    }
    Ok(out)
}

/// Applies a per-column transform across a whole matrix.
fn map_columns(m: &Matrix, f: impl Fn(&[f64]) -> Result<Vec<f64>>) -> Result<Matrix> {
    let mut out = Matrix::zeros(m.n_rows(), m.n_cols());
    for col in 0..m.n_cols() {
        let computed = f(m.column(col))?;
        out.column_mut(col).copy_from_slice(&computed);
    }
    Ok(out)
}

macro_rules! matrix_wrapper {
    ($(#[$meta:meta])* $name:ident, $inner:ident, ($($arg:ident : $ty:ty),*)) => {
        $(#[$meta])*
        ///
        /// # Errors
        ///
        /// As the one-dimensional form.
        pub fn $name(m: &Matrix, $($arg: $ty),*) -> Result<Matrix> {
            map_columns(m, |c| $inner(c, $($arg),*))
        }
    };
}

matrix_wrapper!(
    /// [`rolling_mean_1d`] over every column.
    rolling_mean, rolling_mean_1d, (window: usize, minp: Option<usize>)
);
matrix_wrapper!(
    /// [`rolling_std_1d`] over every column.
    rolling_std, rolling_std_1d, (window: usize, minp: Option<usize>, ddof: usize)
);
matrix_wrapper!(
    /// [`rolling_min_1d`] over every column.
    rolling_min, rolling_min_1d, (window: usize, minp: Option<usize>)
);
matrix_wrapper!(
    /// [`rolling_max_1d`] over every column.
    rolling_max, rolling_max_1d, (window: usize, minp: Option<usize>)
);
matrix_wrapper!(
    /// [`ewm_mean_1d`] over every column.
    ewm_mean, ewm_mean_1d, (span: usize, minp: usize, adjust: bool)
);
matrix_wrapper!(
    /// [`ewm_std_1d`] over every column.
    ewm_std, ewm_std_1d, (span: usize, minp: usize, adjust: bool)
);

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "these assert on exact expected window values and on NaN, which \
              is the contract"
)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn a_full_window_is_the_plain_mean() {
        let r = rolling_mean_1d(&[1.0, 2.0, 3.0, 4.0], 2, None).unwrap();
        assert!(r[0].is_nan(), "one observation is short of a 2-bar window");
        assert!(close(r[1], 1.5) && close(r[2], 2.5) && close(r[3], 3.5));
    }

    #[test]
    fn min_periods_decides_when_values_start() {
        // The same window with minp=1 produces a value on bar 0, computed
        // from one observation. That is the trap this parameter sets.
        let r = rolling_mean_1d(&[1.0, 2.0, 3.0], 3, Some(1)).unwrap();
        assert!(close(r[0], 1.0) && close(r[1], 1.5) && close(r[2], 2.0));
    }

    #[test]
    fn min_periods_above_the_window_is_rejected() {
        assert!(rolling_mean_1d(&[1.0], 2, Some(3)).is_err());
        assert!(rolling_mean_1d(&[1.0], 0, None).is_err());
    }

    #[test]
    fn nan_shrinks_the_window_rather_than_poisoning_it() {
        let r = rolling_mean_1d(&[1.0, f64::NAN, 3.0], 3, Some(1)).unwrap();
        assert!(close(r[0], 1.0));
        assert!(close(r[1], 1.0), "the NaN bar contributes nothing");
        assert!(close(r[2], 2.0), "mean of 1 and 3");
    }

    #[test]
    fn a_window_of_only_nan_has_no_value() {
        let r = rolling_mean_1d(&[f64::NAN, f64::NAN], 2, Some(1)).unwrap();
        assert!(r[0].is_nan() && r[1].is_nan());
        let mx = rolling_max_1d(&[f64::NAN, f64::NAN], 2, Some(1)).unwrap();
        assert!(mx[0].is_nan() && mx[1].is_nan(), "not -inf");
    }

    #[test]
    fn a_constant_series_has_exactly_zero_deviation() {
        // The expanded variance form cancels to a tiny negative number here;
        // without the absolute value the square root would be NaN.
        let r = rolling_std_1d(&[0.1, 0.1, 0.1, 0.1], 3, None, 0).unwrap();
        assert_eq!(r[2], 0.0);
        assert_eq!(r[3], 0.0);
    }

    #[test]
    fn ddof_changes_the_divisor() {
        let pop = rolling_std_1d(&[1.0, 2.0, 3.0], 3, None, 0).unwrap();
        let sample = rolling_std_1d(&[1.0, 2.0, 3.0], 3, None, 1).unwrap();
        assert!(close(pop[2], (2.0f64 / 3.0).sqrt()));
        assert!(close(sample[2], 1.0));
    }

    #[test]
    fn a_window_equal_to_ddof_has_no_value() {
        let r = rolling_std_1d(&[5.0], 1, None, 1).unwrap();
        assert!(r[0].is_nan(), "one observation, one degree of freedom");
    }

    #[test]
    fn rolling_extremes_scan_their_window() {
        let a = [3.0, 1.0, 4.0, 1.0, 5.0];
        let mn = rolling_min_1d(&a, 3, None).unwrap();
        let mx = rolling_max_1d(&a, 3, None).unwrap();
        assert!(mn[0].is_nan() && mn[1].is_nan());
        assert_eq!(mn[2], 1.0);
        assert_eq!(mx[2], 4.0);
        assert_eq!(mx[4], 5.0);
    }

    #[test]
    fn expanding_never_forgets() {
        let a = [3.0, 1.0, 4.0, 1.0];
        assert_eq!(expanding_min_1d(&a, 1), [3.0, 1.0, 1.0, 1.0]);
        assert_eq!(expanding_max_1d(&a, 1), [3.0, 3.0, 4.0, 4.0]);
        let m = expanding_mean_1d(&a, 1).unwrap();
        assert!(close(m[3], 2.25));
    }

    #[test]
    fn ewm_without_adjust_is_a_plain_recursion() {
        // span 3 -> alpha 0.5. Seeded at 1, then 0.5 * prev + 0.5 * cur.
        let r = ewm_mean_1d(&[1.0, 3.0, 5.0], 3, 0, false).unwrap();
        assert!(close(r[0], 1.0));
        assert!(close(r[1], 2.0));
        assert!(close(r[2], 3.5));
    }

    #[test]
    fn ewm_with_adjust_rescales_the_early_bars() {
        let plain = ewm_mean_1d(&[1.0, 3.0, 5.0], 3, 0, false).unwrap();
        let adjusted = ewm_mean_1d(&[1.0, 3.0, 5.0], 3, 0, true).unwrap();
        assert!(close(adjusted[0], 1.0));
        assert!(!close(adjusted[1], plain[1]), "and they stay different");
        assert!(!close(adjusted[2], plain[2]));
    }

    #[test]
    fn ewm_leaves_a_constant_series_constant() {
        // The equality guard exists for this: without it, recomputing the
        // same value through the weighted form lets it drift. Most constants
        // survive the round trip unchanged — 0.1 at span 5 does — so this one
        // was searched for. At span 10, 8.19478 becomes 8.194779999999998.
        //
        // The drift is about 2e-15 relative, below the tolerance the
        // golden fixtures compare with, so this is asserted exactly.
        let r = ewm_mean_1d(&[8.19478; 8], 10, 0, false).unwrap();
        assert!(r.iter().all(|v| *v == 8.19478), "got {r:?}");
    }

    #[test]
    fn the_exponential_mean_does_not_fuse_its_multiply_add() {
        // `a.mul_add(b, c)` rounds once where `a * b + c` rounds twice. It is
        // faster and more accurate, and therefore wrong here: the recursion
        // rounds twice. The difference is about one unit in the last place —
        // below the tolerance the golden fixtures compare with, so no
        // fixture can catch it and this test asserts on bits instead.
        //
        // Rather than a magic literal, the fused form is recomputed here and
        // the two are required to differ.
        let a = [1.0, 2.0, 3.0, 4.0];
        let got = ewm_mean_1d(&a, 2, 0, false).unwrap();

        let alpha = 1.0 / (1.0 + (2.0 - 1.0) / 2.0);
        let old_wt_factor = 1.0 - alpha;
        let mut fused = a[0];
        let mut old_wt: f64 = 1.0;
        let mut fused_out = vec![a[0]];
        for &cur in &a[1..] {
            old_wt *= old_wt_factor;
            if fused != cur {
                fused = old_wt.mul_add(fused, alpha * cur) / (old_wt + alpha);
            }
            old_wt = 1.0;
            fused_out.push(fused);
        }

        assert!(
            got.iter()
                .zip(&fused_out)
                .any(|(x, y)| x.to_bits() != y.to_bits()),
            "the fused and unfused forms agreed bitwise, so this test proves nothing"
        );
        // And the difference really is tiny, which is why the fixtures miss it.
        for (x, y) in got.iter().zip(&fused_out) {
            assert!((x - y).abs() < 1e-12, "got {x}, fused {y}");
        }
    }

    #[test]
    fn ewm_std_has_no_value_on_the_first_bar() {
        let r = ewm_std_1d(&[1.0, 3.0, 5.0], 3, 0, false).unwrap();
        assert!(r[0].is_nan(), "one observation has no deviation");
        assert!(r[1] > 0.0 && r[2] > 0.0);
    }

    #[test]
    fn a_span_shorter_than_min_periods_is_rejected() {
        assert!(ewm_mean_1d(&[1.0], 2, 3, false).is_err());
        assert!(ewm_std_1d(&[1.0], 2, 3, false).is_err());
    }

    #[test]
    fn the_matrix_forms_run_column_by_column() {
        let m = Matrix::from_column_major(vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0], 3, 2).unwrap();
        let r = rolling_mean(&m, 2, None).unwrap();
        assert!(r.column(0)[0].is_nan());
        assert!(close(r.column(0)[1], 1.5));
        assert!(close(r.column(1)[1], 15.0), "columns do not mix");
    }
}
