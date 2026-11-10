//! Shifts, differences, fills and crossovers.
//!
//! Small functions, and three of them carry a hazard worth stating.
//!
//! **[`bshift`] looks ahead.** It pulls a future bar into the present, which
//! in a backtest is lookahead bias with a friendly name. It exists because
//! labelling a training set is a legitimate use; it must never appear between a price and a signal.
//!
//! **[`bfill`] looks ahead too**, for the same reason: it fills a gap from
//! the next known value.
//!
//! **[`ffill`] seeds from element zero**, so a leading `NaN` stays `NaN`:
//! there is nothing behind it to carry. Seeding from `NaN` instead would be
//! *identical* — the seed is only ever read when element zero is itself
//! `NaN` — which mutation testing confirms by surviving. The
//! explicit spelling is kept because it says what it means.

use alphax_core::error::Result;
use alphax_core::matrix::Matrix;

/// Shifts a series forward by `n` bars, filling the start with `fill`.
///
/// This is the safe direction: `out[i]` is `a[i - n]`, a value the past
/// already knew.
#[must_use]
pub fn fshift_1d(a: &[f64], n: usize, fill: f64) -> Vec<f64> {
    (0..a.len())
        .map(|i| if i >= n { a[i - n] } else { fill })
        .collect()
}

/// Shifts a series backward by `n` bars, filling the end with `fill`.
///
/// **Looks ahead.** `out[i]` is `a[i + n]`, which bar `i` cannot know. See
/// the module docs.
#[must_use]
pub fn bshift_1d(a: &[f64], n: usize, fill: f64) -> Vec<f64> {
    (0..a.len())
        .map(|i| if i + n < a.len() { a[i + n] } else { fill })
        .collect()
}

/// Difference against the value `n` bars ago.
#[must_use]
pub fn diff_1d(a: &[f64], n: usize) -> Vec<f64> {
    (0..a.len())
        .map(|i| if i >= n { a[i] - a[i - n] } else { f64::NAN })
        .collect()
}

/// Fractional change against the value `n` bars ago.
#[must_use]
pub fn pct_change_1d(a: &[f64], n: usize) -> Vec<f64> {
    (0..a.len())
        .map(|i| {
            if i >= n {
                a[i] / a[i - n] - 1.0
            } else {
                f64::NAN
            }
        })
        .collect()
}

/// Carries the last known value forward over gaps.
#[must_use]
pub fn ffill_1d(a: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(a.len());
    // Seeded from the first element, so a leading gap stays `NaN` — there is
    // nothing behind it to carry.
    let mut last = a.first().copied().unwrap_or(f64::NAN);
    for &v in a {
        if v.is_nan() {
            out.push(last);
        } else {
            last = v;
            out.push(v);
        }
    }
    out
}

/// Carries the next known value backward over gaps.
///
/// **Looks ahead.** See the module docs.
#[must_use]
pub fn bfill_1d(a: &[f64]) -> Vec<f64> {
    let mut out = vec![f64::NAN; a.len()];
    let mut last = a.last().copied().unwrap_or(f64::NAN);
    for i in (0..a.len()).rev() {
        if a[i].is_nan() {
            out[i] = last;
        } else {
            last = a[i];
            out[i] = a[i];
        }
    }
    out
}

/// Replaces every `NaN` with `value`.
#[must_use]
pub fn fillna_1d(a: &[f64], value: f64) -> Vec<f64> {
    a.iter()
        .map(|v| if v.is_nan() { value } else { *v })
        .collect()
}

/// Where `a` crosses above `b`.
///
/// Three rules are easy to get wrong and all three matter:
///
/// - A cross requires having been *strictly below* first. A series that
///   starts above never crosses, however far it rises.
/// - Equality resets the *counter* but not the memory of having been below,
///   so touching the level and then rising **does** count as a cross. Only a
///   `NaN` clears the memory.
/// - A `NaN` on either side resets everything, so a gap in the data cannot
///   manufacture a crossing across it.
///
/// `wait` fires the signal that many bars *after* the crossing, and only if
/// the series has stayed above throughout.
#[must_use]
#[allow(
    clippy::float_cmp,
    reason = "equality is a distinct state here, not an approximation: two \
              series touching is neither above nor below, and a tolerance \
              would invent a band in which neither branch is taken"
)]
pub fn crossed_above_1d(a: &[f64], b: &[f64], wait: usize) -> Vec<bool> {
    let n = a.len().min(b.len());
    let mut out = vec![false; n];
    let mut was_below = false;
    // Counts bars since the cross; -1 means "not currently above after a
    // below", which is why this is signed.
    let mut crossed_ago: i64 = -1;

    for i in 0..n {
        if a[i].is_nan() || b[i].is_nan() {
            crossed_ago = -1;
            was_below = false;
        } else if a[i] > b[i] {
            if was_below {
                crossed_ago += 1;
                out[i] = crossed_ago == wait as i64;
            }
        } else if a[i] == b[i] {
            crossed_ago = -1;
        } else {
            crossed_ago = -1;
            was_below = true;
        }
    }
    out
}

/// Where `a` crosses below `b`, by symmetry.
#[must_use]
pub fn crossed_below_1d(a: &[f64], b: &[f64], wait: usize) -> Vec<bool> {
    crossed_above_1d(b, a, wait)
}

/// Applies a per-column transform across a whole matrix.
fn map_columns(m: &Matrix, f: impl Fn(&[f64]) -> Vec<f64>) -> Matrix {
    let mut out = Matrix::zeros(m.n_rows(), m.n_cols());
    for col in 0..m.n_cols() {
        out.column_mut(col).copy_from_slice(&f(m.column(col)));
    }
    out
}

/// [`fshift_1d`] over every column.
#[must_use]
pub fn fshift(m: &Matrix, n: usize, fill: f64) -> Matrix {
    map_columns(m, |c| fshift_1d(c, n, fill))
}

/// [`bshift_1d`] over every column. **Looks ahead.**
#[must_use]
pub fn bshift(m: &Matrix, n: usize, fill: f64) -> Matrix {
    map_columns(m, |c| bshift_1d(c, n, fill))
}

/// [`diff_1d`] over every column.
#[must_use]
pub fn diff(m: &Matrix, n: usize) -> Matrix {
    map_columns(m, |c| diff_1d(c, n))
}

/// [`pct_change_1d`] over every column.
#[must_use]
pub fn pct_change(m: &Matrix, n: usize) -> Matrix {
    map_columns(m, |c| pct_change_1d(c, n))
}

/// [`ffill_1d`] over every column.
#[must_use]
pub fn ffill(m: &Matrix) -> Matrix {
    map_columns(m, ffill_1d)
}

/// [`bfill_1d`] over every column. **Looks ahead.**
#[must_use]
pub fn bfill(m: &Matrix) -> Matrix {
    map_columns(m, bfill_1d)
}

/// [`fillna_1d`] over every column.
#[must_use]
pub fn fillna(m: &Matrix, value: f64) -> Matrix {
    map_columns(m, |c| fillna_1d(c, value))
}

/// [`crossed_above_1d`] over every column of two matrices.
///
/// # Errors
///
/// Returns [`alphax_core::Error::Broadcast`] if the shapes differ.
pub fn crossed_above(a: &Matrix, b: &Matrix, wait: usize) -> Result<Vec<Vec<bool>>> {
    if a.n_rows() != b.n_rows() || a.n_cols() != b.n_cols() {
        return Err(alphax_core::Error::Broadcast {
            left: (a.n_rows(), a.n_cols()),
            right: (b.n_rows(), b.n_cols()),
        });
    }
    Ok((0..a.n_cols())
        .map(|c| crossed_above_1d(a.column(c), b.column(c), wait))
        .collect())
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "these assert on exact shifted and filled values"
)]
mod tests {
    use super::*;

    #[test]
    fn fshift_pulls_the_past_forward() {
        assert_eq!(fshift_1d(&[1.0, 2.0, 3.0], 1, 0.0), [0.0, 1.0, 2.0]);
    }

    #[test]
    fn bshift_pulls_the_future_back() {
        // Lookahead by construction; see the module docs.
        assert_eq!(bshift_1d(&[1.0, 2.0, 3.0], 1, 0.0), [2.0, 3.0, 0.0]);
    }

    #[test]
    fn a_shift_longer_than_the_series_is_all_fill() {
        assert_eq!(fshift_1d(&[1.0, 2.0], 5, -1.0), [-1.0, -1.0]);
        assert_eq!(bshift_1d(&[1.0, 2.0], 5, -1.0), [-1.0, -1.0]);
    }

    #[test]
    fn diff_and_pct_change_start_with_nan() {
        let d = diff_1d(&[2.0, 5.0, 9.0], 1);
        assert!(d[0].is_nan());
        assert_eq!(d[1], 3.0);
        let p = pct_change_1d(&[2.0, 4.0], 1);
        assert!(p[0].is_nan());
        assert_eq!(p[1], 1.0);
    }

    #[test]
    fn ffill_leaves_a_leading_gap_alone() {
        let r = ffill_1d(&[f64::NAN, 1.0, f64::NAN, 2.0]);
        assert!(r[0].is_nan(), "nothing precedes it to carry forward");
        assert_eq!(r[1], 1.0);
        assert_eq!(r[2], 1.0);
        assert_eq!(r[3], 2.0);
    }

    #[test]
    fn bfill_leaves_a_trailing_gap_alone() {
        let r = bfill_1d(&[1.0, f64::NAN, 2.0, f64::NAN]);
        assert_eq!(r[1], 2.0);
        assert!(r[3].is_nan());
    }

    #[test]
    fn a_series_that_starts_above_never_crosses() {
        let a = [2.0, 3.0, 4.0];
        let b = [1.0, 1.0, 1.0];
        assert_eq!(crossed_above_1d(&a, &b, 0), [false, false, false]);
    }

    #[test]
    fn a_cross_needs_to_have_been_below_first() {
        let a = [0.0, 2.0, 3.0];
        let b = [1.0, 1.0, 1.0];
        assert_eq!(crossed_above_1d(&a, &b, 0), [false, true, false]);
    }

    #[test]
    fn touching_the_level_and_then_rising_is_a_cross() {
        // Equality resets the counter but not the memory of having been
        // below, so this fires on the rise. Asserted because it is the
        // opposite of what the code reads like at a glance — and it is the
        // intended behaviour.
        let a = [0.0, 1.0, 2.0];
        let b = [1.0, 1.0, 1.0];
        assert_eq!(crossed_above_1d(&a, &b, 0), [false, false, true]);
    }

    #[test]
    fn a_gap_in_the_data_cannot_manufacture_a_cross() {
        let a = [0.0, f64::NAN, 2.0];
        let b = [1.0, 1.0, 1.0];
        assert_eq!(crossed_above_1d(&a, &b, 0), [false, false, false]);
    }

    #[test]
    fn wait_delays_the_signal_and_requires_staying_above() {
        let a = [0.0, 2.0, 3.0, 4.0];
        let b = [1.0, 1.0, 1.0, 1.0];
        assert_eq!(crossed_above_1d(&a, &b, 1), [false, false, true, false]);
        // Dropping back before the wait elapses cancels it.
        let a = [0.0, 2.0, 0.5, 4.0];
        assert_eq!(crossed_above_1d(&a, &b, 1), [false, false, false, false]);
    }

    #[test]
    fn crossed_below_is_the_mirror() {
        let a = [2.0, 0.0];
        let b = [1.0, 1.0];
        assert_eq!(crossed_below_1d(&a, &b, 0), [false, true]);
    }

    #[test]
    fn mismatched_shapes_are_rejected() {
        let a = Matrix::zeros(3, 1);
        let b = Matrix::zeros(2, 1);
        assert!(crossed_above(&a, &b, 0).is_err());
    }
}
