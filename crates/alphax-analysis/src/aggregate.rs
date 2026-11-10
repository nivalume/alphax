//! Reducing a sweep's per-column metrics to an answer.
//!
//! A thousand-column sweep produces a thousand Sharpe ratios. That vector is
//! not an answer by itself — "which quantile threshold was best", "which factor
//! held up across symbols" is. Those questions are all the same shape: group
//! the columns by one label level, then reduce within each group.
//!
//! This is why multi-level column labels live in the foundation crate rather
//! than being a presentation concern. Without them a sweep's output is a pile
//! of numbers with no way back to what produced each one.
//!
//! `NaN` columns are skipped rather than propagated, on the same reasoning as
//! [`crate::metrics`]: a column that could not be measured should not erase
//! the ones that could. A group with nothing measurable reduces to `NaN`.
//!
//! # Infinities are *not* skipped, and that is a trap worth knowing
//!
//! A column whose signals never fired has a flat equity curve: zero return
//! and zero volatility. [`crate::metrics::sharpe_ratio`] divides one by the
//! other and answers `+inf` — so **an untraded column
//! sorts as the best in the sweep**, and one of them makes its whole group's
//! mean infinite.
//!
//! Nothing here silently drops it, because "this column did nothing" is a
//! result a sweep needs to see rather than have hidden. What a caller does
//! instead is say what it is ranking: mask the columns that failed a
//! precondition to `NaN`, which every reduction here skips.
//!
//! ```text
//! let traded: Vec<f64> = sharpe.iter().zip(&stats)
//!     .map(|(s, st)| if st.count == 0 { f64::NAN } else { *s })
//!     .collect();
//! ```
//!
//! `examples/sweep.py` does exactly that, and
//! `crates/alphax/tests/sweep.rs` pins both halves — the naive ranking finding
//! the untraded column, and the filtered one finding a real column.

use alphax_core::columns::{Columns, Label};
use alphax_core::error::Result;

/// The index of the largest value, ignoring `NaN`.
///
/// `None` when every value is `NaN`. Ties go to the first, so a ranking is
/// stable under reordering that does not change values.
#[must_use]
pub fn argmax(values: &[f64]) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (i, &v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        match best {
            Some((_, b)) if v <= b => {}
            _ => best = Some((i, v)),
        }
    }
    best.map(|(i, _)| i)
}

/// The index of the smallest value, ignoring `NaN`.
#[must_use]
pub fn argmin(values: &[f64]) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (i, &v) in values.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        match best {
            Some((_, b)) if v >= b => {}
            _ => best = Some((i, v)),
        }
    }
    best.map(|(i, _)| i)
}

/// How to reduce the columns within a group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduce {
    /// Arithmetic mean of the non-`NaN` values.
    Mean,
    /// Median of the non-`NaN` values.
    Median,
    /// Largest non-`NaN` value.
    Max,
    /// Smallest non-`NaN` value.
    Min,
    /// Number of non-`NaN` values.
    Count,
}

impl Reduce {
    /// Applies this reduction to a group's values.
    #[must_use]
    pub fn apply(self, values: &[f64]) -> f64 {
        let mut live: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
        if self == Self::Count {
            return live.len() as f64;
        }
        if live.is_empty() {
            return f64::NAN;
        }
        match self {
            Self::Mean => live.iter().sum::<f64>() / live.len() as f64,
            Self::Median => {
                live.sort_by(f64::total_cmp);
                let n = live.len();
                if n % 2 == 1 {
                    live[n / 2]
                } else {
                    (live[n / 2 - 1] + live[n / 2]) / 2.0
                }
            }
            Self::Max => live.into_iter().fold(f64::NEG_INFINITY, f64::max),
            Self::Min => live.into_iter().fold(f64::INFINITY, f64::min),
            Self::Count => unreachable!("handled above"),
        }
    }
}

/// Groups a per-column metric by one label level and reduces each group.
///
/// Groups come back in first-appearance order, not sorted, so the result
/// keeps the sweep's own ordering of parameter values.
///
/// # Errors
///
/// Returns [`alphax_core::Error::UnknownLevel`] if `level` does not exist, or
/// [`alphax_core::Error::Invariant`] if `values` is not one per column.
pub fn group_by(
    columns: &Columns,
    values: &[f64],
    level: &str,
    reduce: Reduce,
) -> Result<Vec<(Label, f64)>> {
    if values.len() != columns.n_cols() {
        return Err(alphax_core::Error::Invariant(format!(
            "expected one value per column: got {} for {} columns",
            values.len(),
            columns.n_cols()
        )));
    }
    Ok(columns
        .group_indices(level)?
        .into_iter()
        .map(|(label, cols)| {
            let group: Vec<f64> = cols.into_iter().map(|c| values[c]).collect();
            (label, reduce.apply(&group))
        })
        .collect())
}

/// The best column overall, as its full label set.
///
/// # Errors
///
/// Returns [`alphax_core::Error::Invariant`] if `values` is not one per column.
pub fn best_column(columns: &Columns, values: &[f64]) -> Result<Option<(usize, String, f64)>> {
    if values.len() != columns.n_cols() {
        return Err(alphax_core::Error::Invariant(format!(
            "expected one value per column: got {} for {} columns",
            values.len(),
            columns.n_cols()
        )));
    }
    Ok(argmax(values).map(|col| (col, columns.describe(col), values[col])))
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "reductions of exact inputs have exact expected values"
)]
mod tests {
    use super::*;
    use alphax_core::columns::Level;

    /// Two quantiles crossed with three symbols: 6 columns.
    ///
    /// Parameters repeat and symbols tile, so a parameter's columns are
    /// contiguous — the layout `alphax_core::columns` fixes.
    fn sweep() -> Columns {
        let quantile = Level::repeat("quantile", &[Label::from("0.8"), Label::from("0.9")], 3);
        let symbol = Level::tile(
            "symbol",
            &[Label::from("BTC"), Label::from("ETH"), Label::from("SOL")],
            2,
        );
        Columns::new(vec![quantile, symbol]).unwrap()
    }

    #[test]
    fn argmax_ignores_nan() {
        assert_eq!(argmax(&[1.0, f64::NAN, 3.0, 2.0]), Some(2));
        assert_eq!(argmax(&[f64::NAN, f64::NAN]), None);
        assert_eq!(argmin(&[1.0, f64::NAN, 3.0]), Some(0));
    }

    #[test]
    fn argmax_breaks_ties_toward_the_first() {
        assert_eq!(argmax(&[2.0, 2.0]), Some(0));
    }

    #[test]
    fn grouping_by_parameter_averages_across_symbols() {
        let c = sweep();
        // quantile 0.8 -> cols 0,1,2; quantile 0.9 -> cols 3,4,5.
        let values = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let g = group_by(&c, &values, "quantile", Reduce::Mean).unwrap();
        assert_eq!(g.len(), 2);
        assert_eq!(g[0].1, 2.0);
        assert_eq!(g[1].1, 20.0);
    }

    #[test]
    fn grouping_by_symbol_averages_across_parameters() {
        let c = sweep();
        let values = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let g = group_by(&c, &values, "symbol", Reduce::Mean).unwrap();
        // BTC is columns 0 and 3.
        assert_eq!(g.len(), 3);
        assert_eq!(g[0].0, Label::from("BTC"));
        assert_eq!(g[0].1, 5.5);
    }

    #[test]
    fn a_group_of_only_nan_reduces_to_nan() {
        let c = sweep();
        let values = [f64::NAN, f64::NAN, f64::NAN, 1.0, 2.0, 3.0];
        let g = group_by(&c, &values, "quantile", Reduce::Mean).unwrap();
        assert!(g[0].1.is_nan());
        assert_eq!(g[1].1, 2.0);
        // Count still distinguishes "no columns measured" from "none exist".
        let counted = group_by(&c, &values, "quantile", Reduce::Count).unwrap();
        assert_eq!(counted[0].1, 0.0);
        assert_eq!(counted[1].1, 3.0);
    }

    #[test]
    fn the_best_column_comes_back_labelled() {
        let c = sweep();
        let values = [1.0, 2.0, 3.0, 10.0, 99.0, 30.0];
        let (col, label, value) = best_column(&c, &values).unwrap().unwrap();
        assert_eq!(col, 4);
        assert_eq!(value, 99.0);
        assert!(label.contains("quantile=0.9"), "got {label}");
        assert!(label.contains("symbol=ETH"), "got {label}");
    }

    #[test]
    fn a_metric_of_the_wrong_width_is_rejected() {
        let c = sweep();
        assert!(group_by(&c, &[1.0, 2.0], "symbol", Reduce::Mean).is_err());
        assert!(best_column(&c, &[1.0]).is_err());
    }

    #[test]
    fn an_unknown_level_is_rejected() {
        let c = sweep();
        assert!(group_by(&c, &[0.0; 6], "factor", Reduce::Mean).is_err());
    }
}
