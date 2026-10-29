//! Quantile groups of a factor, and what each group earns.
//!
//! Labels are a same-bar transform of the factor: bar `t` uses only bar `t`'s
//! values. [`quantile_returns`] scores those labels against a forward-return
//! panel, which does read the future, on purpose — it is how a group is
//! judged, and it is not a signal.

use phasor_core::error::{Error, Result};

use crate::corr::Corr;
use crate::panel::Panel;
use crate::rank::Ranker;

/// `q` as a quantile count: at least one group, and small enough that a label
/// fits in `i16`.
pub(crate) fn check_q(q: usize) -> Result<()> {
    if q == 0 || q > i16::MAX as usize {
        return Err(Error::Invariant(format!(
            "q must be from 1 to {}, got {q}",
            i16::MAX
        )));
    }
    Ok(())
}

/// `floor((rank - 0.5) * q / n)`, clamped to `0..q`.
///
/// `rank` is a 1-based average rank. Subtracting a half puts a tied pair on
/// the boundary the same way a unique pair would sit, so a tied row and an
/// untied row of the same length split across `0..q` symmetrically.
fn bucket(rank: f64, q: usize, n: usize) -> i16 {
    let raw = ((rank - 0.5) * (q as f64) / (n as f64)).floor();
    let max = (q - 1) as f64;
    raw.clamp(0.0, max) as i16
}

/// Quantile label of every value, column-major, `-1` where the value is
/// missing or the row has fewer than `q` finite values.
///
/// On a row with `n >= q` finite values, those values are average-ranked
/// (ties share the mean of the ranks they span) and value with rank `r`
/// goes to group `floor((r - 0.5) * q / n)`, clamped to `0..q`. Group `0`
/// holds the smallest factor values. A row with fewer than `q` finite values
/// is all `-1`: it is not split into empty groups.
///
/// # Errors
///
/// [`Error::Invariant`] if `q` is 0 or does not fit in `i16`.
pub fn quantile_labels(factor: &Panel<'_>, q: usize) -> Result<Vec<i16>> {
    check_q(q)?;
    let (n_rows, n_cols) = (factor.n_rows(), factor.n_cols());
    let mut out = vec![-1i16; factor.data().len()];
    let mut finite = Vec::with_capacity(n_cols);
    let mut index = Vec::with_capacity(n_cols);
    let mut ranks = Vec::new();
    let mut ranker = Ranker::default();
    for t in 0..n_rows {
        finite.clear();
        index.clear();
        for c in 0..n_cols {
            let v = factor.col(c)[t];
            if v.is_finite() {
                finite.push(v);
                index.push(c);
            }
        }
        let n = finite.len();
        if n < q {
            continue;
        }
        ranker.rank(&finite, &mut ranks);
        for (&c, &rank) in index.iter().zip(&ranks) {
            out[c * n_rows + t] = bucket(rank, q, n);
        }
    }
    Ok(out)
}

/// Equal-weighted or weighted mean forward return of each quantile group.
///
/// `labels` is column-major `(n_rows, n_cols)`, as [`quantile_labels`]
/// returns it. The result is column-major `(n_rows, q)`: column `g` is group
/// `g`'s return on every bar. A name counts on bar `t` when its label is `g`
/// and its forward return is finite. With no weights the group earns the
/// arithmetic mean. With weights, a non-finite weight is skipped and the
/// group earns `Σ w r / Σ w`; a sum of weights that is not strictly positive
/// is `NaN`, as is a group with no names.
///
/// # Errors
///
/// [`Error::Invariant`] if `q` is out of range, `labels` is not
/// `n_rows * n_cols` long, `weights` has a different shape, or
/// `n_rows * q` overflows.
pub fn quantile_returns(
    labels: &[i16],
    fwd: &Panel<'_>,
    q: usize,
    weights: Option<&Panel<'_>>,
) -> Result<Vec<f64>> {
    check_q(q)?;
    let (n_rows, n_cols) = (fwd.n_rows(), fwd.n_cols());
    let n = n_rows
        .checked_mul(n_cols)
        .ok_or_else(|| Error::Invariant(format!("shape {n_rows} x {n_cols} overflows")))?;
    if labels.len() != n {
        return Err(Error::Invariant(format!(
            "labels have {} values, expected {n_rows} x {n_cols}",
            labels.len()
        )));
    }
    if let Some(w) = weights {
        fwd.same_shape(w, "weights")?;
    }
    let out_len = n_rows
        .checked_mul(q)
        .ok_or_else(|| Error::Invariant(format!("shape {n_rows} x {q} overflows")))?;
    let mut out = vec![f64::NAN; out_len];
    let mut sum = vec![0.0; q];
    let mut weight_sum = vec![0.0; q];
    let mut count = vec![0usize; q];
    for t in 0..n_rows {
        sum.fill(0.0);
        weight_sum.fill(0.0);
        count.fill(0);
        for c in 0..n_cols {
            let label = labels[c * n_rows + t];
            if label < 0 {
                continue;
            }
            let g = label as usize;
            if g >= q {
                continue;
            }
            let ret = fwd.col(c)[t];
            if !ret.is_finite() {
                continue;
            }
            match weights {
                None => {
                    sum[g] += ret;
                    count[g] += 1;
                }
                Some(w) => {
                    let weight = w.col(c)[t];
                    if !weight.is_finite() {
                        continue;
                    }
                    sum[g] += weight * ret;
                    weight_sum[g] += weight;
                }
            }
        }
        for g in 0..q {
            let dst = g * n_rows + t;
            match weights {
                None => {
                    if count[g] > 0 {
                        out[dst] = sum[g] / count[g] as f64;
                    }
                }
                Some(_) if weight_sum[g] > 0.0 => {
                    out[dst] = sum[g] / weight_sum[g];
                }
                Some(_) => {}
            }
        }
    }
    Ok(out)
}

/// Top group minus bottom group, one value per bar.
///
/// `q_returns` is column-major `(n_rows, q)`. Entry `t` is
/// `q_returns[(q - 1) * n_rows + t] - q_returns[t]`. `NaN` propagates.
///
/// # Errors
///
/// [`Error::Invariant`] if `q` is 0 or `q_returns` is not `n_rows * q` long.
pub fn long_short_returns(q_returns: &[f64], n_rows: usize, q: usize) -> Result<Vec<f64>> {
    if q == 0 {
        return Err(Error::Invariant("q must be 1 or more".into()));
    }
    let n = n_rows
        .checked_mul(q)
        .ok_or_else(|| Error::Invariant(format!("shape {n_rows} x {q} overflows")))?;
    if q_returns.len() != n {
        return Err(Error::Invariant(format!(
            "q_returns has {} values, expected {n_rows} x {q}",
            q_returns.len()
        )));
    }
    let top = (q - 1) * n_rows;
    let mut out = vec![f64::NAN; n_rows];
    for t in 0..n_rows {
        out[t] = q_returns[top + t] - q_returns[t];
    }
    Ok(out)
}

/// Spearman rank correlation of `0 .. q-1` with the mean return of each group.
///
/// `NaN` when fewer than two groups are given, any mean is not finite, or the
/// means are constant: a flat profile has no monotone direction.
#[must_use]
pub fn monotonicity(mean_q_returns: &[f64]) -> f64 {
    if mean_q_returns.len() < 2 || mean_q_returns.iter().any(|v| !v.is_finite()) {
        return f64::NAN;
    }
    let mut corr = Corr::default();
    for (i, &mean) in mean_q_returns.iter().enumerate() {
        corr.push(i as f64, mean);
    }
    corr.spearman()
}

#[cfg(test)]
mod tests {
    use super::*;
    use phasor_core::math::is_close;

    /// Row-major `rows` as a column-major panel buffer.
    fn panel(rows: &[&[f64]]) -> (Vec<f64>, usize, usize) {
        let n_rows = rows.len();
        let n_cols = rows.first().map_or(0, |row| row.len());
        let mut data = vec![0.0; n_rows * n_cols];
        for (t, row) in rows.iter().enumerate() {
            assert_eq!(row.len(), n_cols);
            for (c, &v) in row.iter().enumerate() {
                data[c * n_rows + t] = v;
            }
        }
        (data, n_rows, n_cols)
    }

    fn labels_by_row(labels: &[i16], n_rows: usize, n_cols: usize) -> Vec<Vec<i16>> {
        (0..n_rows)
            .map(|t| (0..n_cols).map(|c| labels[c * n_rows + t]).collect())
            .collect()
    }

    #[test]
    fn labels_split_ranks_skip_missing_and_refuse_a_short_row() {
        let (data, n_rows, n_cols) = panel(&[
            &[1.0, 3.0, 2.0, f64::NAN],
            &[1.0, 1.0, 2.0, 3.0],
            &[5.0, f64::NAN, 6.0, f64::NAN],
            &[1.0, f64::NAN, 2.0, 4.0],
        ]);
        let view = Panel::new(&data, n_rows, n_cols).unwrap();
        let labels = quantile_labels(&view, 2).unwrap();
        assert_eq!(
            labels_by_row(&labels, n_rows, n_cols),
            vec![
                vec![0, 1, 1, -1],
                vec![0, 0, 1, 1],
                vec![0, -1, 1, -1],
                vec![0, -1, 1, 1],
            ]
        );

        // Three finite values cannot fill four groups: the whole row is missing.
        let wide = quantile_labels(&view, 4).unwrap();
        assert_eq!(
            labels_by_row(&wide, n_rows, n_cols)[3],
            vec![-1, -1, -1, -1]
        );
        // Tied row, q = 4: ranks 1.5, 1.5, 3, 4 → buckets 1, 1, 2, 3.
        assert_eq!(labels_by_row(&wide, n_rows, n_cols)[1], vec![1, 1, 2, 3]);
    }

    #[test]
    fn ties_include_signed_zero_and_a_constant_row_lands_in_the_middle() {
        let (data, n_rows, n_cols) = panel(&[&[-0.0, 0.0, 1.0, 2.0], &[7.0, 7.0, 7.0, 7.0]]);
        let view = Panel::new(&data, n_rows, n_cols).unwrap();
        let labels = quantile_labels(&view, 2).unwrap();
        assert_eq!(labels_by_row(&labels, n_rows, n_cols)[0], vec![0, 0, 1, 1]);

        let middle = quantile_labels(&view, 4).unwrap();
        // Every value ties at rank 2.5; (2.5 - 0.5) * 4 / 4 = 2.
        assert_eq!(labels_by_row(&middle, n_rows, n_cols)[1], vec![2, 2, 2, 2]);
    }

    #[test]
    fn one_group_labels_every_finite_value_zero() {
        let (data, n_rows, n_cols) = panel(&[&[1.0, f64::NAN, 2.0]]);
        let view = Panel::new(&data, n_rows, n_cols).unwrap();
        let labels = quantile_labels(&view, 1).unwrap();
        assert_eq!(labels_by_row(&labels, n_rows, n_cols), vec![vec![0, -1, 0]]);
        assert!(quantile_labels(&view, 0).is_err());
    }

    #[test]
    fn returns_are_column_major_group_means() {
        // Row 0 labels [0, 1], row 1 labels [1, 0].
        let labels = vec![0, 1, 1, 0];
        let (data, n_rows, n_cols) = panel(&[&[0.1, 0.4], &[0.2, 0.3]]);
        let fwd = Panel::new(&data, n_rows, n_cols).unwrap();
        let got = quantile_returns(&labels, &fwd, 2, None).unwrap();
        // group 0: [0.1, 0.3], group 1: [0.4, 0.2]
        assert_eq!(got.len(), 4);
        assert!(is_close(got[0], 0.1) && is_close(got[1], 0.3));
        assert!(is_close(got[2], 0.4) && is_close(got[3], 0.2));

        let spread = long_short_returns(&got, n_rows, 2).unwrap();
        assert!(is_close(spread[0], 0.3) && is_close(spread[1], -0.1));
    }

    #[test]
    fn missing_forward_and_non_positive_weights_are_nan() {
        let labels = vec![0, 0, 1, -1];
        let (data, n_rows, n_cols) = panel(&[&[0.1, f64::NAN, 0.4, 0.9]]);
        let fwd = Panel::new(&data, n_rows, n_cols).unwrap();
        let equal = quantile_returns(&labels, &fwd, 2, None).unwrap();
        assert!(is_close(equal[0], 0.1));
        assert!(is_close(equal[1], 0.4));

        let (wdata, _, _) = panel(&[&[1.0, 3.0, 0.0, 5.0]]);
        let weights = Panel::new(&wdata, n_rows, n_cols).unwrap();
        let weighted = quantile_returns(&labels, &fwd, 2, Some(&weights)).unwrap();
        // Group 0 keeps only the first name (the second return is missing): weight 1.
        assert!(is_close(weighted[0], 0.1));
        // Group 1's only finite return has weight 0.
        assert!(weighted[1].is_nan());

        // Negative weights are kept. (-1 * 0.1 + 3 * 0.2) / 2 = 0.25.
        let labels = vec![0i16, 0];
        let (both, n_rows, n_cols) = panel(&[&[0.1, 0.2]]);
        let fwd = Panel::new(&both, n_rows, n_cols).unwrap();
        let (wdata, _, _) = panel(&[&[-1.0, 3.0]]);
        let weights = Panel::new(&wdata, n_rows, n_cols).unwrap();
        let got = quantile_returns(&labels, &fwd, 1, Some(&weights)).unwrap();
        assert!(is_close(got[0], 0.25));
    }

    #[test]
    fn shape_mismatches_are_errors() {
        let (data, n_rows, n_cols) = panel(&[&[0.1, 0.2]]);
        let fwd = Panel::new(&data, n_rows, n_cols).unwrap();
        assert!(quantile_returns(&[0], &fwd, 1, None).is_err());
        let (wdata, _, _) = panel(&[&[1.0, 2.0, 3.0]]);
        let weights = Panel::new(&wdata, 1, 3).unwrap();
        assert!(quantile_returns(&[0, 0], &fwd, 1, Some(&weights)).is_err());
        assert!(long_short_returns(&[0.1, 0.2], 2, 2).is_err());
        assert!(long_short_returns(&[0.1], 1, 0).is_err());
    }

    #[test]
    fn monotonicity_is_rank_correlation_with_the_group_index() {
        assert!(is_close(monotonicity(&[0.1, 0.3, 0.2]), 0.5));
        assert!(is_close(monotonicity(&[0.0, 1.0, 2.0, 3.0]), 1.0));
        assert!(is_close(monotonicity(&[3.0, 2.0, 1.0, 0.0]), -1.0));
        assert!(monotonicity(&[1.0, 1.0, 1.0]).is_nan());
        assert!(monotonicity(&[1.0]).is_nan());
        assert!(monotonicity(&[1.0, f64::NAN]).is_nan());
        assert!(monotonicity(&[]).is_nan());
    }
}
