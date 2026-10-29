//! How fast a factor's grouping and its ranking change from bar to bar.
//!
//! Neither function reads a later bar. Turnover compares a bar with an
//! earlier one; rank autocorrelation does the same for the ordering.

use phasor_core::error::{Error, Result};

use crate::corr::Corr;
use crate::panel::Panel;
use crate::quantile::check_q;

/// Turnover of each quantile group, column-major `(n_rows, q)`.
///
/// On bar `t >= lag`, group `g`'s turnover is
/// `1 - |S_t ∩ S_{t-lag}| / |S_t|`, where `S` is the set of symbols labelled
/// `g`. An empty `S_t` is `NaN` (there is no book to turn over). Bars before
/// `lag` are `NaN`. A symbol labelled outside `0..q` is in no group.
///
/// # Errors
///
/// [`Error::Invariant`] if `q` is out of range, or `labels` is not
/// `n_rows * n_cols` long.
pub fn quantile_turnover(
    labels: &[i16],
    n_rows: usize,
    n_cols: usize,
    q: usize,
    lag: usize,
) -> Result<Vec<f64>> {
    check_q(q)?;
    let n = n_rows
        .checked_mul(n_cols)
        .ok_or_else(|| Error::Invariant(format!("shape {n_rows} x {n_cols} overflows")))?;
    if labels.len() != n {
        return Err(Error::Invariant(format!(
            "labels have {} values, expected {n_rows} x {n_cols}",
            labels.len()
        )));
    }
    let out_len = n_rows
        .checked_mul(q)
        .ok_or_else(|| Error::Invariant(format!("shape {n_rows} x {q} overflows")))?;
    let mut out = vec![f64::NAN; out_len];
    let mut count = vec![0usize; q];
    let mut both = vec![0usize; q];
    for t in lag..n_rows {
        count.fill(0);
        both.fill(0);
        let prev = t - lag;
        for c in 0..n_cols {
            let now = labels[c * n_rows + t];
            if now < 0 {
                continue;
            }
            let g = now as usize;
            if g >= q {
                continue;
            }
            count[g] += 1;
            if labels[c * n_rows + prev] == now {
                both[g] += 1;
            }
        }
        for g in 0..q {
            if count[g] == 0 {
                continue;
            }
            out[g * n_rows + t] = 1.0 - (both[g] as f64) / (count[g] as f64);
        }
    }
    Ok(out)
}

/// Spearman rank correlation of each bar with the bar `lag` before it.
///
/// The correlation is taken across symbols where both bars are finite.
/// `NaN` before `lag`, and where fewer than two symbols are in common or
/// either side is constant.
///
/// # Errors
///
/// Never, today; the [`Result`] matches the other kernels.
pub fn rank_autocorr(factor: &Panel<'_>, lag: usize) -> Result<Vec<f64>> {
    let n_rows = factor.n_rows();
    let n_cols = factor.n_cols();
    let mut out = vec![f64::NAN; n_rows];
    let mut corr = Corr::default();
    for (t, slot) in out.iter_mut().enumerate().skip(lag) {
        corr.clear();
        let prev = t - lag;
        for c in 0..n_cols {
            let col = factor.col(c);
            corr.push(col[t], col[prev]);
        }
        *slot = corr.spearman();
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use phasor_core::math::is_close;

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

    /// Column-major labels from row-major rows.
    fn col_major(rows: &[Vec<i16>]) -> Vec<i16> {
        let n_rows = rows.len();
        let n_cols = rows.first().map_or(0, Vec::len);
        let mut out = vec![0i16; n_rows * n_cols];
        for (t, row) in rows.iter().enumerate() {
            for (c, &label) in row.iter().enumerate() {
                out[c * n_rows + t] = label;
            }
        }
        out
    }

    #[test]
    fn turnover_is_one_minus_the_share_that_stayed() {
        // t=0: [0, 0, 1, 1]
        // t=1: [0, 1, 1, 1]
        // t=2: [1, 1, 0, 0]
        let labels = col_major(&[vec![0, 0, 1, 1], vec![0, 1, 1, 1], vec![1, 1, 0, 0]]);
        let got = quantile_turnover(&labels, 3, 4, 2, 1).unwrap();
        // group 0: [nan, 0, 1], group 1: [nan, 1/3, 0.5]
        assert!(got[0].is_nan() && got[3].is_nan());
        assert!(is_close(got[1], 0.0) && is_close(got[2], 1.0));
        assert!(is_close(got[4], 1.0 / 3.0) && is_close(got[5], 0.5));
    }

    #[test]
    fn an_empty_group_and_a_lag_past_the_sample_are_nan() {
        let labels = col_major(&[vec![0, 0], vec![-1, -1]]);
        let got = quantile_turnover(&labels, 2, 2, 2, 1).unwrap();
        assert!(got.iter().all(|v| v.is_nan()));

        let same = quantile_turnover(&labels, 2, 2, 2, 0).unwrap();
        // Lag 0 compares a group with itself: turnover 0 where the group is non-empty.
        assert!(is_close(same[0], 0.0));
        assert!(same[1].is_nan());

        let late = quantile_turnover(&labels, 2, 2, 2, 5).unwrap();
        assert!(late.iter().all(|v| v.is_nan()));
        assert!(quantile_turnover(&[0, 0], 2, 2, 0, 1).is_err());
        assert!(quantile_turnover(&[0], 2, 2, 1, 1).is_err());
    }

    #[test]
    fn rank_autocorr_is_spearman_of_the_two_rows() {
        let (data, n_rows, n_cols) = panel(&[
            &[1.0, 2.0, 3.0],
            &[3.0, 2.0, 1.0],
            &[1.0, f64::NAN, 3.0],
            &[5.0, 5.0, 5.0],
        ]);
        let view = Panel::new(&data, n_rows, n_cols).unwrap();
        let got = rank_autocorr(&view, 1).unwrap();
        assert!(got[0].is_nan());
        assert!(is_close(got[1], -1.0));
        // t=2 against t=1 keeps symbols 0 and 2: (1, 3) vs (3, 1).
        assert!(is_close(got[2], -1.0));
        // t=3 is constant.
        assert!(got[3].is_nan());

        let lag2 = rank_autocorr(&view, 2).unwrap();
        assert!(lag2[0].is_nan() && lag2[1].is_nan());
        // t=2 against t=0 keeps symbols 0 and 2: (1, 3) vs (1, 3).
        assert!(is_close(lag2[2], 1.0));

        let self_corr = rank_autocorr(&view, 0).unwrap();
        assert!(is_close(self_corr[0], 1.0));
        assert!(self_corr[3].is_nan());

        let (thin, n_rows, n_cols) = panel(&[&[1.0, f64::NAN], &[2.0, f64::NAN]]);
        let thin = Panel::new(&thin, n_rows, n_cols).unwrap();
        assert!(rank_autocorr(&thin, 1).unwrap()[1].is_nan());
    }
}
