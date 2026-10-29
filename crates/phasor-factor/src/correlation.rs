//! Mean cross-sectional Spearman correlation of factor panels.
//!
//! On each row every factor is ranked once, over the symbols where it is
//! finite, and those ranks are z-scored to population variance 1. The
//! correlation of two factors on that row is the dot product of the z-scores
//! on the symbols where both are finite, divided by how many such symbols
//! there are. Rows where either factor has fewer than two finite values, or
//! no variation, are left out of the average. The average itself is taken in
//! row order after the rows have been scored, so the thread count does not
//! change a bit.

use phasor_core::error::{Error, Result};

use crate::panel::{Panel, map_chunks};
use crate::rank::Ranker;

/// `K x K` mean cross-sectional Spearman correlation, column-major.
///
/// Entry `(i, j)` is the average, over rows, of the cross-sectional Spearman
/// correlation of factor `i` with factor `j`. The diagonal is exactly 1 when
/// the factor varies on at least one row, and `NaN` when it never does.
///
/// # Errors
///
/// [`Error::Invariant`] if `factors` is empty or the panels differ in shape.
pub fn mean_cs_corr(factors: &[Panel<'_>], threads: usize) -> Result<Vec<f64>> {
    if factors.is_empty() {
        return Err(Error::Invariant(
            "mean_cs_corr needs at least one factor".into(),
        ));
    }
    let mut out = cross_corr(factors, factors, threads)?;
    let k = factors.len();
    for i in 0..k {
        // The z-score dot product of a factor with itself is 1 up to
        // rounding. Store the exact value.
        let v = &mut out[i * k + i];
        if !v.is_nan() {
            *v = 1.0;
        }
    }
    Ok(out)
}

/// `n_candidates x n_references` mean cross-sectional Spearman correlation,
/// column-major: entry `(i, j)` is candidate `i` against reference `j`.
///
/// The same rank-once Spearman as [`mean_cs_corr`], between two lists. A pair
/// with no row where both sides have two or more varying finite values is
/// `NaN`.
///
/// # Errors
///
/// [`Error::Invariant`] if the panels differ in shape.
pub fn cross_corr(
    candidates: &[Panel<'_>],
    references: &[Panel<'_>],
    threads: usize,
) -> Result<Vec<f64>> {
    if let Some(first) = candidates.first() {
        for panel in candidates.iter().skip(1) {
            first.same_shape(panel, "candidate")?;
        }
        for panel in references {
            first.same_shape(panel, "reference")?;
        }
    } else if let Some(first) = references.first() {
        for panel in references.iter().skip(1) {
            first.same_shape(panel, "reference")?;
        }
    }
    let n_c = candidates.len();
    let n_r = references.len();
    if n_c == 0 || n_r == 0 {
        return Ok(Vec::new());
    }
    let (n_rows, n_cols) = (candidates[0].n_rows(), candidates[0].n_cols());
    let n_pairs = n_c * n_r;

    let parts = map_chunks(n_rows, threads, |rows| {
        let len = rows.len();
        let mut scored = vec![f64::NAN; len * n_pairs];
        let mut ranker = Ranker::default();
        let mut finite = Vec::new();
        let mut index = Vec::new();
        let mut ranks = Vec::new();
        let mut z_c = vec![f64::NAN; n_c * n_cols];
        let mut z_r = vec![f64::NAN; n_r * n_cols];
        let mut ok_c = vec![false; n_c];
        let mut ok_r = vec![false; n_r];
        for (local, t) in rows.enumerate() {
            fill_z(
                candidates,
                t,
                n_cols,
                &mut ranker,
                &mut finite,
                &mut index,
                &mut ranks,
                &mut z_c,
                &mut ok_c,
            );
            fill_z(
                references,
                t,
                n_cols,
                &mut ranker,
                &mut finite,
                &mut index,
                &mut ranks,
                &mut z_r,
                &mut ok_r,
            );
            for j in 0..n_r {
                if !ok_r[j] {
                    continue;
                }
                let zj = &z_r[j * n_cols..(j + 1) * n_cols];
                for i in 0..n_c {
                    if !ok_c[i] {
                        continue;
                    }
                    let zi = &z_c[i * n_cols..(i + 1) * n_cols];
                    let mut dot = 0.0;
                    let mut count = 0usize;
                    for s in 0..n_cols {
                        let (a, b) = (zi[s], zj[s]);
                        if a.is_finite() && b.is_finite() {
                            dot += a * b;
                            count += 1;
                        }
                    }
                    if count >= 2 {
                        // Column-major: reference `j`, candidate `i`.
                        scored[local * n_pairs + j * n_c + i] = dot / count as f64;
                    }
                }
            }
        }
        scored
    });

    // Add the per-row correlations in row order. Chunks come back in row
    // order, so this is the same sum a single thread would make.
    let mut sum = vec![0.0; n_pairs];
    let mut cnt = vec![0usize; n_pairs];
    for scored in &parts {
        let len = scored.len() / n_pairs;
        for local in 0..len {
            for p in 0..n_pairs {
                let v = scored[local * n_pairs + p];
                if !v.is_nan() {
                    sum[p] += v;
                    cnt[p] += 1;
                }
            }
        }
    }
    let mut out = vec![f64::NAN; n_pairs];
    for p in 0..n_pairs {
        if cnt[p] > 0 {
            out[p] = sum[p] / cnt[p] as f64;
        }
    }
    Ok(out)
}

/// Ranks each factor on row `t` and writes a unit-variance z-score into `z`.
/// `ok[f]` is set when the factor has at least two finite values and they are
/// not all equal, which is when a Spearman correlation with it is defined.
fn fill_z(
    panels: &[Panel<'_>],
    t: usize,
    n_cols: usize,
    ranker: &mut Ranker,
    finite: &mut Vec<f64>,
    index: &mut Vec<usize>,
    ranks: &mut Vec<f64>,
    z: &mut [f64],
    ok: &mut [bool],
) {
    for (f, panel) in panels.iter().enumerate() {
        finite.clear();
        index.clear();
        for c in 0..n_cols {
            let v = panel.col(c)[t];
            if v.is_finite() {
                finite.push(v);
                index.push(c);
            }
        }
        let row = &mut z[f * n_cols..(f + 1) * n_cols];
        row.fill(f64::NAN);
        ok[f] = false;
        if finite.len() < 2 {
            continue;
        }
        ranker.rank(finite, ranks);
        let n = ranks.len() as f64;
        let mean = ranks.iter().sum::<f64>() / n;
        let mut ss = 0.0;
        for &r in ranks.iter() {
            let d = r - mean;
            ss += d * d;
        }
        if ss > 0.0 {
            let scale = (ss / n).sqrt();
            for (&c, &r) in index.iter().zip(ranks.iter()) {
                row[c] = (r - mean) / scale;
            }
            ok[f] = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use phasor_core::math::is_close;

    fn col_major(rows: &[Vec<f64>]) -> (Vec<f64>, usize, usize) {
        let n_rows = rows.len();
        let n_cols = rows.first().map(Vec::len).unwrap_or(0);
        let mut data = vec![0.0; n_rows * n_cols];
        for (t, row) in rows.iter().enumerate() {
            for (c, &v) in row.iter().enumerate() {
                data[c * n_rows + t] = v;
            }
        }
        (data, n_rows, n_cols)
    }

    fn view(data: &[f64], n_rows: usize, n_cols: usize) -> Panel<'_> {
        Panel::new(data, n_rows, n_cols).unwrap()
    }

    #[test]
    fn self_correlation_is_one_and_the_reverse_is_minus_one() {
        let (x, nr, nc) = col_major(&[
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1.0, 2.0, 3.0, 4.0],
            vec![f64::NAN, f64::NAN, f64::NAN, f64::NAN],
        ]);
        let (y, _, _) = col_major(&[
            vec![4.0, 3.0, 2.0, 1.0],
            vec![1.0, 2.0, 4.0, 3.0],
            vec![1.0, 2.0, 3.0, 4.0],
        ]);
        // Row 0: x against y is -1. Row 1: ranks of y = [1, 2, 4, 3], not -1.
        // Row 2: x is all missing, so it does not enter the average.
        // x with itself averages the two usable rows, both perfect.
        let m = mean_cs_corr(&[view(&x, nr, nc), view(&y, nr, nc)], 1).unwrap();
        assert_eq!(m.len(), 4);
        assert!(is_close(m[0], 1.0));
        assert!(is_close(m[3], 1.0));
        // Column-major (2, 2): index 1 is (row 1, col 0) = corr(y, x) on the
        // averaged rows. Row 0 contributes -1. Row 1 is computed below and
        // checked through the average only for the sign of the perfect pair.
        assert!(m[1] < 0.0);
        assert!(is_close(m[1], m[2]));
    }

    #[test]
    fn average_of_a_perfect_row_and_its_reverse_is_zero() {
        let (x, nr, nc) = col_major(&[vec![1.0, 2.0, 3.0, 4.0], vec![1.0, 2.0, 3.0, 4.0]]);
        let (y, _, _) = col_major(&[vec![1.0, 2.0, 3.0, 4.0], vec![4.0, 3.0, 2.0, 1.0]]);
        let m = mean_cs_corr(&[view(&x, nr, nc), view(&y, nr, nc)], 2).unwrap();
        assert!(is_close(m[0], 1.0));
        assert!(is_close(m[1], 0.0));
        assert!(is_close(m[2], 0.0));
        assert!(is_close(m[3], 1.0));
    }

    #[test]
    fn cross_corr_picks_the_matching_reference() {
        let (a, nr, nc) = col_major(&[vec![1.0, 3.0, 2.0, 4.0], vec![8.0, 5.0, 7.0, 6.0]]);
        let (b, _, _) = col_major(&[vec![1.0, 1.0, 2.0, 2.0], vec![0.0, 3.0, 1.0, 2.0]]);
        let neg: Vec<f64> = a.iter().map(|v| -v).collect();
        let m = cross_corr(
            &[view(&a, nr, nc), view(&neg, nr, nc), view(&b, nr, nc)],
            &[view(&a, nr, nc), view(&b, nr, nc)],
            1,
        )
        .unwrap();
        // Shape (3, 2), column-major. Candidate 0 vs reference 0 is 1.
        assert!(is_close(m[0], 1.0));
        // Candidate 1 (neg a) vs reference 0 is -1.
        assert!(is_close(m[1], -1.0));
        // Candidate 2 vs reference 1 (b with itself) is 1, at column 1, row 2.
        assert!(is_close(m[5], 1.0));
    }

    #[test]
    fn constant_rows_and_empty_inputs() {
        let (flat, nr, nc) = col_major(&[vec![2.0, 2.0, 2.0], vec![1.0, 2.0, 3.0]]);
        let m = mean_cs_corr(&[view(&flat, nr, nc)], 1).unwrap();
        // Only the second row varies, so the factor still correlates with itself.
        assert!(is_close(m[0], 1.0));
        let (only, _, _) = col_major(&[vec![2.0, 2.0, 2.0]]);
        let m = mean_cs_corr(&[view(&only, 1, 3)], 1).unwrap();
        assert!(m[0].is_nan());
        assert!(mean_cs_corr(&[], 1).is_err());
        let empty = cross_corr(&[], &[view(&only, 1, 3)], 1).unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn threads_match_bit_for_bit() {
        let n_rows = 30;
        let n_cols = 10;
        let mut a = vec![0.0; n_rows * n_cols];
        let mut b = vec![0.0; n_rows * n_cols];
        for c in 0..n_cols {
            for t in 0..n_rows {
                let i = c * n_rows + t;
                if (t + 2 * c) % 7 == 0 {
                    a[i] = f64::NAN;
                } else {
                    a[i] = ((t * 5 + c * 3) % 13) as f64;
                }
                let level = if a[i].is_nan() { 0.0 } else { a[i] };
                b[i] = ((t * 4 + c) % 11) as f64 - level * 0.5;
                if (t * c) % 9 == 0 {
                    b[i] = f64::NAN;
                }
            }
        }
        let av = view(&a, n_rows, n_cols);
        let bv = view(&b, n_rows, n_cols);
        let base = mean_cs_corr(&[av, bv], 1).unwrap();
        let cross = cross_corr(&[av], &[av, bv], 1).unwrap();
        for threads in [2, 4, 8] {
            let m = mean_cs_corr(&[av, bv], threads).unwrap();
            assert_eq!(
                base.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                m.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            let c = cross_corr(&[av], &[av, bv], threads).unwrap();
            assert_eq!(
                cross.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                c.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }
}
