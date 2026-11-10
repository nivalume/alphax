//! Random signals: seeded, reproducible, and independent of evaluation order.
//!
//! # Why not a random stream
//!
//! A stream generator makes every draw depend on every draw before it, so the
//! answer changes with the order columns are visited in, the number of
//! threads, or a column added in front. Here every draw is a pure function of
//! `(seed, row, col)` — [`alphax_core::random::uniform`], SplitMix64 chained
//! over the three — so one column's signals are the same whether it is
//! generated alone or as the thousandth of a sweep, and a seed names one
//! answer.
//!
//! # What these are for
//!
//! A null model. A strategy whose entries beat [`random_n`] entries with the
//! same count has an entry rule; one that does not has a sample. The draws
//! are not cryptographic and are not meant to match NumPy's stream.

use alphax_core::error::{Error, Result};
use alphax_core::random::uniform;

/// An index drawn uniformly from `0..m`, for `m > 0`.
fn below(m: usize, seed: u64, row: usize, col: usize) -> usize {
    // The float product can only reach `m` through rounding; clamp it back.
    ((uniform(seed, row, col) * m as f64) as usize).min(m - 1)
}

/// Picks `n` distinct rows of `0..n_rows` for one column, uniformly.
///
/// A partial Fisher-Yates shuffle: step `k` swaps slot `k` with a slot drawn
/// from `k..n_rows`, so the first `n` slots are a uniform sample without
/// replacement. Step `k`'s draw is keyed by `(seed, k, col)`.
fn pick_rows(rows: &mut Vec<usize>, n_rows: usize, n: usize, seed: u64, col: usize) {
    rows.clear();
    rows.extend(0..n_rows);
    for k in 0..n {
        let j = k + below(n_rows - k, seed, k, col);
        rows.swap(k, j);
    }
    rows.truncate(n);
}

/// Fires each bar independently with probability `prob`.
///
/// Column-major, `n_rows * n_cols` long. The count per column is binomial,
/// not fixed; [`random_n`] is the one to use when it must match.
///
/// # Errors
///
/// Returns [`Error::Invariant`] if `prob` is outside `[0, 1]` or NaN.
pub fn random_by_prob(n_rows: usize, n_cols: usize, prob: f64, seed: u64) -> Result<Vec<bool>> {
    if !(0.0..=1.0).contains(&prob) {
        return Err(Error::Invariant(format!(
            "a signal probability must lie in [0, 1], got {prob}"
        )));
    }
    let mut out = vec![false; n_rows * n_cols];
    for col in 0..n_cols {
        let base = col * n_rows;
        for row in 0..n_rows {
            out[base + row] = uniform(seed, row, col) < prob;
        }
    }
    Ok(out)
}

/// Fires exactly `n` bars per column, at distinct rows chosen uniformly.
///
/// # Errors
///
/// Returns [`Error::Invariant`] if `n > n_rows`.
pub fn random_n(n_rows: usize, n_cols: usize, n: usize, seed: u64) -> Result<Vec<bool>> {
    if n > n_rows {
        return Err(Error::Invariant(format!(
            "cannot place {n} signals in a column of {n_rows} rows"
        )));
    }
    let mut out = vec![false; n_rows * n_cols];
    let mut rows = Vec::with_capacity(n_rows);
    for col in 0..n_cols {
        let base = col * n_rows;
        pick_rows(&mut rows, n_rows, n, seed, col);
        for &row in &rows {
            out[base + row] = true;
        }
    }
    Ok(out)
}

/// Places `n` entry/exit pairs per column, strictly alternating.
///
/// `2n` distinct rows are drawn uniformly, sorted, and assigned entry, exit,
/// entry, exit — so every entry has its exit before the next entry, and no
/// bar is both. Returns `(entries, exits)`, each column-major.
///
/// # Errors
///
/// Returns [`Error::Invariant`] if `2n > n_rows`.
pub fn random_entries_exits(
    n_rows: usize,
    n_cols: usize,
    n: usize,
    seed: u64,
) -> Result<(Vec<bool>, Vec<bool>)> {
    let picks = n.checked_mul(2).filter(|&p| p <= n_rows).ok_or_else(|| {
        Error::Invariant(format!(
            "cannot place {n} entry/exit pairs in a column of {n_rows} rows"
        ))
    })?;
    let mut entries = vec![false; n_rows * n_cols];
    let mut exits = vec![false; n_rows * n_cols];
    let mut rows = Vec::with_capacity(n_rows);
    for col in 0..n_cols {
        let base = col * n_rows;
        pick_rows(&mut rows, n_rows, picks, seed, col);
        rows.sort_unstable();
        for pair in rows.chunks_exact(2) {
            entries[base + pair[0]] = true;
            exits[base + pair[1]] = true;
        }
    }
    Ok((entries, exits))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn col(v: &[bool], n_rows: usize, c: usize) -> &[bool] {
        &v[c * n_rows..(c + 1) * n_rows]
    }

    /// The indices of the `true` cells.
    fn on(v: &[bool]) -> Vec<usize> {
        v.iter()
            .enumerate()
            .filter_map(|(i, &b)| b.then_some(i))
            .collect()
    }

    #[test]
    fn outputs_are_pinned() {
        // Recorded from the original implementation: a seed names one answer,
        // so these may never change.
        let pins = [
            ((0, 0, 0), 0x3fc1_c13a_de1c_7e5c),
            ((7, 3, 4), 0x3fdf_59a9_fade_4faa),
            ((7, 4, 3), 0x3fe6_b535_49c1_af99),
            ((42, 1000, 17), 0x3fc0_b892_0c42_3c8c),
            ((u64::MAX, usize::MAX, 99), 0x3fef_4880_6bee_efbf),
        ];
        for ((seed, row, col), want) in pins {
            assert_eq!(
                uniform(seed, row, col).to_bits(),
                want,
                "{seed} {row} {col}"
            );
        }
        assert_eq!(
            on(&random_by_prob(12, 3, 0.25, 42).unwrap()),
            [2, 3, 9, 19, 26, 31]
        );
        assert_eq!(
            on(&random_n(12, 3, 3, 42).unwrap()),
            [2, 4, 7, 16, 20, 21, 27, 34, 35]
        );
        let (entries, exits) = random_entries_exits(12, 3, 2, 42).unwrap();
        assert_eq!(on(&entries), [0, 4, 16, 20, 25, 34]);
        assert_eq!(on(&exits), [2, 7, 18, 21, 27, 35]);
    }

    #[test]
    fn uniform_stays_in_unit_interval() {
        for row in 0..1000 {
            let u = uniform(7, row, 3);
            assert!((0.0..1.0).contains(&u));
        }
    }

    #[test]
    fn same_seed_same_output_different_seed_differs() {
        let a = random_by_prob(200, 5, 0.3, 42).unwrap();
        assert_eq!(a, random_by_prob(200, 5, 0.3, 42).unwrap());
        assert_ne!(a, random_by_prob(200, 5, 0.3, 43).unwrap());

        let b = random_n(200, 5, 10, 42).unwrap();
        assert_eq!(b, random_n(200, 5, 10, 42).unwrap());
        assert_ne!(b, random_n(200, 5, 10, 43).unwrap());

        let c = random_entries_exits(200, 5, 10, 42).unwrap();
        assert_eq!(c, random_entries_exits(200, 5, 10, 42).unwrap());
        assert_ne!(c, random_entries_exits(200, 5, 10, 43).unwrap());
    }

    #[test]
    fn a_column_does_not_depend_on_its_neighbours() {
        // Column 2 of a 4-column draw equals column 2 of a 3-column draw:
        // draws are keyed by position, not by how many came before.
        let wide = random_n(50, 4, 7, 9).unwrap();
        let narrow = random_n(50, 3, 7, 9).unwrap();
        assert_eq!(col(&wide, 50, 2), col(&narrow, 50, 2));

        let wide = random_by_prob(50, 4, 0.5, 9).unwrap();
        let narrow = random_by_prob(50, 3, 0.5, 9).unwrap();
        assert_eq!(col(&wide, 50, 2), col(&narrow, 50, 2));
    }

    #[test]
    fn columns_differ() {
        let v = random_n(100, 2, 10, 1).unwrap();
        assert_ne!(col(&v, 100, 0), col(&v, 100, 1));
    }

    #[test]
    fn by_prob_rate_matches() {
        let (n_rows, n_cols) = (1000, 100);
        let v = random_by_prob(n_rows, n_cols, 0.3, 123).unwrap();
        let rate = v.iter().filter(|&&b| b).count() as f64 / v.len() as f64;
        // Standard error is ~0.0015 at 100k cells; 0.01 is ~7 of them.
        assert!((rate - 0.3).abs() < 0.01, "rate {rate}");
    }

    #[test]
    fn by_prob_edges() {
        assert!(random_by_prob(100, 3, 0.0, 5).unwrap().iter().all(|&b| !b));
        assert!(random_by_prob(100, 3, 1.0, 5).unwrap().iter().all(|&b| b));
    }

    #[test]
    fn random_n_exact_counts() {
        let (n_rows, n_cols) = (37, 20);
        for n in [0, 1, 5, 36, 37] {
            let v = random_n(n_rows, n_cols, n, 77).unwrap();
            assert_eq!(v.len(), n_rows * n_cols);
            for c in 0..n_cols {
                assert_eq!(col(&v, n_rows, c).iter().filter(|&&b| b).count(), n);
            }
        }
    }

    #[test]
    fn random_n_is_roughly_uniform_over_rows() {
        let (n_rows, n_cols, n) = (10, 20_000, 3);
        let v = random_n(n_rows, n_cols, n, 2024).unwrap();
        for row in 0..n_rows {
            let hits = (0..n_cols).filter(|&c| v[c * n_rows + row]).count();
            let rate = hits as f64 / n_cols as f64;
            assert!((rate - 0.3).abs() < 0.02, "row {row} rate {rate}");
        }
    }

    #[test]
    fn entries_exits_alternate() {
        let (n_rows, n_cols) = (60, 50);
        for n in [0, 1, 4, 30] {
            let (en, ex) = random_entries_exits(n_rows, n_cols, n, 314).unwrap();
            assert_eq!(en.len(), n_rows * n_cols);
            assert_eq!(ex.len(), n_rows * n_cols);
            for c in 0..n_cols {
                let (en, ex) = (col(&en, n_rows, c), col(&ex, n_rows, c));
                assert_eq!(en.iter().filter(|&&b| b).count(), n);
                assert_eq!(ex.iter().filter(|&&b| b).count(), n);
                // Walk the column: entry, exit, entry, exit, never both.
                let mut in_pos = false;
                for row in 0..n_rows {
                    assert!(!(en[row] && ex[row]), "col {c} row {row} is both");
                    if en[row] {
                        assert!(!in_pos, "col {c} row {row}: entry while in position");
                        in_pos = true;
                    }
                    if ex[row] {
                        assert!(in_pos, "col {c} row {row}: exit without entry");
                        in_pos = false;
                    }
                }
                assert!(!in_pos, "col {c} ends with an open entry");
            }
        }
    }

    #[test]
    fn argument_errors() {
        for p in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                random_by_prob(10, 2, p, 0),
                Err(Error::Invariant(_))
            ));
        }
        assert!(matches!(random_n(10, 2, 11, 0), Err(Error::Invariant(_))));
        assert!(matches!(
            random_entries_exits(10, 2, 6, 0),
            Err(Error::Invariant(_))
        ));
        assert!(matches!(
            random_entries_exits(10, 2, usize::MAX, 0),
            Err(Error::Invariant(_))
        ));
        assert!(random_entries_exits(10, 2, 5, 0).is_ok());
    }
}
