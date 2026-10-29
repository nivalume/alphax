//! Seeded placebo tests for a rank IC.
//!
//! A t-statistic asks whether the mean IC is far from zero *if the IC series
//! were a sample from a normal distribution*. Overlapping forward returns and
//! a persistent factor make that a poor description, which is why the summary
//! uses Newey–West — and still assumes a shape. A placebo does not. It builds
//! the null by destroying the pairing the IC measures and asks how often that
//! null is at least as large as what was observed.
//!
//! Two destructions, both of the forward return, never of the factor:
//!
//! - [`cs_permutation`] shuffles the symbols' forward returns on each bar.
//!   The cross-section of returns is unchanged; which symbol earned which
//!   return is not. This is the null for a cross-sectional rank IC.
//! - [`ts_circular_shift`] rotates the whole forward-return panel along time
//!   by an offset far from zero. A symbol keeps its own returns, in order,
//!   but they no longer line up with its factor. Offsets near zero (and near
//!   a full turn, which is the same shift) would leave the pairing almost
//!   intact, so they are not drawn.
//!
//! Both score the Spearman rank IC, the series [`crate::ic::cs_ic`] reports
//! as rank IC and the one [`crate::summary::summarize`] describes by default.
//! The p-value is `(1 + #{|null| ≥ |observed|}) / (1 + draws)`, which counts
//! the observed result as one more draw under the null (Phipson & Smyth).
//!
//! Every draw is a pure function of `(seed, draw, row)` through
//! [`phasor_core::random::bits`], with a salt that counts the integers one
//! row needs. A row does not depend on any other row's draws, and a draw
//! does not depend on any other draw, so splitting the draws across threads
//! and summing each draw's rows in row order (its symbols in symbol order)
//! is bit-identical to doing it on one thread.

use phasor_core::error::{Error, Result};
use phasor_core::random::bits;

use crate::corr::Corr;
use crate::ic::Method;
use crate::panel::{Panel, map_chunks};

/// The observed rank IC, the share of placebo draws that match it, and those draws.
#[derive(Debug, Clone, PartialEq)]
pub struct PlaceboResult {
    /// Mean rank IC of the factor against the forward return it was given.
    ///
    /// The mean skips bars (cross-section) or symbols (time series) whose
    /// rank IC is `NaN`. `NaN` when none remain.
    pub observed: f64,
    /// `(1 + #{|null| ≥ |observed|}) / (1 + draws)`.
    ///
    /// `NaN` when `observed` is `NaN`: there is nothing to compare the null to.
    /// A `NaN` null draw does not count, because the comparison is false.
    pub p_value: f64,
    /// One mean null rank IC per draw, in draw order. Length `draws`.
    pub null_distribution: Vec<f64>,
}

/// Cross-sectional placebo: permute each bar's forward returns, then average
/// the per-bar Spearman IC.
///
/// On draw `d` and bar `t` the forward returns of that bar are shuffled by a
/// Fisher–Yates permutation whose random integers are
/// [`bits`]`(seed, d, t, salt)`. The factor is not moved. The draw's null IC
/// is the mean of the finite per-bar rank ICs, in bar order. `min_names` is
/// the smallest number of finite pairs a bar needs, as in [`crate::ic::cs_ic`].
///
/// `threads` 0 uses every core. Draws are split into contiguous ranges; the
/// values do not depend on how many threads that is.
///
/// # Errors
///
/// [`Error::Invariant`] if the panels' shapes differ.
pub fn cs_permutation(
    factor: &Panel<'_>,
    forward: &Panel<'_>,
    draws: usize,
    seed: u64,
    min_names: usize,
    threads: usize,
) -> Result<PlaceboResult> {
    let (n_rows, n_cols) = shapes(factor, forward)?;
    let mut factor_rows = Vec::new();
    let mut forward_rows = Vec::new();
    factor.gather_rows(0..n_rows, &mut factor_rows);
    forward.gather_rows(0..n_rows, &mut forward_rows);
    let min_names = min_names.max(2);

    let observed = {
        let mut corr = Corr::default();
        mean_row_ic(
            &factor_rows,
            &forward_rows,
            n_rows,
            n_cols,
            min_names,
            &mut corr,
        )
    };
    let parts = map_chunks(draws, threads, |range| {
        let mut scratch = CsScratch {
            shuffled: vec![0.0; n_cols],
            ..CsScratch::default()
        };
        let mut out = Vec::with_capacity(range.len());
        for draw in range {
            out.push(permuted_mean(
                seed,
                draw,
                &factor_rows,
                &forward_rows,
                n_rows,
                n_cols,
                min_names,
                &mut scratch,
            ));
        }
        out
    });
    Ok(finish(observed, parts.concat()))
}

/// Time-series placebo: circularly shift the forward return along time, then
/// average the per-symbol Spearman IC.
///
/// Draw `d` rotates every symbol's forward return by the same offset `k`,
/// drawn uniformly from `[n_rows / 10, n_rows - n_rows / 10]` and reduced
/// modulo `n_rows` so a full turn is not a second copy of the unshifted
/// panel. `shifted[t] = forward[(t - k) mod n_rows]`, NumPy's `roll` along
/// the time axis. The null IC is the mean of the finite per-symbol rank ICs,
/// in symbol order. `min_periods` is the smallest number of finite pairs a
/// symbol needs, as in [`crate::ic::ts_ic`].
///
/// `threads` 0 uses every core. The values do not depend on how many.
///
/// # Errors
///
/// [`Error::Invariant`] if the panels' shapes differ, or the panel has fewer
/// than two rows (no rotation exists).
pub fn ts_circular_shift(
    factor: &Panel<'_>,
    forward: &Panel<'_>,
    draws: usize,
    seed: u64,
    min_periods: usize,
    threads: usize,
) -> Result<PlaceboResult> {
    let (n_rows, _) = shapes(factor, forward)?;
    let (lo, hi) = shift_bounds(n_rows)?;
    let min_periods = min_periods.max(2);

    let observed = {
        let mut corr = Corr::default();
        mean_symbol_ic(factor, forward, 0, min_periods, &mut corr)
    };
    let parts = map_chunks(draws, threads, |range| {
        let mut corr = Corr::default();
        let mut out = Vec::with_capacity(range.len());
        for draw in range {
            let k = lo + index_below(seed, draw, 0, hi - lo + 1);
            out.push(mean_symbol_ic(factor, forward, k, min_periods, &mut corr));
        }
        out
    });
    Ok(finish(observed, parts.concat()))
}

fn shapes(factor: &Panel<'_>, forward: &Panel<'_>) -> Result<(usize, usize)> {
    factor.same_shape(forward, "forward return")?;
    Ok((factor.n_rows(), factor.n_cols()))
}

/// `(1 + extremes) / (1 + draws)`, or `NaN` when there is no observed IC.
fn finish(observed: f64, null_distribution: Vec<f64>) -> PlaceboResult {
    let p_value = if observed.is_finite() {
        let mag = observed.abs();
        let extremes = null_distribution
            .iter()
            .filter(|value| value.abs() >= mag)
            .count();
        (1.0 + extremes as f64) / (1.0 + null_distribution.len() as f64)
    } else {
        f64::NAN
    };
    PlaceboResult {
        observed,
        p_value,
        null_distribution,
    }
}

fn mean_of(sum: f64, n: usize) -> f64 {
    if n == 0 { f64::NAN } else { sum / n as f64 }
}

/// Mean of the finite per-bar rank ICs, bars in order. `NaN` when none are finite.
fn mean_row_ic(
    factor_rows: &[f64],
    forward_rows: &[f64],
    n_rows: usize,
    n_cols: usize,
    min_names: usize,
    corr: &mut Corr,
) -> f64 {
    let mut sum = 0.0;
    let mut n = 0usize;
    for t in 0..n_rows {
        let ic = row_rank_ic(
            row(factor_rows, n_cols, t),
            row(forward_rows, n_cols, t),
            min_names,
            corr,
        );
        if ic.is_finite() {
            sum += ic;
            n += 1;
        }
    }
    mean_of(sum, n)
}

fn row(panel: &[f64], n_cols: usize, t: usize) -> &[f64] {
    let start = t * n_cols;
    &panel[start..start + n_cols]
}

/// Spearman of one bar. `NaN` below `min_names` finite pairs or for a constant side.
fn row_rank_ic(x: &[f64], y: &[f64], min_names: usize, corr: &mut Corr) -> f64 {
    corr.clear();
    for (&a, &b) in x.iter().zip(y) {
        corr.push(a, b);
    }
    corr.compute(Method::Spearman, min_names)
}

#[derive(Debug, Default)]
struct CsScratch {
    corr: Corr,
    perm: Vec<usize>,
    shuffled: Vec<f64>,
}

fn permuted_mean(
    seed: u64,
    draw: usize,
    factor_rows: &[f64],
    forward_rows: &[f64],
    n_rows: usize,
    n_cols: usize,
    min_names: usize,
    scratch: &mut CsScratch,
) -> f64 {
    let mut sum = 0.0;
    let mut n = 0usize;
    for t in 0..n_rows {
        fisher_yates(seed, draw, t, n_cols, &mut scratch.perm);
        let src = row(forward_rows, n_cols, t);
        scratch.shuffled.clear();
        scratch
            .shuffled
            .extend(scratch.perm.iter().map(|&j| src[j]));
        let ic = row_rank_ic(
            row(factor_rows, n_cols, t),
            &scratch.shuffled,
            min_names,
            &mut scratch.corr,
        );
        if ic.is_finite() {
            sum += ic;
            n += 1;
        }
    }
    mean_of(sum, n)
}

/// Fisher–Yates into `perm`, a uniform shuffle of `0..n`.
///
/// The integer that swaps position `i` is the next accepted draw of
/// [`bits`]`(seed, draw, t, salt)`. `salt` starts at 0 on every bar and counts
/// rejected draws as well as accepted ones, so the whole permutation is fixed
/// by `(seed, draw, t)`.
fn fisher_yates(seed: u64, draw: usize, t: usize, n: usize, perm: &mut Vec<usize>) {
    perm.clear();
    perm.extend(0..n);
    if n < 2 {
        return;
    }
    let mut salt = 0u64;
    for i in (1..n).rev() {
        let j = index_below_salted(seed, draw, t, &mut salt, i + 1);
        perm.swap(i, j);
    }
}

/// Inclusive `[lo, hi]` of admissible circular-shift offsets, each in `0..n_rows`.
fn shift_bounds(n_rows: usize) -> Result<(usize, usize)> {
    if n_rows < 2 {
        return Err(Error::Invariant(
            "circular shift needs at least 2 rows".into(),
        ));
    }
    let lo = n_rows / 10;
    // `n_rows` is the same rotation as 0, so the top of the spec's closed
    // interval is folded back into `0..n_rows` before the draw.
    let hi = (n_rows - lo).min(n_rows - 1);
    Ok((lo, hi))
}

/// One uniform integer in `0..n` from [`bits`]`(seed, row, col, salt)`, salt starting at 0.
fn index_below(seed: u64, row: usize, col: usize, n: usize) -> usize {
    let mut salt = 0u64;
    index_below_salted(seed, row, col, &mut salt, n)
}

/// Unbiased integer in `0..n`. Advances `salt` once per candidate draw.
///
/// `draw % n` over-represents the residues below `2^64 % n`. Those candidates
/// are rejected and the next salt is tried, which keeps the permutation
/// uniform and still a function of the key alone.
fn index_below_salted(seed: u64, row: usize, col: usize, salt: &mut u64, n: usize) -> usize {
    debug_assert!(n >= 1);
    if n == 1 {
        return 0;
    }
    let n_u = n as u64;
    let threshold = n_u.wrapping_neg() % n_u;
    loop {
        let draw = bits(seed, row, col, *salt);
        *salt = salt.wrapping_add(1);
        if draw >= threshold {
            return (draw % n_u) as usize;
        }
    }
}

/// Mean rank IC across symbols after rotating `forward` by `k` bars.
///
/// `k == 0` is the observed series. Symbols are accumulated in column order.
fn mean_symbol_ic(
    factor: &Panel<'_>,
    forward: &Panel<'_>,
    k: usize,
    min_periods: usize,
    corr: &mut Corr,
) -> f64 {
    let n_rows = factor.n_rows();
    let mut sum = 0.0;
    let mut n = 0usize;
    for c in 0..factor.n_cols() {
        let x = factor.col(c);
        let y = forward.col(c);
        corr.clear();
        for (t, &xv) in x.iter().enumerate() {
            let src = (t + n_rows - k) % n_rows;
            corr.push(xv, y[src]);
        }
        let ic = corr.compute(Method::Spearman, min_periods);
        if ic.is_finite() {
            sum += ic;
            n += 1;
        }
    }
    mean_of(sum, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ic::{cs_ic, ts_ic};
    use phasor_core::math::is_close;

    fn col_major(n_rows: usize, n_cols: usize, value: impl Fn(usize, usize) -> f64) -> Vec<f64> {
        let mut data = vec![0.0; n_rows * n_cols];
        for c in 0..n_cols {
            for r in 0..n_rows {
                data[c * n_rows + r] = value(r, c);
            }
        }
        data
    }

    fn bits_of(values: &[f64]) -> Vec<u64> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    #[test]
    fn a_perfect_cross_section_is_not_a_permutation() {
        // The same ranking on every bar. A shuffle of 5 distinct values is
        // exactly ±1 only when the whole bar is untouched or reversed, and
        // the mean reaches ±1 only when every bar does that together.
        let data = col_major(8, 5, |_r, c| c as f64);
        let panel = Panel::new(&data, 8, 5).unwrap();
        let result = cs_permutation(&panel, &panel, 20, 1, 2, 1).unwrap();
        assert!(is_close(result.observed, 1.0));
        assert_eq!(result.null_distribution.len(), 20);
        assert!(result.null_distribution.iter().all(|v| v.abs() < 1.0));
        assert_eq!(result.p_value.to_bits(), (1.0_f64 / 21.0).to_bits());
    }

    #[test]
    fn the_p_value_is_the_add_one_formula() {
        let x = col_major(15, 6, |r, c| ((r * 3 + c * 5) % 17) as f64);
        let y = col_major(15, 6, |r, c| ((r * 2 + c) % 9) as f64 - 3.0);
        let (px, py) = (
            Panel::new(&x, 15, 6).unwrap(),
            Panel::new(&y, 15, 6).unwrap(),
        );
        let result = cs_permutation(&px, &py, 12, 9, 3, 1).unwrap();
        let mag = result.observed.abs();
        let extremes = result
            .null_distribution
            .iter()
            .filter(|v| v.abs() >= mag)
            .count();
        assert_eq!(
            result.p_value.to_bits(),
            ((1.0 + extremes as f64) / 13.0).to_bits()
        );
    }

    #[test]
    fn observed_rank_ic_matches_the_cross_sectional_kernel() {
        let x = col_major(12, 7, |r, c| {
            if (r + c) % 5 == 0 {
                f64::NAN
            } else {
                (r + 1) as f64 * (c as f64 - 2.0)
            }
        });
        let y = col_major(12, 7, |r, c| {
            if (r * 2 + c) % 6 == 0 {
                f64::NAN
            } else {
                (c * c) as f64 - r as f64
            }
        });
        let (px, py) = (
            Panel::new(&x, 12, 7).unwrap(),
            Panel::new(&y, 12, 7).unwrap(),
        );
        let result = cs_permutation(&px, &py, 4, 2, 3, 1).unwrap();
        let ic = cs_ic(&[px], &[py], 3, 1).unwrap();
        let mut sum = 0.0;
        let mut n = 0usize;
        for &v in ic.series(Method::Spearman, 0, 0) {
            if v.is_finite() {
                sum += v;
                n += 1;
            }
        }
        assert_eq!(result.observed.to_bits(), (sum / n as f64).to_bits());
    }

    #[test]
    fn no_draws_leaves_the_observed_and_an_empty_null() {
        let data = col_major(4, 4, |_r, c| c as f64);
        let panel = Panel::new(&data, 4, 4).unwrap();
        let result = cs_permutation(&panel, &panel, 0, 0, 2, 1).unwrap();
        assert!(result.null_distribution.is_empty());
        assert!(is_close(result.observed, 1.0));
        assert!(is_close(result.p_value, 1.0));
    }

    #[test]
    fn a_constant_side_has_no_p_value() {
        let x = col_major(6, 4, |r, c| (r + c) as f64);
        let y = col_major(6, 4, |_r, _c| 1.0);
        let (px, py) = (Panel::new(&x, 6, 4).unwrap(), Panel::new(&y, 6, 4).unwrap());
        let result = cs_permutation(&px, &py, 5, 0, 2, 1).unwrap();
        assert!(result.observed.is_nan());
        assert!(result.p_value.is_nan());
        assert_eq!(result.null_distribution.len(), 5);
    }

    #[test]
    fn permutation_threads_are_bit_identical() {
        let x = col_major(24, 9, |r, c| {
            if (r * 3 + c) % 11 == 0 {
                f64::NAN
            } else {
                (r * 5 + c * 7) as f64
            }
        });
        let y = col_major(24, 9, |r, c| {
            if (r + c * 2) % 8 == 0 {
                f64::NAN
            } else {
                (c as f64) * 0.5 - r as f64
            }
        });
        let (px, py) = (
            Panel::new(&x, 24, 9).unwrap(),
            Panel::new(&y, 24, 9).unwrap(),
        );
        let base = cs_permutation(&px, &py, 16, 42, 4, 1).unwrap();
        for threads in [2, 3, 4, 8] {
            let other = cs_permutation(&px, &py, 16, 42, 4, threads).unwrap();
            assert_eq!(
                base.observed.to_bits(),
                other.observed.to_bits(),
                "{threads}"
            );
            assert_eq!(base.p_value.to_bits(), other.p_value.to_bits(), "{threads}");
            assert_eq!(
                bits_of(&base.null_distribution),
                bits_of(&other.null_distribution),
                "{threads}"
            );
        }
    }

    #[test]
    fn a_different_seed_is_a_different_null() {
        let data = col_major(10, 6, |r, c| (r + c) as f64);
        let panel = Panel::new(&data, 10, 6).unwrap();
        let a = cs_permutation(&panel, &panel, 3, 1, 2, 1).unwrap();
        let b = cs_permutation(&panel, &panel, 3, 2, 2, 1).unwrap();
        assert_ne!(bits_of(&a.null_distribution), bits_of(&b.null_distribution));
    }

    #[test]
    fn mismatched_shapes_are_rejected() {
        let a = col_major(4, 3, |r, c| (r + c) as f64);
        let b = col_major(5, 3, |r, c| (r + c) as f64);
        let (pa, pb) = (Panel::new(&a, 4, 3).unwrap(), Panel::new(&b, 5, 3).unwrap());
        assert!(cs_permutation(&pa, &pb, 1, 0, 2, 1).is_err());
        assert!(ts_circular_shift(&pa, &pb, 1, 0, 2, 1).is_err());
    }

    #[test]
    fn fisher_yates_is_a_permutation_and_depends_on_the_key() {
        let mut perm = Vec::new();
        fisher_yates(3, 4, 5, 8, &mut perm);
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..8).collect::<Vec<_>>());

        let mut again = Vec::new();
        fisher_yates(3, 4, 5, 8, &mut again);
        assert_eq!(perm, again);
        fisher_yates(3, 4, 6, 8, &mut again);
        assert_ne!(perm, again);
    }

    #[test]
    fn the_first_position_of_a_shuffle_is_spread_out() {
        let mut counts = [0usize; 8];
        let mut perm = Vec::new();
        for draw in 0..4000 {
            fisher_yates(1, draw, 0, 8, &mut perm);
            counts[perm[0]] += 1;
        }
        for count in counts {
            assert!((350..650).contains(&count), "{counts:?}");
        }
    }

    #[test]
    fn one_circular_shift_matches_a_rolled_panel() {
        let n_rows = 20;
        let n_cols = 3;
        let x = col_major(n_rows, n_cols, |r, c| (r * (c + 1)) as f64);
        let y = col_major(n_rows, n_cols, |r, c| ((r + 3) * (c + 2)) as f64 + 0.5);
        let (px, py) = (
            Panel::new(&x, n_rows, n_cols).unwrap(),
            Panel::new(&y, n_rows, n_cols).unwrap(),
        );
        let result = ts_circular_shift(&px, &py, 1, 11, 5, 1).unwrap();
        let (lo, hi) = shift_bounds(n_rows).unwrap();
        assert_eq!((lo, hi), (2, 18));
        let k = lo + index_below(11, 0, 0, hi - lo + 1);
        assert!((lo..=hi).contains(&k));

        let mut rolled = y.clone();
        for c in 0..n_cols {
            for t in 0..n_rows {
                rolled[c * n_rows + t] = y[c * n_rows + (t + n_rows - k) % n_rows];
            }
        }
        let rolled_panel = Panel::new(&rolled, n_rows, n_cols).unwrap();
        let manual = ts_ic(&px, &rolled_panel, Method::Spearman, 5, 1).unwrap();
        let mut sum = 0.0;
        let mut n = 0usize;
        for &v in &manual {
            if v.is_finite() {
                sum += v;
                n += 1;
            }
        }
        assert_eq!(
            result.null_distribution[0].to_bits(),
            (sum / n as f64).to_bits()
        );

        let plain = ts_ic(&px, &py, Method::Spearman, 5, 1).unwrap();
        let mut sum = 0.0;
        let mut n = 0usize;
        for &v in &plain {
            if v.is_finite() {
                sum += v;
                n += 1;
            }
        }
        assert_eq!(result.observed.to_bits(), (sum / n as f64).to_bits());
        assert!(is_close(result.observed, 1.0));
    }

    #[test]
    fn shift_offsets_stay_away_from_a_full_turn() {
        let n_rows = 40;
        let data = col_major(n_rows, 2, |r, c| (r + c) as f64);
        let panel = Panel::new(&data, n_rows, 2).unwrap();
        let result = ts_circular_shift(&panel, &panel, 30, 5, 5, 1).unwrap();
        assert_eq!(result.null_distribution.len(), 30);
        let (lo, hi) = shift_bounds(n_rows).unwrap();
        assert_eq!((lo, hi), (4, 36));
        for draw in 0..30 {
            let k = lo + index_below(5, draw, 0, hi - lo + 1);
            assert!((4..=36).contains(&k), "{k}");
        }
        assert!(result.p_value.is_finite());
    }

    #[test]
    fn a_single_row_cannot_be_shifted() {
        let data = col_major(1, 4, |_r, c| c as f64);
        let panel = Panel::new(&data, 1, 4).unwrap();
        assert!(ts_circular_shift(&panel, &panel, 2, 0, 2, 1).is_err());
    }

    #[test]
    fn circular_shift_threads_are_bit_identical() {
        let x = col_major(30, 6, |r, c| {
            if r % 9 == 0 {
                f64::NAN
            } else {
                (r * c + 1) as f64
            }
        });
        let y = col_major(30, 6, |r, c| (r as f64).sin() + c as f64);
        let (px, py) = (
            Panel::new(&x, 30, 6).unwrap(),
            Panel::new(&y, 30, 6).unwrap(),
        );
        let base = ts_circular_shift(&px, &py, 12, 8, 6, 1).unwrap();
        for threads in [2, 4, 8] {
            let other = ts_circular_shift(&px, &py, 12, 8, 6, threads).unwrap();
            assert_eq!(
                base.observed.to_bits(),
                other.observed.to_bits(),
                "{threads}"
            );
            assert_eq!(base.p_value.to_bits(), other.p_value.to_bits(), "{threads}");
            assert_eq!(
                bits_of(&base.null_distribution),
                bits_of(&other.null_distribution),
                "{threads}"
            );
        }
    }
}
