//! Overfitting and robustness statistics.
//!
//! A sweep that tries a thousand parameter sets and reports the best Sharpe
//! has not measured a strategy, it has measured the maximum of a thousand
//! noisy estimates. The functions here put that number back in context,
//! following Bailey & López de Prado:
//!
//! - [`probabilistic_sharpe`] — the probability that the true Sharpe ratio
//!   exceeds a benchmark, given the sample length and the skew and fat tails
//!   of the returns.
//! - [`expected_max_sharpe`] — the Sharpe ratio the best of `N` worthless
//!   trials would be expected to show by luck alone.
//! - [`deflated_sharpe`] — the probabilistic Sharpe ratio measured against
//!   that luck-only maximum instead of against zero.
//! - [`pbo_cscv`] — the probability that the in-sample winner of a sweep is
//!   below the median out of sample, by combinatorially symmetric
//!   cross-validation.
//!
//! # Sharpe ratios here are per period
//!
//! Every Sharpe ratio in this module is the raw per-bar ratio — mean over
//! sample standard deviation, never annualized. The formulas are stated in
//! those units, and the trial Sharpe ratios fed to [`deflated_sharpe`] must
//! be in the same units as the returns it is given.
//!
//! # `NaN` is a missing bar, not an error
//!
//! As in [`crate::metrics`], every reduction skips `NaN`s. Degenerate inputs
//! — fewer than two bars, a column that never moved — give `NaN` rather than
//! a confident-looking probability.

use alphax_core::error::{Error, Result};

use crate::metrics::{nanmean, nanstd};

/// The Euler–Mascheroni constant, γ.
const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_9;

/// Largest number of blocks [`pbo_cscv`] accepts.
///
/// `C(16, 8) = 12_870` train/test splits, each costing `O(n_cols *
/// n_splits)` once per-block sums are precomputed. Beyond this the number of
/// splits grows by roughly 4x per extra pair of blocks while the estimate
/// stops improving, so a larger request is refused rather than silently
/// taking minutes.
pub const MAX_PBO_SPLITS: usize = 16;

/// Complementary error function, `erfc(z) = 1 - erf(z)`.
///
/// Two regimes, both without cancellation in the tail that matters:
///
/// - `|z| < 2.5`: the all-positive series
///   `erf(z) = 2/√π · e^{-z²} · Σ 2ⁿ z^{2n+1} / (2n+1)!!`, then `1 - erf`.
///   `erfc(2.5) ≈ 4e-4`, so the subtraction costs at most ~3.5 decimal
///   digits, leaving a relative error near `1e-13`.
/// - `|z| ≥ 2.5`: the Laplace continued fraction
///   `erfc(z) = e^{-z²}/√π · 1/(z + ½/(z + 1/(z + 3⁄2/(z + …))))`, evaluated
///   bottom-up over a fixed 100 terms, which is converged to machine
///   precision from `z = 2.5` outward.
///
/// Negative arguments use `erfc(-z) = 2 - erfc(z)`.
fn erfc(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    if z < 0.0 {
        return 2.0 - erfc(-z);
    }
    let two_over_sqrt_pi = std::f64::consts::FRAC_2_SQRT_PI;
    if z < 2.5 {
        let z2 = z * z;
        let mut term = z;
        let mut sum = z;
        let mut n = 1.0;
        while term > sum * 1e-17 {
            term *= 2.0 * z2 / (2.0 * n + 1.0);
            sum += term;
            n += 1.0;
        }
        1.0 - two_over_sqrt_pi * (-z2).exp() * sum
    } else {
        let mut t = z;
        for n in (1..=100).rev() {
            t = z + (f64::from(n) * 0.5) / t;
        }
        // e^{-z²}/√π = (2/√π)/2 · e^{-z²}.
        0.5 * two_over_sqrt_pi * (-z * z).exp() / t
    }
}

/// Standard normal cumulative distribution function, Φ(x).
///
/// Computed as `½ erfc(-x/√2)`, so the lower tail keeps its relative
/// accuracy (about `1e-13` or better) all the way to underflow rather than
/// bottoming out at `1e-16` absolute. `NaN` in, `NaN` out.
#[must_use]
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// Inverse of [`norm_cdf`], Φ⁻¹(p).
///
/// Acklam's rational approximation (relative error below `1.15e-9`) followed
/// by one Halley step against [`norm_cdf`], which brings the result to
/// within a few ulps of the exact quantile — the round trip
/// `norm_cdf(norm_ppf(p))` agrees with `p` to about `1e-13` relative.
///
/// The work is always done in the lower half: `p > ½` is answered as
/// `-Φ⁻¹(1 - p)`, where `1 - p` is exact, so the refinement compares small
/// tail probabilities instead of numbers near one.
///
/// `p = 0` gives `-∞`, `p = 1` gives `+∞`, and anything outside `[0, 1]` or
/// `NaN` gives `NaN`.
#[must_use]
pub fn norm_ppf(p: f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p > 0.5 {
        return -norm_ppf_lower(1.0 - p);
    }
    norm_ppf_lower(p)
}

/// [`norm_ppf`] for `0 < p <= ½`.
fn norm_ppf_lower(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const P_LOW: f64 = 0.024_25;

    let x = if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    };

    // One Halley step: e is the residual in probability, u the Newton step.
    let e = norm_cdf(x) - p;
    let u = e * (2.0 * std::f64::consts::PI).sqrt() * (0.5 * x * x).exp();
    x - u / (1.0 + 0.5 * x * u)
}

/// Probabilistic Sharpe ratio: `P(true SR > benchmark_sr)`.
///
/// ```text
/// PSR = Φ( (SR - SR*) · √(n - 1) / √(1 - γ₃·SR + (γ₄ - 1)/4 · SR²) )
/// ```
///
/// `SR` is the per-period Sharpe ratio (mean over the `ddof = 1` standard
/// deviation, no risk-free rate, not annualized), `n` the number of
/// non-`NaN` bars, `γ₃` the sample skewness and `γ₄` the sample kurtosis —
/// **not** excess kurtosis, so a normal distribution has `γ₄ = 3`. The two
/// moment ratios use the biased (population) central moments, as in Bailey &
/// López de Prado.
///
/// `benchmark_sr` must be per period too. Returns `NaN` for fewer than two
/// bars, a column with zero variance, or the (pathological) case where the
/// variance term under the square root is not positive.
#[must_use]
pub fn probabilistic_sharpe(returns: &[f64], benchmark_sr: f64) -> f64 {
    let n = returns.iter().filter(|v| !v.is_nan()).count();
    if n < 2 {
        return f64::NAN;
    }
    let mean = nanmean(returns);
    let std = nanstd(returns, 1);
    if std.is_nan() || std <= 0.0 {
        return f64::NAN;
    }
    let sr = mean / std;

    let (mut m2, mut m3, mut m4) = (0.0, 0.0, 0.0);
    for &v in returns {
        if !v.is_nan() {
            let d = v - mean;
            let d2 = d * d;
            m2 += d2;
            m3 += d2 * d;
            m4 += d2 * d2;
        }
    }
    let nf = n as f64;
    m2 /= nf;
    m3 /= nf;
    m4 /= nf;
    let skew = m3 / m2.powf(1.5);
    let kurt = m4 / (m2 * m2);

    let denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr;
    if denom.is_nan() || denom <= 0.0 {
        return f64::NAN;
    }
    norm_cdf((sr - benchmark_sr) * (nf - 1.0).sqrt() / denom.sqrt())
}

/// Expected maximum Sharpe ratio of `n_trials` independent trials whose true
/// Sharpe ratio is zero.
///
/// ```text
/// E[max SR] ≈ √V · ( (1 - γ) Φ⁻¹(1 - 1/N) + γ Φ⁻¹(1 - 1/(N·e)) )
/// ```
///
/// with `γ` the Euler–Mascheroni constant and `V` the variance of the trial
/// Sharpe ratios across the sweep. The approximation is asymptotic in `N`;
/// it is used as-is for any `N ≥ 2`.
///
/// One trial has nothing to be lucky against, so `n_trials = 1` gives `0`.
/// `n_trials = 0` and a negative or `NaN` variance give `NaN`.
#[must_use]
pub fn expected_max_sharpe(n_trials: usize, sr_variance: f64) -> f64 {
    if n_trials == 0 || sr_variance.is_nan() || sr_variance < 0.0 {
        return f64::NAN;
    }
    if n_trials == 1 {
        return 0.0;
    }
    let n = n_trials as f64;
    let g = EULER_MASCHERONI;
    sr_variance.sqrt()
        * ((1.0 - g) * norm_ppf(1.0 - 1.0 / n)
            + g * norm_ppf(1.0 - 1.0 / (n * std::f64::consts::E)))
}

/// Deflated Sharpe ratio: [`probabilistic_sharpe`] against the Sharpe ratio
/// the best of the sweep would show by luck.
///
/// `trial_sharpes` is the per-period Sharpe ratio of every trial in the
/// sweep, including the one whose `returns` are given. Its non-`NaN` count
/// is `N` and its `ddof = 1` variance is `V`; the benchmark is
/// [`expected_max_sharpe`]`(N, V)`. Fewer than two non-`NaN` trials leave the
/// variance undefined and give `NaN`.
#[must_use]
pub fn deflated_sharpe(returns: &[f64], trial_sharpes: &[f64]) -> f64 {
    let n_trials = trial_sharpes.iter().filter(|v| !v.is_nan()).count();
    let sd = nanstd(trial_sharpes, 1);
    let benchmark = expected_max_sharpe(n_trials, sd * sd);
    probabilistic_sharpe(returns, benchmark)
}

/// Probability of backtest overfitting, by combinatorially symmetric
/// cross-validation (CSCV).
///
/// `perf` holds per-period returns for `n_cols` strategies, column-major:
/// column `j` is `perf[j * n_rows..(j + 1) * n_rows]`. The rows are cut into
/// `n_splits` contiguous blocks of near-equal size (sizes differ by at most
/// one row). For every way of choosing `n_splits / 2` blocks as the
/// in-sample set, with the rest out of sample:
///
/// 1. the in-sample winner is the column with the highest in-sample **mean
///    return** (not Sharpe: a short block makes Sharpe noisy, and a column
///    with no in-sample variance would win with an infinite ratio);
/// 2. its out-of-sample relative rank is `ω = rank / (n_cols + 1)`, where
///    `rank` runs from 1 (worst) to `n_cols` (best), ties taking the average
///    rank;
/// 3. the logit is `λ = ln(ω / (1 - ω))`.
///
/// PBO is the fraction of splits with `λ ≤ 0` — the in-sample winner
/// landing at or below the out-of-sample median. `NaN` bars are skipped in
/// every mean; a column with no bars in a sample ranks below every column
/// that has some.
///
/// # Errors
///
/// [`Error::Invariant`] if `perf.len() != n_rows * n_cols`, `n_cols < 2`,
/// `n_splits` is odd, zero, larger than `n_rows` or larger than
/// [`MAX_PBO_SPLITS`].
pub fn pbo_cscv(perf: &[f64], n_rows: usize, n_cols: usize, n_splits: usize) -> Result<f64> {
    if n_rows.checked_mul(n_cols) != Some(perf.len()) {
        return Err(Error::Invariant(format!(
            "pbo_cscv: perf has {} values, expected n_rows * n_cols = {n_rows} * {n_cols}",
            perf.len()
        )));
    }
    if n_cols < 2 {
        return Err(Error::Invariant(format!(
            "pbo_cscv: need at least 2 columns to rank, got {n_cols}"
        )));
    }
    if n_splits == 0 || n_splits % 2 != 0 {
        return Err(Error::Invariant(format!(
            "pbo_cscv: n_splits must be even and positive, got {n_splits}"
        )));
    }
    if n_splits > n_rows || n_splits > MAX_PBO_SPLITS {
        return Err(Error::Invariant(format!(
            "pbo_cscv: n_splits = {n_splits} exceeds n_rows = {n_rows} or the cap of {MAX_PBO_SPLITS}"
        )));
    }

    // Per-column, per-block sum and count of non-`NaN` bars, so each split
    // is a sum over blocks rather than a pass over rows.
    let bounds: Vec<usize> = (0..=n_splits).map(|b| b * n_rows / n_splits).collect();
    let mut sums = vec![0.0; n_cols * n_splits];
    let mut counts = vec![0_usize; n_cols * n_splits];
    for j in 0..n_cols {
        let col = &perf[j * n_rows..(j + 1) * n_rows];
        for b in 0..n_splits {
            for &v in &col[bounds[b]..bounds[b + 1]] {
                if !v.is_nan() {
                    sums[j * n_splits + b] += v;
                    counts[j * n_splits + b] += 1;
                }
            }
        }
    }
    // A column with no bars in the sample sorts below everything.
    let mean_over = |j: usize, mask: u32| -> f64 {
        let (mut s, mut c) = (0.0, 0_usize);
        for b in 0..n_splits {
            if mask & (1 << b) != 0 {
                s += sums[j * n_splits + b];
                c += counts[j * n_splits + b];
            }
        }
        if c == 0 {
            f64::NEG_INFINITY
        } else {
            s / c as f64
        }
    };

    let half = (n_splits / 2) as u32;
    let full: u32 = (1 << n_splits) - 1;
    let mut n_combos = 0_usize;
    let mut n_overfit = 0_usize;
    let mut oos = vec![0.0; n_cols];
    for is_mask in 0..=full {
        if is_mask.count_ones() != half {
            continue;
        }
        let oos_mask = full & !is_mask;

        let mut best = 0;
        let mut best_is = mean_over(0, is_mask);
        for j in 1..n_cols {
            let m = mean_over(j, is_mask);
            if m > best_is {
                best = j;
                best_is = m;
            }
        }

        for (j, slot) in oos.iter_mut().enumerate() {
            *slot = mean_over(j, oos_mask);
        }
        let target = oos[best];
        let below = oos.iter().filter(|&&v| v < target).count();
        let tied = oos
            .iter()
            .filter(|&&v| v.partial_cmp(&target) == Some(std::cmp::Ordering::Equal))
            .count();
        // Average 1-based rank over the tied block, which includes `best`.
        let rank = below as f64 + (tied as f64 + 1.0) / 2.0;
        let omega = rank / (n_cols as f64 + 1.0);
        let logit = (omega / (1.0 - omega)).ln();

        n_combos += 1;
        if logit <= 0.0 {
            n_overfit += 1;
        }
    }
    Ok(n_overfit as f64 / n_combos as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    /// Tiny deterministic generator: uniform in `[-1, 1)`.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0
        }
    }

    #[test]
    fn norm_cdf_known_values() {
        assert!(close(norm_cdf(0.0), 0.5, 1e-15));
        assert!(close(norm_cdf(1.96), 0.975_002_104_851_780, 1e-12));
        assert!(close(norm_cdf(-1.0), 0.158_655_253_931_457_05, 1e-13));
        assert!(close(norm_cdf(3.0), 0.998_650_101_968_369_9, 1e-13));
        // Deep tail keeps relative accuracy.
        let tail = norm_cdf(-10.0);
        assert!(close(tail / 7.619_853_024_160_527e-24, 1.0, 1e-10));
        assert!(norm_cdf(f64::NAN).is_nan());
    }

    #[test]
    fn norm_ppf_known_values_and_edges() {
        assert!(close(norm_ppf(0.5), 0.0, 1e-15));
        assert!(close(norm_ppf(0.975), 1.959_963_984_540_054, 1e-12));
        assert!(close(norm_ppf(0.025), -1.959_963_984_540_054, 1e-12));
        assert!(norm_ppf(0.0).is_infinite() && norm_ppf(0.0) < 0.0);
        assert!(norm_ppf(1.0).is_infinite() && norm_ppf(1.0) > 0.0);
        assert!(norm_ppf(-0.1).is_nan());
        assert!(norm_ppf(1.1).is_nan());
        assert!(norm_ppf(f64::NAN).is_nan());
    }

    #[test]
    fn norm_ppf_round_trips_through_norm_cdf() {
        for &p in &[
            1e-12, 1e-6, 0.001, 0.02, 0.024_25, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999_999,
        ] {
            let back = norm_cdf(norm_ppf(p));
            assert!(close(back / p, 1.0, 1e-12), "p = {p}, back = {back}");
        }
        // Above the median Φ(x) is rounded near one, so the upper side stops at
        // x = 4: past that the input itself has lost the digits to recover.
        for &x in &[-8.0, -3.0, -1.0, -0.1, 0.0, 0.4, 2.5, 4.0] {
            assert!(close(norm_ppf(norm_cdf(x)), x, 1e-9), "x = {x}");
        }
    }

    fn sample_returns(len: usize) -> Vec<f64> {
        let mut rng = Lcg(7);
        (0..len).map(|_| 0.001 + 0.01 * rng.next()).collect()
    }

    #[test]
    fn psr_is_one_half_when_sharpe_equals_benchmark() {
        let r = sample_returns(500);
        let sr = nanmean(&r) / nanstd(&r, 1);
        assert!(close(probabilistic_sharpe(&r, sr), 0.5, 1e-12));
    }

    #[test]
    fn psr_increases_with_sample_length() {
        let base = sample_returns(100);
        let long: Vec<f64> = base.iter().cycle().take(base.len() * 4).copied().collect();
        let short_psr = probabilistic_sharpe(&base, 0.0);
        let long_psr = probabilistic_sharpe(&long, 0.0);
        assert!(short_psr > 0.5);
        assert!(long_psr > short_psr, "{long_psr} <= {short_psr}");
    }

    #[test]
    fn psr_skips_nans_and_rejects_degenerate_columns() {
        let r = sample_returns(200);
        let mut gappy = r.clone();
        gappy.insert(10, f64::NAN);
        gappy.push(f64::NAN);
        assert!(close(
            probabilistic_sharpe(&gappy, 0.0),
            probabilistic_sharpe(&r, 0.0),
            1e-15
        ));
        assert!(probabilistic_sharpe(&[0.01], 0.0).is_nan());
        assert!(probabilistic_sharpe(&[0.01, 0.01, 0.01], 0.0).is_nan());
    }

    #[test]
    fn expected_max_sharpe_grows_with_trials() {
        assert!(expected_max_sharpe(0, 1.0).is_nan());
        assert!(close(expected_max_sharpe(1, 1.0), 0.0, 0.0));
        let e10 = expected_max_sharpe(10, 1.0);
        let e1000 = expected_max_sharpe(1000, 1.0);
        assert!(e10 > 0.0 && e1000 > e10);
        // Known value for N = 1000, V = 1: about 3.25 standard deviations.
        assert!(close(e1000, 3.25, 0.02), "{e1000}");
        assert!(close(expected_max_sharpe(1000, 4.0), 2.0 * e1000, 1e-12));
    }

    #[test]
    fn deflated_sharpe_is_below_psr_against_zero_with_many_trials() {
        let r = sample_returns(500);
        let mut rng = Lcg(99);
        let trials: Vec<f64> = (0..200).map(|_| 0.05 * rng.next()).collect();
        let dsr = deflated_sharpe(&r, &trials);
        let psr0 = probabilistic_sharpe(&r, 0.0);
        assert!(dsr < psr0, "{dsr} >= {psr0}");
    }

    fn column_major(cols: &[Vec<f64>]) -> Vec<f64> {
        cols.iter().flatten().copied().collect()
    }

    #[test]
    fn pbo_is_zero_for_a_dominant_column() {
        let n_rows = 400;
        let mut rng = Lcg(1);
        let mut cols: Vec<Vec<f64>> = Vec::new();
        cols.push((0..n_rows).map(|_| 0.05 + 0.001 * rng.next()).collect());
        for _ in 0..7 {
            cols.push((0..n_rows).map(|_| 0.01 * rng.next()).collect());
        }
        let pbo = pbo_cscv(&column_major(&cols), n_rows, cols.len(), 8).unwrap();
        assert!(close(pbo, 0.0, 0.0), "{pbo}");
    }

    #[test]
    fn pbo_is_around_one_half_for_pure_noise() {
        let n_rows = 1000;
        let n_cols = 20;
        let mut rng = Lcg(12_345);
        let cols: Vec<Vec<f64>> = (0..n_cols)
            .map(|_| (0..n_rows).map(|_| rng.next()).collect())
            .collect();
        let pbo = pbo_cscv(&column_major(&cols), n_rows, n_cols, 10).unwrap();
        assert!((0.3..=1.0).contains(&pbo), "{pbo}");
    }

    #[test]
    fn pbo_rejects_bad_shapes() {
        let perf = vec![0.0; 40];
        assert!(pbo_cscv(&perf, 20, 2, 3).is_err());
        assert!(pbo_cscv(&perf, 20, 2, 0).is_err());
        assert!(pbo_cscv(&perf, 40, 1, 4).is_err());
        assert!(pbo_cscv(&perf, 20, 3, 4).is_err());
        assert!(pbo_cscv(&perf, 10, 4, 12).is_err());
        assert!(pbo_cscv(&vec![0.0; 400], 20, 20, 18).is_err());
        assert!(pbo_cscv(&perf, 20, 2, 4).is_ok());
    }
}
