//! Summary statistics of an IC series, with a t-statistic that survives
//! overlapping returns.

use phasor_analysis::robust::norm_cdf;

/// What one IC series says about its factor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Summary {
    /// Mean IC.
    pub mean: f64,
    /// Sample standard deviation (`ddof = 1`).
    pub std: f64,
    /// `mean / std`: the information ratio of the IC, per period.
    pub icir: f64,
    /// `icir * sqrt(n)`: the t-statistic if the ICs were independent.
    pub t: f64,
    /// The Newey–West t-statistic: [`newey_west_variance`] in place of the
    /// variance, so autocorrelated ICs (overlapping forward returns, a
    /// persistent factor) do not overstate it.
    pub t_nw: f64,
    /// Two-sided normal p-value of `t_nw`.
    pub p_nw: f64,
    /// Share of ICs strictly above zero.
    pub hit: f64,
    /// Number of ICs that were not `NaN`.
    pub n: usize,
}

/// The Newey–West estimate of the long-run variance of `x` around `mean`:
/// `γ₀ + 2 Σ_{l=1}^{lags} (1 − l/(lags+1)) γ_l`, `γ_l = (1/n) Σ_{t≥l}
/// (x_t − mean)(x_{t−l} − mean)`, Bartlett weights, no small-sample
/// correction (statsmodels' `HAC` with `use_correction=False`). Lags past
/// the series are ignored.
#[must_use]
pub fn newey_west_variance(x: &[f64], mean: f64, lags: usize) -> f64 {
    let n = x.len();
    let gamma = |l: usize| -> f64 {
        x[l..]
            .iter()
            .zip(x)
            .map(|(&a, &b)| (a - mean) * (b - mean))
            .sum::<f64>()
            / n as f64
    };
    let mut omega = gamma(0);
    for l in 1..=lags.min(n.saturating_sub(1)) {
        let w = 1.0 - l as f64 / (lags as f64 + 1.0);
        omega += 2.0 * w * gamma(l);
    }
    omega
}

/// The Newey–West lag for `n` ICs of `horizon`-bar forward returns: at least
/// `horizon − 1`, which is how many neighbours an overlapping return shares
/// with, and at least the usual rule of thumb `⌊4 (n/100)^{2/9}⌋`.
#[must_use]
pub fn auto_lags(n: usize, horizon: usize) -> usize {
    let rule = (4.0 * (n as f64 / 100.0).powf(2.0 / 9.0)).floor() as usize;
    rule.max(horizon.saturating_sub(1))
}

/// [`Summary`] of `series`, `NaN` entries dropped first (the remaining values
/// are treated as consecutive), with `lags` Newey–West lags.
///
/// With no values everything is `NaN` and `n` is 0; with one, or a constant
/// series, the statistics that divide by its spread are `NaN`.
#[must_use]
pub fn summarize(series: &[f64], lags: usize) -> Summary {
    let x: Vec<f64> = series.iter().copied().filter(|v| !v.is_nan()).collect();
    let n = x.len();
    let nan = f64::NAN;
    if n == 0 {
        return Summary {
            mean: nan,
            std: nan,
            icir: nan,
            t: nan,
            t_nw: nan,
            p_nw: nan,
            hit: nan,
            n: 0,
        };
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let hit = x.iter().filter(|&&v| v > 0.0).count() as f64 / n as f64;
    if n < 2 {
        return Summary {
            mean,
            std: nan,
            icir: nan,
            t: nan,
            t_nw: nan,
            p_nw: nan,
            hit,
            n,
        };
    }
    #[allow(clippy::float_cmp, reason = "constant series check is exact equality")]
    if x.iter().all(|&v| v == x[0]) {
        return Summary {
            mean: x[0],
            std: 0.0,
            icir: nan,
            t: nan,
            t_nw: nan,
            p_nw: nan,
            hit,
            n,
        };
    }
    let ss: f64 = x.iter().map(|v| (v - mean) * (v - mean)).sum();
    let std = (ss / (n - 1) as f64).sqrt();
    let (icir, t) = if std > 0.0 {
        let icir = mean / std;
        (icir, icir * (n as f64).sqrt())
    } else {
        (nan, nan)
    };
    let omega = newey_west_variance(&x, mean, lags);
    let (t_nw, p_nw) = if omega > 0.0 {
        let t_nw = mean / (omega / n as f64).sqrt();
        (t_nw, 2.0 * norm_cdf(-t_nw.abs()))
    } else {
        (nan, nan)
    };
    Summary {
        mean,
        std,
        icir,
        t,
        t_nw,
        p_nw,
        hit,
        n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use phasor_core::math::is_close;

    #[test]
    fn zero_lags_is_the_plain_t_up_to_ddof() {
        let x = [0.1, -0.05, 0.2, 0.03, 0.07];
        let s = summarize(&x, 0);
        let n = x.len() as f64;
        // γ₀ uses 1/n, std uses 1/(n-1).
        assert!(is_close(s.t_nw, s.t * (n / (n - 1.0)).sqrt()));
        assert!(is_close(s.hit, 0.8));
        assert_eq!(s.n, 5);
    }

    #[test]
    fn degenerate_series() {
        assert_eq!(summarize(&[f64::NAN], 1).n, 0);
        assert!(summarize(&[0.3], 1).std.is_nan());
        let c = summarize(&[0.2, 0.2, 0.2], 1);
        assert!(c.icir.is_nan() && c.t_nw.is_nan() && is_close(c.mean, 0.2));
    }

    #[test]
    fn lag_rule() {
        assert_eq!(auto_lags(100, 1), 4);
        assert_eq!(auto_lags(100, 20), 19);
    }
}
