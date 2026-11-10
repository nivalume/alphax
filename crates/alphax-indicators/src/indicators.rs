//! The eight indicators, each swept across its parameters.
//!
//! Every function takes one parameter struct per combination and returns an
//! [`Indicator`] whose outputs are `n_combinations * n_input_cols` wide, in
//! the layout [`crate::sweep`] fixes. That output drops straight into the
//! simulator, which sees only columns.
//!
//! # Shared work is computed once
//!
//! A sweep repeats parameters constantly — a MACD grid over three fast and
//! three slow windows asks for at most six distinct moving averages, not
//! eighteen, and Bollinger Bands over three windows and four alphas needs
//! three. Every function here caches its expensive intermediates by the
//! parameters that determine them. On a wide sweep
//! that is most of the work.

use std::collections::HashMap;

use alphax_core::columns::Label;
use alphax_core::error::{Error, Result};
use alphax_core::matrix::Matrix;
use alphax_generic::elementwise::{diff_1d, fshift_1d};
use alphax_generic::rolling::{
    ewm_mean_1d, ewm_std_1d, rolling_max_1d, rolling_mean_1d, rolling_min_1d, rolling_std_1d,
};

use crate::sweep::{ParamLevel, column_of, sweep_columns};

/// A named set of output matrices sharing one swept column axis.
#[derive(Debug, Clone)]
pub struct Indicator {
    /// The swept column labels: one level per parameter, plus the input's.
    pub columns: alphax_core::columns::Columns,
    outputs: Vec<(&'static str, Matrix)>,
}

impl Indicator {
    /// One output by name.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invariant`] if the indicator has no such output.
    pub fn get(&self, name: &str) -> Result<&Matrix> {
        self.outputs
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, m)| m)
            .ok_or_else(|| {
                Error::Invariant(format!(
                    "no output named {name:?}; this indicator has {:?}",
                    self.names()
                ))
            })
    }

    /// The names of every output, in order.
    #[must_use]
    pub fn names(&self) -> Vec<&'static str> {
        self.outputs.iter().map(|(n, _)| *n).collect()
    }
}

/// Labels for the input columns: the caller's symbols, or `0..n`.
fn input_labels(n: usize, labels: Option<&[Label]>) -> Result<Vec<Label>> {
    match labels {
        Some(l) if l.len() == n => Ok(l.to_vec()),
        Some(l) => Err(Error::Invariant(format!(
            "expected {n} input labels, got {}",
            l.len()
        ))),
        None => Ok((0..n).map(Label::from).collect()),
    }
}

/// A moving average, exponential or simple.
///
/// `minp` is the window, always: an indicator that starts reporting before
/// its window is full reports a different indicator.
fn moving_average(a: &[f64], window: usize, ewm: bool, adjust: bool) -> Result<Vec<f64>> {
    if ewm {
        ewm_mean_1d(a, window, window, adjust)
    } else {
        rolling_mean_1d(a, window, Some(window))
    }
}

/// A moving standard deviation, exponential or simple.
fn moving_std(a: &[f64], window: usize, ewm: bool, adjust: bool, ddof: usize) -> Result<Vec<f64>> {
    if ewm {
        ewm_std_1d(a, window, window, adjust)
    } else {
        rolling_std_1d(a, window, Some(window), ddof)
    }
}

/// Caches a per-column series keyed by whatever determines it.
struct Cache<K> {
    entries: HashMap<K, Matrix>,
}

impl<K: std::hash::Hash + Eq + Clone> Cache<K> {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// The cached matrix for `key`, computing it once if absent.
    fn get_or_insert(&mut self, key: K, compute: impl Fn() -> Result<Matrix>) -> Result<&Matrix> {
        if !self.entries.contains_key(&key) {
            let value = compute()?;
            self.entries.insert(key.clone(), value);
        }
        Ok(&self.entries[&key])
    }
}

/// Applies a per-column transform to a whole matrix.
fn per_column(m: &Matrix, f: impl Fn(&[f64]) -> Result<Vec<f64>>) -> Result<Matrix> {
    let mut out = Matrix::zeros(m.n_rows(), m.n_cols());
    for col in 0..m.n_cols() {
        out.column_mut(col).copy_from_slice(&f(m.column(col))?);
    }
    Ok(out)
}

/// Writes one combination's result into the swept output.
fn place(out: &mut Matrix, source: &Matrix, comb: usize, n_input: usize) {
    for s in 0..n_input {
        let col = column_of(comb, s, n_input);
        out.column_mut(col).copy_from_slice(source.column(s));
    }
}

/// Parameters for [`ma`] and [`mstd`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaParams {
    /// Window length in bars.
    pub window: usize,
    /// Exponential rather than simple.
    pub ewm: bool,
}

/// Moving average, swept.
///
/// # Errors
///
/// Propagates any window error, and rejects mismatched input labels.
pub fn ma(
    close: &Matrix,
    params: &[MaParams],
    adjust: bool,
    labels: Option<&[Label]>,
) -> Result<Indicator> {
    let n_input = close.n_cols();
    let mut out = Matrix::zeros(close.n_rows(), params.len() * n_input);
    let mut cache = Cache::new();
    for (i, p) in params.iter().enumerate() {
        let m = cache.get_or_insert((p.window, p.ewm), || {
            per_column(close, |c| moving_average(c, p.window, p.ewm, adjust))
        })?;
        let m = m.clone();
        place(&mut out, &m, i, n_input);
    }
    Ok(Indicator {
        columns: ma_columns(params, close.n_cols(), labels)?,
        outputs: vec![("ma", out)],
    })
}

fn ma_columns(
    params: &[MaParams],
    n_input: usize,
    labels: Option<&[Label]>,
) -> Result<alphax_core::columns::Columns> {
    let windows: Vec<usize> = params.iter().map(|p| p.window).collect();
    let ewms: Vec<bool> = params.iter().map(|p| p.ewm).collect();
    sweep_columns(
        &[
            ParamLevel::new("window", &windows),
            ParamLevel::new("ewm", &ewms),
        ],
        ("symbol", &input_labels(n_input, labels)?),
    )
}

/// Moving standard deviation, swept.
///
/// # Errors
///
/// As [`ma`].
pub fn mstd(
    close: &Matrix,
    params: &[MaParams],
    adjust: bool,
    ddof: usize,
    labels: Option<&[Label]>,
) -> Result<Indicator> {
    let n_input = close.n_cols();
    let mut out = Matrix::zeros(close.n_rows(), params.len() * n_input);
    let mut cache = Cache::new();
    for (i, p) in params.iter().enumerate() {
        let m = cache
            .get_or_insert((p.window, p.ewm), || {
                per_column(close, |c| moving_std(c, p.window, p.ewm, adjust, ddof))
            })?
            .clone();
        place(&mut out, &m, i, n_input);
    }
    Ok(Indicator {
        columns: ma_columns(params, close.n_cols(), labels)?,
        outputs: vec![("mstd", out)],
    })
}

/// Parameters for [`bbands`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BbParams {
    /// Window length in bars.
    pub window: usize,
    /// Exponential rather than simple.
    pub ewm: bool,
    /// How many standard deviations the bands sit at.
    pub alpha: f64,
}

/// Bollinger bands: `middle`, `upper`, `lower`.
///
/// The middle band is a moving average and the outer two sit `alpha` standard
/// deviations either side. Sweeping `alpha` costs almost nothing — the
/// average and the deviation depend only on the window, so they are computed
/// once and reused across every alpha.
///
/// # Errors
///
/// As [`ma`].
pub fn bbands(
    close: &Matrix,
    params: &[BbParams],
    adjust: bool,
    ddof: usize,
    labels: Option<&[Label]>,
) -> Result<Indicator> {
    let n_input = close.n_cols();
    let n_rows = close.n_rows();
    let mut middle = Matrix::zeros(n_rows, params.len() * n_input);
    let mut upper = Matrix::zeros(n_rows, params.len() * n_input);
    let mut lower = Matrix::zeros(n_rows, params.len() * n_input);
    let mut ma_cache = Cache::new();
    let mut std_cache = Cache::new();

    for (i, p) in params.iter().enumerate() {
        let m = ma_cache
            .get_or_insert((p.window, p.ewm), || {
                per_column(close, |c| moving_average(c, p.window, p.ewm, adjust))
            })?
            .clone();
        let s = std_cache
            .get_or_insert((p.window, p.ewm), || {
                per_column(close, |c| moving_std(c, p.window, p.ewm, adjust, ddof))
            })?
            .clone();

        let mut up = Matrix::zeros(n_rows, n_input);
        let mut low = Matrix::zeros(n_rows, n_input);
        for col in 0..n_input {
            let (mc, sc) = (m.column(col), s.column(col));
            for row in 0..n_rows {
                up.column_mut(col)[row] = mc[row] + p.alpha * sc[row];
                low.column_mut(col)[row] = mc[row] - p.alpha * sc[row];
            }
        }
        place(&mut middle, &m, i, n_input);
        place(&mut upper, &up, i, n_input);
        place(&mut lower, &low, i, n_input);
    }

    let windows: Vec<usize> = params.iter().map(|p| p.window).collect();
    let ewms: Vec<bool> = params.iter().map(|p| p.ewm).collect();
    let alphas: Vec<f64> = params.iter().map(|p| p.alpha).collect();
    Ok(Indicator {
        columns: sweep_columns(
            &[
                ParamLevel::new("window", &windows),
                ParamLevel::new("ewm", &ewms),
                ParamLevel::new("alpha", &alphas),
            ],
            ("symbol", &input_labels(n_input, labels)?),
        )?,
        outputs: vec![("middle", middle), ("upper", upper), ("lower", lower)],
    })
}

/// Relative strength index, swept.
///
/// # A zero denominator is a feature
///
/// `100 - 100 / (1 + up/down)` with no down moves divides by zero, giving an
/// infinite ratio and an RSI of exactly 100. That is the correct reading —
/// every move was up — and it arrives through infinity rather than a special
/// case, which is why the tolerance layer treats infinities as live values.
///
/// # Errors
///
/// As [`ma`].
pub fn rsi(
    close: &Matrix,
    params: &[MaParams],
    adjust: bool,
    labels: Option<&[Label]>,
) -> Result<Indicator> {
    let n_input = close.n_cols();
    let n_rows = close.n_rows();

    // Gains and losses, split from the bar-to-bar change. Losses are carried
    // positive so the ratio is a ratio of magnitudes.
    let mut up = Matrix::zeros(n_rows, n_input);
    let mut down = Matrix::zeros(n_rows, n_input);
    for col in 0..n_input {
        let delta = diff_1d(close.column(col), 1);
        for (row, &d) in delta.iter().enumerate() {
            up.column_mut(col)[row] = if d < 0.0 { 0.0 } else { d };
            down.column_mut(col)[row] = if d > 0.0 { 0.0 } else { -d };
        }
    }

    let mut out = Matrix::zeros(n_rows, params.len() * n_input);
    let mut cache: Cache<(usize, bool)> = Cache::new();
    for (i, p) in params.iter().enumerate() {
        // Both averages share a key, so they are computed together.
        let value = cache
            .get_or_insert((p.window, p.ewm), || {
                let roll_up = per_column(&up, |c| moving_average(c, p.window, p.ewm, adjust))?;
                let roll_down = per_column(&down, |c| moving_average(c, p.window, p.ewm, adjust))?;
                let mut value = Matrix::zeros(n_rows, n_input);
                for col in 0..n_input {
                    let (u, d) = (roll_up.column(col), roll_down.column(col));
                    for row in 0..n_rows {
                        let rs = u[row] / d[row];
                        value.column_mut(col)[row] = 100.0 - 100.0 / (1.0 + rs);
                    }
                }
                Ok(value)
            })?
            .clone();
        place(&mut out, &value, i, n_input);
    }

    Ok(Indicator {
        columns: ma_columns(params, n_input, labels)?,
        outputs: vec![("rsi", out)],
    })
}

/// Parameters for [`stoch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StochParams {
    /// Window for the raw %K.
    pub k_window: usize,
    /// Smoothing window for %D.
    pub d_window: usize,
    /// Whether %D is smoothed exponentially.
    pub d_ewm: bool,
}

/// Stochastic oscillator: `percent_k` and `percent_d`.
///
/// # Errors
///
/// As [`ma`].
pub fn stoch(
    high: &Matrix,
    low: &Matrix,
    close: &Matrix,
    params: &[StochParams],
    adjust: bool,
    labels: Option<&[Label]>,
) -> Result<Indicator> {
    check_shapes(&[high, low, close])?;
    let n_input = close.n_cols();
    let n_rows = close.n_rows();
    let mut k_out = Matrix::zeros(n_rows, params.len() * n_input);
    let mut d_out = Matrix::zeros(n_rows, params.len() * n_input);
    // Keyed on the %K window alone: %D's smoothing does not change the range.
    let mut range_cache: Cache<usize> = Cache::new();

    for (i, p) in params.iter().enumerate() {
        let k = range_cache
            .get_or_insert(p.k_window, || {
                let roll_min = per_column(low, |c| rolling_min_1d(c, p.k_window, None))?;
                let roll_max = per_column(high, |c| rolling_max_1d(c, p.k_window, None))?;
                let mut k = Matrix::zeros(n_rows, n_input);
                for col in 0..n_input {
                    let (lo, hi, cl) = (
                        roll_min.column(col),
                        roll_max.column(col),
                        close.column(col),
                    );
                    for row in 0..n_rows {
                        k.column_mut(col)[row] = 100.0 * (cl[row] - lo[row]) / (hi[row] - lo[row]);
                    }
                }
                Ok(k)
            })?
            .clone();
        let d = per_column(&k, |c| moving_average(c, p.d_window, p.d_ewm, adjust))?;
        place(&mut k_out, &k, i, n_input);
        place(&mut d_out, &d, i, n_input);
    }

    let k_windows: Vec<usize> = params.iter().map(|p| p.k_window).collect();
    let d_windows: Vec<usize> = params.iter().map(|p| p.d_window).collect();
    let d_ewms: Vec<bool> = params.iter().map(|p| p.d_ewm).collect();
    Ok(Indicator {
        columns: sweep_columns(
            &[
                ParamLevel::new("k_window", &k_windows),
                ParamLevel::new("d_window", &d_windows),
                ParamLevel::new("d_ewm", &d_ewms),
            ],
            ("symbol", &input_labels(n_input, labels)?),
        )?,
        outputs: vec![("percent_k", k_out), ("percent_d", d_out)],
    })
}

/// Parameters for [`macd`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MacdParams {
    /// Fast moving-average window.
    pub fast_window: usize,
    /// Slow moving-average window.
    pub slow_window: usize,
    /// Window for smoothing the MACD line into the signal line.
    pub signal_window: usize,
    /// Whether the two moving averages are exponential.
    pub macd_ewm: bool,
    /// Whether the signal line is exponential.
    pub signal_ewm: bool,
}

/// Moving average convergence/divergence: `macd` and `signal`.
///
/// The fast and slow averages are cached under one key space, so a grid that
/// reuses a window as "fast" in one combination and "slow" in another
/// computes it once.
///
/// # Errors
///
/// As [`ma`].
pub fn macd(
    close: &Matrix,
    params: &[MacdParams],
    adjust: bool,
    labels: Option<&[Label]>,
) -> Result<Indicator> {
    let n_input = close.n_cols();
    let n_rows = close.n_rows();
    let mut macd_out = Matrix::zeros(n_rows, params.len() * n_input);
    let mut signal_out = Matrix::zeros(n_rows, params.len() * n_input);
    let mut cache: Cache<(usize, bool)> = Cache::new();

    for (i, p) in params.iter().enumerate() {
        let fast = cache
            .get_or_insert((p.fast_window, p.macd_ewm), || {
                per_column(close, |c| {
                    moving_average(c, p.fast_window, p.macd_ewm, adjust)
                })
            })?
            .clone();
        let slow = cache
            .get_or_insert((p.slow_window, p.macd_ewm), || {
                per_column(close, |c| {
                    moving_average(c, p.slow_window, p.macd_ewm, adjust)
                })
            })?
            .clone();

        let mut line = Matrix::zeros(n_rows, n_input);
        for col in 0..n_input {
            let (f, s) = (fast.column(col), slow.column(col));
            for row in 0..n_rows {
                line.column_mut(col)[row] = f[row] - s[row];
            }
        }
        let signal = per_column(&line, |c| {
            moving_average(c, p.signal_window, p.signal_ewm, adjust)
        })?;
        place(&mut macd_out, &line, i, n_input);
        place(&mut signal_out, &signal, i, n_input);
    }

    let fast: Vec<usize> = params.iter().map(|p| p.fast_window).collect();
    let slow: Vec<usize> = params.iter().map(|p| p.slow_window).collect();
    let signal: Vec<usize> = params.iter().map(|p| p.signal_window).collect();
    let macd_ewms: Vec<bool> = params.iter().map(|p| p.macd_ewm).collect();
    let signal_ewms: Vec<bool> = params.iter().map(|p| p.signal_ewm).collect();
    Ok(Indicator {
        columns: sweep_columns(
            &[
                ParamLevel::new("fast_window", &fast),
                ParamLevel::new("slow_window", &slow),
                ParamLevel::new("signal_window", &signal),
                ParamLevel::new("macd_ewm", &macd_ewms),
                ParamLevel::new("signal_ewm", &signal_ewms),
            ],
            ("symbol", &input_labels(n_input, labels)?),
        )?,
        outputs: vec![("macd", macd_out), ("signal", signal_out)],
    })
}

/// True range: the widest of the three ways a bar can move.
///
/// The first bar has no previous close, so it falls back to the bar's own
/// range rather than reaching for a value that does not exist.
#[must_use]
pub fn true_range(high: &Matrix, low: &Matrix, close: &Matrix) -> Matrix {
    let (n_rows, n_cols) = (high.n_rows(), high.n_cols());
    let mut out = Matrix::zeros(n_rows, n_cols);
    for col in 0..n_cols {
        let (h, l, c) = (high.column(col), low.column(col), close.column(col));
        let dst = out.column_mut(col);
        for row in 0..n_rows {
            let tr1 = h[row] - l[row];
            dst[row] = if row == 0 {
                tr1
            } else {
                let tr2 = (h[row] - c[row - 1]).abs();
                let tr3 = (l[row] - c[row - 1]).abs();
                tr1.max(tr2).max(tr3)
            };
        }
    }
    out
}

/// Average true range: `tr` and `atr`.
///
/// # Errors
///
/// As [`ma`].
pub fn atr(
    high: &Matrix,
    low: &Matrix,
    close: &Matrix,
    params: &[MaParams],
    adjust: bool,
    labels: Option<&[Label]>,
) -> Result<Indicator> {
    check_shapes(&[high, low, close])?;
    let n_input = close.n_cols();
    let n_rows = close.n_rows();
    // Computed once for the whole sweep: the range does not depend on the
    // smoothing window.
    let tr = true_range(high, low, close);

    let mut tr_out = Matrix::zeros(n_rows, params.len() * n_input);
    let mut atr_out = Matrix::zeros(n_rows, params.len() * n_input);
    let mut cache = Cache::new();
    for (i, p) in params.iter().enumerate() {
        let smoothed = cache
            .get_or_insert((p.window, p.ewm), || {
                per_column(&tr, |c| moving_average(c, p.window, p.ewm, adjust))
            })?
            .clone();
        place(&mut tr_out, &tr, i, n_input);
        place(&mut atr_out, &smoothed, i, n_input);
    }

    Ok(Indicator {
        columns: ma_columns(params, n_input, labels)?,
        outputs: vec![("tr", tr_out), ("atr", atr_out)],
    })
}

/// On-balance volume: `obv`. Takes no parameters.
///
/// Volume is added on a bar that closed at or above the previous close and
/// subtracted otherwise, then accumulated. The first bar has no previous
/// close to compare against, so its comparison is false and its volume counts
/// positive — which follows from `NaN` comparing false, not from a special
/// case, and is worth knowing before changing the comparison's direction.
///
/// # Errors
///
/// Rejects mismatched shapes or input labels.
pub fn obv(close: &Matrix, volume: &Matrix, labels: Option<&[Label]>) -> Result<Indicator> {
    check_shapes(&[close, volume])?;
    let (n_rows, n_cols) = (close.n_rows(), close.n_cols());
    let mut out = Matrix::zeros(n_rows, n_cols);
    for col in 0..n_cols {
        let c = close.column(col);
        let prev = fshift_1d(c, 1, f64::NAN);
        let v = volume.column(col);
        let dst = out.column_mut(col);
        // `NaN` contributes zero rather than poisoning the running total,
        // as in a cumulative sum.
        let mut running = 0.0;
        for row in 0..n_rows {
            let signed = if c[row] < prev[row] { -v[row] } else { v[row] };
            if !signed.is_nan() {
                running += signed;
            }
            dst[row] = running;
        }
    }
    Ok(Indicator {
        columns: sweep_columns(&[], ("symbol", &input_labels(n_cols, labels)?))?,
        outputs: vec![("obv", out)],
    })
}

/// Rejects inputs that do not share a shape.
fn check_shapes(inputs: &[&Matrix]) -> Result<()> {
    let Some(first) = inputs.first() else {
        return Ok(());
    };
    for m in inputs {
        if m.n_rows() != first.n_rows() || m.n_cols() != first.n_cols() {
            return Err(Error::Broadcast {
                left: (first.n_rows(), first.n_cols()),
                right: (m.n_rows(), m.n_cols()),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "these assert on exact indicator values, including the infinities \
              and NaNs that are part of the contract"
)]
mod tests {
    use super::*;

    fn col(values: &[f64]) -> Matrix {
        Matrix::from_column_major(values.to_vec(), values.len(), 1).unwrap()
    }

    fn two_cols(a: &[f64], b: &[f64]) -> Matrix {
        let mut data = a.to_vec();
        data.extend_from_slice(b);
        Matrix::from_column_major(data, a.len(), 2).unwrap()
    }

    #[test]
    fn a_sweep_widens_the_output_not_the_result_count() {
        let close = two_cols(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 40.0]);
        let params = [
            MaParams {
                window: 2,
                ewm: false,
            },
            MaParams {
                window: 3,
                ewm: false,
            },
        ];
        let r = ma(&close, &params, false, None).unwrap();
        let m = r.get("ma").unwrap();
        assert_eq!(m.n_cols(), 4, "two windows times two symbols");
        assert_eq!(r.columns.n_cols(), 4);
        // Window 2 on symbol 0, then window 2 on symbol 1, then window 3.
        assert!(m.column(0)[0].is_nan());
        assert_eq!(m.column(0)[1], 1.5);
        assert_eq!(m.column(1)[1], 15.0);
        assert!(m.column(2)[1].is_nan(), "window 3 is not full yet");
        assert_eq!(m.column(2)[2], 2.0);
    }

    #[test]
    fn an_indicator_starts_only_when_its_window_is_full() {
        let close = col(&[1.0, 2.0, 3.0, 4.0]);
        let r = ma(
            &close,
            &[MaParams {
                window: 3,
                ewm: false,
            }],
            false,
            None,
        )
        .unwrap();
        let m = r.get("ma").unwrap();
        assert!(m.column(0)[0].is_nan() && m.column(0)[1].is_nan());
        assert_eq!(m.column(0)[2], 2.0);
    }

    #[test]
    fn an_unknown_output_name_is_rejected() {
        let close = col(&[1.0, 2.0]);
        let r = ma(
            &close,
            &[MaParams {
                window: 1,
                ewm: false,
            }],
            false,
            None,
        )
        .unwrap();
        assert!(r.get("nope").is_err());
        assert_eq!(r.names(), ["ma"]);
    }

    #[test]
    fn bollinger_bands_bracket_their_middle() {
        let close = col(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let p = [BbParams {
            window: 3,
            ewm: false,
            alpha: 2.0,
        }];
        let r = bbands(&close, &p, false, 0, None).unwrap();
        let (mid, up, low) = (
            r.get("middle").unwrap(),
            r.get("upper").unwrap(),
            r.get("lower").unwrap(),
        );
        for row in 2..5 {
            assert!(up.column(0)[row] > mid.column(0)[row]);
            assert!(low.column(0)[row] < mid.column(0)[row]);
            // Symmetric about the middle.
            let a = up.column(0)[row] - mid.column(0)[row];
            let b = mid.column(0)[row] - low.column(0)[row];
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn sweeping_alpha_reuses_the_window_work() {
        // Two alphas over one window: the middle band must be identical.
        let close = col(&[1.0, 3.0, 2.0, 5.0, 4.0]);
        let p = [
            BbParams {
                window: 3,
                ewm: false,
                alpha: 1.0,
            },
            BbParams {
                window: 3,
                ewm: false,
                alpha: 2.0,
            },
        ];
        let r = bbands(&close, &p, false, 0, None).unwrap();
        let mid = r.get("middle").unwrap();
        // Compared on bits: the leading bars are `NaN`, which never compares
        // equal to itself, so `assert_eq!` on the slices would fail on two
        // results that are in fact identical.
        assert!(
            mid.column(0)
                .iter()
                .zip(mid.column(1))
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "the two alphas produced different middle bands"
        );
        // And the second alpha's bands are twice as far out.
        let up = r.get("upper").unwrap();
        let (a, b) = (
            up.column(0)[4] - mid.column(0)[4],
            up.column(1)[4] - mid.column(1)[4],
        );
        assert!((b - 2.0 * a).abs() < 1e-12);
    }

    #[test]
    fn rsi_reads_100_when_nothing_fell() {
        // Every move up means no down moves at all: the ratio is infinite and
        // the RSI is exactly 100, reached through infinity rather than a
        // special case.
        let close = col(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let r = rsi(
            &close,
            &[MaParams {
                window: 2,
                ewm: false,
            }],
            false,
            None,
        )
        .unwrap();
        assert_eq!(r.get("rsi").unwrap().column(0)[4], 100.0);
    }

    #[test]
    fn rsi_reads_0_when_nothing_rose() {
        let close = col(&[5.0, 4.0, 3.0, 2.0, 1.0]);
        let r = rsi(
            &close,
            &[MaParams {
                window: 2,
                ewm: false,
            }],
            false,
            None,
        )
        .unwrap();
        assert_eq!(r.get("rsi").unwrap().column(0)[4], 0.0);
    }

    #[test]
    fn true_range_falls_back_to_the_bar_on_the_first_one() {
        let high = col(&[10.0, 12.0]);
        let low = col(&[9.0, 8.0]);
        let close = col(&[9.5, 11.0]);
        let tr = true_range(&high, &low, &close);
        assert_eq!(tr.column(0)[0], 1.0, "no previous close to reach for");
        // Bar 1: range 4, |12 - 9.5| = 2.5, |8 - 9.5| = 1.5.
        assert_eq!(tr.column(0)[1], 4.0);
    }

    #[test]
    fn true_range_catches_a_gap_the_bar_alone_would_miss() {
        // A bar that opened far above the previous close has a small range
        // but a large true range.
        let high = col(&[10.0, 21.0]);
        let low = col(&[9.0, 20.0]);
        let close = col(&[10.0, 20.5]);
        let tr = true_range(&high, &low, &close);
        assert_eq!(tr.column(0)[1], 11.0, "measured from the previous close");
    }

    #[test]
    fn obv_adds_on_a_rise_and_subtracts_on_a_fall() {
        let close = col(&[10.0, 11.0, 9.0, 9.0]);
        let volume = col(&[100.0, 200.0, 300.0, 400.0]);
        let r = obv(&close, &volume, None).unwrap();
        // Bar 0 has no previous close, so its volume counts positive.
        assert_eq!(r.get("obv").unwrap().column(0), [100.0, 300.0, 0.0, 400.0]);
    }

    #[test]
    fn macd_reuses_a_window_that_is_fast_in_one_row_and_slow_in_another() {
        let close = col(&[1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0]);
        let p = [
            MacdParams {
                fast_window: 2,
                slow_window: 3,
                signal_window: 2,
                macd_ewm: false,
                signal_ewm: false,
            },
            MacdParams {
                fast_window: 3,
                slow_window: 4,
                signal_window: 2,
                macd_ewm: false,
                signal_ewm: false,
            },
        ];
        let r = macd(&close, &p, false, None).unwrap();
        assert_eq!(r.get("macd").unwrap().n_cols(), 2);
        assert_eq!(r.columns.n_cols(), 2);
    }

    #[test]
    fn stoch_sits_between_0_and_100() {
        let high = col(&[10.0, 12.0, 14.0, 13.0, 15.0]);
        let low = col(&[8.0, 9.0, 11.0, 10.0, 12.0]);
        let close = col(&[9.0, 11.0, 13.0, 11.0, 14.0]);
        let p = [StochParams {
            k_window: 3,
            d_window: 2,
            d_ewm: false,
        }];
        let r = stoch(&high, &low, &close, &p, false, None).unwrap();
        let k = r.get("percent_k").unwrap();
        for row in 2..5 {
            assert!((0.0..=100.0).contains(&k.column(0)[row]), "row {row}");
        }
    }

    #[test]
    fn mismatched_input_shapes_are_rejected() {
        let a = Matrix::zeros(3, 1);
        let b = Matrix::zeros(2, 1);
        assert!(obv(&a, &b, None).is_err());
        assert!(stoch(&a, &b, &a, &[], false, None).is_err());
    }

    #[test]
    fn wrong_input_labels_are_rejected() {
        let close = two_cols(&[1.0, 2.0], &[3.0, 4.0]);
        let labels = [Label::from("only-one")];
        assert!(
            ma(
                &close,
                &[MaParams {
                    window: 1,
                    ewm: false
                }],
                false,
                Some(&labels)
            )
            .is_err()
        );
    }
}
