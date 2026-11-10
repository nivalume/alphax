//! The indicators one bar at a time.
//!
//! Each stream holds the state its batch kernel carries from bar to bar and
//! performs the same floating-point operations in the same order, so feeding
//! a series through [`Stream::update`] gives bit for bit what the batch
//! function gives for that column. The rolling mean and deviation, for
//! instance, are differences of prefix sums over the whole history in the
//! batch kernel; the stream keeps the running sums and the last `window` of
//! them, not a re-summed window.
//!
//! State is `O(window)` per stream, and an update is `O(1)` amortized.

use std::collections::VecDeque;

use alphax_core::error::{Error, Result};

use crate::{BbParams, MaParams, MacdParams, StochParams};

fn check_window(window: usize, minp: usize) -> Result<()> {
    if window == 0 {
        return Err(Error::Invariant("the window must be at least 1".into()));
    }
    if minp > window {
        return Err(Error::Invariant(format!(
            "min_periods ({minp}) cannot exceed the window ({window})"
        )));
    }
    Ok(())
}

/// `alphax_generic::rolling::rolling_mean_1d` and `rolling_std_1d`, one bar
/// at a time: prefix sums of the values and of their squares, prefix `NaN`
/// counts, and the last `window` of each.
#[derive(Debug, Clone)]
struct Prefix {
    window: usize,
    bar: usize,
    sum: f64,
    sum_sq: f64,
    nans: usize,
    /// `(sum, sum_sq, nans)` after each of the last `window` bars.
    past: VecDeque<(f64, f64, usize)>,
}

impl Prefix {
    fn new(window: usize) -> Self {
        Self {
            window,
            bar: 0,
            sum: 0.0,
            sum_sq: 0.0,
            nans: 0,
            past: VecDeque::with_capacity(window + 1),
        }
    }

    /// `(count, sum, sum_sq)` of the window ending at this bar.
    fn push(&mut self, x: f64) -> (usize, f64, f64) {
        if x.is_nan() {
            self.nans += 1;
        } else {
            self.sum += x;
            self.sum_sq += x * x;
        }
        self.past.push_back((self.sum, self.sum_sq, self.nans));
        let i = self.bar;
        self.bar += 1;
        if i < self.window {
            (i + 1 - self.nans, self.sum, self.sum_sq)
        } else {
            let (sum, sum_sq, nans) = self.past.pop_front().expect("window + 1 entries");
            (
                self.window - (self.nans - nans),
                self.sum - sum,
                self.sum_sq - sum_sq,
            )
        }
    }
}

/// Rolling mean, as `rolling_mean_1d`.
#[derive(Debug, Clone)]
pub struct RollingMean {
    prefix: Prefix,
    minp: usize,
}

impl RollingMean {
    /// # Errors
    ///
    /// A zero window, or `minp` above it.
    pub fn new(window: usize, minp: usize) -> Result<Self> {
        check_window(window, minp)?;
        Ok(Self {
            prefix: Prefix::new(window),
            minp,
        })
    }

    pub fn update(&mut self, x: f64) -> f64 {
        let (len, sum, _) = self.prefix.push(x);
        if len < self.minp {
            f64::NAN
        } else {
            sum / len as f64
        }
    }
}

/// Rolling standard deviation, as `rolling_std_1d`.
#[derive(Debug, Clone)]
pub struct RollingStd {
    prefix: Prefix,
    minp: usize,
    ddof: usize,
}

impl RollingStd {
    /// # Errors
    ///
    /// A zero window, or `minp` above it.
    pub fn new(window: usize, minp: usize, ddof: usize) -> Result<Self> {
        check_window(window, minp)?;
        Ok(Self {
            prefix: Prefix::new(window),
            minp,
            ddof,
        })
    }

    pub fn update(&mut self, x: f64) -> f64 {
        let (len, sum, sum_sq) = self.prefix.push(x);
        if len < self.minp || len == self.ddof {
            return f64::NAN;
        }
        let n = len as f64;
        let mean = sum / n;
        let var = (sum_sq - 2.0 * sum * mean + n * (mean * mean)).abs();
        (var / (n - self.ddof as f64)).sqrt()
    }
}

/// Rolling minimum or maximum, as `rolling_min_1d`/`rolling_max_1d`.
#[derive(Debug, Clone)]
pub struct RollingExtremum {
    window: usize,
    minp: usize,
    want_min: bool,
    bar: usize,
    count: usize,
    /// The last `window` values.
    recent: VecDeque<f64>,
    /// `(bar, value)` candidates, each strictly better than every later one.
    candidates: VecDeque<(usize, f64)>,
}

impl RollingExtremum {
    /// # Errors
    ///
    /// A zero window, or `minp` above it.
    pub fn new(window: usize, minp: usize, want_min: bool) -> Result<Self> {
        check_window(window, minp)?;
        Ok(Self {
            window,
            minp,
            want_min,
            bar: 0,
            count: 0,
            recent: VecDeque::with_capacity(window + 1),
            candidates: VecDeque::new(),
        })
    }

    pub fn update(&mut self, v: f64) -> f64 {
        let i = self.bar;
        self.bar += 1;
        self.recent.push_back(v);
        if i >= self.window {
            let left = self.recent.pop_front().expect("window + 1 values");
            if !left.is_nan() {
                self.count -= 1;
                if self.candidates.front().map(|c| c.0) == Some(i - self.window) {
                    self.candidates.pop_front();
                }
            }
        }
        if !v.is_nan() {
            self.count += 1;
            let want_min = self.want_min;
            while self
                .candidates
                .back()
                .is_some_and(|&(_, b)| if want_min { b >= v } else { b <= v })
            {
                self.candidates.pop_back();
            }
            self.candidates.push_back((i, v));
        }
        let best = self.candidates.front().map_or(v, |c| c.1);
        if self.count < self.minp {
            f64::NAN
        } else {
            best
        }
    }
}

/// Exponentially weighted mean, as `ewm_mean_1d`.
#[derive(Debug, Clone)]
pub struct EwmMean {
    minp: usize,
    old_wt_factor: f64,
    new_wt: f64,
    adjust: bool,
    started: bool,
    avg: f64,
    nobs: usize,
    old_wt: f64,
}

impl EwmMean {
    /// # Errors
    ///
    /// `minp` above `span`.
    pub fn new(span: usize, minp: usize, adjust: bool) -> Result<Self> {
        if minp > span {
            return Err(Error::Invariant(format!(
                "min_periods ({minp}) cannot exceed the span ({span})"
            )));
        }
        let com = (span as f64 - 1.0) / 2.0;
        let alpha = 1.0 / (1.0 + com);
        Ok(Self {
            minp,
            old_wt_factor: 1.0 - alpha,
            new_wt: if adjust { 1.0 } else { alpha },
            adjust,
            started: false,
            avg: f64::NAN,
            nobs: 0,
            old_wt: 1.0,
        })
    }

    #[allow(
        clippy::float_cmp,
        reason = "the batch kernel's guard against drift on an unmoved value"
    )]
    pub fn update(&mut self, cur: f64) -> f64 {
        let observed = !cur.is_nan();
        if !self.started {
            self.started = true;
            self.avg = cur;
            self.nobs = usize::from(observed);
        } else {
            self.nobs += usize::from(observed);
            if !self.avg.is_nan() {
                self.old_wt *= self.old_wt_factor;
                if observed {
                    if self.avg != cur {
                        self.avg = (self.old_wt * self.avg + self.new_wt * cur)
                            / (self.old_wt + self.new_wt);
                    }
                    if self.adjust {
                        self.old_wt += self.new_wt;
                    } else {
                        self.old_wt = 1.0;
                    }
                }
            } else if observed {
                self.avg = cur;
            }
        }
        if self.nobs >= self.minp {
            self.avg
        } else {
            f64::NAN
        }
    }
}

/// Exponentially weighted standard deviation, as `ewm_std_1d`.
#[derive(Debug, Clone)]
pub struct EwmStd {
    minp: usize,
    old_wt_factor: f64,
    new_wt: f64,
    adjust: bool,
    started: bool,
    mean: f64,
    nobs: usize,
    cov: f64,
    sum_wt: f64,
    sum_wt2: f64,
    old_wt: f64,
}

impl EwmStd {
    /// # Errors
    ///
    /// `minp` above `span`.
    pub fn new(span: usize, minp: usize, adjust: bool) -> Result<Self> {
        if minp > span {
            return Err(Error::Invariant(format!(
                "min_periods ({minp}) cannot exceed the span ({span})"
            )));
        }
        let com = (span as f64 - 1.0) / 2.0;
        let alpha = 1.0 / (1.0 + com);
        Ok(Self {
            minp,
            old_wt_factor: 1.0 - alpha,
            new_wt: if adjust { 1.0 } else { alpha },
            adjust,
            started: false,
            mean: f64::NAN,
            nobs: 0,
            cov: 0.0,
            sum_wt: 1.0,
            sum_wt2: 1.0,
            old_wt: 1.0,
        })
    }

    #[allow(
        clippy::float_cmp,
        reason = "the batch kernel's guard against drift on an unmoved value"
    )]
    pub fn update(&mut self, cur: f64) -> f64 {
        let observed = !cur.is_nan();
        if !self.started {
            // The first bar has no deviation to report.
            self.started = true;
            self.mean = cur;
            self.nobs = usize::from(observed);
            return f64::NAN;
        }
        self.nobs += usize::from(observed);
        if !self.mean.is_nan() {
            let f = self.old_wt_factor;
            self.sum_wt *= f;
            self.sum_wt2 *= f * f;
            self.old_wt *= f;
            if observed {
                let old_mean = self.mean;
                if self.mean != cur {
                    self.mean =
                        (self.old_wt * old_mean + self.new_wt * cur) / (self.old_wt + self.new_wt);
                }
                let mean = self.mean;
                self.cov = (self.old_wt * (self.cov + (old_mean - mean) * (old_mean - mean))
                    + self.new_wt * ((cur - mean) * (cur - mean)))
                    / (self.old_wt + self.new_wt);
                self.sum_wt += self.new_wt;
                self.sum_wt2 += self.new_wt * self.new_wt;
                self.old_wt += self.new_wt;
                if !self.adjust {
                    self.sum_wt /= self.old_wt;
                    self.sum_wt2 /= self.old_wt * self.old_wt;
                    self.old_wt = 1.0;
                }
            }
        } else if observed {
            self.mean = cur;
        }
        if self.nobs >= self.minp {
            let numerator = self.sum_wt * self.sum_wt;
            let denominator = numerator - self.sum_wt2;
            if denominator > 0.0 {
                ((numerator / denominator) * self.cov).sqrt()
            } else {
                f64::NAN
            }
        } else {
            f64::NAN
        }
    }
}

/// The indicators' moving average: simple or exponential, reporting from a
/// full window.
#[derive(Debug, Clone)]
enum Average {
    Simple(RollingMean),
    Exp(EwmMean),
}

impl Average {
    fn new(window: usize, ewm: bool, adjust: bool) -> Result<Self> {
        Ok(if ewm {
            Self::Exp(EwmMean::new(window, window, adjust)?)
        } else {
            Self::Simple(RollingMean::new(window, window)?)
        })
    }

    fn update(&mut self, x: f64) -> f64 {
        match self {
            Self::Simple(m) => m.update(x),
            Self::Exp(m) => m.update(x),
        }
    }
}

#[derive(Debug, Clone)]
enum Deviation {
    Simple(RollingStd),
    Exp(EwmStd),
}

impl Deviation {
    fn new(window: usize, ewm: bool, adjust: bool, ddof: usize) -> Result<Self> {
        Ok(if ewm {
            Self::Exp(EwmStd::new(window, window, adjust)?)
        } else {
            Self::Simple(RollingStd::new(window, window, ddof)?)
        })
    }

    fn update(&mut self, x: f64) -> f64 {
        match self {
            Self::Simple(m) => m.update(x),
            Self::Exp(m) => m.update(x),
        }
    }
}

/// One bar of input; fields an indicator does not read may be `NaN`.
#[derive(Debug, Clone, Copy)]
pub struct Bar {
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl Bar {
    /// A bar with only a close.
    #[must_use]
    pub fn close(close: f64) -> Self {
        Self {
            high: f64::NAN,
            low: f64::NAN,
            close,
            volume: f64::NAN,
        }
    }
}

#[derive(Debug, Clone)]
enum Kind {
    Ma(Average),
    Mstd(Deviation),
    Bbands {
        mean: Average,
        std: Deviation,
        alpha: f64,
    },
    Rsi {
        up: Average,
        down: Average,
    },
    Stoch {
        low: RollingExtremum,
        high: RollingExtremum,
        d: Average,
    },
    Macd {
        fast: Average,
        slow: Average,
        signal: Average,
    },
    Atr(Average),
    Obv {
        running: f64,
    },
}

/// One indicator over one series, a bar at a time.
///
/// Outputs per bar, in [`Stream::names`] order: `ma`; `mstd`; `middle`,
/// `upper`, `lower`; `rsi`; `percent_k`, `percent_d`; `macd`, `signal`;
/// `tr`, `atr`; `obv`. Each equals that bar's value from the batch function
/// with the same parameters, bit for bit.
#[derive(Debug, Clone)]
pub struct Stream {
    kind: Kind,
    /// The previous bar's close; `None` before the first bar.
    prev_close: Option<f64>,
    out: [f64; 3],
}

impl Stream {
    fn of(kind: Kind) -> Self {
        Self {
            kind,
            prev_close: None,
            out: [f64::NAN; 3],
        }
    }

    /// Moving average.
    ///
    /// # Errors
    ///
    /// A zero window.
    pub fn ma(p: MaParams, adjust: bool) -> Result<Self> {
        Ok(Self::of(Kind::Ma(Average::new(p.window, p.ewm, adjust)?)))
    }

    /// Moving standard deviation.
    ///
    /// # Errors
    ///
    /// A zero window.
    pub fn mstd(p: MaParams, adjust: bool, ddof: usize) -> Result<Self> {
        Ok(Self::of(Kind::Mstd(Deviation::new(
            p.window, p.ewm, adjust, ddof,
        )?)))
    }

    /// Bollinger bands.
    ///
    /// # Errors
    ///
    /// A zero window.
    pub fn bbands(p: BbParams, adjust: bool, ddof: usize) -> Result<Self> {
        Ok(Self::of(Kind::Bbands {
            mean: Average::new(p.window, p.ewm, adjust)?,
            std: Deviation::new(p.window, p.ewm, adjust, ddof)?,
            alpha: p.alpha,
        }))
    }

    /// Relative strength index.
    ///
    /// # Errors
    ///
    /// A zero window.
    pub fn rsi(p: MaParams, adjust: bool) -> Result<Self> {
        Ok(Self::of(Kind::Rsi {
            up: Average::new(p.window, p.ewm, adjust)?,
            down: Average::new(p.window, p.ewm, adjust)?,
        }))
    }

    /// Stochastic oscillator.
    ///
    /// # Errors
    ///
    /// A zero window.
    pub fn stoch(p: StochParams, adjust: bool) -> Result<Self> {
        Ok(Self::of(Kind::Stoch {
            low: RollingExtremum::new(p.k_window, p.k_window, true)?,
            high: RollingExtremum::new(p.k_window, p.k_window, false)?,
            d: Average::new(p.d_window, p.d_ewm, adjust)?,
        }))
    }

    /// Moving average convergence divergence.
    ///
    /// # Errors
    ///
    /// A zero window.
    pub fn macd(p: MacdParams, adjust: bool) -> Result<Self> {
        Ok(Self::of(Kind::Macd {
            fast: Average::new(p.fast_window, p.macd_ewm, adjust)?,
            slow: Average::new(p.slow_window, p.macd_ewm, adjust)?,
            signal: Average::new(p.signal_window, p.signal_ewm, adjust)?,
        }))
    }

    /// Average true range.
    ///
    /// # Errors
    ///
    /// A zero window.
    pub fn atr(p: MaParams, adjust: bool) -> Result<Self> {
        Ok(Self::of(Kind::Atr(Average::new(p.window, p.ewm, adjust)?)))
    }

    /// On-balance volume.
    #[must_use]
    pub fn obv() -> Self {
        Self::of(Kind::Obv { running: 0.0 })
    }

    /// The outputs' names.
    #[must_use]
    pub fn names(&self) -> &'static [&'static str] {
        match self.kind {
            Kind::Ma(_) => &["ma"],
            Kind::Mstd(_) => &["mstd"],
            Kind::Bbands { .. } => &["middle", "upper", "lower"],
            Kind::Rsi { .. } => &["rsi"],
            Kind::Stoch { .. } => &["percent_k", "percent_d"],
            Kind::Macd { .. } => &["macd", "signal"],
            Kind::Atr(_) => &["tr", "atr"],
            Kind::Obv { .. } => &["obv"],
        }
    }

    /// Takes one bar and returns this bar's outputs.
    pub fn update(&mut self, bar: Bar) -> &[f64] {
        let first = self.prev_close.is_none();
        let prev = self.prev_close.unwrap_or(f64::NAN);
        let out = &mut self.out;
        let n = match &mut self.kind {
            Kind::Ma(m) => {
                out[0] = m.update(bar.close);
                1
            }
            Kind::Mstd(s) => {
                out[0] = s.update(bar.close);
                1
            }
            Kind::Bbands { mean, std, alpha } => {
                let (m, s) = (mean.update(bar.close), std.update(bar.close));
                *out = [m, m + *alpha * s, m - *alpha * s];
                3
            }
            Kind::Rsi { up, down } => {
                let d = bar.close - prev;
                let u = up.update(if d < 0.0 { 0.0 } else { d });
                let l = down.update(if d > 0.0 { 0.0 } else { -d });
                let rs = u / l;
                out[0] = 100.0 - 100.0 / (1.0 + rs);
                1
            }
            Kind::Stoch { low, high, d } => {
                let (lo, hi) = (low.update(bar.low), high.update(bar.high));
                let k = 100.0 * (bar.close - lo) / (hi - lo);
                out[0] = k;
                out[1] = d.update(k);
                2
            }
            Kind::Macd { fast, slow, signal } => {
                let line = fast.update(bar.close) - slow.update(bar.close);
                out[0] = line;
                out[1] = signal.update(line);
                2
            }
            Kind::Atr(m) => {
                let tr1 = bar.high - bar.low;
                let tr = if first {
                    tr1
                } else {
                    tr1.max((bar.high - prev).abs()).max((bar.low - prev).abs())
                };
                out[0] = tr;
                out[1] = m.update(tr);
                2
            }
            Kind::Obv { running } => {
                let signed = if bar.close < prev {
                    -bar.volume
                } else {
                    bar.volume
                };
                if !signed.is_nan() {
                    *running += signed;
                }
                out[0] = *running;
                1
            }
        };
        self.prev_close = Some(bar.close);
        &self.out[..n]
    }
}
