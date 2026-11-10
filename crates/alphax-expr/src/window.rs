//! `O(1)`-per-row trailing-window kernels for the additive rolling operators.
//!
//! Every accumulator here keeps running sums, adds the row entering the window
//! and subtracts the row leaving it. Two things keep that honest in floating
//! point:
//!
//! * The sums are rebuilt from the window itself every `window` rows, so
//!   add/subtract rounding error never accumulates over more than two windows.
//!   The rebuild costs `O(window)` once per `window` rows, so the total stays
//!   `O(n)`.
//! * Values are summed as offsets from a shift, the first value pushed into an
//!   empty window, so a variance of a price near 100 is not the small
//!   difference of two large squares.
//!
//! Qlib's zero checks (`Corr`'s `va == 0`, `Rsquare`'s `yvar <= 0`) are exact
//! comparisons. A running sum over identical values leaves a
//! rounding residue instead of zero, so [`Moments`] tracks how many trailing
//! observations are identical and reports an exact zero variance when all of
//! them are.
//!
//! `NaN` rows are skipped: they occupy a window slot but
//! contribute to no sum.

/// Additive state over one trailing window.
pub(crate) trait Accumulator<T: Copy> {
    /// Starts over from `window`, whose first element is row `start`.
    fn rebuild(&mut self, window: &[T], start: usize);
    /// Adds the row entering the window.
    fn push(&mut self, row: usize, value: T);
    /// Removes the row leaving the window.
    fn pop(&mut self, row: usize, value: T);
    /// The reduction over `window`, whose first element is row `start`.
    fn value(&self, window: &[T], start: usize) -> f64;
}

/// Runs `acc` over every trailing window of `values`.
#[cfg(test)]
pub(crate) fn slide<T: Copy>(
    values: &[T],
    window: usize,
    acc: &mut impl Accumulator<T>,
) -> Vec<f64> {
    let mut out = vec![0.0; values.len()];
    slide_into(values, window, acc, &mut out);
    out
}

/// [`slide`] into a caller-supplied series of the same length.
pub(crate) fn slide_into<T: Copy>(
    values: &[T],
    window: usize,
    acc: &mut impl Accumulator<T>,
    out: &mut [f64],
) {
    let window = window.max(1);
    let mut rebuilt = 0;
    for (row, slot) in out.iter_mut().enumerate() {
        let start = (row + 1).saturating_sub(window);
        if row == 0 || row - rebuilt >= window {
            acc.rebuild(&values[start..=row], start);
            rebuilt = row;
        } else {
            acc.push(row, values[row]);
            if row >= window {
                acc.pop(row - window, values[row - window]);
            }
        }
        *slot = acc.value(&values[start..=row], start);
    }
}

/// Count, shifted first and second moments, and an exact-constant detector
/// over the non-`NaN` values of a window.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct Moments {
    shift: f64,
    count: usize,
    sum: f64,
    sum_sq: f64,
    last: f64,
    /// How many of the most recently pushed observations equal `last`.
    run: usize,
}

impl Moments {
    fn rebuild<'a>(&mut self, values: impl Iterator<Item = &'a f64>) {
        self.count = 0;
        for &value in values {
            self.push(value);
        }
    }

    #[allow(
        clippy::float_cmp,
        reason = "the constant-window detector is defined by exact equality"
    )]
    fn push(&mut self, value: f64) {
        if value.is_nan() {
            return;
        }
        if self.count == 0 {
            // Nothing depends on the old shift once the window is empty, so
            // re-anchor it on the data (and drop any rounding residue).
            *self = Self {
                shift: value,
                ..Self::default()
            };
        }
        let d = value - self.shift;
        self.count += 1;
        self.sum += d;
        self.sum_sq += d * d;
        if value == self.last {
            self.run += 1;
        } else {
            self.last = value;
            self.run = 1;
        }
    }

    fn pop(&mut self, value: f64) {
        if value.is_nan() {
            return;
        }
        let d = value - self.shift;
        self.count -= 1;
        self.sum -= d;
        self.sum_sq -= d * d;
    }

    /// Whether every observation in the window is the same value.
    fn constant(&self) -> bool {
        self.run >= self.count
    }

    /// `Σ (x - mean)²`, exactly zero for a constant window.
    fn centered_sq(&self) -> f64 {
        if self.constant() {
            return 0.0;
        }
        (self.sum_sq - self.sum * self.sum / self.count as f64).max(0.0)
    }

    fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Mean minus the shift, exact for a constant window.
    fn mean_offset(&self) -> f64 {
        if self.constant() {
            return self.last - self.shift;
        }
        self.sum / self.count as f64
    }

    /// Sample standard deviation with `ddof`, `NaN` below `ddof + 1` values.
    pub(crate) fn std(&self, ddof: usize) -> f64 {
        if self.count <= ddof {
            return f64::NAN;
        }
        (self.centered_sq() / (self.count - ddof) as f64).sqrt()
    }
}

/// Rolling sum of non-`NaN` values; `NaN` for an all-`NaN` window.
#[derive(Debug, Default)]
pub(crate) struct Sum {
    count: usize,
    sum: f64,
}

impl Accumulator<f64> for Sum {
    fn rebuild(&mut self, window: &[f64], _: usize) {
        *self = Self::default();
        for &value in window {
            self.push(0, value);
        }
    }
    fn push(&mut self, _: usize, value: f64) {
        if !value.is_nan() {
            self.count += 1;
            self.sum += value;
        }
    }
    fn pop(&mut self, _: usize, value: f64) {
        if !value.is_nan() {
            self.count -= 1;
            self.sum -= value;
        }
    }
    fn value(&self, _: &[f64], _: usize) -> f64 {
        if self.count == 0 { f64::NAN } else { self.sum }
    }
}

/// Rolling count of non-`NaN` values; `NaN` for an all-`NaN` window.
///
/// Integer bookkeeping, so it never needs the periodic rebuild for accuracy;
/// it takes it anyway because every accumulator does.
#[derive(Debug, Default)]
pub(crate) struct Count(usize);

impl Accumulator<f64> for Count {
    fn rebuild(&mut self, window: &[f64], _: usize) {
        self.0 = window.iter().filter(|value| !value.is_nan()).count();
    }
    fn push(&mut self, _: usize, value: f64) {
        self.0 += usize::from(!value.is_nan());
    }
    fn pop(&mut self, _: usize, value: f64) {
        self.0 -= usize::from(!value.is_nan());
    }
    fn value(&self, _: &[f64], _: usize) -> f64 {
        if self.0 == 0 { f64::NAN } else { self.0 as f64 }
    }
}

/// Rolling sample standard deviation (`ddof = 1`), or its square.
#[derive(Debug, Default)]
pub(crate) struct Deviation {
    moments: Moments,
    squared: bool,
}

impl Deviation {
    pub(crate) fn std() -> Self {
        Self::default()
    }
    pub(crate) fn var() -> Self {
        Self {
            squared: true,
            ..Self::default()
        }
    }
}

impl Accumulator<f64> for Deviation {
    fn rebuild(&mut self, window: &[f64], _: usize) {
        self.moments.rebuild(window.iter());
    }
    fn push(&mut self, _: usize, value: f64) {
        self.moments.push(value);
    }
    fn pop(&mut self, _: usize, value: f64) {
        self.moments.pop(value);
    }
    fn value(&self, _: &[f64], _: usize) -> f64 {
        let std = self.moments.std(1);
        if self.squared { std * std } else { std }
    }
}

/// Qlib's `WMA`: weights `1..=len` over the whole window (including `NaN`
/// slots) normalised to sum to one, then averaged over the non-`NaN` count.
#[derive(Debug, Default)]
pub(crate) struct Wma {
    origin: usize,
    count: usize,
    sum: f64,
    /// `Σ (row - origin) · value`.
    weighted: f64,
}

impl Accumulator<f64> for Wma {
    fn rebuild(&mut self, window: &[f64], start: usize) {
        *self = Self {
            origin: start,
            ..Self::default()
        };
        for (offset, &value) in window.iter().enumerate() {
            self.push(start + offset, value);
        }
    }
    fn push(&mut self, row: usize, value: f64) {
        if !value.is_nan() {
            self.count += 1;
            self.sum += value;
            self.weighted += (row - self.origin) as f64 * value;
        }
    }
    fn pop(&mut self, row: usize, value: f64) {
        if !value.is_nan() {
            self.count -= 1;
            self.sum -= value;
            self.weighted -= (row - self.origin) as f64 * value;
        }
    }
    fn value(&self, window: &[f64], start: usize) -> f64 {
        if self.count == 0 {
            return f64::NAN;
        }
        let len = window.len();
        let weight_sum = (len * (len + 1) / 2) as f64;
        // Weight of row r is r - start + 1 = (r - origin) - (start - origin - 1).
        let lag = (start - self.origin) as f64 - 1.0;
        (self.weighted - lag * self.sum) / weight_sum / self.count as f64
    }
}

/// Which of the three regression outputs to report.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Fit {
    Slope,
    Rsquare,
    Resi,
}

/// Least squares of the window's non-`NaN` values against their positions
/// `1..=len` in the window, matching [`crate::eval::regression`].
#[derive(Debug)]
pub(crate) struct Regression {
    fit: Fit,
    origin: usize,
    y: Moments,
    /// Sums over x = row - origin, which are integers and therefore exact.
    sx: f64,
    sxx: f64,
    /// `Σ x · (y - shift)`.
    sxy: f64,
}

impl Regression {
    pub(crate) fn new(fit: Fit) -> Self {
        Self {
            fit,
            origin: 0,
            y: Moments::default(),
            sx: 0.0,
            sxx: 0.0,
            sxy: 0.0,
        }
    }
}

impl Accumulator<f64> for Regression {
    fn rebuild(&mut self, window: &[f64], start: usize) {
        self.origin = start;
        self.y.rebuild(std::iter::empty());
        for (offset, &value) in window.iter().enumerate() {
            self.push(start + offset, value);
        }
    }
    fn push(&mut self, row: usize, value: f64) {
        if !value.is_nan() {
            let x = (row - self.origin) as f64;
            if self.y.is_empty() {
                (self.sx, self.sxx, self.sxy) = (0.0, 0.0, 0.0);
            }
            self.y.push(value);
            self.sx += x;
            self.sxx += x * x;
            self.sxy += x * (value - self.y.shift);
        }
    }
    fn pop(&mut self, row: usize, value: f64) {
        if !value.is_nan() {
            let x = (row - self.origin) as f64;
            self.y.pop(value);
            self.sx -= x;
            self.sxx -= x * x;
            self.sxy -= x * (value - self.y.shift);
        }
    }
    fn value(&self, window: &[f64], start: usize) -> f64 {
        if self.y.count < 2 {
            return f64::NAN;
        }
        let n = self.y.count as f64;
        // Window positions are 1-based: position = (row - origin) - lag.
        let lag = (start - self.origin) as f64 - 1.0;
        let sx = self.sx - lag * n;
        let sxx = self.sxx - 2.0 * lag * self.sx + lag * lag * n;
        let sy = n * self.y.mean_offset();
        let sxy = self.sxy - lag * sy;
        let xvar = n * sxx - sx * sx;
        // A constant window has an exactly flat fit; the running `Σ x·y` would
        // leave a residue scaled by the distance from the shift.
        let covariance = if self.y.constant() {
            0.0
        } else {
            n * sxy - sx * sy
        };
        let slope = covariance / xvar;
        match self.fit {
            Fit::Slope => slope,
            Fit::Rsquare => {
                let yvar = n * self.y.centered_sq();
                let sample_std = (yvar / (n * (n - 1.0))).sqrt();
                if xvar <= 0.0 || yvar <= 0.0 || sample_std <= 2e-5 {
                    f64::NAN
                } else {
                    covariance * covariance / (xvar * yvar)
                }
            }
            Fit::Resi => {
                let intercept = sy / n - slope * sx / n;
                let last = window.last().copied().unwrap_or(f64::NAN) - self.y.shift;
                last - (slope * window.len() as f64 + intercept)
            }
        }
    }
}

/// Qlib's rolling `Corr` or `Cov` of two aligned series.
///
/// `Corr` first rejects a window where either series alone (every non-`NaN`
/// value, paired or not) has a sample deviation at or below `2e-5`; the
/// correlation itself uses only rows where both are present.
#[derive(Debug, Default)]
pub(crate) struct Pair {
    correlation: bool,
    left: Moments,
    right: Moments,
    paired_left: Moments,
    paired_right: Moments,
    /// `Σ (a - paired_left.shift) · (b - paired_right.shift)` over pairs.
    cross: f64,
}

impl Pair {
    pub(crate) fn corr() -> Self {
        Self {
            correlation: true,
            ..Self::default()
        }
    }
    pub(crate) fn cov() -> Self {
        Self::default()
    }
}

impl Accumulator<(f64, f64)> for Pair {
    fn rebuild(&mut self, window: &[(f64, f64)], start: usize) {
        for moments in [
            &mut self.left,
            &mut self.right,
            &mut self.paired_left,
            &mut self.paired_right,
        ] {
            moments.rebuild(std::iter::empty());
        }
        for (offset, &pair) in window.iter().enumerate() {
            self.push(start + offset, pair);
        }
    }
    fn push(&mut self, _: usize, (a, b): (f64, f64)) {
        self.left.push(a);
        self.right.push(b);
        if !a.is_nan() && !b.is_nan() {
            if self.paired_left.is_empty() {
                self.cross = 0.0;
            }
            self.paired_left.push(a);
            self.paired_right.push(b);
            self.cross += (a - self.paired_left.shift) * (b - self.paired_right.shift);
        }
    }
    fn pop(&mut self, _: usize, (a, b): (f64, f64)) {
        self.left.pop(a);
        self.right.pop(b);
        if !a.is_nan() && !b.is_nan() {
            self.paired_left.pop(a);
            self.paired_right.pop(b);
            self.cross -= (a - self.paired_left.shift) * (b - self.paired_right.shift);
        }
    }
    fn value(&self, _: &[(f64, f64)], _: usize) -> f64 {
        let count = self.paired_left.count;
        if self.correlation && (self.left.std(1) <= 2e-5 || self.right.std(1) <= 2e-5) {
            return f64::NAN;
        }
        if count < 2 {
            return f64::NAN;
        }
        let n = count as f64;
        // A constant side has exactly zero covariance; the running sums would
        // leave a rounding residue scaled by the distance from the shift.
        let covariance = if self.paired_left.constant() || self.paired_right.constant() {
            0.0
        } else {
            self.cross - n * self.paired_left.mean_offset() * self.paired_right.mean_offset()
        };
        if !self.correlation {
            return covariance / (n - 1.0);
        }
        let va = self.paired_left.centered_sq();
        let vb = self.paired_right.centered_sq();
        if va == 0.0 || vb == 0.0 {
            return f64::NAN;
        }
        covariance / (va * vb).sqrt()
    }
}

/// Qlib's rolling `Skew` (adjusted Fisher–Pearson) or `Kurt` (bias-corrected
/// excess kurtosis) from shifted power sums, `NaN` for a constant window (whose
/// central moments are then exactly zero).
#[derive(Debug, Default)]
pub(crate) struct Shape {
    kurtosis: bool,
    moments: Moments,
    sum_cube: f64,
    sum_fourth: f64,
}

impl Shape {
    pub(crate) fn skew() -> Self {
        Self::default()
    }
    pub(crate) fn kurt() -> Self {
        Self {
            kurtosis: true,
            ..Self::default()
        }
    }
}

impl Accumulator<f64> for Shape {
    fn rebuild(&mut self, window: &[f64], start: usize) {
        self.moments.rebuild(std::iter::empty());
        for (offset, &value) in window.iter().enumerate() {
            self.push(start + offset, value);
        }
    }
    fn push(&mut self, _: usize, value: f64) {
        if value.is_nan() {
            return;
        }
        if self.moments.is_empty() {
            (self.sum_cube, self.sum_fourth) = (0.0, 0.0);
        }
        self.moments.push(value);
        let d = value - self.moments.shift;
        self.sum_cube += d * d * d;
        self.sum_fourth += d * d * d * d;
    }
    fn pop(&mut self, _: usize, value: f64) {
        if value.is_nan() {
            return;
        }
        let d = value - self.moments.shift;
        self.moments.pop(value);
        self.sum_cube -= d * d * d;
        self.sum_fourth -= d * d * d * d;
    }
    fn value(&self, _: &[f64], _: usize) -> f64 {
        let count = self.moments.count;
        if count < if self.kurtosis { 4 } else { 3 } || self.moments.constant() {
            return f64::NAN;
        }
        let n = count as f64;
        let mu = self.moments.sum / n;
        let (s2, s3) = (self.moments.sum_sq, self.sum_cube);
        let m2 = self.moments.centered_sq();
        if self.kurtosis {
            let m4 = self.sum_fourth - 4.0 * mu * s3 + 6.0 * mu * mu * s2 - 3.0 * n * mu.powi(4);
            let biased_fisher = n * m4 / (m2 * m2) - 3.0;
            (n - 1.0) / ((n - 2.0) * (n - 3.0)) * ((n + 1.0) * biased_fisher + 6.0)
        } else {
            let m3 = s3 - 3.0 * mu * s2 + 2.0 * n * mu.powi(3);
            n * (n - 1.0).sqrt() / (n - 2.0) * m3 / m2.powf(1.5)
        }
    }
}
