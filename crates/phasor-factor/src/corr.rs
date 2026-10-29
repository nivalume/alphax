//! Correlation of paired, finite samples.

use crate::rank::Ranker;

/// Whether every value of `x` equals the first. Exact: a series that is
/// constant up to rounding still has a (meaningless) variance, and scipy
/// and pandas treat only exact constants as undefined.
#[allow(clippy::float_cmp, reason = "a constant is defined by exact equality")]
fn is_constant(x: &[f64]) -> bool {
    x.iter().all(|&v| v == x[0])
}

/// Pearson correlation of `x` and `y`, both finite and of equal length.
///
/// `NaN` when fewer than two pairs remain or either side is constant. Two
/// passes, mean first, so large offsets do not cancel.
pub(crate) fn pearson(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    if n < 2 || is_constant(x) || is_constant(y) {
        return f64::NAN;
    }
    let inv = 1.0 / n as f64;
    let mx = x.iter().sum::<f64>() * inv;
    let my = y.iter().sum::<f64>() * inv;
    let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
    for (&a, &b) in x.iter().zip(y) {
        let (dx, dy) = (a - mx, b - my);
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    let r = sxy / (sxx * syy).sqrt();
    // Rounding can carry a perfect correlation a hair past 1.
    r.clamp(-1.0, 1.0)
}

/// Scratch for [`Corr::spearman`] and the pair gathering in the IC kernels.
#[derive(Debug, Default)]
pub(crate) struct Corr {
    pub(crate) ranker: Ranker,
    pub(crate) xs: Vec<f64>,
    pub(crate) ys: Vec<f64>,
    rx: Vec<f64>,
    ry: Vec<f64>,
}

impl Corr {
    /// Clears the pair buffers.
    pub(crate) fn clear(&mut self) {
        self.xs.clear();
        self.ys.clear();
    }

    /// Adds the pair if both values are finite.
    #[inline]
    pub(crate) fn push(&mut self, x: f64, y: f64) {
        if x.is_finite() && y.is_finite() {
            self.xs.push(x);
            self.ys.push(y);
        }
    }

    /// Pearson of the gathered pairs.
    pub(crate) fn pearson(&self) -> f64 {
        pearson(&self.xs, &self.ys)
    }

    /// Spearman of the gathered pairs: Pearson of their average ranks.
    pub(crate) fn spearman(&mut self) -> f64 {
        self.ranker.rank(&self.xs, &mut self.rx);
        self.ranker.rank(&self.ys, &mut self.ry);
        pearson(&self.rx, &self.ry)
    }

    /// Pearson or Spearman of the gathered pairs, `NaN` below `min` pairs.
    pub(crate) fn compute(&mut self, method: crate::ic::Method, min: usize) -> f64 {
        if self.xs.len() < min.max(2) {
            return f64::NAN;
        }
        match method {
            crate::ic::Method::Pearson => self.pearson(),
            crate::ic::Method::Spearman => self.spearman(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use phasor_core::math::is_close;

    #[test]
    fn perfect_and_undefined() {
        assert!(is_close(pearson(&[1.0, 2.0, 3.0], &[5.0, 7.0, 9.0]), 1.0));
        assert!(is_close(pearson(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]), -1.0));
        assert!(pearson(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]).is_nan());
        assert!(pearson(&[1.0], &[1.0]).is_nan());
    }

    #[test]
    fn spearman_is_rank_pearson() {
        let mut c = Corr::default();
        for (x, y) in [(1.0, 1.0), (2.0, 8.0), (3.0, 27.0), (f64::NAN, 1.0)] {
            c.push(x, y);
        }
        assert_eq!(c.xs.len(), 3);
        assert!(is_close(c.spearman(), 1.0));
    }
}
