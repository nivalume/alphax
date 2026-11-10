//! Cross-sectional kernels: one row of every symbol in, one row out.
//!
//! `NaN` inputs are left out of every statistic and give `NaN` outputs. Sums
//! run in symbol order, so a row's result does not depend on how rows are
//! split between threads.

use crate::graph::Cross;

/// Row scratch reused across rows.
#[derive(Default)]
pub(crate) struct Scratch {
    order: Vec<usize>,
}

/// Applies `op` to one row: `x` the first input's values, `y` the second's
/// (the group for `IndNeutralize`, the regressor for `CSResi`).
pub(crate) fn row(op: Cross, x: &[f64], y: &[f64], out: &mut [f64], scratch: &mut Scratch) {
    match op {
        Cross::Rank => rank(x, out, scratch),
        Cross::ZScore => {
            let (mean, std) = mean_std(x);
            for (slot, &v) in out.iter_mut().zip(x) {
                *slot = (v - mean) / std;
            }
        }
        Cross::Demean => {
            let mean = mean_std(x).0;
            for (slot, &v) in out.iter_mut().zip(x) {
                *slot = v - mean;
            }
        }
        Cross::Scale(bits) => {
            let scale = f64::from_bits(bits);
            let total: f64 = x.iter().filter(|v| !v.is_nan()).map(|v| v.abs()).sum();
            for (slot, &v) in out.iter_mut().zip(x) {
                *slot = v * scale / total;
            }
        }
        Cross::Neutralize => neutralize(x, y, out, scratch),
        Cross::Resi => residual(x, y, out),
    }
}

/// Mean and sample standard deviation of the values present.
fn mean_std(x: &[f64]) -> (f64, f64) {
    let (mut n, mut sum) = (0usize, 0.0);
    for &v in x.iter().filter(|v| !v.is_nan()) {
        n += 1;
        sum += v;
    }
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    let mean = sum / n as f64;
    if n < 2 {
        return (mean, f64::NAN);
    }
    let ss: f64 = x
        .iter()
        .filter(|v| !v.is_nan())
        .map(|v| (v - mean) * (v - mean))
        .sum();
    (mean, (ss / (n - 1) as f64).sqrt())
}

/// Percentile rank, ties averaged: the k-th smallest of n values present has
/// rank k / n.
#[allow(
    clippy::float_cmp,
    reason = "ties are defined by exact equality, as pandas' rank"
)]
fn rank(x: &[f64], out: &mut [f64], scratch: &mut Scratch) {
    let order = &mut scratch.order;
    order.clear();
    order.extend((0..x.len()).filter(|&i| !x[i].is_nan()));
    order.sort_by(|&a, &b| x[a].total_cmp(&x[b]).then(a.cmp(&b)));
    out.fill(f64::NAN);
    let n = order.len() as f64;
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && x[order[end]] == x[order[start]] {
            end += 1;
        }
        // Ranks start + 1 ..= end, averaged.
        let rank = (start + 1 + end) as f64 / 2.0;
        for &i in &order[start..end] {
            out[i] = rank / n;
        }
        start = end;
    }
}

/// `x` minus its group's mean; the group is `g`'s exact value on the row.
#[allow(clippy::float_cmp, reason = "groups are labels compared exactly")]
fn neutralize(x: &[f64], g: &[f64], out: &mut [f64], scratch: &mut Scratch) {
    let order = &mut scratch.order;
    order.clear();
    order.extend((0..x.len()).filter(|&i| !g[i].is_nan()));
    // Stable by symbol inside a group, so the group's sum runs in symbol order.
    order.sort_by(|&a, &b| g[a].total_cmp(&g[b]).then(a.cmp(&b)));
    out.fill(f64::NAN);
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && g[order[end]] == g[order[start]] {
            end += 1;
        }
        let members = &order[start..end];
        let (mut n, mut sum) = (0usize, 0.0);
        for &i in members.iter().filter(|&&i| !x[i].is_nan()) {
            n += 1;
            sum += x[i];
        }
        let mean = if n == 0 { f64::NAN } else { sum / n as f64 };
        for &i in members {
            out[i] = x[i] - mean;
        }
        start = end;
    }
}

/// `y - (a + b·x)` with `a`, `b` the least-squares fit over the symbols
/// where both are present.
fn residual(y: &[f64], x: &[f64], out: &mut [f64]) {
    let both = || (0..y.len()).filter(|&i| !y[i].is_nan() && !x[i].is_nan());
    let (mut n, mut sx, mut sy) = (0usize, 0.0, 0.0);
    for i in both() {
        n += 1;
        sx += x[i];
        sy += y[i];
    }
    out.fill(f64::NAN);
    if n < 2 {
        return;
    }
    let (mx, my) = (sx / n as f64, sy / n as f64);
    let (mut sxx, mut sxy) = (0.0, 0.0);
    for i in both() {
        let dx = x[i] - mx;
        sxx += dx * dx;
        sxy += dx * (y[i] - my);
    }
    let slope = sxy / sxx;
    for i in both() {
        out[i] = y[i] - my - slope * (x[i] - mx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(op: Cross, x: &[f64], y: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; x.len()];
        row(op, x, y, &mut out, &mut Scratch::default());
        out
    }

    fn same(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!(
                (x.is_nan() && y.is_nan()) || (x - y).abs() < 1e-12,
                "{a:?} != {b:?}"
            );
        }
    }

    const N: f64 = f64::NAN;

    #[test]
    fn rank_averages_ties_and_skips_nan() {
        // pandas: Series([3, 1, nan, 3, 2]).rank(pct=True) = [0.875, 0.25, nan, 0.875, 0.5]
        same(
            &run(Cross::Rank, &[3.0, 1.0, N, 3.0, 2.0], &[]),
            &[0.875, 0.25, N, 0.875, 0.5],
        );
        same(&run(Cross::Rank, &[N, N], &[]), &[N, N]);
    }

    #[test]
    fn zscore_demean_scale() {
        // mean 2, sample std 1.
        same(
            &run(Cross::ZScore, &[1.0, 2.0, 3.0, N], &[]),
            &[-1.0, 0.0, 1.0, N],
        );
        same(&run(Cross::ZScore, &[5.0, N], &[]), &[N, N]);
        same(
            &run(Cross::Demean, &[1.0, 2.0, 6.0], &[]),
            &[-2.0, -1.0, 3.0],
        );
        same(
            &run(Cross::Scale(2.0_f64.to_bits()), &[1.0, -3.0, N], &[]),
            &[0.5, -1.5, N],
        );
    }

    #[test]
    fn neutralize_by_group() {
        // Groups 7: {1, 3} mean 2; 9: {10, nan, 20} mean 15; group nan: nan.
        same(
            &run(
                Cross::Neutralize,
                &[1.0, 10.0, 3.0, N, 20.0, 5.0],
                &[7.0, 9.0, 7.0, 9.0, 9.0, N],
            ),
            &[-1.0, -5.0, 1.0, N, 5.0, N],
        );
    }

    #[test]
    fn residual_of_a_line_is_zero_and_of_a_point_off_it_is_its_distance() {
        same(
            &run(Cross::Resi, &[3.0, 5.0, 7.0, N], &[1.0, 2.0, 3.0, 4.0]),
            &[0.0, 0.0, 0.0, N],
        );
        // y = [0, 0, 3] on x = [0, 1, 2]: slope 1.5, intercept -0.5.
        same(
            &run(Cross::Resi, &[0.0, 0.0, 3.0], &[0.0, 1.0, 2.0]),
            &[0.5, -1.0, 0.5],
        );
    }
}
