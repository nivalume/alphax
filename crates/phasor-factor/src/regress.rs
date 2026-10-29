//! Cross-sectional regression: neutralization and Fama–MacBeth.
//!
//! Both run one ordinary-least-squares fit per bar, over the symbols that are
//! finite on every series that bar uses. An intercept is removed by demeaning
//! (Frisch–Waugh–Lovell); group labels, when given, are removed the same way
//! inside each group, which is the within transformation for group fixed
//! effects. Slopes come from a Cholesky factorization of `XᵀX`.
//!
//! Sums run in symbol order. Rows are split across threads and never combined,
//! so one thread and many write the same bits.

use phasor_core::error::{Error, Result};

use crate::panel::{Panel, map_chunks};

/// A column that still has cross-sectional variation after demeaning.
///
/// Demeaning a constant leaves only rounding dust, on the order of
/// `epsilon * |level|`. Variation below `1e-12` of the column's magnitude is
/// that dust: the column is in the span of the intercept (or of the group
/// means) and does not identify a slope.
fn varies(ss: f64, max_abs: f64, n: usize) -> bool {
    let tol = max_abs * 1e-12;
    ss > (n as f64) * tol * tol
}

/// Solves `A beta = b` for symmetric positive-definite `A` (`k x k`, row-major)
/// by a Cholesky factor `L Lᵀ`. Returns false when a pivot is not positive
/// relative to the largest diagonal, which is the singular case.
fn solve_spd(a: &[f64], b: &[f64], chol: &mut [f64], beta: &mut [f64]) -> bool {
    let k = beta.len();
    if k == 0 {
        return true;
    }
    chol.fill(0.0);
    let mut max_diag = 0.0_f64;
    for i in 0..k {
        max_diag = max_diag.max(a[i * k + i]);
    }
    let tol = max_diag * 1e-12 * (k as f64);
    for i in 0..k {
        for j in 0..i {
            let mut s = a[i * k + j];
            for p in 0..j {
                s -= chol[i * k + p] * chol[j * k + p];
            }
            chol[i * k + j] = s / chol[j * k + j];
        }
        let mut s = a[i * k + i];
        for p in 0..i {
            let v = chol[i * k + p];
            s -= v * v;
        }
        // A non-positive or non-finite pivot means the system is singular.
        if s > tol {
            chol[i * k + i] = s.sqrt();
        } else {
            return false;
        }
    }
    for i in 0..k {
        let mut s = b[i];
        for p in 0..i {
            s -= chol[i * k + p] * beta[p];
        }
        beta[i] = s / chol[i * k + i];
    }
    for i in (0..k).rev() {
        let mut s = beta[i];
        for p in (i + 1)..k {
            s -= chol[p * k + i] * beta[p];
        }
        beta[i] = s / chol[i * k + i];
    }
    true
}

/// Scratch reused across the rows of one thread.
#[derive(Debug, Default)]
struct Scratch {
    valid: Vec<usize>,
    y: Vec<f64>,
    /// Regressors of the valid symbols, row-major (`n * k`).
    x: Vec<f64>,
    labels: Vec<f64>,
    order: Vec<usize>,
    max_abs: Vec<f64>,
    ss: Vec<f64>,
    sum_x: Vec<f64>,
    keep: Vec<usize>,
    xtx: Vec<f64>,
    xty: Vec<f64>,
    chol: Vec<f64>,
    beta: Vec<f64>,
    one: Vec<f64>,
}

impl Scratch {
    fn collect(
        &mut self,
        t: usize,
        y: &Panel<'_>,
        groups: Option<&Panel<'_>>,
        regressors: &[Panel<'_>],
    ) {
        self.valid.clear();
        self.y.clear();
        self.x.clear();
        self.labels.clear();
        for c in 0..y.n_cols() {
            let yv = y.col(c)[t];
            if !yv.is_finite() {
                continue;
            }
            let label = if let Some(g) = groups {
                let gv = g.col(c)[t];
                if !gv.is_finite() {
                    continue;
                }
                Some(gv)
            } else {
                None
            };
            self.one.clear();
            let mut ok = true;
            for reg in regressors {
                let v = reg.col(c)[t];
                if !v.is_finite() {
                    ok = false;
                    break;
                }
                self.one.push(v);
            }
            if !ok {
                continue;
            }
            self.valid.push(c);
            self.y.push(yv);
            self.x.append(&mut self.one);
            if let Some(gv) = label {
                self.labels.push(gv);
            }
        }
    }

    fn column_max_abs(&mut self, k: usize) {
        let n = self.y.len();
        self.max_abs.clear();
        self.max_abs.resize(k, 0.0);
        for j in 0..k {
            let mut m = 0.0_f64;
            for i in 0..n {
                m = m.max(self.x[i * k + j].abs());
            }
            self.max_abs[j] = m;
        }
    }

    fn column_ss(&mut self, k: usize) {
        let n = self.y.len();
        self.ss.clear();
        self.ss.resize(k, 0.0);
        for j in 0..k {
            let mut ss = 0.0;
            for i in 0..n {
                let v = self.x[i * k + j];
                ss += v * v;
            }
            self.ss[j] = ss;
        }
    }

    /// Sums of the raw (not yet demeaned) regressors, and the sum of `y`.
    fn sums(&mut self, k: usize) -> f64 {
        let n = self.y.len();
        self.sum_x.clear();
        self.sum_x.resize(k, 0.0);
        for j in 0..k {
            let mut s = 0.0;
            for i in 0..n {
                s += self.x[i * k + j];
            }
            self.sum_x[j] = s;
        }
        self.y.iter().sum()
    }

    fn demean_global(&mut self, k: usize) {
        let n = self.y.len();
        if n == 0 {
            return;
        }
        let inv = 1.0 / n as f64;
        let mean_y = self.y.iter().sum::<f64>() * inv;
        for v in &mut self.y {
            *v -= mean_y;
        }
        for j in 0..k {
            let mut sum = 0.0;
            for i in 0..n {
                sum += self.x[i * k + j];
            }
            let mean = sum * inv;
            for i in 0..n {
                self.x[i * k + j] -= mean;
            }
        }
    }

    /// Subtracts each group's mean from `y` and from every regressor. Group
    /// labels are discrete codes compared exactly; `-0.0` and `0.0` are one
    /// group. Members of a group are summed in symbol order.
    #[allow(clippy::float_cmp, reason = "group labels are discrete codes")]
    fn demean_groups(&mut self, k: usize) {
        let n = self.y.len();
        self.order.clear();
        self.order.extend(0..n);
        // A copy, so the sort can borrow the codes while it reorders indices.
        let labels = self.labels.clone();
        self.order
            .sort_by(|&a, &b| labels[a].total_cmp(&labels[b]).then(a.cmp(&b)));
        let mut start = 0;
        while start < n {
            let mut end = start + 1;
            while end < n && labels[self.order[end]] == labels[self.order[start]] {
                end += 1;
            }
            let inv = 1.0 / (end - start) as f64;
            let mut sum_y = 0.0;
            for i in start..end {
                sum_y += self.y[self.order[i]];
            }
            let mean_y = sum_y * inv;
            for i in start..end {
                let idx = self.order[i];
                self.y[idx] -= mean_y;
            }
            for j in 0..k {
                let mut sum = 0.0;
                for i in start..end {
                    sum += self.x[self.order[i] * k + j];
                }
                let mean = sum * inv;
                for i in start..end {
                    let idx = self.order[i];
                    self.x[idx * k + j] -= mean;
                }
            }
            start = end;
        }
    }

    fn demean(&mut self, k: usize, grouped: bool) {
        if grouped {
            self.demean_groups(k);
        } else if k > 0 {
            self.demean_global(k);
        }
    }

    /// Normal equations of the columns in `keep`.
    fn normal_eq(&mut self, k_full: usize) {
        let n = self.y.len();
        let kr = self.keep.len();
        self.xtx.clear();
        self.xtx.resize(kr * kr, 0.0);
        self.xty.clear();
        self.xty.resize(kr, 0.0);
        for a in 0..kr {
            let ja = self.keep[a];
            let mut sy = 0.0;
            for i in 0..n {
                sy += self.x[i * k_full + ja] * self.y[i];
            }
            self.xty[a] = sy;
            for b in 0..=a {
                let jb = self.keep[b];
                let mut s = 0.0;
                for i in 0..n {
                    s += self.x[i * k_full + ja] * self.x[i * k_full + jb];
                }
                self.xtx[a * kr + b] = s;
                self.xtx[b * kr + a] = s;
            }
        }
    }

    /// `y ← y − X beta` for the columns in `keep`, and the residual sum of squares.
    fn apply(&mut self, k_full: usize) -> f64 {
        let kr = self.keep.len();
        self.chol.resize(kr * kr, 0.0);
        self.beta.resize(kr, 0.0);
        if !solve_spd(&self.xtx, &self.xty, &mut self.chol, &mut self.beta) {
            return f64::NAN;
        }
        let mut ss = 0.0;
        let n = self.y.len();
        for i in 0..n {
            let mut y_hat = 0.0;
            let row = i * k_full;
            for a in 0..kr {
                y_hat += self.x[row + self.keep[a]] * self.beta[a];
            }
            let r = self.y[i] - y_hat;
            self.y[i] = r;
            ss += r * r;
        }
        ss
    }

    fn scatter(&self, row: &mut [f64]) {
        row.fill(f64::NAN);
        for (&c, &v) in self.valid.iter().zip(&self.y) {
            row[c] = v;
        }
    }
}

fn align(
    y: &Panel<'_>,
    groups: Option<&Panel<'_>>,
    others: &[Panel<'_>],
    what: &str,
) -> Result<()> {
    if let Some(g) = groups {
        y.same_shape(g, "groups")?;
    }
    for panel in others {
        y.same_shape(panel, what)?;
    }
    Ok(())
}

/// Residuals of `y` after removing group means and the span of `exposures`.
///
/// On each row the symbols where `y`, `groups` (if given) and every exposure
/// are finite are kept. With `groups`, `y` and the exposures are demeaned
/// inside each discrete group label; without groups but with exposures, they
/// are demeaned across the row, which is the intercept. The residuals are
/// `y − X β` from a Cholesky solve of the demeaned system.
///
/// An exposure that is constant on the row (or constant inside every group)
/// is absorbed by that demeaning and dropped. If every exposure drops out,
/// the residual is the demeaned factor. If the remaining columns are linearly
/// dependent, or fewer than `K + 1` symbols are valid (`K` = number of
/// exposures), the whole row is `NaN`: the slope is not identified, and the
/// spec reports no residual rather than one particular solution.
///
/// With neither groups nor exposures the factor is returned unchanged, apart
/// from non-finite entries staying `NaN`. The result is column-major
/// `(n_rows, n_cols)`. It does not read any other row.
///
/// # Errors
///
/// [`Error::Invariant`] if a panel's shape differs from `y`.
pub fn neutralize(
    y: &Panel<'_>,
    groups: Option<&Panel<'_>>,
    exposures: &[Panel<'_>],
    threads: usize,
) -> Result<Vec<f64>> {
    align(y, groups, exposures, "exposure")?;
    let (n_rows, n_cols) = (y.n_rows(), y.n_cols());
    let k = exposures.len();
    let grouped = groups.is_some();
    let parts = map_chunks(n_rows, threads, |rows| {
        let len = rows.len();
        let mut chunk = vec![f64::NAN; n_cols * len];
        let mut scratch = Scratch::default();
        let mut row = vec![f64::NAN; n_cols];
        for (local, t) in (rows.start..rows.end).enumerate() {
            neutralize_row(t, y, groups, exposures, k, grouped, &mut scratch, &mut row);
            for c in 0..n_cols {
                chunk[c * len + local] = row[c];
            }
        }
        (rows, chunk)
    });
    let mut out = vec![f64::NAN; n_rows * n_cols];
    for (rows, chunk) in parts {
        let len = rows.len();
        for c in 0..n_cols {
            let dst = c * n_rows + rows.start;
            let src = c * len;
            out[dst..dst + len].copy_from_slice(&chunk[src..src + len]);
        }
    }
    Ok(out)
}

fn neutralize_row(
    t: usize,
    y: &Panel<'_>,
    groups: Option<&Panel<'_>>,
    exposures: &[Panel<'_>],
    k: usize,
    grouped: bool,
    scratch: &mut Scratch,
    row: &mut [f64],
) {
    row.fill(f64::NAN);
    scratch.collect(t, y, groups, exposures);
    let n = scratch.y.len();
    if n == 0 || (k > 0 && n <= k) {
        return;
    }
    scratch.column_max_abs(k);
    scratch.demean(k, grouped);
    scratch.keep.clear();
    if k > 0 {
        scratch.column_ss(k);
        for j in 0..k {
            if varies(scratch.ss[j], scratch.max_abs[j], n) {
                scratch.keep.push(j);
            }
        }
        if !scratch.keep.is_empty() {
            scratch.normal_eq(k);
            if scratch.apply(k).is_nan() {
                return;
            }
        }
    }
    scratch.scatter(row);
}

/// Coefficients and R² of one Fama–MacBeth cross-section, for every bar.
///
/// `coef` is column-major `(n_rows, names.len())`. Column 0 is the intercept.
/// The next columns are `factor_names`, then one `control_{i}` per control.
/// `r2` has one entry per bar.
///
/// On bar `t` the regression is `fwd ~ 1 + factors + controls`, over the
/// symbols that are finite on all of those series (and on `groups`, if
/// given). Group labels add a fixed effect per distinct code, removed by
/// within-group demeaning; the reported intercept is then the size-weighted
/// average of the group intercepts. Without groups the intercept is the
/// ordinary one.
///
/// A bar whose valid count is not strictly greater than the number of slopes,
/// or whose demeaned regressors are constant or linearly dependent, is `NaN`
/// in every coefficient and in `r2`. Slopes are not uniquely identified
/// there, so the bar is left out of the time-series mean rather than filled
/// with one of the solutions.
#[derive(Debug, Clone, PartialEq)]
pub struct FmResult {
    /// Number of bars.
    pub n_rows: usize,
    /// Names of the coefficient columns, intercept first.
    pub names: Vec<String>,
    /// Column-major coefficients, one column per name.
    pub coef: Vec<f64>,
    /// R² of each bar, `NaN` when the total sum of squares is 0 or the fit failed.
    pub r2: Vec<f64>,
}

fn coefficient_names(factor_names: &[String], n_controls: usize) -> Vec<String> {
    let mut names = Vec::with_capacity(1 + factor_names.len() + n_controls);
    names.push("intercept".to_string());
    names.extend(factor_names.iter().cloned());
    for i in 0..n_controls {
        names.push(format!("control_{i}"));
    }
    names
}

/// Fama–MacBeth slopes: a cross-sectional regression on every bar.
///
/// The panels are whatever the caller passes. When `fwd` is a forward return
/// it reads the future, and these coefficients are a description of the
/// factor, not a signal.
///
/// # Errors
///
/// [`Error::Invariant`] if `factor_names` does not have one name per factor,
/// or a panel's shape differs from `fwd`.
pub fn fama_macbeth(
    fwd: &Panel<'_>,
    factors: &[Panel<'_>],
    factor_names: &[String],
    groups: Option<&Panel<'_>>,
    controls: &[Panel<'_>],
    threads: usize,
) -> Result<FmResult> {
    if factor_names.len() != factors.len() {
        return Err(Error::Invariant(format!(
            "factor_names has {} names, expected {}",
            factor_names.len(),
            factors.len()
        )));
    }
    align(fwd, groups, factors, "factor")?;
    align(fwd, None, controls, "control")?;
    let n_rows = fwd.n_rows();
    let mut regressors = Vec::with_capacity(factors.len() + controls.len());
    regressors.extend_from_slice(factors);
    regressors.extend_from_slice(controls);
    let k = regressors.len();
    let n_coefs = 1 + k;
    let names = coefficient_names(factor_names, controls.len());
    let grouped = groups.is_some();

    let parts = map_chunks(n_rows, threads, |rows| {
        let len = rows.len();
        let mut coef = vec![f64::NAN; n_coefs * len];
        let mut r2 = vec![f64::NAN; len];
        let mut scratch = Scratch::default();
        for (local, t) in (rows.start..rows.end).enumerate() {
            fama_row(
                t,
                fwd,
                groups,
                &regressors,
                k,
                grouped,
                &mut scratch,
                &mut coef,
                &mut r2,
                len,
                local,
            );
        }
        (rows, coef, r2)
    });

    let mut coef = vec![f64::NAN; n_coefs * n_rows];
    let mut r2 = vec![f64::NAN; n_rows];
    for (rows, coef_chunk, r2_chunk) in parts {
        let len = rows.len();
        r2[rows.start..rows.end].copy_from_slice(&r2_chunk);
        for j in 0..n_coefs {
            let dst = j * n_rows + rows.start;
            let src = j * len;
            coef[dst..dst + len].copy_from_slice(&coef_chunk[src..src + len]);
        }
    }
    Ok(FmResult {
        n_rows,
        names,
        coef,
        r2,
    })
}

fn fama_row(
    t: usize,
    fwd: &Panel<'_>,
    groups: Option<&Panel<'_>>,
    regressors: &[Panel<'_>],
    k: usize,
    grouped: bool,
    scratch: &mut Scratch,
    coef: &mut [f64],
    r2: &mut [f64],
    len: usize,
    local: usize,
) {
    scratch.collect(t, fwd, groups, regressors);
    let n = scratch.y.len();
    if n == 0 || (k > 0 && n <= k) {
        return;
    }
    let sum_y = scratch.sums(k);
    let mean_y = sum_y / n as f64;
    let ss_tot = ss_around(&scratch.y, mean_y);
    if k == 0 && !grouped {
        coef[local] = mean_y;
        r2[local] = if ss_tot > 0.0 { 0.0 } else { f64::NAN };
        return;
    }
    scratch.column_max_abs(k);
    scratch.demean(k, grouped);
    if k > 0 {
        scratch.column_ss(k);
        for j in 0..k {
            if !varies(scratch.ss[j], scratch.max_abs[j], n) {
                return;
            }
        }
        scratch.keep.clear();
        scratch.keep.extend(0..k);
        scratch.normal_eq(k);
        let ss_res = scratch.apply(k);
        if ss_res.is_nan() {
            return;
        }
        coef[local] = grand_intercept(sum_y, &scratch.sum_x, &scratch.beta, n);
        for j in 0..k {
            coef[(j + 1) * len + local] = scratch.beta[j];
        }
        r2[local] = r_squared(ss_res, ss_tot);
    } else {
        let ss_res: f64 = scratch.y.iter().map(|v| v * v).sum();
        coef[local] = mean_y;
        r2[local] = r_squared(ss_res, ss_tot);
    }
}

fn ss_around(y: &[f64], mean: f64) -> f64 {
    let mut s = 0.0;
    for &v in y {
        let d = v - mean;
        s += d * d;
    }
    s
}

fn grand_intercept(sum_y: f64, sum_x: &[f64], beta: &[f64], n: usize) -> f64 {
    let inv = 1.0 / n as f64;
    let mut intercept = sum_y * inv;
    for (&s, &b) in sum_x.iter().zip(beta) {
        intercept -= (s * inv) * b;
    }
    intercept
}

fn r_squared(ss_res: f64, ss_tot: f64) -> f64 {
    if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        f64::NAN
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
            assert_eq!(row.len(), n_cols);
            for (c, &v) in row.iter().enumerate() {
                data[c * n_rows + t] = v;
            }
        }
        (data, n_rows, n_cols)
    }

    fn view(data: &[f64], n_rows: usize, n_cols: usize) -> Panel<'_> {
        Panel::new(data, n_rows, n_cols).unwrap()
    }

    fn same(got: &[f64], expect: &[f64]) {
        assert_eq!(
            got.len(),
            expect.len(),
            "len {} != {}",
            got.len(),
            expect.len()
        );
        for (i, (&a, &b)) in got.iter().zip(expect).enumerate() {
            assert!(
                (a.is_nan() && b.is_nan()) || is_close(a, b),
                "index {i}: {a} != {b}\n got {got:?}\n want {expect:?}"
            );
        }
    }

    fn row_of(data: &[f64], n_rows: usize, t: usize, n_cols: usize) -> Vec<f64> {
        (0..n_cols).map(|c| data[c * n_rows + t]).collect()
    }

    #[test]
    fn constant_and_zero_exposure_demean_the_row() {
        let (y, nr, nc) = col_major(&[vec![1.0, 2.0, 3.0, 6.0], vec![10.0, 0.0, 20.0, 10.0]]);
        let constant = vec![0.1; nr * nc];
        let zeros = vec![0.0; nr * nc];
        let yv = view(&y, nr, nc);
        for exposure in [&constant, &zeros] {
            let out = neutralize(&yv, None, &[view(exposure, nr, nc)], 1).unwrap();
            same(&row_of(&out, nr, 0, nc), &[-2.0, -1.0, 0.0, 3.0]);
            same(&row_of(&out, nr, 1, nc), &[0.0, -10.0, 10.0, 0.0]);
        }
    }

    #[test]
    fn no_exposure_copies_y_and_groups_demean() {
        let (y, nr, nc) = col_major(&[vec![1.0, 10.0, 3.0, f64::NAN, 20.0, 5.0]]);
        let (g, _, _) = col_major(&[vec![7.0, 9.0, 7.0, 9.0, 9.0, f64::NAN]]);
        let yv = view(&y, nr, nc);
        let copied = neutralize(&yv, None, &[], 1).unwrap();
        same(&copied, &y);
        let out = neutralize(&yv, Some(&view(&g, nr, nc)), &[], 2).unwrap();
        same(&out, &[-1.0, -5.0, 1.0, f64::NAN, 5.0, f64::NAN]);
    }

    #[test]
    fn residuals_of_a_line_are_zero_and_orthogonal() {
        // y = 3.5 + 1.4 x on x = 1, 2, 3, 4 is the textbook fit of [6, 5, 7, 10].
        let (y, nr, nc) = col_major(&[vec![6.0, 5.0, 7.0, 10.0]]);
        let (x, _, _) = col_major(&[vec![1.0, 2.0, 3.0, 4.0]]);
        let out = neutralize(&view(&y, nr, nc), None, &[view(&x, nr, nc)], 1).unwrap();
        same(&out, &[1.1, -1.3, -0.7, 0.9]);

        let x1_row = [1.0, 2.0, 3.0, 4.0, 0.0];
        let x2_row = [1.0, 0.0, 2.0, -1.0, 1.5];
        let exact_row: Vec<f64> = x1_row
            .iter()
            .zip(x2_row)
            .map(|(a, b)| 0.5 + 2.0 * a - 3.0 * b)
            .collect();
        let (exact, _, _) = col_major(&[exact_row]);
        let (x1, _, _) = col_major(&[x1_row.to_vec()]);
        let (x2, _, _) = col_major(&[x2_row.to_vec()]);
        let n = exact.len();
        let resid = neutralize(
            &view(&exact, 1, n),
            None,
            &[view(&x1, 1, n), view(&x2, 1, n)],
            1,
        )
        .unwrap();
        same(&resid, &vec![0.0; n]);
    }

    #[test]
    fn group_exposure_fit_has_zero_residual_and_zero_group_means() {
        // Two groups. y = group effect + 2 x. Neutralizing on the group and
        // on x removes both, so the residual is 0 and each group mean is 0.
        let (x, nr, nc) = col_major(&[vec![1.0, 2.0, 4.0, 1.0, 3.0, 0.0]]);
        let (g, _, _) = col_major(&[vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]]);
        let mut y = vec![0.0; nr * nc];
        for c in 0..nc {
            let effect = if c < 3 { 10.0 } else { -4.0 };
            y[c] = effect + 2.0 * x[c];
        }
        let out = neutralize(
            &view(&y, nr, nc),
            Some(&view(&g, nr, nc)),
            &[view(&x, nr, nc)],
            3,
        )
        .unwrap();
        same(&out, &vec![0.0; nc]);
    }

    #[test]
    fn singular_or_short_rows_are_nan() {
        let (y, nr, nc) = col_major(&[vec![1.0, 2.0, 4.0, 8.0]]);
        let (x, _, _) = col_major(&[vec![1.0, 2.0, 3.0, 4.0]]);
        let mut twice = x.clone();
        for v in &mut twice {
            *v *= 2.0;
        }
        let yv = view(&y, nr, nc);
        let out = neutralize(&yv, None, &[view(&x, nr, nc), view(&twice, nr, nc)], 1).unwrap();
        assert!(out.iter().all(|v| v.is_nan()));

        let short_y = view(&y[..3], 1, 3);
        let short_x1 = view(&x[..3], 1, 3);
        let short_x2 = view(&twice[..3], 1, 3);
        let (z, _, _) = col_major(&[vec![0.0, 1.0, 2.0]]);
        // 3 symbols, 3 exposures: valid count <= K.
        let out = neutralize(&short_y, None, &[short_x1, short_x2, view(&z, 1, 3)], 1).unwrap();
        assert!(out.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn fama_macbeth_recovers_a_known_line() {
        // y = 3.5 + 1.4 x, x = 1..4, y = [6, 5, 7, 10]. R² = 0.7.
        let (y, nr, nc) = col_major(&[vec![6.0, 5.0, 7.0, 10.0]]);
        let (x, _, _) = col_major(&[vec![1.0, 2.0, 3.0, 4.0]]);
        let names = vec!["x".to_string()];
        let fit =
            fama_macbeth(&view(&y, nr, nc), &[view(&x, nr, nc)], &names, None, &[], 1).unwrap();
        assert_eq!(fit.names, vec!["intercept".to_string(), "x".to_string()]);
        assert!(is_close(fit.coef[0], 3.5));
        assert!(is_close(fit.coef[nr], 1.4));
        assert!(is_close(fit.r2[0], 0.7));
    }

    #[test]
    fn fama_macbeth_recovers_two_slopes_a_control_and_group_effects() {
        let f1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.0];
        let f2 = vec![1.0, 0.0, 1.0, 0.0, 1.0, 2.0];
        let control = vec![0.0, 1.0, -1.0, 2.0, 0.5, -0.5];
        let mut fwd = vec![0.0; 6];
        for i in 0..6 {
            fwd[i] = 0.25 + 2.0 * f1[i] - 1.5 * f2[i] + 0.5 * control[i];
        }
        let names = vec!["f1".to_string(), "f2".to_string()];
        let fit = fama_macbeth(
            &view(&fwd, 1, 6),
            &[view(&f1, 1, 6), view(&f2, 1, 6)],
            &names,
            None,
            &[view(&control, 1, 6)],
            1,
        )
        .unwrap();
        assert_eq!(
            fit.names,
            [
                "intercept".to_string(),
                "f1".to_string(),
                "f2".to_string(),
                "control_0".to_string()
            ]
        );
        same(&fit.coef, &[0.25, 2.0, -1.5, 0.5]);
        assert!(is_close(fit.r2[0], 1.0));

        // y = group effect + 2 f1. The within slope is 2. The reported
        // intercept is the size-weighted average of the group intercepts,
        // (3 * 10 + 3 * -4) / 6 = 3.
        let groups = vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
        let mut pure = vec![0.0; 6];
        for i in 0..6 {
            let effect = if i < 3 { 10.0 } else { -4.0 };
            pure[i] = effect + 2.0 * f1[i];
        }
        let fit = fama_macbeth(
            &view(&pure, 1, 6),
            &[view(&f1, 1, 6)],
            &["f1".to_string()],
            Some(&view(&groups, 1, 6)),
            &[],
            1,
        )
        .unwrap();
        assert!(is_close(fit.coef[0], 3.0)); // (3*10 + 3*(-4)) / 6
        assert!(is_close(fit.coef[1], 2.0));
        assert!(is_close(fit.r2[0], 1.0));
    }

    #[test]
    fn threads_match_bit_for_bit() {
        let n_rows = 24;
        let n_cols = 12;
        let mut y = vec![0.0; n_rows * n_cols];
        let mut x = vec![0.0; n_rows * n_cols];
        let mut g = vec![0.0; n_rows * n_cols];
        let mut fwd = vec![0.0; n_rows * n_cols];
        for c in 0..n_cols {
            for t in 0..n_rows {
                let i = c * n_rows + t;
                if (t + c) % 11 == 0 {
                    y[i] = f64::NAN;
                    x[i] = 1.0;
                } else {
                    y[i] = ((t * 3 + c * 5) % 17) as f64 - 4.0;
                    x[i] = ((t * 2 + c) % 9) as f64;
                }
                g[i] = (c % 3) as f64;
                fwd[i] = 0.3 + 1.5 * x[i] + 0.2 * (c % 3) as f64;
                if (t * 3 + c) % 13 == 0 {
                    fwd[i] = f64::NAN;
                }
            }
        }
        let yv = view(&y, n_rows, n_cols);
        let xv = view(&x, n_rows, n_cols);
        let gv = view(&g, n_rows, n_cols);
        let fv = view(&fwd, n_rows, n_cols);
        let names = vec!["x".to_string()];
        let base_n = neutralize(&yv, Some(&gv), &[xv], 1).unwrap();
        let base_f = fama_macbeth(&fv, &[xv], &names, Some(&gv), &[], 1).unwrap();
        for threads in [2, 3, 8] {
            let n = neutralize(&yv, Some(&gv), &[xv], threads).unwrap();
            assert_eq!(
                base_n.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                n.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            let f = fama_macbeth(&fv, &[xv], &names, Some(&gv), &[], threads).unwrap();
            assert_eq!(
                base_f.coef.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                f.coef.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(
                base_f.r2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                f.r2.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn names_and_shapes_are_checked() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let y = view(&data, 2, 2);
        let bad = view(&data, 4, 1);
        assert!(neutralize(&y, None, &[bad], 1).is_err());
        assert!(fama_macbeth(&y, &[y], &["a".to_string(), "b".to_string()], None, &[], 1).is_err());
    }
}
