//! Factor evaluation: forward returns, information coefficients, quantiles
//! and turnover.
//!
//! The Python module `phasor.factor_eval` wraps these. Forward returns and
//! the coefficients scored against them read bars after the one they are
//! stamped on, on purpose — that is how a factor is judged, and none of that
//! may be fed back into a simulation as a signal. Quantile labels, turnover
//! and rank autocorrelation use the factor's own bars.
//!
//! Arrays cross as they do everywhere else: F-ordered in, through
//! [`columns_of`], and F-ordered out, through [`columns_out`]. The
//! cross-sectional IC is the exception on the way out. Its buffer is
//! `[factor][horizon][row]`, last index fastest, so it is reshaped C-ordered
//! into `(n_factors, n_horizons, n_rows)`.
//!
//! Neutralization, Fama–MacBeth and the Spearman correlation matrices are
//! the same shape convention: column-major in, column-major out.
//!
//! Placebos return `(observed, p_value, null_distribution)`. The null is one
//! mean rank IC per draw. A cross-sectional draw shuffles each bar; a
//! time-series draw rotates the forward return. The thread count is not part
//! of the result.

use numpy::npyffi::NPY_ORDER;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
};
use phasor_factor::{Method, Panel};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::array::{columns_of, columns_out, to_py_err};

/// A column-major `(n_rows, n_cols)` result.
type Columns<'py> = Bound<'py, PyArray2<f64>>;

/// One cross-sectional IC, shaped `(n_factors, n_horizons, n_rows)`.
type IcCube<'py> = Bound<'py, PyArray3<f64>>;

/// Pearson and Spearman cubes from one cross-sectional pass.
type CsOut<'py> = (IcCube<'py>, IcCube<'py>);

/// Block ids, and the IC of each block by symbol.
type BlockOut<'py> = (Bound<'py, PyArray1<i64>>, Columns<'py>);

/// Mean, std, ICIR, t, Newey–West t, its p-value, hit rate, and `n`.
type SummaryOut = (f64, f64, f64, f64, f64, f64, f64, usize);

/// `(n_rows, n_cols)` view of one F-ordered array.
fn panel<'a>(array: &'a PyReadonlyArray2<'a, f64>) -> PyResult<Panel<'a>> {
    let (data, n_rows, n_cols) = columns_of(array)?;
    Panel::new(data, n_rows, n_cols).map_err(to_py_err)
}

fn panels<'a>(arrays: &'a [PyReadonlyArray2<'a, f64>]) -> PyResult<Vec<Panel<'a>>> {
    arrays.iter().map(panel).collect()
}

/// `"pearson"`, or `"spearman"` / `"rank"`.
fn parse_method(name: &str) -> PyResult<Method> {
    match name {
        "pearson" => Ok(Method::Pearson),
        "spearman" | "rank" => Ok(Method::Spearman),
        _ => Err(PyValueError::new_err(format!(
            "method must be \"pearson\", \"spearman\" or \"rank\"; got {name:?}"
        ))),
    }
}

/// `[factor][horizon][row]` as `(n_factors, n_horizons, n_rows)`.
///
/// A 1-D buffer is both C- and F-contiguous, and NumPy's `'A'` order then
/// follows Fortran. The engine's layout is last-index-fastest, so the reshape
/// has to ask for C order by name.
fn cube<'py>(
    py: Python<'py>,
    data: Vec<f64>,
    n_factors: usize,
    n_horizons: usize,
    n_rows: usize,
) -> PyResult<IcCube<'py>> {
    data.into_pyarray(py)
        .reshape_with_order([n_factors, n_horizons, n_rows], NPY_ORDER::NPY_CORDER)
}

/// `close[t + lag + horizon] / close[t + lag] - 1`, column-major.
///
/// NaN where the window runs past the column or either close is not finite.
/// Reads `lag + horizon` bars ahead.
#[pyfunction]
pub fn factor_forward_returns<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'_, f64>,
    horizon: usize,
    lag: usize,
) -> PyResult<Columns<'py>> {
    let (data, n_rows, n_cols) = columns_of(&close)?;
    let view = Panel::new(data, n_rows, n_cols).map_err(to_py_err)?;
    let out = py
        .detach(|| phasor_factor::forward_returns(&view, horizon, lag))
        .map_err(to_py_err)?;
    columns_out(py, out, n_rows, n_cols)
}

/// Pearson and Spearman IC of every factor against every forward return, per row.
///
/// Each result is `(n_factors, n_horizons, n_rows)`. A row with fewer than
/// `min_names` finite pairs, or a constant side, is NaN. `threads` 0 uses
/// every core; the values do not depend on how many.
#[pyfunction]
pub fn factor_cs_ic<'py>(
    py: Python<'py>,
    factors: Vec<PyReadonlyArray2<'_, f64>>,
    forwards: Vec<PyReadonlyArray2<'_, f64>>,
    min_names: usize,
    threads: usize,
) -> PyResult<CsOut<'py>> {
    let factor_panels = panels(&factors)?;
    let forward_panels = panels(&forwards)?;
    let ic = py
        .detach(|| phasor_factor::cs_ic(&factor_panels, &forward_panels, min_names, threads))
        .map_err(to_py_err)?;
    Ok((
        cube(py, ic.pearson, ic.n_factors, ic.n_horizons, ic.n_rows)?,
        cube(py, ic.spearman, ic.n_factors, ic.n_horizons, ic.n_rows)?,
    ))
}

/// Time-series IC of each symbol over the whole sample. One value per column.
#[pyfunction]
pub fn factor_ts_ic<'py>(
    py: Python<'py>,
    factor: PyReadonlyArray2<'_, f64>,
    forward: PyReadonlyArray2<'_, f64>,
    method: &str,
    min_periods: usize,
    threads: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let how = parse_method(method)?;
    let x = panel(&factor)?;
    let y = panel(&forward)?;
    let out = py
        .detach(|| phasor_factor::ts_ic(&x, &y, how, min_periods, threads))
        .map_err(to_py_err)?;
    Ok(out.into_pyarray(py))
}

/// Time-series IC of each symbol over a trailing window. Shape of `factor`.
///
/// On row `t` the window is `t + 1 - window ..= t`. The value uses forward
/// returns that are only known later: it describes the factor, it is not a
/// signal.
#[pyfunction]
pub fn factor_ts_rolling_ic<'py>(
    py: Python<'py>,
    factor: PyReadonlyArray2<'_, f64>,
    forward: PyReadonlyArray2<'_, f64>,
    window: usize,
    min_periods: usize,
    method: &str,
    threads: usize,
) -> PyResult<Columns<'py>> {
    let how = parse_method(method)?;
    let (data, n_rows, n_cols) = columns_of(&factor)?;
    let x = Panel::new(data, n_rows, n_cols).map_err(to_py_err)?;
    let y = panel(&forward)?;
    let out = py
        .detach(|| phasor_factor::ts_rolling_ic(&x, &y, window, min_periods, how, threads))
        .map_err(to_py_err)?;
    columns_out(py, out, n_rows, n_cols)
}

/// Time-series IC of each symbol inside each block of rows.
///
/// `blocks[t]` names row `t`'s block. Rows of a block need not be adjacent.
/// Returns the distinct ids in ascending order and a column-major
/// `(n_blocks, n_cols)` matrix.
#[pyfunction]
pub fn factor_ts_block_ic<'py>(
    py: Python<'py>,
    factor: PyReadonlyArray2<'_, f64>,
    forward: PyReadonlyArray2<'_, f64>,
    blocks: PyReadonlyArray1<'_, i64>,
    min_periods: usize,
    method: &str,
    threads: usize,
) -> PyResult<BlockOut<'py>> {
    let how = parse_method(method)?;
    let x = panel(&factor)?;
    let y = panel(&forward)?;
    let ids = blocks.as_slice().map_err(|_| {
        PyValueError::new_err(
            "blocks is not contiguous; pass a contiguous int64 array of one id per row",
        )
    })?;
    let (ids, ic) = py
        .detach(|| phasor_factor::ts_block_ic(&x, &y, ids, min_periods, how, threads))
        .map_err(to_py_err)?;
    let n_blocks = ids.len();
    let n_cols = x.n_cols();
    Ok((ids.into_pyarray(py), columns_out(py, ic, n_blocks, n_cols)?))
}

/// Mean, standard deviation, ICIR, t, Newey–West t and its p-value, hit rate, and `n`.
///
/// NaN entries of `series` are dropped first. `lags` is the Newey–West lag.
#[pyfunction]
pub fn factor_summarize(
    py: Python<'_>,
    series: PyReadonlyArray1<'_, f64>,
    lags: usize,
) -> PyResult<SummaryOut> {
    let values = series.as_slice().map_err(|_| {
        PyValueError::new_err("series is not contiguous; pass a contiguous float64 array")
    })?;
    let summary = py.detach(|| phasor_factor::summarize(values, lags));
    Ok((
        summary.mean,
        summary.std,
        summary.icir,
        summary.t,
        summary.t_nw,
        summary.p_nw,
        summary.hit,
        summary.n,
    ))
}

/// Newey–West lag for `n` ICs of `horizon`-bar forward returns.
///
/// At least `horizon - 1`, and at least `floor(4 (n/100)^{2/9})`.
#[pyfunction]
pub fn factor_auto_lags(n: usize, horizon: usize) -> usize {
    phasor_factor::auto_lags(n, horizon)
}

/// Quantile label of every factor value, column-major `int16`.
///
/// `-1` where the value is missing or the row has fewer than `q` finite
/// values. Group `0` holds the smallest values.
#[pyfunction]
pub fn factor_quantile_labels<'py>(
    py: Python<'py>,
    factor: PyReadonlyArray2<'_, f64>,
    q: usize,
) -> PyResult<Bound<'py, PyArray2<i16>>> {
    let (data, n_rows, n_cols) = columns_of(&factor)?;
    let view = Panel::new(data, n_rows, n_cols).map_err(to_py_err)?;
    let out = py
        .detach(|| phasor_factor::quantile_labels(&view, q))
        .map_err(to_py_err)?;
    columns_out(py, out, n_rows, n_cols)
}

/// Mean forward return of each quantile group, column-major `(n_rows, q)`.
///
/// `weights`, when given, must have `fwd`'s shape. A non-finite weight is
/// skipped. A group with no names, or a weight sum that is not strictly
/// positive, is NaN.
#[pyfunction]
#[pyo3(signature = (labels, fwd, q, weights=None))]
pub fn factor_quantile_returns<'py>(
    py: Python<'py>,
    labels: PyReadonlyArray2<'_, i16>,
    fwd: PyReadonlyArray2<'_, f64>,
    q: usize,
    weights: Option<PyReadonlyArray2<'_, f64>>,
) -> PyResult<Columns<'py>> {
    let (label_data, label_rows, label_cols) = columns_of(&labels)?;
    let (fwd_data, n_rows, n_cols) = columns_of(&fwd)?;
    if (label_rows, label_cols) != (n_rows, n_cols) {
        return Err(PyValueError::new_err(format!(
            "labels are {label_rows}x{label_cols}, expected {n_rows}x{n_cols}"
        )));
    }
    let forward = Panel::new(fwd_data, n_rows, n_cols).map_err(to_py_err)?;
    let weight_panel = match &weights {
        None => None,
        Some(array) => {
            let (data, rows, cols) = columns_of(array)?;
            if (rows, cols) != (n_rows, n_cols) {
                return Err(PyValueError::new_err(format!(
                    "weights are {rows}x{cols}, expected {n_rows}x{n_cols}"
                )));
            }
            Some(Panel::new(data, rows, cols).map_err(to_py_err)?)
        }
    };
    let out = py
        .detach(|| phasor_factor::quantile_returns(label_data, &forward, q, weight_panel.as_ref()))
        .map_err(to_py_err)?;
    columns_out(py, out, n_rows, q)
}

/// Top quantile minus bottom quantile, one value per bar.
///
/// `q_returns` is `(n_rows, q)`, column-major.
#[pyfunction]
pub fn factor_long_short_returns<'py>(
    py: Python<'py>,
    q_returns: PyReadonlyArray2<'_, f64>,
    q: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (data, n_rows, n_cols) = columns_of(&q_returns)?;
    if n_cols != q {
        return Err(PyValueError::new_err(format!(
            "q_returns has {n_cols} columns, expected {q}"
        )));
    }
    let out = py
        .detach(|| phasor_factor::long_short_returns(data, n_rows, q))
        .map_err(to_py_err)?;
    Ok(out.into_pyarray(py))
}

/// Spearman correlation of `0 .. q-1` with each group's mean return.
///
/// NaN below two groups, when a mean is not finite, or when the means are
/// constant.
#[pyfunction]
pub fn factor_monotonicity(mean_q_returns: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let values = mean_q_returns.as_slice().map_err(|_| {
        PyValueError::new_err("mean_q_returns is not contiguous; pass a contiguous float64 array")
    })?;
    Ok(phasor_factor::monotonicity(values))
}

/// Turnover of each quantile group, column-major `(n_rows, q)`.
///
/// Bar `t < lag` is NaN. Otherwise group `g` is
/// `1 - |symbols in g now and lag bars ago| / |symbols in g now|`, or NaN
/// when the group is empty now.
#[pyfunction]
pub fn factor_quantile_turnover<'py>(
    py: Python<'py>,
    labels: PyReadonlyArray2<'_, i16>,
    q: usize,
    lag: usize,
) -> PyResult<Columns<'py>> {
    let (data, n_rows, n_cols) = columns_of(&labels)?;
    let out = py
        .detach(|| phasor_factor::quantile_turnover(data, n_rows, n_cols, q, lag))
        .map_err(to_py_err)?;
    columns_out(py, out, n_rows, q)
}

/// Spearman correlation of each bar's factor with the bar `lag` before it.
///
/// NaN before `lag`, and where fewer than two symbols are finite on both
/// bars or either side is constant.
#[pyfunction]
pub fn factor_rank_autocorr<'py>(
    py: Python<'py>,
    factor: PyReadonlyArray2<'_, f64>,
    lag: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (data, n_rows, n_cols) = columns_of(&factor)?;
    let view = Panel::new(data, n_rows, n_cols).map_err(to_py_err)?;
    let out = py
        .detach(|| phasor_factor::rank_autocorr(&view, lag))
        .map_err(to_py_err)?;
    Ok(out.into_pyarray(py))
}

/// Residuals of `y` after group means and exposures, column-major.
///
/// `groups` and each exposure must have `y`'s shape. A symbol that is not
/// finite on `y`, on `groups`, or on an exposure is `NaN`. `threads` 0 uses
/// every core; the values do not depend on how many.
#[pyfunction]
pub fn factor_neutralize<'py>(
    py: Python<'py>,
    y: PyReadonlyArray2<'_, f64>,
    groups: Option<PyReadonlyArray2<'_, f64>>,
    exposures: Vec<PyReadonlyArray2<'_, f64>>,
    threads: usize,
) -> PyResult<Columns<'py>> {
    let (data, n_rows, n_cols) = columns_of(&y)?;
    let outcome = Panel::new(data, n_rows, n_cols).map_err(to_py_err)?;
    let group_panel = match &groups {
        None => None,
        Some(array) => Some(panel(array)?),
    };
    let exposure_panels = panels(&exposures)?;
    let out = py
        .detach(|| {
            phasor_factor::neutralize(&outcome, group_panel.as_ref(), &exposure_panels, threads)
        })
        .map_err(to_py_err)?;
    columns_out(py, out, n_rows, n_cols)
}

/// Names, column-major coefficients `(bars, names)`, and R² per bar.
type FmOut<'py> = (Vec<String>, Columns<'py>, Bound<'py, PyArray1<f64>>);

/// Fama–MacBeth slopes: `fwd ~ 1 + factors + controls` on every bar.
///
/// The first coefficient is the intercept, then one column per factor name,
/// then one `control_{i}` per control. With `groups`, labels are fixed
/// effects and the intercept is the average of the group intercepts. A bar
/// that cannot identify the slopes is `NaN`. `threads` 0 uses every core.
#[pyfunction]
pub fn factor_fama_macbeth<'py>(
    py: Python<'py>,
    fwd: PyReadonlyArray2<'_, f64>,
    factors: Vec<PyReadonlyArray2<'_, f64>>,
    factor_names: Vec<String>,
    groups: Option<PyReadonlyArray2<'_, f64>>,
    controls: Vec<PyReadonlyArray2<'_, f64>>,
    threads: usize,
) -> PyResult<FmOut<'py>> {
    let outcome = panel(&fwd)?;
    let factor_panels = panels(&factors)?;
    let group_panel = match &groups {
        None => None,
        Some(array) => Some(panel(array)?),
    };
    let control_panels = panels(&controls)?;
    let n_rows = outcome.n_rows();
    let n_coefs = 1 + factor_panels.len() + control_panels.len();
    let fit = py
        .detach(|| {
            phasor_factor::fama_macbeth(
                &outcome,
                &factor_panels,
                &factor_names,
                group_panel.as_ref(),
                &control_panels,
                threads,
            )
        })
        .map_err(to_py_err)?;
    let names = fit.names;
    let coef = columns_out(py, fit.coef, n_rows, n_coefs)?;
    Ok((names, coef, fit.r2.into_pyarray(py)))
}

/// `K x K` mean cross-sectional Spearman correlation, column-major.
///
/// The diagonal is 1 where a factor varies on at least one row. `threads` 0
/// uses every core; the values do not depend on how many.
#[pyfunction]
pub fn factor_mean_cs_corr<'py>(
    py: Python<'py>,
    factors: Vec<PyReadonlyArray2<'_, f64>>,
    threads: usize,
) -> PyResult<Columns<'py>> {
    let factor_panels = panels(&factors)?;
    let k = factor_panels.len();
    let out = py
        .detach(|| phasor_factor::mean_cs_corr(&factor_panels, threads))
        .map_err(to_py_err)?;
    columns_out(py, out, k, k)
}

/// `n_candidates x n_references` mean cross-sectional Spearman correlation.
///
/// Column-major. Entry `(i, j)` is candidate `i` against reference `j`.
#[pyfunction]
pub fn factor_cross_corr<'py>(
    py: Python<'py>,
    candidates: Vec<PyReadonlyArray2<'_, f64>>,
    references: Vec<PyReadonlyArray2<'_, f64>>,
    threads: usize,
) -> PyResult<Columns<'py>> {
    let candidate_panels = panels(&candidates)?;
    let reference_panels = panels(&references)?;
    let (n_c, n_r) = (candidate_panels.len(), reference_panels.len());
    let out = py
        .detach(|| phasor_factor::cross_corr(&candidate_panels, &reference_panels, threads))
        .map_err(to_py_err)?;
    columns_out(py, out, n_c, n_r)
}

/// Observed rank IC, its placebo p-value, and one null mean per draw.
type PlaceboOut<'py> = (f64, f64, Bound<'py, PyArray1<f64>>);

fn placebo_out<'py>(py: Python<'py>, result: phasor_factor::PlaceboResult) -> PlaceboOut<'py> {
    (
        result.observed,
        result.p_value,
        result.null_distribution.into_pyarray(py),
    )
}

/// Cross-sectional placebo. Each draw shuffles every bar's forward returns
/// and averages the per-bar Spearman IC.
///
/// Returns `(observed, p_value, null_distribution)`. `threads` 0 uses every
/// core; the values do not depend on how many.
#[pyfunction]
pub fn factor_cs_permutation<'py>(
    py: Python<'py>,
    factor: PyReadonlyArray2<'_, f64>,
    forward: PyReadonlyArray2<'_, f64>,
    draws: usize,
    seed: u64,
    min_names: usize,
    threads: usize,
) -> PyResult<PlaceboOut<'py>> {
    let x = panel(&factor)?;
    let y = panel(&forward)?;
    let result = py
        .detach(|| phasor_factor::cs_permutation(&x, &y, draws, seed, min_names, threads))
        .map_err(to_py_err)?;
    Ok(placebo_out(py, result))
}

/// Time-series placebo. Each draw circularly shifts `forward` along time and
/// averages the per-symbol Spearman IC.
///
/// The offset is drawn from `[n_rows / 10, n_rows - n_rows / 10]`. Returns
/// `(observed, p_value, null_distribution)`.
#[pyfunction]
pub fn factor_ts_circular_shift<'py>(
    py: Python<'py>,
    factor: PyReadonlyArray2<'_, f64>,
    forward: PyReadonlyArray2<'_, f64>,
    draws: usize,
    seed: u64,
    min_periods: usize,
    threads: usize,
) -> PyResult<PlaceboOut<'py>> {
    let x = panel(&factor)?;
    let y = panel(&forward)?;
    let result = py
        .detach(|| phasor_factor::ts_circular_shift(&x, &y, draws, seed, min_periods, threads))
        .map_err(to_py_err)?;
    Ok(placebo_out(py, result))
}

/// Registers the functions above on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(factor_forward_returns, m)?)?;
    m.add_function(wrap_pyfunction!(factor_cs_ic, m)?)?;
    m.add_function(wrap_pyfunction!(factor_ts_ic, m)?)?;
    m.add_function(wrap_pyfunction!(factor_ts_rolling_ic, m)?)?;
    m.add_function(wrap_pyfunction!(factor_ts_block_ic, m)?)?;
    m.add_function(wrap_pyfunction!(factor_summarize, m)?)?;
    m.add_function(wrap_pyfunction!(factor_auto_lags, m)?)?;
    m.add_function(wrap_pyfunction!(factor_quantile_labels, m)?)?;
    m.add_function(wrap_pyfunction!(factor_quantile_returns, m)?)?;
    m.add_function(wrap_pyfunction!(factor_long_short_returns, m)?)?;
    m.add_function(wrap_pyfunction!(factor_monotonicity, m)?)?;
    m.add_function(wrap_pyfunction!(factor_quantile_turnover, m)?)?;
    m.add_function(wrap_pyfunction!(factor_rank_autocorr, m)?)?;
    m.add_function(wrap_pyfunction!(factor_neutralize, m)?)?;
    m.add_function(wrap_pyfunction!(factor_fama_macbeth, m)?)?;
    m.add_function(wrap_pyfunction!(factor_mean_cs_corr, m)?)?;
    m.add_function(wrap_pyfunction!(factor_cross_corr, m)?)?;
    m.add_function(wrap_pyfunction!(factor_cs_permutation, m)?)?;
    m.add_function(wrap_pyfunction!(factor_ts_circular_shift, m)?)?;
    Ok(())
}
