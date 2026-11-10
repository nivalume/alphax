//! Metrics of return series, scalar and rolling, and drawdown records: the
//! Rust side of `alphax.returns` and of `Portfolio.rolling` and
//! `Portfolio.drawdowns`. A series is a column of an `(n_rows, n_cols)`
//! array; a benchmark, where a metric needs one, is shaped the same.

use alphax_analysis::{benchmark as bm, metrics as m, rolling as rl};
use alphax_records::{DrawdownRecords, find_drawdowns};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::array::{columns_of, columns_out, to_py_err};

/// The options the metrics read.
#[derive(Debug, Clone, Copy)]
pub struct Opts {
    pub ann_factor: f64,
    pub risk_free: f64,
    pub required_return: f64,
    pub cutoff: f64,
    pub ddof: usize,
}

/// Metrics that read only the series.
pub const SOLO: &[&str] = &[
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "max_drawdown",
    "calmar_ratio",
    "omega_ratio",
    "sharpe_ratio",
    "downside_risk",
    "sortino_ratio",
    "tail_ratio",
    "value_at_risk",
    "cond_value_at_risk",
    "hit_ratio",
    "return_concentration",
];

/// Metrics that read a benchmark too.
pub const PAIRED: &[&str] = &[
    "information_ratio",
    "beta",
    "alpha",
    "capture",
    "up_capture",
    "down_capture",
];

fn unknown(name: &str) -> PyErr {
    PyValueError::new_err(format!(
        "no metric {name:?}; metrics are {} and, with a benchmark, {}",
        SOLO.join(", "),
        PAIRED.join(", ")
    ))
}

fn need_benchmark(name: &str) -> PyErr {
    PyValueError::new_err(format!("{name} needs a benchmark"))
}

/// One metric of one series.
///
/// # Errors
///
/// A `ValueError` for an unknown metric or a missing benchmark.
pub fn scalar(name: &str, r: &[f64], b: Option<&[f64]>, o: Opts) -> PyResult<f64> {
    Ok(match name {
        "total_return" => m::total_return(r),
        "annualized_return" => m::annualized_return(r, o.ann_factor),
        "annualized_volatility" => m::annualized_volatility(r, o.ann_factor),
        "max_drawdown" => m::max_drawdown(r),
        "calmar_ratio" => m::calmar_ratio(r, o.ann_factor),
        "omega_ratio" => m::omega_ratio(r, o.ann_factor, o.risk_free, o.required_return),
        "sharpe_ratio" => m::sharpe_ratio(r, o.ann_factor, o.risk_free),
        "downside_risk" => m::downside_risk(r, o.ann_factor, o.required_return),
        "sortino_ratio" => m::sortino_ratio(r, o.ann_factor, o.required_return),
        "tail_ratio" => m::tail_ratio(r),
        "value_at_risk" => m::value_at_risk(r, o.cutoff),
        "cond_value_at_risk" => m::cond_value_at_risk(r, o.cutoff),
        "hit_ratio" => m::hit_ratio(r),
        "return_concentration" => m::return_concentration(r, o.cutoff),
        paired if PAIRED.contains(&paired) => {
            let b = b.ok_or_else(|| need_benchmark(paired))?;
            match paired {
                "information_ratio" => bm::information_ratio(r, b, o.ddof),
                "beta" => bm::beta(r, b),
                "alpha" => bm::alpha(r, b, o.ann_factor, o.risk_free),
                "capture" => bm::capture(r, b, o.ann_factor),
                "up_capture" => bm::up_capture(r, b, o.ann_factor),
                _ => bm::down_capture(r, b, o.ann_factor),
            }
        }
        other => return Err(unknown(other)),
    })
}

/// One metric of one series, over each trailing `window`.
///
/// # Errors
///
/// As [`scalar`], and the rolling functions' own.
pub fn rolling(
    name: &str,
    r: &[f64],
    b: Option<&[f64]>,
    window: usize,
    min_periods: Option<usize>,
    o: Opts,
) -> PyResult<Vec<f64>> {
    let (w, p) = (window, min_periods);
    let out = match name {
        "total_return" => rl::rolling_total_return(r, w, p),
        "annualized_return" => rl::rolling_annualized_return(r, w, p, o.ann_factor),
        "annualized_volatility" => rl::rolling_annualized_volatility(r, w, p, o.ann_factor),
        "max_drawdown" => rl::rolling_max_drawdown(r, w, p),
        "calmar_ratio" => rl::rolling_calmar_ratio(r, w, p, o.ann_factor),
        "omega_ratio" => {
            rl::rolling_omega_ratio(r, w, p, o.ann_factor, o.risk_free, o.required_return)
        }
        "sharpe_ratio" => rl::rolling_sharpe_ratio(r, w, p, o.ann_factor, o.risk_free),
        "downside_risk" => rl::rolling_downside_risk(r, w, p, o.ann_factor, o.required_return),
        "sortino_ratio" => rl::rolling_sortino_ratio(r, w, p, o.ann_factor, o.required_return),
        "tail_ratio" => rl::rolling_tail_ratio(r, w, p),
        "value_at_risk" => rl::rolling_value_at_risk(r, w, p, o.cutoff),
        "cond_value_at_risk" => rl::rolling_cond_value_at_risk(r, w, p, o.cutoff),
        "hit_ratio" => rl::rolling_hit_ratio(r, w, p),
        "return_concentration" => rl::rolling_return_concentration(r, w, p, o.cutoff),
        paired if PAIRED.contains(&paired) => {
            let b = b.ok_or_else(|| need_benchmark(paired))?;
            match paired {
                "information_ratio" => rl::rolling_information_ratio(r, b, w, p, o.ddof),
                "beta" => rl::rolling_beta(r, b, w, p),
                "alpha" => rl::rolling_alpha(r, b, w, p, o.ann_factor, o.risk_free),
                "capture" => rl::rolling_capture(r, b, w, p, o.ann_factor),
                "up_capture" => rl::rolling_up_capture(r, b, w, p, o.ann_factor),
                _ => rl::rolling_down_capture(r, b, w, p, o.ann_factor),
            }
        }
        other => return Err(unknown(other)),
    };
    out.map_err(to_py_err)
}

/// Drawdown records as a dict of arrays, with their derived fields.
pub fn drawdowns_dict<'py>(py: Python<'py>, d: &DrawdownRecords) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("col", d.col.clone().into_pyarray(py))?;
    out.set_item("peak_idx", d.peak_idx.clone().into_pyarray(py))?;
    out.set_item("start_idx", d.start_idx.clone().into_pyarray(py))?;
    out.set_item("valley_idx", d.valley_idx.clone().into_pyarray(py))?;
    out.set_item("end_idx", d.end_idx.clone().into_pyarray(py))?;
    out.set_item("peak_val", d.peak_val.clone().into_pyarray(py))?;
    out.set_item("valley_val", d.valley_val.clone().into_pyarray(py))?;
    out.set_item("end_val", d.end_val.clone().into_pyarray(py))?;
    let status: Vec<i64> = d.status.iter().map(|&s| i64::from(s)).collect();
    out.set_item("status", status.into_pyarray(py))?;
    out.set_item("drawdown", d.drawdown().into_pyarray(py))?;
    out.set_item("duration", d.duration().into_pyarray(py))?;
    out.set_item("decline_duration", d.decline_duration().into_pyarray(py))?;
    out.set_item("recovery_duration", d.recovery_duration().into_pyarray(py))?;
    out.set_item("recovery_return", d.recovery_return().into_pyarray(py))?;
    out.set_item("recovered", d.recovered().into_pyarray(py))?;
    Ok(out)
}

fn benchmark_cols<'a>(
    benchmark: Option<&'a PyReadonlyArray2<'_, f64>>,
    shape: (usize, usize),
) -> PyResult<Option<&'a [f64]>> {
    let Some(b) = benchmark else {
        return Ok(None);
    };
    let (flat, r, c) = columns_of(b)?;
    if (r, c) != shape {
        return Err(PyValueError::new_err(format!(
            "benchmark must be shaped like returns: {:?} and {shape:?}",
            (r, c)
        )));
    }
    Ok(Some(flat))
}

/// A metric of each column.
#[pyfunction]
#[pyo3(signature = (returns, metric, benchmark, ann_factor, risk_free, required_return, cutoff, ddof))]
#[allow(clippy::too_many_arguments, reason = "the metrics' options")]
pub fn ret_metric<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'_, f64>,
    metric: &str,
    benchmark: Option<PyReadonlyArray2<'_, f64>>,
    ann_factor: f64,
    risk_free: f64,
    required_return: f64,
    cutoff: f64,
    ddof: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (flat, n_rows, n_cols) = columns_of(&returns)?;
    let bench = benchmark_cols(benchmark.as_ref(), (n_rows, n_cols))?;
    let o = Opts {
        ann_factor,
        risk_free,
        required_return,
        cutoff,
        ddof,
    };
    let out = (0..n_cols)
        .map(|c| {
            let span = c * n_rows..(c + 1) * n_rows;
            scalar(metric, &flat[span.clone()], bench.map(|b| &b[span]), o)
        })
        .collect::<PyResult<Vec<f64>>>()?;
    Ok(out.into_pyarray(py))
}

/// A metric of each column over each trailing window.
#[pyfunction]
#[pyo3(signature = (returns, metric, window, min_periods, benchmark, ann_factor, risk_free, required_return, cutoff, ddof))]
#[allow(
    clippy::too_many_arguments,
    reason = "the window and the metrics' options"
)]
pub fn ret_rolling<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'_, f64>,
    metric: &str,
    window: usize,
    min_periods: Option<usize>,
    benchmark: Option<PyReadonlyArray2<'_, f64>>,
    ann_factor: f64,
    risk_free: f64,
    required_return: f64,
    cutoff: f64,
    ddof: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (flat, n_rows, n_cols) = columns_of(&returns)?;
    let bench = benchmark_cols(benchmark.as_ref(), (n_rows, n_cols))?;
    let o = Opts {
        ann_factor,
        risk_free,
        required_return,
        cutoff,
        ddof,
    };
    let mut out = Vec::with_capacity(n_rows * n_cols);
    for c in 0..n_cols {
        let span = c * n_rows..(c + 1) * n_rows;
        out.extend(rolling(
            metric,
            &flat[span.clone()],
            bench.map(|b| &b[span]),
            window,
            min_periods,
            o,
        )?);
    }
    columns_out(py, out, n_rows, n_cols)
}

/// Drawdown records of each column of a value curve.
#[pyfunction]
pub fn ret_drawdowns<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let (flat, n_rows, n_cols) = columns_of(&values)?;
    let d = find_drawdowns(flat, n_rows, n_cols).map_err(to_py_err)?;
    drawdowns_dict(py, &d)
}

/// Adds the functions to the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ret_metric, m)?)?;
    m.add_function(wrap_pyfunction!(ret_rolling, m)?)?;
    m.add_function(wrap_pyfunction!(ret_drawdowns, m)?)?;
    Ok(())
}
