//! Research tools that sit beside a backtest rather than inside one.
//!
//! Event studies, overfitting statistics, supervised-learning labels, and
//! seeded random signals and prices. None of them runs a simulation, and
//! several of them read the future on purpose — the labels and the forward
//! paths — so none of them may feed one either. The Python modules
//! `alphax.events`, `alphax.robust`, `alphax.labels`, `alphax.signals` and
//! `alphax.synthetic` wrap these and say so on every function.
//!
//! Every array crosses as it does for the simulations: F-ordered in, through
//! [`columns_of`], and F-ordered out, through [`columns_out`], with no copy
//! either way. The functions here are one engine call each; the shape checks
//! and the NumPy plumbing are the whole of the work.

use alphax_analysis::{events, robust};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::array::{columns_of, columns_out, to_py_err};

/// A 2-D result handed back to Python, `(n_rows, n_cols)` and F-ordered.
type Columns<'py, T> = Bound<'py, PyArray2<T>>;

/// The price paths around every event of a study.
///
/// Built by [`forward_paths`]. One row per event, one column per offset on
/// `-before..=after`; each value is the close at that offset over the close
/// on the event bar, minus one.
#[pyclass(name = "EventPaths", module = "alphax.events", frozen)]
#[derive(Debug)]
pub struct PyEventPaths {
    inner: events::EventPaths,
}

#[pymethods]
impl PyEventPaths {
    /// Column of each event, in row order.
    #[getter]
    fn col<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.inner.col.clone().into_pyarray(py)
    }

    /// Bar of each event within its column, in row order.
    #[getter]
    fn idx<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.inner.idx.clone().into_pyarray(py)
    }

    /// The offset axis, `-before..=after`.
    #[getter]
    fn offsets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.inner.offsets.clone().into_pyarray(py)
    }

    /// `(n_events, before + after + 1)` returns relative to each event's bar.
    ///
    /// Offsets that fall outside the event's column are NaN, as is every
    /// offset of an event whose own close is NaN.
    #[getter]
    fn paths<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        // Row-major in the engine — an event is the unit — so a C-ordered
        // reshape is the right reading of the same buffer.
        self.inner
            .paths
            .clone()
            .into_pyarray(py)
            .reshape([self.inner.n_events(), self.inner.width()])
    }

    /// Mean across events at each offset, NaN skipped.
    fn mean_path<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        events::mean_path(&self.inner).into_pyarray(py)
    }

    /// Median across events at each offset, NaN skipped.
    fn median_path<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        events::median_path(&self.inner).into_pyarray(py)
    }

    /// Fraction of events strictly above their event-bar close at `offset`.
    ///
    /// NaN when `offset` is off the axis or no event has a value there.
    fn hit_rate(&self, offset: i64) -> f64 {
        events::hit_rate(&self.inner, offset)
    }

    fn __len__(&self) -> usize {
        self.inner.n_events()
    }

    fn __repr__(&self) -> String {
        let offsets = &self.inner.offsets;
        format!(
            "EventPaths(n_events={}, offsets={}..={})",
            self.inner.n_events(),
            offsets.first().copied().unwrap_or(0),
            offsets.last().copied().unwrap_or(0),
        )
    }
}

/// The return path around every bar where `events` is true.
#[pyfunction]
pub fn forward_paths(
    py: Python<'_>,
    close: PyReadonlyArray2<'_, f64>,
    events: PyReadonlyArray2<'_, bool>,
    before: usize,
    after: usize,
) -> PyResult<PyEventPaths> {
    let (close, n_rows, n_cols) = columns_of(&close)?;
    let (fired, rows, cols) = columns_of(&events)?;
    if (rows, cols) != (n_rows, n_cols) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "events is {rows}x{cols}; close is {n_rows}x{n_cols}"
        )));
    }
    let inner = py
        .detach(|| events::forward_paths(close, fired, n_rows, n_cols, before, after))
        .map_err(to_py_err)?;
    Ok(PyEventPaths { inner })
}

/// Probabilistic Sharpe ratio of each column of `returns`, per period.
#[pyfunction]
pub fn probabilistic_sharpe<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'_, f64>,
    benchmark_sr: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (flat, n_rows, n_cols) = columns_of(&returns)?;
    let out: Vec<f64> = py.detach(|| {
        column_slices(flat, n_rows, n_cols)
            .map(|r| robust::probabilistic_sharpe(r, benchmark_sr))
            .collect()
    });
    Ok(out.into_pyarray(py))
}

/// The Sharpe ratio the best of `n_trials` worthless trials shows by luck.
#[pyfunction]
pub fn expected_max_sharpe(n_trials: usize, sr_variance: f64) -> f64 {
    robust::expected_max_sharpe(n_trials, sr_variance)
}

/// Deflated Sharpe ratio of each column of `returns` against one sweep.
#[pyfunction]
pub fn deflated_sharpe<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'_, f64>,
    trial_sharpes: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (flat, n_rows, n_cols) = columns_of(&returns)?;
    let trials = trial_sharpes.as_array();
    let trials = trials.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("trial_sharpes must be contiguous")
    })?;
    let out: Vec<f64> = py.detach(|| {
        column_slices(flat, n_rows, n_cols)
            .map(|r| robust::deflated_sharpe(r, trials))
            .collect()
    });
    Ok(out.into_pyarray(py))
}

/// Probability of backtest overfitting across the columns of `perf`.
#[pyfunction]
pub fn pbo_cscv(py: Python<'_>, perf: PyReadonlyArray2<'_, f64>, n_splits: usize) -> PyResult<f64> {
    let (flat, n_rows, n_cols) = columns_of(&perf)?;
    py.detach(|| robust::pbo_cscv(flat, n_rows, n_cols, n_splits))
        .map_err(to_py_err)
}

/// Simple return from each bar to the bar `horizon` rows later.
#[pyfunction]
pub fn fixed_horizon_returns<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'_, f64>,
    horizon: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (flat, n_rows, n_cols) = columns_of(&close)?;
    let out = py
        .detach(|| alphax_signals::fixed_horizon_returns(flat, n_rows, n_cols, horizon))
        .map_err(to_py_err)?;
    columns_out(py, out, n_rows, n_cols)
}

/// +1, 0 or -1 by the forward return over `horizon` against `threshold`.
#[pyfunction]
pub fn trend_labels<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'_, f64>,
    horizon: usize,
    threshold: f64,
) -> PyResult<Bound<'py, PyArray2<i8>>> {
    let (flat, n_rows, n_cols) = columns_of(&close)?;
    let out = py
        .detach(|| alphax_signals::trend_labels(flat, n_rows, n_cols, horizon, threshold))
        .map_err(to_py_err)?;
    columns_out(py, out, n_rows, n_cols)
}

/// Which of a profit-take, a stop-loss or a time limit each bar meets first,
/// and the row it is met on.
#[pyfunction]
pub fn triple_barrier<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'_, f64>,
    up: f64,
    down: f64,
    max_hold: usize,
) -> PyResult<(Columns<'py, i8>, Columns<'py, i64>)> {
    let (flat, n_rows, n_cols) = columns_of(&close)?;
    let (labels, decided) = py
        .detach(|| alphax_signals::triple_barrier(flat, n_rows, n_cols, up, down, max_hold))
        .map_err(to_py_err)?;
    Ok((
        columns_out(py, labels, n_rows, n_cols)?,
        columns_out(py, decided, n_rows, n_cols)?,
    ))
}

/// +1 at confirmed zig-zag troughs, -1 at confirmed peaks, 0 elsewhere.
#[pyfunction]
pub fn pivot_labels<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'_, f64>,
    threshold: f64,
) -> PyResult<Bound<'py, PyArray2<i8>>> {
    let (flat, n_rows, n_cols) = columns_of(&close)?;
    let out = py
        .detach(|| alphax_signals::pivot_labels(flat, n_rows, n_cols, threshold))
        .map_err(to_py_err)?;
    columns_out(py, out, n_rows, n_cols)
}

/// The exits, the price each fired at, which stop and the share closed.
type StopExitArrays<'py> = (
    Columns<'py, bool>,
    Columns<'py, f64>,
    Columns<'py, i8>,
    Columns<'py, f64>,
);

/// Where the stops of each entry would fire, without a simulation: the
/// exits, the price each fired at, which stop (0 stop-loss, 1 trailing,
/// 2 take-profit, 3 time, -1 none) and the share of the entry's position
/// each exit closes. See `alphax_sim::stop_exits`.
#[pyfunction]
#[pyo3(signature = (
    entries, close, *, open = None, high = None, low = None, sl_stop = None,
    sl_trail = false, tsl_stop = None, tsl_th = None, tp_stop = None, td_stop = None,
    dt_stop = None, short = false, wait = 1, until_next = true, skip_until_exit = false,
    delta_format = "percent", stop_ladder = "disabled", stop_order_type = "market",
    stop_limit_delta = None, limit_order_price = "limit",
))]
#[allow(
    clippy::too_many_arguments,
    reason = "one argument per input, as the Python signature has"
)]
pub fn generate_ohlc_stop_exits<'py>(
    py: Python<'py>,
    entries: PyReadonlyArray2<'py, bool>,
    close: PyReadonlyArray2<'py, f64>,
    open: Option<PyReadonlyArray2<'py, f64>>,
    high: Option<PyReadonlyArray2<'py, f64>>,
    low: Option<PyReadonlyArray2<'py, f64>>,
    sl_stop: Option<&Bound<'py, PyAny>>,
    sl_trail: bool,
    tsl_stop: Option<&Bound<'py, PyAny>>,
    tsl_th: Option<&Bound<'py, PyAny>>,
    tp_stop: Option<&Bound<'py, PyAny>>,
    td_stop: Option<&Bound<'py, PyAny>>,
    dt_stop: Option<&Bound<'py, PyAny>>,
    short: bool,
    wait: usize,
    until_next: bool,
    skip_until_exit: bool,
    delta_format: &str,
    stop_ladder: &str,
    stop_order_type: &str,
    stop_limit_delta: Option<&Bound<'py, PyAny>>,
    limit_order_price: &str,
) -> PyResult<StopExitArrays<'py>> {
    use crate::array::{OneDim, borrow_opt_f64, flex_or_f64};
    use alphax_core::flex::FlexArray;
    let (close_flat, n_rows, n_cols) = columns_of(&close)?;
    let (entry_flat, r, c) = columns_of(&entries)?;
    if (r, c) != (n_rows, n_cols) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "entries must have close's shape ({n_rows}, {n_cols}); got ({r}, {c})"
        )));
    }
    fn bar<'a>(
        a: &'a Option<PyReadonlyArray2<'_, f64>>,
        name: &str,
        n_rows: usize,
        n_cols: usize,
    ) -> PyResult<FlexArray<'a, f64>> {
        let Some(a) = a else {
            return Ok(FlexArray::Scalar(f64::NAN));
        };
        let (flat, r, c) = columns_of(a)?;
        if (r, c) != (n_rows, n_cols) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{name} must have close's shape ({n_rows}, {n_cols}); got ({r}, {c})"
            )));
        }
        FlexArray::full(flat, n_rows, n_cols).map_err(to_py_err)
    }
    let stops = [
        borrow_opt_f64(sl_stop, "sl_stop")?,
        borrow_opt_f64(tsl_stop, "tsl_stop")?,
        borrow_opt_f64(tsl_th, "tsl_th")?,
        borrow_opt_f64(tp_stop, "tp_stop")?,
        borrow_opt_f64(td_stop, "td_stop")?,
        borrow_opt_f64(dt_stop, "dt_stop")?,
        borrow_opt_f64(stop_limit_delta, "stop_limit_delta")?,
    ];
    let stop = |k: usize| {
        flex_or_f64(
            &stops[k],
            n_rows,
            n_cols,
            OneDim::PerColumn,
            FlexArray::Scalar(f64::NAN),
        )
    };
    let bad = pyo3::exceptions::PyValueError::new_err;
    let ladder = crate::parse::stop_ladder(stop_ladder).map_err(bad)?;
    // A ladder's rows are its steps: a 1-D array is every column's steps.
    let price_stop = |k: usize, name: &str| {
        if ladder == alphax_model::StopLadderMode::Disabled {
            stop(k)
        } else {
            crate::array::ladder_steps(&stops[k], n_cols, name, FlexArray::Scalar(f64::NAN))
        }
    };
    let args = alphax_sim::stop_exits::OhlcStopExits {
        n_rows,
        n_cols,
        entries: FlexArray::full(entry_flat, n_rows, n_cols).map_err(to_py_err)?,
        open: bar(&open, "open", n_rows, n_cols)?,
        high: bar(&high, "high", n_rows, n_cols)?,
        low: bar(&low, "low", n_rows, n_cols)?,
        close: FlexArray::full(close_flat, n_rows, n_cols).map_err(to_py_err)?,
        sl_stop: price_stop(0, "sl_stop")?,
        sl_trail: FlexArray::Scalar(sl_trail),
        tsl_stop: price_stop(1, "tsl_stop")?,
        tsl_th: stop(2)?,
        tp_stop: price_stop(3, "tp_stop")?,
        td_stop: stop(4)?,
        dt_stop: stop(5)?,
        delta_format: FlexArray::Scalar(crate::parse::delta_format(delta_format).map_err(bad)?),
        stop_ladder: ladder,
        stop_order_type: FlexArray::Scalar(crate::parse::order_type(stop_order_type).map_err(bad)?),
        stop_limit_delta: stop(6)?,
        limit_order_price: FlexArray::Scalar(
            crate::parse::limit_order_price(limit_order_price).map_err(bad)?,
        ),
        short,
        wait,
        until_next,
        skip_until_exit,
    };
    let out = py
        .detach(|| alphax_sim::stop_exits::generate_ohlc_stop_exits(&args))
        .map_err(to_py_err)?;
    Ok((
        columns_out(py, out.exits, n_rows, n_cols)?,
        columns_out(py, out.price, n_rows, n_cols)?,
        columns_out(py, out.kind, n_rows, n_cols)?,
        columns_out(py, out.size, n_rows, n_cols)?,
    ))
}

/// Each bar fires independently with probability `prob`.
#[pyfunction]
pub fn random_by_prob(
    py: Python<'_>,
    n_rows: usize,
    n_cols: usize,
    prob: f64,
    seed: u64,
) -> PyResult<Bound<'_, PyArray2<bool>>> {
    let out = py
        .detach(|| alphax_signals::random_by_prob(n_rows, n_cols, prob, seed))
        .map_err(to_py_err)?;
    columns_out(py, out, n_rows, n_cols)
}

/// Exactly `n` signals per column, at distinct rows.
#[pyfunction]
pub fn random_n(
    py: Python<'_>,
    n_rows: usize,
    n_cols: usize,
    n: usize,
    seed: u64,
) -> PyResult<Bound<'_, PyArray2<bool>>> {
    let out = py
        .detach(|| alphax_signals::random_n(n_rows, n_cols, n, seed))
        .map_err(to_py_err)?;
    columns_out(py, out, n_rows, n_cols)
}

/// `n` strictly alternating entry/exit pairs per column.
#[pyfunction]
pub fn random_entries_exits(
    py: Python<'_>,
    n_rows: usize,
    n_cols: usize,
    n: usize,
    seed: u64,
) -> PyResult<(Columns<'_, bool>, Columns<'_, bool>)> {
    let (entries, exits) = py
        .detach(|| alphax_signals::random_entries_exits(n_rows, n_cols, n, seed))
        .map_err(to_py_err)?;
    Ok((
        columns_out(py, entries, n_rows, n_cols)?,
        columns_out(py, exits, n_rows, n_cols)?,
    ))
}

/// Gaussian random walks starting at `start`.
#[pyfunction]
pub fn random_walk(
    py: Python<'_>,
    n_rows: usize,
    n_cols: usize,
    start: f64,
    step_std: f64,
    seed: u64,
) -> PyResult<Bound<'_, PyArray2<f64>>> {
    let out = py
        .detach(|| alphax_generic::random_walk(n_rows, n_cols, start, step_std, seed))
        .map_err(to_py_err)?;
    columns_out(py, out, n_rows, n_cols)
}

/// Geometric Brownian motion, discretized exactly.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn gbm(
    py: Python<'_>,
    n_rows: usize,
    n_cols: usize,
    start: f64,
    mu: f64,
    sigma: f64,
    dt: f64,
    seed: u64,
) -> PyResult<Bound<'_, PyArray2<f64>>> {
    let out = py
        .detach(|| alphax_generic::gbm(n_rows, n_cols, start, mu, sigma, dt, seed))
        .map_err(to_py_err)?;
    columns_out(py, out, n_rows, n_cols)
}

/// Opens, highs and lows consistent with `close`.
#[pyfunction]
pub fn ohlc_from_close<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'_, f64>,
    spread: f64,
    seed: u64,
) -> PyResult<(Columns<'py, f64>, Columns<'py, f64>, Columns<'py, f64>)> {
    let (flat, n_rows, n_cols) = columns_of(&close)?;
    let (open, high, low) = py
        .detach(|| alphax_generic::ohlc_from_close(flat, n_rows, n_cols, spread, seed))
        .map_err(to_py_err)?;
    Ok((
        columns_out(py, open, n_rows, n_cols)?,
        columns_out(py, high, n_rows, n_cols)?,
        columns_out(py, low, n_rows, n_cols)?,
    ))
}

/// Each column of a column-major buffer as its own slice.
fn column_slices(flat: &[f64], n_rows: usize, n_cols: usize) -> impl Iterator<Item = &[f64]> {
    // `chunks_exact` would yield nothing for zero rows, where every column
    // still needs its (NaN) answer.
    (0..n_cols).map(move |c| &flat[c * n_rows..(c + 1) * n_rows])
}

/// Registers the classes and functions above on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEventPaths>()?;
    m.add_function(wrap_pyfunction!(forward_paths, m)?)?;
    m.add_function(wrap_pyfunction!(probabilistic_sharpe, m)?)?;
    m.add_function(wrap_pyfunction!(expected_max_sharpe, m)?)?;
    m.add_function(wrap_pyfunction!(deflated_sharpe, m)?)?;
    m.add_function(wrap_pyfunction!(pbo_cscv, m)?)?;
    m.add_function(wrap_pyfunction!(fixed_horizon_returns, m)?)?;
    m.add_function(wrap_pyfunction!(trend_labels, m)?)?;
    m.add_function(wrap_pyfunction!(triple_barrier, m)?)?;
    m.add_function(wrap_pyfunction!(pivot_labels, m)?)?;
    m.add_function(wrap_pyfunction!(random_by_prob, m)?)?;
    m.add_function(wrap_pyfunction!(random_n, m)?)?;
    m.add_function(wrap_pyfunction!(random_entries_exits, m)?)?;
    m.add_function(wrap_pyfunction!(generate_ohlc_stop_exits, m)?)?;
    m.add_function(wrap_pyfunction!(random_walk, m)?)?;
    m.add_function(wrap_pyfunction!(gbm, m)?)?;
    m.add_function(wrap_pyfunction!(ohlc_from_close, m)?)?;
    Ok(())
}
