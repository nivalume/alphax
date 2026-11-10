//! The Rust indicators, swept over parameter lists: `alphax.indicators`.
//!
//! Each function takes one list per parameter, of the same length: the
//! combinations to compute, built by the Python layer (a product of the
//! values by default). The output's columns are combination-major: every
//! input column under the first combination, then under the second. Each
//! comes back as a dict of `(n_rows, n_combinations × n_input)` arrays by
//! output name, and `columns`, each level's label per output column.

use alphax_core::columns::Label;
use alphax_core::matrix::Matrix;
use alphax_indicators::{Bar, BbParams, Indicator, MaParams, MacdParams, StochParams, Stream};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::array::{columns_of, columns_out, to_py_err};

fn matrix(a: &PyReadonlyArray2<'_, f64>) -> PyResult<Matrix> {
    let (flat, n_rows, n_cols) = columns_of(a)?;
    Matrix::from_column_major(flat.to_vec(), n_rows, n_cols).map_err(to_py_err)
}

fn same_len(lens: &[usize]) -> PyResult<usize> {
    let n = lens[0];
    if lens.iter().any(|&l| l != n) {
        return Err(PyValueError::new_err(
            "every parameter list must be as long as the others: one entry per combination",
        ));
    }
    Ok(n)
}

fn label(py: Python<'_>, l: &Label) -> PyResult<Py<PyAny>> {
    Ok(match l {
        Label::Int(v) => v.into_pyobject(py)?.into_any().unbind(),
        Label::Float(v) => v.into_pyobject(py)?.into_any().unbind(),
        Label::Str(s) => s.as_ref().into_pyobject(py)?.into_any().unbind(),
        Label::Bool(b) => b.into_pyobject(py)?.to_owned().into_any().unbind(),
    })
}

fn out<'py>(py: Python<'py>, ind: Indicator) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let outputs = PyDict::new(py);
    for name in ind.names() {
        let m = ind.get(name).map_err(to_py_err)?;
        let (r, c) = (m.n_rows(), m.n_cols());
        let a: Bound<'py, PyArray2<f64>> = columns_out(py, m.as_slice().to_vec(), r, c)?;
        outputs.set_item(name, a)?;
    }
    d.set_item("outputs", outputs)?;
    let columns = PyDict::new(py);
    for level in ind.columns.levels() {
        let labels = level
            .labels()
            .iter()
            .map(|l| label(py, l))
            .collect::<PyResult<Vec<_>>>()?;
        columns.set_item(level.name(), PyList::new(py, labels)?)?;
    }
    d.set_item("columns", columns)?;
    Ok(d)
}

fn ma_params(windows: &[usize], ewms: &[bool]) -> PyResult<Vec<MaParams>> {
    same_len(&[windows.len(), ewms.len()])?;
    Ok(windows
        .iter()
        .zip(ewms)
        .map(|(&window, &ewm)| MaParams { window, ewm })
        .collect())
}

/// Moving average.
#[pyfunction]
pub fn ind_ma<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'_, f64>,
    window: Vec<usize>,
    ewm: Vec<bool>,
    adjust: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let (close, params) = (matrix(&close)?, ma_params(&window, &ewm)?);
    let ind = py
        .detach(|| alphax_indicators::ma(&close, &params, adjust, None))
        .map_err(to_py_err)?;
    out(py, ind)
}

/// Moving standard deviation.
#[pyfunction]
pub fn ind_mstd<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'_, f64>,
    window: Vec<usize>,
    ewm: Vec<bool>,
    adjust: bool,
    ddof: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let (close, params) = (matrix(&close)?, ma_params(&window, &ewm)?);
    let ind = py
        .detach(|| alphax_indicators::mstd(&close, &params, adjust, ddof, None))
        .map_err(to_py_err)?;
    out(py, ind)
}

/// Bollinger bands.
#[pyfunction]
pub fn ind_bbands<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'_, f64>,
    window: Vec<usize>,
    ewm: Vec<bool>,
    alpha: Vec<f64>,
    adjust: bool,
    ddof: usize,
) -> PyResult<Bound<'py, PyDict>> {
    same_len(&[window.len(), ewm.len(), alpha.len()])?;
    let params: Vec<BbParams> = (0..window.len())
        .map(|i| BbParams {
            window: window[i],
            ewm: ewm[i],
            alpha: alpha[i],
        })
        .collect();
    let close = matrix(&close)?;
    let ind = py
        .detach(|| alphax_indicators::bbands(&close, &params, adjust, ddof, None))
        .map_err(to_py_err)?;
    out(py, ind)
}

/// Relative strength index.
#[pyfunction]
pub fn ind_rsi<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'_, f64>,
    window: Vec<usize>,
    ewm: Vec<bool>,
    adjust: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let (close, params) = (matrix(&close)?, ma_params(&window, &ewm)?);
    let ind = py
        .detach(|| alphax_indicators::rsi(&close, &params, adjust, None))
        .map_err(to_py_err)?;
    out(py, ind)
}

/// Stochastic oscillator.
#[pyfunction]
#[allow(
    clippy::too_many_arguments,
    reason = "three inputs and three parameter lists"
)]
pub fn ind_stoch<'py>(
    py: Python<'py>,
    high: PyReadonlyArray2<'_, f64>,
    low: PyReadonlyArray2<'_, f64>,
    close: PyReadonlyArray2<'_, f64>,
    k_window: Vec<usize>,
    d_window: Vec<usize>,
    d_ewm: Vec<bool>,
    adjust: bool,
) -> PyResult<Bound<'py, PyDict>> {
    same_len(&[k_window.len(), d_window.len(), d_ewm.len()])?;
    let params: Vec<StochParams> = (0..k_window.len())
        .map(|i| StochParams {
            k_window: k_window[i],
            d_window: d_window[i],
            d_ewm: d_ewm[i],
        })
        .collect();
    let (high, low, close) = (matrix(&high)?, matrix(&low)?, matrix(&close)?);
    let ind = py
        .detach(|| alphax_indicators::stoch(&high, &low, &close, &params, adjust, None))
        .map_err(to_py_err)?;
    out(py, ind)
}

/// Moving average convergence divergence.
#[pyfunction]
#[allow(clippy::too_many_arguments, reason = "one list per parameter")]
pub fn ind_macd<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'_, f64>,
    fast_window: Vec<usize>,
    slow_window: Vec<usize>,
    signal_window: Vec<usize>,
    macd_ewm: Vec<bool>,
    signal_ewm: Vec<bool>,
    adjust: bool,
) -> PyResult<Bound<'py, PyDict>> {
    same_len(&[
        fast_window.len(),
        slow_window.len(),
        signal_window.len(),
        macd_ewm.len(),
        signal_ewm.len(),
    ])?;
    let params: Vec<MacdParams> = (0..fast_window.len())
        .map(|i| MacdParams {
            fast_window: fast_window[i],
            slow_window: slow_window[i],
            signal_window: signal_window[i],
            macd_ewm: macd_ewm[i],
            signal_ewm: signal_ewm[i],
        })
        .collect();
    let close = matrix(&close)?;
    let ind = py
        .detach(|| alphax_indicators::macd(&close, &params, adjust, None))
        .map_err(to_py_err)?;
    out(py, ind)
}

/// Average true range.
#[pyfunction]
#[allow(
    clippy::too_many_arguments,
    reason = "three inputs and two parameter lists"
)]
pub fn ind_atr<'py>(
    py: Python<'py>,
    high: PyReadonlyArray2<'_, f64>,
    low: PyReadonlyArray2<'_, f64>,
    close: PyReadonlyArray2<'_, f64>,
    window: Vec<usize>,
    ewm: Vec<bool>,
    adjust: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let params = ma_params(&window, &ewm)?;
    let (high, low, close) = (matrix(&high)?, matrix(&low)?, matrix(&close)?);
    let ind = py
        .detach(|| alphax_indicators::atr(&high, &low, &close, &params, adjust, None))
        .map_err(to_py_err)?;
    out(py, ind)
}

/// On-balance volume.
#[pyfunction]
pub fn ind_obv<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'_, f64>,
    volume: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let (close, volume) = (matrix(&close)?, matrix(&volume)?);
    let ind = py
        .detach(|| alphax_indicators::obv(&close, &volume, None))
        .map_err(to_py_err)?;
    out(py, ind)
}

/// True range.
#[pyfunction]
pub fn ind_true_range<'py>(
    py: Python<'py>,
    high: PyReadonlyArray2<'_, f64>,
    low: PyReadonlyArray2<'_, f64>,
    close: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (high, low, close) = (matrix(&high)?, matrix(&low)?, matrix(&close)?);
    if high.n_rows() != close.n_rows() || low.n_rows() != close.n_rows() {
        return Err(PyValueError::new_err(
            "high, low and close must be the same shape",
        ));
    }
    let m = py.detach(|| alphax_indicators::true_range(&high, &low, &close));
    let (r, c) = (m.n_rows(), m.n_cols());
    columns_out(py, m.into_vec(), r, c)
}

/// One indicator per column, a bar at a time: `alphax.indicators.Stream`.
#[pyclass(name = "_IndicatorStream", module = "alphax._alphax")]
#[derive(Debug)]
pub struct PyStream {
    streams: Vec<Stream>,
}

fn param<'py, T: pyo3::conversion::FromPyObjectOwned<'py>>(
    params: &Bound<'py, PyDict>,
    name: &str,
    default: T,
) -> PyResult<T> {
    match params.get_item(name)? {
        Some(v) => v.extract().map_err(Into::into),
        None => Ok(default),
    }
}

#[pymethods]
impl PyStream {
    /// `kind` is an indicator name; `params` its parameters, as the batch
    /// function takes them but one value each, with the Python functions'
    /// defaults (`adjust=True`; `window=20` and `ddof=0` for `bbands`,
    /// `ddof=1` for `mstd`).
    #[new]
    fn new(kind: &str, n_cols: usize, params: &Bound<'_, PyDict>) -> PyResult<Self> {
        let adjust = param(params, "adjust", true)?;
        let window = param(
            params,
            "window",
            if kind == "bbands" { 20usize } else { 14 },
        )?;
        let ewm = param(params, "ewm", false)?;
        let ma = MaParams { window, ewm };
        let make = || -> PyResult<Stream> {
            let stream = match kind {
                "ma" => Stream::ma(ma, adjust),
                "mstd" => Stream::mstd(ma, adjust, param(params, "ddof", 1usize)?),
                "bbands" => Stream::bbands(
                    BbParams {
                        window,
                        ewm,
                        alpha: param(params, "alpha", 2.0)?,
                    },
                    adjust,
                    param(params, "ddof", 0usize)?,
                ),
                "rsi" => Stream::rsi(ma, adjust),
                "stoch" => Stream::stoch(
                    StochParams {
                        k_window: param(params, "k_window", 14usize)?,
                        d_window: param(params, "d_window", 3usize)?,
                        d_ewm: param(params, "d_ewm", false)?,
                    },
                    adjust,
                ),
                "macd" => Stream::macd(
                    MacdParams {
                        fast_window: param(params, "fast_window", 12usize)?,
                        slow_window: param(params, "slow_window", 26usize)?,
                        signal_window: param(params, "signal_window", 9usize)?,
                        macd_ewm: param(params, "macd_ewm", false)?,
                        signal_ewm: param(params, "signal_ewm", false)?,
                    },
                    adjust,
                ),
                "atr" => Stream::atr(ma, adjust),
                "obv" => Ok(Stream::obv()),
                other => {
                    return Err(PyValueError::new_err(format!(
                        "no streaming indicator {other:?}; there are ma, mstd, bbands, rsi, \
                         stoch, macd, atr and obv"
                    )));
                }
            };
            stream.map_err(to_py_err)
        };
        Ok(Self {
            streams: (0..n_cols).map(|_| make()).collect::<PyResult<_>>()?,
        })
    }

    #[getter]
    fn names(&self) -> Vec<&'static str> {
        self.streams
            .first()
            .map_or(Vec::new(), |s| s.names().to_vec())
    }

    /// One bar for every column; returns each output's values.
    #[pyo3(signature = (close, high=None, low=None, volume=None))]
    fn update<'py>(
        &mut self,
        py: Python<'py>,
        close: Vec<f64>,
        high: Option<Vec<f64>>,
        low: Option<Vec<f64>>,
        volume: Option<Vec<f64>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let n = self.streams.len();
        let field = |v: Option<Vec<f64>>, name: &str| -> PyResult<Vec<f64>> {
            let v = v.unwrap_or_else(|| vec![f64::NAN; n]);
            if v.len() == n {
                Ok(v)
            } else {
                Err(PyValueError::new_err(format!(
                    "{name} has {} values for {n} columns",
                    v.len()
                )))
            }
        };
        let close = field(Some(close), "close")?;
        let (high, low, volume) = (
            field(high, "high")?,
            field(low, "low")?,
            field(volume, "volume")?,
        );
        let names = self.names();
        let mut outs = vec![Vec::with_capacity(n); names.len()];
        for (c, stream) in self.streams.iter_mut().enumerate() {
            let bar = Bar {
                high: high[c],
                low: low[c],
                close: close[c],
                volume: volume[c],
            };
            for (k, &v) in stream.update(bar).iter().enumerate() {
                outs[k].push(v);
            }
        }
        let d = PyDict::new(py);
        for (name, values) in names.iter().zip(outs) {
            d.set_item(name, values.into_pyarray(py))?;
        }
        Ok(d)
    }
}

/// Adds the functions to the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStream>()?;
    m.add_function(wrap_pyfunction!(ind_ma, m)?)?;
    m.add_function(wrap_pyfunction!(ind_mstd, m)?)?;
    m.add_function(wrap_pyfunction!(ind_bbands, m)?)?;
    m.add_function(wrap_pyfunction!(ind_rsi, m)?)?;
    m.add_function(wrap_pyfunction!(ind_stoch, m)?)?;
    m.add_function(wrap_pyfunction!(ind_macd, m)?)?;
    m.add_function(wrap_pyfunction!(ind_atr, m)?)?;
    m.add_function(wrap_pyfunction!(ind_obv, m)?)?;
    m.add_function(wrap_pyfunction!(ind_true_range, m)?)?;
    Ok(())
}
