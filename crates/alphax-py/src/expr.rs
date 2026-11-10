//! Python boundary for the factor-expression engine.

use alphax_expr::{
    Alpha158Config, Alpha158Price, Alpha158Rolling, FactorGrid, FactorInputs, FactorSet, PitRecord,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

use crate::array::{columns_of, to_py_err};

/// The crossings, as one column-major boolean array the caller reshapes with
/// `order="F"`. Returned flat rather than as nested lists: a list of lists is
/// one Python object per cell, and `np.asarray` over it was the slowest step
/// of a sweep that used it.
#[pyfunction]
pub fn crossed_above<'py>(
    py: Python<'py>,
    left: PyReadonlyArray2<'_, f64>,
    right: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let (a, rows, cols) = columns_of(&left)?;
    let (b, other_rows, other_cols) = columns_of(&right)?;
    if (rows, cols) != (other_rows, other_cols) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "crossed_above inputs must have the same shape",
        ));
    }
    let flat = py.detach(|| {
        let mut out = Vec::with_capacity(rows * cols);
        for col in 0..cols {
            let start = col * rows;
            out.extend(alphax_generic::crossed_above_1d(
                &a[start..start + rows],
                &b[start..start + rows],
                0,
            ));
        }
        out
    });
    Ok(flat.into_pyarray(py))
}

/// F-ordered float64 fields by name.
type Fields<'py> = Vec<(String, PyReadonlyArray2<'py, f64>)>;
/// Per instrument outside the panel: its one-dimensional fields by name.
type Instruments<'py> = Vec<(String, Vec<(String, PyReadonlyArray1<'py, f64>)>)>;
/// Per point-in-time field: the records' columns, days since 1970-01-01,
/// periods and values.
type Pit<'py> = Vec<(
    String,
    PyReadonlyArray1<'py, i64>,
    PyReadonlyArray1<'py, i64>,
    PyReadonlyArray1<'py, i64>,
    PyReadonlyArray1<'py, f64>,
)>;

/// Everything an evaluation reads besides the panel's fields.
struct Context<'py> {
    shape: (usize, usize),
    timestamps: Option<PyReadonlyArray1<'py, i64>>,
    symbols: Option<Vec<String>>,
    instruments: Instruments<'py>,
    pit: Pit<'py>,
}

/// A compiled set of named factor expressions.
#[pyclass(name = "_FactorSet", module = "alphax._alphax")]
#[derive(Debug)]
pub struct PyFactorSet {
    inner: FactorSet,
}

#[pymethods]
impl PyFactorSet {
    /// Compiles `(name, expression)` pairs in their supplied order.
    #[staticmethod]
    fn compile(factors: Vec<(String, String)>) -> PyResult<Self> {
        Ok(Self {
            inner: FactorSet::compile(factors).map_err(to_py_err)?,
        })
    }

    #[getter]
    fn names(&self) -> Vec<String> {
        self.inner.names().map(str::to_owned).collect()
    }

    #[getter]
    fn required_fields(&self) -> Vec<String> {
        self.inner
            .required_fields()
            .into_iter()
            .map(str::to_owned)
            .collect()
    }

    #[getter]
    fn uses_future(&self) -> bool {
        self.inner.uses_future()
    }

    #[getter]
    fn instrument_names(&self) -> Vec<String> {
        self.inner.instrument_names()
    }

    #[getter]
    fn needs_timestamps(&self) -> bool {
        self.inner.needs_timestamps()
    }

    #[getter]
    fn lookbacks(&self) -> Vec<usize> {
        self.inner.lookbacks()
    }

    #[getter]
    fn warmup(&self) -> bool {
        self.inner.warmup()
    }

    /// The same set with warm-up on or off.
    fn with_warmup(&self, on: bool) -> Self {
        Self {
            inner: self.inner.clone().with_warmup(on),
        }
    }

    /// Evaluates F-ordered float64 fields of `shape` and returns column-major
    /// flat arrays. `timestamps` are wall-clock nanoseconds per row.
    #[pyo3(signature = (fields, shape, timestamps=None, symbols=None, instruments=Vec::new(), pit=Vec::new()))]
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        fields: Fields<'py>,
        shape: (usize, usize),
        timestamps: Option<PyReadonlyArray1<'py, i64>>,
        symbols: Option<Vec<String>>,
        instruments: Instruments<'py>,
        pit: Pit<'py>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let context = Context {
            shape,
            timestamps,
            symbols,
            instruments,
            pit,
        };
        let inputs = inputs_of(&fields, &context)?;

        // The inputs borrow NumPy buffers that stay alive for this call, and
        // the evaluation touches no Python object, so other threads may run.
        let values = py
            .detach(|| self.inner.evaluate(&inputs))
            .map_err(to_py_err)?;
        values_dict(py, values)
    }

    /// Evaluates `batch` symbols at a time and calls `sink(start, stop,
    /// values)` for each batch, `values` as [`Self::evaluate`] returns them.
    #[pyo3(signature = (fields, shape, batch, sink, timestamps=None, symbols=None, instruments=Vec::new(), pit=Vec::new()))]
    #[allow(clippy::too_many_arguments, reason = "the Python keyword arguments")]
    fn evaluate_batches<'py>(
        &self,
        py: Python<'py>,
        fields: Fields<'py>,
        shape: (usize, usize),
        batch: usize,
        sink: Bound<'py, PyAny>,
        timestamps: Option<PyReadonlyArray1<'py, i64>>,
        symbols: Option<Vec<String>>,
        instruments: Instruments<'py>,
        pit: Pit<'py>,
    ) -> PyResult<()> {
        let context = Context {
            shape,
            timestamps,
            symbols,
            instruments,
            pit,
        };
        let inputs = inputs_of(&fields, &context)?;
        let mut failure = None;
        let run = self.inner.evaluate_batches(&inputs, batch, |cols, values| {
            let called = values_dict(py, values)
                .and_then(|d| sink.call1((cols.start, cols.end, d)).map(|_| ()));
            called.map_err(|e| {
                failure = Some(e);
                alphax_core::Error::Invariant("the sink raised".into())
            })
        });
        if let Some(error) = failure {
            return Err(error);
        }
        run.map_err(to_py_err)
    }
}

/// Factor expressions with `{name}` placeholders compiled for every case of a
/// parameter grid.
#[pyclass(name = "_FactorGrid", module = "alphax._alphax")]
#[derive(Debug)]
pub struct PyFactorGrid {
    inner: FactorGrid,
}

#[pymethods]
impl PyFactorGrid {
    /// Compiles `(name, expression)` pairs for every case, each case a list
    /// of `(parameter, value text)`.
    #[staticmethod]
    fn compile(
        factors: Vec<(String, String)>,
        cases: Vec<Vec<(String, String)>>,
    ) -> PyResult<Self> {
        let cases = cases
            .into_iter()
            .map(|case| case.into_iter().collect())
            .collect();
        Ok(Self {
            inner: FactorGrid::compile(factors, cases).map_err(to_py_err)?,
        })
    }

    #[getter]
    fn names(&self) -> Vec<String> {
        self.inner.names().map(str::to_owned).collect()
    }

    #[getter]
    fn lookbacks(&self) -> Vec<Vec<usize>> {
        self.inner.lookbacks()
    }

    #[getter]
    fn uses_future(&self) -> bool {
        self.inner.set().uses_future()
    }

    #[getter]
    fn needs_timestamps(&self) -> bool {
        self.inner.set().needs_timestamps()
    }

    #[getter]
    fn instrument_names(&self) -> Vec<String> {
        self.inner.set().instrument_names()
    }

    #[getter]
    fn required_fields(&self) -> Vec<String> {
        self.inner
            .set()
            .required_fields()
            .into_iter()
            .map(str::to_owned)
            .collect()
    }

    #[getter]
    fn warmup(&self) -> bool {
        self.inner.set().warmup()
    }

    fn with_warmup(&self, on: bool) -> Self {
        Self {
            inner: self.inner.clone().with_warmup(on),
        }
    }

    /// As [`PyFactorSet::evaluate`]; each factor is `(rows, cases * symbols)`.
    #[pyo3(signature = (fields, shape, timestamps=None, symbols=None, instruments=Vec::new(), pit=Vec::new()))]
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        fields: Fields<'py>,
        shape: (usize, usize),
        timestamps: Option<PyReadonlyArray1<'py, i64>>,
        symbols: Option<Vec<String>>,
        instruments: Instruments<'py>,
        pit: Pit<'py>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let context = Context {
            shape,
            timestamps,
            symbols,
            instruments,
            pit,
        };
        let inputs = inputs_of(&fields, &context)?;
        let values = py
            .detach(|| self.inner.evaluate(&inputs))
            .map_err(to_py_err)?;
        values_dict(py, values)
    }
}

fn values_dict(py: Python<'_>, values: alphax_expr::FactorValues) -> PyResult<Bound<'_, PyDict>> {
    let out = PyDict::new(py);
    for (name, value) in values.into_matrices() {
        out.set_item(name, value.into_vec().into_pyarray(py))?;
    }
    Ok(out)
}

fn inputs_of<'a>(fields: &'a Fields<'_>, context: &'a Context<'_>) -> PyResult<FactorInputs<'a>> {
    let invalid = |message: String| to_py_err(alphax_core::Error::Invariant(message));
    let (n_rows, n_cols) = context.shape;
    let mut inputs = FactorInputs::new(n_rows, n_cols).map_err(to_py_err)?;
    for (name, array) in fields {
        let (values, rows, cols) = columns_of(array)?;
        if (rows, cols) != (n_rows, n_cols) {
            return Err(invalid(format!(
                "factor field {name:?} has shape {rows}x{cols}, expected {n_rows}x{n_cols}"
            )));
        }
        inputs = inputs.with_field(name, values).map_err(to_py_err)?;
    }
    if let Some(timestamps) = &context.timestamps {
        inputs = inputs
            .with_timestamps(timestamps.as_slice()?)
            .map_err(to_py_err)?;
    }
    if let Some(symbols) = &context.symbols {
        inputs = inputs.with_symbols(symbols).map_err(to_py_err)?;
    }
    for (name, own) in &context.instruments {
        let mut instrument = FactorInputs::new(n_rows, 1).map_err(to_py_err)?;
        for (field, values) in own {
            instrument = instrument
                .with_field(field, values.as_slice()?)
                .map_err(to_py_err)?;
        }
        inputs = inputs
            .with_instrument(name, instrument)
            .map_err(to_py_err)?;
    }
    for (field, cols, days, periods, values) in &context.pit {
        let (cols, days, periods, values) = (
            cols.as_slice()?,
            days.as_slice()?,
            periods.as_slice()?,
            values.as_slice()?,
        );
        if [days.len(), periods.len(), values.len()] != [cols.len(); 3] {
            return Err(invalid(format!(
                "point-in-time field {field:?} has columns of different lengths"
            )));
        }
        let records = (0..cols.len())
            .map(|k| {
                Ok(PitRecord {
                    col: usize::try_from(cols[k])
                        .map_err(|_| invalid(format!("{field:?}: a negative symbol index")))?,
                    day: days[k],
                    period: u32::try_from(periods[k]).map_err(|_| {
                        invalid(format!("{field:?}: invalid period {}", periods[k]))
                    })?,
                    value: values[k],
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        inputs = inputs.with_pit(field, records).map_err(to_py_err)?;
    }
    Ok(inputs)
}

/// Qlib's Alpha158 config as plain values: `kbar`, `price` as `(windows,
/// features)`, `volume` windows and `rolling` as `(windows, include,
/// exclude)`; an absent section is `None`.
type Alpha158Spec = (
    bool,
    Option<(Vec<usize>, Vec<String>)>,
    Option<Vec<usize>>,
    Option<(Vec<usize>, Option<Vec<String>>, Vec<String>)>,
);

fn alpha158_config(spec: Option<Alpha158Spec>) -> Alpha158Config {
    match spec {
        None => Alpha158Config::default(),
        Some((kbar, price, volume, rolling)) => Alpha158Config {
            kbar,
            price: price.map(|(windows, features)| Alpha158Price { windows, features }),
            volume,
            rolling: rolling.map(|(windows, include, exclude)| Alpha158Rolling {
                windows,
                include,
                exclude,
            }),
        },
    }
}

/// Qlib's Alpha158 factor set: the default, or the features `config`
/// selects.
#[pyfunction]
#[pyo3(signature = (config=None))]
pub fn alpha158(config: Option<Alpha158Spec>) -> PyResult<PyFactorSet> {
    Ok(PyFactorSet {
        inner: alphax_expr::alpha158_with(&alpha158_config(config)).map_err(to_py_err)?,
    })
}

/// The `(name, expression)` pairs of the Alpha158 features `config` selects.
#[pyfunction]
#[pyo3(signature = (config=None))]
pub fn alpha158_features(config: Option<Alpha158Spec>) -> PyResult<Vec<(String, String)>> {
    alphax_expr::alpha158_features(&alpha158_config(config)).map_err(to_py_err)
}

/// Qlib's label `LABEL0` on `price`.
#[pyfunction]
pub fn qlib_label(price: &str) -> PyResult<PyFactorSet> {
    Ok(PyFactorSet {
        inner: alphax_expr::qlib_label(price).map_err(to_py_err)?,
    })
}

/// Qlib's default Alpha360 factor set.
#[pyfunction]
pub fn alpha360() -> PyResult<PyFactorSet> {
    Ok(PyFactorSet {
        inner: alphax_expr::alpha360().map_err(to_py_err)?,
    })
}

/// WorldQuant's 101 alphas: the 82 that read only prices and volume, or all
/// 101 (`all`), which also read `cap`, `sector`, `industry`, `subindustry`.
#[pyfunction]
#[pyo3(signature = (all=false))]
pub fn wq101(all: bool) -> PyResult<PyFactorSet> {
    let inner = if all {
        alphax_expr::wq101_all()
    } else {
        alphax_expr::wq101()
    };
    Ok(PyFactorSet {
        inner: inner.map_err(to_py_err)?,
    })
}

/// The paper's formula for alpha `k`.
#[pyfunction]
pub fn wq101_formula(k: usize) -> Option<&'static str> {
    alphax_expr::wq101_formula(k)
}
