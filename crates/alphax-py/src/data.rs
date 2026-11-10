//! Identity bindings for local sources; reading them is `python/alphax/data.py`.
use alphax_model::{BarType, ids::InstrumentId};
use pyo3::{exceptions::PyValueError, prelude::*};

fn bad(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

#[pyclass(
    name = "InstrumentId",
    module = "alphax",
    frozen,
    eq,
    hash,
    skip_from_py_object
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PyInstrumentId(pub InstrumentId);
#[pymethods]
impl PyInstrumentId {
    #[new]
    fn new(symbol: &str, venue: &str) -> PyResult<Self> {
        Ok(Self(InstrumentId::from_parts(symbol, venue).map_err(bad)?))
    }
    #[staticmethod]
    fn parse(value: &str) -> PyResult<Self> {
        Ok(Self(value.parse().map_err(bad)?))
    }
    #[getter]
    fn symbol(&self) -> &str {
        self.0.symbol.as_str()
    }
    #[getter]
    fn venue(&self) -> &str {
        self.0.venue.as_str()
    }
    fn __str__(&self) -> String {
        self.0.to_string()
    }
    fn __repr__(&self) -> String {
        format!("InstrumentId('{}')", self.0)
    }
}

#[pyclass(
    name = "BarType",
    module = "alphax",
    frozen,
    eq,
    hash,
    skip_from_py_object
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PyBarType(pub BarType);
#[pymethods]
impl PyBarType {
    #[staticmethod]
    fn parse(value: &str) -> PyResult<Self> {
        Ok(Self(value.parse().map_err(bad)?))
    }
    #[getter]
    fn instrument(&self) -> PyInstrumentId {
        PyInstrumentId(self.0.instrument.clone())
    }
    #[getter]
    fn step(&self) -> u64 {
        self.0.step
    }
    #[getter]
    fn aggregation(&self) -> &str {
        &self.0.aggregation
    }
    #[getter]
    fn price_type(&self) -> &str {
        &self.0.price_type
    }
    #[getter]
    fn source(&self) -> &str {
        &self.0.source
    }
    fn __str__(&self) -> String {
        self.0.to_string()
    }
    fn __repr__(&self) -> String {
        format!("BarType('{}')", self.0)
    }
}
