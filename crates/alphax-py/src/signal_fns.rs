//! Signal cleaning, generated exits, ranges and ranks: the Rust side of
//! `alphax.signals`. Arrays are `(n_rows, n_cols)` booleans, column-major.

use alphax_signals::{ExitWindow, FirstSignal, Gaps, LeadingRun, RangeRecords};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::array::{columns_of, columns_out, to_py_err};

type Columns<'py, T> = Bound<'py, PyArray2<T>>;

fn window(name: &str) -> PyResult<ExitWindow> {
    match name {
        "next_entry" => Ok(ExitWindow::UntilNextEntry),
        "end_skip" => Ok(ExitWindow::ToEndSkippingResolved),
        "end" => Ok(ExitWindow::ToEnd),
        other => Err(PyValueError::new_err(format!(
            "until must be 'next_entry', 'end_skip' or 'end'; got {other:?}"
        ))),
    }
}

fn same_shape(a: (usize, usize), b: (usize, usize), what: &str) -> PyResult<()> {
    if a == b {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{what} must be the same shape: {a:?} and {b:?}"
        )))
    }
}

/// Entries and exits made to alternate, starting with an entry (or with
/// either, `entry_first=False`); a bar with both is neither.
#[pyfunction]
pub fn sig_clean_enex<'py>(
    py: Python<'py>,
    entries: PyReadonlyArray2<'_, bool>,
    exits: PyReadonlyArray2<'_, bool>,
    entry_first: bool,
) -> PyResult<(Columns<'py, bool>, Columns<'py, bool>)> {
    let (en, n_rows, n_cols) = columns_of(&entries)?;
    let (ex, r2, c2) = columns_of(&exits)?;
    same_shape((n_rows, n_cols), (r2, c2), "entries and exits")?;
    let first = if entry_first {
        FirstSignal::EntryFirst
    } else {
        FirstSignal::Either
    };
    let (a, b) = py
        .detach(|| alphax_signals::clean_enex(en, ex, n_rows, n_cols, first))
        .map_err(to_py_err)?;
    Ok((
        columns_out(py, a, n_rows, n_cols)?,
        columns_out(py, b, n_rows, n_cols)?,
    ))
}

/// An exit `bars` after each entry.
#[pyfunction]
pub fn sig_exit_after<'py>(
    py: Python<'py>,
    entries: PyReadonlyArray2<'_, bool>,
    bars: usize,
    until: &str,
) -> PyResult<Columns<'py, bool>> {
    let (en, n_rows, n_cols) = columns_of(&entries)?;
    let w = window(until)?;
    let out = py
        .detach(|| alphax_signals::exit_after(en, n_rows, n_cols, bars, w))
        .map_err(to_py_err)?;
    columns_out(py, out, n_rows, n_cols)
}

/// An exit on the first bar after each entry (and `wait` bars) where
/// `condition` holds.
#[pyfunction]
pub fn sig_exit_on<'py>(
    py: Python<'py>,
    entries: PyReadonlyArray2<'_, bool>,
    condition: PyReadonlyArray2<'_, bool>,
    wait: usize,
    until: &str,
) -> PyResult<Columns<'py, bool>> {
    let (en, n_rows, n_cols) = columns_of(&entries)?;
    let (cond, r2, c2) = columns_of(&condition)?;
    same_shape((n_rows, n_cols), (r2, c2), "entries and condition")?;
    let w = window(until)?;
    let out = py
        .detach(|| alphax_signals::exit_on(en, cond, n_rows, n_cols, wait, w))
        .map_err(to_py_err)?;
    columns_out(py, out, n_rows, n_cols)
}

fn ranges_out<'py>(py: Python<'py>, r: RangeRecords) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("duration", r.duration().into_pyarray(py))?;
    d.set_item("col", r.col.into_pyarray(py))?;
    d.set_item("start_idx", r.start_idx.into_pyarray(py))?;
    d.set_item("end_idx", r.end_idx.into_pyarray(py))?;
    let status: Vec<i64> = r.status.iter().map(|&s| i64::from(s)).collect();
    d.set_item("status", status.into_pyarray(py))?;
    Ok(d)
}

/// The array's ranges as records: `"partition"` (runs of `True`),
/// `"between"` (one signal to the next) or `"between_partition"` (the end of
/// one run to the start of the next).
#[pyfunction]
pub fn sig_ranges<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, bool>,
    kind: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let (flat, n_rows, n_cols) = columns_of(&a)?;
    let f = match kind {
        "partition" => alphax_signals::partition_ranges,
        "between" => alphax_signals::between_ranges,
        "between_partition" => alphax_signals::between_partition_ranges,
        other => {
            return Err(PyValueError::new_err(format!(
                "kind must be 'partition', 'between' or 'between_partition'; got {other:?}"
            )));
        }
    };
    let r = py.detach(|| f(flat, n_rows, n_cols)).map_err(to_py_err)?;
    ranges_out(py, r)
}

/// Each `True`'s rank in its run (from 1; 0 where `False`), or with
/// `partition` each run's rank; `reset_by` restarts the count.
#[pyfunction]
#[pyo3(signature = (a, reset_by, partition, allow_gaps, skip_leading))]
pub fn sig_rank<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, bool>,
    reset_by: Option<PyReadonlyArray2<'_, bool>>,
    partition: bool,
    allow_gaps: bool,
    skip_leading: bool,
) -> PyResult<Columns<'py, i64>> {
    let (flat, n_rows, n_cols) = columns_of(&a)?;
    let reset = match &reset_by {
        Some(r) => {
            let (flat_r, r2, c2) = columns_of(r)?;
            same_shape((n_rows, n_cols), (r2, c2), "a and reset_by")?;
            Some(flat_r)
        }
        None => None,
    };
    let leading = if skip_leading {
        LeadingRun::Skip
    } else {
        LeadingRun::Rank
    };
    let gaps = if allow_gaps {
        Gaps::Allow
    } else {
        Gaps::ResetEachRun
    };
    let out = py.detach(|| {
        if partition {
            alphax_signals::partition_rank(flat, reset, n_rows, n_cols, leading)
        } else {
            alphax_signals::signal_rank(flat, reset, n_rows, n_cols, gaps, leading)
        }
    });
    columns_out(py, out, n_rows, n_cols)
}

/// Each column's `n`th `True` row (negative counts from the end); -1 where
/// there is none.
#[pyfunction]
pub fn sig_nth_index<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, bool>,
    n: i64,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let (flat, n_rows, n_cols) = columns_of(&a)?;
    Ok(alphax_signals::nth_index(flat, n_rows, n_cols, n).into_pyarray(py))
}

/// Each column's mean `True` row, scaled to -1 (the start) to 1 (the end).
#[pyfunction]
pub fn sig_normalized_average_index<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, bool>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (flat, n_rows, n_cols) = columns_of(&a)?;
    Ok(alphax_signals::normalized_average_index(flat, n_rows, n_cols).into_pyarray(py))
}

/// Adds the functions to the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sig_clean_enex, m)?)?;
    m.add_function(wrap_pyfunction!(sig_exit_after, m)?)?;
    m.add_function(wrap_pyfunction!(sig_exit_on, m)?)?;
    m.add_function(wrap_pyfunction!(sig_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(sig_rank, m)?)?;
    m.add_function(wrap_pyfunction!(sig_nth_index, m)?)?;
    m.add_function(wrap_pyfunction!(sig_normalized_average_index, m)?)?;
    Ok(())
}
