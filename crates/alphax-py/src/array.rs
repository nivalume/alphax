//! Borrowing NumPy arrays as the engine's parameters, without copying them.
//!
//! Every simulation parameter is a [`FlexArray`], which is the engine's way of
//! accepting a scalar, a value per column, a value per bar or a full matrix
//! through one code path with no allocation. Python's side of that is: a
//! number, or a 1-D array, or a 2-D array. This module is the translation, and
//! it is small except for one thing that is not.
//!
//! # The memory-order trap
//!
//! **This engine is column-major. NumPy is row-major by default.** The two
//! disagree about what the bytes mean, and the disagreement is silent.
//!
//! A `(3, 2)` array holding `0..6` is `[0,1,2,3,4,5]` in memory when C-ordered
//! and `[0,2,4,1,3,5]` when F-ordered. Read the first as column-major and
//! column 0 comes out as `[0,1,2]` where it should be `[0,2,4]` — a
//! *transposed* price series, which runs, produces fills, and reports a
//! perfectly ordinary-looking backtest on data nobody asked for.
//!
//! Three things about that are worth stating because each one removes a
//! defence someone might assume:
//!
//! - Contiguity is not the check. A C-ordered array is contiguous, so
//!   "contiguous therefore fine" is wrong.
//! - The shape is not the check either. On a square array the transposed
//!   reading has the same shape as the intended one, so nothing downstream
//!   notices.
//! - It cannot be left to the Python wrapper. A wrapper is easy to bypass —
//!   `_alphax` is importable — so the guard belongs where the bytes are actually
//!   read.
//!
//! So [`columns_of`] refuses a C-ordered 2-D array and names the fix. The
//! Python layer calls `np.asfortranarray` first, which is free when the array
//! is already F-ordered and a visible copy when it is not.
//!
//! 1-D arrays have no such problem: one row or one column of anything is
//! contiguous either way.

use alphax_core::flex::FlexArray;
use numpy::{Element, IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;

/// An owned buffer plus the shape it was read at.
///
/// Owned because a [`FlexArray`] borrows, and the borrow has to outlive the
/// call that built it. **The bytes are not copied** — [`Borrowed::Slice`]
/// holds a borrow of NumPy's own buffer; the owned variants exist only for the
/// cases where Python handed over a scalar or a list rather than an array.
#[derive(Debug)]
pub enum Borrowed<'py> {
    /// One value for everything.
    Scalar(f64),
    /// A borrow of NumPy's buffer, in the engine's layout already.
    Slice(PyReadonlyArray2<'py, f64>),
    /// A borrow of a 1-D NumPy buffer.
    Row(PyReadonlyArray1<'py, f64>),
}

/// What a 1-D array means, which the engine cannot infer.
///
/// A 1-D array of length `n` against an `(n_rows, n_cols)` shape is ambiguous
/// whenever `n_rows == n_cols`, and means different things otherwise. The
/// caller says which; guessing from the length would be right most of the time
/// and silently wrong the rest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OneDim {
    /// One value per bar, shared by every column.
    PerRow,
    /// One value per column, shared by every bar.
    PerColumn,
}

/// The engine's column-major buffer for a 2-D array, borrowed not copied.
///
/// # Errors
///
/// Returns a Python `ValueError` if the array is not contiguous, or is
/// C-ordered — see the module docs for why the second is a refusal rather than
/// something to transparently fix here.
pub fn columns_of<'a, T: Element>(
    array: &'a PyReadonlyArray2<'a, T>,
) -> PyResult<(&'a [T], usize, usize)> {
    let view = array.as_array();
    let (n_rows, n_cols) = view.dim();
    // A single column is contiguous in both orders, so the stride test would
    // reject a perfectly usable array without this.
    let f_ordered = n_rows == 1 || n_cols == 1 || view.strides()[0] == 1;
    if !f_ordered {
        return Err(PyValueError::new_err(format!(
            "this array is C-ordered ({n_rows}x{n_cols}, strides {:?}) and this engine is \
             column-major. Pass np.asfortranarray(x). Reading it as given would backtest \
             the transposed series and report an ordinary-looking result",
            view.strides()
        )));
    }
    // `to_slice_memory_order` returns the buffer in whatever order it is
    // stored, so for an F-ordered array that is exactly column-major. A second
    // `as_array()` rather than reusing `view` above: on a view the method
    // returns a slice borrowing the *array's* data, not the view, and that is
    // the lifetime this function has to hand back.
    let flat = array.as_array().to_slice_memory_order().ok_or_else(|| {
        PyValueError::new_err(
            "the array is not contiguous — a slice like x[:, ::2] cannot be \
             borrowed; pass np.ascontiguousarray(x, order='F')",
        )
    })?;
    Ok((flat, n_rows, n_cols))
}

/// A column-major buffer handed back as an `(n_rows, n_cols)` NumPy array.
///
/// The buffer moves into NumPy and is viewed F-ordered, so nothing is copied
/// and nothing is transposed: [`columns_of`] in reverse.
///
/// # Errors
///
/// A Python `ValueError` if `data` is not `n_rows * n_cols` long.
pub fn columns_out<T: Element>(
    py: Python<'_>,
    data: Vec<T>,
    n_rows: usize,
    n_cols: usize,
) -> PyResult<Bound<'_, PyArray2<T>>> {
    data.into_pyarray(py)
        .reshape_with_order([n_rows, n_cols], numpy::npyffi::NPY_ORDER::NPY_FORTRANORDER)
}

/// Reads whatever Python passed into a float [`FlexArray`].
///
/// Accepts a number, a 1-D array or a 2-D array.
///
/// # Errors
///
/// A Python `TypeError` for anything else, and whatever [`columns_of`] rejects.
pub fn flex_f64<'a>(
    value: &'a Borrowed<'a>,
    n_rows: usize,
    n_cols: usize,
    one_dim: OneDim,
) -> PyResult<FlexArray<'a, f64>> {
    match value {
        Borrowed::Scalar(v) => Ok(FlexArray::Scalar(*v)),
        Borrowed::Row(a) => {
            let slice = a.as_slice().map_err(|e| {
                PyValueError::new_err(format!("the 1-D array is not contiguous: {e}"))
            })?;
            let built = match one_dim {
                OneDim::PerRow => FlexArray::per_row(slice, n_rows),
                OneDim::PerColumn => FlexArray::per_column(slice, n_cols),
            };
            built.map_err(|e| PyValueError::new_err(e.to_string()))
        }
        Borrowed::Slice(a) => {
            let (flat, rows, cols) = columns_of(a)?;
            if rows != n_rows || cols != n_cols {
                return Err(PyValueError::new_err(format!(
                    "a {rows}x{cols} array against a {n_rows}x{n_cols} run"
                )));
            }
            FlexArray::full(flat, n_rows, n_cols).map_err(|e| PyValueError::new_err(e.to_string()))
        }
    }
}

/// Reads whatever Python passed, keeping the borrow alive.
///
/// Separate from [`flex_f64`] because the borrow has to be held by the caller
/// for as long as the `FlexArray` lives, and a function returning both cannot
/// express that.
///
/// # Errors
///
/// A Python `TypeError` for a value that is neither a number nor a float
/// array.
pub fn borrow_f64<'py>(value: &Bound<'py, PyAny>, name: &str) -> PyResult<Borrowed<'py>> {
    if let Ok(v) = value.extract::<f64>() {
        return Ok(Borrowed::Scalar(v));
    }
    if let Ok(a) = value.extract::<PyReadonlyArray2<'py, f64>>() {
        return Ok(Borrowed::Slice(a));
    }
    if let Ok(a) = value.extract::<PyReadonlyArray1<'py, f64>>() {
        return Ok(Borrowed::Row(a));
    }
    Err(PyTypeError::new_err(format!(
        "{name} must be a float, or a 1-D or 2-D float64 array; got {}. An integer \
         array is not accepted, because silently viewing it as float would copy",
        value
            .get_type()
            .name()
            .map_or_else(|_| "an unknown type".to_owned(), |n| n.to_string())
    )))
}

/// A boolean buffer, borrowed. The signal arrays' equivalent of [`Borrowed`].
#[derive(Debug)]
pub enum BorrowedBool<'py> {
    /// One value for everything.
    Scalar(bool),
    /// A borrow of a 2-D NumPy buffer.
    Slice(PyReadonlyArray2<'py, bool>),
}

/// Reads a bool parameter, keeping the borrow alive.
///
/// # Errors
///
/// A Python `TypeError` for a value that is neither a bool nor a bool array.
pub fn borrow_bool<'py>(value: &Bound<'py, PyAny>, name: &str) -> PyResult<BorrowedBool<'py>> {
    if let Ok(v) = value.extract::<bool>() {
        return Ok(BorrowedBool::Scalar(v));
    }
    if let Ok(a) = value.extract::<PyReadonlyArray2<'py, bool>>() {
        return Ok(BorrowedBool::Slice(a));
    }
    Err(PyTypeError::new_err(format!(
        "{name} must be a bool or a 2-D bool array; got {}",
        value
            .get_type()
            .name()
            .map_or_else(|_| "an unknown type".to_owned(), |n| n.to_string())
    )))
}

/// The engine's view of a bool parameter.
///
/// # Errors
///
/// As [`flex_f64`].
pub fn flex_bool<'a>(
    value: &'a BorrowedBool<'a>,
    n_rows: usize,
    n_cols: usize,
) -> PyResult<FlexArray<'a, bool>> {
    match value {
        BorrowedBool::Scalar(v) => Ok(FlexArray::Scalar(*v)),
        BorrowedBool::Slice(a) => {
            let view = a.as_array();
            let (rows, cols) = view.dim();
            if rows != n_rows || cols != n_cols {
                return Err(PyValueError::new_err(format!(
                    "a {rows}x{cols} array against a {n_rows}x{n_cols} run"
                )));
            }
            // Same trap as the float case, same guard. A transposed signal
            // matrix is arguably worse: it produces entries on the wrong bars
            // for the wrong column and still looks like a strategy.
            let f_ordered = rows == 1 || cols == 1 || view.strides()[0] == 1;
            if !f_ordered {
                return Err(PyValueError::new_err(format!(
                    "this bool array is C-ordered ({rows}x{cols}); this engine is \
                     column-major. Pass np.asfortranarray(x)"
                )));
            }
            let flat = a
                .as_array()
                .to_slice_memory_order()
                .ok_or_else(|| PyValueError::new_err("the array is not contiguous".to_owned()))?;
            FlexArray::full(flat, n_rows, n_cols).map_err(|e| PyValueError::new_err(e.to_string()))
        }
    }
}

/// Reads an optional float parameter, keeping the borrow alive.
///
/// # Why every parameter's Python default is `None`
///
/// The default for `fees` is `0.0`, for `size` it is `inf`, for
/// `size_granularity` it is `NaN` — and each of those lives in
/// `FromSignals::default()`. Repeating them in this layer's signature would
/// create a second copy that compiles fine while disagreeing with the engine,
/// which is the kind of drift no test looks for. So the signature says only
/// "not supplied" and [`flex_or_f64`] substitutes the engine's own value.
///
/// # Errors
///
/// As [`borrow_f64`].
pub fn borrow_opt_f64<'py>(
    value: Option<&Bound<'py, PyAny>>,
    name: &str,
) -> PyResult<Option<Borrowed<'py>>> {
    value.map(|v| borrow_f64(v, name)).transpose()
}

/// The engine's view of an optional float parameter, or the engine's default.
///
/// # Errors
///
/// As [`flex_f64`].
pub fn flex_or_f64<'a>(
    value: &'a Option<Borrowed<'a>>,
    n_rows: usize,
    n_cols: usize,
    one_dim: OneDim,
    default: FlexArray<'a, f64>,
) -> PyResult<FlexArray<'a, f64>> {
    match value {
        Some(v) => flex_f64(v, n_rows, n_cols, one_dim),
        None => Ok(default),
    }
}

/// The steps of a stop ladder: a number is one step, a 1-D array the steps
/// of every column, and an `(n_steps, n_cols)` array each column's own.
///
/// # Errors
///
/// A Python `ValueError` for a 2-D array with the wrong number of columns,
/// or one that is not contiguous.
pub fn ladder_steps<'a>(
    value: &'a Option<Borrowed<'a>>,
    n_cols: usize,
    name: &str,
    default: FlexArray<'a, f64>,
) -> PyResult<FlexArray<'a, f64>> {
    match value {
        None => Ok(default),
        Some(Borrowed::Scalar(v)) => Ok(FlexArray::Scalar(*v)),
        Some(Borrowed::Row(a)) => {
            let steps = a.as_slice().map_err(|e| {
                PyValueError::new_err(format!("{name}'s steps are not contiguous: {e}"))
            })?;
            FlexArray::per_row(steps, steps.len()).map_err(to_py_err)
        }
        Some(Borrowed::Slice(a)) => {
            let (data, rows, cols) = columns_of(a)?;
            if cols != n_cols {
                return Err(PyValueError::new_err(format!(
                    "a {name} ladder must have one column of steps per column; got {cols} for \
                     {n_cols}"
                )));
            }
            FlexArray::full(data, rows, n_cols).map_err(to_py_err)
        }
    }
}

/// Reads an optional bool parameter, keeping the borrow alive.
///
/// # Errors
///
/// As [`borrow_bool`].
pub fn borrow_opt_bool<'py>(
    value: Option<&Bound<'py, PyAny>>,
    name: &str,
) -> PyResult<Option<BorrowedBool<'py>>> {
    value.map(|v| borrow_bool(v, name)).transpose()
}

/// The engine's view of an optional bool parameter, or the engine's default.
///
/// # Errors
///
/// As [`flex_bool`].
pub fn flex_or_bool<'a>(
    value: &'a Option<BorrowedBool<'a>>,
    n_rows: usize,
    n_cols: usize,
    default: FlexArray<'a, bool>,
) -> PyResult<FlexArray<'a, bool>> {
    match value {
        Some(v) => flex_bool(v, n_rows, n_cols),
        None => Ok(default),
    }
}

/// Turns an engine error into a Python exception, keeping the message.
///
/// The engine's errors say things like "order rejected at row 3, column 1:
/// insufficient cash". Collapsing that to an error code, which a hand-written
/// C ABI would have to, throws away the only part a user can act on.
pub fn to_py_err(e: alphax_core::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// A count parameter (`from_ago`), borrowed: a number, or a 2-D `uintp`
/// array in the engine's column-major layout.
#[derive(Debug)]
pub enum BorrowedCount<'py> {
    /// One value for everything.
    Scalar(usize),
    /// A borrow of a 2-D NumPy buffer.
    Slice(PyReadonlyArray2<'py, usize>),
}

/// Reads a count parameter, keeping the borrow alive.
///
/// # Errors
///
/// A Python `TypeError` for a value that is neither a nonnegative integer
/// nor a 2-D `uintp` array.
pub fn borrow_count<'py>(value: &Bound<'py, PyAny>, name: &str) -> PyResult<BorrowedCount<'py>> {
    if let Ok(v) = value.extract::<usize>() {
        return Ok(BorrowedCount::Scalar(v));
    }
    if let Ok(a) = value.extract::<PyReadonlyArray2<'py, usize>>() {
        return Ok(BorrowedCount::Slice(a));
    }
    Err(PyTypeError::new_err(format!(
        "{name} must be a nonnegative integer or a 2-D uintp array"
    )))
}

/// The engine's view of a count parameter.
///
/// # Errors
///
/// As [`flex_bool`].
pub fn flex_count<'a>(
    value: &'a BorrowedCount<'a>,
    n_rows: usize,
    n_cols: usize,
) -> PyResult<FlexArray<'a, usize>> {
    match value {
        BorrowedCount::Scalar(v) => Ok(FlexArray::Scalar(*v)),
        BorrowedCount::Slice(a) => {
            let (flat, rows, cols) = columns_of(a)?;
            if rows != n_rows || cols != n_cols {
                return Err(PyValueError::new_err(format!(
                    "a {rows}x{cols} array against a {n_rows}x{n_cols} run"
                )));
            }
            FlexArray::full(flat, n_rows, n_cols).map_err(|e| PyValueError::new_err(e.to_string()))
        }
    }
}
