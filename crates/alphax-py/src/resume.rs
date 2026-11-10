//! Continuing a portfolio's run, and keeping a portfolio as bytes.
//!
//! `from_signals(..., resume=pf)` (and `from_orders`) takes the whole
//! history again, the old bars and the new, simulates only the bars after
//! `pf`'s last, and returns one portfolio over all of them: `pf`'s records
//! followed by the new ones, as one run over the whole history writes them.
//! The old bars' inputs must be the ones `pf` ran on; the close is checked.
//!
//! A portfolio's bytes hold everything its analysis reads (records, logs,
//! closes, starting cash, grouping and scoring) and the state to continue
//! from. Callback states are Python objects and are not kept. Writing them
//! to a file is the Python layer's.

use alphax_core::error::{Error, Result};
use alphax_model::TradesType;
use alphax_records::Journal;
use alphax_records::bytes::{Put, Reader, put_header};
use alphax_sim::groups::Groups;
use alphax_sim::snapshot::SimState;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::PyPortfolio;

/// A run's state after its last bar, as `pf.last_state` shows it.
#[pyclass(name = "SimState", module = "alphax", frozen, skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct PySimState(pub SimState);

#[pymethods]
impl PySimState {
    /// The first bar a run continuing it simulates.
    #[getter]
    fn next_row(&self) -> usize {
        self.0.next_row
    }

    /// Each account group's cash.
    #[getter]
    fn cash(&self) -> Vec<f64> {
        self.0.groups.iter().map(|g| g.cash).collect()
    }

    /// Each account group's cash not tied up by positions.
    #[getter]
    fn free_cash(&self) -> Vec<f64> {
        self.0.groups.iter().map(|g| g.free_cash).collect()
    }

    /// Each column's position, in column order.
    #[getter]
    fn position(&self) -> Vec<f64> {
        self.0
            .groups
            .iter()
            .flat_map(|g| g.columns.iter().map(|c| c.position))
            .collect()
    }

    /// The state as bytes, versioned.
    fn to_bytes(&self) -> Vec<u8> {
        self.0.to_bytes()
    }

    /// A state from `to_bytes`.
    #[staticmethod]
    fn from_bytes(bytes: &[u8]) -> PyResult<Self> {
        SimState::from_bytes(bytes)
            .map(Self)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "SimState(next_row={}, groups={}, cash={:?})",
            self.0.next_row,
            self.0.groups.len(),
            self.cash()
        )
    }
}

/// What a run continuing a portfolio takes from it.
pub struct Prior {
    /// Its records and logs, as written.
    pub journal: Journal,
    /// Its starting cash per column, which the whole history keeps.
    pub init_cash: Vec<f64>,
    /// The state to continue from.
    pub state: SimState,
}

/// What a run continues: a portfolio, whose records it is appended to, or
/// a bare state, whose run's records are the caller's to keep.
pub enum Resumed {
    /// `resume=pf`.
    Portfolio(Box<Prior>),
    /// `resume=state`, with arrays from `row_offset`.
    State(SimState),
}

impl Resumed {
    /// The state the run starts from.
    pub const fn state(&self) -> &SimState {
        match self {
            Self::Portfolio(prior) => &prior.state,
            Self::State(state) => state,
        }
    }
}

/// Reads `resume`: a portfolio, checked against the run with [`prior`], or
/// a state.
///
/// # Errors
///
/// A `TypeError` for anything else; a `ValueError` for a portfolio with a
/// `row_offset`, which continues on the whole history, or a `row_offset`
/// without a state.
pub fn resolve(
    resume: Option<&Bound<'_, PyAny>>,
    shape: &Shape<'_>,
    row_offset: usize,
) -> PyResult<Option<Resumed>> {
    let Some(resume) = resume else {
        if row_offset != 0 {
            return Err(PyValueError::new_err(
                "row_offset places arrays that continue a state; pass the state as resume",
            ));
        }
        return Ok(None);
    };
    if let Ok(pf) = resume.cast::<PyPortfolio>() {
        if row_offset != 0 {
            return Err(PyValueError::new_err(
                "resume=Portfolio continues on the whole history, which starts at row 0",
            ));
        }
        return prior(&pf.borrow(), shape).map(|p| Some(Resumed::Portfolio(Box::new(p))));
    }
    if let Ok(state) = resume.cast::<PySimState>() {
        return Ok(Some(Resumed::State(state.get().0.clone())));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "resume must be a Portfolio or a SimState; got {}",
        resume.get_type().name()?
    )))
}

/// The records of a run that continued a bare state, on the run's rows,
/// and the state it ended in: what a stepper keeps, bar by bar. Not a
/// portfolio: its records start mid-run, from cash it does not know.
#[pyclass(name = "Segment", module = "alphax", frozen)]
#[derive(Debug)]
pub struct PySegment {
    records: alphax_records::OrderRecords,
    state: SimState,
}

#[pymethods]
impl PySegment {
    /// The fills, as `pf.orders()` gives them, on the run's rows.
    fn orders<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        crate::orders_dict(py, &self.records)
    }

    /// The cash flows, as `pf.cash_records()` gives them.
    fn cash_records<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        crate::cash_dict(py, &self.records.cash)
    }

    /// The state after the segment's last bar.
    #[getter]
    fn last_state(&self) -> PySimState {
        PySimState(self.state.clone())
    }
}

/// A run that continued a bare state, as a [`PySegment`].
pub fn segment(py: Python<'_>, run: crate::PyRun) -> PyResult<Py<PyAny>> {
    let seg = PySegment {
        records: run.records.orders,
        state: run.last_state,
    };
    Ok(Bound::new(py, seg)?.into_any().unbind())
}

/// The shape of the run about to continue a portfolio, to check it against.
pub struct Shape<'s> {
    pub close: &'s [f64],
    pub n_rows: usize,
    pub n_cols: usize,
    pub source_cols: usize,
    pub column_map: Option<&'s [usize]>,
    pub groups: &'s Groups,
    pub cash_sharing: bool,
}

/// `pf`'s records and state, after checking the new run continues it.
///
/// # Errors
///
/// A `ValueError` for a portfolio without a state (one from before states
/// were kept, or loaded from a version without one) or a run that is not
/// its continuation: another shape, grouping or column map, fewer bars, or
/// other closes on the bars it ran.
pub fn prior(pf: &PyPortfolio, shape: &Shape<'_>) -> PyResult<Prior> {
    let bad = |m: String| PyValueError::new_err(m);
    let Some(state) = pf.last_state.clone() else {
        return Err(bad("this portfolio has no state to continue from".into()));
    };
    if pf.n_cols != shape.n_cols
        || pf.source_cols != shape.source_cols
        || pf.column_map.as_deref() != shape.column_map
    {
        return Err(bad(
            "resume needs the columns the portfolio ran on: the same close \
                        columns, column_map and parameter combinations"
                .into(),
        ));
    }
    if pf.cash_sharing != shape.cash_sharing || !same_groups(&pf.groups, shape.groups) {
        return Err(bad(
            "resume needs the grouping the portfolio ran with".into()
        ));
    }
    if shape.n_rows < state.next_row {
        return Err(bad(format!(
            "resume needs the whole history: the portfolio ran {} bars and close has {}",
            state.next_row, shape.n_rows
        )));
    }
    let old_rows = pf.n_rows;
    for c in 0..shape.source_cols {
        let old = &pf.close[c * old_rows..c * old_rows + state.next_row];
        let new = &shape.close[c * shape.n_rows..c * shape.n_rows + state.next_row];
        if let Some(r) = old
            .iter()
            .zip(new)
            .position(|(a, b)| a.to_bits() != b.to_bits())
        {
            return Err(bad(format!(
                "close differs from the portfolio's at bar {r} of column {c}: resume continues \
                 the bars it ran on, it does not rerun them"
            )));
        }
    }
    Ok(Prior {
        journal: Journal {
            orders: pf.orders.clone(),
            logs: pf.logs.clone(),
        },
        init_cash: pf.init_cash.clone(),
        state,
    })
}

fn same_groups(a: &Groups, b: &Groups) -> bool {
    a.len() == b.len() && (0..a.len()).all(|g| a.range(g) == b.range(g))
}

/// Each column's group, as [`Journal::chain`] takes it.
pub fn group_of(groups: &Groups, n_cols: usize) -> Vec<usize> {
    (0..n_cols).map(|c| groups.of_column(c)).collect()
}

const MAGIC: &[u8; 4] = b"PHPF";
const VERSION: u32 = 1;

/// `pf` as bytes; see the [module](self).
pub fn to_bytes(pf: &PyPortfolio) -> Vec<u8> {
    let mut out = Vec::new();
    put_header(&mut out, MAGIC, VERSION);
    pf.n_rows.put(&mut out);
    pf.n_cols.put(&mut out);
    pf.source_cols.put(&mut out);
    pf.column_map.is_some().put(&mut out);
    pf.column_map.clone().unwrap_or_default().put(&mut out);
    pf.init_cash.put(&mut out);
    (0..pf.groups.len())
        .map(|g| pf.groups.len_of(g))
        .collect::<Vec<_>>()
        .put(&mut out);
    pf.cash_sharing.put(&mut out);
    pf.ann_factor.put(&mut out);
    pf.risk_free.put(&mut out);
    pf.incl_open.put(&mut out);
    (pf.trades_type as i8).put(&mut out);
    pf.close.put(&mut out);
    let journal = Journal {
        orders: pf.orders.clone(),
        logs: pf.logs.clone(),
    }
    .to_bytes();
    journal.put(&mut out);
    let state = pf.last_state.as_ref().map(SimState::to_bytes);
    state.is_some().put(&mut out);
    state.unwrap_or_default().put(&mut out);
    out
}

/// A portfolio from [`to_bytes`]' output.
///
/// # Errors
///
/// Returns [`Error::Invariant`] for bytes that are not a portfolio of this
/// version or are inconsistent.
pub fn from_bytes(bytes: &[u8]) -> Result<PyPortfolio> {
    let mut r = Reader::new(bytes);
    r.header(MAGIC, VERSION, "portfolio")?;
    let n_rows: usize = r.get()?;
    let n_cols: usize = r.get()?;
    let source_cols: usize = r.get()?;
    let mapped: bool = r.get()?;
    let map: Vec<usize> = r.get()?;
    let init_cash: Vec<f64> = r.get()?;
    let lens: Vec<usize> = r.get()?;
    let cash_sharing: bool = r.get()?;
    let ann_factor: f64 = r.get()?;
    let risk_free: f64 = r.get()?;
    let incl_open: bool = r.get()?;
    let trades_type = match r.get::<i8>()? {
        0 => TradesType::EntryTrades,
        1 => TradesType::ExitTrades,
        2 => TradesType::Positions,
        b => return Err(Error::Invariant(format!("{b} is not a trades type"))),
    };
    let close: Vec<f64> = r.get()?;
    let journal = Journal::from_bytes(&r.get::<Vec<u8>>()?)?;
    let has_state: bool = r.get()?;
    let state: Vec<u8> = r.get()?;
    r.end()?;
    let last_state = has_state
        .then(|| SimState::from_bytes(&state))
        .transpose()?;
    if close.len() != n_rows * source_cols || init_cash.len() != n_cols {
        return Err(Error::Invariant(
            "portfolio arrays of the wrong shape".into(),
        ));
    }
    let groups = Groups::from_lens(&lens, n_cols)?;
    let mut pf = PyPortfolio::assemble(
        journal.orders,
        journal.logs,
        close,
        source_cols,
        mapped.then_some(map),
        init_cash,
        groups,
        cash_sharing,
        n_rows,
        n_cols,
        alphax_analysis::Scoring {
            ann_factor,
            risk_free,
            trades_type,
            incl_open,
        },
    );
    pf.last_state = last_state;
    Ok(pf)
}
