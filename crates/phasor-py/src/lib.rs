//! Python bindings for phasor.
//!
//! A thin shell, deliberately. The engine's own `Portfolio` already holds the
//! interesting design — lazy derivation, one cached rebuild chain, per-column
//! metrics — so this layer's whole job is to get NumPy's buffers in and results
//! out without copying and without lying about layout.
//!
//! ```python
//! import numpy as np, phasor
//!
//! pf = phasor.from_signals(close, entries, exits, init_cash=10_000, fees=0.001)
//! print(pf.stats())            # the 28-row table
//! sharpe = pf.sharpe_ratio()   # one value per column
//! ```
//!
//! # What is exposed, and what is not
//!
//! Both vectorized loops — `from_signals` and `from_orders` — the portfolio they
//! return, its metrics, its statistics table, its exposures and trade
//! excursions, and its order, trade and log records as NumPy arrays. Factor
//! expressions, Alpha158, and Alpha360 are exposed through the Python shell
//! over this module's compiled [`expr::PyFactorSet`], with WorldQuant 101 and
//! batched evaluation. Event studies, overfitting statistics, labels, random
//! signals and synthetic prices are in [`research`]; the indicator sweep and
//! streams in `indicators`; signal tools in `signal_fns`; rolling
//! metrics and drawdown records in `returns_fns`. Forward returns,
//! information coefficients, quantile groups and turnover are in `factor_eval`.
//!
//! # Two things worth knowing before using it
//!
//! **Arrays must be column-major.** The engine is; NumPy is not by default.
//! The Python wrapper converts, and the Rust layer refuses anything it cannot
//! read correctly rather than reading it wrongly — see `array` for why that
//! is a refusal rather than a silent transpose.
//!
//! **A column that never traded has an infinite Sharpe.** That is the
//! defined answer and it tops a naive ranking. `traded_only` masks those to
//! `NaN`, which every reduction skips.
//!
//! **`log=True` answers why a bar did not trade**, which no order record can:
//! absence of a fill is consistent with a dozen causes. It is off by default
//! because a log row is 38 fields written for every order *considered*.

#![deny(unsafe_code)]

pub mod array;
mod callbacks;
pub mod data;
pub mod expr;
mod factor_eval;
mod indicators;
pub mod params;
pub mod parse;
pub mod research;
mod resume;
mod returns_fns;
mod signal_fns;

use std::sync::OnceLock;

use numpy::{IntoPyArray, PyArray1, PyArray2};
use phasor_analysis::stats::{Select, Value};
use phasor_analysis::{Cache, Portfolio as CorePortfolio, Scoring};
use phasor_core::flex::FlexArray;
use phasor_core::matrix::Matrix;
use phasor_model::TradesType;
use phasor_records::{LogRecords, OrderRecords};
use phasor_sim::from_orders::FromOrders;
use phasor_sim::from_signals::FromSignals;
use phasor_sim::groups::{CallSeq, Groups};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

use array::{
    Borrowed, OneDim, borrow_bool, borrow_f64, borrow_opt_f64, columns_of, columns_out, flex_bool,
    flex_or_bool, flex_or_f64, to_py_err,
};
use expr::PyFactorSet;
use params::{Holding, Liquidation, Shared};

/// A completed run, and everything derivable from it.
///
/// # Why the arrays are owned here
///
/// The engine's `Portfolio` borrows its prices, which is right for Rust: a
/// sweep's close matrix is the largest thing in the system and the result of a
/// run has no reason to own a second copy. Across the FFI boundary that does
/// not survive — Python may free its array the moment this returns — so the
/// close prices and starting cash are copied once, here, and the borrow is
/// rebuilt per call from the copy.
///
/// The copy is one array, once. The rebuild chain — the value curve, returns,
/// trades and drawdowns — is derived on demand into `Self::cache`, which
/// outlives the per-call engine view, so a curve computed for one call is
/// still there for the next. `Self::clear_cache` gives the memory back.
#[pyclass(name = "Portfolio", module = "phasor")]
#[derive(Debug)]
pub struct PyPortfolio {
    orders: OrderRecords,
    /// The fills in column-major order when the run shared cash and emitted
    /// them in execution order; `None` when `orders` already is. Sorted once
    /// here rather than on every rebuild.
    sorted_orders: Option<OrderRecords>,
    /// Derivations kept across calls; the engine view borrows it.
    cache: Cache,
    /// Buy-and-hold returns, the default benchmark for `stats`, computed once.
    benchmark: OnceLock<Matrix>,
    /// Empty unless the run asked for logs. Not part of the derivation chain:
    /// nothing computed from a portfolio reads a log, which is the point —
    /// logs explain the run rather than contributing to it.
    logs: LogRecords,
    close: Vec<f64>,
    source_cols: usize,
    column_map: Option<Vec<usize>>,
    init_cash: Vec<f64>,
    groups: Groups,
    cash_sharing: bool,
    n_rows: usize,
    n_cols: usize,
    ann_factor: f64,
    risk_free: f64,
    incl_open: bool,
    trades_type: TradesType,
    /// Each group's callback state, in group order, when the run had
    /// callbacks.
    in_outputs: Option<Py<PyAny>>,
    /// The state after the last bar, to continue the run from.
    last_state: Option<phasor_sim::SimState>,
}

impl PyPortfolio {
    /// Rebuilds the engine's portfolio over this object's own buffers.
    ///
    /// Called per method rather than held, because the engine's type borrows
    /// and a `#[pyclass]` cannot hold a borrow of itself. The records and the
    /// cache are lent rather than cloned, so the rebuild itself is free and
    /// whatever one call derives, the next one finds.
    fn core(&self) -> PyResult<CorePortfolio<'_>> {
        let close = market_array(
            &self.close,
            self.n_rows,
            self.source_cols,
            self.column_map.as_deref(),
        )?;
        let cash = FlexArray::per_column(&self.init_cash, self.n_cols).map_err(to_py_err)?;
        CorePortfolio::from_borrowed(
            self.sorted_orders.as_ref().unwrap_or(&self.orders),
            close,
            cash,
            self.n_rows,
            self.n_cols,
            Scoring {
                ann_factor: self.ann_factor,
                risk_free: self.risk_free,
                trades_type: self.trades_type,
                incl_open: self.incl_open,
            },
            self.groups.clone(),
            &self.cache,
        )
        .map_err(to_py_err)
    }

    /// Assembles the result of a run into a portfolio, sorting the records
    /// once if a shared account emitted them in execution order.
    #[allow(
        clippy::too_many_arguments,
        reason = "one argument per thing a run leaves behind"
    )]
    fn assemble(
        orders: OrderRecords,
        logs: LogRecords,
        close: Vec<f64>,
        source_cols: usize,
        column_map: Option<Vec<usize>>,
        init_cash: Vec<f64>,
        groups: Groups,
        cash_sharing: bool,
        n_rows: usize,
        n_cols: usize,
        scoring: Scoring,
    ) -> Self {
        let sorted_orders = groups.is_shared().then(|| {
            let mut sorted = orders.clone();
            sorted.sort_by_column();
            sorted
        });
        Self {
            orders,
            sorted_orders,
            cache: Cache::default(),
            benchmark: OnceLock::new(),
            logs,
            close,
            source_cols,
            column_map,
            init_cash,
            groups,
            cash_sharing,
            n_rows,
            n_cols,
            ann_factor: scoring.ann_factor,
            risk_free: scoring.risk_free,
            incl_open: scoring.incl_open,
            trades_type: scoring.trades_type,
            in_outputs: None,
            last_state: None,
        }
    }
}

#[pymethods]
impl PyPortfolio {
    /// Each group's state from a run with callbacks, in group order, as
    /// `pre_group_func` made it; `None` for a run without.
    #[getter]
    fn in_outputs(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.in_outputs.as_ref().map(|v| v.clone_ref(py))
    }

    /// The state after the last bar: what `resume=pf` continues from.
    #[getter]
    fn last_state(&self) -> Option<resume::PySimState> {
        self.last_state.clone().map(resume::PySimState)
    }

    /// The portfolio as bytes: records, logs, the inputs its analysis reads
    /// and its state; `phasor.save` writes them to a file. Callback states
    /// (`in_outputs`) are not kept.
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyBytes> {
        pyo3::types::PyBytes::new(py, &resume::to_bytes(self))
    }

    /// The closes it was valued at, `(n_rows, source columns)`.
    #[getter]
    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        columns_out(py, self.close.clone(), self.n_rows, self.source_cols)
    }

    /// The run column -> close column map, or `None`.
    #[getter]
    fn column_map(&self) -> Option<Vec<usize>> {
        self.column_map.clone()
    }

    /// Starting cash per column (per group, repeated, when cash is shared).
    #[getter]
    fn init_cash<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.init_cash.clone().into_pyarray(py)
    }

    /// What `from_records` needs besides the records to analyse as this
    /// portfolio does: `ann_factor`, `risk_free`, `trades`, `incl_open`,
    /// `allow_negative_price`.
    #[getter]
    fn scoring<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("ann_factor", self.ann_factor)?;
        d.set_item("risk_free", self.risk_free)?;
        d.set_item(
            "trades",
            match self.trades_type {
                TradesType::EntryTrades => "entry",
                TradesType::ExitTrades => "exit",
                TradesType::Positions => "positions",
            },
        )?;
        d.set_item("incl_open", self.incl_open)?;
        d.set_item("allow_negative_price", self.orders.allow_negative_price)?;
        Ok(d)
    }

    /// A portfolio from fill records: `phasor.Portfolio.from_records`.
    #[staticmethod]
    #[pyo3(signature = (
        orders,
        close,
        *,
        init_cash = None,
        cash_records = None,
        group_lens = None,
        cash_sharing = false,
        column_map = None,
        allow_negative_price = false,
        ann_factor = 365.0,
        risk_free = 0.0,
        trades = "exit",
        incl_open = false,
    ))]
    #[allow(
        clippy::too_many_arguments,
        reason = "one argument per input of the analysis"
    )]
    fn from_records(
        orders: &Bound<'_, PyDict>,
        close: &Bound<'_, PyAny>,
        init_cash: Option<&Bound<'_, PyAny>>,
        cash_records: Option<&Bound<'_, PyDict>>,
        group_lens: Option<Vec<usize>>,
        cash_sharing: bool,
        column_map: Option<Vec<usize>>,
        allow_negative_price: bool,
        ann_factor: f64,
        risk_free: f64,
        trades: &str,
        incl_open: bool,
    ) -> PyResult<Self> {
        records_portfolio(
            orders,
            close,
            init_cash,
            cash_records,
            group_lens,
            cash_sharing,
            column_map,
            allow_negative_price,
            Scoring {
                ann_factor,
                risk_free,
                trades_type: parse::trades_type(trades)
                    .map_err(pyo3::exceptions::PyValueError::new_err)?,
                incl_open,
            },
        )
    }

    /// A portfolio from `to_bytes`.
    #[staticmethod]
    fn from_bytes(bytes: &[u8]) -> PyResult<Self> {
        resume::from_bytes(bytes).map_err(to_py_err)
    }

    /// Bytes retained for source prices, independent of parameter count.
    #[getter]
    fn price_nbytes(&self) -> usize {
        self.close.len() * std::mem::size_of::<f64>()
    }

    /// Number of bars.
    #[getter]
    const fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of columns.
    #[getter]
    const fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// Number of independent cash accounts.
    #[getter]
    fn n_groups(&self) -> usize {
        self.groups.len()
    }

    /// Whether cash sharing was requested.
    #[getter]
    const fn cash_sharing(&self) -> bool {
        self.cash_sharing
    }

    /// Sizes of the contiguous account groups.
    #[getter]
    fn group_lens<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        pyo3::types::PyTuple::new(py, (0..self.groups.len()).map(|g| self.groups.len_of(g)))
    }

    /// Number of fills across every column.
    #[getter]
    fn n_fills(&self) -> usize {
        self.orders.len()
    }

    /// The statistics table for one column, as a dict of title to value.
    ///
    /// Row order is fixed and stable, which is what lets two runs
    /// be diffed. A metric with no answer comes back as `float('nan')`; one
    /// whose inputs were not supplied comes back as `None`, which is a
    /// different statement.
    ///
    /// With the bars' `high` and `low` (shaped like `close`), the table adds
    /// MAE and MFE by side: `avg_mae_long`, `avg_mfe_long`,
    /// `edge_ratio_long` and the same for short trades, as returns.
    #[pyo3(signature = (col = None, *, group = None, high = None, low = None))]
    fn stats<'py>(
        &self,
        py: Python<'py>,
        col: Option<usize>,
        group: Option<usize>,
        high: Option<&Bound<'py, PyAny>>,
        low: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let col = self.account_index(col, group)?;
        let excursions = high.is_some() || low.is_some();
        let high = self.bar_field(high, "high")?;
        let low = self.bar_field(low, "low")?;
        let (table, split) = py.detach(|| -> PyResult<_> {
            let pf = self.core()?;
            let split = if excursions {
                let (high, low) = (self.bar_view(&high)?, self.bar_view(&low)?);
                pf.excursion_stats(&high, &low, col).map_err(to_py_err)?
            } else {
                Vec::new()
            };
            let pf = if self.cash_sharing {
                pf
            } else {
                let benchmark = self.benchmark.get_or_init(|| pf.buy_and_hold());
                pf.with_benchmark_ref(benchmark).map_err(to_py_err)?
            };
            Ok((pf.stats(col, &Select::all()).map_err(to_py_err)?, split))
        })?;
        let out = PyDict::new(py);
        for (title, value) in table {
            match value {
                Value::Count(n) => out.set_item(title, n)?,
                Value::Float(v) => out.set_item(title, v)?,
                // Distinct from NaN on purpose: NaN says the metric was
                // computed and had no answer, None says nobody supplied the
                // inputs for it.
                Value::Missing => out.set_item(title, py.None())?,
            }
        }
        for (key, value) in split {
            out.set_item(key, value)?;
        }
        Ok(out)
    }

    /// The common per-account metrics, computed from one rebuild.
    ///
    /// Returned together rather than one at a time because each individual
    /// accessor rebuilds the chain; asking for six of them separately does six
    /// times the work for the same answer.
    fn metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let (columns, counts) = py.detach(|| -> PyResult<_> {
            let pf = self.core()?;
            let columns = [
                ("total_return", pf.total_return()),
                ("annualized_return", pf.annualized_return()),
                ("max_drawdown", pf.max_drawdown()),
                ("sharpe_ratio", pf.sharpe_ratio()),
                ("sortino_ratio", pf.sortino_ratio()),
                ("calmar_ratio", pf.calmar_ratio()),
            ];
            // An integer array, matching `trade_count()`. A float count would
            // round-trip fine and then read as a measurement rather than a tally.
            let counts: Vec<i64> = pf
                .trade_counts()
                .map_err(to_py_err)?
                .into_iter()
                .map(|n| n as i64)
                .collect();
            Ok((columns, counts))
        })?;
        let out = PyDict::new(py);
        for (name, values) in columns {
            out.set_item(name, values.into_pyarray(py))?;
        }
        out.set_item("trade_count", counts.into_pyarray(py))?;
        Ok(out)
    }

    /// Sharpe ratio per account.
    ///
    /// A column that never traded has a flat curve and answers `inf`, which
    /// sorts as the best column in a sweep. [`Self::traded_only`] is how a
    /// ranking says what it is ranking.
    fn sharpe_ratio<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(py
            .detach(|| self.core().map(|pf| pf.sharpe_ratio()))?
            .into_pyarray(py))
    }

    /// Total return per account.
    fn total_return<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(py
            .detach(|| self.core().map(|pf| pf.total_return()))?
            .into_pyarray(py))
    }

    /// Worst drawdown per column, as a negative fraction.
    fn max_drawdown<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(py
            .detach(|| self.core().map(|pf| pf.max_drawdown()))?
            .into_pyarray(py))
    }

    /// Number of trades per account.
    fn trade_count<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let counts: Vec<i64> = py
            .detach(|| self.core()?.trade_counts().map_err(to_py_err))?
            .into_iter()
            .map(|n| n as i64)
            .collect();
        Ok(counts.into_pyarray(py))
    }

    /// A per-column metric with the untraded columns masked to `NaN`.
    ///
    /// The trap this exists for: a column whose signals never fired has zero
    /// return over zero volatility, so its Sharpe is `inf` and it tops a naive
    /// ranking. `NaN` is skipped by every reduction, so masking is how a caller
    /// says "rank the columns that actually did something".
    fn traded_only<'py>(
        &self,
        py: Python<'py>,
        values: numpy::PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let slice = values
            .as_slice()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let masked = py.detach(|| self.core()?.traded_only(slice).map_err(to_py_err))?;
        Ok(masked.into_pyarray(py))
    }

    /// The value curve of an account. Shared runs require group=.
    #[pyo3(signature = (col = None, *, group = None))]
    fn value<'py>(
        &self,
        py: Python<'py>,
        col: Option<usize>,
        group: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let index = self.account_index(col, group)?;
        Ok(py
            .detach(|| self.core().map(|pf| pf.value().column(index).to_vec()))?
            .into_pyarray(py))
    }

    /// Per-bar account returns.
    #[pyo3(signature = (col = None, *, group = None))]
    fn returns<'py>(
        &self,
        py: Python<'py>,
        col: Option<usize>,
        group: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let index = self.account_index(col, group)?;
        Ok(py
            .detach(|| self.core().map(|pf| pf.returns().column(index).to_vec()))?
            .into_pyarray(py))
    }

    /// Drawdown episodes of each account's value, as a dict of arrays:
    /// where each peaked, bottomed and recovered (`peak_idx`, `valley_idx`,
    /// `end_idx`), the values there, `status` (0 still open, 1 recovered),
    /// and derived `drawdown`, `duration`, `decline_duration`,
    /// `recovery_duration`, `recovery_return` and `recovered`.
    fn drawdowns<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = py.detach(|| self.core().map(|pf| pf.drawdowns().clone()))?;
        returns_fns::drawdowns_dict(py, &d)
    }

    /// A metric of each account over each trailing `window` bars,
    /// `(n_rows, n_accounts)`: one of `total_return`, `annualized_return`,
    /// `annualized_volatility`, `max_drawdown`, `calmar_ratio`,
    /// `omega_ratio`, `sharpe_ratio`, `downside_risk`, `sortino_ratio`,
    /// `tail_ratio`, `value_at_risk`, `cond_value_at_risk`, or against
    /// buy-and-hold `information_ratio`, `beta`, `alpha`, `capture`,
    /// `up_capture`, `down_capture`. `min_periods` (a full window by
    /// default) counts bars with a return.
    #[pyo3(signature = (metric, window, *, min_periods = None, required_return = 0.0, cutoff = 0.05, ddof = 1))]
    fn rolling<'py>(
        &self,
        py: Python<'py>,
        metric: &str,
        window: usize,
        min_periods: Option<usize>,
        required_return: f64,
        cutoff: f64,
        ddof: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let paired = returns_fns::PAIRED.contains(&metric);
        if paired && self.cash_sharing {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "a benchmark metric needs one instrument per account; this run shares cash",
            ));
        }
        let o = returns_fns::Opts {
            ann_factor: self.ann_factor,
            risk_free: self.risk_free,
            required_return,
            cutoff,
            ddof,
        };
        let (data, n_rows, n_cols) = py.detach(|| -> PyResult<_> {
            let pf = self.core()?;
            let returns = pf.returns();
            let bench = paired.then(|| pf.buy_and_hold());
            let mut out = Vec::with_capacity(returns.n_rows() * returns.n_cols());
            for c in 0..returns.n_cols() {
                out.extend(returns_fns::rolling(
                    metric,
                    returns.column(c),
                    bench.as_ref().map(|b| b.column(c)),
                    window,
                    min_periods,
                    o,
                )?);
            }
            Ok((out, returns.n_rows(), returns.n_cols()))
        })?;
        columns_out(py, data, n_rows, n_cols)
    }

    /// Cash held by an account at the end of each bar.
    #[pyo3(signature = (col = None, *, group = None))]
    fn cash<'py>(
        &self,
        py: Python<'py>,
        col: Option<usize>,
        group: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let index = self.account_index(col, group)?;
        Ok(py
            .detach(|| self.core().map(|pf| pf.cash().column(index).to_vec()))?
            .into_pyarray(py))
    }

    /// Cash an account has free at the end of each bar: its cash less what
    /// its open shorts reserve and its margin columns post, with each margin
    /// position counted at its profit.
    #[pyo3(signature = (col = None, *, group = None))]
    fn free_cash<'py>(
        &self,
        py: Python<'py>,
        col: Option<usize>,
        group: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let index = self.account_index(col, group)?;
        Ok(py
            .detach(|| -> PyResult<_> {
                let pf = self.core()?;
                Ok(pf.free_cash().map_err(to_py_err)?.column(index).to_vec())
            })?
            .into_pyarray(py))
    }

    /// Position quantity of one symbol column.
    fn assets<'py>(&self, py: Python<'py>, col: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.column_of(py, col, |pf, c| pf.assets().column(c).to_vec())
    }

    /// Each symbol column's signed position value over its account's total
    /// value, `(n_rows, n_cols)`.
    ///
    /// The weight a column actually carries, drift between rebalances
    /// included. Shorts are negative; NaN where the account is worth nothing.
    ///
    /// The exposure methods return every column at once, unlike
    /// [`Self::value`]: they are computed whole and not cached, so one column
    /// per call would redo the whole matrix per column.
    fn allocations<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = py.detach(|| self.core().map(|pf| pf.allocations()))?;
        matrix_out(py, m)
    }

    /// Long position value over total value, `(n_rows, n_groups)`.
    ///
    /// Divided by the account's value, as a weight is quoted. A hedged book
    /// can be fully long and fully short at once; netting would hide both.
    fn long_exposure<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = py.detach(|| self.core().map(|pf| pf.long_exposure()))?;
        matrix_out(py, m)
    }

    /// Absolute short position value over total value, `(n_rows, n_groups)`.
    fn short_exposure<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = py.detach(|| self.core().map(|pf| pf.short_exposure()))?;
        matrix_out(py, m)
    }

    /// Long minus short exposure, `(n_rows, n_groups)`: the directional bet.
    fn net_exposure<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = py.detach(|| self.core().map(|pf| pf.net_exposure()))?;
        matrix_out(py, m)
    }

    /// Long plus short exposure, `(n_rows, n_groups)`: how much is at work.
    ///
    /// Over total value. The engine's `gross_exposure` divides by free cash
    /// plus the position instead; the two agree only on a long-only run.
    fn gross_long_short<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = py.detach(|| self.core().map(|pf| pf.gross_long_short()))?;
        matrix_out(py, m)
    }

    /// Fraction of bars each symbol column holds a non-zero position.
    fn position_coverage<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(py
            .detach(|| self.core().map(|pf| pf.position_coverage()))?
            .into_pyarray(py))
    }

    /// Maximum adverse and favorable excursion of each trade, as a dict of
    /// arrays aligned row for row with [`Self::trades`].
    ///
    /// `mae` and `mfe` are price distances per unit (MAE never positive, MFE
    /// never negative); `mae_ret` and `mfe_ret` divide them by the entry price.
    /// The window runs from the entry bar to the exit bar inclusive, and the
    /// entry bar's whole range counts.
    ///
    /// A portfolio keeps only its closes, so pass the bars' `high` and `low`,
    /// shaped like `close` was. Without them both sides are measured on
    /// closes, which can only understate them.
    #[pyo3(signature = (high = None, low = None))]
    fn excursions<'py>(
        &self,
        py: Python<'py>,
        high: Option<&Bound<'py, PyAny>>,
        low: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let high = self.bar_field(high, "high")?;
        let low = self.bar_field(low, "low")?;
        let ex = py.detach(|| -> PyResult<_> {
            let (high, low) = (self.bar_view(&high)?, self.bar_view(&low)?);
            self.core()?.excursions(&high, &low).map_err(to_py_err)
        })?;
        let out = PyDict::new(py);
        out.set_item("mae", ex.mae.into_pyarray(py))?;
        out.set_item("mfe", ex.mfe.into_pyarray(py))?;
        out.set_item("mae_ret", ex.mae_ret.into_pyarray(py))?;
        out.set_item("mfe_ret", ex.mfe_ret.into_pyarray(py))?;
        Ok(out)
    }

    /// Edge ratio per account: mean MFE over mean MAE of its trades, both as
    /// returns.
    ///
    /// Above one, trades went further in their favor than against them on
    /// average — an entry edge measured before the exit rule has a say. NaN
    /// for an account with no trades, or whose trades never went against
    /// their entry. `high` and `low` as for [`Self::excursions`].
    #[pyo3(signature = (high = None, low = None))]
    fn edge_ratio<'py>(
        &self,
        py: Python<'py>,
        high: Option<&Bound<'py, PyAny>>,
        low: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let high = self.bar_field(high, "high")?;
        let low = self.bar_field(low, "low")?;
        Ok(py
            .detach(|| -> PyResult<_> {
                let (high, low) = (self.bar_view(&high)?, self.bar_view(&low)?);
                self.core()?.edge_ratio(&high, &low).map_err(to_py_err)
            })?
            .into_pyarray(py))
    }

    /// The fills, as a dict of NumPy arrays.
    ///
    /// Struct-of-arrays, as the record layer stores them, so a caller can build
    /// a DataFrame from it without a transpose.
    ///
    /// A run with any column above leverage 1, any margin column or any
    /// holding charge adds `debt`, `locked_cash`, `margin`, `isolated` and
    /// `liquidation`: what each fill left the column owing and tying up of
    /// its own cash, whether the column trades on margin and on isolated
    /// margin, and whether the fill was a liquidation. A margin column's
    /// `debt` is its position's cost basis and its `locked_cash` the initial
    /// margin posted. Such a run, and any with a contract multiplier, also adds
    /// `multiplier`, the fill's column's. Other runs leave them out
    /// rather than return them empty, so every array in the dict has one entry
    /// per fill.
    fn orders<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        orders_dict(py, &self.orders)
    }

    /// Cash that moved without a fill, as a dict of NumPy arrays.
    ///
    /// `amount` is what was applied, after any cap — a withdrawal or an
    /// earning's charge larger than the free cash is applied in part — so it
    /// can differ from what was passed in. `kind` is 0 for a deposit or
    /// withdrawal, 1 for an earning, 2 for a dividend, 3 for funding, 4 for
    /// financing, 5 for interest on a long's debt, 6 for a short's borrow fee
    /// and 7 for a liquidation's loss written off. Rows are in column order,
    /// then bar order.
    fn cash_records<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        cash_dict(
            py,
            &self.sorted_orders.as_ref().unwrap_or(&self.orders).cash,
        )
    }

    /// The cash flows of one kind, `(n_rows, n_cols)`: per bar and per
    /// column, zero where nothing of that kind moved.
    ///
    /// `kind` is `deposit`, `earning`, `dividend`, `funding`, `financing`,
    /// `interest`, `borrowfee`, `liquidation` or `initialposition`. Per column even in a
    /// shared account, so a
    /// pair's funding stays with the perpetual that paid or received it.
    ///
    /// # Errors
    ///
    /// A Python `ValueError` for an unknown kind.
    fn cash_flows<'py>(&self, py: Python<'py>, kind: &str) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let kind = parse::cash_flow_kind(kind).map_err(pyo3::exceptions::PyValueError::new_err)?;
        matrix_out(py, self.core()?.cash_flow_by_kind(kind))
    }

    /// Number of log rows, which is zero unless the run asked for them.
    #[getter]
    fn n_logs(&self) -> usize {
        self.logs.len()
    }

    /// The logs, as a dict of NumPy arrays — the 38-field base schema
    /// plus leverage's four: `locked_cash` and `new_locked_cash` on either
    /// side of the order, `req_leverage` and `req_leverage_mode` in the
    /// request. `req_leverage_mode` is 0 for lazy and 1 for eager. And
    /// margin's three, all in the request: `req_account_type` (0 cash, 1
    /// margin), `req_initial_margin` and `req_margin_basis` (0 notional, 1
    /// unit). And `liquidation`, true for a row the exchange's liquidation
    /// wrote rather than the strategy's order, and `multiplier`, the column's
    /// contract multiplier: prices are prices, cash and values are cash.
    ///
    /// One row per order **considered**, filled or not, which is the difference
    /// from [`Self::orders`] and the whole reason to ask for logs. Without them
    /// a bar that did not trade leaves no evidence at all, and "no order record"
    /// is consistent with a dozen causes: the size resolved to zero, the cash
    /// ran out, the granularity floored it away, the direction forbade it, the
    /// price was `NaN`.
    ///
    /// `order_id` joins a row to the fill it produced, or is `-1` when nothing
    /// filled. `id` is the row's own index.
    ///
    /// # Errors
    ///
    /// Only if building an array fails.
    fn logs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let l = &self.logs;
        let out = PyDict::new(py);
        let put_i = |name: &str, v: Vec<i64>| -> PyResult<()> {
            out.set_item(name, v.into_pyarray(py))?;
            Ok(())
        };
        put_i("id", l.ids())?;
        put_i("group", l.group.clone())?;
        put_i("col", l.col.clone())?;
        put_i("idx", l.idx.clone())?;
        put_i("req_size_type", l.req_size_type.clone())?;
        put_i("req_direction", l.req_direction.clone())?;
        put_i("req_leverage_mode", l.req_leverage_mode.clone())?;
        put_i("req_account_type", l.req_account_type.clone())?;
        put_i("req_margin_basis", l.req_margin_basis.clone())?;
        put_i("res_side", l.res_side.clone())?;
        put_i("res_status", l.res_status.clone())?;
        put_i("res_status_info", l.res_status_info.clone())?;
        put_i("order_id", l.order_id.clone())?;

        let put_f = |name: &str, v: Vec<f64>| -> PyResult<()> {
            out.set_item(name, v.into_pyarray(py))?;
            Ok(())
        };
        put_f("cash", l.cash.clone())?;
        put_f("position", l.position.clone())?;
        put_f("debt", l.debt.clone())?;
        put_f("locked_cash", l.locked_cash.clone())?;
        put_f("free_cash", l.free_cash.clone())?;
        put_f("val_price", l.val_price.clone())?;
        put_f("value", l.value.clone())?;
        put_f("req_size", l.req_size.clone())?;
        put_f("req_price", l.req_price.clone())?;
        put_f("req_fees", l.req_fees.clone())?;
        put_f("req_fixed_fees", l.req_fixed_fees.clone())?;
        put_f("req_slippage", l.req_slippage.clone())?;
        put_f("req_min_size", l.req_min_size.clone())?;
        put_f("req_max_size", l.req_max_size.clone())?;
        put_f("req_size_granularity", l.req_size_granularity.clone())?;
        put_f("req_leverage", l.req_leverage.clone())?;
        put_f("req_initial_margin", l.req_initial_margin.clone())?;
        put_f("req_reject_prob", l.req_reject_prob.clone())?;
        put_f("new_cash", l.new_cash.clone())?;
        put_f("new_position", l.new_position.clone())?;
        put_f("new_debt", l.new_debt.clone())?;
        put_f("new_locked_cash", l.new_locked_cash.clone())?;
        put_f("new_free_cash", l.new_free_cash.clone())?;
        put_f("new_val_price", l.new_val_price.clone())?;
        put_f("new_value", l.new_value.clone())?;
        put_f("res_size", l.res_size.clone())?;
        put_f("res_price", l.res_price.clone())?;
        put_f("res_fees", l.res_fees.clone())?;

        out.set_item("req_lock_cash", l.req_lock_cash.clone())?;
        out.set_item("req_allow_partial", l.req_allow_partial.clone())?;
        out.set_item("req_raise_reject", l.req_raise_reject.clone())?;
        out.set_item("req_log", l.req_log.clone())?;
        out.set_item("liquidation", l.liquidation.clone())?;
        put_f("multiplier", l.multiplier.clone())?;
        Ok(out)
    }

    /// Why each logged order ended the way it did, in words.
    ///
    /// `"filled"` for the ones that did, and the rejection's own reason for the
    /// ones that did not. The fields are enough to work it out; this is the
    /// answer without the lookup table.
    fn log_reasons(&self) -> Vec<&'static str> {
        self.logs.reasons()
    }

    /// How often each outcome occurred in one column, commonest first.
    ///
    /// The first diagnostic to reach for on a column that traded far less than
    /// expected: there is almost always one reason dominating.
    ///
    /// # Errors
    ///
    /// A Python `ValueError` if `col` is outside the run's shape.
    #[pyo3(signature = (col = 0))]
    fn reason_counts<'py>(&self, py: Python<'py>, col: usize) -> PyResult<Bound<'py, PyDict>> {
        if col >= self.n_cols {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "column {col} is outside 0..{}",
                self.n_cols
            )));
        }
        let tally = self
            .logs
            .reason_counts(col, self.n_cols)
            .map_err(to_py_err)?;
        let out = PyDict::new(py);
        for (reason, count) in tally {
            out.set_item(reason, count)?;
        }
        Ok(out)
    }

    /// The trades, in whichever view this portfolio is scored with.
    ///
    /// `multiplier` is the column's contract multiplier: `pnl` is in cash, so
    /// multiplied by it, and `ret`, a fraction of the entry value, is not.
    fn trades<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let pf = self.core()?;
        // Derived (and cached) without the GIL; the arrays are built with it.
        let t = py.detach(|| pf.trades().map_err(to_py_err))?;
        let out = PyDict::new(py);
        out.set_item("col", t.col.clone().into_pyarray(py))?;
        out.set_item("size", t.size.clone().into_pyarray(py))?;
        out.set_item("entry_idx", t.entry_idx.clone().into_pyarray(py))?;
        out.set_item("entry_price", t.entry_price.clone().into_pyarray(py))?;
        out.set_item("entry_fees", t.entry_fees.clone().into_pyarray(py))?;
        out.set_item("exit_idx", t.exit_idx.clone().into_pyarray(py))?;
        out.set_item("exit_price", t.exit_price.clone().into_pyarray(py))?;
        out.set_item("exit_fees", t.exit_fees.clone().into_pyarray(py))?;
        out.set_item("pnl", t.pnl.clone().into_pyarray(py))?;
        out.set_item("ret", t.ret.clone().into_pyarray(py))?;
        // Derived rather than stored, and not `exit_idx - entry_idx`: a closed
        // trade is measured exclusively and an open one inclusively.
        out.set_item("duration", t.duration().into_pyarray(py))?;
        let direction: Vec<i64> = t.direction.iter().map(|d| i64::from(*d)).collect();
        out.set_item("direction", direction.into_pyarray(py))?;
        let status: Vec<i64> = t.status.iter().map(|s| i64::from(*s)).collect();
        out.set_item("status", status.into_pyarray(py))?;
        out.set_item("parent_id", t.parent_id.clone().into_pyarray(py))?;
        out.set_item("multiplier", t.multiplier.clone().into_pyarray(py))?;
        Ok(out)
    }

    /// Drops every derived series this portfolio has cached.
    ///
    /// The value curve, returns, trades and drawdowns are kept between calls
    /// so that reading six metrics costs one derivation, not six. Each is a
    /// `bars × accounts` matrix; on a large sweep that is memory worth giving
    /// back once the numbers are in hand. The next call derives again.
    fn clear_cache(&mut self) {
        self.cache.clear();
        self.benchmark = OnceLock::new();
    }

    fn __repr__(&self) -> String {
        format!(
            "Portfolio({} bars x {} columns, {} fills)",
            self.n_rows,
            self.n_cols,
            self.orders.len()
        )
    }
}

impl PyPortfolio {
    /// One column of a derived series, checking the column exists first.
    fn account_index(&self, col: Option<usize>, group: Option<usize>) -> PyResult<usize> {
        if col.is_some() && group.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "specify col or group, not both",
            ));
        }
        if self.cash_sharing && col.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "shared-account series require group=, not col",
            ));
        }
        let index = group.or(col).unwrap_or(0);
        if index >= self.groups.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "account {index} is outside 0..{}",
                self.groups.len()
            )));
        }
        Ok(index)
    }

    fn column_of<'py>(
        &self,
        py: Python<'py>,
        col: usize,
        pick: impl Fn(&CorePortfolio<'_>, usize) -> Vec<f64> + Send + Sync,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if col >= self.n_cols {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "column {col} is outside 0..{}",
                self.n_cols
            )));
        }
        Ok(py
            .detach(|| self.core().map(|pf| pick(&pf, col)))?
            .into_pyarray(py))
    }

    /// A bar field passed to an analysis method, copied column-major, or
    /// `None` when not given.
    ///
    /// Shaped like the run's source `close`: `(n_rows, source columns)`, or
    /// 1-D for a single source column. Copied rather than borrowed, and in
    /// either memory order: the copy reads by index, so a C-ordered array
    /// comes out right rather than transposed. The no-copy rule in [`array`]
    /// is for simulation inputs, read on every bar of every run; this is read
    /// once, after the run.
    fn bar_field(
        &self,
        value: Option<&Bound<'_, PyAny>>,
        name: &str,
    ) -> PyResult<Option<Vec<f64>>> {
        let Some(value) = value else {
            return Ok(None);
        };
        let expected = (self.n_rows, self.source_cols);
        if let Ok(a) = value.extract::<numpy::PyReadonlyArray2<'_, f64>>() {
            let view = a.as_array();
            if view.dim() != expected {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "{name} is {:?}; close was {expected:?}",
                    view.dim()
                )));
            }
            return Ok(Some(view.t().iter().copied().collect()));
        }
        if let Ok(a) = value.extract::<numpy::PyReadonlyArray1<'_, f64>>() {
            let view = a.as_array();
            if expected.1 != 1 || view.len() != expected.0 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "{name} has {} values; a 1-D {name} needs a single-column run of {} bars",
                    view.len(),
                    expected.0
                )));
            }
            return Ok(Some(view.to_vec()));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "{name} must be a float64 array shaped like close"
        )))
    }

    /// The engine's view of a [`Self::bar_field`]: through `column_map` like
    /// `close`, or NaN everywhere — read as "use the close" — when absent.
    fn bar_view<'a>(&'a self, field: &'a Option<Vec<f64>>) -> PyResult<FlexArray<'a, f64>> {
        match field {
            Some(data) => market_array(
                data,
                self.n_rows,
                self.source_cols,
                self.column_map.as_deref(),
            ),
            None => Ok(FlexArray::Scalar(f64::NAN)),
        }
    }
}

/// A matrix as an `(n_rows, n_cols)` NumPy array over the same column-major
/// buffer: an F-ordered view, not a copy.
fn matrix_out(py: Python<'_>, m: Matrix) -> PyResult<Bound<'_, PyArray2<f64>>> {
    let (rows, cols) = (m.n_rows(), m.n_cols());
    columns_out(py, m.into_vec(), rows, cols)
}

/// `account_type` as one type per column: `None` keeps the engine's default.
///
/// A string applies to every column, and a list gives each column its own,
/// which is how a spot leg and a perpetual leg share one account. A type is a
/// column's for the whole run, so there is no matrix form.
fn account_types(
    value: Option<&Bound<'_, PyAny>>,
    n_cols: usize,
) -> PyResult<Option<Vec<phasor_model::AccountType>>> {
    let bad = pyo3::exceptions::PyValueError::new_err;
    let Some(value) = value else {
        return Ok(None);
    };
    if let Ok(name) = value.extract::<String>() {
        let ty = parse::account_type(&name).map_err(bad)?;
        return Ok(Some(vec![ty; n_cols]));
    }
    let names: Vec<String> = value.extract().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "account_type must be 'cash', 'margin', or one of them per column",
        )
    })?;
    if names.len() != n_cols {
        return Err(bad(format!(
            "account_type has {} entries for {n_cols} columns",
            names.len()
        )));
    }
    names
        .iter()
        .map(|name| parse::account_type(name).map_err(bad))
        .collect::<PyResult<_>>()
        .map(Some)
}

/// An enum parameter given as one name, or one name per column: `None`
/// keeps the engine's default.
fn per_column<T: Copy>(
    value: Option<&Bound<'_, PyAny>>,
    n_cols: usize,
    name: &str,
    expected: &str,
    parse: fn(&str) -> Result<T, String>,
) -> PyResult<Option<Vec<T>>> {
    let bad = pyo3::exceptions::PyValueError::new_err;
    let Some(value) = value else {
        return Ok(None);
    };
    if let Ok(one) = value.extract::<String>() {
        return Ok(Some(vec![parse(&one).map_err(bad)?; n_cols]));
    }
    let names: Vec<String> = value.extract().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(format!("{name} must be {expected}"))
    })?;
    if names.len() != n_cols {
        return Err(bad(format!(
            "{name} has {} entries for {n_cols} columns",
            names.len()
        )));
    }
    names
        .iter()
        .map(|one| parse(one).map_err(bad))
        .collect::<PyResult<_>>()
        .map(Some)
}

/// `margin_mode` as one mode per column: `None` keeps the engine's default.
///
/// A string applies to every column and a list gives each its own, as for
/// `account_type`; a cash column ignores it.
fn margin_modes(
    value: Option<&Bound<'_, PyAny>>,
    n_cols: usize,
) -> PyResult<Option<Vec<phasor_model::MarginMode>>> {
    let bad = pyo3::exceptions::PyValueError::new_err;
    let Some(value) = value else {
        return Ok(None);
    };
    if let Ok(name) = value.extract::<String>() {
        let mode = parse::margin_mode(&name).map_err(bad)?;
        return Ok(Some(vec![mode; n_cols]));
    }
    let names: Vec<String> = value.extract().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "margin_mode must be 'cross', 'isolated', or one of them per column",
        )
    })?;
    if names.len() != n_cols {
        return Err(bad(format!(
            "margin_mode has {} entries for {n_cols} columns",
            names.len()
        )));
    }
    names
        .iter()
        .map(|name| parse::margin_mode(name).map_err(bad))
        .collect::<PyResult<_>>()
        .map(Some)
}

/// Resolves group capital once, expanding only for the simulator's per-column API.
fn account_inputs(
    shared: bool,
    lens: Option<Vec<usize>>,
    cash: &Option<Borrowed<'_>>,
    n_rows: usize,
    n_cols: usize,
    default: FlexArray<'_, f64>,
) -> PyResult<(Groups, Vec<f64>)> {
    let bad = pyo3::exceptions::PyValueError::new_err;
    if !shared && lens.is_some() {
        return Err(bad("group_lens requires cash_sharing=True"));
    }
    let groups = if shared {
        Groups::from_lens(&lens.unwrap_or_else(|| vec![n_cols]), n_cols).map_err(to_py_err)?
    } else {
        Groups::ungrouped(n_cols)
    };
    if shared && matches!(cash, Some(Borrowed::Slice(_))) {
        return Err(bad(
            "shared init_cash must be a scalar or one value per group",
        ));
    }
    let balances = flex_or_f64(
        cash,
        if shared { 1 } else { n_rows },
        groups.len(),
        OneDim::PerColumn,
        default,
    )?;
    let expanded = (0..n_cols)
        .map(|col| balances.get(0, groups.of_column(col)))
        .collect();
    Ok((groups, expanded))
}

/// `cash_deposits` as the engine reads it: once per account.
///
/// The engine books a group's deposit from its first column, as it reads
/// `init_cash`. A shared run therefore takes a matrix with one column per
/// group — the same shape rule as a shared `init_cash` — and every column of
/// a group is pointed at its group's column without copying. A scalar or a
/// 1-D array (one value per bar) means the same for every account.
fn deposits_view<'a>(
    value: &'a Option<Borrowed<'a>>,
    n_rows: usize,
    n_cols: usize,
    groups: &Groups,
    group_of: &'a [usize],
) -> PyResult<FlexArray<'a, f64>> {
    match value {
        Some(Borrowed::Slice(a)) if groups.is_shared() => {
            let (flat, rows, cols) = columns_of(a)?;
            if rows != n_rows || cols != groups.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "shared cash_deposits must be (n_rows, n_groups) = ({n_rows}, {}); got \
                     ({rows}, {cols})",
                    groups.len()
                )));
            }
            FlexArray::mapped(flat, n_rows, cols, group_of).map_err(to_py_err)
        }
        _ => flex_or_f64(
            value,
            n_rows,
            n_cols,
            OneDim::PerRow,
            FromOrders::default().cash_deposits,
        ),
    }
}

/// Runs a signal-driven backtest.
///
/// `close`, `entries` and `exits` are `(n_rows, n_cols)` arrays and set the
/// shape. Everything else takes a scalar, a 1-D array or a full matrix, the
/// same flexibility the engine has internally.
///
/// # Errors
///
/// A Python `ValueError` for a shape or layout that cannot be read, or for an
/// invariant the execution kernel refuses — with the engine's own message,
/// which names the row and column.
#[pyfunction]
#[pyo3(signature = (
    close,
    entries,
    exits,
    *,
    open = None,
    high = None,
    low = None,
    init_cash = None,
    size = None,
    size_type = "amount",
    price = None,
    direction = "longonly",
    fees = None,
    fixed_fees = None,
    slippage = None,
    min_size = None,
    max_size = None,
    size_granularity = None,
    leverage = None,
    leverage_mode = "lazy",
    account_type = None,
    initial_margin = None,
    margin_basis = None,
    multiplier = None,
    init_position = None,
    init_price = None,
    allow_negative_price = false,
    price_type = None,
    from_ago = None,
    bid_price = None,
    ask_price = None,
    price_area_vio_mode = None,
    force_exit = None,
    cash_limit = None,
    reject_prob = None,
    seed = 0,
    lock_cash = None,
    allow_partial = None,
    raise_reject = None,
    log = None,
    accumulate = "disabled",
    sl_stop = None,
    sl_trail = None,
    tp_stop = None,
    stop_entry_price = None,
    stop_exit_price = None,
    upon_stop_exit = None,
    upon_stop_update = None,
    order_type = None,
    limit_delta = None,
    delta_format = None,
    limit_reverse = None,
    limit_order_price = None,
    limit_delay = None,
    limit_tif = None,
    limit_expiry = None,
    upon_adj_limit_conflict = None,
    upon_opp_limit_conflict = None,
    upon_adj_stop_conflict = None,
    upon_opp_stop_conflict = None,
    order_mode = false,
    tsl_stop = None,
    tsl_th = None,
    td_stop = None,
    dt_stop = None,
    stop_ladder = None,
    stop_order_type = None,
    stop_limit_delta = None,
    val_price = None,
    cash_deposits = None,
    cash_dividends = None,
    cash_earnings = None,
    funding_rate = None,
    long_financing_rate = None,
    short_financing_rate = None,
    debt_rate = None,
    borrow_rate = None,
    mark_price = None,
    margin_mode = None,
    maintenance_margin = None,
    liquidation_fee = None,
    cash_maintenance_margin = None,
    margin_tier_notional = None,
    margin_tier_rate = None,
    liquidation_step = None,
    auto_deleverage = None,
    ffill_val_price = true,
    update_value = false,
    ann_factor = 365.0,
    risk_free = 0.0,
    trades = "exit",
    incl_open = false,
    threads = 1,
    cash_sharing = false,
    group_lens = None,
    call_seq = None,
    column_map = None,
    callbacks = None,
    builtin = None,
    flexible = false,
    row_wise = false,
    resume = None,
    row_offset = 0,
))]
#[allow(
    clippy::too_many_arguments,
    reason = "one argument per simulation parameter, which is the API"
)]
fn from_signals(
    py: Python<'_>,
    close: &Bound<'_, PyAny>,
    entries: &Bound<'_, PyAny>,
    exits: &Bound<'_, PyAny>,
    open: Option<&Bound<'_, PyAny>>,
    high: Option<&Bound<'_, PyAny>>,
    low: Option<&Bound<'_, PyAny>>,
    init_cash: Option<&Bound<'_, PyAny>>,
    size: Option<&Bound<'_, PyAny>>,
    size_type: &str,
    price: Option<&Bound<'_, PyAny>>,
    direction: &str,
    fees: Option<&Bound<'_, PyAny>>,
    fixed_fees: Option<&Bound<'_, PyAny>>,
    slippage: Option<&Bound<'_, PyAny>>,
    min_size: Option<&Bound<'_, PyAny>>,
    max_size: Option<&Bound<'_, PyAny>>,
    size_granularity: Option<&Bound<'_, PyAny>>,
    leverage: Option<&Bound<'_, PyAny>>,
    leverage_mode: &str,
    account_type: Option<&Bound<'_, PyAny>>,
    initial_margin: Option<&Bound<'_, PyAny>>,
    margin_basis: Option<&Bound<'_, PyAny>>,
    multiplier: Option<&Bound<'_, PyAny>>,
    init_position: Option<&Bound<'_, PyAny>>,
    init_price: Option<&Bound<'_, PyAny>>,
    allow_negative_price: bool,
    price_type: Option<&str>,
    from_ago: Option<&Bound<'_, PyAny>>,
    bid_price: Option<&Bound<'_, PyAny>>,
    ask_price: Option<&Bound<'_, PyAny>>,
    price_area_vio_mode: Option<&str>,
    force_exit: Option<&Bound<'_, PyAny>>,
    cash_limit: Option<&Bound<'_, PyAny>>,
    reject_prob: Option<&Bound<'_, PyAny>>,
    seed: u64,
    lock_cash: Option<&Bound<'_, PyAny>>,
    allow_partial: Option<&Bound<'_, PyAny>>,
    raise_reject: Option<&Bound<'_, PyAny>>,
    log: Option<&Bound<'_, PyAny>>,
    accumulate: &str,
    sl_stop: Option<&Bound<'_, PyAny>>,
    sl_trail: Option<&Bound<'_, PyAny>>,
    tp_stop: Option<&Bound<'_, PyAny>>,
    stop_entry_price: Option<&str>,
    stop_exit_price: Option<&str>,
    upon_stop_exit: Option<&str>,
    upon_stop_update: Option<&str>,
    order_type: Option<&Bound<'_, PyAny>>,
    limit_delta: Option<&Bound<'_, PyAny>>,
    delta_format: Option<&str>,
    limit_reverse: Option<&Bound<'_, PyAny>>,
    limit_order_price: Option<&str>,
    limit_delay: Option<&Bound<'_, PyAny>>,
    limit_tif: Option<&Bound<'_, PyAny>>,
    limit_expiry: Option<&Bound<'_, PyAny>>,
    upon_adj_limit_conflict: Option<&str>,
    upon_opp_limit_conflict: Option<&str>,
    upon_adj_stop_conflict: Option<&str>,
    upon_opp_stop_conflict: Option<&str>,
    order_mode: bool,
    tsl_stop: Option<&Bound<'_, PyAny>>,
    tsl_th: Option<&Bound<'_, PyAny>>,
    td_stop: Option<&Bound<'_, PyAny>>,
    dt_stop: Option<&Bound<'_, PyAny>>,
    stop_ladder: Option<&str>,
    stop_order_type: Option<&Bound<'_, PyAny>>,
    stop_limit_delta: Option<&Bound<'_, PyAny>>,
    val_price: Option<&Bound<'_, PyAny>>,
    cash_deposits: Option<&Bound<'_, PyAny>>,
    cash_dividends: Option<&Bound<'_, PyAny>>,
    cash_earnings: Option<&Bound<'_, PyAny>>,
    funding_rate: Option<&Bound<'_, PyAny>>,
    long_financing_rate: Option<&Bound<'_, PyAny>>,
    short_financing_rate: Option<&Bound<'_, PyAny>>,
    debt_rate: Option<&Bound<'_, PyAny>>,
    borrow_rate: Option<&Bound<'_, PyAny>>,
    mark_price: Option<&Bound<'_, PyAny>>,
    margin_mode: Option<&Bound<'_, PyAny>>,
    maintenance_margin: Option<&Bound<'_, PyAny>>,
    liquidation_fee: Option<&Bound<'_, PyAny>>,
    cash_maintenance_margin: Option<&Bound<'_, PyAny>>,
    margin_tier_notional: Option<&Bound<'_, PyAny>>,
    margin_tier_rate: Option<&Bound<'_, PyAny>>,
    liquidation_step: Option<&Bound<'_, PyAny>>,
    auto_deleverage: Option<&Bound<'_, PyAny>>,
    ffill_val_price: bool,
    update_value: bool,
    ann_factor: f64,
    risk_free: f64,
    trades: &str,
    incl_open: bool,
    threads: usize,
    cash_sharing: bool,
    group_lens: Option<Vec<usize>>,
    call_seq: Option<&Bound<'_, PyAny>>,
    column_map: Option<Vec<usize>>,
    callbacks: Option<&Bound<'_, PyDict>>,
    builtin: Option<&Bound<'_, PyDict>>,
    flexible: bool,
    row_wise: bool,
    resume: Option<&Bound<'_, PyAny>>,
    row_offset: usize,
) -> PyResult<Py<PyAny>> {
    // `close` first, so every other parameter can be checked against a known
    // shape rather than one inferred from whichever array arrived first.
    let close_b = borrow_f64(close, "close")?;
    let Borrowed::Slice(close_arr) = &close_b else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "close must be a 2-D float64 array; it is what sets the run's shape",
        ));
    };
    let (close_flat, n_rows, source_cols) = columns_of(close_arr)?;
    let close_view = market_array(close_flat, n_rows, source_cols, column_map.as_deref())?;
    let n_cols = column_map.as_ref().map_or(source_cols, Vec::len);

    // A stop that can never fire is worse than a rejected argument: the run
    // succeeds and reports a strategy that never stopped out. The stop level is
    // tested against the bar's open, low and high, all of which default to NaN,
    // so every comparison is false and the stop is silently inert.
    if (sl_stop.is_some() || tp_stop.is_some()) && (high.is_none() || low.is_none()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "sl_stop or tp_stop was given without high and low. A stop level is tested \
             against the bar's range, and with no range no stop can ever fire — the run \
             would report a stopless strategy. Pass high and low (open too, for gaps)",
        ));
    }

    let entries_b = borrow_bool(entries, "entries")?;
    let exits_b = borrow_bool(exits, "exits")?;
    let open_b = borrow_opt_f64(open, "open")?;
    let high_b = borrow_opt_f64(high, "high")?;
    let low_b = borrow_opt_f64(low, "low")?;
    let init_cash_b = borrow_opt_f64(init_cash, "init_cash")?;
    let val_price_b = borrow_opt_f64(val_price, "val_price")?;
    let deposits_b = borrow_opt_f64(cash_deposits, "cash_deposits")?;
    let earnings_b = borrow_opt_f64(cash_earnings, "cash_earnings")?;
    let dividends_b = borrow_opt_f64(cash_dividends, "cash_dividends")?;
    let h = Holding::borrow(
        funding_rate,
        long_financing_rate,
        short_financing_rate,
        debt_rate,
        borrow_rate,
        mark_price,
    )?;
    let lq = Liquidation::borrow(
        maintenance_margin,
        liquidation_fee,
        cash_maintenance_margin,
        margin_tier_notional,
        margin_tier_rate,
        liquidation_step,
    )?;
    let adl_b = array::borrow_opt_bool(auto_deleverage, "auto_deleverage")?;
    let multiplier_b = borrow_opt_f64(multiplier, "multiplier")?;
    let init_position_b = borrow_opt_f64(init_position, "init_position")?;
    let init_price_b = borrow_opt_f64(init_price, "init_price")?;
    let bid_b = borrow_opt_f64(bid_price, "bid_price")?;
    let force_exit_b = array::borrow_opt_bool(force_exit, "force_exit")?;
    let from_ago_b = from_ago
        .map(|v| array::borrow_count(v, "from_ago"))
        .transpose()?;
    let ask_b = borrow_opt_f64(ask_price, "ask_price")?;
    let sl_stop_b = borrow_opt_f64(sl_stop, "sl_stop")?;
    let tp_stop_b = borrow_opt_f64(tp_stop, "tp_stop")?;
    let sl_trail_b = array::borrow_opt_bool(sl_trail, "sl_trail")?;
    let limit_delta_b = borrow_opt_f64(limit_delta, "limit_delta")?;
    let tsl_stop_b = borrow_opt_f64(tsl_stop, "tsl_stop")?;
    let tsl_th_b = borrow_opt_f64(tsl_th, "tsl_th")?;
    let td_stop_b = borrow_opt_f64(td_stop, "td_stop")?;
    let dt_stop_b = borrow_opt_f64(dt_stop, "dt_stop")?;
    let stop_limit_delta_b = borrow_opt_f64(stop_limit_delta, "stop_limit_delta")?;
    let limit_tif_b = borrow_opt_f64(limit_tif, "limit_tif")?;
    let limit_expiry_b = borrow_opt_f64(limit_expiry, "limit_expiry")?;
    let limit_reverse_b = array::borrow_opt_bool(limit_reverse, "limit_reverse")?;
    let limit_delay_b = array::borrow_opt_bool(limit_delay, "limit_delay")?;
    let p = Shared::borrow(
        size,
        price,
        fees,
        fixed_fees,
        slippage,
        min_size,
        max_size,
        size_granularity,
        leverage,
        initial_margin,
        cash_limit,
        reject_prob,
        lock_cash,
        allow_partial,
        raise_reject,
        log,
    )?;

    let bad = pyo3::exceptions::PyValueError::new_err;
    let dir = parse::direction(direction).map_err(bad)?;
    let leverage_mode = parse::leverage_mode(leverage_mode).map_err(bad)?;
    let margin_bases = per_column(
        margin_basis,
        n_cols,
        "margin_basis",
        "'notional', 'unit', or one of them per column",
        parse::margin_basis,
    )?;
    let account_types = account_types(account_type, n_cols)?;
    let margin_modes = margin_modes(margin_mode, n_cols)?;
    let st = parse::size_type(size_type).map_err(bad)?;
    let acc = parse::accumulate(accumulate).map_err(bad)?;
    let trades_type = parse::trades_type(trades).map_err(bad)?;

    // Every default comes from here rather than from this function's signature,
    // so the two cannot drift. See `array::borrow_opt_f64`.
    let d = FromSignals::default();
    let per_col = OneDim::PerColumn;
    // The stop modes take their defaults the same way: `None` from Python is
    // the engine's own choice, not a second copy of it spelled here.
    let stop_entry_price = stop_entry_price
        .map(parse::stop_entry_price)
        .transpose()
        .map_err(bad)?
        .map_or(d.stop_entry_price, FlexArray::Scalar);
    let stop_exit_price = stop_exit_price
        .map(parse::stop_exit_price)
        .transpose()
        .map_err(bad)?
        .map_or(d.stop_exit_price, FlexArray::Scalar);
    let upon_stop_exit = upon_stop_exit
        .map(parse::upon_stop_exit)
        .transpose()
        .map_err(bad)?
        .map_or(d.upon_stop_exit, FlexArray::Scalar);
    let upon_stop_update = upon_stop_update
        .map(parse::upon_stop_update)
        .transpose()
        .map_err(bad)?
        .map_or(d.upon_stop_update, FlexArray::Scalar);
    let order_types = per_column(
        order_type,
        n_cols,
        "order_type",
        "'market', 'limit', or one of them per column",
        parse::order_type,
    )?;
    let conflict = |name: Option<&str>, default| {
        name.map(parse::pending_conflict)
            .transpose()
            .map_err(bad)
            .map(|mode| mode.map_or(default, FlexArray::Scalar))
    };
    let upon_adj_limit_conflict = conflict(upon_adj_limit_conflict, d.upon_adj_limit_conflict)?;
    let upon_opp_limit_conflict = conflict(upon_opp_limit_conflict, d.upon_opp_limit_conflict)?;
    let upon_adj_stop_conflict = conflict(upon_adj_stop_conflict, d.upon_adj_stop_conflict)?;
    let upon_opp_stop_conflict = conflict(upon_opp_stop_conflict, d.upon_opp_stop_conflict)?;
    let stop_ladder = stop_ladder
        .map(parse::stop_ladder)
        .transpose()
        .map_err(bad)?
        .unwrap_or(d.stop_ladder);
    let stop_order_types = per_column(
        stop_order_type,
        n_cols,
        "stop_order_type",
        "'market', 'limit', or one of them per column",
        parse::order_type,
    )?;
    let delta_format = delta_format
        .map(parse::delta_format)
        .transpose()
        .map_err(bad)?
        .map_or(d.delta_format, FlexArray::Scalar);
    let limit_order_price = limit_order_price
        .map(parse::limit_order_price)
        .transpose()
        .map_err(bad)?
        .map_or(d.limit_order_price, FlexArray::Scalar);
    let (groups, account_cash) = account_inputs(
        cash_sharing,
        group_lens,
        &init_cash_b,
        n_rows,
        n_cols,
        d.init_cash,
    )?;
    let call_seq_b = call_seq_arg(call_seq)?;
    let call_seq = resolve_call_seq(&call_seq_b, n_rows, n_cols)?;
    let group_of: Vec<usize> = (0..n_cols).map(|c| groups.of_column(c)).collect();
    let deposits = deposits_view(&deposits_b, n_rows, n_cols, &groups, &group_of)?;
    let earnings = flex_or_f64(&earnings_b, n_rows, n_cols, OneDim::PerRow, d.cash_earnings)?;
    // Per share, so it is market data: source-shaped, through `column_map`.
    let dividends = market_optional(
        "cash_dividends",
        &dividends_b,
        n_rows,
        source_cols,
        column_map.as_deref(),
        d.cash_dividends,
    )?;
    let map = column_map.as_deref();
    let market =
        |name, value, default| market_optional(name, value, n_rows, source_cols, map, default);
    let funding = market("funding_rate", &h.funding_rate, d.funding_rate)?;
    let long_financing = market(
        "long_financing_rate",
        &h.long_financing_rate,
        d.long_financing_rate,
    )?;
    let short_financing = market(
        "short_financing_rate",
        &h.short_financing_rate,
        d.short_financing_rate,
    )?;
    let borrow = market("borrow_rate", &h.borrow_rate, d.borrow_rate)?;
    let mark = market("mark_price", &h.mark_price, d.mark_price)?;
    let bid = market("bid_price", &bid_b, d.bid_price)?;
    let ask = market("ask_price", &ask_b, d.ask_price)?;
    let price_type = price_type
        .map(parse::price_type)
        .transpose()
        .map_err(bad)?
        .map_or(d.price_type, FlexArray::Scalar);
    let price_area_vio_mode = price_area_vio_mode
        .map(parse::price_area_vio_mode)
        .transpose()
        .map_err(bad)?
        .map_or(d.price_area_vio_mode, FlexArray::Scalar);
    // The account's cost of cash, so a parameter: a 1-D array is per bar.
    let debt = flex_or_f64(&h.debt_rate, n_rows, n_cols, OneDim::PerRow, d.debt_rate)?;

    // A run that continues a portfolio takes its records and state; see
    // `resume`.
    let prior = resume::resolve(
        resume,
        &resume::Shape {
            close: close_flat,
            n_rows,
            n_cols,
            source_cols,
            column_map: column_map.as_deref(),
            groups: &groups,
            cash_sharing,
        },
        row_offset,
    )?;

    let args = FromSignals {
        groups: Some(groups.clone()),
        call_seq,
        val_price: flex_or_f64(&val_price_b, n_rows, n_cols, per_col, d.val_price)?,
        ffill_val_price,
        update_value,
        n_rows,
        n_cols,
        close: close_view,
        open: market_optional(
            "open",
            &open_b,
            n_rows,
            source_cols,
            column_map.as_deref(),
            d.open,
        )?,
        high: market_optional(
            "high",
            &high_b,
            n_rows,
            source_cols,
            column_map.as_deref(),
            d.high,
        )?,
        low: market_optional(
            "low",
            &low_b,
            n_rows,
            source_cols,
            column_map.as_deref(),
            d.low,
        )?,
        entries: flex_bool(&entries_b, n_rows, n_cols)?,
        exits: flex_bool(&exits_b, n_rows, n_cols)?,
        init_cash: FlexArray::per_column(&account_cash, n_cols).map_err(to_py_err)?,
        direction: FlexArray::Scalar(dir),
        size_type: FlexArray::Scalar(st),
        accumulate: FlexArray::Scalar(acc),
        size: flex_or_f64(&p.size, n_rows, n_cols, per_col, d.size)?,
        price: flex_or_f64(&p.price, n_rows, n_cols, per_col, d.price)?,
        fees: flex_or_f64(&p.fees, n_rows, n_cols, per_col, d.fees)?,
        fixed_fees: flex_or_f64(&p.fixed_fees, n_rows, n_cols, per_col, d.fixed_fees)?,
        slippage: flex_or_f64(&p.slippage, n_rows, n_cols, per_col, d.slippage)?,
        min_size: flex_or_f64(&p.min_size, n_rows, n_cols, per_col, d.min_size)?,
        max_size: flex_or_f64(&p.max_size, n_rows, n_cols, per_col, d.max_size)?,
        size_granularity: flex_or_f64(
            &p.size_granularity,
            n_rows,
            n_cols,
            per_col,
            d.size_granularity,
        )?,
        leverage: flex_or_f64(&p.leverage, n_rows, n_cols, per_col, d.leverage)?,
        leverage_mode: FlexArray::Scalar(leverage_mode),
        account_type: account_types.as_ref().map_or(Ok(d.account_type), |types| {
            FlexArray::per_column(types, n_cols).map_err(to_py_err)
        })?,
        initial_margin: flex_or_f64(&p.initial_margin, n_rows, n_cols, per_col, d.initial_margin)?,
        margin_basis: margin_bases.as_ref().map_or(Ok(d.margin_basis), |bases| {
            FlexArray::per_column(bases, n_cols).map_err(to_py_err)
        })?,
        // One per column: the wrapper maps a source column's through
        // `column_map`, and the engine refuses one that changes down a column.
        multiplier: flex_or_f64(
            &multiplier_b,
            n_rows,
            n_cols,
            OneDim::PerColumn,
            d.multiplier,
        )?,
        init_position: flex_or_f64(
            &init_position_b,
            n_rows,
            n_cols,
            OneDim::PerColumn,
            d.init_position,
        )?,
        init_price: flex_or_f64(
            &init_price_b,
            n_rows,
            n_cols,
            OneDim::PerColumn,
            d.init_price,
        )?,
        allow_negative_price,
        price_type,
        from_ago: match &from_ago_b {
            Some(v) => array::flex_count(v, n_rows, n_cols)?,
            None => d.from_ago,
        },
        bid_price: bid,
        ask_price: ask,
        price_area_vio_mode,
        force_exit: flex_or_bool(&force_exit_b, n_rows, n_cols, d.force_exit)?,
        lock_cash: flex_or_bool(&p.lock_cash, n_rows, n_cols, d.lock_cash)?,
        allow_partial: flex_or_bool(&p.allow_partial, n_rows, n_cols, d.allow_partial)?,
        raise_reject: flex_or_bool(&p.raise_reject, n_rows, n_cols, d.raise_reject)?,
        log: flex_or_bool(&p.log, n_rows, n_cols, d.log)?,
        // With a ladder, the rows of the stops are its steps.
        sl_stop: if stop_ladder == phasor_model::StopLadderMode::Disabled {
            flex_or_f64(&sl_stop_b, n_rows, n_cols, per_col, d.sl_stop)?
        } else {
            array::ladder_steps(&sl_stop_b, n_cols, "sl_stop", d.sl_stop)?
        },
        sl_trail: flex_or_bool(&sl_trail_b, n_rows, n_cols, d.sl_trail)?,
        tp_stop: if stop_ladder == phasor_model::StopLadderMode::Disabled {
            flex_or_f64(&tp_stop_b, n_rows, n_cols, per_col, d.tp_stop)?
        } else {
            array::ladder_steps(&tp_stop_b, n_cols, "tp_stop", d.tp_stop)?
        },
        tsl_stop: if stop_ladder == phasor_model::StopLadderMode::Disabled {
            flex_or_f64(&tsl_stop_b, n_rows, n_cols, per_col, d.tsl_stop)?
        } else {
            array::ladder_steps(&tsl_stop_b, n_cols, "tsl_stop", d.tsl_stop)?
        },
        tsl_th: flex_or_f64(&tsl_th_b, n_rows, n_cols, per_col, d.tsl_th)?,
        td_stop: flex_or_f64(&td_stop_b, n_rows, n_cols, per_col, d.td_stop)?,
        dt_stop: flex_or_f64(&dt_stop_b, n_rows, n_cols, per_col, d.dt_stop)?,
        stop_ladder,
        stop_order_type: stop_order_types
            .as_ref()
            .map_or(Ok(d.stop_order_type), |types| {
                FlexArray::per_column(types, n_cols).map_err(to_py_err)
            })?,
        stop_limit_delta: flex_or_f64(
            &stop_limit_delta_b,
            n_rows,
            n_cols,
            per_col,
            d.stop_limit_delta,
        )?,
        stop_entry_price,
        stop_exit_price,
        upon_stop_exit,
        upon_stop_update,
        order_type: order_types.as_ref().map_or(Ok(d.order_type), |types| {
            FlexArray::per_column(types, n_cols).map_err(to_py_err)
        })?,
        limit_delta: flex_or_f64(&limit_delta_b, n_rows, n_cols, per_col, d.limit_delta)?,
        delta_format,
        limit_reverse: flex_or_bool(&limit_reverse_b, n_rows, n_cols, d.limit_reverse)?,
        limit_order_price,
        limit_delay: flex_or_bool(&limit_delay_b, n_rows, n_cols, d.limit_delay)?,
        limit_tif: flex_or_f64(&limit_tif_b, n_rows, n_cols, per_col, d.limit_tif)?,
        limit_expiry: flex_or_f64(&limit_expiry_b, n_rows, n_cols, per_col, d.limit_expiry)?,
        upon_adj_limit_conflict,
        upon_opp_limit_conflict,
        upon_adj_stop_conflict,
        upon_opp_stop_conflict,
        order_mode,
        cash_deposits: deposits,
        cash_dividends: dividends,
        cash_earnings: earnings,
        funding_rate: funding,
        long_financing_rate: long_financing,
        short_financing_rate: short_financing,
        debt_rate: debt,
        borrow_rate: borrow,
        mark_price: mark,
        margin_mode: margin_modes.as_ref().map_or(Ok(d.margin_mode), |modes| {
            FlexArray::per_column(modes, n_cols).map_err(to_py_err)
        })?,
        maintenance_margin: flex_or_f64(
            &lq.maintenance_margin,
            n_rows,
            n_cols,
            OneDim::PerColumn,
            d.maintenance_margin,
        )?,
        liquidation_fee: flex_or_f64(
            &lq.liquidation_fee,
            n_rows,
            n_cols,
            OneDim::PerColumn,
            d.liquidation_fee,
        )?,
        cash_maintenance_margin: flex_or_f64(
            &lq.cash_maintenance_margin,
            n_rows,
            n_cols,
            OneDim::PerColumn,
            d.cash_maintenance_margin,
        )?,
        // The rows of a tier table are its tiers, as a ladder's are its steps.
        margin_tier_notional: array::ladder_steps(
            &lq.tier_notional,
            n_cols,
            "margin_tier_notional",
            d.margin_tier_notional,
        )?,
        margin_tier_rate: array::ladder_steps(
            &lq.tier_rate,
            n_cols,
            "margin_tier_rate",
            d.margin_tier_rate,
        )?,
        liquidation_step: flex_or_f64(
            &lq.liquidation_step,
            n_rows,
            n_cols,
            OneDim::PerColumn,
            d.liquidation_step,
        )?,
        auto_deleverage: flex_or_bool(&adl_b, n_rows, n_cols, d.auto_deleverage)?,
        resume: prior.as_ref().map(resume::Resumed::state),
        row_offset,
        cash_limit: flex_or_f64(&p.cash_limit, n_rows, n_cols, per_col, d.cash_limit)?,
        reject_prob: flex_or_f64(&p.reject_prob, n_rows, n_cols, per_col, d.reject_prob)?,
        seed,
        ..FromSignals::default()
    };

    // Two entry points rather than always taking the logged one: a run that is
    // not investigating anything should not allocate a journal. The GIL is
    // released for the run: every input is a borrowed slice the guards above
    // keep alive, and nothing in the loop touches Python.
    if flexible || row_wise {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "flexible and row_wise are for from_order_func",
        ));
    }
    let cbs = callbacks.map(callbacks::PyCallbacks::new).transpose()?;
    let builtin = builtin.map(builtin_of).transpose()?;
    if let Some(b) = &builtin {
        b.validate(args.use_stops).map_err(to_py_err)?;
    }
    if builtin.is_some() && cbs.is_some() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "the built-in rules (risk, breakeven, max_positions, max_exposure) and Python \
             callbacks cannot be combined; write the rule in the callback",
        ));
    }
    let logged = p.wants_logs();
    let run = py
        .detach(|| match &builtin {
            Some(b) => builtin_run(&args, b, threads, logged),
            None => signals_run(&args, cbs.as_ref(), threads, logged),
        })
        .map_err(|e| failed(cbs.as_ref(), e))?;

    // The starting cash is materialized per column so the portfolio can be
    // rebuilt without keeping Python's array alive.
    let init_cash_owned: Vec<f64> = (0..n_cols).map(|c| args.init_cash.get(0, c)).collect();
    if matches!(prior, Some(resume::Resumed::State(_))) {
        return resume::segment(py, run);
    }
    let (journal, init_cash_owned, in_outputs, last_state) = finished(
        py,
        run,
        prior,
        &groups,
        n_cols,
        init_cash_owned,
        cbs.is_some(),
    )?;
    let (orders, logs) = (journal.orders, journal.logs);

    let mut pf = PyPortfolio::assemble(
        orders,
        logs,
        close_flat.to_vec(),
        source_cols,
        column_map,
        init_cash_owned,
        groups,
        cash_sharing,
        n_rows,
        n_cols,
        Scoring {
            ann_factor,
            risk_free,
            trades_type,
            incl_open,
        },
    );
    pf.in_outputs = in_outputs;
    pf.last_state = Some(last_state);
    Ok(Bound::new(py, pf)?.into_any().unbind())
}

/// Runs a backtest from one order request per element.
///
/// The other shape of a vectorized run, and the more general one: instead of
/// signals resolved into sizes, `size` *is* the request, element by element. A
/// rebalance is a matrix of target weights; a schedule is a matrix that is zero
/// except on the bars that trade.
///
/// All six size types work here, unlike the signal path — nothing has to be
/// reconciled with a direction, so a target is unambiguous.
///
/// # Errors
///
/// A Python `ValueError` for a shape or layout that cannot be read, or for an
/// invariant the execution kernel refuses.
#[pyfunction]
#[pyo3(signature = (
    close,
    *,
    size = None,
    open = None,
    high = None,
    low = None,
    size_type = "amount",
    price = None,
    direction = "both",
    init_cash = None,
    fees = None,
    fixed_fees = None,
    slippage = None,
    min_size = None,
    max_size = None,
    size_granularity = None,
    leverage = None,
    leverage_mode = "lazy",
    account_type = None,
    initial_margin = None,
    margin_basis = None,
    multiplier = None,
    init_position = None,
    init_price = None,
    allow_negative_price = false,
    price_type = None,
    from_ago = None,
    bid_price = None,
    ask_price = None,
    price_area_vio_mode = None,
    force_exit = None,
    cash_limit = None,
    reject_prob = None,
    seed = 0,
    lock_cash = None,
    allow_partial = None,
    raise_reject = None,
    log = None,
    val_price = None,
    cash_deposits = None,
    cash_dividends = None,
    cash_earnings = None,
    funding_rate = None,
    long_financing_rate = None,
    short_financing_rate = None,
    debt_rate = None,
    borrow_rate = None,
    mark_price = None,
    margin_mode = None,
    maintenance_margin = None,
    liquidation_fee = None,
    cash_maintenance_margin = None,
    margin_tier_notional = None,
    margin_tier_rate = None,
    liquidation_step = None,
    auto_deleverage = None,
    ffill_val_price = true,
    update_value = false,
    ann_factor = 365.0,
    risk_free = 0.0,
    trades = "exit",
    incl_open = false,
    threads = 1,
    cash_sharing = false,
    group_lens = None,
    call_seq = None,
    column_map = None,
    callbacks = None,
    flexible = false,
    row_wise = false,
    resume = None,
    row_offset = 0,
))]
#[allow(
    clippy::too_many_arguments,
    reason = "one argument per simulation parameter, which is the API"
)]
fn from_orders(
    py: Python<'_>,
    close: &Bound<'_, PyAny>,
    size: Option<&Bound<'_, PyAny>>,
    open: Option<&Bound<'_, PyAny>>,
    high: Option<&Bound<'_, PyAny>>,
    low: Option<&Bound<'_, PyAny>>,
    size_type: &str,
    price: Option<&Bound<'_, PyAny>>,
    direction: &str,
    init_cash: Option<&Bound<'_, PyAny>>,
    fees: Option<&Bound<'_, PyAny>>,
    fixed_fees: Option<&Bound<'_, PyAny>>,
    slippage: Option<&Bound<'_, PyAny>>,
    min_size: Option<&Bound<'_, PyAny>>,
    max_size: Option<&Bound<'_, PyAny>>,
    size_granularity: Option<&Bound<'_, PyAny>>,
    leverage: Option<&Bound<'_, PyAny>>,
    leverage_mode: &str,
    account_type: Option<&Bound<'_, PyAny>>,
    initial_margin: Option<&Bound<'_, PyAny>>,
    margin_basis: Option<&Bound<'_, PyAny>>,
    multiplier: Option<&Bound<'_, PyAny>>,
    init_position: Option<&Bound<'_, PyAny>>,
    init_price: Option<&Bound<'_, PyAny>>,
    allow_negative_price: bool,
    price_type: Option<&str>,
    from_ago: Option<&Bound<'_, PyAny>>,
    bid_price: Option<&Bound<'_, PyAny>>,
    ask_price: Option<&Bound<'_, PyAny>>,
    price_area_vio_mode: Option<&str>,
    force_exit: Option<&Bound<'_, PyAny>>,
    cash_limit: Option<&Bound<'_, PyAny>>,
    reject_prob: Option<&Bound<'_, PyAny>>,
    seed: u64,
    lock_cash: Option<&Bound<'_, PyAny>>,
    allow_partial: Option<&Bound<'_, PyAny>>,
    raise_reject: Option<&Bound<'_, PyAny>>,
    log: Option<&Bound<'_, PyAny>>,
    val_price: Option<&Bound<'_, PyAny>>,
    cash_deposits: Option<&Bound<'_, PyAny>>,
    cash_dividends: Option<&Bound<'_, PyAny>>,
    cash_earnings: Option<&Bound<'_, PyAny>>,
    funding_rate: Option<&Bound<'_, PyAny>>,
    long_financing_rate: Option<&Bound<'_, PyAny>>,
    short_financing_rate: Option<&Bound<'_, PyAny>>,
    debt_rate: Option<&Bound<'_, PyAny>>,
    borrow_rate: Option<&Bound<'_, PyAny>>,
    mark_price: Option<&Bound<'_, PyAny>>,
    margin_mode: Option<&Bound<'_, PyAny>>,
    maintenance_margin: Option<&Bound<'_, PyAny>>,
    liquidation_fee: Option<&Bound<'_, PyAny>>,
    cash_maintenance_margin: Option<&Bound<'_, PyAny>>,
    margin_tier_notional: Option<&Bound<'_, PyAny>>,
    margin_tier_rate: Option<&Bound<'_, PyAny>>,
    liquidation_step: Option<&Bound<'_, PyAny>>,
    auto_deleverage: Option<&Bound<'_, PyAny>>,
    ffill_val_price: bool,
    update_value: bool,
    ann_factor: f64,
    risk_free: f64,
    trades: &str,
    incl_open: bool,
    threads: usize,
    cash_sharing: bool,
    group_lens: Option<Vec<usize>>,
    call_seq: Option<&Bound<'_, PyAny>>,
    column_map: Option<Vec<usize>>,
    callbacks: Option<&Bound<'_, PyDict>>,
    flexible: bool,
    row_wise: bool,
    resume: Option<&Bound<'_, PyAny>>,
    row_offset: usize,
) -> PyResult<Py<PyAny>> {
    let close_b = borrow_f64(close, "close")?;
    let Borrowed::Slice(close_arr) = &close_b else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "close must be a 2-D float64 array; it is what sets the run's shape",
        ));
    };
    let (close_flat, n_rows, source_cols) = columns_of(close_arr)?;
    let close_view = market_array(close_flat, n_rows, source_cols, column_map.as_deref())?;
    let n_cols = column_map.as_ref().map_or(source_cols, Vec::len);

    let init_cash_b = borrow_opt_f64(init_cash, "init_cash")?;
    let val_price_b = borrow_opt_f64(val_price, "val_price")?;
    let deposits_b = borrow_opt_f64(cash_deposits, "cash_deposits")?;
    let earnings_b = borrow_opt_f64(cash_earnings, "cash_earnings")?;
    let dividends_b = borrow_opt_f64(cash_dividends, "cash_dividends")?;
    let open_b = borrow_opt_f64(open, "open")?;
    let high_b = borrow_opt_f64(high, "high")?;
    let low_b = borrow_opt_f64(low, "low")?;
    let h = Holding::borrow(
        funding_rate,
        long_financing_rate,
        short_financing_rate,
        debt_rate,
        borrow_rate,
        mark_price,
    )?;
    let lq = Liquidation::borrow(
        maintenance_margin,
        liquidation_fee,
        cash_maintenance_margin,
        margin_tier_notional,
        margin_tier_rate,
        liquidation_step,
    )?;
    let adl_b = array::borrow_opt_bool(auto_deleverage, "auto_deleverage")?;
    let multiplier_b = borrow_opt_f64(multiplier, "multiplier")?;
    let init_position_b = borrow_opt_f64(init_position, "init_position")?;
    let init_price_b = borrow_opt_f64(init_price, "init_price")?;
    let bid_b = borrow_opt_f64(bid_price, "bid_price")?;
    let force_exit_b = array::borrow_opt_bool(force_exit, "force_exit")?;
    let from_ago_b = from_ago
        .map(|v| array::borrow_count(v, "from_ago"))
        .transpose()?;
    let ask_b = borrow_opt_f64(ask_price, "ask_price")?;
    let p = Shared::borrow(
        size,
        price,
        fees,
        fixed_fees,
        slippage,
        min_size,
        max_size,
        size_granularity,
        leverage,
        initial_margin,
        cash_limit,
        reject_prob,
        lock_cash,
        allow_partial,
        raise_reject,
        log,
    )?;

    let bad = pyo3::exceptions::PyValueError::new_err;
    let dir = parse::direction(direction).map_err(bad)?;
    let leverage_mode = parse::leverage_mode(leverage_mode).map_err(bad)?;
    let margin_bases = per_column(
        margin_basis,
        n_cols,
        "margin_basis",
        "'notional', 'unit', or one of them per column",
        parse::margin_basis,
    )?;
    let account_types = account_types(account_type, n_cols)?;
    let margin_modes = margin_modes(margin_mode, n_cols)?;
    let st = parse::size_type(size_type).map_err(bad)?;
    let trades_type = parse::trades_type(trades).map_err(bad)?;

    let d = FromOrders::default();
    let per_col = OneDim::PerColumn;
    let (groups, account_cash) = account_inputs(
        cash_sharing,
        group_lens,
        &init_cash_b,
        n_rows,
        n_cols,
        d.init_cash,
    )?;
    let call_seq_b = call_seq_arg(call_seq)?;
    let call_seq = resolve_call_seq(&call_seq_b, n_rows, n_cols)?;
    let group_of: Vec<usize> = (0..n_cols).map(|c| groups.of_column(c)).collect();
    let deposits = deposits_view(&deposits_b, n_rows, n_cols, &groups, &group_of)?;
    let earnings = flex_or_f64(&earnings_b, n_rows, n_cols, OneDim::PerRow, d.cash_earnings)?;
    // Per share, so it is market data: source-shaped, through `column_map`.
    let dividends = market_optional(
        "cash_dividends",
        &dividends_b,
        n_rows,
        source_cols,
        column_map.as_deref(),
        d.cash_dividends,
    )?;
    let map = column_map.as_deref();
    let market =
        |name, value, default| market_optional(name, value, n_rows, source_cols, map, default);
    let funding = market("funding_rate", &h.funding_rate, d.funding_rate)?;
    let long_financing = market(
        "long_financing_rate",
        &h.long_financing_rate,
        d.long_financing_rate,
    )?;
    let short_financing = market(
        "short_financing_rate",
        &h.short_financing_rate,
        d.short_financing_rate,
    )?;
    let borrow = market("borrow_rate", &h.borrow_rate, d.borrow_rate)?;
    let mark = market("mark_price", &h.mark_price, d.mark_price)?;
    let bid = market("bid_price", &bid_b, d.bid_price)?;
    let ask = market("ask_price", &ask_b, d.ask_price)?;
    let price_type = price_type
        .map(parse::price_type)
        .transpose()
        .map_err(bad)?
        .map_or(d.price_type, FlexArray::Scalar);
    let price_area_vio_mode = price_area_vio_mode
        .map(parse::price_area_vio_mode)
        .transpose()
        .map_err(bad)?
        .map_or(d.price_area_vio_mode, FlexArray::Scalar);
    // The account's cost of cash, so a parameter: a 1-D array is per bar.
    let debt = flex_or_f64(&h.debt_rate, n_rows, n_cols, OneDim::PerRow, d.debt_rate)?;

    // A run that continues a portfolio takes its records and state; see
    // `resume`.
    let prior = resume::resolve(
        resume,
        &resume::Shape {
            close: close_flat,
            n_rows,
            n_cols,
            source_cols,
            column_map: column_map.as_deref(),
            groups: &groups,
            cash_sharing,
        },
        row_offset,
    )?;

    let args = FromOrders {
        n_rows,
        n_cols,
        groups: groups.clone(),
        call_seq,
        close: close_view,
        open: market("open", &open_b, d.open)?,
        high: market("high", &high_b, d.high)?,
        low: market("low", &low_b, d.low)?,
        init_cash: FlexArray::per_column(&account_cash, n_cols).map_err(to_py_err)?,
        val_price: flex_or_f64(&val_price_b, n_rows, n_cols, per_col, d.val_price)?,
        direction: FlexArray::Scalar(dir),
        size_type: FlexArray::Scalar(st),
        size: flex_or_f64(&p.size, n_rows, n_cols, per_col, d.size)?,
        price: flex_or_f64(&p.price, n_rows, n_cols, per_col, d.price)?,
        fees: flex_or_f64(&p.fees, n_rows, n_cols, per_col, d.fees)?,
        fixed_fees: flex_or_f64(&p.fixed_fees, n_rows, n_cols, per_col, d.fixed_fees)?,
        slippage: flex_or_f64(&p.slippage, n_rows, n_cols, per_col, d.slippage)?,
        min_size: flex_or_f64(&p.min_size, n_rows, n_cols, per_col, d.min_size)?,
        max_size: flex_or_f64(&p.max_size, n_rows, n_cols, per_col, d.max_size)?,
        size_granularity: flex_or_f64(
            &p.size_granularity,
            n_rows,
            n_cols,
            per_col,
            d.size_granularity,
        )?,
        leverage: flex_or_f64(&p.leverage, n_rows, n_cols, per_col, d.leverage)?,
        leverage_mode: FlexArray::Scalar(leverage_mode),
        account_type: account_types.as_ref().map_or(Ok(d.account_type), |types| {
            FlexArray::per_column(types, n_cols).map_err(to_py_err)
        })?,
        initial_margin: flex_or_f64(&p.initial_margin, n_rows, n_cols, per_col, d.initial_margin)?,
        margin_basis: margin_bases.as_ref().map_or(Ok(d.margin_basis), |bases| {
            FlexArray::per_column(bases, n_cols).map_err(to_py_err)
        })?,
        // One per column: the wrapper maps a source column's through
        // `column_map`, and the engine refuses one that changes down a column.
        multiplier: flex_or_f64(
            &multiplier_b,
            n_rows,
            n_cols,
            OneDim::PerColumn,
            d.multiplier,
        )?,
        init_position: flex_or_f64(
            &init_position_b,
            n_rows,
            n_cols,
            OneDim::PerColumn,
            d.init_position,
        )?,
        init_price: flex_or_f64(
            &init_price_b,
            n_rows,
            n_cols,
            OneDim::PerColumn,
            d.init_price,
        )?,
        allow_negative_price,
        price_type,
        from_ago: match &from_ago_b {
            Some(v) => array::flex_count(v, n_rows, n_cols)?,
            None => d.from_ago,
        },
        bid_price: bid,
        ask_price: ask,
        price_area_vio_mode,
        force_exit: flex_or_bool(&force_exit_b, n_rows, n_cols, d.force_exit)?,
        lock_cash: flex_or_bool(&p.lock_cash, n_rows, n_cols, d.lock_cash)?,
        allow_partial: flex_or_bool(&p.allow_partial, n_rows, n_cols, d.allow_partial)?,
        raise_reject: flex_or_bool(&p.raise_reject, n_rows, n_cols, d.raise_reject)?,
        log: flex_or_bool(&p.log, n_rows, n_cols, d.log)?,
        cash_deposits: deposits,
        cash_dividends: dividends,
        cash_earnings: earnings,
        funding_rate: funding,
        long_financing_rate: long_financing,
        short_financing_rate: short_financing,
        debt_rate: debt,
        borrow_rate: borrow,
        mark_price: mark,
        margin_mode: margin_modes.as_ref().map_or(Ok(d.margin_mode), |modes| {
            FlexArray::per_column(modes, n_cols).map_err(to_py_err)
        })?,
        maintenance_margin: flex_or_f64(
            &lq.maintenance_margin,
            n_rows,
            n_cols,
            OneDim::PerColumn,
            d.maintenance_margin,
        )?,
        liquidation_fee: flex_or_f64(
            &lq.liquidation_fee,
            n_rows,
            n_cols,
            OneDim::PerColumn,
            d.liquidation_fee,
        )?,
        cash_maintenance_margin: flex_or_f64(
            &lq.cash_maintenance_margin,
            n_rows,
            n_cols,
            OneDim::PerColumn,
            d.cash_maintenance_margin,
        )?,
        // The rows of a tier table are its tiers, as a ladder's are its steps.
        margin_tier_notional: array::ladder_steps(
            &lq.tier_notional,
            n_cols,
            "margin_tier_notional",
            d.margin_tier_notional,
        )?,
        margin_tier_rate: array::ladder_steps(
            &lq.tier_rate,
            n_cols,
            "margin_tier_rate",
            d.margin_tier_rate,
        )?,
        liquidation_step: flex_or_f64(
            &lq.liquidation_step,
            n_rows,
            n_cols,
            OneDim::PerColumn,
            d.liquidation_step,
        )?,
        auto_deleverage: flex_or_bool(&adl_b, n_rows, n_cols, d.auto_deleverage)?,
        resume: prior.as_ref().map(resume::Resumed::state),
        row_offset,
        cash_limit: flex_or_f64(&p.cash_limit, n_rows, n_cols, per_col, d.cash_limit)?,
        reject_prob: flex_or_f64(&p.reject_prob, n_rows, n_cols, per_col, d.reject_prob)?,
        seed,
        ffill_val_price,
        update_value,
    };

    let cbs = callbacks.map(callbacks::PyCallbacks::new).transpose()?;
    let schedule = if row_wise {
        phasor_sim::Schedule::RowWise
    } else {
        phasor_sim::Schedule::Groups { n_threads: threads }
    };
    let logged = p.wants_logs();
    let run = py
        .detach(|| orders_run(&args, cbs.as_ref(), schedule, flexible, logged))
        .map_err(|e| failed(cbs.as_ref(), e))?;

    let init_cash_owned: Vec<f64> = (0..n_cols).map(|c| args.init_cash.get(0, c)).collect();
    if matches!(prior, Some(resume::Resumed::State(_))) {
        return resume::segment(py, run);
    }
    let (journal, init_cash_owned, in_outputs, last_state) = finished(
        py,
        run,
        prior,
        &groups,
        n_cols,
        init_cash_owned,
        cbs.is_some(),
    )?;
    let (orders, logs) = (journal.orders, journal.logs);

    let mut pf = PyPortfolio::assemble(
        orders,
        logs,
        close_flat.to_vec(),
        source_cols,
        column_map,
        init_cash_owned,
        groups,
        cash_sharing,
        n_rows,
        n_cols,
        Scoring {
            ann_factor,
            risk_free,
            trades_type,
            incl_open,
        },
    );
    pf.in_outputs = in_outputs;
    pf.last_state = Some(last_state);
    Ok(Bound::new(py, pf)?.into_any().unbind())
}

/// A run's records with logs or without, as one type, and its callback
/// states.
type PyRun = phasor_sim::Run<phasor_records::Journal, Py<PyAny>>;

/// A run without logs, or without callbacks, as a [`PyRun`].
fn as_py_run<T>(run: phasor_sim::Run<OrderRecords, T>) -> PyRun {
    phasor_sim::Run {
        records: phasor_records::Journal {
            orders: run.records,
            logs: LogRecords::new(),
        },
        states: Vec::new(),
        last_state: run.last_state,
    }
}

/// [`as_py_run`] for a logged run without callbacks.
fn logged_py_run(run: phasor_sim::Run<phasor_records::Journal>) -> PyRun {
    phasor_sim::Run {
        records: run.records,
        states: Vec::new(),
        last_state: run.last_state,
    }
}

/// [`as_py_run`] for a callback run without logs.
fn unlogged_py_run(run: phasor_sim::Run<OrderRecords, Py<PyAny>>) -> PyRun {
    phasor_sim::Run {
        records: phasor_records::Journal {
            orders: run.records,
            logs: LogRecords::new(),
        },
        states: run.states,
        last_state: run.last_state,
    }
}

/// `from_signals`, with Python callbacks or not, logged or not.
fn signals_run(
    args: &FromSignals<'_>,
    cbs: Option<&callbacks::PyCallbacks>,
    threads: usize,
    logged: bool,
) -> phasor_core::error::Result<PyRun> {
    use phasor_sim::{simulate_from_signal_func as func, simulate_from_signals_run as plain};
    match (cbs, logged) {
        (Some(cbs), true) => func(args, cbs, threads),
        (Some(cbs), false) => func::<OrderRecords, _>(args, cbs, threads).map(unlogged_py_run),
        (None, true) => plain(args, threads).map(logged_py_run),
        (None, false) => plain::<OrderRecords>(args, threads).map(as_py_run),
    }
}

/// The built-in rules from their Python parameters.
fn builtin_of(d: &Bound<'_, PyDict>) -> PyResult<phasor_sim::Builtin> {
    let get = |name: &str| -> PyResult<f64> {
        d.get_item(name)?
            .filter(|v| !v.is_none())
            .map_or(Ok(f64::NAN), |v| v.extract())
    };
    let b = phasor_sim::Builtin {
        risk: get("risk")?,
        risk_stop: get("risk_stop")?,
        breakeven: get("breakeven")?,
        max_positions: d
            .get_item("max_positions")?
            .filter(|v| !v.is_none())
            .map_or(Ok(0), |v| v.extract())?,
        max_exposure: get("max_exposure")?,
    };
    Ok(b)
}

/// `from_signals` with the built-in rules; their states are not kept.
fn builtin_run(
    args: &FromSignals<'_>,
    b: &phasor_sim::Builtin,
    threads: usize,
    logged: bool,
) -> phasor_core::error::Result<PyRun> {
    use phasor_sim::simulate_from_signal_func as func;
    if logged {
        func::<phasor_records::Journal, _>(args, b, threads).map(|run| phasor_sim::Run {
            records: run.records,
            states: Vec::new(),
            last_state: run.last_state,
        })
    } else {
        func::<OrderRecords, _>(args, b, threads).map(as_py_run)
    }
}

/// `from_orders`, with Python callbacks or not, flexible or not, logged or
/// not.
fn orders_run(
    args: &FromOrders<'_>,
    cbs: Option<&callbacks::PyCallbacks>,
    schedule: phasor_sim::Schedule,
    flexible: bool,
    logged: bool,
) -> phasor_core::error::Result<PyRun> {
    use phasor_records::Journal;
    use phasor_sim::simulate_from_order_func as func;
    let Some(cbs) = cbs else {
        return if logged {
            func::<Journal, _>(args, &phasor_sim::NoCallbacks, schedule).map(logged_py_run)
        } else {
            func::<OrderRecords, _>(args, &phasor_sim::NoCallbacks, schedule).map(as_py_run)
        };
    };
    let (flex, plain) = (
        callbacks::PyOrderFunc::<true>(cbs),
        callbacks::PyOrderFunc::<false>(cbs),
    );
    match (flexible, logged) {
        (false, false) => func::<OrderRecords, _>(args, &plain, schedule).map(unlogged_py_run),
        (true, false) => func::<OrderRecords, _>(args, &flex, schedule).map(unlogged_py_run),
        (false, true) => func(args, &plain, schedule),
        (true, true) => func(args, &flex, schedule),
    }
}

/// The error a run ended on: a callback's own exception when one raised.
fn failed(cbs: Option<&callbacks::PyCallbacks>, e: phasor_core::Error) -> PyErr {
    cbs.and_then(callbacks::PyCallbacks::take_error)
        .unwrap_or_else(|| to_py_err(e))
}

/// What the portfolio is built from: the records (after the prior's, when
/// the run continues one), the starting cash (the prior's), the callback
/// states and the state to continue from.
#[allow(
    clippy::type_complexity,
    reason = "the four parts a binding assembles a portfolio from"
)]
fn finished(
    py: Python<'_>,
    run: PyRun,
    prior: Option<resume::Resumed>,
    groups: &Groups,
    n_cols: usize,
    init_cash: Vec<f64>,
    with_callbacks: bool,
) -> PyResult<(
    phasor_records::Journal,
    Vec<f64>,
    Option<Py<PyAny>>,
    phasor_sim::SimState,
)> {
    let in_outputs = with_callbacks
        .then(|| pyo3::types::PyList::new(py, run.states).map(|l| l.into_any().unbind()))
        .transpose()?;
    let (journal, init_cash) = match prior {
        Some(resume::Resumed::Portfolio(mut prior)) => (
            phasor_records::Journal::chain(
                &[std::mem::take(&mut prior.journal), run.records],
                &resume::group_of(groups, n_cols),
            ),
            prior.init_cash,
        ),
        _ => (run.records, init_cash),
    };
    Ok((journal, init_cash, in_outputs, run.last_state))
}

/// Fills as a dict of NumPy arrays, as `pf.orders()` gives them.
fn orders_dict<'py>(py: Python<'py>, orders: &OrderRecords) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("col", orders.col.clone().into_pyarray(py))?;
    out.set_item("idx", orders.idx.clone().into_pyarray(py))?;
    out.set_item("size", orders.size.clone().into_pyarray(py))?;
    out.set_item("price", orders.price.clone().into_pyarray(py))?;
    out.set_item("fees", orders.fees.clone().into_pyarray(py))?;
    let side: Vec<i64> = orders.side.iter().map(|s| i64::from(*s)).collect();
    out.set_item("side", side.into_pyarray(py))?;
    if !orders.debt.is_empty() {
        out.set_item("debt", orders.debt.clone().into_pyarray(py))?;
        out.set_item("locked_cash", orders.locked_cash.clone().into_pyarray(py))?;
        out.set_item("margin", orders.margin.clone().into_pyarray(py))?;
        out.set_item("isolated", orders.isolated.clone().into_pyarray(py))?;
        out.set_item("liquidation", orders.liquidation.clone().into_pyarray(py))?;
        out.set_item("multiplier", orders.multiplier.clone().into_pyarray(py))?;
    }
    Ok(out)
}

/// Cash flows as a dict of NumPy arrays, as `pf.cash_records()` gives them.
fn cash_dict<'py>(
    py: Python<'py>,
    cash: &phasor_records::cash::CashRecords,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("col", cash.col.clone().into_pyarray(py))?;
    out.set_item("idx", cash.idx.clone().into_pyarray(py))?;
    out.set_item("amount", cash.amount.clone().into_pyarray(py))?;
    let kind: Vec<i64> = cash.kind.iter().map(|k| i64::from(*k)).collect();
    out.set_item("kind", kind.into_pyarray(py))?;
    Ok(out)
}

/// One field of a records dict, `None` when it is not there.
fn field<'py, T: FromPyObjectOwned<'py>>(
    d: &Bound<'py, PyDict>,
    name: &str,
) -> PyResult<Option<Vec<T>>> {
    d.get_item(name)?.map(|v| v.extract()).transpose()
}

/// A field a records dict must have.
fn required<'py, T: FromPyObjectOwned<'py>>(
    d: &Bound<'py, PyDict>,
    name: &str,
) -> PyResult<Vec<T>> {
    field(d, name)?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("the records have no {name:?}"))
    })
}

/// [`PyPortfolio::from_records`].
#[allow(clippy::too_many_arguments, reason = "the inputs of the analysis")]
fn records_portfolio(
    orders: &Bound<'_, PyDict>,
    close: &Bound<'_, PyAny>,
    init_cash: Option<&Bound<'_, PyAny>>,
    cash_records: Option<&Bound<'_, PyDict>>,
    group_lens: Option<Vec<usize>>,
    cash_sharing: bool,
    column_map: Option<Vec<usize>>,
    allow_negative_price: bool,
    scoring: Scoring,
) -> PyResult<PyPortfolio> {
    let bad = |m: String| pyo3::exceptions::PyValueError::new_err(m);
    let close_b = borrow_f64(close, "close")?;
    let Borrowed::Slice(close_arr) = &close_b else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "close must be a 2-D float64 array",
        ));
    };
    let (close_flat, n_rows, source_cols) = columns_of(close_arr)?;
    if let Some(map) = &column_map {
        if let Some(&c) = map.iter().find(|&&c| c >= source_cols) {
            return Err(bad(format!(
                "column_map names close column {c} of {source_cols}"
            )));
        }
    }
    let n_cols = column_map.as_ref().map_or(source_cols, Vec::len);
    let init_cash_b = borrow_opt_f64(init_cash, "init_cash")?;
    let (groups, cash) = account_inputs(
        cash_sharing,
        group_lens,
        &init_cash_b,
        n_rows,
        n_cols,
        FlexArray::Scalar(100.0),
    )?;
    let mut rec = OrderRecords::new();
    rec.col = required(orders, "col")?;
    rec.idx = required(orders, "idx")?;
    rec.size = required(orders, "size")?;
    rec.price = required(orders, "price")?;
    rec.fees = required(orders, "fees")?;
    rec.side = required::<i64>(orders, "side")?
        .into_iter()
        .map(|s| i8::try_from(s).unwrap_or(-1))
        .collect();
    if let Some(debt) = field(orders, "debt")? {
        rec.debt = debt;
        rec.locked_cash = required(orders, "locked_cash")?;
        rec.margin = required(orders, "margin")?;
        rec.isolated = field(orders, "isolated")?.unwrap_or_else(|| vec![false; rec.col.len()]);
        rec.liquidation =
            field(orders, "liquidation")?.unwrap_or_else(|| vec![false; rec.col.len()]);
        rec.multiplier = field(orders, "multiplier")?.unwrap_or_else(|| vec![1.0; rec.col.len()]);
    }
    let n = rec.col.len();
    let lens = [
        rec.idx.len(),
        rec.size.len(),
        rec.price.len(),
        rec.fees.len(),
        rec.side.len(),
    ];
    let balances = [
        rec.debt.len(),
        rec.locked_cash.len(),
        rec.margin.len(),
        rec.isolated.len(),
        rec.liquidation.len(),
        rec.multiplier.len(),
    ];
    if lens.iter().any(|&l| l != n) || balances.iter().any(|&l| l != 0 && l != n) {
        return Err(bad("the records' fields are of different lengths".into()));
    }
    for i in 0..n {
        if rec.col[i] < 0 || rec.col[i] as usize >= n_cols {
            return Err(bad(format!(
                "fill {i} is in column {} of {n_cols}",
                rec.col[i]
            )));
        }
        if rec.idx[i] < 0 || rec.idx[i] as usize >= n_rows {
            return Err(bad(format!(
                "fill {i} is on bar {} of {n_rows}",
                rec.idx[i]
            )));
        }
        if !(0..=1).contains(&rec.side[i]) {
            return Err(bad(format!("fill {i}'s side is not 0 (buy) or 1 (sell)")));
        }
    }
    if let Some(c) = cash_records {
        rec.cash.col = required(c, "col")?;
        rec.cash.idx = required(c, "idx")?;
        rec.cash.amount = required(c, "amount")?;
        rec.cash.kind = required::<i64>(c, "kind")?
            .into_iter()
            .map(|k| i8::try_from(k).unwrap_or(-1))
            .collect();
        let m = rec.cash.col.len();
        if [
            rec.cash.idx.len(),
            rec.cash.amount.len(),
            rec.cash.kind.len(),
        ]
        .iter()
        .any(|&l| l != m)
        {
            return Err(bad(
                "the cash records' fields are of different lengths".into()
            ));
        }
        for i in 0..m {
            if rec.cash.col[i] < 0 || rec.cash.col[i] as usize >= n_cols {
                return Err(bad(format!(
                    "cash flow {i} is in column {}",
                    rec.cash.col[i]
                )));
            }
            if rec.cash.idx[i] < 0 || rec.cash.idx[i] as usize >= n_rows {
                return Err(bad(format!("cash flow {i} is on bar {}", rec.cash.idx[i])));
            }
            // `CashFlowKind`'s discriminants run 0 (deposit) to 8 (initial
            // position).
            if !(0..=8).contains(&rec.cash.kind[i]) {
                return Err(bad(format!(
                    "cash flow {i}'s kind {} is not one",
                    rec.cash.kind[i]
                )));
            }
        }
    }
    rec.allow_negative_price = allow_negative_price;
    let pf = PyPortfolio::assemble(
        rec,
        LogRecords::new(),
        close_flat.to_vec(),
        source_cols,
        column_map,
        cash,
        groups,
        cash_sharing,
        n_rows,
        n_cols,
        scoring,
    );
    // Checked here rather than on first use: a fill the analysis refuses
    // should say so where the records came in.
    pf.core()?;
    Ok(pf)
}

fn market_array<'a>(
    data: &'a [f64],
    rows: usize,
    cols: usize,
    map: Option<&'a [usize]>,
) -> PyResult<FlexArray<'a, f64>> {
    match map {
        Some(map) => FlexArray::mapped(data, rows, cols, map),
        None => FlexArray::full(data, rows, cols),
    }
    .map_err(to_py_err)
}

/// Market data beside the close — the rest of the bar, dividends, rates an
/// instrument carries, a mark price — as the engine reads it: source-shaped,
/// through `column_map` when there is one.
fn market_optional<'a>(
    name: &str,
    value: &'a Option<Borrowed<'a>>,
    rows: usize,
    cols: usize,
    map: Option<&'a [usize]>,
    default: FlexArray<'a, f64>,
) -> PyResult<FlexArray<'a, f64>> {
    match value {
        Some(Borrowed::Slice(a)) => {
            let (data, r, c) = columns_of(a)?;
            if (r, c) != (rows, cols) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "{name} must have the source close shape ({rows}, {cols}); got ({r}, {c})"
                )));
            }
            market_array(data, rows, cols, map)
        }
        Some(Borrowed::Row(_)) if map.is_some() => Err(pyo3::exceptions::PyValueError::new_err(
            format!("a mapped {name} must be a source-shaped matrix or a scalar"),
        )),
        _ => flex_or_f64(value, rows, cols, OneDim::PerColumn, default),
    }
}

/// The extension module.
#[pymodule]
fn _phasor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<data::PyInstrumentId>()?;
    m.add_class::<data::PyBarType>()?;
    m.add_class::<PyPortfolio>()?;
    m.add_class::<callbacks::PyContext>()?;
    m.add_class::<resume::PySimState>()?;
    m.add_class::<resume::PySegment>()?;
    m.add_class::<PyFactorSet>()?;
    m.add_class::<expr::PyFactorGrid>()?;
    m.add_function(wrap_pyfunction!(from_signals, m)?)?;
    m.add_function(wrap_pyfunction!(from_orders, m)?)?;
    m.add_function(wrap_pyfunction!(expr::alpha158, m)?)?;
    m.add_function(wrap_pyfunction!(expr::alpha158_features, m)?)?;
    m.add_function(wrap_pyfunction!(expr::qlib_label, m)?)?;
    m.add_function(wrap_pyfunction!(expr::alpha360, m)?)?;
    m.add_function(wrap_pyfunction!(expr::wq101, m)?)?;
    m.add_function(wrap_pyfunction!(expr::wq101_formula, m)?)?;
    m.add_function(wrap_pyfunction!(expr::crossed_above, m)?)?;
    research::register(m)?;
    indicators::register(m)?;
    signal_fns::register(m)?;
    returns_fns::register(m)?;
    factor_eval::register(m)?;
    m.add(
        "__doc__",
        "Rust engine for phasor. Use the `phasor` package, not this.",
    )?;
    Ok(())
}

/// `call_seq` as Python passed it: a name, or a `(n_rows, n_cols)` `uintp`
/// matrix of within-group indices, column-major.
enum CallSeqArg<'py> {
    Named(String),
    Explicit(numpy::PyReadonlyArray2<'py, usize>),
}

fn call_seq_arg<'py>(value: Option<&Bound<'py, PyAny>>) -> PyResult<CallSeqArg<'py>> {
    let Some(value) = value else {
        return Ok(CallSeqArg::Named("default".to_owned()));
    };
    if let Ok(name) = value.extract::<String>() {
        return Ok(CallSeqArg::Named(name));
    }
    value
        .extract::<numpy::PyReadonlyArray2<'py, usize>>()
        .map(CallSeqArg::Explicit)
        .map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "call_seq must be 'default', 'auto' or a 2-D uintp array of call indices",
            )
        })
}

fn resolve_call_seq<'a>(
    value: &'a CallSeqArg<'a>,
    n_rows: usize,
    n_cols: usize,
) -> PyResult<CallSeq<'a>> {
    let bad = pyo3::exceptions::PyValueError::new_err;
    match value {
        CallSeqArg::Named(name) => match name.as_str() {
            "default" => Ok(CallSeq::Default),
            "auto" => Ok(CallSeq::Auto),
            _ => Err(bad(
                "call_seq must be 'default', 'auto' or an array".to_owned()
            )),
        },
        CallSeqArg::Explicit(a) => {
            let (flat, rows, cols) = array::columns_of(a)?;
            if rows != n_rows || cols != n_cols {
                return Err(bad(format!(
                    "a {rows}x{cols} call_seq against a {n_rows}x{n_cols} run"
                )));
            }
            Ok(CallSeq::Explicit(flat))
        }
    }
}
