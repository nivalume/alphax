//! Python callbacks for `from_orders` and `from_signals`, called directly.
//!
//! Each call takes the GIL, so a run with Python callbacks is orders of
//! magnitude slower than one without: it is for trying a rule out and for
//! debugging, and a rule worth running at scale belongs in Rust, through
//! [`alphax_sim::callbacks`].
//!
//! A callback that raises ends the run, and its exception is what the caller
//! sees: it is kept here and raised in place of the engine's error.

use std::ops::Range;
use std::sync::Mutex;

use alphax_core::error::{Error, Result};
use alphax_exec::{Order, OrderResult};
use alphax_model::{OrderSide, OrderStatus};
use alphax_sim::callbacks::{Context, Decision, Flow, OrderFunc, SignalFunc};
use alphax_sim::state::StopState;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyTuple};

use crate::parse;

/// The account on a bar, as a Python callback reads it. Per-column lists are
/// indexed by the column's place in its group.
#[pyclass(name = "Context", module = "alphax", frozen)]
#[derive(Debug)]
pub struct PyContext {
    /// The group.
    #[pyo3(get)]
    group: usize,
    /// The bar.
    #[pyo3(get)]
    row: usize,
    /// The group's first column.
    #[pyo3(get)]
    from_col: usize,
    /// One past the group's last column.
    #[pyo3(get)]
    to_col: usize,
    /// The group's cash.
    #[pyo3(get)]
    cash: f64,
    /// The cash not tied up by positions.
    #[pyo3(get)]
    free_cash: f64,
    /// What the group is worth at its valuation prices.
    #[pyo3(get)]
    value: f64,
    /// Each column's signed position.
    #[pyo3(get)]
    position: Vec<f64>,
    /// Each column's debt.
    #[pyo3(get)]
    debt: Vec<f64>,
    /// The cash each column's position ties up, or its posted margin.
    #[pyo3(get)]
    locked_cash: Vec<f64>,
    /// Each column's valuation price: in `from_signals`' `signal_func` and
    /// `adjust_func`, the last bar's.
    #[pyo3(get)]
    val_price: Vec<f64>,
    /// Each column's close on this bar.
    #[pyo3(get)]
    close: Vec<f64>,
    /// Each column's contract multiplier.
    #[pyo3(get)]
    multiplier: Vec<f64>,
}

#[pymethods]
impl PyContext {
    fn __repr__(&self) -> String {
        format!(
            "Context(group={}, row={}, cols={}..{}, cash={}, value={}, position={:?})",
            self.group, self.row, self.from_col, self.to_col, self.cash, self.value, self.position
        )
    }
}

impl PyContext {
    fn of(ctx: &Context<'_>) -> Self {
        let n = ctx.len();
        Self {
            group: ctx.group,
            row: ctx.row,
            from_col: ctx.cols.start,
            to_col: ctx.cols.end,
            cash: ctx.cash,
            free_cash: ctx.free_cash,
            value: ctx.value,
            position: (0..n).map(|k| ctx.position(k)).collect(),
            debt: (0..n).map(|k| ctx.debt(k)).collect(),
            locked_cash: (0..n).map(|k| ctx.locked_cash(k)).collect(),
            val_price: (0..n).map(|k| ctx.val_price(k)).collect(),
            close: (0..n).map(|k| ctx.close(k)).collect(),
            multiplier: (0..n).map(|k| ctx.multiplier(k)).collect(),
        }
    }
}

/// The Python functions of a run, by hook. Any may be missing.
pub struct PyCallbacks {
    pre_group: Option<Py<PyAny>>,
    pre_segment: Option<Py<PyAny>>,
    order: Option<Py<PyAny>>,
    post_order: Option<Py<PyAny>>,
    post_segment: Option<Py<PyAny>>,
    adjust: Option<Py<PyAny>>,
    signal: Option<Py<PyAny>>,
    /// The first exception a callback raised.
    error: Mutex<Option<PyErr>>,
}

const HOOKS: &str = "pre_group, pre_segment, order, post_order, post_segment, adjust, signal";

impl PyCallbacks {
    /// From `{hook: function}`; `None` values are no function.
    ///
    /// # Errors
    ///
    /// A `ValueError` for a hook that does not exist.
    pub fn new(funcs: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut cbs = Self {
            pre_group: None,
            pre_segment: None,
            order: None,
            post_order: None,
            post_segment: None,
            adjust: None,
            signal: None,
            error: Mutex::new(None),
        };
        for (key, value) in funcs.iter() {
            let name: String = key.extract()?;
            let func = (!value.is_none()).then(|| value.unbind());
            let slot = match name.as_str() {
                "pre_group" => &mut cbs.pre_group,
                "pre_segment" => &mut cbs.pre_segment,
                "order" => &mut cbs.order,
                "post_order" => &mut cbs.post_order,
                "post_segment" => &mut cbs.post_segment,
                "adjust" => &mut cbs.adjust,
                "signal" => &mut cbs.signal,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "no callback {other:?}; the hooks are {HOOKS}"
                    )));
                }
            };
            *slot = func;
        }
        Ok(cbs)
    }

    /// The exception a callback raised, if the run ended on one.
    pub fn take_error(&self) -> Option<PyErr> {
        self.error.lock().ok().and_then(|mut e| e.take())
    }

    /// Runs `f` with the GIL, keeping its exception.
    fn call<T>(&self, f: impl FnOnce(Python<'_>) -> PyResult<T>) -> Result<T> {
        Python::attach(|py| {
            f(py).map_err(|err| {
                let message = err.to_string();
                if let Ok(mut slot) = self.error.lock() {
                    if slot.is_none() {
                        *slot = Some(err);
                    }
                }
                Error::Invariant(format!("a Python callback raised: {message}"))
            })
        })
    }

    fn make_state(&self, group: usize, cols: Range<usize>) -> Result<Py<PyAny>> {
        self.call(|py| match &self.pre_group {
            Some(f) => Ok(f.call1(py, (group, cols.start, cols.end))?),
            None => Ok(py.None()),
        })
    }

    /// Calls `f(ctx, *args, state)` and reads its answer as a [`Flow`].
    fn flow<A>(
        &self,
        func: Option<&Py<PyAny>>,
        ctx: &Context<'_>,
        args: A,
        state: &Py<PyAny>,
    ) -> Result<Flow>
    where
        A: for<'py> FnOnce(Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>>,
    {
        let Some(f) = func else {
            return Ok(Flow::Continue);
        };
        self.call(|py| {
            let out = call(py, f, ctx, args, state)?;
            read_flow(&out)
        })
    }
}

/// Calls `f(ctx, *args(py), state)`.
fn call<'py, A>(
    py: Python<'py>,
    f: &Py<PyAny>,
    ctx: &Context<'_>,
    args: A,
    state: &Py<PyAny>,
) -> PyResult<Bound<'py, PyAny>>
where
    A: FnOnce(Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>>,
{
    let mut all = vec![Bound::new(py, PyContext::of(ctx))?.into_any()];
    all.extend(args(py)?);
    all.push(state.bind(py).clone());
    f.bind(py).call1(PyTuple::new(py, all)?)
}

fn read_flow(out: &Bound<'_, PyAny>) -> PyResult<Flow> {
    if out.is_none() {
        return Ok(Flow::Continue);
    }
    match out.extract::<String>().as_deref() {
        Ok("skip") => Ok(Flow::SkipSegment),
        Ok("stop") => Ok(Flow::StopGroup),
        _ => Err(PyValueError::new_err(format!(
            "a callback returns None, 'skip' or 'stop'; got {out}"
        ))),
    }
}

/// Applies a callback's answer to `order`: a number is its size, a dict sets
/// the fields it names, `None` leaves it.
fn apply_order(out: &Bound<'_, PyAny>, order: &mut Order) -> PyResult<()> {
    if out.is_none() {
        return Ok(());
    }
    if let Ok(size) = out.extract::<f64>() {
        order.size = size;
        return Ok(());
    }
    let dict = out.cast::<PyDict>().map_err(|_| {
        PyValueError::new_err(format!(
            "an order callback returns None, a size or a dict of order fields; got {out}"
        ))
    })?;
    for (key, value) in dict.iter() {
        let name: String = key.extract()?;
        let float = || value.extract::<f64>();
        let flag = || value.extract::<bool>();
        match name.as_str() {
            "size" => order.size = float()?,
            "price" => order.price = float()?,
            "size_type" => {
                order.size_type =
                    parse::size_type(&value.extract::<String>()?).map_err(PyValueError::new_err)?;
            }
            "direction" => {
                order.direction =
                    parse::direction(&value.extract::<String>()?).map_err(PyValueError::new_err)?;
            }
            "fees" => order.fees = float()?,
            "fixed_fees" => order.fixed_fees = float()?,
            "slippage" => order.slippage = float()?,
            "min_size" => order.min_size = float()?,
            "max_size" => order.max_size = float()?,
            "size_granularity" => order.size_granularity = float()?,
            "leverage" => order.leverage = float()?,
            "initial_margin" => order.initial_margin = float()?,
            "cash_limit" => order.cash_limit = float()?,
            "reject_prob" => order.reject_prob = float()?,
            "lock_cash" => order.lock_cash = flag()?,
            "allow_partial" => order.allow_partial = flag()?,
            "raise_reject" => order.raise_reject = flag()?,
            "log" => order.log = flag()?,
            other => {
                return Err(PyValueError::new_err(format!("no order field {other:?}")));
            }
        }
    }
    Ok(())
}

fn result_dict<'py>(py: Python<'py>, result: &OrderResult) -> PyResult<Bound<'py, PyAny>> {
    let d = PyDict::new(py);
    d.set_item("filled", result.is_filled())?;
    d.set_item("size", result.size)?;
    d.set_item("price", result.price)?;
    d.set_item("fees", result.fees)?;
    d.set_item(
        "side",
        result.side.map(|s| match s {
            OrderSide::Buy => "buy",
            OrderSide::Sell => "sell",
        }),
    )?;
    d.set_item(
        "status",
        match result.status {
            OrderStatus::Filled => "filled",
            OrderStatus::Ignored => "ignored",
            OrderStatus::Rejected => "rejected",
        },
    )?;
    d.set_item("status_info", result.status_info.map(|i| i.describe()))?;
    Ok(d.into_any())
}

/// `from_orders`' hooks; `FLEX` says whether `order` is a flex function.
pub struct PyOrderFunc<'c, const FLEX: bool>(pub &'c PyCallbacks);

impl<const FLEX: bool> OrderFunc for PyOrderFunc<'_, FLEX> {
    type State = Py<PyAny>;
    const FLEX: bool = FLEX;

    fn pre_group(&self, group: usize, cols: Range<usize>) -> Result<Py<PyAny>> {
        self.0.make_state(group, cols)
    }

    fn pre_segment(&self, ctx: &Context<'_>, state: &mut Py<PyAny>) -> Result<Flow> {
        self.0
            .flow(self.0.pre_segment.as_ref(), ctx, |_| Ok(Vec::new()), state)
    }

    fn order(
        &self,
        ctx: &Context<'_>,
        k: usize,
        order: &mut Order,
        state: &mut Py<PyAny>,
    ) -> Result<()> {
        let Some(f) = &self.0.order else {
            return Ok(());
        };
        self.0.call(|py| {
            let out = call(
                py,
                f,
                ctx,
                |py| Ok(vec![k.into_pyobject(py)?.into_any()]),
                state,
            )?;
            apply_order(&out, order)
        })
    }

    fn flex_order(
        &self,
        ctx: &Context<'_>,
        call_idx: usize,
        state: &mut Py<PyAny>,
    ) -> Result<Option<(usize, Order)>> {
        let Some(f) = &self.0.order else {
            return Ok(None);
        };
        self.0.call(|py| {
            let out = call(
                py,
                f,
                ctx,
                |py| Ok(vec![call_idx.into_pyobject(py)?.into_any()]),
                state,
            )?;
            if out.is_none() {
                return Ok(None);
            }
            let (k, request): (usize, Bound<'_, PyAny>) = out.extract().map_err(|_| {
                PyValueError::new_err(format!(
                    "a flexible order callback returns None or (column, order); got {out}"
                ))
            })?;
            if k >= ctx.len() {
                return Err(PyValueError::new_err(format!(
                    "column {k} is not in a group of {}",
                    ctx.len()
                )));
            }
            let mut order = ctx.order(k);
            apply_order(&request, &mut order)?;
            Ok(Some((k, order)))
        })
    }

    fn post_order(
        &self,
        ctx: &Context<'_>,
        k: usize,
        _order: &Order,
        result: &OrderResult,
        state: &mut Py<PyAny>,
    ) -> Result<Flow> {
        self.0.flow(
            self.0.post_order.as_ref(),
            ctx,
            |py| {
                Ok(vec![
                    k.into_pyobject(py)?.into_any(),
                    result_dict(py, result)?,
                ])
            },
            state,
        )
    }

    fn post_segment(&self, ctx: &Context<'_>, state: &mut Py<PyAny>) -> Result<Flow> {
        self.0
            .flow(self.0.post_segment.as_ref(), ctx, |_| Ok(Vec::new()), state)
    }
}

/// The stop fields a callback reads and may set.
fn stops_dict<'py>(py: Python<'py>, stops: &StopState) -> PyResult<Bound<'py, PyAny>> {
    let d = PyDict::new(py);
    d.set_item("sl_price", stops.sl_price)?;
    d.set_item("sl_stop", stops.sl_stop)?;
    d.set_item("sl_trail", stops.sl_trail)?;
    d.set_item("tp_price", stops.tp_price)?;
    d.set_item("tp_stop", stops.tp_stop)?;
    d.set_item("tsl_price", stops.tsl_price)?;
    d.set_item("tsl_stop", stops.tsl_stop)?;
    d.set_item("tsl_th", stops.tsl_th)?;
    Ok(d.into_any())
}

fn apply_stops(out: &Bound<'_, PyAny>, stops: &mut StopState) -> PyResult<()> {
    if out.is_none() {
        return Ok(());
    }
    let dict = out.cast::<PyDict>().map_err(|_| {
        PyValueError::new_err(format!(
            "adjust_func returns None or a dict of stop fields; got {out}"
        ))
    })?;
    for (key, value) in dict.iter() {
        let name: String = key.extract()?;
        match name.as_str() {
            "sl_price" => stops.sl_price = value.extract()?,
            "sl_stop" => stops.sl_stop = value.extract()?,
            "sl_trail" => stops.sl_trail = value.extract()?,
            "tp_price" => stops.tp_price = value.extract()?,
            "tp_stop" => stops.tp_stop = value.extract()?,
            "tsl_price" => stops.tsl_price = value.extract()?,
            "tsl_stop" => stops.tsl_stop = value.extract()?,
            "tsl_th" => stops.tsl_th = value.extract()?,
            other => {
                return Err(PyValueError::new_err(format!(
                    "no stop field {other:?}; they are sl_price, sl_stop, sl_trail, tp_price, \
                     tp_stop, tsl_price, tsl_stop and tsl_th"
                )));
            }
        }
    }
    Ok(())
}

fn apply_decision(out: &Bound<'_, PyAny>, decision: &mut Decision) -> PyResult<()> {
    if out.is_none() {
        return Ok(());
    }
    if let Ok((le, lx, se, sx)) = out.extract::<(bool, bool, bool, bool)>() {
        decision.signals.long_entry = le;
        decision.signals.long_exit = lx;
        decision.signals.short_entry = se;
        decision.signals.short_exit = sx;
        return Ok(());
    }
    let dict = out.cast::<PyDict>().map_err(|_| {
        PyValueError::new_err(format!(
            "signal_func returns None, (long_entry, long_exit, short_entry, short_exit) or a \
             dict; got {out}"
        ))
    })?;
    for (key, value) in dict.iter() {
        let name: String = key.extract()?;
        match name.as_str() {
            "long_entry" | "entry" => decision.signals.long_entry = value.extract()?,
            "long_exit" | "exit" => decision.signals.long_exit = value.extract()?,
            "short_entry" => decision.signals.short_entry = value.extract()?,
            "short_exit" => decision.signals.short_exit = value.extract()?,
            "size" => decision.size = value.extract()?,
            "size_type" => {
                decision.size_type =
                    parse::size_type(&value.extract::<String>()?).map_err(PyValueError::new_err)?;
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "no signal field {other:?}; they are long_entry, long_exit, short_entry, \
                     short_exit, size and size_type"
                )));
            }
        }
    }
    Ok(())
}

impl SignalFunc for PyCallbacks {
    type State = Py<PyAny>;

    fn pre_group(&self, group: usize, cols: Range<usize>) -> Result<Py<PyAny>> {
        self.make_state(group, cols)
    }

    fn adjust(
        &self,
        ctx: &Context<'_>,
        k: usize,
        stops: &mut StopState,
        state: &mut Py<PyAny>,
    ) -> Result<()> {
        let Some(f) = &self.adjust else {
            return Ok(());
        };
        self.call(|py| {
            let out = call(
                py,
                f,
                ctx,
                |py| {
                    Ok(vec![
                        k.into_pyobject(py)?.into_any(),
                        stops_dict(py, stops)?,
                    ])
                },
                state,
            )?;
            apply_stops(&out, stops)
        })
    }

    fn signals(
        &self,
        ctx: &Context<'_>,
        k: usize,
        decision: &mut Decision,
        state: &mut Py<PyAny>,
    ) -> Result<()> {
        let Some(f) = &self.signal else {
            return Ok(());
        };
        self.call(|py| {
            let out = call(
                py,
                f,
                ctx,
                |py| Ok(vec![k.into_pyobject(py)?.into_any()]),
                state,
            )?;
            apply_decision(&out, decision)
        })
    }

    fn post_order(
        &self,
        ctx: &Context<'_>,
        k: usize,
        result: &OrderResult,
        state: &mut Py<PyAny>,
    ) -> Result<Flow> {
        self.flow(
            self.post_order.as_ref(),
            ctx,
            |py| {
                Ok(vec![
                    k.into_pyobject(py)?.into_any(),
                    result_dict(py, result)?,
                ])
            },
            state,
        )
    }

    fn post_segment(&self, ctx: &Context<'_>, state: &mut Py<PyAny>) -> Result<Flow> {
        self.flow(self.post_segment.as_ref(), ctx, |_| Ok(Vec::new()), state)
    }
}
