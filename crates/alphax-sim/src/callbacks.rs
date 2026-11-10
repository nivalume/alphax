//! User code inside the bar loop: callbacks for `from_orders` and
//! `from_signals`.
//!
//! A callback is not an event handler. The loop is the same bar-by-bar loop,
//! with the same kernel and the same records; a callback is called at fixed
//! points of each bar and may change what the loop would have done there:
//!
//! | `from_orders` ([`OrderFunc`]) | `from_signals` ([`SignalFunc`]) | Called |
//! |---|---|---|
//! | [`pre_group`](OrderFunc::pre_group) | [`pre_group`](SignalFunc::pre_group) | once per group, before its first bar; returns the group's state |
//! | [`pre_segment`](OrderFunc::pre_segment) | | each bar, after prices, liquidation and the group's value, before any order |
//! | | [`adjust`](SignalFunc::adjust) | each bar and column, before its stops are tested |
//! | [`order`](OrderFunc::order) or [`flex_order`](OrderFunc::flex_order) | [`signals`](SignalFunc::signals) | each column's order (or signals and size) for the bar |
//! | [`post_order`](OrderFunc::post_order) | [`post_order`](SignalFunc::post_order) | after each order executes, filled or not |
//! | [`post_segment`](OrderFunc::post_segment) | [`post_segment`](SignalFunc::post_segment) | each bar, after earnings and holding charges |
//!
//! Each group gets its own state (`State`), made by `pre_group` and handed
//! back with the records, in group order. The callback itself is shared
//! (`&self`) and must be `Sync` for the parallel entry points; groups run in
//! parallel as they do without callbacks, so anything a callback shares
//! across groups needs [`Schedule::RowWise`](crate::from_orders::Schedule::RowWise).
//!
//! A callback reads the account through a [`Context`] and changes it only
//! through orders: everything that moves cash or a position still goes
//! through the kernel and into the records.
//!
//! # Cost when unused
//!
//! The loops are generic over the callback. [`NoCallbacks`] sets `ON` to
//! false, and every call site is behind `if C::ON`, so the copy the plain
//! entry points run has no code for any of it.

use std::ops::Range;

use alphax_core::error::Result;
use alphax_core::flex::FlexArray;
use alphax_exec::{Order, OrderResult, SizeType, ValidatedOrder};

use crate::signals::Signals;
use crate::state::{ColumnState, StopState};

/// What the loop does after a callback returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Flow {
    /// Carry on.
    #[default]
    Continue,
    /// From `pre_segment`: place no order on this bar. Anywhere else: as
    /// `Continue`.
    SkipSegment,
    /// Stop trading the group: no further order and no further callback.
    /// The bars that remain still book cash flows, holding charges and
    /// liquidations, which belong to the positions left open.
    StopGroup,
}

/// The group's account on the current bar, read-only.
///
/// Positions, debts and prices are per column of the group, indexed from 0
/// (`k`); the column of the run is `cols.start + k`. Prices are as the
/// strategy wrote them, whatever the contract multiplier.
#[derive(Debug, Clone, Copy)]
pub struct Context<'s> {
    /// The group.
    pub group: usize,
    /// The bar.
    pub row: usize,
    /// The group's columns.
    pub cols: &'s Range<usize>,
    /// The group's cash.
    pub cash: f64,
    /// The cash not tied up by positions.
    pub free_cash: f64,
    /// What the group is worth at its valuation prices: the value a
    /// target-percent order is a fraction of. In `from_orders` it moves
    /// within a bar only with `update_value`.
    pub value: f64,
    pub(crate) columns: &'s [ColumnState],
    pub(crate) mult: &'s [f64],
    pub(crate) scaled: bool,
    pub(crate) orders: &'s [ValidatedOrder],
    pub(crate) close: &'s FlexArray<'s, f64>,
}

impl Context<'_> {
    /// The number of columns in the group.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.columns.len()
    }

    /// Whether the group has no columns.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    /// Column `k`'s contract multiplier.
    #[must_use]
    pub fn multiplier(&self, k: usize) -> f64 {
        self.mult.get(k).copied().unwrap_or(1.0)
    }

    /// Column `k`'s signed position.
    #[must_use]
    pub fn position(&self, k: usize) -> f64 {
        self.columns[k].position
    }

    /// Column `k`'s debt: see [`ColumnState::debt`].
    #[must_use]
    pub fn debt(&self, k: usize) -> f64 {
        self.columns[k].debt
    }

    /// The cash column `k`'s position ties up, or its posted margin.
    #[must_use]
    pub fn locked_cash(&self, k: usize) -> f64 {
        self.columns[k].locked_cash
    }

    /// Column `k`'s close on this bar.
    #[must_use]
    pub fn close(&self, k: usize) -> f64 {
        self.close.get(self.row, self.cols.start + k)
    }

    /// Column `k`'s valuation price: this bar's in `from_orders`; in
    /// `from_signals`' `adjust` and `signals`, which run before the bar's
    /// prices are resolved, the last bar's.
    #[must_use]
    pub fn val_price(&self, k: usize) -> f64 {
        let p = self.columns[k].last_val_price;
        if self.scaled {
            p / self.multiplier(k)
        } else {
            p
        }
    }

    /// Column `k`'s stops (`from_signals`).
    #[must_use]
    pub fn stops(&self, k: usize) -> &StopState {
        &self.columns[k].stops
    }

    /// Column `k`'s state as the loop keeps it. In `from_orders` its
    /// valuation price is times the multiplier.
    #[must_use]
    pub fn column(&self, k: usize) -> &ColumnState {
        &self.columns[k]
    }

    /// Column `k`'s request on this bar: its fees, size limits, account
    /// settings and the rest as the run's arrays give them. In `from_orders`
    /// its size, size type and direction are this bar's and its price is
    /// the bar's resolved order price.
    #[must_use]
    pub fn order(&self, k: usize) -> Order {
        let mut order = *self.orders[k].order();
        if self.scaled {
            order.price /= self.multiplier(k);
        }
        order
    }
}

/// Callbacks for `from_orders`: see the [module](self) for when each runs.
///
/// Every method but [`pre_group`](Self::pre_group) has a default that
/// changes nothing. An error from any of them ends the run with that error.
pub trait OrderFunc {
    /// What each group carries from bar to bar; returned with the records.
    type State: Send;

    /// Whether the loop calls the callbacks at all. Only [`NoCallbacks`]
    /// turns it off.
    const ON: bool = true;

    /// Whether orders come from [`flex_order`](Self::flex_order) rather
    /// than [`order`](Self::order): any number per bar, for any column, in
    /// any sequence.
    const FLEX: bool = false;

    /// The state of `group`, whose columns are `cols`.
    ///
    /// # Errors
    ///
    /// Any error ends the run.
    fn pre_group(&self, group: usize, cols: Range<usize>) -> Result<Self::State>;

    /// Before the bar's orders; [`Flow::SkipSegment`] places none.
    ///
    /// # Errors
    ///
    /// Any error ends the run.
    fn pre_segment(&self, ctx: &Context<'_>, state: &mut Self::State) -> Result<Flow> {
        let _ = (ctx, state);
        Ok(Flow::Continue)
    }

    /// Column `k`'s order on this bar, in the sequence `call_seq` gives.
    ///
    /// `order` comes in as the run's arrays make it (a `NaN` size is no
    /// order) and is executed as it goes out. Its price may be `±inf`, read
    /// as in the arrays: `+inf` the bar's close, `-inf` the previous close.
    ///
    /// # Errors
    ///
    /// Any error ends the run.
    fn order(
        &self,
        ctx: &Context<'_>,
        k: usize,
        order: &mut Order,
        state: &mut Self::State,
    ) -> Result<()> {
        let _ = (ctx, k, order, state);
        Ok(())
    }

    /// With [`FLEX`](Self::FLEX): the bar's next order and its column, or
    /// `None` when the bar has no more. `call` counts the calls on this bar
    /// from 0. [`Context::order`] gives a column's request to start from.
    ///
    /// # Errors
    ///
    /// Any error ends the run.
    fn flex_order(
        &self,
        ctx: &Context<'_>,
        call: usize,
        state: &mut Self::State,
    ) -> Result<Option<(usize, Order)>> {
        let _ = (ctx, call, state);
        Ok(None)
    }

    /// After column `k`'s `order` executed: the request as executed and the
    /// result, both at the strategy's prices.
    ///
    /// # Errors
    ///
    /// Any error ends the run.
    fn post_order(
        &self,
        ctx: &Context<'_>,
        k: usize,
        order: &Order,
        result: &OrderResult,
        state: &mut Self::State,
    ) -> Result<Flow> {
        let _ = (ctx, k, order, result, state);
        Ok(Flow::Continue)
    }

    /// At the bar's end, after earnings and holding charges.
    ///
    /// # Errors
    ///
    /// Any error ends the run.
    fn post_segment(&self, ctx: &Context<'_>, state: &mut Self::State) -> Result<Flow> {
        let _ = (ctx, state);
        Ok(Flow::Continue)
    }
}

/// What a `from_signals` column decides on a bar before the signal pipeline
/// resolves it: its four signals, and its size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Decision {
    /// The bar's signals, as the arrays give them.
    pub signals: Signals,
    /// The size of the order the signals make, as `size`.
    pub size: f64,
    /// How `size` is read.
    pub size_type: SizeType,
}

/// Callbacks for `from_signals`: see the [module](self) for when each runs.
pub trait SignalFunc {
    /// What each group carries from bar to bar; returned with the records.
    type State: Send;

    /// Whether the loop calls the callbacks at all. Only [`NoCallbacks`]
    /// turns it off.
    const ON: bool = true;

    /// The state of `group`, whose columns are `cols`.
    ///
    /// # Errors
    ///
    /// Any error ends the run.
    fn pre_group(&self, group: usize, cols: Range<usize>) -> Result<Self::State>;

    /// Column `k`'s stops, before this bar tests them: move a stop-loss to
    /// break-even, tighten a ladder, disarm one. Only the stops of an open
    /// position are tested.
    ///
    /// # Errors
    ///
    /// Any error ends the run.
    fn adjust(
        &self,
        ctx: &Context<'_>,
        k: usize,
        stops: &mut StopState,
        state: &mut Self::State,
    ) -> Result<()> {
        let _ = (ctx, k, stops, state);
        Ok(())
    }

    /// Column `k`'s signals and size on this bar, in place of the arrays'.
    /// `decision` comes in as the arrays give it; with `order_mode` its size
    /// is the order.
    ///
    /// # Errors
    ///
    /// Any error ends the run.
    fn signals(
        &self,
        ctx: &Context<'_>,
        k: usize,
        decision: &mut Decision,
        state: &mut Self::State,
    ) -> Result<()> {
        let _ = (ctx, k, decision, state);
        Ok(())
    }

    /// After an order of column `k` executed, filled or not; `result`'s price
    /// is the strategy's.
    ///
    /// # Errors
    ///
    /// Any error ends the run.
    fn post_order(
        &self,
        ctx: &Context<'_>,
        k: usize,
        result: &OrderResult,
        state: &mut Self::State,
    ) -> Result<Flow> {
        let _ = (ctx, k, result, state);
        Ok(Flow::Continue)
    }

    /// At the bar's end, after earnings and holding charges.
    ///
    /// # Errors
    ///
    /// Any error ends the run.
    fn post_segment(&self, ctx: &Context<'_>, state: &mut Self::State) -> Result<Flow> {
        let _ = (ctx, state);
        Ok(Flow::Continue)
    }
}

/// No callbacks: what the plain entry points run.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoCallbacks;

impl OrderFunc for NoCallbacks {
    type State = ();
    const ON: bool = false;

    fn pre_group(&self, _group: usize, _cols: Range<usize>) -> Result<()> {
        Ok(())
    }
}

impl SignalFunc for NoCallbacks {
    type State = ();
    const ON: bool = false;

    fn pre_group(&self, _group: usize, _cols: Range<usize>) -> Result<()> {
        Ok(())
    }
}

/// How many units to buy or sell so that a move from `entry` to `stop`
/// loses `risk` of `value`: `risk × value / |entry − stop|`, over the
/// contract multiplier. `NaN` when the distance is 0 or not finite, which
/// the kernel ignores.
///
/// The usual sizing for a strategy with a stop: an [`OrderFunc::order`] or
/// [`SignalFunc::signals`] sets its order's size to it, as `Amount`.
#[must_use]
pub fn risk_size(value: f64, risk: f64, entry: f64, stop: f64, multiplier: f64) -> f64 {
    let distance = (entry - stop).abs() * multiplier;
    if distance.is_finite() && distance > 0.0 {
        risk * value / distance
    } else {
        f64::NAN
    }
}
