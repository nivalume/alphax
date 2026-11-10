//! What the loops carry for each column from one bar to the next.
//!
//! One structure for both loops: `from_orders` reads the account part, and
//! `from_signals` the stops and the resting limit order as well. Every field
//! is public and plain data, so a run's state can be read, saved and handed
//! back once the loops take it (F). The account's own cash and free cash
//! are still locals of the loops; they join it when something reads them.

use alphax_exec::{Direction, SizeType};
use alphax_model::{DeltaFormat, LimitOrderPrice};

/// One column's state between bars.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColumnState {
    /// Signed position; negative is short.
    pub position: f64,
    /// What the column owes: a long's borrowing, a short's proceeds, or a
    /// margin column's cost basis. See [`alphax_exec::ExecuteOrderState`].
    pub debt: f64,
    /// The own cash the position ties up, or the margin it posted.
    pub locked_cash: f64,
    /// The last valuation price. `from_orders` keeps it multiplied by the
    /// column's contract multiplier, `from_signals` keeps the price.
    pub last_val_price: f64,
    /// The stops armed on the position (`from_signals`).
    pub stops: StopState,
    /// The limit order resting in the column (`from_signals`).
    pub limit: LimitOrder,
}

impl ColumnState {
    /// A flat column with nothing armed or resting and no price yet.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            position: 0.0,
            debt: 0.0,
            locked_cash: 0.0,
            last_val_price: f64::NAN,
            stops: StopState::cleared(),
            limit: LimitOrder::none(),
        }
    }

    /// What the column owes on a long: its debt, which on a short is a
    /// reservation instead and already in the position's value.
    pub(crate) fn long_debt(&self) -> f64 {
        if self.position > 0.0 { self.debt } else { 0.0 }
    }

    /// What is subtracted from `position × price` to leave the column's
    /// share of the value: a cash column's [`long_debt`](Self::long_debt), a
    /// margin column's signed cost basis.
    pub(crate) fn owed(&self, margin: bool) -> f64 {
        if margin {
            crate::margin::owed(self.position, self.debt)
        } else {
            self.long_debt()
        }
    }
}

impl Default for ColumnState {
    fn default() -> Self {
        Self::new()
    }
}

/// The stops armed on a column's position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StopState {
    /// The price the stop-loss is measured from; trails with the position.
    pub sl_price: f64,
    /// The stop-loss distance, as a fraction; `NaN` is none.
    pub sl_stop: f64,
    /// Whether the stop-loss trails the best price reached.
    pub sl_trail: bool,
    /// The price the take-profit is measured from; fixed at entry.
    pub tp_price: f64,
    /// The take-profit distance, as a fraction; `NaN` is none.
    pub tp_stop: f64,
    /// The trailing stop's best price since the entry. See `crate::stops`.
    pub tsl_price: f64,
    /// The trailing stop's distance from it, as a fraction; `NaN` is none.
    pub tsl_stop: f64,
    /// How far the price must first move from `tsl_anchor` for the trailing
    /// stop to be live.
    pub tsl_th: f64,
    /// The price the trailing stop's threshold is measured from.
    pub tsl_anchor: f64,
    /// Whether the trailing stop is live.
    pub tsl_active: bool,
    /// The first row a `td_stop` exits on; `usize::MAX` is none.
    pub td_row: usize,
    /// The first row a `dt_stop` exits on; `usize::MAX` is none.
    pub dt_row: usize,
    /// The stop-loss ladder's current step.
    pub sl_step: usize,
    /// The take-profit ladder's current step.
    pub tp_step: usize,
    /// The trailing stop ladder's current step.
    pub tsl_step: usize,
    /// How the distances read: fractions, percentages, prices or levels.
    pub delta_format: DeltaFormat,
    /// The size of the position the ladders were armed on.
    pub ladder_base: f64,
    /// Whether the stops were armed by `crate::stops`, which a trailing
    /// stop, a time stop or a ladder needs.
    pub extended: bool,
}

impl StopState {
    /// No stop armed.
    #[must_use]
    pub const fn cleared() -> Self {
        Self {
            sl_price: f64::NAN,
            sl_stop: f64::NAN,
            sl_trail: false,
            tp_price: f64::NAN,
            tp_stop: f64::NAN,
            tsl_price: f64::NAN,
            tsl_stop: f64::NAN,
            tsl_th: f64::NAN,
            tsl_anchor: f64::NAN,
            tsl_active: false,
            td_row: usize::MAX,
            dt_row: usize::MAX,
            sl_step: 0,
            tp_step: 0,
            tsl_step: 0,
            delta_format: DeltaFormat::Percent,
            ladder_base: 0.0,
            extended: false,
        }
    }

    /// Whether any stop is armed.
    #[must_use]
    pub const fn armed(&self) -> bool {
        !self.sl_stop.is_nan() || !self.tp_stop.is_nan() || self.extended
    }
}

/// A limit order resting in a column. See [`crate::limit`].
///
/// It carries the order its signal was sized into, against the position the
/// column held when it was placed: that is why any other fill in the column
/// cancels it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LimitOrder {
    /// The limit price; `NaN` when nothing rests.
    pub price: f64,
    /// Whether it buys.
    pub buy: bool,
    /// Whether it fills when the market moves through the price the other
    /// way: a buy at or above it, a sell at or below it.
    pub reverse: bool,
    /// What it fills at.
    pub order_price: LimitOrderPrice,
    /// The sized request, as [`alphax_exec::ValidatedOrder::set_bar`] takes it.
    pub size: f64,
    /// How `size` is read.
    pub size_type: SizeType,
    /// Which sides the request may take.
    pub direction: Direction,
    /// The bar it was placed on.
    pub created: usize,
    /// The last bar it may fill on; `usize::MAX` is good till cancelled.
    pub last_row: usize,
}

impl LimitOrder {
    /// Nothing resting.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            price: f64::NAN,
            buy: false,
            reverse: false,
            order_price: LimitOrderPrice::Limit,
            size: 0.0,
            size_type: SizeType::Amount,
            direction: Direction::Both,
            created: 0,
            last_row: 0,
        }
    }

    /// Whether an order rests.
    #[must_use]
    pub const fn is_pending(&self) -> bool {
        !self.price.is_nan()
    }
}
