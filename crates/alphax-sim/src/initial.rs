//! Positions held before the first bar: `init_position` at `init_price`.
//!
//! Analysis rebuilds every balance from the records, so an initial position
//! is recorded as what it would have taken to open: a fill on the first bar
//! at `init_price` (the first bar's close when `NaN`), with no fees or
//! slippage, through the kernel so that its debt, collateral and margin are
//! the kernel's own.
//!
//! - **A cash column** paid for its position before the run. Its fill is
//!   balanced by a [`CashFlowKind::InitialPosition`] record of the cost, so
//!   the cash stays `init_cash`, and the account starts worth `init_cash`
//!   plus the position at `init_price`. Analysis counts that record as
//!   capital brought in, like a deposit, so the first bar's return is the
//!   position's move from `init_price` to the close.
//! - **A margin column** has posted margin and holds an open trade: its fill
//!   opens it as any fill would, the margin comes out of free cash, and its
//!   equity is `init_cash` at `init_price`.
//!
//! No stop is armed for an initial position: it was not entered in the run.

use alphax_core::error::{Error, Result};
use alphax_core::flex::FlexArray;
use alphax_exec::{Direction, ProcessOrderState, SizeType, ValidatedOrder, execute_validated};
use alphax_model::CashFlowKind;
use alphax_records::{Balances, RecordSink};

use crate::state::ColumnState;

/// Whether any column of `cols` starts with a position.
pub(crate) fn any(positions: &FlexArray<'_, f64>, cols: std::ops::Range<usize>) -> bool {
    !matches!(positions, FlexArray::Scalar(v) if *v == 0.0)
        && cols.into_iter().any(|c| positions.get(0, c) != 0.0)
}

/// The run's per-column inputs an initial fill needs.
pub(crate) struct Columns<'s> {
    /// Each column's request on the first bar, to take its account settings.
    pub(crate) orders: &'s [ValidatedOrder],
    /// Which are margin columns; empty in a run without balances.
    pub(crate) margin: &'s [bool],
    /// Which are isolated; empty in a run without balances.
    pub(crate) isolated: &'s [bool],
    /// Each column's multiplier; empty in a run without balances.
    pub(crate) mult: &'s [f64],
    /// Whether the loop keeps valuation prices times the multiplier
    /// (`from_orders`) rather than multiplying them where read.
    pub(crate) scaled_val_price: bool,
}

/// Opens every column's initial position, before the first bar.
///
/// # Errors
///
/// Returns [`Error::Invariant`] for a position or price that is not finite,
/// a price that is not above 0, or a position the account cannot hold: a
/// short whose collateral, or a margin position whose margin, `init_cash`
/// does not cover.
#[allow(
    clippy::too_many_arguments,
    reason = "the column inputs, the group's balances and the sink"
)]
pub(crate) fn open<S: RecordSink>(
    positions: &FlexArray<'_, f64>,
    prices: &FlexArray<'_, f64>,
    close: &FlexArray<'_, f64>,
    cols: std::ops::Range<usize>,
    inputs: &Columns<'_>,
    state: &mut [ColumnState],
    cash: &mut f64,
    free_cash: &mut f64,
    sink: &mut S,
    fills: &mut i64,
) -> Result<()> {
    let levered = !inputs.mult.is_empty();
    for (k, col) in cols.enumerate() {
        let position = positions.get(0, col);
        if position == 0.0 {
            continue;
        }
        let mut price = prices.get(0, col);
        if price.is_nan() {
            price = close.get(0, col);
        }
        if !position.is_finite() || !price.is_finite() || price <= 0.0 {
            return Err(Error::Invariant(format!(
                "column {col}: init_position must be finite and init_price (or the first close) \
                 finite and above 0; got {position} at {price}"
            )));
        }
        let m = if levered { inputs.mult[k] } else { 1.0 };
        let margin = levered && inputs.margin[k];
        let mut order = *inputs.orders[k].order();
        order.size = position;
        order.size_type = SizeType::Amount;
        order.direction = Direction::Both;
        order.price = price * m;
        order.fees = 0.0;
        order.fixed_fees = 0.0;
        order.slippage = 0.0;
        order.min_size = 0.0;
        order.max_size = f64::INFINITY;
        order.size_granularity = f64::NAN;
        order.cash_limit = f64::INFINITY;
        order.reject_prob = 0.0;
        order.lock_cash = false;
        order.allow_partial = false;
        order.raise_reject = true;
        order.log = false;
        // A cash column paid before the run: lend it the cost to pay again.
        let cost = if margin { 0.0 } else { position * price * m };
        let process = ProcessOrderState {
            cash: *cash + cost,
            position: 0.0,
            debt: 0.0,
            locked_cash: 0.0,
            free_cash: *free_cash + cost,
            val_price: price * m,
            value: *cash + cost,
            oidx: 0,
            lidx: 0,
        };
        let (after, result) = execute_validated(&process, &ValidatedOrder::new(order), None)?;
        let filled = result.is_filled() && after.position.to_bits() == position.to_bits();
        let Some(side) = result.side.filter(|_| filled) else {
            return Err(Error::Invariant(format!(
                "column {col}: init_position {position} at {price} cannot be held with this cash: {}",
                result
                    .status_info
                    .map_or("only part of it fills", alphax_model::StatusInfo::describe)
            )));
        };
        if cost != 0.0 {
            sink.push_cash(0, col, cost, CashFlowKind::InitialPosition);
        }
        sink.push_order(0, col, result.size, result.price / m, 0.0, side);
        if levered {
            sink.push_balances(Balances {
                debt: after.debt,
                locked_cash: after.locked_cash,
                margin,
                isolated: inputs.isolated[k],
                liquidation: false,
                multiplier: m,
            });
        }
        *fills += 1;
        *cash = after.cash;
        *free_cash = after.free_cash;
        state[k].position = after.position;
        state[k].debt = after.debt;
        state[k].locked_cash = after.locked_cash;
        state[k].last_val_price = if inputs.scaled_val_price {
            price * m
        } else {
            price
        };
    }
    Ok(())
}
