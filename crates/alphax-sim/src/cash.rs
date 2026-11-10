//! Cash that enters or leaves an account without a fill.
//!
//! Two inputs, both shared by the order and signal loops:
//!
//! | Input | Booked | Read at | Recorded as |
//! |---|---|---|---|
//! | `cash_deposits` | at the bar's open, before any order is sized | the group's first column | [`CashFlowKind::Deposit`] |
//! | `cash_dividends` | at the bar's open, after the deposit | each column, times the position held | [`CashFlowKind::Dividend`] |
//! | `cash_earnings` | at the bar's close, after every order | each column | [`CashFlowKind::Earning`] |
//!
//! A deposit lands before the bar is valued, so a target-sized order on the
//! same bar already counts it: that is what lets a recurring deposit be
//! invested by the rebalance it arrives with. An earning lands after the
//! orders, so it is part of the bar's return and of the next bar's sizing.
//!
//! **A negative amount is capped at the free cash available**, never taking
//! the account below zero: a withdrawal or a cost the account cannot pay is
//! applied in part. The record holds the amount actually applied, which is
//! why the cap needs no second source of truth — the analysis layer rebuilds
//! the balance from what was booked, not from what was asked.
//!
//! `NaN` means no flow, as a `NaN` size means no order. An infinite amount is
//! an error: there is no finite balance it could produce.

use alphax_core::error::{Error, Result};
use alphax_core::flex::FlexArray;
use alphax_core::math::add;
use alphax_model::CashFlowKind;
use alphax_records::RecordSink;

/// Whether an input can never move cash, so the loop can skip it entirely.
///
/// The defaults are a scalar zero; checking for that once keeps a run that
/// uses neither input from paying a read per bar.
pub fn is_off(array: &FlexArray<'_, f64>) -> bool {
    matches!(*array, FlexArray::Scalar(v) if v.is_nan() || v == 0.0)
}

/// Whether a run books no cash flow at all, so a loop can run its copy that has
/// no code for one.
///
/// Branches on flags fixed for the run and never taken still cost the loops
/// time: the copy without them runs `from_orders` about 5% faster and
/// `from_signals` about 6% (one thread, interleaved rounds). On `from_orders`
/// it executes about as many instructions, so the cost is not work the
/// branches do; see `benchmarks/README.md`.
pub fn all_off(
    deposits: &FlexArray<'_, f64>,
    dividends: &FlexArray<'_, f64>,
    earnings: &FlexArray<'_, f64>,
) -> bool {
    is_off(deposits) && is_off(dividends) && is_off(earnings)
}

/// Books `amount` to the balance, capping an outflow at the free cash.
///
/// Returns what was applied. `free_cash` never exceeds `cash` — it is cash
/// less twice the open shorts' debt — so an outflow capped at it cannot take
/// cash below zero, which the execution kernel checks on every order. Free
/// cash itself can already be negative, after a short was bought back above
/// its entry; then nothing can be taken out at all.
fn apply(amount: f64, cash: &mut f64, free_cash: &mut f64) -> f64 {
    let applied = if amount >= 0.0 {
        amount
    } else {
        -(-amount).min(free_cash.max(0.0))
    };
    if applied != 0.0 {
        *cash = add(*cash, applied);
        *free_cash = add(*free_cash, applied);
    }
    applied
}

/// Reads one amount and rejects the values no balance can absorb.
pub fn read(array: &FlexArray<'_, f64>, name: &str, row: usize, col: usize) -> Result<f64> {
    let amount = array.get(row, col);
    if amount.is_infinite() {
        return Err(Error::Invariant(format!(
            "{name} is infinite at row {row}, column {col}"
        )));
    }
    Ok(if amount.is_nan() { 0.0 } else { amount })
}

/// Books the deposit for bar `row` to the group whose first column is `first`.
///
/// # Errors
///
/// Returns [`Error::Invariant`] for an infinite amount.
pub fn deposit<S: RecordSink>(
    deposits: &FlexArray<'_, f64>,
    row: usize,
    first: usize,
    cash: &mut f64,
    free_cash: &mut f64,
    sink: &mut S,
) -> Result<()> {
    let amount = read(deposits, "cash_deposits", row, first)?;
    if amount == 0.0 {
        return Ok(());
    }
    let applied = apply(amount, cash, free_cash);
    if applied != 0.0 {
        sink.push_cash(row, first, applied, CashFlowKind::Deposit);
    }
    Ok(())
}

/// Books every column's dividend for bar `row`, in layout order.
///
/// `dividends` is per share; `position(k)` is the `k`-th column's position at
/// the bar's open, before any of the bar's orders. That is the holding a
/// dividend on its ex-date is owed to: buying on the bar does not earn it and
/// selling on the bar does not forfeit it. Booked at the open, so the bar's
/// own rebalance can reinvest it.
///
/// # Errors
///
/// Returns [`Error::Invariant`] for an infinite amount.
pub fn pay_dividends<S: RecordSink>(
    dividends: &FlexArray<'_, f64>,
    row: usize,
    cols: std::ops::Range<usize>,
    position: impl Fn(usize) -> f64,
    cash: &mut f64,
    free_cash: &mut f64,
    sink: &mut S,
) -> Result<()> {
    for (k, col) in cols.enumerate() {
        let per_share = read(dividends, "cash_dividends", row, col)?;
        let held = position(k);
        if per_share == 0.0 || held == 0.0 {
            continue;
        }
        let applied = apply(held * per_share, cash, free_cash);
        if applied != 0.0 {
            sink.push_cash(row, col, applied, CashFlowKind::Dividend);
        }
    }
    Ok(())
}

/// Everything booked at a bar's open: the deposit, then the dividends.
///
/// One call behind one flag, and never inlined, so a run without either
/// input pays a single predictable branch per bar: each flag tested in the
/// bar loop itself cost about three instructions a bar, the loop having more
/// live values than registers.
///
/// # Errors
///
/// As [`deposit`] and [`pay_dividends`].
#[inline(never)]
#[allow(
    clippy::too_many_arguments,
    reason = "the two inputs, the bar, and the balances they move"
)]
pub fn at_open<S: RecordSink>(
    deposits: &FlexArray<'_, f64>,
    dividends: &FlexArray<'_, f64>,
    row: usize,
    cols: std::ops::Range<usize>,
    position: impl Fn(usize) -> f64,
    cash: &mut f64,
    free_cash: &mut f64,
    sink: &mut S,
) -> Result<()> {
    if !is_off(deposits) {
        deposit(deposits, row, cols.start, cash, free_cash, sink)?;
    }
    if !is_off(dividends) {
        pay_dividends(dividends, row, cols, position, cash, free_cash, sink)?;
    }
    Ok(())
}

/// Books every column's earning for bar `row`, in layout order.
///
/// # Errors
///
/// Returns [`Error::Invariant`] for an infinite amount.
#[inline(never)]
pub fn earn<S: RecordSink>(
    earnings: &FlexArray<'_, f64>,
    row: usize,
    cols: std::ops::Range<usize>,
    cash: &mut f64,
    free_cash: &mut f64,
    sink: &mut S,
) -> Result<()> {
    for col in cols {
        let amount = read(earnings, "cash_earnings", row, col)?;
        if amount == 0.0 {
            continue;
        }
        let applied = apply(amount, cash, free_cash);
        if applied != 0.0 {
            sink.push_cash(row, col, applied, CashFlowKind::Earning);
        }
    }
    Ok(())
}

/// Books `amount` of `kind` to column `col`'s account on `row` and records
/// it, as the loops book their cash flows: `NaN` is nothing and an outflow
/// is capped at the free cash. Returns what was applied.
///
/// For a custom loop: any cash that enters or leaves an account outside a
/// fill has to be a record, or the analysis cannot rebuild the balance.
///
/// # Errors
///
/// Returns [`Error::Invariant`] for an infinite amount.
pub fn book<S: RecordSink>(
    amount: f64,
    kind: CashFlowKind,
    row: usize,
    col: usize,
    cash: &mut f64,
    free_cash: &mut f64,
    sink: &mut S,
) -> Result<f64> {
    if amount.is_infinite() {
        return Err(Error::Invariant(format!(
            "a cash flow is infinite at row {row}, column {col}"
        )));
    }
    if amount.is_nan() || amount == 0.0 {
        return Ok(0.0);
    }
    let applied = apply(amount, cash, free_cash);
    if applied != 0.0 {
        sink.push_cash(row, col, applied, kind);
    }
    Ok(applied)
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "the balances here are exact sums of small binary fractions"
)]
mod tests {
    use super::*;
    use alphax_records::OrderRecords;

    #[test]
    fn the_default_inputs_are_off() {
        assert!(is_off(&FlexArray::Scalar(0.0)));
        assert!(is_off(&FlexArray::Scalar(-0.0)));
        assert!(is_off(&FlexArray::Scalar(f64::NAN)));
        assert!(!is_off(&FlexArray::Scalar(1.0)));
        assert!(!is_off(&FlexArray::PerColumn(&[0.0])));
    }

    #[test]
    fn an_outflow_is_capped_at_free_cash() {
        let (mut cash, mut free) = (100.0, 40.0);
        assert_eq!(apply(-50.0, &mut cash, &mut free), -40.0);
        assert_eq!((cash, free), (60.0, 0.0));
        // Nothing left to take: nothing is applied and nothing recorded.
        assert_eq!(apply(-1.0, &mut cash, &mut free), 0.0);
        assert_eq!((cash, free), (60.0, 0.0));
        assert_eq!(apply(5.0, &mut cash, &mut free), 5.0);
        assert_eq!((cash, free), (65.0, 5.0));
    }

    #[test]
    fn a_capped_withdrawal_records_what_was_applied() {
        let mut sink = OrderRecords::new();
        let (mut cash, mut free) = (10.0, 10.0);
        let deposits = [f64::NAN, -25.0];
        let array = FlexArray::per_row(&deposits, 2).unwrap();
        deposit(&array, 0, 3, &mut cash, &mut free, &mut sink).unwrap();
        assert!(sink.cash.is_empty(), "NaN is no flow");
        deposit(&array, 1, 3, &mut cash, &mut free, &mut sink).unwrap();
        assert_eq!(sink.cash.amount, [-10.0]);
        assert_eq!(sink.cash.col, [3]);
        assert_eq!(cash, 0.0);
    }

    #[test]
    fn an_infinite_amount_is_an_error() {
        let mut sink = OrderRecords::new();
        let (mut cash, mut free) = (10.0, 10.0);
        let array = FlexArray::Scalar(f64::INFINITY);
        assert!(earn(&array, 0, 0..1, &mut cash, &mut free, &mut sink).is_err());
    }
}
