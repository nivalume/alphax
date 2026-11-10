//! What holding a position costs or earns at each bar's close.
//!
//! Four charges, booked per column after the bar's orders and its
//! `cash_earnings`, on the position the orders left:
//!
//! | Input | Charged on | Amount | Recorded as |
//! |---|---|---|---|
//! | `funding_rate` | any position | `−position × price × rate`: longs pay a positive rate, shorts receive it | [`CashFlowKind::Funding`] |
//! | `long_financing_rate`, `short_financing_rate` | any position | `−|position| × price × rate`, the rate for the position's side | [`CashFlowKind::Financing`] |
//! | `debt_rate` | a cash column's long | `−debt × rate`: what the long borrowed | [`CashFlowKind::Interest`] |
//! | `borrow_rate` | a cash column's short | `−|position| × price × rate` | [`CashFlowKind::BorrowFee`] |
//! | `debt_rate` (the account's first column's) | an account's cash below zero, beyond what its leveraged longs borrowed | `−overdraft × rate` | [`CashFlowKind::Interest`], to the first column |
//!
//! `price` is `mark_price`, or the close where it is `NaN`, or the column's
//! last valuation price where both are; with none of them the charge is
//! skipped, as settling is. A positive rate is a cost to the holder. For a
//! column with a contract multiplier, `price` is multiplied by it: the
//! charge is on the notional.
//!
//! Every rate is **per bar**: what one bar's holding costs, for the time
//! until the next bar. A bar with no settlement is `NaN` or zero. Turning an
//! annual rate into one per bar needs the bars' timestamps, which the engine
//! does not have; the Python layer does it (`alphax.rates.per_bar`).
//!
//! A perpetual's funding is the case where the short's rate is minus the
//! long's, so it is one input; financing has two, because a broker's rates
//! for the two sides differ and both can be costs. Interest and the borrow fee
//! are the two halves of a cash account's leverage: a long that borrowed cash
//! pays on the cash, a short that borrowed shares pays on their value. A margin
//! column borrows neither, so neither applies to it.
//!
//! **A charge is never capped.** It is owed whatever the balance, unlike a
//! `cash_earnings` charge, which stops at the free cash. Any of these inputs
//! therefore runs the account as one that may borrow — the loops' `LEV` copy
//! — so cash may go below zero, after which the account can close but not
//! open. That also keeps their code out of a run that uses none of them.
//!
//! An **isolated** margin column pays its charges out of its own margin, as
//! an exchange takes funding from an isolated position: the margin it posted
//! and the account's cash both change, its free cash does not. A margin
//! worn below the maintenance margin by charges is liquidated as one worn
//! by the price.
//!
//! Like every flow that is not a fill, a charge is recorded with the amount
//! applied, so the analysis layer rebuilds the balance from records alone.

use alphax_core::error::Result;
use alphax_core::flex::FlexArray;
use alphax_core::math::add;
use alphax_model::CashFlowKind;
use alphax_records::RecordSink;

use crate::cash::{is_off, read};
use crate::state::ColumnState;

/// The inputs of the charges, borrowed from a loop's arguments.
#[derive(Clone, Copy)]
pub(crate) struct Rates<'s, 'a> {
    pub(crate) funding: &'s FlexArray<'a, f64>,
    pub(crate) long_financing: &'s FlexArray<'a, f64>,
    pub(crate) short_financing: &'s FlexArray<'a, f64>,
    pub(crate) debt: &'s FlexArray<'a, f64>,
    pub(crate) borrow: &'s FlexArray<'a, f64>,
    pub(crate) mark_price: &'s FlexArray<'a, f64>,
    pub(crate) close: &'s FlexArray<'a, f64>,
}

impl Rates<'_, '_> {
    /// Whether no rate can ever charge anything, so a run can skip them all.
    pub(crate) fn all_off(&self) -> bool {
        is_off(self.funding)
            && is_off(self.long_financing)
            && is_off(self.short_financing)
            && is_off(self.debt)
            && is_off(self.borrow)
    }
}

/// The group's columns as the charges read them: their state after the
/// bar's orders, which an isolated column's charge changes.
pub(crate) struct Columns<'s> {
    pub(crate) states: &'s mut [ColumnState],
    pub(crate) margin: &'s [bool],
    pub(crate) isolated: &'s [bool],
    /// Each column's contract multiplier, which the charges' price is
    /// multiplied by.
    pub(crate) mult: &'s [f64],
    /// Whether the states' valuation prices are already times the
    /// multiplier (`from_orders`) rather than multiplied where read.
    pub(crate) scaled_val_price: bool,
}

/// Applies `amount`, uncapped, and records it: out of the free cash, or out
/// of an isolated column's own margin.
#[allow(
    clippy::too_many_arguments,
    reason = "the charge, where it is booked and the balances it moves"
)]
fn book<S: RecordSink>(
    amount: f64,
    row: usize,
    col: usize,
    kind: CashFlowKind,
    cash: &mut f64,
    free_cash: &mut f64,
    own_margin: Option<&mut f64>,
    sink: &mut S,
) {
    // `NaN` only when no price was known; zero when the rate was.
    if amount == 0.0 || amount.is_nan() {
        return;
    }
    *cash = add(*cash, amount);
    match own_margin {
        Some(locked) => *locked = add(*locked, amount),
        None => *free_cash = add(*free_cash, amount),
    }
    sink.push_cash(row, col, amount, kind);
}

/// Books every column's charges for bar `row`, in layout order, and within a
/// column in the order of the module table.
///
/// # Errors
///
/// Returns [`alphax_core::Error::Invariant`] for an infinite rate.
#[inline(never)]
pub(crate) fn charge<S: RecordSink>(
    rates: &Rates<'_, '_>,
    row: usize,
    cols: std::ops::Range<usize>,
    columns: Columns<'_>,
    cash: &mut f64,
    free_cash: &mut f64,
    sink: &mut S,
) -> Result<()> {
    let first = cols.start;
    for (k, col) in cols.enumerate() {
        let st = &mut columns.states[k];
        let (position, debt) = (st.position, st.debt);
        if position == 0.0 {
            continue;
        }
        let margin = columns.margin[k];
        let mut price = rates.mark_price.get(row, col);
        if price.is_nan() {
            price = rates.close.get(row, col);
        }
        price *= columns.mult[k];
        if price.is_nan() {
            price = if columns.scaled_val_price {
                st.last_val_price
            } else {
                st.last_val_price * columns.mult[k]
            };
        }
        let long = position > 0.0;
        let isolated = columns.isolated[k];

        let funding = read(rates.funding, "funding_rate", row, col)?;
        if funding != 0.0 {
            let amount = -position * price * funding;
            book(
                amount,
                row,
                col,
                CashFlowKind::Funding,
                cash,
                free_cash,
                isolated.then_some(&mut st.locked_cash),
                sink,
            );
        }
        let financing = if long {
            read(rates.long_financing, "long_financing_rate", row, col)?
        } else {
            read(rates.short_financing, "short_financing_rate", row, col)?
        };
        if financing != 0.0 {
            let amount = -position.abs() * price * financing;
            book(
                amount,
                row,
                col,
                CashFlowKind::Financing,
                cash,
                free_cash,
                isolated.then_some(&mut st.locked_cash),
                sink,
            );
        }
        if margin {
            continue;
        }
        if long {
            let rate = read(rates.debt, "debt_rate", row, col)?;
            if rate != 0.0 && debt > 0.0 {
                book(
                    -debt * rate,
                    row,
                    col,
                    CashFlowKind::Interest,
                    cash,
                    free_cash,
                    None,
                    sink,
                );
            }
        } else {
            let rate = read(rates.borrow, "borrow_rate", row, col)?;
            if rate != 0.0 {
                let amount = -position.abs() * price * rate;
                book(
                    amount,
                    row,
                    col,
                    CashFlowKind::BorrowFee,
                    cash,
                    free_cash,
                    None,
                    sink,
                );
            }
        }
    }
    // An account that has lost more than its cash owes the difference, and
    // pays interest on it as a long pays on what it borrowed. A leveraged
    // long's borrowing is in the cash too, and pays on its own debt above:
    // only what is owed beyond it. After every column's charges, so it is on
    // what the bar leaves owed.
    let borrowed: f64 = columns
        .states
        .iter()
        .zip(columns.margin)
        .filter(|(_, margin)| !**margin)
        .map(|(st, _)| st.long_debt())
        .sum();
    let overdraft = -(*cash + borrowed);
    if overdraft > 0.0 {
        let rate = read(rates.debt, "debt_rate", row, first)?;
        if rate != 0.0 {
            book(
                -overdraft * rate,
                row,
                first,
                CashFlowKind::Interest,
                cash,
                free_cash,
                None,
                sink,
            );
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "every amount here is an exact product of small binary fractions"
)]
mod tests {
    use super::*;
    use alphax_records::OrderRecords;

    const ZERO: FlexArray<'static, f64> = FlexArray::Scalar(0.0);
    const NAN: FlexArray<'static, f64> = FlexArray::Scalar(f64::NAN);

    /// A column as these tests give it.
    #[derive(Debug, Clone, Copy)]
    struct Held {
        position: f64,
        debt: f64,
        margin: bool,
        val_price: f64,
        multiplier: f64,
    }

    /// [`charge`] on columns given as [`Held`], none isolated.
    fn charge_held<S: RecordSink>(
        rates: &Rates<'_, '_>,
        row: usize,
        cols: std::ops::Range<usize>,
        held: impl Fn(usize) -> Held,
        cash: &mut f64,
        free_cash: &mut f64,
        sink: &mut S,
    ) -> Result<()> {
        let hs: Vec<Held> = (0..cols.len()).map(held).collect();
        let mut states: Vec<ColumnState> = hs
            .iter()
            .map(|h| ColumnState {
                position: h.position,
                debt: h.debt,
                last_val_price: h.val_price,
                ..ColumnState::new()
            })
            .collect();
        let margin: Vec<bool> = hs.iter().map(|h| h.margin).collect();
        let isolated = vec![false; hs.len()];
        let mult: Vec<f64> = hs.iter().map(|h| h.multiplier).collect();
        let columns = Columns {
            states: &mut states,
            margin: &margin,
            isolated: &isolated,
            mult: &mult,
            scaled_val_price: true,
        };
        charge(rates, row, cols, columns, cash, free_cash, sink)
    }

    fn held(position: f64, debt: f64, margin: bool) -> Held {
        Held {
            position,
            debt,
            margin,
            val_price: 8.0,
            multiplier: 1.0,
        }
    }

    #[test]
    fn funding_is_paid_by_longs_and_received_by_shorts() {
        let rate = FlexArray::Scalar(0.25);
        let close = FlexArray::Scalar(10.0);
        let rates = Rates {
            funding: &rate,
            long_financing: &ZERO,
            short_financing: &ZERO,
            debt: &ZERO,
            borrow: &ZERO,
            mark_price: &NAN,
            close: &close,
        };
        assert!(!rates.all_off());
        let mut sink = OrderRecords::new();
        let (mut cash, mut free) = (100.0, 50.0);
        let pos = [2.0, -4.0];
        charge_held(
            &rates,
            3,
            0..2,
            |k| held(pos[k], 0.0, true),
            &mut cash,
            &mut free,
            &mut sink,
        )
        .unwrap();
        // The long pays 2 × 10 × 0.25, the short receives 4 × 10 × 0.25.
        assert_eq!(sink.cash.amount, [-5.0, 10.0]);
        assert_eq!(sink.cash.kind, [3, 3]);
        assert_eq!(sink.cash.idx, [3, 3]);
        assert_eq!((cash, free), (105.0, 55.0));
    }

    #[test]
    fn financing_takes_the_rate_of_the_positions_side_at_the_mark() {
        let long = FlexArray::Scalar(0.5);
        let short = FlexArray::Scalar(-0.25);
        let mark = FlexArray::Scalar(4.0);
        let close = FlexArray::Scalar(10.0);
        let rates = Rates {
            funding: &ZERO,
            long_financing: &long,
            short_financing: &short,
            debt: &ZERO,
            borrow: &ZERO,
            mark_price: &mark,
            close: &close,
        };
        let mut sink = OrderRecords::new();
        let (mut cash, mut free) = (0.0, 0.0);
        let pos = [2.0, -2.0];
        charge_held(
            &rates,
            0,
            0..2,
            |k| held(pos[k], 0.0, false),
            &mut cash,
            &mut free,
            &mut sink,
        )
        .unwrap();
        // 2 × 4 × 0.5 paid; a negative short rate is received.
        assert_eq!(sink.cash.amount, [-4.0, 2.0]);
        assert_eq!(sink.cash.kind, [4, 4]);
        // Uncapped: the account had nothing and owes the difference.
        assert_eq!(cash, -2.0);
    }

    #[test]
    fn interest_is_on_a_cash_longs_debt_and_the_borrow_fee_on_a_cash_shorts_value() {
        let debt = FlexArray::Scalar(0.125);
        let borrow = FlexArray::Scalar(0.5);
        let rates = Rates {
            funding: &ZERO,
            long_financing: &ZERO,
            short_financing: &ZERO,
            debt: &debt,
            borrow: &borrow,
            mark_price: &NAN,
            close: &NAN,
        };
        let mut sink = OrderRecords::new();
        let (mut cash, mut free) = (100.0, 100.0);
        // A cash long owing 40, a cash short, and a margin column on each
        // side, which borrows neither.
        let cols = [
            held(3.0, 40.0, false),
            held(-1.0, 8.0, false),
            held(3.0, 40.0, true),
            held(-1.0, 8.0, true),
        ];
        charge_held(
            &rates,
            0,
            0..4,
            |k| cols[k],
            &mut cash,
            &mut free,
            &mut sink,
        )
        .unwrap();
        // No mark and no close: the short is valued at its last valuation
        // price, 1 × 8 × 0.5.
        assert_eq!(sink.cash.amount, [-5.0, -4.0]);
        assert_eq!(sink.cash.kind, [5, 6]);
        assert_eq!(sink.cash.col, [0, 1]);
        assert_eq!((cash, free), (91.0, 91.0));
    }

    #[test]
    fn a_flat_column_an_unpriced_one_and_a_nan_rate_charge_nothing() {
        let funding = FlexArray::Scalar(f64::NAN);
        let long = FlexArray::Scalar(0.5);
        let rates = Rates {
            funding: &funding,
            long_financing: &long,
            short_financing: &ZERO,
            debt: &ZERO,
            borrow: &ZERO,
            mark_price: &NAN,
            close: &NAN,
        };
        let mut sink = OrderRecords::new();
        let (mut cash, mut free) = (1.0, 1.0);
        let cols = [
            held(0.0, 0.0, false),
            Held {
                position: 1.0,
                debt: 0.0,
                margin: true,
                val_price: f64::NAN,
                multiplier: 1.0,
            },
        ];
        charge_held(
            &rates,
            0,
            0..2,
            |k| cols[k],
            &mut cash,
            &mut free,
            &mut sink,
        )
        .unwrap();
        assert!(sink.cash.is_empty());
        assert_eq!((cash, free), (1.0, 1.0));
    }

    #[test]
    fn an_infinite_rate_is_an_error_and_zero_rates_are_off() {
        let inf = FlexArray::Scalar(f64::INFINITY);
        let close = FlexArray::Scalar(1.0);
        let rates = Rates {
            funding: &inf,
            long_financing: &ZERO,
            short_financing: &ZERO,
            debt: &ZERO,
            borrow: &ZERO,
            mark_price: &NAN,
            close: &close,
        };
        let mut sink = OrderRecords::new();
        let (mut cash, mut free) = (1.0, 1.0);
        let err = charge_held(
            &rates,
            0,
            0..1,
            |_| held(1.0, 0.0, true),
            &mut cash,
            &mut free,
            &mut sink,
        );
        assert!(err.is_err());
        let off = Rates {
            funding: &NAN,
            ..rates
        };
        assert!(off.all_off());
    }

    #[test]
    fn an_isolated_column_pays_out_of_its_own_margin() {
        let rate = FlexArray::Scalar(0.25);
        let close = FlexArray::Scalar(10.0);
        let rates = Rates {
            funding: &rate,
            long_financing: &ZERO,
            short_financing: &ZERO,
            debt: &ZERO,
            borrow: &ZERO,
            mark_price: &NAN,
            close: &close,
        };
        let mut sink = OrderRecords::new();
        let mut states = vec![ColumnState {
            position: 2.0,
            debt: 20.0,
            locked_cash: 8.0,
            last_val_price: 10.0,
            ..ColumnState::new()
        }];
        let (mut cash, mut free) = (100.0, 92.0);
        let columns = Columns {
            states: &mut states,
            margin: &[true],
            isolated: &[true],
            mult: &[1.0],
            scaled_val_price: true,
        };
        charge(&rates, 0, 0..1, columns, &mut cash, &mut free, &mut sink).unwrap();
        // 2 × 10 × 0.25 out of the cash and the margin; the free cash is
        // untouched.
        assert_eq!((cash, free, states[0].locked_cash), (95.0, 92.0, 3.0));
        assert_eq!(sink.cash.amount, [-5.0]);
    }

    #[test]
    fn cash_below_zero_pays_interest_at_the_first_column_s_rate() {
        let debt = FlexArray::Scalar(0.1);
        let rates = Rates {
            funding: &ZERO,
            long_financing: &ZERO,
            short_financing: &ZERO,
            debt: &debt,
            borrow: &ZERO,
            mark_price: &NAN,
            close: &NAN,
        };
        let mut sink = OrderRecords::new();
        let (mut cash, mut free) = (-50.0, -60.0);
        charge_held(
            &rates,
            2,
            3..5,
            |_| Held {
                position: 0.0,
                debt: 0.0,
                margin: true,
                val_price: 1.0,
                multiplier: 1.0,
            },
            &mut cash,
            &mut free,
            &mut sink,
        )
        .unwrap();
        assert_eq!((cash, free), (-55.0, -65.0));
        assert_eq!(sink.cash.col, [3]);
        assert_eq!(sink.cash.kind, [CashFlowKind::Interest as i8]);
    }
}
