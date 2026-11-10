//! The lazy rebuild chain: records back to a value curve.
//!
//! A portfolio stores its order records, the close prices and the starting
//! cash. Everything else is derived on demand through this chain:
//!
//! ```text
//! records ──▶ asset_flow ──▶ assets ─┐
//!         └─▶ cash_flow  ──▶ cash  ──┴─▶ value ──▶ returns
//!                                          ▲
//!                                     close prices
//! ```
//!
//! Storing the chain instead would mean holding six full matrices per sweep
//! when most columns are only ever asked for one scalar. Rebuilding is cheap
//! — one pass over the records plus one over the bars — and the arrays are
//! the largest thing in the system.
//!
//! Every function here takes records in the engine's ordering: grouped by
//! column, ascending in row within a column.

use alphax_core::error::Result;
use alphax_core::flex::FlexArray;
use alphax_core::math::{add, is_close};
use alphax_core::matrix::Matrix;
use alphax_model::{CashFlowKind, Direction, OrderSide};
use alphax_records::{ColMap, OrderRecords};

/// The signed size a fill adds to a position.
fn signed_size(size: f64, side: i8) -> f64 {
    if side == OrderSide::Sell as i8 {
        -size
    } else {
        size
    }
}

/// The long part of a position change.
///
/// A reversal is not one move but two: the old side closing and the new one
/// opening. These two functions split it, so that "how much long exposure
/// changed" has an answer even on a bar that crossed through zero.
fn long_size(before: f64, now: f64) -> f64 {
    if before <= 0.0 && now <= 0.0 {
        return 0.0;
    }
    if before >= 0.0 && now < 0.0 {
        return -before;
    }
    if before < 0.0 && now >= 0.0 {
        return now;
    }
    add(now, -before)
}

/// The short part of a position change.
fn short_size(before: f64, now: f64) -> f64 {
    if before >= 0.0 && now >= 0.0 {
        return 0.0;
    }
    if before >= 0.0 && now < 0.0 {
        return -now;
    }
    if before < 0.0 && now >= 0.0 {
        return before;
    }
    add(before, -now)
}

/// Assets transacted on each bar, per column.
///
/// `direction` picks which side of a reversal is counted;
/// [`Direction::Both`] counts the net move.
#[must_use]
pub fn asset_flow(
    orders: &OrderRecords,
    n_rows: usize,
    n_cols: usize,
    direction: Direction,
) -> Matrix {
    let mut out = Matrix::zeros(n_rows, n_cols);
    let mut position = 0.0_f64;
    let mut current_col = usize::MAX;

    for i in 0..orders.len() {
        let col = orders.col[i] as usize;
        if col != current_col {
            current_col = col;
            position = 0.0;
        }
        let size = signed_size(orders.size[i], orders.side[i]);
        let new_position = add(position, size);
        let flow = match direction {
            Direction::LongOnly => long_size(position, new_position),
            Direction::ShortOnly => short_size(position, new_position),
            Direction::Both => size,
        };
        let row = orders.idx[i] as usize;
        let cell = &mut out.column_mut(col)[row];
        // Several fills can land on one bar, so this accumulates rather than
        // assigns — and does so tolerantly, so a flat round trip within a bar
        // leaves exactly zero.
        *cell = add(*cell, flow);
        position = new_position;
    }
    out
}

/// Position held at the end of each bar, per column.
#[must_use]
pub fn assets(orders: &OrderRecords, n_rows: usize, n_cols: usize) -> Matrix {
    let flow = asset_flow(orders, n_rows, n_cols, Direction::Both);
    let mut out = Matrix::zeros(n_rows, n_cols);
    for col in 0..n_cols {
        let src = flow.column(col);
        let dst = out.column_mut(col);
        let mut running = 0.0;
        for row in 0..n_rows {
            running = add(running, src[row]);
            dst[row] = running;
        }
    }
    out
}

/// Each column's contract multiplier, read off its fills, or `None` when
/// every column's is 1.
///
/// Only a run that keeps balances can have one other than 1, and it is
/// constant down a column, so a column's first fill gives it; a column that
/// never filled holds nothing to multiply.
#[must_use]
pub fn multipliers(orders: &OrderRecords, n_cols: usize) -> Option<Vec<f64>> {
    if orders
        .multiplier
        .iter()
        .all(|m| m.to_bits() == 1.0_f64.to_bits())
    {
        return None;
    }
    let mut out = vec![1.0; n_cols];
    for (r, &m) in orders.multiplier.iter().enumerate() {
        out[orders.col[r] as usize] = m;
    }
    Some(out)
}

/// Cash moved on each bar, per column: `-size * price * multiplier - fees`
/// for every fill, plus every cash flow that was not a fill.
#[must_use]
pub fn cash_flow(orders: &OrderRecords, n_rows: usize, n_cols: usize) -> Matrix {
    let mut out = Matrix::zeros(n_rows, n_cols);
    for i in 0..orders.len() {
        let col = orders.col[i] as usize;
        let size = signed_size(orders.size[i], orders.side[i]);
        let flow = -size * (orders.price[i] * orders.multiplier_of(i)) - orders.fees[i];
        let row = orders.idx[i] as usize;
        let cell = &mut out.column_mut(col)[row];
        *cell = add(*cell, flow);
    }
    add_cash_records(&mut out, orders);
    out
}

/// Adds the non-fill cash flows to a per-column flow matrix.
///
/// They move cash and free cash alike — none of them reserves or releases
/// margin — so both flow series take them unchanged.
fn add_cash_records(out: &mut Matrix, orders: &OrderRecords) {
    let cash = &orders.cash;
    for i in 0..cash.len() {
        let cell = &mut out.column_mut(cash.col[i] as usize)[cash.idx[i] as usize];
        *cell = add(*cell, cash.amount[i]);
    }
}

/// Deposits and withdrawals per bar and column, or `None` if there were none.
///
/// The one kind of cash flow that is not a return: money brought into the
/// account changes its value without the account having earned it.
#[must_use]
pub fn deposits(orders: &OrderRecords, n_rows: usize, n_cols: usize) -> Option<Matrix> {
    let cash = &orders.cash;
    let mut out: Option<Matrix> = None;
    for i in 0..cash.len() {
        if !cash.is_deposit(i) {
            continue;
        }
        let matrix = out.get_or_insert_with(|| Matrix::zeros(n_rows, n_cols));
        let cell = &mut matrix.column_mut(cash.col[i] as usize)[cash.idx[i] as usize];
        *cell = add(*cell, cash.amount[i]);
    }
    out
}

/// The cash flows of one kind per bar and column: what was funded, financed,
/// charged in interest, paid out in dividends.
///
/// Zero where nothing of that kind moved. Per column, as the records book
/// them, so a shared account's funding stays attributed to the symbol that
/// paid it; summing a group's columns gives the account's.
#[must_use]
pub fn cash_flow_by_kind(
    orders: &OrderRecords,
    kind: CashFlowKind,
    n_rows: usize,
    n_cols: usize,
) -> Matrix {
    let cash = &orders.cash;
    let mut out = Matrix::zeros(n_rows, n_cols);
    for i in 0..cash.len() {
        if cash.kind[i] == kind as i8 {
            let cell = &mut out.column_mut(cash.col[i] as usize)[cash.idx[i] as usize];
            *cell = add(*cell, cash.amount[i]);
        }
    }
    out
}

/// Cash moved on each bar net of the margin an open short reserves.
///
/// The difference from [`cash_flow`] is entirely about shorts. Opening one
/// brings in cash *and* commits twice the short's value as margin, so free
/// cash falls where plain cash rises; closing one releases it again. On a
/// long-only column the two series are identical, which is why the difference
/// is easy to miss and why the fixtures that caught it are the shorting ones.
///
/// This replays the kernel's `debt` bookkeeping from the records rather than
/// carrying it out of the simulation. That is possible because `debt` is a
/// function of the fill sequence alone — the same reason the whole rebuild
/// chain works — and it keeps the portfolio's stored state down to records
/// plus prices.
///
/// A levered run is the exception: there a short ties up its proceeds plus
/// `value / leverage`, not twice its value, and the fills alone do not say what
/// leverage each one was taken at. Such a run records each fill's `debt` and
/// `locked_cash`, and the reserve is read from them instead. A levered long
/// reserves nothing, but its borrowing is already out of [`cash_flow`], which
/// charges the whole price: free cash is then own cash less what was borrowed,
/// negative when more was borrowed than is held.
///
/// A margin column reserves its initial margin, read from `locked_cash` the
/// same way. Its fills are charged their whole price here too, so what this
/// leaves out is the position's market value: the free cash of a margin
/// column is this flow's running sum plus `position × close`, which
/// [`Portfolio::free_cash`](crate::Portfolio::free_cash) adds.
///
/// # Errors
///
/// Returns [`alphax_core::Error::Invariant`] if the records are not grouped by
/// column.
pub fn free_cash_flow(orders: &OrderRecords, n_rows: usize, n_cols: usize) -> Result<Matrix> {
    let map = ColMap::build(&orders.col, n_cols)?;
    let mut out = Matrix::zeros(n_rows, n_cols);
    if !orders.debt.is_empty() {
        let charges = isolated_charges(orders, n_cols);
        levered_free_cash_flow(orders, &map, &charges, &mut out);
        let cash = &orders.cash;
        for i in 0..cash.len() {
            let col = cash.col[i] as usize;
            if charges[col].is_some() && is_holding_charge(cash.kind[i]) {
                // Out of the column's own margin, not the free cash.
                continue;
            }
            let cell = &mut out.column_mut(col)[cash.idx[i] as usize];
            *cell = add(*cell, cash.amount[i]);
        }
        return Ok(out);
    }
    for col in 0..n_cols {
        let mut position = 0.0_f64;
        let mut debt = 0.0_f64;
        for &r in map.column(col) {
            let size = signed_size(orders.size[r], orders.side[r]);
            let price = orders.price[r];
            let new_position = add(position, size);
            let final_cash = -size * price - orders.fees[r];

            let diff = if is_close(size, 0.0) {
                0.0
            } else if size > 0.0 {
                if position < 0.0 {
                    // Buying back part or all of a short releases the margin
                    // it reserved: twice the debt retired, since the reserve
                    // covered the round trip.
                    let short_size = if new_position < 0.0 {
                        size.abs()
                    } else {
                        position.abs()
                    };
                    let avg_entry = debt / position.abs();
                    let debt_diff = short_size * avg_entry;
                    debt = add(debt, -debt_diff);
                    add(2.0 * debt_diff, final_cash)
                } else {
                    final_cash
                }
            } else if new_position < 0.0 {
                let short_size = if position < 0.0 {
                    size.abs()
                } else {
                    new_position.abs()
                };
                let short_value = short_size * price;
                debt += short_value;
                add(final_cash, -2.0 * short_value)
            } else {
                final_cash
            };

            let cell = &mut out.column_mut(col)[orders.idx[r] as usize];
            *cell = add(*cell, diff);
            position = new_position;
        }
    }
    add_cash_records(&mut out, orders);
    Ok(out)
}

/// Whether a cash record of `kind` is a holding charge a margin column pays.
fn is_holding_charge(kind: i8) -> bool {
    kind == CashFlowKind::Funding as i8 || kind == CashFlowKind::Financing as i8
}

/// Each isolated column's holding charges as `(row, amount)`, by row; `None`
/// for a column that is not isolated. An isolated column pays them out of
/// the margin it posted, so they move its reserve, not the free cash.
fn isolated_charges(orders: &OrderRecords, n_cols: usize) -> Vec<Option<Vec<(i64, f64)>>> {
    let mut out: Vec<Option<Vec<(i64, f64)>>> = vec![None; n_cols];
    for (r, &on) in orders.isolated.iter().enumerate() {
        if on {
            out[orders.col[r] as usize].get_or_insert_with(Vec::new);
        }
    }
    let cash = &orders.cash;
    for i in 0..cash.len() {
        if let Some(list) = out[cash.col[i] as usize].as_mut() {
            if is_holding_charge(cash.kind[i]) {
                list.push((cash.idx[i], cash.amount[i]));
            }
        }
    }
    for list in out.iter_mut().flatten() {
        list.sort_by_key(|&(row, _)| row);
    }
    out
}

/// [`free_cash_flow`] for a run that recorded its balances.
///
/// A short's reserve is the `debt` plus `locked_cash` its last fill left
/// behind, and a margin column's is its `locked_cash`, the initial margin, on
/// either side; anything else reserves nothing. Each fill moves free cash by
/// its cash flow less the change in that reserve.
fn levered_free_cash_flow(
    orders: &OrderRecords,
    map: &ColMap,
    charges: &[Option<Vec<(i64, f64)>>],
    out: &mut Matrix,
) {
    for (col, own) in charges.iter().enumerate().take(out.n_cols()) {
        let mut position = 0.0_f64;
        let mut reserve = 0.0_f64;
        let own = own.as_deref().unwrap_or(&[]);
        let mut next = 0;
        for &r in map.column(col) {
            // An isolated column's charges since its last fill came out of
            // its margin: the reserve the fill starts from is smaller.
            while next < own.len() && own[next].0 < orders.idx[r] {
                reserve += own[next].1;
                next += 1;
            }
            let size = signed_size(orders.size[r], orders.side[r]);
            position = add(position, size);
            // Records built without margin flags have no margin columns.
            let new_reserve = if orders.margin.get(r) == Some(&true) {
                orders.locked_cash[r]
            } else if position < 0.0 {
                orders.debt[r] + orders.locked_cash[r]
            } else {
                0.0
            };
            let final_cash = -size * (orders.price[r] * orders.multiplier_of(r)) - orders.fees[r];
            let diff = add(final_cash, -add(new_reserve, -reserve));
            reserve = new_reserve;
            let cell = &mut out.column_mut(col)[orders.idx[r] as usize];
            *cell = add(*cell, diff);
        }
    }
}

/// Gross exposure per bar: how much of the account is in the position.
///
/// `|asset value| / (|asset value| + free cash)`, which is **not** the same as
/// dividing by portfolio value. The two agree on a long-only column and part
/// company on a short, where free cash is held down by the margin the short
/// reserves and the ratio is correspondingly higher. Getting this wrong is
/// invisible until someone shorts.
///
/// A zero denominator gives `0.0` rather than `NaN`, as an
/// account with nothing in it has no exposure.
#[must_use]
pub fn gross_exposure(asset_value: &Matrix, free_cash: &Matrix) -> Matrix {
    let (n_rows, n_cols) = (asset_value.n_rows(), asset_value.n_cols());
    let mut out = Matrix::zeros(n_rows, n_cols);
    for col in 0..n_cols {
        for row in 0..n_rows {
            let abs_value = asset_value.get(row, col).abs();
            let denom = add(abs_value, free_cash.get(row, col));
            out.column_mut(col)[row] = if denom == 0.0 { 0.0 } else { abs_value / denom };
        }
    }
    out
}

/// Cash held at the end of each bar, per column.
#[must_use]
pub fn cash(flow: &Matrix, init_cash: &FlexArray<'_, f64>) -> Matrix {
    let (n_rows, n_cols) = (flow.n_rows(), flow.n_cols());
    let mut out = Matrix::zeros(n_rows, n_cols);
    for col in 0..n_cols {
        let src = flow.column(col);
        let dst = out.column_mut(col);
        let mut running = init_cash.get(0, col);
        for row in 0..n_rows {
            running = add(running, src[row]);
            dst[row] = running;
        }
    }
    out
}

/// Position value at each bar: close times assets, times each column's
/// contract multiplier where there are any (see [`multipliers`]).
#[must_use]
pub fn asset_value(
    assets: &Matrix,
    close: &FlexArray<'_, f64>,
    multiplier: Option<&[f64]>,
) -> Matrix {
    let (n_rows, n_cols) = (assets.n_rows(), assets.n_cols());
    let mut out = Matrix::zeros(n_rows, n_cols);
    for col in 0..n_cols {
        let src = assets.column(col);
        let dst = out.column_mut(col);
        if let Some(m) = multiplier.map(|m| m[col]) {
            for row in 0..n_rows {
                dst[row] = close.get(row, col) * m * src[row];
            }
        } else {
            for row in 0..n_rows {
                dst[row] = close.get(row, col) * src[row];
            }
        }
    }
    out
}

/// Portfolio value at each bar: cash plus position value.
#[must_use]
pub fn value(cash: &Matrix, asset_value: &Matrix) -> Matrix {
    let (n_rows, n_cols) = (cash.n_rows(), cash.n_cols());
    let mut out = Matrix::zeros(n_rows, n_cols);
    for col in 0..n_cols {
        let (a, b) = (cash.column(col), asset_value.column(col));
        let dst = out.column_mut(col);
        for row in 0..n_rows {
            dst[row] = a[row] + b[row];
        }
    }
    out
}

/// One period's return, from the value before to the value after.
///
/// Two cases are not the textbook formula, and both matter. A zero starting
/// value cannot produce a ratio, so the result is signed infinity — or zero
/// when nothing changed, rather than `NaN`. A *negative* starting value has
/// its return sign flipped, so that a short account moving further into
/// deficit reads as a loss instead of a gain.
#[must_use]
pub fn period_return(before: f64, after: f64) -> f64 {
    if before == 0.0 {
        if after == 0.0 {
            return 0.0;
        }
        return if after > 0.0 {
            f64::INFINITY
        } else if after < 0.0 {
            f64::NEG_INFINITY
        } else {
            f64::NAN
        };
    }
    let r = (after - before) / before;
    if before < 0.0 { -r } else { r }
}

/// Per-bar returns of the value curve, seeded from the starting cash.
///
/// The first bar's return is measured against `init_cash`, not against the
/// first value — otherwise the first bar's trading would be invisible.
#[must_use]
pub fn returns(value: &Matrix, init_cash: &FlexArray<'_, f64>) -> Matrix {
    returns_with_deposits(value, init_cash, None)
}

/// [`returns`], with deposits taken out.
///
/// A deposit is booked at the bar's open, so it is part of what the bar
/// started with: the bar's return is measured from the previous value *plus*
/// the deposit. Without that, a deposit reads as a gain and a withdrawal as a
/// loss. `deposits` has the value's shape; `None` is the plain chain, bit for
/// bit.
#[must_use]
pub fn returns_with_deposits(
    value: &Matrix,
    init_cash: &FlexArray<'_, f64>,
    deposits: Option<&Matrix>,
) -> Matrix {
    let (n_rows, n_cols) = (value.n_rows(), value.n_cols());
    let mut out = Matrix::zeros(n_rows, n_cols);
    for col in 0..n_cols {
        let src = value.column(col);
        let added = deposits.map(|d| d.column(col));
        let dst = out.column_mut(col);
        let mut before = init_cash.get(0, col);
        for row in 0..n_rows {
            if let Some(added) = added {
                if added[row] != 0.0 {
                    before = add(before, added[row]);
                }
            }
            dst[row] = period_return(before, src[row]);
            before = src[row];
        }
    }
    out
}

/// A value curve with deposits taken out: the starting cash compounded by the
/// returns.
///
/// What drawdowns are measured on when money was added or withdrawn, because
/// a withdrawal is not a drawdown and a deposit is not a recovery.
#[must_use]
pub fn compounded_value(returns: &Matrix, init_cash: &FlexArray<'_, f64>) -> Matrix {
    let (n_rows, n_cols) = (returns.n_rows(), returns.n_cols());
    let mut out = Matrix::zeros(n_rows, n_cols);
    for col in 0..n_cols {
        let src = returns.column(col);
        let dst = out.column_mut(col);
        let mut running = init_cash.get(0, col);
        for row in 0..n_rows {
            running *= 1.0 + src[row];
            dst[row] = running;
        }
    }
    out
}

/// The whole chain in one call: records and prices to a return series.
///
/// # Errors
///
/// Propagates nothing today; the signature is fallible so that the chain can
/// grow a shape check without breaking callers.
pub fn returns_from_orders(
    orders: &OrderRecords,
    close: &FlexArray<'_, f64>,
    init_cash: &FlexArray<'_, f64>,
    n_rows: usize,
    n_cols: usize,
) -> Result<Matrix> {
    let assets = assets(orders, n_rows, n_cols);
    let cash = cash(&cash_flow(orders, n_rows, n_cols), init_cash);
    let multiplier = multipliers(orders, n_cols);
    let value = value(&cash, &asset_value(&assets, close, multiplier.as_deref()));
    Ok(returns(&value, init_cash))
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "these assert on exact expected balances; an approximate \
              assertion would not catch the arithmetic errors they exist for"
)]
mod tests {
    use super::*;

    #[test]
    fn free_cash_reserves_a_shorts_margin_where_plain_cash_does_not() {
        // Opening a short brings in cash and commits twice its value as
        // margin, so the two series move in opposite directions on that bar.
        // On a long-only column they are identical, which is exactly why the
        // difference went unnoticed until a shorting fixture reached it.
        let mut o = OrderRecords::new();
        o.push_order(0, 0, 1.0, 10.0, 0.0, OrderSide::Sell);
        o.push_order(2, 0, 1.0, 8.0, 0.0, OrderSide::Buy);

        let plain = cash_flow(&o, 3, 1);
        let free = free_cash_flow(&o, 3, 1).unwrap();
        assert_eq!(plain.column(0)[0], 10.0, "the short brings cash in");
        assert_eq!(
            free.column(0)[0],
            -10.0,
            "and commits twice that as margin, netting the other way"
        );
        // Closing it releases the reserve.
        assert_eq!(plain.column(0)[2], -8.0);
        assert_eq!(free.column(0)[2], 12.0);

        // Long only: the two agree bar for bar.
        let mut long = OrderRecords::new();
        long.push_order(0, 0, 1.0, 10.0, 0.0, OrderSide::Buy);
        long.push_order(2, 0, 1.0, 12.0, 0.0, OrderSide::Sell);
        assert_eq!(
            cash_flow(&long, 3, 1).as_slice(),
            free_cash_flow(&long, 3, 1).unwrap().as_slice()
        );
    }

    #[test]
    fn recorded_balances_without_margin_flags_read_as_cash_columns() {
        // Records filled in by hand, as before margin columns existed, carry
        // `debt` and `locked_cash` and no flags.
        let mut o = OrderRecords::new();
        o.push_order(0, 0, 1.0, 10.0, 0.0, OrderSide::Sell);
        o.push_order(2, 0, 1.0, 8.0, 0.0, OrderSide::Buy);
        o.debt = vec![10.0, 0.0];
        o.locked_cash = vec![5.0, 0.0];
        let free = free_cash_flow(&o, 3, 1).unwrap();
        assert_eq!(free.column(0), &[-5.0, 0.0, 7.0]);
    }

    #[test]
    fn gross_exposure_divides_by_free_cash_not_by_value() {
        // Dividing by portfolio value is the obvious reading and it is wrong
        // for a short, where free cash is held down by the margin. Same
        // asset value, different denominator, different answer.
        let asset_value = Matrix::from_column_major(vec![-10.0], 1, 1).unwrap();
        let free = Matrix::from_column_major(vec![90.0], 1, 1).unwrap();
        let e = gross_exposure(&asset_value, &free);
        assert!((e.get(0, 0) - 0.1).abs() < 1e-12, "got {}", e.get(0, 0));
        // An empty account has no exposure rather than an undefined one.
        let zero = Matrix::zeros(1, 1);
        assert_eq!(gross_exposure(&zero, &zero).get(0, 0), 0.0);
    }
    use alphax_records::RecordSink;

    /// Buy 2 @ 10 on bar 0, sell 2 @ 15 on bar 2.
    fn round_trip() -> OrderRecords {
        let mut o = OrderRecords::new();
        o.push_order(0, 0, 2.0, 10.0, 0.0, OrderSide::Buy);
        o.push_order(2, 0, 2.0, 15.0, 0.0, OrderSide::Sell);
        o
    }

    #[test]
    fn the_chain_reproduces_a_round_trip() {
        let close = FlexArray::per_row(&[10.0, 12.0, 15.0], 3).unwrap();
        let init = FlexArray::scalar(100.0);
        let o = round_trip();

        assert_eq!(assets(&o, 3, 1).column(0), [2.0, 2.0, 0.0]);
        let c = cash(&cash_flow(&o, 3, 1), &init);
        assert_eq!(c.column(0), [80.0, 80.0, 110.0]);
        let v = value(&c, &asset_value(&assets(&o, 3, 1), &close, None));
        // 80 + 2*10, 80 + 2*12, 110 + 0.
        assert_eq!(v.column(0), [100.0, 104.0, 110.0]);
        let r = returns(&v, &init);
        // Spelled as the engine computes it. `110.0 / 104.0 - 1.0` is the same
        // number mathematically and one ULP away in floating point, which is
        // the kind of difference the golden fixtures are there to catch.
        assert_eq!(r.column(0), [0.0, 0.04, (110.0 - 104.0) / 104.0]);
    }

    #[test]
    fn fees_leave_the_value_curve_lower() {
        let mut o = OrderRecords::new();
        o.push_order(0, 0, 1.0, 10.0, 1.0, OrderSide::Buy);
        let close = FlexArray::scalar(10.0);
        let init = FlexArray::scalar(100.0);
        let v = value(
            &cash(&cash_flow(&o, 1, 1), &init),
            &asset_value(&assets(&o, 1, 1), &close, None),
        );
        // Paid 10 for the asset plus 1 in fees; the asset is still worth 10.
        assert_eq!(v.column(0), [99.0]);
    }

    #[test]
    fn several_fills_on_one_bar_accumulate() {
        let mut o = OrderRecords::new();
        o.push_order(0, 0, 1.0, 10.0, 0.0, OrderSide::Buy);
        o.push_order(0, 0, 3.0, 10.0, 0.0, OrderSide::Buy);
        assert_eq!(assets(&o, 2, 1).column(0), [4.0, 4.0]);
        assert_eq!(cash_flow(&o, 2, 1).column(0), [-40.0, 0.0]);
    }

    #[test]
    fn a_flat_round_trip_within_one_bar_leaves_exactly_zero() {
        // Tolerant accumulation, so the position does not end on a residue
        // that would keep a closed position nominally open.
        let mut o = OrderRecords::new();
        o.push_order(0, 0, 0.1, 10.0, 0.0, OrderSide::Buy);
        o.push_order(0, 0, 0.2, 10.0, 0.0, OrderSide::Buy);
        o.push_order(0, 0, 0.3, 10.0, 0.0, OrderSide::Sell);
        assert_eq!(asset_flow(&o, 1, 1, Direction::Both).column(0), [0.0]);
    }

    #[test]
    fn columns_are_independent() {
        let mut o = OrderRecords::new();
        o.push_order(0, 0, 2.0, 10.0, 0.0, OrderSide::Buy);
        o.push_order(0, 1, 5.0, 10.0, 0.0, OrderSide::Buy);
        let a = assets(&o, 2, 2);
        assert_eq!(a.column(0), [2.0, 2.0]);
        assert_eq!(a.column(1), [5.0, 5.0]);
    }

    #[test]
    fn a_reversal_splits_into_a_long_and_a_short_leg() {
        // Long 1, then sell 3: one unit of long closes, two of short open.
        let mut o = OrderRecords::new();
        o.push_order(0, 0, 1.0, 10.0, 0.0, OrderSide::Buy);
        o.push_order(1, 0, 3.0, 10.0, 0.0, OrderSide::Sell);
        assert_eq!(asset_flow(&o, 2, 1, Direction::Both).column(0), [1.0, -3.0]);
        assert_eq!(
            asset_flow(&o, 2, 1, Direction::LongOnly).column(0),
            [1.0, -1.0]
        );
        assert_eq!(
            asset_flow(&o, 2, 1, Direction::ShortOnly).column(0),
            [0.0, 2.0]
        );
    }

    #[test]
    fn the_position_resets_between_columns() {
        // Only the directional flows read the running position, and only they
        // can see this: with `Both` the flow is the signed size regardless, so
        // a position leaking across a column boundary is invisible. Column 0
        // ends short; if that carried into column 1, its opening buy would be
        // classified as covering rather than as opening a long.
        let mut o = OrderRecords::new();
        o.push_order(0, 0, 5.0, 10.0, 0.0, OrderSide::Sell);
        o.push_order(0, 1, 5.0, 10.0, 0.0, OrderSide::Buy);
        let long = asset_flow(&o, 1, 2, Direction::LongOnly);
        assert_eq!(long.column(0), [0.0], "a short opening is not long flow");
        assert_eq!(long.column(1), [5.0], "column 1 opens a fresh long");
        let short = asset_flow(&o, 1, 2, Direction::ShortOnly);
        assert_eq!(short.column(0), [5.0]);
        assert_eq!(short.column(1), [0.0]);
    }

    #[test]
    fn a_return_from_zero_is_a_signed_infinity() {
        assert_eq!(period_return(0.0, 0.0), 0.0);
        assert_eq!(period_return(0.0, 5.0), f64::INFINITY);
        assert_eq!(period_return(0.0, -5.0), f64::NEG_INFINITY);
    }

    #[test]
    fn a_return_from_a_negative_value_flips_sign() {
        // Going from -100 to -110 is a further loss, not a 10% gain.
        assert_eq!(period_return(-100.0, -110.0), -0.1);
        assert_eq!(period_return(-100.0, -90.0), 0.1);
    }
}
