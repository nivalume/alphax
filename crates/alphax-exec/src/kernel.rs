//! The order execution kernel.
//!
//! Bit-identical fills against the recorded golden outputs are the acceptance
//! criterion, so where a rewrite would be tidier but round differently, the
//! arithmetic is kept as it is. Places where that produces un-idiomatic Rust
//! are commented rather than tidied up.

use alphax_core::error::{Error, Result};
use alphax_core::math::{add, is_close, is_close_or_less, is_less};
use alphax_model::{
    AccountType, Direction, LeverageMode, MarginBasis, OrderSide, OrderStatus, SizeType, StatusInfo,
};

use crate::types::{ExecuteOrderState, Order, OrderResult, ProcessOrderState};

/// Outcome of a buy/sell leg: either the new state and a fill, or a
/// non-filling result that leaves the state untouched.
type Leg = (ExecuteOrderState, OrderResult);

/// Three-way sign with `numpy` semantics: both zeros and `NaN` give `0.0`.
///
/// Used only to turn a percentage into a signed infinity. `NaN` mapping to
/// `0.0` is not a deviation: `0.0 * inf` is `NaN`, which is what
/// `sign(NaN) * inf` produces too.
#[inline]
fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Floor to a multiple of `granularity`, matching Python's `a // g * g`.
#[inline]
fn floor_to(value: f64, granularity: f64) -> f64 {
    (value / granularity).floor() * granularity
}

/// The cash a `lock_cash` buy may spend, before the caps every buy applies.
///
/// Out of line, like [`sell_size_limit`]: most runs never lock cash, and
/// [`buy`] is inlined into the kernel, so this keeps the copy small.
#[cold]
#[inline(never)]
fn locked_cash_limit(state: &ExecuteOrderState, adj_price: f64, order: &Order) -> f64 {
    if state.position >= 0.0 {
        // In a long position cash == free_cash, unless another column has
        // locked some of it.
        return state.free_cash;
    }
    // Closing the short gives back what it ties up: its proceeds and its
    // collateral. Without leverage the two are equal, and their sum is
    // `2 * debt` to the last bit.
    let reserved = state.debt + state.locked_cash;
    // Covering the short costs this much; what is left over is what a new long
    // may use.
    let cover_req_cash = state.position.abs() * adj_price * (1.0 + order.fees) + order.fixed_fees;
    let cover_free_cash = add(state.free_cash + reserved, -cover_req_cash);
    if cover_free_cash > 0.0 {
        // Enough to close the short and open a long.
        state.free_cash + reserved
    } else if cover_free_cash < 0.0 {
        // Not enough to close the short: solve for the part that can be
        // covered. The two per-unit terms are the reservation released by
        // shrinking the short: `2 * avg_entry_price`.
        let avg_entry_price = state.debt / state.position.abs();
        let avg_locked = state.locked_cash / state.position.abs();
        let max_short_size = (state.free_cash - order.fixed_fees)
            / (adj_price * (1.0 + order.fees) - (avg_entry_price + avg_locked));
        max_short_size * adj_price * (1.0 + order.fees) + order.fixed_fees
    } else {
        // Exactly enough to close the short.
        state.cash
    }
}

/// Buys, or covers an existing short.
///
/// Comparisons against zero here are exact, not tolerance-based, `is_close` at these sites changes which orders fill.
///
/// `size` is positive. `percent`, when not `NaN`, caps the spend at that
/// fraction of the available cash.
///
/// Inlining is forced here and in [`sell`]: left to itself, the compiler
/// stopped inlining either once the kernel grew, and the calls cost
/// `from_orders` about 2.5% more instructions (`benchmarks/instructions.py`).
#[inline(always)]
fn buy(
    state: ExecuteOrderState,
    size: f64,
    price: f64,
    order: &Order,
    percent: f64,
) -> Result<Leg> {
    let adj_price = price * (1.0 + order.slippage);

    // How much cash this order may spend.
    let mut cash_limit = if order.lock_cash {
        locked_cash_limit(&state, adj_price, order)
    } else {
        state.cash
    };
    cash_limit = cash_limit.min(state.cash).min(order.cash_limit);
    if !percent.is_nan() {
        cash_limit = cash_limit.min(percent * cash_limit);
    }

    let levered = order.leverage > 1.0;
    if matches!(order.direction, Direction::LongOnly | Direction::Both) {
        // Below zero only in a leveraged account, where a long sold for less
        // than its debt left the account owing; the limit is then below zero
        // too. Without leverage that is an invalid state, which validation
        // stops first, and this is the `cash_limit == 0` case.
        if cash_limit <= 0.0 {
            let info = if levered {
                StatusInfo::NoCashLeverage
            } else {
                StatusInfo::NoCashLong
            };
            return Ok((state, OrderResult::not_filled(OrderStatus::Rejected, info)));
        }
        if size.is_infinite() && cash_limit.is_infinite() {
            return Err(Error::Invariant(
                "attempt to go long infinitely: both size and available cash are infinite".into(),
            ));
        }
    } else if state.position == 0.0 {
        return Ok((
            state,
            OrderResult::not_filled(OrderStatus::Rejected, StatusInfo::NoOpenPosition),
        ));
    }

    // In a short-only column a buy can only cover, never go long.
    let mut adj_size = if order.direction == Direction::ShortOnly {
        (-state.position).min(size)
    } else {
        size
    };

    if adj_size == 0.0 {
        return Ok((
            state,
            OrderResult::not_filled(OrderStatus::Ignored, StatusInfo::SizeZero),
        ));
    }

    if adj_size > order.max_size {
        if !order.allow_partial {
            return Ok((
                state,
                OrderResult::not_filled(OrderStatus::Rejected, StatusInfo::MaxSizeExceeded),
            ));
        }
        adj_size = order.max_size;
    }

    if !order.size_granularity.is_nan() {
        adj_size = floor_to(adj_size, order.size_granularity);
    }

    let req_cash = adj_size * adj_price;
    let req_fees = req_cash * order.fees + order.fixed_fees;
    let total_req_cash = req_cash + req_fees;

    // `final_req_cash` is the own cash the fill spends and `borrowed` the rest
    // of what it costs, which goes into debt.
    let (final_size, fees_paid, final_req_cash, borrowed) = if levered {
        match levered_buy_cost(&state, adj_size, adj_price, cash_limit, order) {
            Ok(cost) => cost,
            Err(info) => return Ok((state, OrderResult::not_filled(OrderStatus::Rejected, info))),
        }
    } else if is_close_or_less(total_req_cash, cash_limit) {
        (adj_size, req_fees, total_req_cash, 0.0)
    } else {
        // Not enough cash. Fees are inside the cap, not added on top, so the
        // affordable notional has to be back-solved: at 10% fees and $1 fixed,
        // $100 of cash buys $90 of stock.
        let max_req_cash = add(cash_limit, -order.fixed_fees) / (1.0 + order.fees);
        if max_req_cash <= 0.0 {
            return Ok((
                state,
                OrderResult::not_filled(OrderStatus::Rejected, StatusInfo::CantCoverFees),
            ));
        }
        let max_acq_size = max_req_cash / adj_price;

        if order.size_granularity.is_nan() {
            (max_acq_size, cash_limit - max_req_cash, cash_limit, 0.0)
        } else {
            // Granularity is applied a second time here, after the cash
            // constraint, so lot discipline survives a partial fill.
            let final_size = floor_to(max_acq_size, order.size_granularity);
            let new_req_cash = final_size * adj_price;
            let fees_paid = new_req_cash * order.fees + order.fixed_fees;
            (final_size, fees_paid, new_req_cash + fees_paid, 0.0)
        }
    };

    // Note this tests `adj_size`, not `final_size`.
    if is_close(adj_size, 0.0) {
        return Ok((
            state,
            OrderResult::not_filled(OrderStatus::Ignored, StatusInfo::SizeZero),
        ));
    }
    if is_less(final_size, order.min_size) {
        return Ok((
            state,
            OrderResult::not_filled(OrderStatus::Rejected, StatusInfo::MinSizeNotReached),
        ));
    }
    // An infinite request cannot be "partial", by definition.
    if size.is_finite() && is_less(final_size, size) && !order.allow_partial {
        return Ok((
            state,
            OrderResult::not_filled(OrderStatus::Rejected, StatusInfo::PartialFill),
        ));
    }

    let new_cash = add(state.cash, -final_req_cash);
    let new_position = add(state.position, final_size);

    // The part of the fill that ends up long ties up the own cash paid for it:
    // its cost less what was borrowed.
    let long_size = if state.position < 0.0 {
        new_position.max(0.0)
    } else {
        final_size
    };
    let own_long = long_size * adj_price - borrowed;

    let (new_debt, new_locked_cash, new_free_cash) = if state.position < 0.0 {
        // Covering releases the reservation proportionally to the part closed:
        // the short's proceeds and its collateral, which without leverage are
        // equal and sum to `2 * debt_diff`.
        let short_size = if new_position < 0.0 {
            final_size
        } else {
            state.position.abs()
        };
        let avg_entry_price = state.debt / state.position.abs();
        let debt_diff = short_size * avg_entry_price;
        let locked_diff = short_size * (state.locked_cash / state.position.abs());
        (
            add(state.debt, -debt_diff) + borrowed,
            add(state.locked_cash, -locked_diff) + own_long,
            add(state.free_cash + (debt_diff + locked_diff), -final_req_cash),
        )
    } else {
        (
            state.debt + borrowed,
            state.locked_cash + own_long,
            add(state.free_cash, -final_req_cash),
        )
    };

    Ok((
        ExecuteOrderState {
            cash: new_cash,
            position: new_position,
            debt: new_debt,
            locked_cash: new_locked_cash,
            free_cash: new_free_cash,
        },
        OrderResult::filled(final_size, adj_price, fees_paid, OrderSide::Buy),
    ))
}

/// What a leveraged buy fills and what it costs: `(size, fees, own cash
/// spent, borrowed)`. `Err` is the reason it is rejected.
///
/// Own cash pays for any short being bought back at full price, for `1 /
/// leverage` of what goes long, and for every fee; the rest of the long part
/// is borrowed. `size` has already been capped and rounded as any buy is, and
/// `cash_limit` is the own cash the buy may spend.
///
/// [`LeverageMode::Eager`] spends exactly that minimum. [`LeverageMode::Lazy`]
/// spends own cash first, up to `cash_limit`, and borrows only what is left —
/// which is nothing when the whole cost fits.
///
/// Out of line so that [`buy`], which is inlined into the kernel, carries no
/// copy of it when nothing is leveraged.
#[inline(never)]
fn levered_buy_cost(
    state: &ExecuteOrderState,
    size: f64,
    adj_price: f64,
    cash_limit: f64,
    order: &Order,
) -> core::result::Result<(f64, f64, f64, f64), StatusInfo> {
    let leverage = order.leverage;
    // How much of a buy closes a short rather than going long.
    let cover_room = (-state.position).max(0.0);
    let own_needed = |q: f64| {
        let cover = q.min(cover_room);
        let notional = q * adj_price;
        cover * adj_price
            + (q - cover) * adj_price / leverage
            + (notional * order.fees + order.fixed_fees)
    };

    let mut final_size = size;
    if !is_close_or_less(own_needed(size), cash_limit) {
        // Not enough own cash: back-solve the largest size it carries.
        let cover_cost = cover_room * adj_price * (1.0 + order.fees) + order.fixed_fees;
        let max_size = if cover_room > 0.0 && cash_limit < cover_cost {
            // Not even the short can be closed, and closing is never leveraged.
            add(cash_limit, -order.fixed_fees) / (1.0 + order.fees) / adj_price
        } else {
            // The cover at full price, the rest at `1 / leverage`.
            add(
                cash_limit,
                -(order.fixed_fees + cover_room * adj_price * (1.0 - 1.0 / leverage)),
            ) / (1.0 / leverage + order.fees)
                / adj_price
        };
        if max_size <= 0.0 {
            return Err(StatusInfo::CantCoverFees);
        }
        final_size = if order.size_granularity.is_nan() {
            max_size
        } else {
            floor_to(max_size, order.size_granularity)
        };
    }

    let notional = final_size * adj_price;
    let fees = notional * order.fees + order.fixed_fees;
    let total = notional + fees;
    let own_min = own_needed(final_size);
    let own = match order.leverage_mode {
        LeverageMode::Eager => own_min,
        LeverageMode::Lazy if is_close_or_less(total, cash_limit) => total,
        LeverageMode::Lazy => cash_limit.max(own_min),
    };
    Ok((final_size, fees, own, add(total, -own).max(0.0)))
}

/// The most a `sell` may sell and short when cash is locked or the request is
/// an infinite percentage, and the part of `percent` still to apply after it.
/// `Err` is the reason the order is rejected.
///
/// Out of line because most runs reach neither case, as [`locked_cash_limit`].
#[cold]
#[inline(never)]
fn sell_size_limit(
    state: &ExecuteOrderState,
    size: f64,
    adj_price: f64,
    order: &Order,
    percent: f64,
) -> core::result::Result<(f64, f64), StatusInfo> {
    // Work out the largest position that can be sold and shorted. Selling a
    // long frees its proceeds less the debt they repay, which is zero without
    // leverage.
    let long_size = state.position.max(0.0);
    let long_debt = if state.position > 0.0 {
        state.debt
    } else {
        0.0
    };
    let long_cash = long_size * adj_price * (1.0 - order.fees) - long_debt;
    let total_free_cash = add(state.free_cash, long_cash);

    let max_size_limit = if total_free_cash <= 0.0 {
        if state.position <= 0.0 {
            // Nothing to sell and no free cash to short with.
            return Err(StatusInfo::NoCashShort);
        }
        // A position to close, but no free cash to short beyond it.
        long_size
    } else {
        // A short ties up `1 / leverage` of its value besides its proceeds,
        // and pays its fees from free cash: `1 + fees` per unit of value at
        // leverage 1.
        let max_short_size = add(total_free_cash, -order.fixed_fees)
            / (adj_price * (1.0 / order.leverage + order.fees));
        let limit = add(long_size, max_short_size);
        if limit <= 0.0 {
            return Err(StatusInfo::CantCoverFees);
        }
        limit
    };

    Ok(if order.lock_cash {
        // The percentage is consumed here rather than applied again in `sell`.
        if size.is_infinite() && !percent.is_nan() {
            ((percent * max_size_limit).min(max_size_limit), f64::NAN)
        } else if !percent.is_nan() {
            ((percent * size).min(max_size_limit), f64::NAN)
        } else {
            (size.min(max_size_limit), percent)
        }
    } else {
        // Reached only when size is infinite and a percentage is set.
        (max_size_limit, percent)
    })
}

/// What a leveraged long being sold repays: `(cash, debt, free_cash)` after
/// repaying its debt in proportion to the part sold, or all of it when the
/// sale closes the long.
///
/// Out of line because only leveraged runs have a long with debt, as
/// [`levered_buy_cost`].
#[cold]
#[inline(never)]
fn repay_long(state: &ExecuteOrderState, size: f64, new_position: f64) -> (f64, f64, f64) {
    let repaid = if new_position > 0.0 {
        size * (state.debt / state.position)
    } else {
        state.debt
    };
    (
        add(state.cash, -repaid),
        add(state.debt, -repaid),
        add(state.free_cash, -repaid),
    )
}

/// Sells, or opens/extends a short.
///
/// As in [`buy`], comparisons against zero are exact by design.
///
/// `size` is positive.
#[inline(always)]
fn sell(
    state: ExecuteOrderState,
    size: f64,
    price: f64,
    order: &Order,
    mut percent: f64,
) -> Result<Leg> {
    let adj_price = price * (1.0 - order.slippage);

    let mut size_limit = if order.direction == Direction::LongOnly {
        state.position.min(size)
    } else if order.lock_cash || (size.is_infinite() && !percent.is_nan()) {
        match sell_size_limit(&state, size, adj_price, order, percent) {
            Ok((limit, rest)) => {
                percent = rest;
                limit
            }
            Err(info) => {
                return Ok((state, OrderResult::not_filled(OrderStatus::Rejected, info)));
            }
        }
    } else {
        size
    };

    if !percent.is_nan() {
        size_limit *= percent;
    }

    if size_limit > order.max_size {
        if !order.allow_partial {
            return Ok((
                state,
                OrderResult::not_filled(OrderStatus::Rejected, StatusInfo::MaxSizeExceeded),
            ));
        }
        size_limit = order.max_size;
    }

    if matches!(order.direction, Direction::ShortOnly | Direction::Both) {
        if size_limit.is_infinite() {
            return Err(Error::Invariant(
                "attempt to go short infinitely: the sellable size is unbounded".into(),
            ));
        }
    } else if state.position == 0.0 {
        return Ok((
            state,
            OrderResult::not_filled(OrderStatus::Rejected, StatusInfo::NoOpenPosition),
        ));
    }

    if !order.size_granularity.is_nan() {
        size_limit = floor_to(size_limit, order.size_granularity);
    }

    if is_close(size_limit, 0.0) {
        return Ok((
            state,
            OrderResult::not_filled(OrderStatus::Ignored, StatusInfo::SizeZero),
        ));
    }
    if is_less(size_limit, order.min_size) {
        return Ok((
            state,
            OrderResult::not_filled(OrderStatus::Rejected, StatusInfo::MinSizeNotReached),
        ));
    }
    if size.is_finite() && is_less(size_limit, size) && !order.allow_partial {
        return Ok((
            state,
            OrderResult::not_filled(OrderStatus::Rejected, StatusInfo::PartialFill),
        ));
    }

    let acq_cash = size_limit * adj_price;
    let fees_paid = acq_cash * order.fees + order.fixed_fees;
    let final_acq_cash = add(acq_cash, -fees_paid);
    if final_acq_cash < 0.0 {
        return Ok((
            state,
            OrderResult::not_filled(OrderStatus::Rejected, StatusInfo::CantCoverFees),
        ));
    }

    let new_position = add(state.position, -size_limit);

    // Selling a long frees its share of the own cash it tied up, and repays
    // its share of what was borrowed for it. Without leverage a long has no
    // debt, and the other balances it starts from are the state's own.
    let (cash, debt, locked_cash, free_cash) = if state.position > 0.0 {
        // Plain arithmetic, not `add`: closing the whole long frees it all,
        // and a partial sale leaves a share well clear of zero.
        let locked_cash = if new_position > 0.0 {
            state.locked_cash - size_limit * (state.locked_cash / state.position)
        } else {
            0.0
        };
        if state.debt > 0.0 {
            let (cash, debt, free_cash) = repay_long(&state, size_limit, new_position);
            (cash, debt, locked_cash, free_cash)
        } else {
            (state.cash, state.debt, locked_cash, state.free_cash)
        }
    } else {
        (state.cash, state.debt, state.locked_cash, state.free_cash)
    };

    let new_cash = cash + final_acq_cash;
    let (new_debt, new_locked_cash, new_free_cash) = if new_position < 0.0 {
        // Only the part that ends up short creates debt and ties up margin.
        let short_size = if state.position < 0.0 {
            size_limit
        } else {
            new_position.abs()
        };
        let short_value = short_size * adj_price;
        // The proceeds are reserved, and so is the collateral: the short's
        // value over the leverage. At leverage 1 the two are the
        // doubled short value, the round-trip reservation, to the last bit.
        let collateral = short_value / order.leverage;
        let free_cash_diff = add(final_acq_cash, -(short_value + collateral));
        (
            debt + short_value,
            locked_cash + collateral,
            add(free_cash, free_cash_diff),
        )
    } else {
        (debt, locked_cash, free_cash + final_acq_cash)
    };

    Ok((
        ExecuteOrderState {
            cash: new_cash,
            position: new_position,
            debt: new_debt,
            locked_cash: new_locked_cash,
            free_cash: new_free_cash,
        },
        OrderResult::filled(size_limit, adj_price, fees_paid, OrderSide::Sell),
    ))
}

/// The margin one unit opened at `adj_price` posts.
#[inline]
fn unit_margin(order: &Order, adj_price: f64) -> f64 {
    match order.margin_basis {
        MarginBasis::Notional => {
            let rate = if order.initial_margin.is_nan() {
                1.0 / order.leverage
            } else {
                order.initial_margin
            };
            rate * adj_price
        }
        MarginBasis::Unit => order.initial_margin,
    }
}

/// Trades a margin column: closes the part of the position the order runs
/// against, then opens the rest by posting initial margin.
///
/// `size` is signed as the size cascade leaves it, positive to buy; zero and
/// `NaN` sell, as at the dispatch. `percent`, when not `NaN`, fills that
/// fraction of the most the order could fill on its side.
///
/// Closing is always allowed. The part closed realizes its profit against the
/// average entry price, `debt / |position|`, and releases its share of the
/// margin. Opening needs free cash for its margin and fees, counting what
/// closing the whole opposite position would realize and release, and
/// `cash_limit` caps what it may post, fees included. Every fee is paid from
/// cash, which may go below zero, as in any account that may borrow.
///
/// `leverage_mode` and `lock_cash` do not apply: nothing is borrowed, and
/// margin is reserved for every position anyway.
///
/// Out of line and cold: a run without margin columns never reaches it, and
/// the dispatch keeps [`buy`] and [`sell`] on the path it falls through to.
#[cold]
#[inline(never)]
fn margin_trade(
    state: ExecuteOrderState,
    size: f64,
    price: f64,
    order: &Order,
    percent: f64,
) -> Result<Leg> {
    let reject = |info| Ok((state, OrderResult::not_filled(OrderStatus::Rejected, info)));
    let ignore_zero = || {
        Ok((
            state,
            OrderResult::not_filled(OrderStatus::Ignored, StatusInfo::SizeZero),
        ))
    };

    let buying = size > 0.0;
    let adj_price = if price > 0.0 {
        if buying {
            price * (1.0 + order.slippage)
        } else {
            price * (1.0 - order.slippage)
        }
    } else {
        against(price, buying, order.slippage)
    };
    // Fees and margin are on the price's size, which for a price above 0 is
    // the price itself, bit for bit.
    let size_price = adj_price.abs();
    let held = state.position.abs();
    // The part of the position this order runs against.
    let closable = if state.position != 0.0 && (state.position > 0.0) != buying {
        held
    } else {
        0.0
    };
    let may_open = match order.direction {
        Direction::Both => true,
        Direction::LongOnly => buying,
        Direction::ShortOnly => !buying,
    };
    if !may_open && closable == 0.0 {
        return reject(StatusInfo::NoOpenPosition);
    }
    let zero = if percent.is_nan() {
        size == 0.0
    } else {
        percent == 0.0
    };
    if zero {
        return ignore_zero();
    }

    let open_max = if may_open {
        if closable == 0.0 && state.free_cash.min(order.cash_limit) <= 0.0 {
            return reject(if buying {
                StatusInfo::NoCashLong
            } else {
                StatusInfo::NoCashShort
            });
        }
        // Closing all of `closable` first realizes its profit and releases its
        // margin, both of which the opening part may then post.
        let freed = if closable > 0.0 {
            let proceeds = closable * adj_price;
            let pnl = if state.position > 0.0 {
                proceeds - state.debt
            } else {
                state.debt - proceeds
            };
            pnl + state.locked_cash - closable * size_price * order.fees
        } else {
            0.0
        };
        let budget = add(
            add(state.free_cash, freed).min(order.cash_limit),
            -order.fixed_fees,
        );
        if budget > 0.0 {
            budget / (unit_margin(order, size_price) + size_price * order.fees)
        } else if closable == 0.0 {
            return reject(StatusInfo::CantCoverFees);
        } else {
            0.0
        }
    } else {
        0.0
    };
    let max_total = closable + open_max;

    let mut fill = if percent.is_nan() {
        size.abs()
    } else {
        percent * max_total
    };
    if fill > order.max_size {
        if !order.allow_partial {
            return reject(StatusInfo::MaxSizeExceeded);
        }
        fill = order.max_size;
    }
    fill = fill.min(max_total);
    if fill.is_infinite() {
        return Err(Error::Invariant(
            "attempt to trade a margin column infinitely: both the size and the free cash are \
             infinite"
                .into(),
        ));
    }
    if !order.size_granularity.is_nan() {
        fill = floor_to(fill, order.size_granularity);
    }

    if is_close(fill, 0.0) {
        return ignore_zero();
    }
    if is_less(fill, order.min_size) {
        return reject(StatusInfo::MinSizeNotReached);
    }
    if size.is_finite() && is_less(fill, size.abs()) && !order.allow_partial {
        return reject(StatusInfo::PartialFill);
    }

    let new_position = add(state.position, if buying { fill } else { -fill });
    let closed = fill.min(closable);
    let opened = fill - closed;
    let fees_paid = fill * size_price * order.fees + order.fixed_fees;

    // Closing realizes the closed part's profit and releases its margin. A
    // close that leaves nothing of the old position takes all of both, so no
    // residue survives in either balance.
    let (realized, released, debt, locked_cash) = if closed > 0.0 {
        let (cost, released) = if new_position * state.position > 0.0 {
            (
                closed * (state.debt / held),
                closed * (state.locked_cash / held),
            )
        } else {
            (state.debt, state.locked_cash)
        };
        let proceeds = closed * adj_price;
        let realized = if state.position > 0.0 {
            proceeds - cost
        } else {
            cost - proceeds
        };
        (
            realized,
            released,
            add(state.debt, -cost),
            add(state.locked_cash, -released),
        )
    } else {
        (0.0, 0.0, state.debt, state.locked_cash)
    };
    // Opening adds its cost basis and posts its margin.
    let posted = opened * unit_margin(order, size_price);

    Ok((
        ExecuteOrderState {
            cash: add(state.cash, realized - fees_paid),
            position: new_position,
            debt: debt + opened * adj_price,
            locked_cash: locked_cash + posted,
            free_cash: add(
                state.free_cash,
                (realized + released) - (fees_paid + posted),
            ),
        },
        OrderResult::filled(
            fill,
            adj_price,
            fees_paid,
            if buying {
                OrderSide::Buy
            } else {
                OrderSide::Sell
            },
        ),
    ))
}

/// A price of 0 or below moved against a trade by `slippage`: up for a buy,
/// down for a sell, by a fraction of its size.
fn against(price: f64, buying: bool, slippage: f64) -> f64 {
    if buying {
        price + price.abs() * slippage
    } else {
        price - price.abs() * slippage
    }
}

/// A cash column's trade at a price of 0 or below (`allow_negative_price`):
/// a buy, or a sell of what the column holds, sized in units.
///
/// A buy at a price below 0 is paid to take the position; a sell pays to be
/// rid of it. Either is refused for want of cash when the cash and fees it
/// takes exceed the free cash, as a buy above 0 is. Shorting, leverage and a
/// size in percent need a price above 0.
#[cold]
#[inline(never)]
fn signed_price_trade(
    state: ExecuteOrderState,
    size: f64,
    price: f64,
    order: &Order,
    percent: f64,
) -> Result<Leg> {
    let reject = |info| Ok((state, OrderResult::not_filled(OrderStatus::Rejected, info)));
    if !percent.is_nan() || !size.is_finite() {
        return Err(Error::Invariant(
            "a price of 0 or below needs a size in units, not in percent or infinite".into(),
        ));
    }
    if order.leverage > 1.0 {
        return Err(Error::Invariant(
            "a price of 0 or below in a cash column needs leverage 1".into(),
        ));
    }
    if size == 0.0 {
        return Ok((
            state,
            OrderResult::not_filled(OrderStatus::Ignored, StatusInfo::SizeZero),
        ));
    }
    let buying = size > 0.0;
    let shorts = if buying {
        state.position < 0.0
    } else {
        // A long-only sell of more than is held sells what is held.
        order.direction != Direction::LongOnly && is_less(state.position, size.abs())
    };
    if shorts {
        return Err(Error::Invariant(
            "a cash column cannot short, or cover a short, at a price of 0 or below".into(),
        ));
    }
    if order.direction == Direction::ShortOnly || (!buying && state.position <= 0.0) {
        return reject(StatusInfo::NoOpenPosition);
    }
    let mut fill = size.abs();
    if !buying {
        fill = fill.min(state.position);
    }
    if fill > order.max_size {
        if !order.allow_partial {
            return reject(StatusInfo::MaxSizeExceeded);
        }
        fill = order.max_size;
    }
    if !order.size_granularity.is_nan() {
        fill = floor_to(fill, order.size_granularity);
    }
    if is_close(fill, 0.0) {
        return Ok((
            state,
            OrderResult::not_filled(OrderStatus::Ignored, StatusInfo::SizeZero),
        ));
    }
    if is_less(fill, order.min_size) {
        return reject(StatusInfo::MinSizeNotReached);
    }
    let adj_price = against(price, buying, order.slippage);
    let fees_paid = fill * adj_price.abs() * order.fees + order.fixed_fees;
    // What the trade takes out of the cash: a buy's cost, a sell's negative
    // proceeds, and the fees.
    let outflow = if buying {
        fill * adj_price
    } else {
        -fill * adj_price
    } + fees_paid;
    if outflow > 0.0 && !is_close_or_less(outflow, state.free_cash.min(order.cash_limit)) {
        return reject(if buying {
            StatusInfo::NoCashLong
        } else {
            StatusInfo::CantCoverFees
        });
    }
    Ok((
        ExecuteOrderState {
            cash: add(state.cash, -outflow),
            position: add(state.position, if buying { fill } else { -fill }),
            debt: state.debt,
            locked_cash: state.locked_cash,
            free_cash: add(state.free_cash, -outflow),
        },
        OrderResult::filled(
            fill,
            adj_price,
            fees_paid,
            if buying {
                OrderSide::Buy
            } else {
                OrderSide::Sell
            },
        ),
    ))
}

/// Validates the inputs the kernel's arithmetic depends on.
///
/// These are invariant violations, not rejections — see the crate docs. Two
/// checks that a dynamically typed caller would need are absent here because
/// the type system already makes them unrepresentable: `size_type` and `direction` are enums, so they
/// cannot hold an out-of-range discriminant.
///
/// The eight checks that read only the bar: state, price and direction.
///
/// `levered_account` admits cash below zero, which a leveraged account reaches
/// when a long is sold for less than its debt: the account then owes the
/// difference, and can sell but not buy until it is paid. Anywhere else a
/// negative balance is a bad input.
fn validate_bar(
    state: &ExecuteOrderState,
    order: &Order,
    position: f64,
    levered_account: bool,
) -> Result<()> {
    let bail = |msg: &str| Err(Error::Invariant(msg.to_owned()));

    if state.cash.is_nan() || (state.cash < 0.0 && !levered_account) {
        return bail("cash cannot be NaN and must be 0 or greater");
    }
    if !position.is_finite() {
        return bail("position must be finite");
    }
    // Both balances in one test, which lets through nothing the two exact
    // checks would stop. Written as two, each compiles to an integer
    // classification of the float, about 17 instructions on aarch64, and
    // this runs on every bar.
    if !(state.debt + state.locked_cash).is_finite() || state.debt.min(state.locked_cash) < 0.0 {
        balance_violation(state, order)?;
    }
    if state.free_cash.is_nan() {
        return bail("free_cash cannot be NaN");
    }

    if (!order.price.is_finite() || order.price <= 0.0)
        && !(order.allow_negative_price && order.price.is_finite())
    {
        return bail("order.price must be finite and greater than 0");
    }
    if order.direction == Direction::LongOnly && position < 0.0 {
        return bail("position is negative but order.direction is LongOnly");
    }
    if order.direction == Direction::ShortOnly && position > 0.0 {
        return bail("position is positive but order.direction is ShortOnly");
    }
    Ok(())
}

/// The exact checks behind [`validate_bar`]'s test of `debt` and
/// `locked_cash`. That test also fails when two finite balances add up to
/// more than the largest float, which this lets through.
#[cold]
#[inline(never)]
fn balance_violation(state: &ExecuteOrderState, order: &Order) -> Result<()> {
    // A margin position opened below 0 has a cost basis below 0.
    let signed_basis = order.allow_negative_price && order.account_type == AccountType::Margin;
    if !state.debt.is_finite() || (state.debt < 0.0 && !signed_basis) {
        return Err(Error::Invariant(
            "debt must be finite and 0 or greater".into(),
        ));
    }
    if !state.locked_cash.is_finite() || state.locked_cash < 0.0 {
        return Err(Error::Invariant(
            "locked_cash must be finite and 0 or greater".into(),
        ));
    }
    Ok(())
}

/// The nine checks that read only fields a whole column usually shares, and
/// the two a margin order adds for its initial margin.
///
/// Separated from [`validate_bar`] so a caller with column-constant
/// parameters can run them once instead of once per bar — see
/// [`ValidatedOrder`]. Every check in either half is independent of the
/// others, so splitting them changes no outcome; what it must not change is
/// which message a doubly-invalid order reports, and the call site keeps the
/// two halves in their original order for that reason.
///
/// Returns the message rather than an [`Error`] so that a [`ValidatedOrder`]
/// stays `Copy` and one pointer wide.
fn param_violation(order: &Order) -> Option<&'static str> {
    if !order.fees.is_finite() {
        return Some("order.fees must be finite");
    }
    if !order.fixed_fees.is_finite() {
        return Some("order.fixed_fees must be finite");
    }
    if !order.slippage.is_finite() || order.slippage < 0.0 {
        return Some("order.slippage must be finite and 0 or greater");
    }
    if !order.min_size.is_finite() || order.min_size < 0.0 {
        return Some("order.min_size must be finite and 0 or greater");
    }
    if order.max_size.is_nan() || order.max_size <= 0.0 {
        return Some("order.max_size must be greater than 0");
    }
    // NaN is the "no granularity" sentinel and passes both tests.
    if order.size_granularity.is_infinite() || order.size_granularity <= 0.0 {
        return Some("order.size_granularity must be NaN, or finite and greater than 0");
    }
    if !order.leverage.is_finite() || order.leverage < 1.0 {
        return Some("order.leverage must be finite and 1 or greater");
    }
    if !order.reject_prob.is_finite() || !(0.0..=1.0).contains(&order.reject_prob) {
        return Some("order.reject_prob must be between 0 and 1");
    }
    if order.cash_limit.is_nan() || order.cash_limit < 0.0 {
        return Some("order.cash_limit must be 0 or greater");
    }
    if order.account_type == AccountType::Margin {
        // NaN is `1 / leverage` and passes, as granularity's sentinel does.
        if order.initial_margin.is_infinite() || order.initial_margin <= 0.0 {
            return Some("order.initial_margin must be NaN, or finite and greater than 0");
        }
        if order.margin_basis == MarginBasis::Unit && order.initial_margin.is_nan() {
            return Some("order.initial_margin must be set when order.margin_basis is Unit");
        }
    }
    None
}

/// An [`Order`] whose column-invariant parameters have already been checked.
///
/// Nine of the seventeen checks [`execute_order`] runs look only at fees,
/// slippage, size bounds, granularity, leverage, rejection probability and
/// the cash cap — fields a whole column normally shares. A margin order adds
/// two more on its initial margin. A bar that produces
/// no fill still pays for all nine, and on a realistic strategy most bars produce no fill. Checking
/// them once per column instead is worth 12–15% there.
///
/// Build one per column, then call [`set_bar`](Self::set_bar) each bar with
/// the four fields the parameter checks do not cover. Rebuild it when a
/// parameter genuinely varies down the column.
///
/// # Why construction cannot fail
///
/// A `NaN` size is ignored *before* anything is validated, so a
/// column of `NaN` sizes and an invalid fee rate must produce ignored orders,
/// not an error. Failing eagerly here would change that. The violation is
/// therefore carried and raised at the exact point the per-bar check would
/// have raised it — which costs one branch instead of nine.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValidatedOrder {
    order: Order,
    violation: Option<&'static str>,
    levered_account: bool,
}

impl ValidatedOrder {
    /// Checks `order`'s column-invariant parameters.
    ///
    /// An order whose `leverage` is above 1, and any margin order, accepts a
    /// negative cash balance; see [`in_levered_account`](Self::in_levered_account).
    #[must_use]
    pub fn new(order: Order) -> Self {
        Self {
            violation: param_violation(&order),
            levered_account: order.leverage > 1.0 || order.account_type == AccountType::Margin,
            order,
        }
    }

    /// Accepts a negative cash balance whatever this order's own leverage.
    ///
    /// A leveraged long sold for less than its debt leaves the account owing
    /// the difference, so its cash goes below zero, and so can a margin
    /// position's loss. Every order on that account then has to accept the
    /// state — an unleveraged one in the same group included — and is
    /// rejected for want of cash rather than stopping the run. A loop marks
    /// every order of an account that may borrow or trade on margin.
    #[must_use]
    pub const fn in_levered_account(mut self) -> Self {
        self.levered_account = true;
        self
    }

    /// Overwrites the four fields that vary bar to bar.
    ///
    /// None of them is covered by the parameter checks, so the validation
    /// this type carries stays sound.
    #[inline]
    pub const fn set_bar(
        &mut self,
        size: f64,
        price: f64,
        size_type: SizeType,
        direction: Direction,
    ) {
        self.order.size = size;
        self.order.price = price;
        self.order.size_type = size_type;
        self.order.direction = direction;
    }

    /// The underlying request.
    #[must_use]
    pub const fn order(&self) -> &Order {
        &self.order
    }

    /// Reports a parameter violation now rather than at the next execution.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invariant`] if a parameter check failed.
    pub fn check(&self) -> Result<()> {
        match self.violation {
            Some(msg) => Err(Error::Invariant(msg.to_owned())),
            None => Ok(()),
        }
    }
}

/// Executes one order against one state.
///
/// Pure and deterministic. When `order.reject_prob` is non-zero the caller
/// supplies `reject_draw`, a sample from `U(0, 1)`; the order is rejected when
/// that draw is below the probability. Inverting control this way — rather
/// than reaching for a global RNG — is what lets a
/// parameter sweep seed each column reproducibly and still run in parallel.
///
/// # Errors
///
/// Returns [`Error::Invariant`] when an input violates something the
/// arithmetic depends on, or when `reject_prob` is non-zero and no draw was
/// supplied. A *rejected* order is not an error; it comes back as an
/// [`OrderResult`] with [`OrderStatus::Rejected`].
pub fn execute_order(
    state: &ProcessOrderState,
    order: &Order,
    reject_draw: Option<f64>,
) -> Result<Leg> {
    execute_validated(state, &ValidatedOrder::new(*order), reject_draw)
}

/// The percent spellings are the fractions they write, a hundredth;
/// `ValuePercent` is `Value` of `size × value`, as `TargetPercent` is
/// `TargetValue`. A `TargetPercent100` comes back as `TargetPercent`.
#[cold]
#[inline(never)]
fn percent_spelling(
    size: f64,
    size_type: SizeType,
    value: f64,
) -> std::result::Result<(f64, SizeType), OrderResult> {
    let (size, size_type) = match size_type {
        SizeType::Percent100 => (size / 100.0, SizeType::Percent),
        SizeType::ValuePercent100 => (size / 100.0, SizeType::ValuePercent),
        SizeType::TargetPercent100 => (size / 100.0, SizeType::TargetPercent),
        other => (size, other),
    };
    if size_type != SizeType::ValuePercent {
        return Ok((size, size_type));
    }
    if value.is_nan() {
        return Err(OrderResult::not_filled(
            OrderStatus::Ignored,
            StatusInfo::ValueNaN,
        ));
    }
    if value <= 0.0 {
        return Err(OrderResult::not_filled(
            OrderStatus::Rejected,
            StatusInfo::ValueZeroNeg,
        ));
    }
    Ok((size * value, SizeType::Value))
}

/// [`execute_order`], with the parameter checks already done.
///
/// Identical in every observable way — same fills, same rejections, same
/// error at the same point. It exists only so a column loop can hoist nine
/// checks out of itself; see [`ValidatedOrder`].
///
/// # Errors
///
/// As [`execute_order`].
pub fn execute_validated(
    state: &ProcessOrderState,
    validated: &ValidatedOrder,
    reject_draw: Option<f64>,
) -> Result<Leg> {
    let order = &validated.order;
    // Numerical stability: snap everything that should be zero to zero before
    // any comparison sees it. Residue here keeps closed positions nominally
    // open and makes affordability tests flap.
    // `locked_cash` is left alone: it is only ever read in proportion to a
    // position, and the kernel never leaves residue in it.
    let snap = |v: f64| if is_close(v, 0.0) { 0.0 } else { v };
    let cash = snap(state.cash);
    let position = snap(state.position);
    let debt = snap(state.debt);
    let locked_cash = state.locked_cash;
    let free_cash = snap(state.free_cash);
    let val_price = snap(state.val_price);
    let value = snap(state.value);

    let exec_state = ExecuteOrderState {
        cash,
        position,
        debt,
        locked_cash,
        free_cash,
    };

    // Ignore checks come before validation: a NaN size is ignored even when
    // the state would otherwise be rejected.
    if order.size.is_nan() {
        return Ok((
            exec_state,
            OrderResult::not_filled(OrderStatus::Ignored, StatusInfo::SizeNaN),
        ));
    }
    if order.price.is_nan() {
        return Ok((
            exec_state,
            OrderResult::not_filled(OrderStatus::Ignored, StatusInfo::PriceNaN),
        ));
    }

    // The bar half first, then the parameter half: that is the order the
    // checks ran in when they were one function, and it decides which message
    // an order that violates both reports.
    validate_bar(&exec_state, order, position, validated.levered_account)?;
    validated.check()?;

    if order.reject_prob > 0.0 && reject_draw.is_none() {
        return Err(Error::Invariant(
            "order.reject_prob is non-zero but no random draw was supplied".into(),
        ));
    }

    // The size cascade. Order matters: each stage rewrites the size type, and
    // only `Amount` reaches the buy/sell dispatch.
    let mut order_size = order.size;
    let mut size_type = order.size_type;

    if order.direction == Direction::ShortOnly {
        // In a short-only column a positive request means "short more".
        order_size = -order_size;
    }

    // `TargetPercent` (5), and `ValuePercent` and the `*100` spellings (6–9)
    // out of line: one comparison when none of them is used.
    if size_type as i8 >= SizeType::TargetPercent as i8 {
        if size_type != SizeType::TargetPercent {
            match percent_spelling(order_size, size_type, value) {
                Ok(rewritten) => (order_size, size_type) = rewritten,
                Err(status) => return Ok((exec_state, status)),
            }
        }
        if size_type == SizeType::TargetPercent {
            if value.is_nan() {
                return Ok((
                    exec_state,
                    OrderResult::not_filled(OrderStatus::Ignored, StatusInfo::ValueNaN),
                ));
            }
            if value <= 0.0 {
                return Ok((
                    exec_state,
                    OrderResult::not_filled(OrderStatus::Rejected, StatusInfo::ValueZeroNeg),
                ));
            }
            order_size *= value;
            size_type = SizeType::TargetValue;
        }
    }

    if matches!(size_type, SizeType::Value | SizeType::TargetValue) {
        if val_price.is_infinite() || val_price <= 0.0 {
            return Err(Error::Invariant(
                "val_price must be finite and greater than 0".into(),
            ));
        }
        if val_price.is_nan() {
            return Ok((
                exec_state,
                OrderResult::not_filled(OrderStatus::Ignored, StatusInfo::ValPriceNaN),
            ));
        }
        order_size /= val_price;
        size_type = if size_type == SizeType::Value {
            SizeType::Amount
        } else {
            SizeType::TargetAmount
        };
    }

    if size_type == SizeType::TargetAmount {
        order_size -= position;
        size_type = SizeType::Amount;
    }

    if size_type == SizeType::Amount
        && matches!(order.direction, Direction::ShortOnly | Direction::Both)
        && order_size.is_infinite()
        && order_size < 0.0
    {
        // Negative infinity has a special meaning: short 100% of resources.
        order_size = -1.0;
        size_type = SizeType::Percent;
    }

    let mut percent = f64::NAN;
    if size_type == SizeType::Percent {
        percent = order_size.abs();
        // A zero request yields `0.0 * inf` == NaN here, by design.
        order_size = sign(order_size) * f64::INFINITY;
    }

    // Random rejection replaces whatever buying or selling would have
    // produced and leaves the state as it was, so it is decided before either
    // runs rather than after. The draw overrides every result that reaches
    // this point, and an order it rejects never reaches the two invariant
    // errors below, which only an infinite request can raise. Deciding first also means
    // nothing has to be kept alive across the dispatch for a rollback.
    if let Some(draw) = reject_draw {
        if draw < order.reject_prob {
            return Ok((
                exec_state,
                OrderResult::not_filled(OrderStatus::Rejected, StatusInfo::RandomEvent),
            ));
        }
    }

    if order.account_type == AccountType::Margin {
        return margin_trade(exec_state, order_size, order.price, order, percent);
    }
    if order.price <= 0.0 {
        return signed_price_trade(exec_state, order_size, order.price, order, percent);
    }
    // Note zero and NaN both take the sell branch.
    if order_size > 0.0 {
        buy(exec_state, order_size, order.price, order, percent)
    } else {
        sell(exec_state, -order_size, order.price, order, percent)
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "these assert on exact expected balances (50.0, 48.5, 0.0); an \
              approximate assertion here would not catch the arithmetic errors \
              the tests exist to catch"
)]
mod tests {
    use super::*;

    fn flat() -> ProcessOrderState {
        ProcessOrderState::flat(100.0, 10.0)
    }

    #[test]
    fn the_percent_spellings_are_the_types_they_rewrite_to() {
        let held = ProcessOrderState {
            cash: 1000.0,
            position: 5.0,
            debt: 0.0,
            locked_cash: 0.0,
            free_cash: 1000.0,
            val_price: 10.0,
            value: 1050.0,
            oidx: 0,
            lidx: 0,
        };
        for state in [flat(), held] {
            let run = |size, size_type, direction| {
                let order = Order {
                    size,
                    price: 10.0,
                    size_type,
                    direction,
                    ..Default::default()
                };
                execute_order(&state, &order, None).unwrap()
            };
            // A long cannot be held in a short-only column.
            let directions: &[Direction] = if state.position > 0.0 {
                &[Direction::Both, Direction::LongOnly]
            } else {
                &[Direction::Both, Direction::LongOnly, Direction::ShortOnly]
            };
            for &direction in directions {
                let pairs = [
                    (50.0, SizeType::Percent100, 0.5, SizeType::Percent),
                    (-50.0, SizeType::Percent100, -0.5, SizeType::Percent),
                    (
                        30.0,
                        SizeType::TargetPercent100,
                        0.3,
                        SizeType::TargetPercent,
                    ),
                    (
                        0.3,
                        SizeType::ValuePercent,
                        0.3 * state.value,
                        SizeType::Value,
                    ),
                    (
                        -20.0,
                        SizeType::ValuePercent100,
                        -0.2 * state.value,
                        SizeType::Value,
                    ),
                ];
                for (size, spelled, plain, as_type) in pairs {
                    // As text: a refused order's NaNs are equal there.
                    assert_eq!(
                        format!("{:?}", run(size, spelled, direction)),
                        format!("{:?}", run(plain, as_type, direction)),
                        "{spelled:?} {size} on {direction:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn plain_buy_spends_cash_and_takes_a_position() {
        let order = Order {
            size: 5.0,
            price: 10.0,
            ..Default::default()
        };
        let (state, result) = execute_order(&flat(), &order, None).unwrap();
        assert_eq!(state.cash, 50.0);
        assert_eq!(state.position, 5.0);
        assert!(result.is_filled());
        assert_eq!(result.side, Some(OrderSide::Buy));
    }

    #[test]
    fn fees_come_out_of_the_cash_cap_not_on_top() {
        // 5 @ 10 = 50 notional, 1% = 0.5, plus 1 fixed => 48.5 left.
        let order = Order {
            size: 5.0,
            price: 10.0,
            fees: 0.01,
            fixed_fees: 1.0,
            ..Default::default()
        };
        let (state, result) = execute_order(&flat(), &order, None).unwrap();
        assert_eq!(result.fees, 1.5);
        assert_eq!(state.cash, 48.5);
    }

    #[test]
    fn a_bad_parameter_is_an_error() {
        let order = Order {
            size: 5.0,
            price: 10.0,
            fees: f64::INFINITY,
            ..Default::default()
        };
        assert!(execute_order(&flat(), &order, None).is_err());
    }

    #[test]
    fn a_nan_size_is_ignored_even_when_a_parameter_is_bad() {
        // The whole reason `ValidatedOrder` carries its violation instead of
        // returning it: hoisting the parameter checks out of a column loop
        // must not turn this column's ignored orders into an error.
        let order = Order {
            size: f64::NAN,
            price: 10.0,
            fees: f64::INFINITY,
            ..Default::default()
        };
        let (_, result) = execute_order(&flat(), &order, None).unwrap();
        assert_eq!(result.status, OrderStatus::Ignored);
        assert_eq!(result.status_info, Some(StatusInfo::SizeNaN));
    }

    #[test]
    fn a_bad_state_outranks_a_bad_parameter() {
        // Both halves are violated. The bar half ran first before the split
        // and must still, or the error message changes.
        let mut state = flat();
        state.cash = -1.0;
        let order = Order {
            size: 5.0,
            price: 10.0,
            fees: f64::INFINITY,
            ..Default::default()
        };
        let err = execute_order(&state, &order, None).unwrap_err();
        assert!(format!("{err}").contains("cash"));
    }

    #[test]
    fn set_bar_reuses_the_validation() {
        let mut v = ValidatedOrder::new(Order {
            fees: 0.01,
            fixed_fees: 1.0,
            ..Default::default()
        });
        v.set_bar(5.0, 10.0, SizeType::Amount, Direction::Both);
        let (hoisted, _) = execute_validated(&flat(), &v, None).unwrap();

        let (inline, _) = execute_order(
            &flat(),
            &Order {
                size: 5.0,
                price: 10.0,
                fees: 0.01,
                fixed_fees: 1.0,
                ..Default::default()
            },
            None,
        )
        .unwrap();
        assert_eq!(hoisted, inline);
    }

    #[test]
    fn rejection_is_a_status_not_an_error() {
        let order = Order {
            size: 5.0,
            price: 10.0,
            ..Default::default()
        };
        let broke = ProcessOrderState::flat(0.0, 10.0);
        let (state, result) = execute_order(&broke, &order, None).unwrap();
        assert_eq!(result.status, OrderStatus::Rejected);
        assert_eq!(result.status_info, Some(StatusInfo::NoCashLong));
        // State is untouched by a rejection.
        assert_eq!(state.cash, 0.0);
        assert_eq!(state.position, 0.0);
    }

    #[test]
    fn invariant_violation_is_an_error() {
        let order = Order {
            size: 5.0,
            price: 10.0,
            ..Default::default()
        };
        let bad = ProcessOrderState {
            cash: -1.0,
            ..ProcessOrderState::flat(0.0, 10.0)
        };
        assert!(execute_order(&bad, &order, None).is_err());
    }

    #[test]
    fn nan_size_is_ignored_even_when_the_state_is_invalid() {
        // The ignore checks run before validation.
        let order = Order {
            size: f64::NAN,
            price: 10.0,
            ..Default::default()
        };
        let bad = ProcessOrderState {
            cash: -1.0,
            ..ProcessOrderState::flat(0.0, 10.0)
        };
        let (_, result) = execute_order(&bad, &order, None).unwrap();
        assert_eq!(result.status, OrderStatus::Ignored);
        assert_eq!(result.status_info, Some(StatusInfo::SizeNaN));
    }

    #[test]
    fn reject_prob_requires_a_draw() {
        let order = Order {
            size: 5.0,
            price: 10.0,
            reject_prob: 0.5,
            ..Default::default()
        };
        assert!(execute_order(&flat(), &order, None).is_err());

        // Draw below the probability rejects and restores the state.
        let (state, result) = execute_order(&flat(), &order, Some(0.1)).unwrap();
        assert_eq!(result.status_info, Some(StatusInfo::RandomEvent));
        assert_eq!(state.cash, 100.0);

        // Draw above it fills.
        let (state, result) = execute_order(&flat(), &order, Some(0.9)).unwrap();
        assert!(result.is_filled());
        assert_eq!(state.cash, 50.0);
    }

    #[test]
    fn shorting_records_debt_and_reserves_margin() {
        let order = Order {
            size: -5.0,
            price: 10.0,
            ..Default::default()
        };
        let (state, result) = execute_order(&flat(), &order, None).unwrap();
        assert!(result.is_filled());
        assert_eq!(state.position, -5.0);
        // Proceeds land in cash, debt records them, and free_cash is reduced
        // by the round-trip reservation: 100 + 50 - 2*50 = 50.
        assert_eq!(state.cash, 150.0);
        assert_eq!(state.debt, 50.0);
        assert_eq!(state.free_cash, 50.0);
    }

    /// `start` with the balances an execution left.
    fn carry(state: ExecuteOrderState, start: ProcessOrderState) -> ProcessOrderState {
        ProcessOrderState {
            cash: state.cash,
            position: state.position,
            debt: state.debt,
            locked_cash: state.locked_cash,
            free_cash: state.free_cash,
            ..start
        }
    }

    fn levered(size: f64, leverage: f64, leverage_mode: LeverageMode) -> Order {
        Order {
            size,
            price: 100.0,
            leverage,
            leverage_mode,
            ..Default::default()
        }
    }

    #[test]
    fn a_percent_buy_uses_the_whole_leveraged_buying_power() {
        // $1000 at 3x buys $3000 of stock: 30 shares at $100.
        let order = Order {
            size_type: SizeType::Percent,
            ..levered(1.0, 3.0, LeverageMode::Lazy)
        };
        let start = ProcessOrderState::flat(1000.0, 100.0);
        let (state, result) = execute_order(&start, &order, None).unwrap();
        assert_eq!(result.size, 30.0);
        assert_eq!(state.cash, 0.0);
        assert_eq!(state.debt, 2000.0);
        assert_eq!(state.locked_cash, 1000.0);
        assert_eq!(state.free_cash, 0.0);
    }

    #[test]
    fn eager_leverage_borrows_as_much_as_it_can() {
        // 10 @ 100 at 2x: half paid, half borrowed, though the cash would
        // cover all of it.
        let start = ProcessOrderState::flat(1000.0, 100.0);
        let order = levered(10.0, 2.0, LeverageMode::Eager);
        let (state, result) = execute_order(&start, &order, None).unwrap();
        assert_eq!(result.size, 10.0);
        assert_eq!(state.cash, 500.0);
        assert_eq!(state.debt, 500.0);
        assert_eq!(state.locked_cash, 500.0);
        assert_eq!(state.free_cash, 500.0);
    }

    #[test]
    fn lazy_leverage_borrows_only_what_cash_cannot_cover() {
        let order = levered(10.0, 2.0, LeverageMode::Lazy);

        let (state, _) =
            execute_order(&ProcessOrderState::flat(1000.0, 100.0), &order, None).unwrap();
        assert_eq!(state.cash, 0.0);
        assert_eq!(state.debt, 0.0);
        assert_eq!(state.locked_cash, 1000.0);

        let (state, _) =
            execute_order(&ProcessOrderState::flat(600.0, 100.0), &order, None).unwrap();
        assert_eq!(state.cash, 0.0);
        assert_eq!(state.debt, 400.0);
        assert_eq!(state.locked_cash, 600.0);
    }

    #[test]
    fn a_leveraged_buy_short_of_cash_fills_what_the_cash_carries() {
        // $300 at 2x carries $600 of stock.
        let start = ProcessOrderState::flat(300.0, 100.0);
        let order = levered(10.0, 2.0, LeverageMode::Lazy);
        let (state, result) = execute_order(&start, &order, None).unwrap();
        assert_eq!(result.size, 6.0);
        assert_eq!(state.debt, 300.0);
        assert_eq!(state.cash, 0.0);

        let strict = Order {
            allow_partial: false,
            ..order
        };
        let (_, result) = execute_order(&start, &strict, None).unwrap();
        assert_eq!(result.status_info, Some(StatusInfo::PartialFill));
    }

    #[test]
    fn fees_are_paid_from_own_cash() {
        // 10 @ 100 at 2x with 1% fees: $500 own, $10 fees, $500 borrowed.
        let start = ProcessOrderState::flat(1000.0, 100.0);
        let order = Order {
            fees: 0.01,
            ..levered(10.0, 2.0, LeverageMode::Eager)
        };
        let (state, result) = execute_order(&start, &order, None).unwrap();
        assert_eq!(result.fees, 10.0);
        assert_eq!(state.cash, 490.0);
        assert_eq!(state.debt, 500.0);
        assert_eq!(state.locked_cash, 500.0);
    }

    #[test]
    fn selling_a_leveraged_long_repays_its_debt() {
        let start = ProcessOrderState::flat(1000.0, 100.0);
        let (long, _) =
            execute_order(&start, &levered(10.0, 2.0, LeverageMode::Eager), None).unwrap();
        let held = carry(long, start);
        let sell = |size: f64| Order {
            price: 120.0,
            ..levered(size, 2.0, LeverageMode::Eager)
        };

        // Half: repays half the debt and frees half the own cash.
        let (state, _) = execute_order(&held, &sell(-5.0), None).unwrap();
        assert_eq!(state.cash, 850.0);
        assert_eq!(state.debt, 250.0);
        assert_eq!(state.locked_cash, 250.0);
        assert_eq!(state.free_cash, 850.0);

        // All of it: 1200 of proceeds less the 500 owed.
        let (state, _) = execute_order(&held, &sell(-10.0), None).unwrap();
        assert_eq!(state.cash, 1200.0);
        assert_eq!(state.debt, 0.0);
        assert_eq!(state.locked_cash, 0.0);
        assert_eq!(state.free_cash, 1200.0);
    }

    #[test]
    fn a_long_sold_below_its_debt_leaves_the_account_owing() {
        // $250 at 4x holds $1000 of stock; at half the price it repays only
        // 500 of the 750 borrowed.
        let start = ProcessOrderState::flat(250.0, 100.0);
        let (long, _) =
            execute_order(&start, &levered(10.0, 4.0, LeverageMode::Lazy), None).unwrap();
        assert_eq!(long.debt, 750.0);
        let held = carry(long, start);
        let crash = Order {
            price: 50.0,
            ..levered(-10.0, 4.0, LeverageMode::Lazy)
        };
        let (owing, _) = execute_order(&held, &crash, None).unwrap();
        assert_eq!(owing.cash, -250.0);
        assert_eq!(owing.debt, 0.0);
        let owing = carry(owing, start);

        // A leveraged buy has no own cash to put up.
        let (_, result) =
            execute_order(&owing, &levered(1.0, 4.0, LeverageMode::Lazy), None).unwrap();
        assert_eq!(result.status_info, Some(StatusInfo::NoCashLeverage));

        // An unleveraged one in the same account is rejected, not an error;
        // outside a leveraged account the balance is invalid.
        let plain = levered(1.0, 1.0, LeverageMode::Lazy);
        let in_account = ValidatedOrder::new(plain).in_levered_account();
        let (_, result) = execute_validated(&owing, &in_account, None).unwrap();
        assert_eq!(result.status_info, Some(StatusInfo::NoCashLong));
        assert!(execute_order(&owing, &plain, None).is_err());
    }

    #[test]
    fn a_short_ties_up_its_value_over_the_leverage() {
        let start = ProcessOrderState::flat(1000.0, 100.0);
        let short = Order {
            fees: 0.01,
            ..levered(-10.0, 2.0, LeverageMode::Lazy)
        };
        let (state, _) = execute_order(&start, &short, None).unwrap();
        // Proceeds of 1000 less 10 of fees; 1000 reserved as debt and 500 as
        // collateral, so free cash drops by 500 + 10.
        assert_eq!(state.cash, 1990.0);
        assert_eq!(state.debt, 1000.0);
        assert_eq!(state.locked_cash, 500.0);
        assert_eq!(state.free_cash, 490.0);

        // Covering at the same price releases both and costs only the fees.
        let held = carry(state, start);
        let cover = Order {
            fees: 0.01,
            ..levered(10.0, 2.0, LeverageMode::Lazy)
        };
        let (state, _) = execute_order(&held, &cover, None).unwrap();
        assert_eq!(state.position, 0.0);
        assert_eq!(state.cash, 980.0);
        assert_eq!(state.debt, 0.0);
        assert_eq!(state.locked_cash, 0.0);
        assert_eq!(state.free_cash, 980.0);
    }

    #[test]
    fn a_locked_short_is_capped_by_leveraged_collateral() {
        // $1000 of free cash backs $2000 of short at 2x.
        let start = ProcessOrderState::flat(1000.0, 100.0);
        let order = Order {
            lock_cash: true,
            ..levered(-30.0, 2.0, LeverageMode::Lazy)
        };
        let (state, result) = execute_order(&start, &order, None).unwrap();
        assert_eq!(result.size, 20.0);
        assert_eq!(state.free_cash, 0.0);
    }

    #[test]
    fn a_buy_through_a_short_covers_at_full_price_and_levers_the_rest() {
        let start = ProcessOrderState::flat(1000.0, 100.0);
        let (short, _) =
            execute_order(&start, &levered(-5.0, 2.0, LeverageMode::Eager), None).unwrap();
        let held = carry(short, start);
        // Cover 5 for 500, then 10 long for 500 own and 500 borrowed: the
        // same account as an eager 10 long opened from flat.
        let (state, _) =
            execute_order(&held, &levered(15.0, 2.0, LeverageMode::Eager), None).unwrap();
        assert_eq!(state.position, 10.0);
        assert_eq!(state.cash, 500.0);
        assert_eq!(state.debt, 500.0);
        assert_eq!(state.locked_cash, 500.0);
        assert_eq!(state.free_cash, 500.0);
    }

    #[test]
    fn leverage_below_one_or_not_finite_is_an_error() {
        let start = ProcessOrderState::flat(1000.0, 100.0);
        for leverage in [0.5, 0.0, -2.0, f64::NAN, f64::INFINITY] {
            let order = levered(1.0, leverage, LeverageMode::Lazy);
            let err = execute_order(&start, &order, None).unwrap_err();
            assert!(format!("{err}").contains("leverage"), "{leverage}");
        }
    }

    #[test]
    fn leverage_one_is_the_unleveraged_kernel_in_either_mode() {
        let start = ProcessOrderState::flat(1000.0, 100.0);
        for size in [3.0, 25.0, -4.0, f64::INFINITY] {
            let plain = execute_order(&start, &levered(size, 1.0, LeverageMode::Lazy), None);
            let eager = execute_order(&start, &levered(size, 1.0, LeverageMode::Eager), None);
            assert_eq!(plain.unwrap(), eager.unwrap(), "{size}");
        }
    }

    /// A margin order at a quarter of the notional, 4x.
    fn margin(size: f64, price: f64) -> Order {
        Order {
            size,
            price,
            account_type: AccountType::Margin,
            initial_margin: 0.25,
            ..Default::default()
        }
    }

    /// Long 20 at 100 on 1000 of cash: 2000 of cost, 500 of margin posted.
    fn margin_long() -> ProcessOrderState {
        let start = ProcessOrderState::flat(1000.0, 100.0);
        let (state, _) = execute_order(&start, &margin(20.0, 100.0), None).unwrap();
        carry(state, start)
    }

    #[test]
    fn a_margin_long_posts_margin_and_keeps_its_cost_as_debt() {
        let held = margin_long();
        assert_eq!(held.cash, 1000.0);
        assert_eq!(held.position, 20.0);
        assert_eq!(held.debt, 2000.0);
        assert_eq!(held.locked_cash, 500.0);
        assert_eq!(held.free_cash, 500.0);
    }

    #[test]
    fn a_margin_short_posts_the_same_margin_and_keeps_its_proceeds_out_of_cash() {
        let start = ProcessOrderState::flat(1000.0, 100.0);
        let (state, result) = execute_order(&start, &margin(-20.0, 100.0), None).unwrap();
        assert_eq!(result.side, Some(OrderSide::Sell));
        assert_eq!(state.cash, 1000.0);
        assert_eq!(state.position, -20.0);
        assert_eq!(state.debt, 2000.0);
        assert_eq!(state.locked_cash, 500.0);
        assert_eq!(state.free_cash, 500.0);
    }

    #[test]
    fn closing_realizes_against_the_average_entry_and_releases_margin() {
        let held = margin_long();
        // 8 of 20 at 110: cost 800, proceeds 880, margin 200 back.
        let (part, _) = execute_order(&held, &margin(-8.0, 110.0), None).unwrap();
        assert_eq!(part.cash, 1080.0);
        assert_eq!(part.position, 12.0);
        assert_eq!(part.debt, 1200.0);
        assert_eq!(part.locked_cash, 300.0);
        assert_eq!(part.free_cash, 780.0);
        // The other 12 at 90: cost 1200, proceeds 1080. Flat, free is cash.
        let (flat, _) = execute_order(&carry(part, held), &margin(-12.0, 90.0), None).unwrap();
        assert_eq!(flat.cash, 960.0);
        assert_eq!(flat.position, 0.0);
        assert_eq!(flat.debt, 0.0);
        assert_eq!(flat.locked_cash, 0.0);
        assert_eq!(flat.free_cash, 960.0);
    }

    #[test]
    fn a_short_covered_higher_loses_the_difference() {
        let start = ProcessOrderState::flat(1000.0, 100.0);
        let (short, _) = execute_order(&start, &margin(-20.0, 100.0), None).unwrap();
        let (flat, _) = execute_order(&carry(short, start), &margin(20.0, 110.0), None).unwrap();
        assert_eq!(flat.cash, 800.0);
        assert_eq!(flat.free_cash, 800.0);
        assert_eq!((flat.debt, flat.locked_cash), (0.0, 0.0));
    }

    #[test]
    fn a_reversal_opens_with_what_closing_frees() {
        // All of the free cash is posted, yet selling everything can still
        // short: closing the long hands back its 500 of margin first, and
        // 1000 at 25 a unit opens 40.
        let held = margin_long();
        let order = Order {
            size_type: SizeType::Percent,
            ..margin(-1.0, 100.0)
        };
        let (state, result) = execute_order(&held, &order, None).unwrap();
        assert_eq!(result.size, 60.0);
        assert_eq!(state.position, -40.0);
        assert_eq!(state.cash, 1000.0);
        assert_eq!(state.debt, 4000.0);
        assert_eq!(state.locked_cash, 1000.0);
        assert_eq!(state.free_cash, 0.0);
    }

    #[test]
    fn an_opening_pays_margin_and_fees_from_free_cash() {
        let start = ProcessOrderState::flat(1000.0, 100.0);
        let order = Order {
            fees: 0.01,
            fixed_fees: 5.0,
            ..margin(f64::INFINITY, 100.0)
        };
        let (state, result) = execute_order(&start, &order, None).unwrap();
        // 25 of margin and 1 of fees a unit, after the fixed 5.
        assert_eq!(result.size, 995.0 / 26.0);
        assert_eq!(result.fees, result.size * 100.0 * 0.01 + 5.0);
        assert_eq!(state.cash, 1000.0 - result.fees);
        assert!(state.free_cash.abs() < 1e-9);
    }

    #[test]
    fn cash_limit_caps_what_an_opening_posts() {
        let start = ProcessOrderState::flat(1000.0, 100.0);
        let order = Order {
            cash_limit: 100.0,
            ..margin(f64::INFINITY, 100.0)
        };
        let (state, result) = execute_order(&start, &order, None).unwrap();
        assert_eq!(result.size, 4.0);
        assert_eq!(state.locked_cash, 100.0);
    }

    #[test]
    fn without_free_cash_an_opening_is_rejected_but_a_close_is_not() {
        let start = ProcessOrderState::flat(500.0, 100.0);
        let (full, _) = execute_order(&start, &margin(20.0, 100.0), None).unwrap();
        let full = carry(full, start);
        assert_eq!(full.free_cash, 0.0);
        let (_, more) = execute_order(&full, &margin(1.0, 100.0), None).unwrap();
        assert_eq!(more.status_info, Some(StatusInfo::NoCashLong));
        let (_, close) = execute_order(&full, &margin(-1.0, 100.0), None).unwrap();
        assert!(close.is_filled());

        // An account that owes, as a loss can leave it, is rejected rather
        // than stopped.
        let owing = ProcessOrderState::flat(-50.0, 100.0);
        let (_, short) = execute_order(&owing, &margin(-1.0, 100.0), None).unwrap();
        assert_eq!(short.status_info, Some(StatusInfo::NoCashShort));
    }

    #[test]
    fn a_one_sided_margin_column_only_closes_the_other_side() {
        let long_only = |size| Order {
            direction: Direction::LongOnly,
            ..margin(size, 100.0)
        };
        let flat = ProcessOrderState::flat(1000.0, 100.0);
        let (_, none) = execute_order(&flat, &long_only(-5.0), None).unwrap();
        assert_eq!(none.status_info, Some(StatusInfo::NoOpenPosition));
        let (state, result) = execute_order(&margin_long(), &long_only(-30.0), None).unwrap();
        assert_eq!(result.size, 20.0);
        assert_eq!(state.position, 0.0);
        let strict = Order {
            allow_partial: false,
            ..long_only(-30.0)
        };
        let (_, partial) = execute_order(&margin_long(), &strict, None).unwrap();
        assert_eq!(partial.status_info, Some(StatusInfo::PartialFill));

        // In a short-only column a negative size buys back, and there is
        // nothing to buy back.
        let short_only = Order {
            direction: Direction::ShortOnly,
            ..margin(-5.0, 100.0)
        };
        let (_, none) = execute_order(&flat, &short_only, None).unwrap();
        assert_eq!(none.status_info, Some(StatusInfo::NoOpenPosition));
    }

    #[test]
    fn unit_margin_is_cash_per_unit_and_a_nan_margin_is_one_over_leverage() {
        let start = ProcessOrderState::flat(1000.0, 100.0);
        let per_unit = Order {
            margin_basis: MarginBasis::Unit,
            initial_margin: 30.0,
            ..margin(10.0, 100.0)
        };
        let (state, _) = execute_order(&start, &per_unit, None).unwrap();
        assert_eq!(state.locked_cash, 300.0);
        assert_eq!(state.debt, 1000.0);
        assert_eq!(state.free_cash, 700.0);

        let from_leverage = Order {
            initial_margin: f64::NAN,
            leverage: 4.0,
            ..margin(20.0, 100.0)
        };
        let (state, _) = execute_order(&start, &from_leverage, None).unwrap();
        assert_eq!(state.locked_cash, 500.0);
    }

    #[test]
    fn margin_parameters_are_checked_only_on_margin_orders() {
        let start = ProcessOrderState::flat(1000.0, 100.0);
        for initial_margin in [0.0, -0.1, f64::INFINITY] {
            let order = Order {
                initial_margin,
                ..margin(1.0, 100.0)
            };
            assert!(execute_order(&start, &order, None).is_err());
            let cash = Order {
                account_type: AccountType::Cash,
                ..order
            };
            assert!(execute_order(&start, &cash, None).is_ok());
        }
        let unset = Order {
            margin_basis: MarginBasis::Unit,
            initial_margin: f64::NAN,
            ..margin(1.0, 100.0)
        };
        assert!(execute_order(&start, &unset, None).is_err());
    }

    #[test]
    fn sign_matches_numpy_for_zero_and_nan() {
        assert_eq!(sign(3.0), 1.0);
        assert_eq!(sign(-3.0), -1.0);
        assert_eq!(sign(0.0), 0.0);
        assert_eq!(sign(-0.0), 0.0);
        assert_eq!(sign(f64::NAN), 0.0);
        // Which matters because this is what it feeds:
        assert!((sign(0.0) * f64::INFINITY).is_nan());
    }

    #[test]
    fn a_price_below_zero_is_a_price_only_when_allowed() {
        let start = flat();
        let buy = Order {
            size: 1.0,
            price: -20.0,
            fees: 0.01,
            allow_negative_price: true,
            ..Default::default()
        };
        // Paid 20 to take it, less a fee on the price's size.
        let (state, result) = execute_order(&start, &buy, None).unwrap();
        assert_eq!(
            (state.cash, state.free_cash, state.position),
            (119.8, 119.8, 1.0)
        );
        assert_eq!((result.price, result.fees), (-20.0, 0.2));
        // Slippage is against the trade: a buy pays more, here less received.
        let slipped = Order {
            slippage: 0.1,
            fees: 0.0,
            ..buy
        };
        let (_, result) = execute_order(&start, &slipped, None).unwrap();
        assert_eq!(result.price, -18.0);
        assert!(
            execute_order(
                &start,
                &Order {
                    allow_negative_price: false,
                    ..buy
                },
                None
            )
            .is_err()
        );
        assert!(
            execute_order(
                &start,
                &Order {
                    size_type: SizeType::Percent,
                    ..buy
                },
                None
            )
            .is_err()
        );
        // A cash column cannot short below zero.
        assert!(execute_order(&start, &Order { size: -1.0, ..buy }, None).is_err());
        // A sell of what is held, at a price below 0, pays to be rid of it.
        let held = ProcessOrderState {
            position: 1.0,
            ..start
        };
        let sell = Order {
            size: -1.0,
            fees: 0.0,
            ..buy
        };
        let (state, _) = execute_order(&held, &sell, None).unwrap();
        assert_eq!((state.cash, state.position), (80.0, 0.0));
    }

    #[test]
    fn a_margin_position_below_zero_has_a_cost_basis_below_zero() {
        let start = flat();
        let short = Order {
            size: -2.0,
            price: -10.0,
            slippage: 0.1,
            account_type: AccountType::Margin,
            initial_margin: 0.1,
            allow_negative_price: true,
            ..Default::default()
        };
        // Sold at -11: margin on its size, 2 × 11 × 0.1.
        let (state, result) = execute_order(&start, &short, None).unwrap();
        assert_eq!(result.price, -11.0);
        assert_eq!((state.position, state.debt), (-2.0, -22.0));
        assert!((state.locked_cash - 2.2).abs() < 1e-12);
        assert!((state.free_cash - 97.8).abs() < 1e-12);
        // Bought back at -15: 4 a contract made on the short.
        let after = ProcessOrderState {
            cash: state.cash,
            position: state.position,
            debt: state.debt,
            locked_cash: state.locked_cash,
            free_cash: state.free_cash,
            ..start
        };
        let cover = Order {
            size: 2.0,
            price: -15.0,
            slippage: 0.0,
            ..short
        };
        let (state, _) = execute_order(&after, &cover, None).unwrap();
        assert_eq!(state.position, 0.0);
        assert!((state.cash - 108.0).abs() < 1e-9);
        assert!((state.free_cash - 108.0).abs() < 1e-9);
    }
}
