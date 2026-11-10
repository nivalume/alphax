//! Log records: the full before-and-after of one order.
//!
//! Order records say what happened. A log says what was *asked for* and what
//! the account looked like on both sides of the answer — which is the only
//! thing that answers the question people actually have when a backtest
//! surprises them: **why did this bar not trade?**
//!
//! Without logs the available evidence is that no order record exists, which
//! is consistent with a dozen causes: the size resolved to zero, the cash ran
//! out, the granularity floored it away, the direction forbade it, the price
//! was `NaN`. With a log the reason is a field.
//!
//! # Why this is off by default
//!
//! A log is 38 fields against an order record's 7, and it is written for every
//! order *considered* rather than every order filled — on a sparse strategy
//! that is twenty times as many rows at five times the width. On a
//! thousand-column sweep that is gigabytes for something nobody will read.
//!
//! So logging is per order, through the request's own `log` flag, and it is a
//! [`FlexArray`](alphax_core::flex::FlexArray) like every other parameter. The
//! useful shapes are: off everywhere (the default), on for one column being
//! investigated, or on for one row across a sweep.
//!
//! # The `id` fields
//!
//! `id` is the record's index, as everywhere else here. `order_id` points at
//! the order record this log produced, or `-1` when nothing filled — which is
//! the join between the two sets, and the reason a rejected order still gets
//! a row.
//!
//! # Two logs are rarely `==`
//!
//! [`LogRecords`] derives `PartialEq`, and it is a trap: a rejected order fills
//! nothing, so its `res_size`, `res_price` and `res_fees` are `NaN` — and
//! `NaN != NaN`. Any run that rejected anything therefore compares unequal to
//! *itself*. Compare the `Debug` rendering, or the fields that matter, when
//! what is wanted is "the same log twice".
//!
//! **`order_id` is relative to the buffer it was written into, and is re-based
//! when buffers merge.** That is the same rule `id` follows, and for the same
//! reason: a simulation that wrote global ids would need a shared counter, and
//! then the numbering would depend on how threads interleaved. [`Journal`]
//! owns both halves of a run, so [`Journal::concat`] is the one place that
//! knows a chunk's order offset — which is why merging goes through it rather
//! than concatenating the two record sets separately.

use alphax_core::error::Result;
use alphax_model::{OrderSide, OrderStatus, StatusInfo};

use crate::mapped::ColMap;

/// One order's full context, as the simulation saw it.
///
/// Built by the simulation loop and handed to [`LogSink::push_log`]. A struct
/// rather than forty-two arguments, because a positional call that long is
/// a swap waiting to happen — and `cash` and `free_cash` sitting next to each
/// other, both `f64`, is exactly the pair that would swap silently.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogEntry {
    /// Cash-sharing group the column belongs to.
    pub group: usize,
    /// Column.
    pub col: usize,
    /// Row.
    pub idx: usize,

    /// Cash before the order.
    pub cash: f64,
    /// Position before the order.
    pub position: f64,
    /// Debt before the order.
    pub debt: f64,
    /// Own cash the position tied up before the order.
    pub locked_cash: f64,
    /// Free cash before the order.
    pub free_cash: f64,
    /// Valuation price used to size the order.
    pub val_price: f64,
    /// Portfolio value the order was sized against.
    pub value: f64,

    /// The request, exactly as submitted.
    pub request: alphax_exec::Order,

    /// Cash after.
    pub new_cash: f64,
    /// Position after.
    pub new_position: f64,
    /// Debt after.
    pub new_debt: f64,
    /// Own cash the position ties up after.
    pub new_locked_cash: f64,
    /// Free cash after.
    pub new_free_cash: f64,
    /// Valuation price after, which `update_value` may have re-marked.
    pub new_val_price: f64,
    /// Portfolio value after.
    pub new_value: f64,

    /// What the kernel answered.
    pub result: alphax_exec::OrderResult,
    /// Index of the order record this produced, or `-1` if nothing filled.
    pub order_id: i64,
    /// Whether the order was a liquidation, placed by the exchange rather
    /// than by the strategy.
    pub liquidation: bool,
    /// The column's contract multiplier. Prices here are prices, as the
    /// strategy wrote them; cash and values are cash, so a fill of `size`
    /// at `price` moved `size × price × multiplier`.
    pub multiplier: f64,
}

/// Logs, struct-of-arrays, one row per order considered.
///
/// Field names and order follow a fixed schema, with leverage's four fields beside the ones
/// they extend: `locked_cash` after `debt`, the leverage request after the
/// granularity. `id` is not stored: it is the record's index.
#[derive(Debug, Clone, Default, PartialEq)]
#[allow(
    clippy::struct_field_names,
    reason = "the `new_` prefix is the schema's, and renaming it would break \
              the correspondence a log exists to support"
)]
pub struct LogRecords {
    /// Cash-sharing group.
    pub group: Vec<i64>,
    /// Column.
    pub col: Vec<i64>,
    /// Row.
    pub idx: Vec<i64>,

    /// Cash before.
    pub cash: Vec<f64>,
    /// Position before.
    pub position: Vec<f64>,
    /// Debt before.
    pub debt: Vec<f64>,
    /// Own cash the position tied up before.
    pub locked_cash: Vec<f64>,
    /// Free cash before.
    pub free_cash: Vec<f64>,
    /// Valuation price.
    pub val_price: Vec<f64>,
    /// Portfolio value.
    pub value: Vec<f64>,

    /// Requested size.
    pub req_size: Vec<f64>,
    /// Requested price.
    pub req_price: Vec<f64>,
    /// Requested size type, as its discriminant.
    pub req_size_type: Vec<i64>,
    /// Requested direction, as its discriminant.
    pub req_direction: Vec<i64>,
    /// Requested proportional fee.
    pub req_fees: Vec<f64>,
    /// Requested fixed fee.
    pub req_fixed_fees: Vec<f64>,
    /// Requested slippage.
    pub req_slippage: Vec<f64>,
    /// Requested minimum size.
    pub req_min_size: Vec<f64>,
    /// Requested maximum size.
    pub req_max_size: Vec<f64>,
    /// Requested size granularity.
    pub req_size_granularity: Vec<f64>,
    /// Requested leverage.
    pub req_leverage: Vec<f64>,
    /// Requested leverage mode, as its discriminant.
    pub req_leverage_mode: Vec<i64>,
    /// The column's account type, as its discriminant.
    pub req_account_type: Vec<i64>,
    /// Requested initial margin; `NaN` is `1 / leverage` of the notional.
    pub req_initial_margin: Vec<f64>,
    /// What the initial margin is measured in, as its discriminant.
    pub req_margin_basis: Vec<i64>,
    /// Requested random rejection probability.
    pub req_reject_prob: Vec<f64>,
    /// Whether cash was locked.
    pub req_lock_cash: Vec<bool>,
    /// Whether a partial fill was allowed.
    pub req_allow_partial: Vec<bool>,
    /// Whether a rejection was to be raised.
    pub req_raise_reject: Vec<bool>,
    /// Whether this order asked to be logged, which it did.
    pub req_log: Vec<bool>,

    /// Cash after.
    pub new_cash: Vec<f64>,
    /// Position after.
    pub new_position: Vec<f64>,
    /// Debt after.
    pub new_debt: Vec<f64>,
    /// Own cash the position ties up after.
    pub new_locked_cash: Vec<f64>,
    /// Free cash after.
    pub new_free_cash: Vec<f64>,
    /// Valuation price after.
    pub new_val_price: Vec<f64>,
    /// Portfolio value after.
    pub new_value: Vec<f64>,

    /// Filled size.
    pub res_size: Vec<f64>,
    /// Fill price.
    pub res_price: Vec<f64>,
    /// Fees charged.
    pub res_fees: Vec<f64>,
    /// Side, as its discriminant; `-1` when nothing filled.
    pub res_side: Vec<i64>,
    /// Status, as its discriminant.
    pub res_status: Vec<i64>,
    /// Why, as [`StatusInfo`]'s discriminant; `-1` when there is no reason to
    /// give.
    pub res_status_info: Vec<i64>,
    /// The order record this produced, or `-1`.
    pub order_id: Vec<i64>,
    /// Whether the order was a liquidation.
    pub liquidation: Vec<bool>,
    /// The column's contract multiplier.
    pub multiplier: Vec<f64>,
}

impl LogRecords {
    /// An empty set.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            group: Vec::new(),
            col: Vec::new(),
            idx: Vec::new(),
            cash: Vec::new(),
            position: Vec::new(),
            debt: Vec::new(),
            locked_cash: Vec::new(),
            free_cash: Vec::new(),
            val_price: Vec::new(),
            value: Vec::new(),
            req_size: Vec::new(),
            req_price: Vec::new(),
            req_size_type: Vec::new(),
            req_direction: Vec::new(),
            req_fees: Vec::new(),
            req_fixed_fees: Vec::new(),
            req_slippage: Vec::new(),
            req_min_size: Vec::new(),
            req_max_size: Vec::new(),
            req_size_granularity: Vec::new(),
            req_leverage: Vec::new(),
            req_leverage_mode: Vec::new(),
            req_account_type: Vec::new(),
            req_initial_margin: Vec::new(),
            req_margin_basis: Vec::new(),
            req_reject_prob: Vec::new(),
            req_lock_cash: Vec::new(),
            req_allow_partial: Vec::new(),
            req_raise_reject: Vec::new(),
            req_log: Vec::new(),
            new_cash: Vec::new(),
            new_position: Vec::new(),
            new_debt: Vec::new(),
            new_locked_cash: Vec::new(),
            new_free_cash: Vec::new(),
            new_val_price: Vec::new(),
            new_value: Vec::new(),
            res_size: Vec::new(),
            res_price: Vec::new(),
            res_fees: Vec::new(),
            res_side: Vec::new(),
            res_status: Vec::new(),
            res_status_info: Vec::new(),
            order_id: Vec::new(),
            liquidation: Vec::new(),
            multiplier: Vec::new(),
        }
    }

    /// Number of logged orders.
    #[must_use]
    pub fn len(&self) -> usize {
        self.col.len()
    }

    /// Whether nothing was logged.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.col.is_empty()
    }

    /// The `id` of each record: its index.
    #[must_use]
    pub fn ids(&self) -> Vec<i64> {
        (0..self.len() as i64).collect()
    }

    /// Groups these logs by column.
    ///
    /// # Errors
    ///
    /// Returns [`alphax_core::Error::Invariant`] if a log names a column outside
    /// `n_cols`.
    pub fn col_map(&self, n_cols: usize) -> Result<ColMap> {
        ColMap::build(&self.col, n_cols)
    }

    /// Appends one entry.
    pub fn push(&mut self, e: &LogEntry) {
        self.group.push(e.group as i64);
        self.col.push(e.col as i64);
        self.idx.push(e.idx as i64);

        self.cash.push(e.cash);
        self.position.push(e.position);
        self.debt.push(e.debt);
        self.locked_cash.push(e.locked_cash);
        self.free_cash.push(e.free_cash);
        self.val_price.push(e.val_price);
        self.value.push(e.value);

        let r = &e.request;
        self.req_size.push(r.size);
        self.req_price.push(r.price);
        self.req_size_type.push(r.size_type as i64);
        self.req_direction.push(r.direction as i64);
        self.req_fees.push(r.fees);
        self.req_fixed_fees.push(r.fixed_fees);
        self.req_slippage.push(r.slippage);
        self.req_min_size.push(r.min_size);
        self.req_max_size.push(r.max_size);
        self.req_size_granularity.push(r.size_granularity);
        self.req_leverage.push(r.leverage);
        self.req_leverage_mode.push(r.leverage_mode as i64);
        self.req_account_type.push(r.account_type as i64);
        self.req_initial_margin.push(r.initial_margin);
        self.req_margin_basis.push(r.margin_basis as i64);
        self.req_reject_prob.push(r.reject_prob);
        self.req_lock_cash.push(r.lock_cash);
        self.req_allow_partial.push(r.allow_partial);
        self.req_raise_reject.push(r.raise_reject);
        self.req_log.push(r.log);

        self.new_cash.push(e.new_cash);
        self.new_position.push(e.new_position);
        self.new_debt.push(e.new_debt);
        self.new_locked_cash.push(e.new_locked_cash);
        self.new_free_cash.push(e.new_free_cash);
        self.new_val_price.push(e.new_val_price);
        self.new_value.push(e.new_value);

        self.res_size.push(e.result.size);
        self.res_price.push(e.result.price);
        self.res_fees.push(e.result.fees);
        // `-1` rather than a default side: an order that did not fill has no
        // side, and picking one would read as a buy that never happened.
        self.res_side.push(e.result.side.map_or(-1, |s| s as i64));
        self.res_status.push(e.result.status as i64);
        self.res_status_info
            .push(e.result.status_info.map_or(-1, |s| s as i64));
        self.order_id.push(e.order_id);
        self.liquidation.push(e.liquidation);
        self.multiplier.push(e.multiplier);
    }

    /// Appends another set, preserving order.
    pub fn extend(&mut self, other: &Self) {
        macro_rules! join {
            ($($f:ident),+ $(,)?) => { $( self.$f.extend_from_slice(&other.$f); )+ };
        }
        join!(
            group,
            col,
            idx,
            cash,
            position,
            debt,
            locked_cash,
            free_cash,
            val_price,
            value,
            req_size,
            req_price,
            req_size_type,
            req_direction,
            req_fees,
            req_fixed_fees,
            req_slippage,
            req_min_size,
            req_max_size,
            req_size_granularity,
            req_leverage,
            req_leverage_mode,
            req_account_type,
            req_initial_margin,
            req_margin_basis,
            req_reject_prob,
            req_lock_cash,
            req_allow_partial,
            req_raise_reject,
            req_log,
            new_cash,
            new_position,
            new_debt,
            new_locked_cash,
            new_free_cash,
            new_val_price,
            new_value,
            res_size,
            res_price,
            res_fees,
            res_side,
            res_status,
            res_status_info,
            order_id,
            liquidation,
            multiplier,
        );
    }

    /// Merges sets that are already in column order.
    #[must_use]
    pub fn concat(parts: &[Self]) -> Self {
        let mut out = Self::new();
        for part in parts {
            out.extend(part);
        }
        out
    }

    /// Moves every row `by` rows later; see
    /// [`OrderRecords::shift_rows`](crate::OrderRecords::shift_rows).
    pub fn shift_rows(&mut self, by: i64) {
        if by != 0 {
            for idx in &mut self.idx {
                *idx += by;
            }
        }
    }

    /// The length of every column, to check they agree.
    pub(crate) fn lens(&self) -> Vec<usize> {
        macro_rules! lens {
            ($($f:ident),+ $(,)?) => { vec![$( self.$f.len(), )+] };
        }
        lens!(
            group,
            col,
            idx,
            cash,
            position,
            debt,
            locked_cash,
            free_cash,
            val_price,
            value,
            req_size,
            req_price,
            req_size_type,
            req_direction,
            req_fees,
            req_fixed_fees,
            req_slippage,
            req_min_size,
            req_max_size,
            req_size_granularity,
            req_leverage,
            req_leverage_mode,
            req_account_type,
            req_initial_margin,
            req_margin_basis,
            req_reject_prob,
            req_lock_cash,
            req_allow_partial,
            req_raise_reject,
            req_log,
            new_cash,
            new_position,
            new_debt,
            new_locked_cash,
            new_free_cash,
            new_val_price,
            new_value,
            res_size,
            res_price,
            res_fees,
            res_side,
            res_status,
            res_status_info,
            order_id,
            liquidation,
            multiplier,
        )
    }

    /// Reorders the rows: the `k`th becomes the one at `order[k]`.
    pub fn permute(&mut self, order: &[usize]) {
        macro_rules! reorder {
            ($($f:ident),+ $(,)?) => { $( self.$f = order.iter().map(|&i| self.$f[i]).collect(); )+ };
        }
        reorder!(
            group,
            col,
            idx,
            cash,
            position,
            debt,
            locked_cash,
            free_cash,
            val_price,
            value,
            req_size,
            req_price,
            req_size_type,
            req_direction,
            req_fees,
            req_fixed_fees,
            req_slippage,
            req_min_size,
            req_max_size,
            req_size_granularity,
            req_leverage,
            req_leverage_mode,
            req_account_type,
            req_initial_margin,
            req_margin_basis,
            req_reject_prob,
            req_lock_cash,
            req_allow_partial,
            req_raise_reject,
            req_log,
            new_cash,
            new_position,
            new_debt,
            new_locked_cash,
            new_free_cash,
            new_val_price,
            new_value,
            res_size,
            res_price,
            res_fees,
            res_side,
            res_status,
            res_status_info,
            order_id,
            liquidation,
            multiplier,
        );
    }

    /// Why each logged order ended the way it did, in words.
    ///
    /// The whole reason logs exist. `"filled"` for the ones that did, and the
    /// rejection's own reason for the ones that did not.
    #[must_use]
    pub fn reasons(&self) -> Vec<&'static str> {
        (0..self.len())
            .map(|i| {
                if self.res_status[i] == OrderStatus::Filled as i64 {
                    return "filled";
                }
                StatusInfo::from_discriminant(self.res_status_info[i])
                    .map_or("no reason recorded", StatusInfo::describe)
            })
            .collect()
    }

    /// How often each outcome occurred in one column.
    ///
    /// A sweep's first diagnostic: a column that traded far less than expected
    /// usually has one reason dominating this tally.
    ///
    /// # Errors
    ///
    /// As [`Self::col_map`].
    pub fn reason_counts(&self, col: usize, n_cols: usize) -> Result<Vec<(&'static str, usize)>> {
        let map = self.col_map(n_cols)?;
        let reasons = self.reasons();
        let mut tally: Vec<(&'static str, usize)> = Vec::new();
        for &i in map.column(col) {
            let reason = reasons[i];
            if let Some(slot) = tally.iter_mut().find(|(r, _)| *r == reason) {
                slot.1 += 1;
            } else {
                tally.push((reason, 1));
            }
        }
        tally.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        Ok(tally)
    }
}

/// Where logs are written, when an order asks to be logged.
///
/// Separate from [`crate::RecordSink`] and defaulted to a no-op, so a sink
/// that does not care about logs — which is most of them, and all of the
/// existing ones — needs no change.
pub trait LogSink {
    /// Records one order's full context.
    fn push_log(&mut self, entry: &LogEntry) {
        let _ = entry;
    }
}

impl LogSink for crate::OrderRecords {}

/// Orders and logs together, for a run that wants both.
///
/// The sink a caller passes when logging is on. Order records come out
/// identical to a run without logging — the logs are additional, not
/// alternative — which is what lets an investigation re-run a workload with
/// logging enabled and compare.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Journal {
    /// The fills.
    pub orders: crate::OrderRecords,
    /// The logged orders, filled or not.
    pub logs: LogRecords,
}

impl Journal {
    /// An empty journal.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            orders: crate::OrderRecords::new(),
            logs: LogRecords::new(),
        }
    }

    /// Appends another journal, re-basing its `order_id`s onto this one.
    ///
    /// The step that makes a merge correct rather than merely concatenated.
    /// Each buffer numbers its fills from zero, so `other`'s log rows point at
    /// indices in `other.orders`; after the merge those fills sit at an offset,
    /// and every non-negative `order_id` has to move with them. A `-1` stays
    /// `-1`: nothing filled, so there is nothing to point at.
    ///
    /// Getting this wrong is quiet. The arrays are the right length, every
    /// index is in range, and the join simply names a different fill — one
    /// belonging to an earlier group, on some other bar, with a plausible size.
    pub fn extend(&mut self, other: &Self) {
        let offset = self.orders.len() as i64;
        let first = self.logs.len();
        self.orders.extend(&other.orders);
        self.logs.extend(&other.logs);
        for id in &mut self.logs.order_id[first..] {
            if *id >= 0 {
                *id += offset;
            }
        }
    }

    /// Merges journals that are already in group order.
    #[must_use]
    pub fn concat(parts: &[Self]) -> Self {
        let mut out = Self::new();
        for part in parts {
            out.extend(part);
        }
        out
    }

    /// [`OrderRecords::chain`](crate::OrderRecords::chain) for a journal:
    /// the logs are put in the same order, and their `order_id`s follow
    /// the fills they name.
    #[must_use]
    pub fn chain(parts: &[Self], group_of: &[usize]) -> Self {
        let joined = Self::concat(parts);
        let orders = crate::OrderRecords::chain(std::slice::from_ref(&joined.orders), group_of);
        // Where each fill of the concatenation went.
        let order = crate::chain_order(&joined.orders.col, group_of);
        let mut moved = vec![0_i64; order.len()];
        for (new, &old) in order.iter().enumerate() {
            moved[old] = new as i64;
        }
        let mut logs = joined.logs;
        for id in &mut logs.order_id {
            if *id >= 0 {
                *id = moved[*id as usize];
            }
        }
        let order = crate::chain_order(&logs.col, group_of);
        logs.permute(&order);
        Self { orders, logs }
    }
}

impl crate::RecordSink for Journal {
    fn push_order(
        &mut self,
        row: usize,
        col: usize,
        size: f64,
        price: f64,
        fees: f64,
        side: OrderSide,
    ) {
        self.orders.push_order(row, col, size, price, fees, side);
    }

    fn push_cash(&mut self, row: usize, col: usize, amount: f64, kind: alphax_model::CashFlowKind) {
        crate::RecordSink::push_cash(&mut self.orders, row, col, amount, kind);
    }

    fn push_balances(&mut self, balances: crate::Balances) {
        self.orders.push_balances(balances);
    }
}

impl LogSink for Journal {
    fn push_log(&mut self, entry: &LogEntry) {
        self.logs.push(entry);
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp, reason = "a log must record exactly what happened")]
mod tests {
    use super::*;
    use crate::RecordSink;
    use alphax_exec::{Order, OrderResult};
    use alphax_model::{AccountType, Direction, LeverageMode, MarginBasis, SizeType};

    fn request() -> Order {
        Order {
            size: 10.0,
            price: 100.0,
            size_type: SizeType::Amount,
            direction: Direction::Both,
            fees: 0.001,
            fixed_fees: 0.0,
            slippage: 0.0,
            min_size: 0.0,
            max_size: f64::INFINITY,
            size_granularity: f64::NAN,
            leverage: 1.0,
            leverage_mode: LeverageMode::Lazy,
            account_type: AccountType::Cash,
            initial_margin: f64::NAN,
            margin_basis: MarginBasis::Notional,
            cash_limit: f64::INFINITY,
            reject_prob: 0.0,
            lock_cash: false,
            allow_partial: true,
            raise_reject: false,
            log: true,
            allow_negative_price: false,
        }
    }

    fn entry(col: usize, idx: usize, result: OrderResult, order_id: i64) -> LogEntry {
        LogEntry {
            group: col,
            col,
            idx,
            cash: 1000.0,
            position: 0.0,
            debt: 0.0,
            locked_cash: 0.0,
            free_cash: 1000.0,
            val_price: 100.0,
            value: 1000.0,
            request: request(),
            new_cash: 0.0,
            new_position: 10.0,
            new_debt: 0.0,
            new_locked_cash: 0.0,
            new_free_cash: 0.0,
            new_val_price: 100.0,
            new_value: 1000.0,
            result,
            order_id,
            liquidation: false,
            multiplier: 1.0,
        }
    }

    fn filled() -> OrderResult {
        OrderResult {
            size: 10.0,
            price: 100.0,
            fees: 1.0,
            side: Some(OrderSide::Buy),
            status: OrderStatus::Filled,
            status_info: None,
        }
    }

    fn rejected(info: StatusInfo) -> OrderResult {
        OrderResult {
            size: f64::NAN,
            price: f64::NAN,
            fees: f64::NAN,
            side: None,
            status: OrderStatus::Rejected,
            status_info: Some(info),
        }
    }

    #[test]
    fn a_log_records_both_sides_of_the_order() {
        let mut logs = LogRecords::new();
        logs.push(&entry(0, 3, filled(), 7));
        assert_eq!(logs.len(), 1);
        assert_eq!(logs.col, [0]);
        assert_eq!(logs.idx, [3]);
        assert_eq!(logs.cash, [1000.0]);
        assert_eq!(logs.new_cash, [0.0]);
        assert_eq!(logs.position, [0.0]);
        assert_eq!(logs.new_position, [10.0]);
        assert_eq!(logs.req_size, [10.0]);
        assert_eq!(logs.res_size, [10.0]);
        assert_eq!(logs.order_id, [7], "the join back to the order record");
    }

    #[test]
    fn a_rejected_order_still_gets_a_row_and_that_is_the_point() {
        // Without it the only evidence is that no order record exists, which
        // is consistent with a dozen different causes.
        let mut logs = LogRecords::new();
        logs.push(&entry(0, 1, rejected(StatusInfo::NoCashLong), -1));
        assert_eq!(logs.len(), 1);
        assert_eq!(logs.order_id, [-1], "nothing filled, so nothing to join to");
        assert_eq!(logs.res_side, [-1], "an unfilled order has no side");
        assert_eq!(logs.res_status, [OrderStatus::Rejected as i64]);
        assert_eq!(logs.res_status_info, [StatusInfo::NoCashLong as i64]);
        assert_eq!(logs.reasons(), [StatusInfo::NoCashLong.describe()]);
    }

    #[test]
    fn an_unfilled_side_is_minus_one_not_a_default() {
        // A default of `Buy` would read as a purchase that never happened,
        // and it would look entirely ordinary in a table.
        let mut logs = LogRecords::new();
        logs.push(&entry(0, 0, rejected(StatusInfo::SizeNaN), -1));
        logs.push(&entry(0, 1, filled(), 0));
        assert_eq!(logs.res_side, [-1, OrderSide::Buy as i64]);
    }

    #[test]
    fn the_reasons_tally_is_the_first_diagnostic() {
        // A column that traded far less than expected almost always has one
        // reason dominating.
        let mut logs = LogRecords::new();
        for i in 0..5 {
            logs.push(&entry(0, i, rejected(StatusInfo::NoCashLong), -1));
        }
        logs.push(&entry(0, 5, rejected(StatusInfo::SizeZero), -1));
        logs.push(&entry(0, 6, filled(), 0));
        // A second column, which must not be counted into the first's tally.
        logs.push(&entry(1, 0, rejected(StatusInfo::SizeNaN), -1));

        let tally = logs.reason_counts(0, 2).unwrap();
        assert_eq!(tally[0], (StatusInfo::NoCashLong.describe(), 5));
        assert_eq!(tally.len(), 3, "three distinct outcomes in column 0");
        assert!(tally.iter().any(|(r, n)| *r == "filled" && *n == 1));

        let other = logs.reason_counts(1, 2).unwrap();
        assert_eq!(other.len(), 1);
        assert_eq!(other[0].1, 1);
    }

    #[test]
    fn a_journal_writes_both_and_the_orders_are_unaffected() {
        // Logging is additional, not alternative: the same workload run with
        // logs on must produce the same fills, or an investigation would be
        // investigating a different run.
        let mut journal = Journal::new();
        journal.push_order(0, 0, 10.0, 100.0, 1.0, OrderSide::Buy);
        journal.push_log(&entry(0, 0, filled(), 0));
        assert_eq!(journal.orders.len(), 1);
        assert_eq!(journal.logs.len(), 1);

        let mut plain = crate::OrderRecords::new();
        plain.push_order(0, 0, 10.0, 100.0, 1.0, OrderSide::Buy);
        // The default `push_log` is a no-op, so a plain sink is unchanged.
        plain.push_log(&entry(0, 0, filled(), 0));
        assert_eq!(journal.orders, plain);
    }

    #[test]
    fn concatenating_keeps_column_order() {
        let mut a = LogRecords::new();
        a.push(&entry(0, 0, filled(), 0));
        let mut b = LogRecords::new();
        b.push(&entry(1, 0, filled(), 1));
        b.push(&entry(1, 1, rejected(StatusInfo::SizeZero), -1));
        let merged = LogRecords::concat(&[a, b]);
        assert_eq!(merged.len(), 3);
        assert_eq!(merged.col, [0, 1, 1]);
        assert_eq!(merged.ids(), [0, 1, 2]);
        assert!(merged.col_map(2).unwrap().is_contiguous());
    }

    #[test]
    fn every_field_of_the_schema_is_carried() {
        // The log's value is that nothing about the order is missing from it,
        // so the count is asserted rather than trusted. 38 base fields,
        // less `id`, which is the record's index here as everywhere, plus
        // leverage's four, margin's three, the liquidation flag and the
        // contract multiplier.
        let mut logs = LogRecords::new();
        logs.push(&entry(0, 0, filled(), 0));
        let widths = [
            logs.group.len(),
            logs.col.len(),
            logs.idx.len(),
            logs.cash.len(),
            logs.position.len(),
            logs.debt.len(),
            logs.locked_cash.len(),
            logs.free_cash.len(),
            logs.val_price.len(),
            logs.value.len(),
            logs.req_size.len(),
            logs.req_price.len(),
            logs.req_size_type.len(),
            logs.req_direction.len(),
            logs.req_fees.len(),
            logs.req_fixed_fees.len(),
            logs.req_slippage.len(),
            logs.req_min_size.len(),
            logs.req_max_size.len(),
            logs.req_size_granularity.len(),
            logs.req_leverage.len(),
            logs.req_leverage_mode.len(),
            logs.req_account_type.len(),
            logs.req_initial_margin.len(),
            logs.req_margin_basis.len(),
            logs.req_reject_prob.len(),
            logs.req_lock_cash.len(),
            logs.req_allow_partial.len(),
            logs.req_raise_reject.len(),
            logs.req_log.len(),
            logs.new_cash.len(),
            logs.new_position.len(),
            logs.new_debt.len(),
            logs.new_locked_cash.len(),
            logs.new_free_cash.len(),
            logs.new_val_price.len(),
            logs.new_value.len(),
            logs.res_size.len(),
            logs.res_price.len(),
            logs.res_fees.len(),
            logs.res_side.len(),
            logs.res_status.len(),
            logs.res_status_info.len(),
            logs.order_id.len(),
            logs.liquidation.len(),
            logs.multiplier.len(),
        ];
        assert_eq!(
            widths.len(),
            46,
            "the 38 base fields, less the index, plus leverage's four, margin's three, \
             the liquidation flag and the contract multiplier"
        );
        assert!(widths.iter().all(|&w| w == 1), "a field was not written");
    }
}
