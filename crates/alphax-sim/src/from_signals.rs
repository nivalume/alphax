//! `simulate_from_signals`: signals in, fills out.
//!
//! The pipeline per bar, in this order, because each stage narrows what the
//! next one has to consider:
//!
//! ```text
//! resolve prices ──▶ test stops ──▶ signals ──▶ conflicts ──▶ size ──▶ execute ──▶ re-arm stops
//! ```
//!
//! A fired stop replaces the input signals rather than competing with them:
//! it is not an opinion about the market, it is a position limit being hit.
//!
//! # Shared accounts
//!
//! Resolve all symbols' prices and signals before executing the group's orders.
//! Each fill updates shared cash; positions and stops remain per symbol.
//! Groups, rather than symbols within an account, are the parallel unit.
//!
//! # State that is not carried
//!
//! The bar index at which each stop was set and the stop-loss anchor price at
//! the time the position opened are not tracked in this loop: nothing reads
//! them here, so carrying them would be state no fixture could observe.
//!
//! # Guards that look redundant
//!
//! Earlier mutation tests found checks that did not change ungrouped fills.
//! They remain part of the pipeline:
//!
//! - **Clearing the stops when a position closes.** [`stop_price`] returns
//!   `NaN` while flat, and opening a position always re-anchors, so a stale
//!   stop can never fire. Clearing is belt and braces.
//! - **Resolving conflicts only when an entry fired.** All three resolvers
//!   guard on an entry being present, so the outer check is an optimization,
//!   not a semantic — a real one on sparse signals, where most bars have
//!   nothing at all.
//! - **Testing `use_stops` before acting on a trigger.** The trigger is only
//!   ever assigned inside a block that already tested it.
//!
//! Account valuation includes every held symbol before ordering the fills;
//! the value and its fill updates are also observable in account logs.

use alphax_core::error::{Error, Result};
use alphax_core::flex::FlexArray;
use alphax_exec::{
    AccountType, Direction, LeverageMode, MarginBasis, MarginMode, Order, OrderStatus,
    ProcessOrderState, SizeType, ValidatedOrder, execute_validated,
};
use alphax_model::{
    AccumulationMode, ConflictMode, DeltaFormat, DirectionConflictMode, LimitOrderPrice,
    OppositeEntryMode, OrderType, PendingConflictMode, PriceAreaVioMode, PriceType, StopEntryPrice,
    StopExitMode, StopExitPrice, StopLadderMode, StopUpdateMode,
};
use alphax_records::{Balances, Journal, LogEntry, LogSink, OrderRecords, RecordSink, SinkBuffer};

use crate::callbacks::{Context, Decision, Flow, NoCallbacks, SignalFunc};
use crate::from_orders::updated_value;
use crate::groups::{CallSeq, Groups, approx_cash_need, approx_order_value};
use crate::signals::{
    Signals, from_stop, resolve_dir_conflict, resolve_opposite_entry, resolve_pending_conflict,
    resolve_signal_conflict, size_signals,
};
use crate::snapshot::{GroupState, Loop, Run, SimState};
use crate::state::{ColumnState, LimitOrder, StopState};

/// Inputs to [`simulate_from_signals`].
///
/// Two calling conventions are supported and can be mixed. The
/// *direction* convention is `entries` / `exits` read through `direction`;
/// the *long/short* convention is four already-direction-aware arrays. Where
/// both fire they are OR-ed together — which is why `entries` in a two-sided
/// column becomes a long entry and `exits` becomes a **short entry**, not a
/// long exit.
#[derive(Debug, Clone)]
pub struct FromSignals<'a> {
    /// Number of bars.
    pub n_rows: usize,
    /// Number of columns: parameter combinations times symbols.
    pub n_cols: usize,
    /// Starting cash; a shared group uses its first column's value once.
    pub init_cash: FlexArray<'a, f64>,
    /// Units held before the first bar, per column (negative: short); 0 is
    /// none. See `crate::initial`.
    pub init_position: FlexArray<'a, f64>,
    /// What the initial position was opened at; `NaN` is the first close.
    pub init_price: FlexArray<'a, f64>,
    /// Whether prices of 0 or below are prices, as a futures contract's can
    /// be: see [`alphax_exec::Order::allow_negative_price`]. Off by default.
    pub allow_negative_price: bool,

    /// Bar opens. Used for stop gaps and for the `-inf` price sentinel.
    pub open: FlexArray<'a, f64>,
    /// Bar highs. Used to test stops and to trail them.
    pub high: FlexArray<'a, f64>,
    /// Bar lows.
    pub low: FlexArray<'a, f64>,
    /// Bar closes.
    pub close: FlexArray<'a, f64>,

    /// Direction-agnostic entries, read through `direction`.
    pub entries: FlexArray<'a, bool>,
    /// Direction-agnostic exits.
    pub exits: FlexArray<'a, bool>,
    /// How to read `entries` and `exits`.
    pub direction: FlexArray<'a, Direction>,
    /// Direction-aware long entries.
    pub long_entries: FlexArray<'a, bool>,
    /// Direction-aware long exits.
    pub long_exits: FlexArray<'a, bool>,
    /// Direction-aware short entries.
    pub short_entries: FlexArray<'a, bool>,
    /// Direction-aware short exits.
    pub short_exits: FlexArray<'a, bool>,

    /// Requested size. `inf` means "as much as possible".
    pub size: FlexArray<'a, f64>,
    /// How to interpret it: `Amount`, `Value` or `Percent` only.
    pub size_type: FlexArray<'a, SizeType>,
    /// Order price. `+inf` is this bar's close, `-inf` its open.
    pub price: FlexArray<'a, f64>,
    /// Whether an order executes at `price`, or at the bar's open or close.
    /// See `crate::prices`.
    pub price_type: FlexArray<'a, PriceType>,
    /// How many bars before the one an order executes on it was decided:
    /// its signals, size, size type and an explicit price are read there.
    /// Read on the executing bar, per column or per bar: a bar whose
    /// `from_ago` reaches before the first row executes nothing.
    pub from_ago: FlexArray<'a, usize>,
    /// Where given, what a sell executes at in place of the order price.
    pub bid_price: FlexArray<'a, f64>,
    /// Where given, what a buy executes at in place of the order price.
    pub ask_price: FlexArray<'a, f64>,
    /// What to do with an order price outside the bar's low–high range.
    pub price_area_vio_mode: FlexArray<'a, PriceAreaVioMode>,
    /// Bars that close the column's position at the order price, in place
    /// of any other order on them: the end of a trading session. A stop
    /// that triggers on the bar still fills first, at its level; otherwise
    /// the bar's signals and a resting limit order are dropped and the stops
    /// cleared. A flat column does nothing.
    pub force_exit: FlexArray<'a, bool>,
    /// Proportional fee rate.
    pub fees: FlexArray<'a, f64>,
    /// Flat fee per fill.
    pub fixed_fees: FlexArray<'a, f64>,
    /// Proportional slippage.
    pub slippage: FlexArray<'a, f64>,
    /// Reject fills below this size.
    pub min_size: FlexArray<'a, f64>,
    /// Clamp fills to this size.
    pub max_size: FlexArray<'a, f64>,
    /// Round sizes down to a multiple of this.
    pub size_granularity: FlexArray<'a, f64>,
    /// Buying power per unit of own cash, 1 or more. `1` borrows nothing.
    /// See [`crate::FromOrders::leverage`].
    pub leverage: FlexArray<'a, f64>,
    /// When a leveraged long borrows: as late as possible or as much as
    /// possible.
    pub leverage_mode: FlexArray<'a, LeverageMode>,
    /// Whether a column pays for its trades in full or posts margin, per
    /// column. See [`crate::FromOrders::account_type`].
    pub account_type: FlexArray<'a, AccountType>,
    /// Initial margin per unit a margin column opens, in `margin_basis`.
    /// `NaN` is `1 / leverage` of the notional.
    pub initial_margin: FlexArray<'a, f64>,
    /// Whether `initial_margin` is a fraction of the notional or cash per
    /// unit of size.
    pub margin_basis: FlexArray<'a, MarginBasis>,
    /// What one unit of position is worth per unit of price, per column.
    /// See [`crate::FromOrders::multiplier`]. Stops are levels of the price,
    /// so a multiplier does not move them.
    pub multiplier: FlexArray<'a, f64>,
    /// The most cash one buy may spend, fees included. `inf` is no cap.
    pub cash_limit: FlexArray<'a, f64>,
    /// Probability that an order is rejected at random, in `[0, 1]`.
    pub reject_prob: FlexArray<'a, f64>,
    /// Seed for the `reject_prob` draws. Each draw depends on the seed, the
    /// bar and the column only, so a run is reproducible on any thread count.
    pub seed: u64,
    /// Whether to reserve cash to cover an existing short.
    pub lock_cash: FlexArray<'a, bool>,
    /// Whether a partial fill is acceptable.
    pub allow_partial: FlexArray<'a, bool>,
    /// Whether a rejection should abort the simulation.
    pub raise_reject: FlexArray<'a, bool>,

    /// Whether each order records a full before-and-after log.
    ///
    /// Off everywhere by default, and per order rather than per run: a log is
    /// 38 fields against an order record's 7, so a sweep with this on
    /// everywhere produces gigabytes nobody reads. The useful shapes are one
    /// column under investigation, or one row across a sweep.
    ///
    /// **A row is written for every order that reaches the execution kernel,
    /// filled or not — and only for those.** A bar whose signals resolved to no
    /// order at all gets no row, because no order was considered. That is the
    /// rule, and it is what makes the log answer "why did this
    /// signal not trade" rather than "why was this bar quiet".
    pub log: FlexArray<'a, bool>,

    /// Whether `size` is the order, as in `from_orders`, instead of the
    /// signals: each bar's `size` in `size_type`, on `direction`'s sides, is
    /// an order, `NaN` or zero none, and the signal arrays are not read.
    /// Stops, limit orders and the price options apply to it as to a
    /// signal's order.
    pub order_mode: bool,
    /// Whether a signal may resize an existing position.
    pub accumulate: FlexArray<'a, AccumulationMode>,
    /// How to resolve a long entry and a long exit on one bar.
    pub upon_long_conflict: FlexArray<'a, ConflictMode>,
    /// How to resolve a short entry and a short exit on one bar.
    pub upon_short_conflict: FlexArray<'a, ConflictMode>,
    /// How to resolve a long entry and a short entry on one bar.
    pub upon_dir_conflict: FlexArray<'a, DirectionConflictMode>,
    /// How to treat an entry against the position held.
    pub upon_opposite_entry: FlexArray<'a, OppositeEntryMode>,

    /// Stop-loss distance, as a fraction. `NaN` disables it.
    pub sl_stop: FlexArray<'a, f64>,
    /// Whether the stop-loss trails the best price reached.
    pub sl_trail: FlexArray<'a, bool>,
    /// Take-profit distance, as a fraction. `NaN` disables it.
    pub tp_stop: FlexArray<'a, f64>,
    /// Which price a stop is anchored to when the position opens.
    pub stop_entry_price: FlexArray<'a, StopEntryPrice>,
    /// Which price a stop order fills at.
    pub stop_exit_price: FlexArray<'a, StopExitPrice>,
    /// Whether a stop closes or reverses the position.
    pub upon_stop_exit: FlexArray<'a, StopExitMode>,
    /// Whether adding to a position re-arms its stops.
    pub upon_stop_update: FlexArray<'a, StopUpdateMode>,
    /// Whether a signal's order executes on its bar or rests as a limit
    /// order until the market reaches it. Stop exits are always market
    /// orders. See [`crate::limit`].
    pub order_type: FlexArray<'a, OrderType>,
    /// How far from the order price a limit order rests, in `delta_format`,
    /// read on the bar it is placed. `NaN` is at the order price.
    pub limit_delta: FlexArray<'a, f64>,
    /// How `limit_delta`, `stop_limit_delta` and the stop distances
    /// (`sl_stop`, `tsl_stop`, `tsl_th`, `tp_stop`) are read, on the bar an
    /// order is placed or a position opened.
    pub delta_format: FlexArray<'a, DeltaFormat>,
    /// Whether a limit order fills with the market moving through it the
    /// other way: a buy at or above it, a sell at or below. A breakout entry.
    pub limit_reverse: FlexArray<'a, bool>,
    /// What a limit order fills at once the market reaches it.
    pub limit_order_price: FlexArray<'a, LimitOrderPrice>,
    /// Whether a limit order is first tested on the bar after the one it is
    /// placed on, even when the price already reaches it.
    pub limit_delay: FlexArray<'a, bool>,
    /// How many bars after the one it is placed on a limit order may fill.
    /// `NaN` is until it fills or is cancelled.
    pub limit_tif: FlexArray<'a, f64>,
    /// The row a limit order is cancelled at, before that bar is tested.
    /// `NaN` is none.
    pub limit_expiry: FlexArray<'a, f64>,
    /// What a signal on the same side as a resting limit order does to it,
    /// and whether it executes. See [`PendingConflictMode`].
    pub upon_adj_limit_conflict: FlexArray<'a, PendingConflictMode>,
    /// The same for a signal on the other side.
    pub upon_opp_limit_conflict: FlexArray<'a, PendingConflictMode>,
    /// What a signal on the same side as a stop that fired on its bar does
    /// to the stop, and whether it executes. `KeepExecute` fills the stop,
    /// then the signal re-sized against what the stop left.
    pub upon_adj_stop_conflict: FlexArray<'a, PendingConflictMode>,
    /// The same for a signal on the other side, such as a short entry on the
    /// bar a long is stopped out.
    pub upon_opp_stop_conflict: FlexArray<'a, PendingConflictMode>,
    /// A trailing stop's distance from the best price since the entry, as a
    /// fraction. `NaN` disables it. See `crate::stops`.
    pub tsl_stop: FlexArray<'a, f64>,
    /// How far the price must first move in the position's favour, as a
    /// fraction of the anchor, for the trailing stop to be live. `NaN` or 0:
    /// at once.
    pub tsl_th: FlexArray<'a, f64>,
    /// Exit at the open this many bars after the entry bar. `NaN` disables it.
    pub td_stop: FlexArray<'a, f64>,
    /// Exit at the open of this row. `NaN` disables it.
    pub dt_stop: FlexArray<'a, f64>,
    /// Whether `sl_stop`, `tsl_stop` and `tp_stop` are ladders, their rows
    /// steps, and how a step's share is sized. See `crate::stops`.
    pub stop_ladder: StopLadderMode,
    /// Whether a fired stop exits at market or places a limit order at its
    /// level (a time stop's: the open), `stop_limit_delta` worse (a
    /// stop-limit order).
    pub stop_order_type: FlexArray<'a, OrderType>,
    /// How far beyond a stop's level its limit order rests, in
    /// `delta_format`. `NaN` is at the level.
    pub stop_limit_delta: FlexArray<'a, f64>,
    /// Whether to run the stop engine at all.
    pub use_stops: bool,

    /// Valuation price. `+inf` is the order price, `-inf` the previous close.
    pub val_price: FlexArray<'a, f64>,
    /// Cash added (positive) or withdrawn (negative) at each bar's open,
    /// read from the group's first column. See [`crate::cash`].
    pub cash_deposits: FlexArray<'a, f64>,
    /// Dividend per share, paid at each bar's open on the position held
    /// before the bar's orders. A short position pays it.
    pub cash_dividends: FlexArray<'a, f64>,
    /// Cash credited (positive) or charged (negative) at each bar's close,
    /// per column. Booked on quiet bars too.
    pub cash_earnings: FlexArray<'a, f64>,
    /// A perpetual's funding rate per bar, charged at the bar's close on
    /// quiet bars too. See [`crate::FromOrders::funding_rate`].
    pub funding_rate: FlexArray<'a, f64>,
    /// Overnight financing per bar on a long. See
    /// [`crate::FromOrders::long_financing_rate`].
    pub long_financing_rate: FlexArray<'a, f64>,
    /// Overnight financing per bar on a short.
    pub short_financing_rate: FlexArray<'a, f64>,
    /// Interest per bar on what a cash column's long borrowed.
    pub debt_rate: FlexArray<'a, f64>,
    /// The fee per bar for the shares a cash column's short borrowed.
    pub borrow_rate: FlexArray<'a, f64>,
    /// The price the holding charges are computed at, and liquidation is
    /// tested at. `NaN` is the close for the charges and the bar's adverse
    /// extreme for the test.
    pub mark_price: FlexArray<'a, f64>,
    /// A margin column's maintenance margin, in `margin_basis`. `NaN` or zero
    /// carries none. See [`crate::FromOrders::maintenance_margin`].
    pub maintenance_margin: FlexArray<'a, f64>,
    /// The maintenance margin of a cash column, as a fraction of the
    /// position's notional below 1: the equity a margin-lending account
    /// (borrowing cash under `leverage`, or shares to short) must keep. Below
    /// it the account's cash columns that carry one are liquidated as margin
    /// columns are. `NaN` or zero, the default, carries none: a cash column
    /// is then collateral and never closed. See `crate::liquidation`.
    pub cash_maintenance_margin: FlexArray<'a, f64>,
    /// A tier table of maintenance margins by notional, replacing
    /// `maintenance_margin` where given: the rows are tiers, `NaN` after the
    /// last; `margin_tier_rate[t]` applies up to `margin_tier_notional[t]`
    /// (the last tier's bound is ignored). Margin columns with a notional
    /// basis only. See `crate::liquidation`.
    pub margin_tier_notional: FlexArray<'a, f64>,
    /// The tiers' maintenance margin rates. See `margin_tier_notional`.
    pub margin_tier_rate: FlexArray<'a, f64>,
    /// The fraction of a position each breach closes, in `(0, 1]`; `NaN`,
    /// the default, closes it all. See `crate::liquidation`.
    pub liquidation_step: FlexArray<'a, f64>,
    /// Bars on which an exchange deleverages: a margin column's position is
    /// closed at the mark price (the close without one), with no fee. See
    /// `crate::liquidation`.
    pub auto_deleverage: FlexArray<'a, bool>,
    /// The state to continue from, on its `next_row`, in place of
    /// `init_cash` and `init_position`; see [`crate::snapshot`]. `None`
    /// starts a run.
    pub resume: Option<&'a crate::snapshot::SimState>,
    /// The run's row the arrays' first row is: they cover rows
    /// `row_offset..row_offset + n_rows`. Records and the state are written
    /// on the run's rows. 0 unless a long history runs in pieces.
    pub row_offset: usize,
    /// What a liquidation fill pays, as a fraction of its notional.
    pub liquidation_fee: FlexArray<'a, f64>,
    /// Whether a margin column draws on the account's margin or only its
    /// own, per column.
    pub margin_mode: FlexArray<'a, MarginMode>,
    /// Carry the last non-`NaN` valuation price forward.
    pub ffill_val_price: bool,
    /// Mark to market after each fill rather than only at the bar's start.
    pub update_value: bool,
    /// Cash-sharing groups. None keeps the independent-column default.
    pub groups: Option<Groups>,
    /// Execution order within each group.
    pub call_seq: CallSeq<'a>,
}

impl Default for FromSignals<'_> {
    fn default() -> Self {
        Self {
            n_rows: 0,
            n_cols: 0,
            init_cash: FlexArray::Scalar(100.0),
            init_position: FlexArray::Scalar(0.0),
            init_price: FlexArray::Scalar(f64::NAN),
            allow_negative_price: false,
            open: FlexArray::Scalar(f64::NAN),
            high: FlexArray::Scalar(f64::NAN),
            low: FlexArray::Scalar(f64::NAN),
            close: FlexArray::Scalar(f64::NAN),
            entries: FlexArray::Scalar(false),
            exits: FlexArray::Scalar(false),
            direction: FlexArray::Scalar(Direction::LongOnly),
            long_entries: FlexArray::Scalar(false),
            long_exits: FlexArray::Scalar(false),
            short_entries: FlexArray::Scalar(false),
            short_exits: FlexArray::Scalar(false),
            size: FlexArray::Scalar(f64::INFINITY),
            size_type: FlexArray::Scalar(SizeType::Amount),
            price: FlexArray::Scalar(f64::INFINITY),
            price_type: FlexArray::Scalar(PriceType::Price),
            from_ago: FlexArray::Scalar(0),
            bid_price: FlexArray::Scalar(f64::NAN),
            ask_price: FlexArray::Scalar(f64::NAN),
            price_area_vio_mode: FlexArray::Scalar(PriceAreaVioMode::Ignore),
            force_exit: FlexArray::Scalar(false),
            fees: FlexArray::Scalar(0.0),
            fixed_fees: FlexArray::Scalar(0.0),
            slippage: FlexArray::Scalar(0.0),
            min_size: FlexArray::Scalar(crate::DEFAULT_MIN_SIZE),
            max_size: FlexArray::Scalar(f64::INFINITY),
            size_granularity: FlexArray::Scalar(f64::NAN),
            leverage: FlexArray::Scalar(1.0),
            leverage_mode: FlexArray::Scalar(LeverageMode::Lazy),
            account_type: FlexArray::Scalar(AccountType::Cash),
            initial_margin: FlexArray::Scalar(f64::NAN),
            margin_basis: FlexArray::Scalar(MarginBasis::Notional),
            multiplier: FlexArray::Scalar(1.0),
            cash_limit: FlexArray::Scalar(f64::INFINITY),
            reject_prob: FlexArray::Scalar(0.0),
            seed: 0,
            lock_cash: FlexArray::Scalar(false),
            allow_partial: FlexArray::Scalar(true),
            raise_reject: FlexArray::Scalar(false),
            log: FlexArray::Scalar(false),
            order_mode: false,
            accumulate: FlexArray::Scalar(AccumulationMode::Disabled),
            upon_long_conflict: FlexArray::Scalar(ConflictMode::Ignore),
            upon_short_conflict: FlexArray::Scalar(ConflictMode::Ignore),
            upon_dir_conflict: FlexArray::Scalar(DirectionConflictMode::Ignore),
            upon_opposite_entry: FlexArray::Scalar(OppositeEntryMode::ReverseReduce),
            sl_stop: FlexArray::Scalar(f64::NAN),
            sl_trail: FlexArray::Scalar(false),
            tp_stop: FlexArray::Scalar(f64::NAN),
            stop_entry_price: FlexArray::Scalar(StopEntryPrice::Close),
            stop_exit_price: FlexArray::Scalar(StopExitPrice::StopLimit),
            upon_stop_exit: FlexArray::Scalar(StopExitMode::Close),
            upon_stop_update: FlexArray::Scalar(StopUpdateMode::Override),
            order_type: FlexArray::Scalar(OrderType::Market),
            limit_delta: FlexArray::Scalar(f64::NAN),
            delta_format: FlexArray::Scalar(DeltaFormat::Percent),
            limit_reverse: FlexArray::Scalar(false),
            limit_order_price: FlexArray::Scalar(LimitOrderPrice::Limit),
            limit_delay: FlexArray::Scalar(false),
            limit_tif: FlexArray::Scalar(f64::NAN),
            limit_expiry: FlexArray::Scalar(f64::NAN),
            upon_adj_limit_conflict: FlexArray::Scalar(PendingConflictMode::KeepIgnore),
            upon_opp_limit_conflict: FlexArray::Scalar(PendingConflictMode::CancelExecute),
            upon_adj_stop_conflict: FlexArray::Scalar(PendingConflictMode::KeepIgnore),
            upon_opp_stop_conflict: FlexArray::Scalar(PendingConflictMode::KeepIgnore),
            tsl_stop: FlexArray::Scalar(f64::NAN),
            tsl_th: FlexArray::Scalar(f64::NAN),
            td_stop: FlexArray::Scalar(f64::NAN),
            dt_stop: FlexArray::Scalar(f64::NAN),
            stop_ladder: StopLadderMode::Disabled,
            stop_order_type: FlexArray::Scalar(OrderType::Market),
            stop_limit_delta: FlexArray::Scalar(f64::NAN),
            use_stops: true,
            val_price: FlexArray::Scalar(f64::INFINITY),
            cash_deposits: FlexArray::Scalar(0.0),
            cash_dividends: FlexArray::Scalar(0.0),
            cash_earnings: FlexArray::Scalar(0.0),
            funding_rate: FlexArray::Scalar(0.0),
            long_financing_rate: FlexArray::Scalar(0.0),
            short_financing_rate: FlexArray::Scalar(0.0),
            debt_rate: FlexArray::Scalar(0.0),
            borrow_rate: FlexArray::Scalar(0.0),
            mark_price: FlexArray::Scalar(f64::NAN),
            maintenance_margin: FlexArray::Scalar(0.0),
            cash_maintenance_margin: FlexArray::Scalar(f64::NAN),
            margin_tier_notional: FlexArray::Scalar(f64::NAN),
            margin_tier_rate: FlexArray::Scalar(f64::NAN),
            liquidation_step: FlexArray::Scalar(f64::NAN),
            auto_deleverage: FlexArray::Scalar(false),
            resume: None,
            row_offset: 0,
            liquidation_fee: FlexArray::Scalar(0.0),
            margin_mode: FlexArray::Scalar(MarginMode::Cross),
            ffill_val_price: true,
            update_value: false,
            groups: None,
            call_seq: CallSeq::Default,
        }
    }
}

/// The price a stop fills at, or `NaN` if it did not trigger this bar.
///
/// # The gap branch is the whole point
///
/// A stop is not a price you get; it is a price you asked for. If the bar
/// *opened* through the level, the fill is at the open — worse than the stop,
/// which is exactly what a gap does to a real account. Only when the level
/// sits inside the bar's range does the stop fill at the stop.
///
/// Dropping that branch makes every gap look like a perfectly executed stop
/// and quietly removes the largest losses from any backtest that uses them.
///
/// `hit_below` says which way the level is crossed: a stop-loss on a long is
/// hit from below, a take-profit from above.
///
/// # Errors
///
/// Returns [`Error::Invariant`] if `stop` is negative.
pub fn stop_price(
    position: f64,
    anchor: f64,
    stop: f64,
    open: f64,
    low: f64,
    high: f64,
    hit_below: bool,
) -> Result<f64> {
    if stop < 0.0 {
        return Err(Error::Invariant(
            "a stop distance must be 0 or greater".into(),
        ));
    }
    let downward = (position > 0.0 && hit_below) || (position < 0.0 && !hit_below);
    let upward = (position < 0.0 && hit_below) || (position > 0.0 && !hit_below);

    if downward {
        let level = anchor * (1.0 - stop);
        if open <= level {
            return Ok(open);
        }
        return Ok(if low <= level && level <= high {
            level
        } else {
            f64::NAN
        });
    }
    if upward {
        let level = anchor * (1.0 + stop);
        if level <= open {
            return Ok(open);
        }
        return Ok(if low <= level && level <= high {
            level
        } else {
            f64::NAN
        });
    }
    Ok(f64::NAN)
}

/// Whether a stop already in place should be replaced by a new one.
const fn should_update_stop(stop: f64, mode: StopUpdateMode) -> bool {
    match mode {
        StopUpdateMode::Keep => false,
        StopUpdateMode::Override => !stop.is_nan(),
        StopUpdateMode::OverrideNaN => true,
    }
}

/// Reads the four signals for one bar and merges the two conventions.
#[inline(always)]
fn read_signals(args: &FromSignals<'_>, i: usize, col: usize) -> Signals {
    let mut s = Signals {
        long_entry: args.long_entries.get(i, col),
        long_exit: args.long_exits.get(i, col),
        short_entry: args.short_entries.get(i, col),
        short_exit: args.short_exits.get(i, col),
    };
    let entry = args.entries.get(i, col);
    let exit = args.exits.get(i, col);
    if entry || exit {
        // Merged by OR, not replaced: a column may drive both conventions.
        match args.direction.get(i, col) {
            Direction::LongOnly => {
                s.long_entry |= entry;
                s.long_exit |= exit;
            }
            Direction::ShortOnly => {
                s.short_entry |= entry;
                s.short_exit |= exit;
            }
            Direction::Both => {
                // Note an "exit" here is a *short entry*. In a two-sided
                // column the pair means long-or-short, not in-or-out.
                s.long_entry |= entry;
                s.short_entry |= exit;
            }
        }
    }
    s
}

impl FromSignals<'_> {
    /// An upper bound on the fills columns `cols` can produce.
    ///
    /// Every fill needs a signal or an armed stop, a stop only fires once per
    /// position, and a position needs an entry signal, so fills never exceed
    /// twice the signal cells that are set. The record buffers are
    /// preallocated to that rather than to one fill per cell, which is what
    /// `from_orders` measured as worth +6% to +19% over growing by doubling.
    fn fill_bound(&self, cols: std::ops::Range<usize>) -> usize {
        if self.order_mode {
            // An order per sized cell, and a stop exit after each at most.
            let orders = crate::count_where(&self.size, self.n_rows, cols.clone(), |size| {
                size != 0.0 && !size.is_nan()
            });
            return orders.saturating_mul(2).min(self.n_rows * cols.len());
        }
        let set = |array| crate::count_where(array, self.n_rows, cols.clone(), |on| on);
        let signals = set(&self.entries)
            + set(&self.exits)
            + set(&self.long_entries)
            + set(&self.long_exits)
            + set(&self.short_entries)
            + set(&self.short_exits);
        signals.saturating_mul(2).min(self.n_rows * cols.len())
    }

    /// Whether any parameter [`ValidatedOrder`] covers changes down a column;
    /// if none does, each column's request is built once rather than per bar.
    /// `log` is read off the built request, so it counts too.
    fn params_vary_by_row(&self) -> bool {
        self.log.varies_by_row()
            || self.fees.varies_by_row()
            || self.fixed_fees.varies_by_row()
            || self.slippage.varies_by_row()
            || self.min_size.varies_by_row()
            || self.max_size.varies_by_row()
            || self.size_granularity.varies_by_row()
            || self.leverage.varies_by_row()
            || self.leverage_mode.varies_by_row()
            || self.initial_margin.varies_by_row()
            || self.margin_basis.varies_by_row()
            || self.cash_limit.varies_by_row()
            || self.reject_prob.varies_by_row()
            || self.lock_cash.varies_by_row()
            || self.allow_partial.varies_by_row()
            || self.raise_reject.varies_by_row()
    }

    /// The inputs of the holding charges.
    fn rates(&self) -> crate::holding::Rates<'_, '_> {
        crate::holding::Rates {
            funding: &self.funding_rate,
            long_financing: &self.long_financing_rate,
            short_financing: &self.short_financing_rate,
            debt: &self.debt_rate,
            borrow: &self.borrow_rate,
            mark_price: &self.mark_price,
            close: &self.close,
        }
    }

    /// The inputs of the liquidation test.
    fn liquidation(&self) -> crate::liquidation::Inputs<'_, '_> {
        crate::liquidation::Inputs {
            maintenance_margin: &self.maintenance_margin,
            cash_maintenance_margin: &self.cash_maintenance_margin,
            tier_bounds: &self.margin_tier_notional,
            tier_rates: &self.margin_tier_rate,
            liquidation_step: &self.liquidation_step,
            liquidation_fee: &self.liquidation_fee,
            mark_price: &self.mark_price,
            open: &self.open,
            high: &self.high,
            low: &self.low,
            close: &self.close,
        }
    }

    /// The price inputs beyond `price`.
    pub(crate) fn quotes(&self) -> crate::prices::Quotes<'_, '_> {
        crate::prices::Quotes {
            price_type: &self.price_type,
            bid: &self.bid_price,
            ask: &self.ask_price,
            vio: &self.price_area_vio_mode,
            low: &self.low,
            high: &self.high,
        }
    }

    /// The stop inputs.
    pub(crate) fn stop_inputs(&self) -> crate::stops::Inputs<'_, '_> {
        crate::stops::Inputs {
            sl_stop: &self.sl_stop,
            sl_trail: &self.sl_trail,
            tp_stop: &self.tp_stop,
            tsl_stop: &self.tsl_stop,
            tsl_th: &self.tsl_th,
            td_stop: &self.td_stop,
            dt_stop: &self.dt_stop,
            delta_format: &self.delta_format,
            ladder: self.stop_ladder,
        }
    }

    /// Whether the run's accounts may go below zero cash, as in
    /// `from_orders`: leverage, a margin column or a holding charge.
    fn levered(&self) -> bool {
        crate::any_levered(
            &self.leverage,
            &self.account_type,
            &self.multiplier,
            self.n_rows,
            self.n_cols,
        ) || !self.rates().all_off()
            || !self.liquidation().cash_off()
    }

    /// Resolves the default grouping and validates its width.
    ///
    /// # Errors
    /// Returns an invariant error if the groups do not cover all columns.
    pub fn resolved_groups(&self) -> Result<Groups> {
        let groups = self
            .groups
            .clone()
            .unwrap_or_else(|| Groups::ungrouped(self.n_cols));
        let covered: usize = (0..groups.len()).map(|g| groups.len_of(g)).sum();
        if covered != self.n_cols {
            return Err(Error::Invariant(format!(
                "the grouping covers {covered} columns, but the shape has {}",
                self.n_cols
            )));
        }
        Ok(groups)
    }

    /// The first row to simulate, counted from the arrays' first: 0, or
    /// where `resume` continues.
    fn start_row(&self, n_groups: usize) -> Result<usize> {
        self.resume.map_or(Ok(0), |state| {
            state.start_in(Loop::FromSignals, n_groups, self.row_offset, self.n_rows)
        })
    }

    /// [`levered`](Self::levered), kept from the state a run continues; as
    /// in `from_orders`.
    fn run_levered(&self) -> Result<bool> {
        let levered = self.levered();
        match self.resume {
            Some(state) if state.levered => Ok(true),
            Some(_) if levered => Err(Error::Invariant(
                "this run borrows or trades on margin and the run it continues did not; start \
                 that run with the leverage, account type or charges it will need"
                    .into(),
            )),
            _ => Ok(levered),
        }
    }

    /// The run's records on its own rows, and its state.
    fn finish_run<S: SinkBuffer, T>(
        &self,
        mut records: S,
        states: Vec<T>,
        levered: bool,
        groups: Vec<GroupState>,
    ) -> Run<S, T> {
        records.records_mut().allow_negative_price = self.allow_negative_price;
        records.shift_rows(self.row_offset as i64);
        Run {
            records,
            states,
            last_state: SimState {
                source: Loop::FromSignals,
                next_row: self.row_offset + self.n_rows,
                levered,
                groups,
            },
        }
    }
}

/// What [`prepare_order`] decided for one column on one bar: the four fields
/// [`ValidatedOrder::set_bar`] takes, whether the fill pays no slippage (a
/// stop resting at its level, or a limit order), and the level a stop that
/// fired was hit at, which the liquidation test credits.
///
/// `then` says the bar's signal follows a stop's fill (see
/// [`PendingConflictMode::KeepExecute`]), at the bar's `order_price`, placed
/// at the open if `at_open`.
#[derive(Debug, Clone, Copy)]
struct Pending {
    size: f64,
    price: f64,
    size_type: SizeType,
    direction: Direction,
    stop_limit: bool,
    stop_hit: f64,
    then: bool,
    order_price: f64,
    at_open: bool,
}

impl Pending {
    /// A market order for `sized` at `price`.
    fn new(price: f64, sized: crate::signals::SizedOrder) -> Self {
        // A short-only column expresses "more short" as a positive size,
        // so the sign is flipped back on the way in and the kernel flips
        // it again. Two flips rather than one.
        let size = if sized.direction == Direction::ShortOnly {
            -sized.size
        } else {
            sized.size
        };
        Self {
            size,
            size_type: sized.size_type,
            direction: sized.direction,
            ..Self::default_at(price)
        }
    }

    /// An empty market order at `price`, to fill in.
    const fn default_at(price: f64) -> Self {
        Self {
            size: 0.0,
            price,
            size_type: SizeType::Amount,
            direction: Direction::Both,
            stop_limit: false,
            stop_hit: f64::NAN,
            then: false,
            order_price: f64::NAN,
            at_open: false,
        }
    }
}

/// Builds the request for row `i` of column `col`, with the parameter checks
/// done once. `stop_limit` builds the variant a resting stop fills through,
/// which differs only in paying no slippage; `levered` marks a request from
/// an account that may borrow.
fn build_order(
    args: &FromSignals<'_>,
    i: usize,
    col: usize,
    stop_limit: bool,
    levered: bool,
) -> ValidatedOrder {
    let order = ValidatedOrder::new(Order {
        size: f64::NAN,
        price: f64::NAN,
        size_type: SizeType::Amount,
        direction: Direction::Both,
        fees: args.fees.get(i, col),
        fixed_fees: args.fixed_fees.get(i, col),
        slippage: if stop_limit {
            0.0
        } else {
            args.slippage.get(i, col)
        },
        min_size: args.min_size.get(i, col),
        max_size: args.max_size.get(i, col),
        size_granularity: args.size_granularity.get(i, col),
        leverage: args.leverage.get(i, col),
        leverage_mode: args.leverage_mode.get(i, col),
        account_type: args.account_type.get(i, col),
        initial_margin: args.initial_margin.get(i, col),
        margin_basis: args.margin_basis.get(i, col),
        cash_limit: args.cash_limit.get(i, col),
        reject_prob: args.reject_prob.get(i, col),
        lock_cash: args.lock_cash.get(i, col),
        allow_partial: args.allow_partial.get(i, col),
        raise_reject: args.raise_reject.get(i, col),
        log: args.log.get(i, col),
        allow_negative_price: args.allow_negative_price,
    });
    if levered {
        order.in_levered_account()
    } else {
        order
    }
}

/// The bar's open, high, low and close, with a missing open, high or low
/// reconstructed from what is known, so a close-only series still tests
/// stops and limit orders rather than silently never triggering them.
fn bar_range(args: &FromSignals<'_>, i: usize, col: usize) -> (f64, f64, f64, f64) {
    let close = args.close.get(i, col);
    let mut open = args.open.get(i, col);
    if open.is_nan() {
        open = close;
    }
    let mut low = args.low.get(i, col);
    if low.is_nan() {
        low = open.min(close);
    }
    let mut high = args.high.get(i, col);
    if high.is_nan() {
        high = open.max(close);
    }
    (open, high, low, close)
}

/// Resolves prices, stops, conflicts and size before any group order fills.
///
/// `sizes_checked` says the size and size type were validated once for the
/// column, so a bar with neither a signal nor a stop can be skipped without
/// changing where an invalid size is reported.
///
/// `EXTENDED` says whether the column's group uses anything beyond the
/// base pipeline: limit orders, a stop conflict mode other than
/// ignoring the signal, `from_ago`, a price type, bid and ask or a price
/// range check, or the stops of `crate::stops`. The copy without has no
/// code for any of them.
#[allow(
    clippy::too_many_lines,
    reason = "keep the ordered signal pipeline together"
)]
fn prepare_order<const EXTENDED: bool>(
    args: &FromSignals<'_>,
    i: usize,
    col: usize,
    state: &mut ColumnState,
    sizes_checked: bool,
    multiplier: f64,
    over: Option<&Decision>,
) -> Result<Option<Pending>> {
    // Only the extended copy takes a callback's decision.
    let over = if EXTENDED { over } else { None };
    let position = state.position;
    let mut stops = state.stops;
    let mut last_val_price = state.last_val_price;
    // --- prices -------------------------------------------------------
    // `-inf` means this bar's open here, where `from_orders` has no open
    // to reach for and falls back to the previous close.
    // The row the bar's order was decided on (`from_ago`), if it exists.
    let src = if EXTENDED {
        crate::prices::source_row(i, args.from_ago.get(i, col))
    } else {
        Some(i)
    };
    let mut price = src.map_or(f64::NAN, |r| args.price.get(r, col));
    if EXTENDED {
        // The open and the close are the executing bar's.
        match args.price_type.get(i, col) {
            PriceType::Price => {}
            PriceType::Open => price = f64::NEG_INFINITY,
            PriceType::Close => price = f64::INFINITY,
        }
    }
    // Whether the order is placed at the open, which leaves the rest of the
    // bar for a limit order to be tested against.
    let mut at_open = false;
    if price.is_infinite() {
        price = if price > 0.0 {
            args.close.get(i, col)
        } else {
            let open = args.open.get(i, col);
            if !open.is_nan() {
                at_open = true;
                open
            } else if i > 0 {
                args.close.get(i - 1, col)
            } else {
                f64::NAN
            }
        };
    }

    // Resolved before the stop engine can rewrite `price`, so a stop fill
    // does not retroactively change what the bar was valued at.
    let mut val_price = args.val_price.get(i, col);
    if val_price.is_infinite() {
        val_price = if val_price > 0.0 {
            price
        } else if i > 0 {
            args.close.get(i - 1, col)
        } else {
            f64::NAN
        };
    }
    if !val_price.is_nan() || !args.ffill_val_price {
        last_val_price = val_price;
    }
    let val_price_now = last_val_price;

    // --- stops --------------------------------------------------------
    let mut triggered = f64::NAN;
    let mut hit: Option<crate::stops::Hit> = None;
    if args.use_stops && stops.armed() {
        let (open, high, low, _) = bar_range(args, i, col);
        if EXTENDED {
            // The same stop-loss and take-profit, plus the trailing, time
            // and laddered stops, and which of them fired.
            hit = crate::stops::test(&stops, position, i, open, high, low)?;
            if let Some(h) = hit {
                triggered = h.price;
            }
            crate::stops::trail(&mut stops, position, high, low);
        } else {
            if !stops.sl_stop.is_nan() {
                triggered = stop_price(
                    position,
                    stops.sl_price,
                    stops.sl_stop,
                    open,
                    low,
                    high,
                    true,
                )?;
            }
            if triggered.is_nan() && !stops.tp_stop.is_nan() {
                triggered = stop_price(
                    position,
                    stops.tp_price,
                    stops.tp_stop,
                    open,
                    low,
                    high,
                    false,
                )?;
            }

            // Trailed *after* the test, so this bar is judged against the
            // peak as of the previous one. Trailing first would let a bar
            // that made a new high also stop out on its own pullback.
            if !stops.sl_stop.is_nan() && stops.sl_trail {
                if position > 0.0 && high > stops.sl_price {
                    stops.sl_price = high;
                } else if position < 0.0 && low < stops.sl_price {
                    stops.sl_price = low;
                }
            }
        }
    }

    // --- a forced exit ------------------------------------------------
    // After the stops, which fire inside the bar: the position left is
    // closed at the order price, and nothing else is ordered or rests.
    if EXTENDED && triggered.is_nan() && args.force_exit.get(i, col) {
        state.limit = LimitOrder::none();
        state.stops = stops;
        state.last_val_price = last_val_price;
        if position == 0.0 {
            return Ok(None);
        }
        let long = position > 0.0;
        let close = crate::signals::SizedOrder {
            size: -position,
            size_type: SizeType::Amount,
            direction: if long {
                Direction::LongOnly
            } else {
                Direction::ShortOnly
            },
        };
        let price = args.quotes().order_price(i, col, price, !long)?;
        return Ok(Some(Pending::new(price, close)));
    }

    // --- the resting limit order --------------------------------------
    // Tested when no stop fired, which comes first; its time running out
    // cancels it either way.
    if EXTENDED && state.limit.is_pending() {
        if let Some(fill) = test_resting(args, i, col, state, triggered) {
            state.stops = stops;
            state.last_val_price = last_val_price;
            return Ok(Some(fill));
        }
    }

    // --- the bar's signal ---------------------------------------------
    // A fired stop replaces it, unless the stop conflict modes say
    // otherwise. `triggered` is only ever set with `use_stops` on.
    if triggered.is_nan() {
        let user = user_order::<EXTENDED>(
            args,
            src,
            col,
            position,
            val_price_now * multiplier,
            sizes_checked,
            over,
        )?;
        state.stops = stops;
        state.last_val_price = last_val_price;
        let Some(sized) = user else {
            return Ok(None);
        };
        return signal_order::<EXTENDED>(args, i, col, state, price, at_open, sized);
    }

    // --- the stop's order ---------------------------------------------
    let (signals, accumulate) = from_stop(
        position,
        args.upon_stop_exit.get(i, col),
        args.accumulate.get(i, col),
    );
    // A stop order's price, and whether slippage applies to it: a limit
    // resting at the stop pays none, a market order does.
    // A time stop fires at the open, so `triggered` is the open for it.
    let exit_price = args.stop_exit_price.get(i, col);
    let (mut stop_fill, stop_limit) = match exit_price {
        StopExitPrice::StopMarket => (triggered, false),
        StopExitPrice::StopLimit => (triggered, true),
        StopExitPrice::Close => (args.close.get(i, col), false),
        StopExitPrice::Price => (price, false),
    };
    let (size, size_type) = over.map_or_else(
        || (args.size.get(i, col), args.size_type.get(i, col)),
        |d| (d.size, d.size_type),
    );
    let mut sized = size_signals(
        position,
        signals,
        size,
        size_type,
        accumulate,
        // A value reversal is so many contracts' worth.
        val_price_now * multiplier,
    )?;
    // A ladder step short of the last closes only its share.
    let mut partial = false;
    if EXTENDED {
        if let Some(h) = hit {
            let inputs = args.stop_inputs();
            if let Some(q) = crate::stops::ladder_exit(&inputs, col, h.kind, position, &mut stops) {
                partial = true;
                let long = position > 0.0;
                sized = crate::signals::SizedOrder {
                    size: if long { -q } else { q },
                    size_type: SizeType::Amount,
                    direction: if long {
                        Direction::LongOnly
                    } else {
                        Direction::ShortOnly
                    },
                };
            }
        }
        if exit_price == StopExitPrice::Price {
            stop_fill = args.quotes().order_price(i, col, price, sized.size > 0.0)?;
        }
    }
    state.stops = stops;
    state.last_val_price = last_val_price;

    let mut then = false;
    if EXTENDED {
        // The bar's own signal, sized against the position the stop is about
        // to close — or, if it asks for nothing there, such as an entry on
        // the side already held, against the flat the stop leaves. Its side
        // is its order's.
        let val = val_price_now * multiplier;
        let held = user_order::<EXTENDED>(args, src, col, position, val, sizes_checked, over)?;
        let side = match held {
            Some(user) => Some(user),
            None => user_order::<EXTENDED>(args, src, col, 0.0, val, sizes_checked, over)?,
        };
        if let Some(user) = side {
            let (keep, execute) = resolve_pending_conflict(
                sized.size > 0.0,
                user.size > 0.0,
                args.upon_adj_stop_conflict.get(i, col),
                args.upon_opp_stop_conflict.get(i, col),
            );
            if !keep {
                // The stop is disarmed, and the signal, against the position
                // held, decides the bar.
                state.stops = StopState::cleared();
                return match held {
                    Some(user) if execute => {
                        signal_order::<EXTENDED>(args, i, col, state, price, at_open, user)
                    }
                    _ => Ok(None),
                };
            }
            // After the stop's fill, re-sized against what it left.
            then = execute;
        }
        // The stop fills in the column, which cancels a resting order.
        state.limit = LimitOrder::none();
        if let Some(h) = hit {
            if sized.size != 0.0 && args.stop_order_type.get(i, col) == OrderType::Limit {
                if !partial {
                    // The stop has fired; its order now does its work.
                    match h.kind {
                        crate::stops::Kind::StopLoss => state.stops.sl_stop = f64::NAN,
                        crate::stops::Kind::Trailing => state.stops.tsl_stop = f64::NAN,
                        crate::stops::Kind::TakeProfit => state.stops.tp_stop = f64::NAN,
                        crate::stops::Kind::Time => {
                            state.stops.td_row = usize::MAX;
                            state.stops.dt_row = usize::MAX;
                        }
                    }
                }
                return stop_limit_order(args, i, col, state, h, sized);
            }
        }
    }
    if sized.size == 0.0 {
        return Ok(None);
    }
    Ok(Some(Pending {
        stop_limit,
        stop_hit: triggered,
        then,
        order_price: price,
        at_open,
        ..Pending::new(stop_fill, sized)
    }))
}

/// The signal decided on `row` for a column holding `position`, read,
/// resolved and sized, or `None` when it asks for nothing. `val_price` is
/// already multiplied by the column's contract multiplier.
///
/// `sizes_checked`: see [`prepare_order`].
#[inline(always)]
fn user_order<const EXTENDED: bool>(
    args: &FromSignals<'_>,
    row: Option<usize>,
    col: usize,
    position: f64,
    val_price: f64,
    sizes_checked: bool,
    over: Option<&Decision>,
) -> Result<Option<crate::signals::SizedOrder>> {
    let over = if EXTENDED { over } else { None };
    // Read on the row the order was decided on; none before it exists.
    let Some(i) = row else {
        return Ok(None);
    };
    // A callback's decision in place of the arrays', read only when sized.
    let sizing = || {
        over.map_or_else(
            || (args.size.get(i, col), args.size_type.get(i, col)),
            |d| (d.size, d.size_type),
        )
    };
    if EXTENDED && args.order_mode {
        // The order is `size` itself; before the short-only flip a positive
        // size buys, as for a signal's order.
        let (size, size_type) = sizing();
        if size.is_nan() || size == 0.0 {
            return Ok(None);
        }
        let direction = args.direction.get(i, col);
        return Ok(Some(crate::signals::SizedOrder {
            size: if direction == Direction::ShortOnly {
                -size
            } else {
                size
            },
            size_type,
            direction,
        }));
    }
    let mut s = over.map_or_else(|| read_signals(args, i, col), |d| d.signals);
    if sizes_checked && !s.any() {
        // A quiet bar: nothing to size, and the size parameters were
        // already checked, so the sizing step has nothing to report.
        return Ok(None);
    }
    let mut accumulate = args.accumulate.get(i, col);
    // Conflicts are only worth resolving when an entry is involved.
    // Two exits cannot conflict: they act on different sides.
    if s.long_entry || s.short_entry {
        let (le, lx) = resolve_signal_conflict(
            position,
            s.long_entry,
            s.long_exit,
            Direction::LongOnly,
            args.upon_long_conflict.get(i, col),
        );
        s.long_entry = le;
        s.long_exit = lx;
        let (se, sx) = resolve_signal_conflict(
            position,
            s.short_entry,
            s.short_exit,
            Direction::ShortOnly,
            args.upon_short_conflict.get(i, col),
        );
        s.short_entry = se;
        s.short_exit = sx;

        let (le, se) = resolve_dir_conflict(
            position,
            s.long_entry,
            s.short_entry,
            args.upon_dir_conflict.get(i, col),
        );
        s.long_entry = le;
        s.short_entry = se;

        let (resolved, acc) = resolve_opposite_entry(
            position,
            s,
            args.upon_opposite_entry.get(i, col),
            accumulate,
        );
        s = resolved;
        accumulate = acc;
    }
    let (size, size_type) = sizing();
    let sized = size_signals(
        position, s, size, size_type, accumulate,
        // A value reversal is so many contracts' worth.
        val_price,
    )?;
    Ok((sized.size != 0.0).then_some(sized))
}

/// The bar's sized signal as the order it becomes: a market order at
/// `price`, or with `EXTENDED`, what the resting limit order and the
/// signal's `order_type` make of it.
#[inline(always)]
fn signal_order<const EXTENDED: bool>(
    args: &FromSignals<'_>,
    i: usize,
    col: usize,
    state: &mut ColumnState,
    price: f64,
    at_open: bool,
    sized: crate::signals::SizedOrder,
) -> Result<Option<Pending>> {
    if EXTENDED {
        let price = args.quotes().order_price(i, col, price, sized.size > 0.0)?;
        if let Some(routed) = route_limit(args, i, col, state, price, at_open, sized)? {
            return Ok(routed);
        }
        return Ok(Some(Pending::new(price, sized)));
    }
    Ok(Some(Pending::new(price, sized)))
}

/// The order that follows a stop's fill on its bar, under a stop conflict
/// mode that keeps the stop and executes the signal: the bar's signal,
/// re-sized against the position the stop left. `multiplier` is 1 outside
/// the leveraged copy.
#[inline(never)]
fn follow_up(
    args: &FromSignals<'_>,
    i: usize,
    col: usize,
    state: &mut ColumnState,
    after: &Pending,
    multiplier: f64,
    sizes_checked: bool,
    over: Option<&Decision>,
) -> Result<Option<Pending>> {
    let user = user_order::<true>(
        args,
        crate::prices::source_row(i, args.from_ago.get(i, col)),
        col,
        state.position,
        state.last_val_price * multiplier,
        sizes_checked,
        over,
    )?;
    let Some(sized) = user else {
        return Ok(None);
    };
    signal_order::<true>(args, i, col, state, after.order_price, after.at_open, sized)
}

/// Tests the order resting in a column on bar `i`: cancels it once its time
/// has run out, and returns its fill if no stop fired and the bar reaches
/// it.
///
/// Out of line, with [`route_limit`], so a run without limit orders keeps
/// the signal pipeline as small as it was.
#[inline(never)]
fn test_resting(
    args: &FromSignals<'_>,
    i: usize,
    col: usize,
    state: &mut ColumnState,
    triggered: f64,
) -> Option<Pending> {
    let l = state.limit;
    if i > l.last_row {
        state.limit = LimitOrder::none();
        return None;
    }
    if !triggered.is_nan() {
        return None;
    }
    let (open, high, low, close) = bar_range(args, i, col);
    let fill = crate::limit::fill_price(
        l.price,
        l.buy,
        l.reverse,
        l.order_price,
        open,
        high,
        low,
        close,
    );
    if fill.is_nan() {
        return None;
    }
    state.limit = LimitOrder::none();
    Some(Pending {
        size: l.size,
        size_type: l.size_type,
        direction: l.direction,
        stop_limit: true,
        ..Pending::default_at(fill)
    })
}

/// Routes a signal's sized order where pending orders are on: `None`
/// executes it as a market order, `Some` is what the bar does instead —
/// nothing, when the limit conflict modes keep a resting order and ignore
/// the signal, or the order placed as a limit order and, if the bar already
/// reaches it, its fill.
#[allow(
    clippy::too_many_arguments,
    reason = "the column, its bar, the order price and the sized order"
)]
#[inline(never)]
fn route_limit(
    args: &FromSignals<'_>,
    i: usize,
    col: usize,
    state: &mut ColumnState,
    price: f64,
    at_open: bool,
    sized: crate::signals::SizedOrder,
) -> Result<Option<Option<Pending>>> {
    // Before the short-only flip a positive size buys, whatever the
    // direction.
    let buy = sized.size > 0.0;
    if state.limit.is_pending() {
        let (keep, execute) = resolve_pending_conflict(
            state.limit.buy,
            buy,
            args.upon_adj_limit_conflict.get(i, col),
            args.upon_opp_limit_conflict.get(i, col),
        );
        if !keep {
            state.limit = LimitOrder::none();
        }
        if !execute {
            return Ok(Some(None));
        }
    }
    if args.order_type.get(i, col) == OrderType::Limit {
        // One order rests per column: a new one takes the place of one kept.
        return place_limit(args, i, col, state, price, at_open, buy, sized).map(Some);
    }
    Ok(None)
}

/// Turns a fired price stop's exit into a limit order at its level,
/// `stop_limit_delta` worse — below it for a sell, above it for a buy — and
/// fills it at once if the price the stop fired at reaches it: a stop-limit
/// order. A bar that gaps past the limit leaves it resting.
#[inline(never)]
fn stop_limit_order(
    args: &FromSignals<'_>,
    i: usize,
    col: usize,
    state: &mut ColumnState,
    hit: crate::stops::Hit,
    sized: crate::signals::SizedOrder,
) -> Result<Option<Pending>> {
    let at = Pending::new(hit.level, sized);
    let buy = sized.size > 0.0;
    // Worse than the level is the way a reversed limit's delta goes.
    let level = crate::limit::limit_price(
        hit.level,
        buy,
        true,
        args.stop_limit_delta.get(i, col),
        args.delta_format.get(i, col),
    )?;
    let Some(last_row) =
        crate::limit::last_row(i, args.limit_tif.get(i, col), args.limit_expiry.get(i, col))?
    else {
        return Ok(None);
    };
    let order = LimitOrder {
        price: level,
        buy,
        reverse: false,
        order_price: args.limit_order_price.get(i, col),
        size: at.size,
        size_type: at.size_type,
        direction: at.direction,
        created: i,
        last_row,
    };
    let fill = crate::limit::fill_price(
        level,
        buy,
        false,
        order.order_price,
        hit.price,
        f64::NAN,
        f64::NAN,
        args.close.get(i, col),
    );
    if fill.is_nan() {
        state.limit = order;
        return Ok(None);
    }
    Ok(Some(Pending {
        price: fill,
        stop_limit: true,
        stop_hit: hit.price,
        ..at
    }))
}

/// Places a signal's sized order as a limit order at `price`, the bar's
/// order price, and fills it at once if the bar already reaches it.
#[allow(
    clippy::too_many_arguments,
    reason = "the column, its bar, the order price and the sized order"
)]
fn place_limit(
    args: &FromSignals<'_>,
    i: usize,
    col: usize,
    state: &mut ColumnState,
    price: f64,
    at_open: bool,
    buy: bool,
    sized: crate::signals::SizedOrder,
) -> Result<Option<Pending>> {
    let at_price = Pending::new(price, sized);
    let reverse = args.limit_reverse.get(i, col);
    let level = crate::limit::limit_price(
        price,
        buy,
        reverse,
        args.limit_delta.get(i, col),
        args.delta_format.get(i, col),
    )?;
    // No order price, nothing to rest at.
    if level.is_nan() {
        return Ok(None);
    }
    let Some(last_row) =
        crate::limit::last_row(i, args.limit_tif.get(i, col), args.limit_expiry.get(i, col))?
    else {
        return Ok(None);
    };
    let order = LimitOrder {
        price: level,
        buy,
        reverse,
        order_price: args.limit_order_price.get(i, col),
        size: at_price.size,
        size_type: at_price.size_type,
        direction: at_price.direction,
        created: i,
        last_row,
    };
    if !args.limit_delay.get(i, col) {
        // From the order price on: the rest of the bar only after its open.
        let (high, low) = if at_open {
            let (_, high, low, _) = bar_range(args, i, col);
            (high, low)
        } else {
            (f64::NAN, f64::NAN)
        };
        let fill = crate::limit::fill_price(
            level,
            buy,
            reverse,
            order.order_price,
            price,
            high,
            low,
            args.close.get(i, col),
        );
        if !fill.is_nan() {
            return Ok(Some(Pending {
                price: fill,
                stop_limit: true,
                ..at_price
            }));
        }
    }
    state.limit = order;
    Ok(None)
}

/// Executes a group sequentially within each bar; groups are independent.
///
/// `levered` is decided once for the run, as in `from_orders`.
fn simulate_group<S: RecordSink + LogSink, C: SignalFunc>(
    args: &FromSignals<'_>,
    groups: &Groups,
    group: usize,
    levered: bool,
    cb: &C,
    sink: &mut S,
    fills: &mut i64,
) -> Result<(C::State, GroupState)> {
    let flows = !crate::cash::all_off(
        &args.cash_deposits,
        &args.cash_dividends,
        &args.cash_earnings,
    );
    match (flows, levered) {
        (false, false) => group_loop::<S, C, false, false>(args, groups, group, cb, sink, fills),
        (true, false) => group_loop::<S, C, true, false>(args, groups, group, cb, sink, fills),
        (false, true) => group_loop::<S, C, false, true>(args, groups, group, cb, sink, fills),
        (true, true) => group_loop::<S, C, true, true>(args, groups, group, cb, sink, fills),
    }
}

/// The loop of [`simulate_group`], in four copies: with `FLOWS` false it has
/// no code for cash flows at all (see [`crate::cash::all_off`]), and with
/// `LEV` false none for leverage, margin columns or holding charges — valuing
/// a long net of its debt and recording each fill's balances, as in
/// `from_orders`.
#[allow(
    clippy::too_many_lines,
    reason = "keep fill, stop state and log updates together"
)]
fn group_loop<S: RecordSink + LogSink, C: SignalFunc, const FLOWS: bool, const LEV: bool>(
    args: &FromSignals<'_>,
    groups: &Groups,
    group: usize,
    cb: &C,
    sink: &mut S,
    fills: &mut i64,
) -> Result<(C::State, GroupState)> {
    let cols = groups.range(group);
    let mut cs = cb.pre_group(group, cols.clone())?;
    let start = args.start_row(groups.len())?;
    if C::ON && !matches!(args.from_ago, FlexArray::Scalar(0)) {
        return Err(Error::Invariant(
            "from_ago does not apply to signals a callback decides; lag them in the callback"
                .into(),
        ));
    }
    let len = cols.len();
    let shared = len > 1;
    // A run that continues another starts from where that one ended.
    let resumed = args
        .resume
        .map(|state| state.group_for(group, len, args.row_offset))
        .transpose()?;
    let (mut cash, mut free_cash, mut states, mut stopped) = match resumed {
        Some(g) => (g.cash, g.free_cash, g.columns, g.stopped),
        None => {
            let cash = args.init_cash.get(0, cols.start);
            (cash, cash, vec![ColumnState::new(); len], false)
        }
    };
    // The request per column, and the variant a resting stop fills through,
    // built once unless a parameter varies down the column. This hoists the
    // parameter reads and checks that `from_orders` measured at +30% on
    // sparse workloads.
    let params_vary = args.params_vary_by_row();
    let mut orders: Vec<ValidatedOrder> = cols
        .clone()
        .map(|c| build_order(args, 0, c, false, LEV))
        .collect();
    let mut stop_orders: Vec<ValidatedOrder> = cols
        .clone()
        .map(|c| build_order(args, 0, c, true, LEV))
        .collect();
    // Which columns are margin columns, and which of those isolated; none
    // outside a leveraged run.
    let margin = if LEV {
        crate::margin_columns(&args.account_type, args.n_rows, cols.clone())?
    } else {
        Vec::new()
    };
    let isolated = if LEV {
        crate::isolated_columns(&args.margin_mode, args.n_rows, cols.clone(), &margin)?
    } else {
        Vec::new()
    };
    // Each column's contract multiplier. Unlike `from_orders`, a column's
    // state keeps its valuation price unscaled, because the stops read it;
    // the copy multiplies it where it values, settles, charges or executes.
    let mult = if LEV {
        crate::multipliers(&args.multiplier, args.n_rows, cols.clone())?
    } else {
        Vec::new()
    };
    if args.resume.is_none() && crate::initial::any(&args.init_position, cols.clone()) {
        let inputs = crate::initial::Columns {
            orders: &orders,
            margin: &margin,
            isolated: &isolated,
            mult: &mult,
            scaled_val_price: false,
        };
        crate::initial::open(
            &args.init_position,
            &args.init_price,
            &args.close,
            cols.clone(),
            &inputs,
            &mut states,
            &mut cash,
            &mut free_cash,
            sink,
            fills,
        )?;
    }
    // Whether any of the group's columns uses anything beyond the
    // base pipeline (see `prepare_order`); without, no bar has code
    // for it.
    let limit_orders = |types: &FlexArray<'_, OrderType>| {
        crate::count_where(types, args.n_rows, cols.clone(), |t| t == OrderType::Limit) > 0
    };
    // Whether the group arms the stops of `crate::stops`.
    let ext_stops = args.stop_inputs().on();
    let extended = limit_orders(&args.order_type)
        || limit_orders(&args.stop_order_type)
        || [&args.upon_adj_stop_conflict, &args.upon_opp_stop_conflict]
            .iter()
            .any(|mode| {
                crate::count_where(mode, args.n_rows, cols.clone(), |m| {
                    m != PendingConflictMode::KeepIgnore
                }) > 0
            })
        || !matches!(args.from_ago, FlexArray::Scalar(0))
        || !matches!(args.force_exit, FlexArray::Scalar(false))
        || args.order_mode
        || args.quotes().on()
        || ext_stops
        // A callback's decision goes through the extended copy.
        || C::ON;
    let liquidates = LEV && {
        let inputs = args.liquidation();
        ((!inputs.off() || !matches!(args.auto_deleverage, FlexArray::Scalar(false)))
            && margin.contains(&true))
            || !inputs.cash_off()
    };
    let mut to_close = Vec::new();
    // The size checks the sizing step makes are column-constant when the size
    // is; doing them here lets a quiet bar skip that step. They are made at
    // the same point they would otherwise first be made: before bar 0.
    let sizes_checked = !C::ON && !args.size.varies_by_row() && !args.size_type.varies_by_row();
    if sizes_checked {
        for col in cols.clone() {
            size_signals(
                0.0,
                Signals {
                    long_entry: false,
                    long_exit: false,
                    short_entry: false,
                    short_exit: false,
                },
                args.size.get(0, col),
                args.size_type.get(0, col),
                args.accumulate.get(0, col),
                f64::NAN,
            )?;
        }
    }
    let mut pending: Vec<Option<Pending>> = vec![None; len];
    let mut order_values = vec![0.0; len];
    let mut seq: Vec<usize> = (0..len).collect();
    let at_open = FLOWS
        && (!crate::cash::is_off(&args.cash_deposits)
            || !crate::cash::is_off(&args.cash_dividends));
    let earnings = FLOWS && !crate::cash::is_off(&args.cash_earnings);
    let charges = LEV && !args.rates().all_off();
    // Each column's decision on the bar, when a callback makes it.
    let mut decisions: Vec<Decision> = Vec::new();
    if C::ON {
        decisions.resize(
            len,
            Decision {
                signals: Signals {
                    long_entry: false,
                    long_exit: false,
                    short_entry: false,
                    short_exit: false,
                },
                size: f64::NAN,
                size_type: SizeType::Amount,
            },
        );
    }
    for i in start..args.n_rows {
        if at_open {
            crate::cash::at_open(
                &args.cash_deposits,
                &args.cash_dividends,
                i,
                cols.clone(),
                // A dividend per share is paid on every share a contract
                // holds.
                |k| {
                    if LEV {
                        states[k].position * mult[k]
                    } else {
                        states[k].position
                    }
                },
                &mut cash,
                &mut free_cash,
                sink,
            )?;
        }
        if params_vary {
            for (k, col) in cols.clone().enumerate() {
                orders[k] = build_order(args, i, col, false, LEV);
                stop_orders[k] = build_order(args, i, col, true, LEV);
            }
        }
        let mut any = false;
        let deciding = C::ON && !stopped;
        let ctx_value = if deciding {
            signal_value::<LEV>(cash, &states, &margin, &mult)
        } else {
            0.0
        };
        for (k, col) in cols.clone().enumerate() {
            let m = if LEV { mult[k] } else { 1.0 };
            if deciding {
                let ctx = Context {
                    group,
                    row: i,
                    cols: &cols,
                    cash,
                    free_cash,
                    value: ctx_value,
                    columns: &states,
                    mult: &mult,
                    scaled: false,
                    close: &args.close,
                    orders: &orders,
                };
                let (stops, decision) = decide(cb, &mut cs, &ctx, k, args, i, col)?;
                states[k].stops = stops;
                decisions[k] = decision;
            }
            let over = if C::ON { decisions.get(k) } else { None };
            // Two copies, so a run without limit orders has no code for them.
            pending[k] = if C::ON && stopped {
                None
            } else if extended {
                prepare_order::<true>(args, i, col, &mut states[k], sizes_checked, m, over)?
            } else {
                prepare_order::<false>(args, i, col, &mut states[k], sizes_checked, m, None)?
            };
            any |= pending[k].is_some();
            // Every bar, quiet or not: the profit is the account's whether
            // or not a signal comes.
            if LEV && margin[k] && !isolated[k] {
                let st = &mut states[k];
                crate::margin::settle(
                    st.position,
                    &mut st.debt,
                    st.last_val_price * mult[k],
                    &mut cash,
                    &mut free_cash,
                );
            }
        }
        if liquidates {
            let books = Books {
                margin: &margin,
                isolated: &isolated,
                mult: &mult,
                orders: &orders,
            };
            any = liquidate(
                args,
                group,
                cols.clone(),
                i,
                &books,
                &mut states,
                &mut pending,
                &mut cash,
                &mut free_cash,
                &mut to_close,
                sink,
                fills,
            )?;
        }
        if !any {
            // A quiet bar still earns and is charged: interest does not wait
            // for a signal.
            if earnings {
                crate::cash::earn(
                    &args.cash_earnings,
                    i,
                    cols.clone(),
                    &mut cash,
                    &mut free_cash,
                    sink,
                )?;
            }
            if charges {
                crate::holding::charge(
                    &args.rates(),
                    i,
                    cols.clone(),
                    crate::holding::Columns {
                        states: &mut states,
                        margin: &margin,
                        isolated: &isolated,
                        mult: &mult,
                        scaled_val_price: false,
                    },
                    &mut cash,
                    &mut free_cash,
                    sink,
                )?;
            }
            if C::ON && !stopped {
                stopped = segment_end::<C, LEV>(
                    args, cb, &mut cs, group, i, &cols, cash, free_cash, &states, &margin, &mult,
                    &orders,
                )?;
            }
            continue;
        }
        let mut group_value = cash;
        for (k, st) in states.iter().enumerate() {
            if LEV && margin[k] {
                group_value +=
                    crate::margin::unsettled(st.position, st.debt, st.last_val_price * mult[k]);
                continue;
            }
            if st.position != 0.0 {
                group_value += if LEV {
                    st.position * (st.last_val_price * mult[k])
                } else {
                    st.position * st.last_val_price
                };
            }
            if LEV {
                group_value -= st.long_debt();
            }
        }
        // A group of one has one order and nothing to sequence.
        if shared {
            if args.call_seq == CallSeq::Auto {
                for (k, p) in pending.iter().enumerate() {
                    order_values[k] = p.as_ref().map_or(0.0, |p| {
                        let val_price = if LEV {
                            states[k].last_val_price * mult[k]
                        } else {
                            states[k].last_val_price
                        };
                        let value = approx_order_value(
                            p.size,
                            p.size_type,
                            p.direction,
                            cash,
                            states[k].position,
                            free_cash,
                            val_price,
                            group_value,
                        );
                        if LEV {
                            approx_cash_need(
                                value,
                                states[k].position * val_price,
                                val_price,
                                orders[k].order(),
                            )
                        } else {
                            value
                        }
                    });
                }
            }
            args.call_seq
                .resolve(cols.clone(), i, args.n_rows, &mut order_values, &mut seq)?;
        }
        'orders: for &k in &seq {
            let Some(mut p) = pending[k] else {
                continue;
            };
            // A second pass only for the signal that follows a stop's fill
            // (`then`), re-sized against what the stop left.
            loop {
                let order = if p.stop_limit {
                    &mut stop_orders[k]
                } else {
                    &mut orders[k]
                };
                // The kernel runs a multiplied column at its prices times the
                // multiplier; the stops and records keep the prices.
                let m = if LEV { mult[k] } else { 1.0 };
                if LEV {
                    order.set_bar(p.size, p.price * m, p.size_type, p.direction);
                } else {
                    order.set_bar(p.size, p.price, p.size_type, p.direction);
                }
                let col = cols.start + k;
                let mut position = states[k].position;
                let debt = states[k].debt;
                let locked_cash = states[k].locked_cash;
                let val_price_now = states[k].last_val_price;
                let mut stops = states[k].stops;
                let price = p.price;
                let value = group_value;
                let state = ProcessOrderState {
                    cash,
                    position,
                    debt,
                    locked_cash,
                    free_cash,
                    val_price: if LEV {
                        val_price_now * m
                    } else {
                        val_price_now
                    },
                    value,
                    oidx: 0,
                    lidx: 0,
                };
                // Drawn only when asked for, so a run without random rejection
                // never touches the generator.
                let draw = (order.order().reject_prob > 0.0)
                    .then(|| crate::random::uniform(args.seed, i + args.row_offset, col));
                let (exec_state, result) = execute_validated(&state, order, draw)?;

                if result.status == OrderStatus::Rejected && order.order().raise_reject {
                    return Err(Error::Invariant(format!(
                        "order rejected at row {i}, column {col}: {}",
                        result
                            .status_info
                            .map_or("unknown reason", alphax_model::StatusInfo::describe)
                    )));
                }

                // Thread the account valuation through fills when requested.
                let (new_val_price, new_value) = if result.is_filled() && args.update_value {
                    let (val_price, value) = updated_value(
                        state.cash,
                        exec_state.cash,
                        state.position,
                        exec_state.position,
                        state.val_price,
                        result.price,
                        state.value,
                    );
                    if LEV {
                        // Borrowing moves no value: it adds as much debt as cash.
                        // A margin fill adds as much cost basis as position value.
                        let now = ColumnState {
                            position: exec_state.position,
                            debt: exec_state.debt,
                            ..states[k]
                        };
                        let m = margin[k];
                        (val_price, value - (now.owed(m) - states[k].owed(m)))
                    } else {
                        (val_price, value)
                    }
                } else {
                    (state.val_price, state.value)
                };

                let position_before = position;
                cash = exec_state.cash;
                position = exec_state.position;
                states[k].debt = exec_state.debt;
                states[k].locked_cash = exec_state.locked_cash;
                free_cash = exec_state.free_cash;
                // Re-marked to the fill's price, the price its record keeps.
                states[k].last_val_price = if !LEV {
                    new_val_price
                } else if result.is_filled() && args.update_value {
                    result.price / m
                } else {
                    val_price_now
                };
                group_value = new_value;

                // --- re-arm stops ---------------------------------------------
                if args.use_stops && result.is_filled() {
                    if position == 0.0 {
                        stops = StopState::cleared();
                    } else {
                        let anchor = match args.stop_entry_price.get(i, col) {
                            StopEntryPrice::ValPrice => val_price_now,
                            StopEntryPrice::Price => price,
                            StopEntryPrice::FillPrice => {
                                if LEV {
                                    result.price / m
                                } else {
                                    result.price
                                }
                            }
                            StopEntryPrice::Close => args.close.get(i, col),
                        };
                        let opened_or_reversed = position_before == 0.0
                            || position.is_sign_positive() != position_before.is_sign_positive();

                        if ext_stops {
                            // A trailing stop, a time stop or a ladder.
                            let inputs = args.stop_inputs();
                            if opened_or_reversed {
                                stops = crate::stops::arm(&inputs, i, col, anchor, position)?;
                            } else if position.abs() > position_before.abs() {
                                let mode = args.upon_stop_update.get(i, col);
                                crate::stops::update(&inputs, i, col, anchor, mode, &mut stops)?;
                            }
                        } else if opened_or_reversed {
                            stops = StopState {
                                sl_price: anchor,
                                sl_stop: args.sl_stop.get(i, col),
                                sl_trail: args.sl_trail.get(i, col),
                                tp_price: anchor,
                                tp_stop: args.tp_stop.get(i, col),
                                ..StopState::cleared()
                            };
                        } else if position.abs() > position_before.abs() {
                            // Added to the position. Whether that re-anchors the
                            // stops is the caller's choice: averaging down and
                            // keeping the original stop is a different strategy
                            // from averaging down and moving it.
                            let (sl, tp) = (args.sl_stop.get(i, col), args.tp_stop.get(i, col));
                            let mode = args.upon_stop_update.get(i, col);
                            if should_update_stop(sl, mode) {
                                stops.sl_price = anchor;
                                stops.sl_stop = sl;
                                stops.sl_trail = args.sl_trail.get(i, col);
                            }
                            if should_update_stop(tp, mode) {
                                stops.tp_price = anchor;
                                stops.tp_stop = tp;
                            }
                        }
                    }
                }

                // The fill's own index, so a log joins back to it. Relative to this
                // buffer; the merge re-bases it. `-1` when nothing filled, which is
                // most of the interesting log rows.
                let mut order_id = -1_i64;
                if result.is_filled() {
                    let side = result.side.expect("a filled order always carries a side");
                    if LEV {
                        sink.push_order(i, col, result.size, result.price / m, result.fees, side);
                        sink.push_balances(Balances {
                            debt: exec_state.debt,
                            locked_cash: exec_state.locked_cash,
                            margin: margin[k],
                            isolated: isolated[k],
                            liquidation: false,
                            multiplier: m,
                        });
                    } else {
                        sink.push_order(i, col, result.size, result.price, result.fees, side);
                    }
                    order_id = *fills;
                    *fills += 1;
                }

                if order.order().log {
                    let mut entry = LogEntry {
                        group,
                        col,
                        idx: i,
                        cash: state.cash,
                        position: state.position,
                        debt: state.debt,
                        locked_cash: state.locked_cash,
                        free_cash: state.free_cash,
                        val_price: state.val_price,
                        value: state.value,
                        request: *order.order(),
                        new_cash: exec_state.cash,
                        new_position: exec_state.position,
                        new_debt: exec_state.debt,
                        new_locked_cash: exec_state.locked_cash,
                        new_free_cash: exec_state.free_cash,
                        new_val_price,
                        new_value,
                        result,
                        order_id,
                        liquidation: false,
                        multiplier: 1.0,
                    };
                    if LEV {
                        crate::unscale_log(&mut entry, m);
                        // The state's prices, exactly, rather than divided back.
                        entry.val_price = val_price_now;
                        entry.new_val_price = states[k].last_val_price;
                        entry.request.price = p.price;
                    }
                    sink.push_log(&entry);
                }
                states[k].position = position;
                states[k].stops = stops;
                if C::ON {
                    let mut result = result;
                    if LEV {
                        result.price /= m;
                    }
                    let ctx = Context {
                        group,
                        row: i,
                        cols: &cols,
                        cash,
                        free_cash,
                        value: group_value,
                        columns: &states,
                        mult: &mult,
                        scaled: false,
                        close: &args.close,
                        orders: &orders,
                    };
                    if cb.post_order(&ctx, k, &result, &mut cs)? == Flow::StopGroup {
                        stopped = true;
                        break 'orders;
                    }
                }
                if p.then {
                    let m = if LEV { mult[k] } else { 1.0 };
                    let over = if C::ON { decisions.get(k) } else { None };
                    let next = follow_up(args, i, col, &mut states[k], &p, m, sizes_checked, over)?;
                    if let Some(next) = next {
                        p = next;
                        continue;
                    }
                }
                break;
            }
        }
        if earnings {
            crate::cash::earn(
                &args.cash_earnings,
                i,
                cols.clone(),
                &mut cash,
                &mut free_cash,
                sink,
            )?;
        }
        if charges {
            crate::holding::charge(
                &args.rates(),
                i,
                cols.clone(),
                crate::holding::Columns {
                    states: &mut states,
                    margin: &margin,
                    isolated: &isolated,
                    mult: &mult,
                    scaled_val_price: false,
                },
                &mut cash,
                &mut free_cash,
                sink,
            )?;
        }
        if C::ON && !stopped {
            stopped = segment_end::<C, LEV>(
                args, cb, &mut cs, group, i, &cols, cash, free_cash, &states, &margin, &mult,
                &orders,
            )?;
        }
    }
    let ended = crate::snapshot::finish(cash, free_cash, states, stopped, args.row_offset);
    Ok((cs, ended))
}

/// Calls [`SignalFunc::adjust`] and [`SignalFunc::signals`] for column `k`:
/// its stops and its decision on bar `i`.
#[cold]
#[inline(never)]
fn decide<C: SignalFunc>(
    cb: &C,
    cs: &mut C::State,
    ctx: &Context<'_>,
    k: usize,
    args: &FromSignals<'_>,
    i: usize,
    col: usize,
) -> Result<(StopState, Decision)> {
    let mut stops = ctx.columns[k].stops;
    cb.adjust(ctx, k, &mut stops, cs)?;
    let mut decision = Decision {
        signals: read_signals(args, i, col),
        size: args.size.get(i, col),
        size_type: args.size_type.get(i, col),
    };
    cb.signals(ctx, k, &mut decision, cs)?;
    Ok((stops, decision))
}

/// Calls [`SignalFunc::post_segment`]; whether it stopped the group.
#[allow(clippy::too_many_arguments, reason = "the context's fields")]
#[cold]
#[inline(never)]
fn segment_end<C: SignalFunc, const LEV: bool>(
    args: &FromSignals<'_>,
    cb: &C,
    cs: &mut C::State,
    group: usize,
    row: usize,
    cols: &std::ops::Range<usize>,
    cash: f64,
    free_cash: f64,
    states: &[ColumnState],
    margin: &[bool],
    mult: &[f64],
    orders: &[ValidatedOrder],
) -> Result<bool> {
    let ctx = Context {
        group,
        row,
        cols,
        cash,
        free_cash,
        value: signal_value::<LEV>(cash, states, margin, mult),
        columns: states,
        mult,
        scaled: false,
        close: &args.close,
        orders,
    };
    Ok(cb.post_segment(&ctx, cs)? == Flow::StopGroup)
}

/// What a group is worth, as [`group_value`] has it, in either copy.
fn signal_value<const LEV: bool>(
    cash: f64,
    states: &[ColumnState],
    margin: &[bool],
    mult: &[f64],
) -> f64 {
    if LEV {
        return group_value(cash, states, margin, mult);
    }
    let mut value = cash;
    for st in states {
        if st.position != 0.0 {
            value += st.position * st.last_val_price;
        }
    }
    value
}

/// What a group's columns are, for [`liquidate`]: which trade on margin,
/// which of those are isolated, and each column's request for the bar.
struct Books<'b> {
    margin: &'b [bool],
    isolated: &'b [bool],
    mult: &'b [f64],
    orders: &'b [ValidatedOrder],
}

/// Tests the group for liquidation on bar `row` and closes what fails, after
/// the bar's stops and signals are read and its cross columns settled.
///
/// A stop that fired on the bar is credited: its column is tested at the
/// less adverse of its check price and the stop's level. A liquidated
/// column's pending order is dropped and its stops cleared — the order was
/// sized against a position that no longer exists. Returns whether any
/// order is still pending.
///
/// Out of line, behind a flag fixed for the group, as in `from_orders`.
#[allow(
    clippy::too_many_arguments,
    reason = "the group, its bar, its state and the scratch the test reuses"
)]
#[inline(never)]
fn liquidate<S: RecordSink + LogSink>(
    args: &FromSignals<'_>,
    group: usize,
    cols: std::ops::Range<usize>,
    row: usize,
    books: &Books<'_>,
    states: &mut [ColumnState],
    pending: &mut [Option<Pending>],
    cash: &mut f64,
    free_cash: &mut f64,
    to_close: &mut Vec<crate::liquidation::Close>,
    sink: &mut S,
    fills: &mut i64,
) -> Result<bool> {
    let (margin, isolated, mult) = (books.margin, books.isolated, books.mult);
    let inputs = args.liquidation();
    let start: Vec<f64> = states.iter().map(|st| st.position.abs()).collect();
    let mut on_path = Vec::new();
    let mut from = 0.0;
    let mut cross = false;
    for round in 0..crate::liquidation::ROUNDS {
        crate::liquidation::test(
            &inputs,
            row,
            cols.clone(),
            |k| crate::liquidation::Column {
                position: states[k].position,
                owed: states[k].owed(margin[k]),
                locked_cash: states[k].locked_cash,
                margin: margin[k],
                isolated: isolated[k],
                basis: args.margin_basis.get(row, cols.start + k),
                val_price: states[k].last_val_price * mult[k],
                stop: pending[k].map_or(f64::NAN, |p| p.stop_hit),
                multiplier: mult[k],
            },
            *cash,
            from,
            &mut on_path,
            to_close,
        )?;
        if to_close.is_empty() {
            break;
        }
        for c in to_close.iter() {
            let k = c.k;
            let col = cols.start + k;
            let st = states[k];
            let size = crate::liquidation::step_size(
                inputs.step(row, col)?,
                st.position,
                start[k],
                round + 1 == crate::liquidation::ROUNDS,
            );
            let before = ProcessOrderState {
                cash: *cash,
                position: st.position,
                debt: st.debt,
                locked_cash: st.locked_cash,
                free_cash: *free_cash,
                val_price: st.last_val_price * mult[k],
                value: group_value(*cash, states, margin, mult),
                oidx: 0,
                lidx: 0,
            };
            let at = crate::liquidation::At {
                group,
                row,
                col,
                template: books.orders[k].order(),
                isolated: isolated[k],
                margin: margin[k],
                multiplier: mult[k],
            };
            let after =
                crate::liquidation::fill(&inputs, &at, c.price, size, None, &before, sink, fills)?;
            let returned = after.free_cash - *free_cash;
            *cash = after.cash;
            *free_cash = after.free_cash;
            states[k].position = after.position;
            states[k].debt = after.debt;
            states[k].locked_cash = after.locked_cash;
            // Its orders were sized against what it held.
            states[k].stops = StopState::cleared();
            states[k].limit = LimitOrder::none();
            pending[k] = None;
            if !isolated[k] {
                cross = true;
            } else if after.position == 0.0 {
                crate::liquidation::write_off(returned, row, col, cash, free_cash, sink);
            }
            from = f64::max(from, c.alpha);
        }
    }
    if cross {
        crate::liquidation::write_off_account(
            *cash,
            states
                .iter()
                .zip(isolated)
                .map(|(st, &iso)| (st.position, st.locked_cash, iso)),
            row,
            cols.start,
            cash,
            free_cash,
            sink,
        );
    }
    if !matches!(args.auto_deleverage, FlexArray::Scalar(false)) {
        for (k, col) in cols.clone().enumerate() {
            let st = states[k];
            if !margin[k] || st.position == 0.0 || !args.auto_deleverage.get(row, col) {
                continue;
            }
            let mut price = args.mark_price.get(row, col);
            if price.is_nan() {
                price = args.close.get(row, col);
            }
            if price.is_nan() {
                continue;
            }
            let before = ProcessOrderState {
                cash: *cash,
                position: st.position,
                debt: st.debt,
                locked_cash: st.locked_cash,
                free_cash: *free_cash,
                val_price: st.last_val_price * mult[k],
                value: group_value(*cash, states, margin, mult),
                oidx: 0,
                lidx: 0,
            };
            let at = crate::liquidation::At {
                group,
                row,
                col,
                template: books.orders[k].order(),
                isolated: isolated[k],
                margin: true,
                multiplier: mult[k],
            };
            let after = crate::liquidation::fill(
                &inputs,
                &at,
                price * mult[k],
                f64::NAN,
                Some(0.0),
                &before,
                sink,
                fills,
            )?;
            *cash = after.cash;
            *free_cash = after.free_cash;
            states[k].position = after.position;
            states[k].debt = after.debt;
            states[k].locked_cash = after.locked_cash;
            states[k].stops = StopState::cleared();
            states[k].limit = LimitOrder::none();
            pending[k] = None;
        }
    }
    Ok(pending.iter().any(Option::is_some))
}

/// What a group is worth at its columns' valuation prices, as the loop
/// values it before a bar's orders.
fn group_value(cash: f64, states: &[ColumnState], margin: &[bool], mult: &[f64]) -> f64 {
    let mut value = cash;
    for (k, st) in states.iter().enumerate() {
        let val_price = st.last_val_price * mult[k];
        if margin[k] {
            value += crate::margin::unsettled(st.position, st.debt, val_price);
            continue;
        }
        if st.position != 0.0 {
            value += st.position * val_price;
        }
        value -= st.long_debt();
    }
    value
}

fn simulate_into<S: SinkBuffer, C: SignalFunc>(
    args: &FromSignals<'_>,
    cb: &C,
) -> Result<Run<S, C::State>> {
    let groups = args.resolved_groups()?;
    let mut sink = S::with_capacity(args.fill_bound(0..args.n_cols));
    let mut fills = 0_i64;
    let levered = args.run_levered()?;
    let mut states = Vec::new();
    let mut ended = Vec::new();
    for group in 0..groups.len() {
        let (cs, end) = simulate_group(args, &groups, group, levered, cb, &mut sink, &mut fills)?;
        states.push(cs);
        ended.push(end);
    }
    Ok(args.finish_run(sink, states, levered, ended))
}

/// Simulates every column on one thread.
///
/// # Errors
///
/// Propagates any invariant violation from the execution kernel or the signal
/// pipeline, and any rejection on a column whose `raise_reject` is set.
pub fn simulate_from_signals(args: &FromSignals<'_>) -> Result<OrderRecords> {
    Ok(simulate_into(args, &NoCallbacks)?.records)
}

/// Simulates every column, keeping the logs of the orders that asked for one.
///
/// Identical fills to [`simulate_from_signals`]; the logs are additional. One
/// row per order that reached the kernel, filled or not — see the `log` field
/// on [`FromSignals`] for why a quiet bar gets no row.
///
/// # Errors
///
/// As [`simulate_from_signals`].
pub fn simulate_from_signals_logged(args: &FromSignals<'_>) -> Result<Journal> {
    Ok(simulate_into(args, &NoCallbacks)?.records)
}

/// Simulates every column across `n_threads`, in column-range chunks.
///
/// Output is bit-identical to [`simulate_from_signals`]: each thread fills
/// its own buffer and the buffers are concatenated in column order, so record
/// ids never depend on thread interleaving.
///
/// # Errors
///
/// As [`simulate_from_signals`]. If several threads error, the one from the
/// lowest column chunk is returned, so the failure does not depend on timing.
pub fn simulate_from_signals_parallel(
    args: &FromSignals<'_>,
    n_threads: usize,
) -> Result<OrderRecords> {
    Ok(simulate_into_parallel(args, &NoCallbacks, n_threads)?.records)
}

/// Simulates every column across `n_threads`, keeping logs.
///
/// Output equals [`simulate_from_signals_logged`]'s, which for the logs means
/// the merge has to re-base each thread's `order_id`s — see
/// [`Journal::concat`].
///
/// # Errors
///
/// As [`simulate_from_signals_parallel`].
pub fn simulate_from_signals_logged_parallel(
    args: &FromSignals<'_>,
    n_threads: usize,
) -> Result<Journal> {
    Ok(simulate_into_parallel(args, &NoCallbacks, n_threads)?.records)
}

/// `from_signals` with callbacks: `cb` adjusts each column's stops and
/// decides its signals and size, bar by bar; see [`crate::callbacks`].
/// Groups run across `n_threads` (1: serial), as without callbacks. Returns
/// the records, plain ([`OrderRecords`]) or with logs ([`Journal`]), and each
/// group's callback state in group order.
///
/// A callback's decision goes through the whole signal pipeline: conflicts,
/// accumulation, stops, limit orders and `order_mode` as with the arrays.
///
/// # Errors
///
/// As [`simulate_from_signals`], and any error a callback returns. `from_ago`
/// other than 0 is an error: a callback decides on its own bar.
pub fn simulate_from_signal_func<S: SinkBuffer, C: SignalFunc + Sync>(
    args: &FromSignals<'_>,
    cb: &C,
    n_threads: usize,
) -> Result<Run<S, C::State>> {
    simulate_into_parallel(args, cb, n_threads)
}

/// `from_signals` returning its state as well as its records, to continue
/// it later (see [`crate::snapshot`]); groups across `n_threads`.
///
/// # Errors
///
/// As [`simulate_from_signals`], and a `resume` that does not fit the run.
pub fn simulate_from_signals_run<S: SinkBuffer>(
    args: &FromSignals<'_>,
    n_threads: usize,
) -> Result<Run<S>> {
    simulate_into_parallel(args, &NoCallbacks, n_threads)
}

/// Simulates every column across `n_threads` into any sink.
///
/// # Errors
///
/// As [`simulate_from_signals_parallel`].
fn simulate_into_parallel<S: SinkBuffer, C: SignalFunc + Sync>(
    args: &FromSignals<'_>,
    cb: &C,
    n_threads: usize,
) -> Result<Run<S, C::State>> {
    let groups = args.resolved_groups()?;
    let n_groups = groups.len();
    let n_threads = n_threads.max(1).min(n_groups.max(1));
    if n_threads == 1 {
        return simulate_into(args, cb);
    }

    let ranges = groups.chunks(n_threads);
    let levered = args.run_levered()?;
    type Part<S, T> = (S, Vec<(T, GroupState)>);
    let results: Vec<Result<Part<S, C::State>>> = std::thread::scope(|scope| {
        let handles: Vec<_> = ranges
            .iter()
            .map(|&(start, end)| {
                let groups = &groups;
                scope.spawn(move || {
                    let cols = groups.range(start).start..groups.range(end - 1).end;
                    let mut sink = S::with_capacity(args.fill_bound(cols));
                    // Per buffer, so ids do not depend on interleaving.
                    let mut fills = 0_i64;
                    let mut ended = Vec::new();
                    for group in start..end {
                        ended.push(simulate_group(
                            args, groups, group, levered, cb, &mut sink, &mut fills,
                        )?);
                    }
                    Ok((sink, ended))
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err(Error::Invariant("simulation thread panicked".into())))
            })
            .collect()
    });

    // Report the lowest-numbered failure so errors are deterministic.
    let mut parts = Vec::with_capacity(results.len());
    let mut states = Vec::with_capacity(n_groups);
    let mut ended = Vec::with_capacity(n_groups);
    for result in results {
        let (sink, part) = result?;
        parts.push(sink);
        for (cs, end) in part {
            states.push(cs);
            ended.push(end);
        }
    }
    Ok(args.finish_run(S::merge(parts), states, levered, ended))
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "these assert on exact fill prices and sizes, which is the contract"
)]
mod tests {
    use super::*;

    fn prices(v: &[f64]) -> Vec<f64> {
        v.to_vec()
    }

    #[test]
    fn an_entry_then_an_exit_makes_one_round_trip() {
        let close = prices(&[10.0, 11.0, 12.0]);
        let entries = [true, false, false];
        let exits = [false, false, true];
        let args = FromSignals {
            n_rows: 3,
            n_cols: 1,
            init_cash: FlexArray::scalar(1000.0),
            close: FlexArray::per_row(&close, 3).unwrap(),
            entries: FlexArray::per_row(&entries, 3).unwrap(),
            exits: FlexArray::per_row(&exits, 3).unwrap(),
            size: FlexArray::scalar(5.0),
            use_stops: false,
            ..FromSignals::default()
        };
        let r = simulate_from_signals(&args).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r.idx, [0, 2]);
        assert_eq!(r.size, [5.0, 5.0]);
        assert_eq!(r.price, [10.0, 12.0]);
    }

    #[test]
    fn a_stop_loss_fills_at_the_stop_when_the_bar_straddles_it() {
        // Enter at 10 with a 10% stop, so the level is 9. The next bar's
        // range contains 9, so that is the fill.
        let close = prices(&[10.0, 9.5]);
        let low = prices(&[10.0, 8.0]);
        let high = prices(&[10.0, 10.0]);
        let open = prices(&[10.0, 9.8]);
        let entries = [true, false];
        let args = FromSignals {
            n_rows: 2,
            n_cols: 1,
            init_cash: FlexArray::scalar(1000.0),
            open: FlexArray::per_row(&open, 2).unwrap(),
            high: FlexArray::per_row(&high, 2).unwrap(),
            low: FlexArray::per_row(&low, 2).unwrap(),
            close: FlexArray::per_row(&close, 2).unwrap(),
            entries: FlexArray::per_row(&entries, 2).unwrap(),
            size: FlexArray::scalar(5.0),
            sl_stop: FlexArray::scalar(0.1),
            ..FromSignals::default()
        };
        let r = simulate_from_signals(&args).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r.price[1], 9.0, "filled at the stop level");
    }

    #[test]
    fn a_gap_through_the_stop_fills_at_the_open() {
        // The same stop at 9, but the bar opens at 7. A real account gets 7,
        // not 9 — this branch is what keeps a backtest honest about gaps.
        let close = prices(&[10.0, 7.5]);
        let low = prices(&[10.0, 7.0]);
        let high = prices(&[10.0, 8.0]);
        let open = prices(&[10.0, 7.0]);
        let entries = [true, false];
        let args = FromSignals {
            n_rows: 2,
            n_cols: 1,
            init_cash: FlexArray::scalar(1000.0),
            open: FlexArray::per_row(&open, 2).unwrap(),
            high: FlexArray::per_row(&high, 2).unwrap(),
            low: FlexArray::per_row(&low, 2).unwrap(),
            close: FlexArray::per_row(&close, 2).unwrap(),
            entries: FlexArray::per_row(&entries, 2).unwrap(),
            size: FlexArray::scalar(5.0),
            sl_stop: FlexArray::scalar(0.1),
            ..FromSignals::default()
        };
        let r = simulate_from_signals(&args).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r.price[1], 7.0, "gapped through: filled at the open");
    }

    #[test]
    fn a_take_profit_fires_on_the_way_up() {
        let close = prices(&[10.0, 11.5]);
        let low = prices(&[10.0, 10.5]);
        let high = prices(&[10.0, 12.0]);
        let open = prices(&[10.0, 10.5]);
        let entries = [true, false];
        let args = FromSignals {
            n_rows: 2,
            n_cols: 1,
            init_cash: FlexArray::scalar(1000.0),
            open: FlexArray::per_row(&open, 2).unwrap(),
            high: FlexArray::per_row(&high, 2).unwrap(),
            low: FlexArray::per_row(&low, 2).unwrap(),
            close: FlexArray::per_row(&close, 2).unwrap(),
            entries: FlexArray::per_row(&entries, 2).unwrap(),
            size: FlexArray::scalar(5.0),
            tp_stop: FlexArray::scalar(0.1),
            ..FromSignals::default()
        };
        let r = simulate_from_signals(&args).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r.price[1], 11.0);
    }

    #[test]
    fn a_trailing_stop_follows_the_high_but_only_from_the_next_bar() {
        // Bar 1 makes a high of 20 and closes at 19; a stop trailing 10%
        // would sit at 18 *after* that bar, so bar 1 itself must not stop
        // out. Bar 2 dips to 17 and does.
        let close = prices(&[10.0, 19.0, 17.5]);
        let open = prices(&[10.0, 11.0, 19.0]);
        let high = prices(&[10.0, 20.0, 19.0]);
        let low = prices(&[10.0, 11.0, 17.0]);
        let entries = [true, false, false];
        let args = FromSignals {
            n_rows: 3,
            n_cols: 1,
            init_cash: FlexArray::scalar(1000.0),
            open: FlexArray::per_row(&open, 3).unwrap(),
            high: FlexArray::per_row(&high, 3).unwrap(),
            low: FlexArray::per_row(&low, 3).unwrap(),
            close: FlexArray::per_row(&close, 3).unwrap(),
            entries: FlexArray::per_row(&entries, 3).unwrap(),
            size: FlexArray::scalar(5.0),
            sl_stop: FlexArray::scalar(0.1),
            sl_trail: FlexArray::scalar(true),
            ..FromSignals::default()
        };
        let r = simulate_from_signals(&args).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r.idx[1], 2, "stopped out on bar 2, not bar 1");
        assert_eq!(r.price[1], 18.0, "trailed to 10% below the high of 20");
    }

    #[test]
    fn a_stop_does_not_fire_before_a_position_exists() {
        let close = prices(&[10.0, 1.0, 1.0]);
        let entries = [false, false, true];
        let args = FromSignals {
            n_rows: 3,
            n_cols: 1,
            init_cash: FlexArray::scalar(1000.0),
            close: FlexArray::per_row(&close, 3).unwrap(),
            entries: FlexArray::per_row(&entries, 3).unwrap(),
            size: FlexArray::scalar(5.0),
            sl_stop: FlexArray::scalar(0.1),
            ..FromSignals::default()
        };
        let r = simulate_from_signals(&args).unwrap();
        assert_eq!(r.len(), 1, "only the entry");
    }

    #[test]
    fn closing_a_position_clears_its_stops() {
        // Enter, exit on a signal, then re-enter. If the stop from the first
        // position survived, the re-entry would immediately stop out.
        let close = prices(&[10.0, 10.0, 10.0, 10.0]);
        let entries = [true, false, true, false];
        let exits = [false, true, false, false];
        let args = FromSignals {
            n_rows: 4,
            n_cols: 1,
            init_cash: FlexArray::scalar(1000.0),
            close: FlexArray::per_row(&close, 4).unwrap(),
            entries: FlexArray::per_row(&entries, 4).unwrap(),
            exits: FlexArray::per_row(&exits, 4).unwrap(),
            size: FlexArray::scalar(5.0),
            sl_stop: FlexArray::scalar(0.1),
            ..FromSignals::default()
        };
        let r = simulate_from_signals(&args).unwrap();
        assert_eq!(r.len(), 3, "entry, exit, entry");
    }

    #[test]
    fn a_two_sided_column_reads_an_exit_as_a_short_entry() {
        let close = prices(&[10.0, 10.0]);
        let entries = [true, false];
        let exits = [false, true];
        let args = FromSignals {
            n_rows: 2,
            n_cols: 1,
            init_cash: FlexArray::scalar(1000.0),
            close: FlexArray::per_row(&close, 2).unwrap(),
            entries: FlexArray::per_row(&entries, 2).unwrap(),
            exits: FlexArray::per_row(&exits, 2).unwrap(),
            direction: FlexArray::scalar(Direction::Both),
            size: FlexArray::scalar(5.0),
            use_stops: false,
            ..FromSignals::default()
        };
        let r = simulate_from_signals(&args).unwrap();
        assert_eq!(r.len(), 2);
        // Bar 1 reverses: 5 long closed and 5 short opened in one fill.
        assert_eq!(r.size[1], 10.0);
    }

    #[test]
    fn no_signals_means_no_records() {
        let close = prices(&[10.0, 11.0]);
        let args = FromSignals {
            n_rows: 2,
            n_cols: 1,
            close: FlexArray::per_row(&close, 2).unwrap(),
            ..FromSignals::default()
        };
        assert!(simulate_from_signals(&args).unwrap().is_empty());
    }

    #[test]
    fn the_parallel_path_is_identical_to_the_serial_one() {
        let n_rows = 40;
        let n_cols = 9;
        let close: Vec<f64> = (0..n_rows * n_cols)
            .map(|k| 10.0 + ((k * 7) % 13) as f64)
            .collect();
        let entries: Vec<bool> = (0..n_rows * n_cols).map(|k| k % 5 == 0).collect();
        let exits: Vec<bool> = (0..n_rows * n_cols).map(|k| k % 7 == 0).collect();
        let args = FromSignals {
            n_rows,
            n_cols,
            init_cash: FlexArray::scalar(1000.0),
            close: FlexArray::full(&close, n_rows, n_cols).unwrap(),
            entries: FlexArray::full(&entries, n_rows, n_cols).unwrap(),
            exits: FlexArray::full(&exits, n_rows, n_cols).unwrap(),
            size: FlexArray::scalar(3.0),
            sl_stop: FlexArray::scalar(0.05),
            ..FromSignals::default()
        };
        let serial = simulate_from_signals(&args).unwrap();
        assert!(!serial.is_empty(), "the workload must actually trade");
        for threads in [2, 3, 4, 8] {
            let parallel = simulate_from_signals_parallel(&args, threads).unwrap();
            assert_eq!(serial, parallel, "with {threads} threads");
        }
    }

    #[test]
    fn a_negative_stop_distance_is_rejected() {
        assert!(stop_price(5.0, 10.0, -0.1, 10.0, 9.0, 11.0, true).is_err());
    }

    /// Five bars, one column: enter, nothing, exit, enter, nothing.
    fn signalled() -> (Vec<f64>, Vec<bool>, Vec<bool>) {
        (
            vec![10.0, 11.0, 12.0, 11.0, 13.0],
            vec![true, false, false, true, false],
            vec![false, false, true, false, false],
        )
    }

    fn signal_args<'a>(
        close: &'a [f64],
        entries: &'a [bool],
        exits: &'a [bool],
        log: bool,
    ) -> FromSignals<'a> {
        FromSignals {
            n_rows: 5,
            n_cols: 1,
            init_cash: FlexArray::Scalar(100.0),
            close: FlexArray::full(close, 5, 1).unwrap(),
            entries: FlexArray::full(entries, 5, 1).unwrap(),
            exits: FlexArray::full(exits, 5, 1).unwrap(),
            log: FlexArray::Scalar(log),
            ..FromSignals::default()
        }
    }

    #[test]
    fn a_quiet_bar_gets_no_log_row() {
        // The rule, and the one design decision in the signal loop's logging:
        // a row per order that reached the kernel, not a row per bar. Bars 1
        // and 4 resolve to no order at all, so there is nothing to explain
        // about them — and a row per bar would bury the three that matter under
        // the ones where the strategy simply had no opinion.
        let (close, entries, exits) = signalled();
        let args = signal_args(&close, &entries, &exits, true);
        let journal = simulate_from_signals_logged(&args).unwrap();

        assert_eq!(journal.logs.len(), 3, "one row per order considered");
        assert_eq!(journal.logs.idx, [0, 2, 3], "the bars that submitted");
        assert_eq!(journal.orders.len(), 3);
        assert!(journal.logs.reasons().iter().all(|r| *r == "filled"));
        // Every log carries the request beside the result, which is the whole
        // point: an order record shows a fill and says nothing about the size
        // that was asked for.
        assert!(journal.logs.req_size[0].is_infinite());
        assert_eq!(journal.logs.res_size[0], 10.0);
        assert!(journal.logs.req_log.iter().all(|f| *f));
    }

    #[test]
    fn a_log_says_why_a_signal_did_not_trade() {
        // A signal that fires against no cash. Without the log the evidence is
        // an absent order record, which is equally consistent with the signal
        // never having fired.
        let (close, entries, exits) = signalled();
        let args = FromSignals {
            init_cash: FlexArray::Scalar(0.0),
            ..signal_args(&close, &entries, &exits, true)
        };
        let journal = simulate_from_signals_logged(&args).unwrap();

        assert!(journal.orders.is_empty(), "no cash, so no fills");
        assert_eq!(journal.logs.len(), 2, "two entries were still considered");
        assert_eq!(
            journal.logs.reasons()[0],
            alphax_model::StatusInfo::NoCashLong.describe()
        );
        assert!(
            journal.logs.order_id.iter().all(|id| *id == -1),
            "nothing filled, so nothing to join to"
        );
    }

    #[test]
    fn logging_does_not_change_what_filled() {
        // Additional, never alternative. An investigation that changed the run
        // would be investigating something else.
        let (close, entries, exits) = signalled();
        let quiet = simulate_from_signals(&signal_args(&close, &entries, &exits, false)).unwrap();
        let loud =
            simulate_from_signals_logged(&signal_args(&close, &entries, &exits, true)).unwrap();
        assert_eq!(quiet, loud.orders);
        assert!(
            !loud.logs.is_empty(),
            "nothing was logged, so this proves nothing"
        );
    }

    #[test]
    fn a_logged_parallel_run_equals_the_serial_one() {
        // Six columns so the chunking actually splits, and staggered signals so
        // each column's fills land at different indices — which is what makes a
        // mis-based `order_id` visible rather than accidentally right.
        let n_rows = 6;
        let n_cols = 6;
        let mut close = vec![0.0; n_rows * n_cols];
        let mut entries = vec![false; n_rows * n_cols];
        let mut exits = vec![false; n_rows * n_cols];
        for c in 0..n_cols {
            for r in 0..n_rows {
                close[c * n_rows + r] = 10.0 + r as f64;
            }
            entries[c * n_rows + c % n_rows] = true;
            exits[c * n_rows + (c + 2) % n_rows] = true;
        }
        let args = FromSignals {
            n_rows,
            n_cols,
            init_cash: FlexArray::Scalar(1000.0),
            close: FlexArray::full(&close, n_rows, n_cols).unwrap(),
            entries: FlexArray::full(&entries, n_rows, n_cols).unwrap(),
            exits: FlexArray::full(&exits, n_rows, n_cols).unwrap(),
            log: FlexArray::Scalar(true),
            ..FromSignals::default()
        };
        let serial = simulate_from_signals_logged(&args).unwrap();
        assert!(serial.orders.len() > 3, "too few fills to prove anything");

        // Each filled log must name a record in its own column and bar. The
        // check is on what is pointed at, not on the number, so it tests the
        // join rather than restating how it is computed.
        for i in 0..serial.logs.len() {
            let id = serial.logs.order_id[i];
            if id < 0 {
                continue;
            }
            let at = id as usize;
            assert_eq!(serial.orders.col[at], serial.logs.col[i], "log {i}");
            assert_eq!(serial.orders.idx[at], serial.logs.idx[i], "log {i}");
        }

        for threads in [2, 3, 6] {
            let parallel = simulate_from_signals_logged_parallel(&args, threads).unwrap();
            assert_eq!(serial.orders, parallel.orders, "{threads} threads");
            // Through `Debug`, because a log holding `NaN` is never `==` itself.
            assert_eq!(
                format!("{:?}", serial.logs),
                format!("{:?}", parallel.logs),
                "{threads} threads changed the logs"
            );
        }
    }
}
