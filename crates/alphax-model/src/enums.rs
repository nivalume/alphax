//! Execution state-machine enums.
//!
//! Discriminants are pinned because they are written into record arrays and
//! read back by the Python shell and the golden tests. Changing one is a
//! breaking change to the on-disk format, not just to the API.

/// How a requested size should be interpreted.
///
/// These are resolved by a strictly ordered cascade in `alphax-exec`, each stage
/// rewriting the size type until only [`SizeType::Amount`] is left. The order
/// matters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum SizeType {
    /// A number of units.
    Amount = 0,
    /// A cash value; divided by the valuation price to get units.
    Value = 1,
    /// A fraction of available cash, in `[0, 1]`.
    Percent = 2,
    /// Target a number of units; the delta against the current position.
    TargetAmount = 3,
    /// Target a cash value.
    TargetValue = 4,
    /// Target a fraction of portfolio value.
    TargetPercent = 5,
    /// A fraction of portfolio value, as a cash value: `Value` of
    /// `size × value`. Negative sells.
    ValuePercent = 6,
    /// `Percent`, written in percent: 50 is half.
    Percent100 = 7,
    /// `ValuePercent`, written in percent.
    ValuePercent100 = 8,
    /// `TargetPercent`, written in percent.
    TargetPercent100 = 9,
}

/// Which side of the market a column may take.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum Direction {
    /// Long positions only; shorting is rejected.
    LongOnly = 0,
    /// Short positions only.
    ShortOnly = 1,
    /// Both, including reversals through zero.
    Both = 2,
}

/// How a leveraged buy is financed.
///
/// Both modes cap a buy at the same size — own cash covers `1 / leverage` of
/// the notional plus the fees — and differ only in how much of it is borrowed.
/// A short is unaffected: its collateral is `value / leverage` in either mode,
/// because posting more than the requirement would only move free cash into a
/// reserve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(i8)]
pub enum LeverageMode {
    /// Spend own cash first and borrow only the part it cannot cover.
    #[default]
    Lazy = 0,
    /// Borrow `1 - 1 / leverage` of every buy, however much cash is free.
    Eager = 1,
}

/// How a column's trades are paid for.
///
/// Set per column, so one account group can hold spot columns next to futures
/// or perpetual columns and share their cash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(i8)]
pub enum AccountType {
    /// A long pays its price in full, a short ties up its proceeds and its
    /// collateral. `leverage` borrows part of a long's price.
    #[default]
    Cash = 0,
    /// Either side only posts initial margin out of free cash, and the
    /// position's profit or loss is settled into cash every bar: futures,
    /// perpetuals, CFDs.
    Margin = 1,
}

/// What a margin column's `initial_margin` is measured in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(i8)]
pub enum MarginBasis {
    /// A fraction of the notional at the fill price, as perpetuals, forex and
    /// CFDs quote it. Left `NaN`, it is `1 / leverage`.
    #[default]
    Notional = 0,
    /// A fixed amount of cash per unit of size, as futures exchanges publish
    /// it per contract.
    Unit = 1,
}

/// Which margin a margin column's position may draw on.
///
/// Set per column, for the whole run, like [`AccountType`]; a cash column
/// ignores it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(i8)]
pub enum MarginMode {
    /// The account's: the position's profit settles into the account's cash
    /// every bar, and the account is liquidated as a whole when its equity
    /// falls below the maintenance margin of its cross positions.
    #[default]
    Cross = 0,
    /// Its own: the position is not settled into the account's cash, is
    /// liquidated alone when its margin plus its profit falls below its
    /// maintenance margin, and can lose at most the margin it posted.
    Isolated = 1,
}

/// Which side an order executed on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum OrderSide {
    /// Bought.
    Buy = 0,
    /// Sold.
    Sell = 1,
}

/// Outcome of an order request.
///
/// Note that [`OrderStatus::Rejected`] is an ordinary outcome carried in the
/// return value, **not** an `Err`. Errors are reserved for invariant
/// violations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum OrderStatus {
    /// Executed, in full or in part.
    Filled = 0,
    /// Nothing to do (no signal, zero size).
    Ignored = 1,
    /// Could not execute; see the accompanying [`StatusInfo`].
    Rejected = 2,
}

/// Why an order was ignored or rejected.
///
/// Fifteen distinct reasons, kept separate because "why did this backtest
/// not trade" is the most common question a user brings to a sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum StatusInfo {
    /// Size was `NaN`.
    SizeNaN = 0,
    /// Price was `NaN`.
    PriceNaN = 1,
    /// Valuation price was `NaN`.
    ValPriceNaN = 2,
    /// Portfolio value was `NaN`.
    ValueNaN = 3,
    /// Portfolio value was zero or negative.
    ValueZeroNeg = 4,
    /// Size resolved to zero.
    SizeZero = 5,
    /// Not enough cash to open or extend a short.
    NoCashShort = 6,
    /// Not enough cash to open or extend a long.
    NoCashLong = 7,
    /// Tried to close a position that is not open.
    NoOpenPosition = 8,
    /// Size exceeded the configured maximum.
    MaxSizeExceeded = 9,
    /// Randomly rejected via `reject_prob`.
    RandomEvent = 10,
    /// Proceeds would not cover fees.
    CantCoverFees = 11,
    /// Size fell below the configured minimum.
    MinSizeNotReached = 12,
    /// Partial fill was required but not allowed.
    PartialFill = 13,
    /// No own cash to put up for a leveraged buy.
    NoCashLeverage = 14,
}

impl StatusInfo {
    /// Recovers a reason from the discriminant a record stored.
    ///
    /// Records keep the discriminant rather than the enum, because the layout
    /// has to survive a round trip through Arrow or a file. This is the way
    /// back, and it returns `None` for anything that is not a reason — which
    /// includes the `-1` a log writes for an order that filled and therefore
    /// had nothing to explain.
    #[must_use]
    pub const fn from_discriminant(value: i64) -> Option<Self> {
        Some(match value {
            0 => Self::SizeNaN,
            1 => Self::PriceNaN,
            2 => Self::ValPriceNaN,
            3 => Self::ValueNaN,
            4 => Self::ValueZeroNeg,
            5 => Self::SizeZero,
            6 => Self::NoCashShort,
            7 => Self::NoCashLong,
            8 => Self::NoOpenPosition,
            9 => Self::MaxSizeExceeded,
            10 => Self::RandomEvent,
            11 => Self::CantCoverFees,
            12 => Self::MinSizeNotReached,
            13 => Self::PartialFill,
            14 => Self::NoCashLeverage,
            _ => return None,
        })
    }

    /// A human-readable reason, for reports and error messages.
    #[must_use]
    pub const fn describe(self) -> &'static str {
        match self {
            Self::SizeNaN => "size is NaN",
            Self::PriceNaN => "price is NaN",
            Self::ValPriceNaN => "valuation price is NaN",
            Self::ValueNaN => "portfolio value is NaN",
            Self::ValueZeroNeg => "portfolio value is zero or negative",
            Self::SizeZero => "size is zero",
            Self::NoCashShort => "not enough cash to short",
            Self::NoCashLong => "not enough cash to go long",
            Self::NoOpenPosition => "no open position to close",
            Self::MaxSizeExceeded => "size exceeds the maximum",
            Self::RandomEvent => "randomly rejected",
            Self::CantCoverFees => "proceeds do not cover fees",
            Self::MinSizeNotReached => "size is below the minimum",
            Self::PartialFill => "partial fill required but not allowed",
            Self::NoCashLeverage => "no own cash to put up for a leveraged buy",
        }
    }
}

/// Whether a same-direction signal adds to a position or replaces it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum AccumulationMode {
    /// Signals never change an existing position's size.
    Disabled = 0,
    /// Signals both add to and remove from a position.
    Both = 1,
    /// Signals only add.
    AddOnly = 2,
    /// Signals only remove.
    RemoveOnly = 3,
}

/// How to resolve an entry and an exit firing on the same bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum ConflictMode {
    /// Do nothing.
    Ignore = 0,
    /// The entry wins.
    Entry = 1,
    /// The exit wins.
    Exit = 2,
    /// Whichever is adjacent to the current position wins.
    Adjacent = 3,
    /// Whichever is opposite to the current position wins.
    Opposite = 4,
}

/// How to resolve a long and a short signal firing on the same bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum DirectionConflictMode {
    /// Do nothing.
    Ignore = 0,
    /// Long wins.
    Long = 1,
    /// Short wins.
    Short = 2,
    /// Whichever matches the current position wins.
    Adjacent = 3,
    /// Whichever opposes the current position wins.
    Opposite = 4,
}

/// What an entry signal opposite to the open position should do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum OppositeEntryMode {
    /// Do nothing.
    Ignore = 0,
    /// Close the position.
    Close = 1,
    /// Close, then apply accumulation rules.
    CloseReduce = 2,
    /// Close and open the opposite position.
    Reverse = 3,
    /// Reverse, then apply accumulation rules.
    ReverseReduce = 4,
}

/// Which price a stop is anchored to when the position opens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum StopEntryPrice {
    /// The valuation price.
    ValPrice = 0,
    /// The requested order price.
    Price = 1,
    /// The actual fill price.
    FillPrice = 2,
    /// The bar's close.
    Close = 3,
}

/// Which price a triggered stop fills at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum StopExitPrice {
    /// The stop price itself, with no slippage.
    StopLimit = 0,
    /// The stop price, plus slippage.
    StopMarket = 1,
    /// The configured order price.
    Price = 2,
    /// The bar's close.
    Close = 3,
}

/// What a triggered stop does to the position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum StopExitMode {
    /// Close it.
    Close = 0,
    /// Close, then apply accumulation rules.
    CloseReduce = 1,
    /// Reverse it.
    Reverse = 2,
    /// Reverse, then apply accumulation rules.
    ReverseReduce = 3,
}

/// Whether a new stop value replaces one that is already live.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum StopUpdateMode {
    /// Keep the live stop.
    Keep = 0,
    /// Replace it, unless the new value is `NaN`.
    Override = 1,
    /// Replace it, even with `NaN` (which clears the stop).
    OverrideNaN = 2,
}

/// Whether a signal's order executes on its bar or rests as a limit order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum OrderType {
    /// Execute at the order price, with slippage.
    Market = 0,
    /// Rest at a limit price until the market reaches it, its time runs out,
    /// or something else fills in the column. No slippage.
    Limit = 1,
}

/// What a limit order fills at once the market reaches it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum LimitOrderPrice {
    /// The limit price, or the open when the bar opened beyond it.
    Limit = 0,
    /// The limit price, even when the bar opened beyond it.
    HardLimit = 1,
    /// The bar's close.
    Close = 2,
}

/// How a limit order's `limit_delta` is read against the order price.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum DeltaFormat {
    /// A fraction of the order price.
    Percent = 0,
    /// An amount of price.
    Absolute = 1,
    /// The limit price itself.
    Target = 2,
    /// A percentage of the order price: `Percent` written in percent.
    Percent100 = 3,
}

/// What happens when a bar's signal meets an order already pending in its
/// column: a resting limit order, or a stop that fired on the bar. The first
/// half is the pending order's fate, the second the signal's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum PendingConflictMode {
    /// Keep the pending order; ignore the signal.
    KeepIgnore = 0,
    /// Keep the pending order; execute the signal too.
    KeepExecute = 1,
    /// Cancel the pending order; ignore the signal.
    CancelIgnore = 2,
    /// Cancel the pending order; execute the signal.
    CancelExecute = 3,
}

/// Which of the bar's prices an order executes at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum PriceType {
    /// The `price` input, with its `+inf` and `-inf` conventions.
    Price = 0,
    /// The bar's open.
    Open = 1,
    /// The bar's close.
    Close = 2,
}

/// What to do with an order price outside the bar's low–high range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum PriceAreaVioMode {
    /// Execute at it anyway.
    Ignore = 0,
    /// Move it to the nearer of the low and the high.
    Cap = 1,
    /// Stop the run with an error.
    Error = 2,
}

/// How a stop ladder splits a position across its steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum StopLadderMode {
    /// No ladder: a stop is one level and closes the whole position.
    Disabled = 0,
    /// Each step closes an equal share of the position it was armed on.
    Uniform = 1,
    /// Each step closes a share proportional to its distance from the last.
    Weighted = 2,
    /// Each step closes an equal share of what is left: the position when
    /// it fires over the steps still to fire.
    AdaptUniform = 3,
    /// Each step closes a share of what is left proportional to its
    /// distance from the last over the distance still to go.
    AdaptWeighted = 4,
}

/// Which side a trade was on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum TradeDirection {
    /// Long.
    Long = 0,
    /// Short.
    Short = 1,
}

/// Whether a trade is still open at the end of the simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum TradeStatus {
    /// Still open.
    Open = 0,
    /// Closed.
    Closed = 1,
}

/// Which view of the trade records to present.
///
/// All three share one record schema and differ only in how fills are
/// attributed. Getting this distinction wrong is the most common
/// mistake.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum TradesType {
    /// One record per entry fill; exits allocated back FIFO.
    EntryTrades = 0,
    /// One record per exit fill; entries averaged into it.
    ExitTrades = 1,
    /// Trades sharing a parent aggregated into one position.
    Positions = 2,
}

/// Why cash moved without a fill.
///
/// Stored in alphax-records' `CashRecords` as the discriminant, so the analysis
/// layer can tell money brought into the account from money the account
/// earned: a deposit changes the value without being a return, every other
/// kind is part of the return.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum CashFlowKind {
    /// Cash added to or withdrawn from the account at the bar's open.
    Deposit = 0,
    /// Interest, fees or any other income or cost, credited at the bar's close.
    Earning = 1,
    /// A per-share dividend times the position held at the bar's open. A short
    /// position pays it.
    Dividend = 2,
    /// A perpetual's funding payment at the bar's close: the position's
    /// notional at the mark price times the funding rate, paid by longs and
    /// received by shorts when the rate is positive.
    Funding = 3,
    /// Overnight financing at the bar's close, as a broker charges on a
    /// forex, CFD or futures position: the notional at the mark price times
    /// the rate for the position's side.
    Financing = 4,
    /// Interest at the bar's close on the cash a leveraged long borrowed.
    Interest = 5,
    /// The fee for borrowing the shares a cash column's short sold, at the
    /// bar's close: the short's value at the mark price times the rate.
    BorrowFee = 6,
    /// A loss a liquidation left that the account cannot pay, written off as
    /// an exchange's insurance fund takes it: positive, and booked when an
    /// isolated column's margin or a cross account with nothing left open
    /// would otherwise end below zero.
    Liquidation = 7,
    /// The cost of a cash column's `init_position` at its `init_price`, on
    /// the first bar: it balances the fill that opens the position, so the
    /// cash stays `init_cash`. Like a deposit it is capital brought in, not
    /// a return.
    InitialPosition = 8,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discriminants_are_pinned() {
        // These are written into record arrays; changing one is a breaking
        // change to the on-disk format.
        assert_eq!(SizeType::TargetPercent as i8, 5);
        assert_eq!(SizeType::ValuePercent as i8, 6);
        assert_eq!(SizeType::TargetPercent100 as i8, 9);
        assert_eq!(Direction::Both as i8, 2);
        assert_eq!(OrderStatus::Rejected as i8, 2);
        assert_eq!(StatusInfo::PartialFill as i8, 13);
        assert_eq!(TradesType::Positions as i8, 2);
        assert_eq!(CashFlowKind::Deposit as i8, 0);
        assert_eq!(CashFlowKind::Earning as i8, 1);
        assert_eq!(CashFlowKind::Dividend as i8, 2);
        assert_eq!(CashFlowKind::Funding as i8, 3);
        assert_eq!(CashFlowKind::Financing as i8, 4);
        assert_eq!(CashFlowKind::Interest as i8, 5);
        assert_eq!(CashFlowKind::BorrowFee as i8, 6);
        assert_eq!(CashFlowKind::Liquidation as i8, 7);
        assert_eq!(CashFlowKind::InitialPosition as i8, 8);
        assert_eq!(MarginMode::Isolated as i8, 1);
        assert_eq!(AccountType::Margin as i8, 1);
        assert_eq!(MarginBasis::Unit as i8, 1);
        assert_eq!(OrderType::Limit as i8, 1);
        assert_eq!(LimitOrderPrice::Close as i8, 2);
        assert_eq!(DeltaFormat::Target as i8, 2);
        assert_eq!(DeltaFormat::Percent100 as i8, 3);
        assert_eq!(PendingConflictMode::CancelExecute as i8, 3);
        assert_eq!(PriceType::Close as i8, 2);
        assert_eq!(PriceAreaVioMode::Error as i8, 2);
        assert_eq!(StopLadderMode::Weighted as i8, 2);
        assert_eq!(StopLadderMode::AdaptWeighted as i8, 4);
    }

    #[test]
    fn every_status_info_has_a_description() {
        let all = [
            StatusInfo::SizeNaN,
            StatusInfo::PriceNaN,
            StatusInfo::ValPriceNaN,
            StatusInfo::ValueNaN,
            StatusInfo::ValueZeroNeg,
            StatusInfo::SizeZero,
            StatusInfo::NoCashShort,
            StatusInfo::NoCashLong,
            StatusInfo::NoOpenPosition,
            StatusInfo::MaxSizeExceeded,
            StatusInfo::RandomEvent,
            StatusInfo::CantCoverFees,
            StatusInfo::MinSizeNotReached,
            StatusInfo::PartialFill,
        ];
        assert_eq!(all.len(), 14);
        for s in all {
            assert!(!s.describe().is_empty());
        }
    }
}
