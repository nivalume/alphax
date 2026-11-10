//! Turning Python's strings into the engine's enums.
//!
//! Small, and tested one variant at a time on purpose. A mapping table is the
//! kind of code that is obviously right while being wrong in one row —
//! `"targetvalue"` landing on `TargetAmount` compiles, runs, and produces a
//! backtest that is merely different. Nothing downstream can notice, because
//! both are valid size types.
//!
//! These return `Result<_, String>` rather than `PyResult` so they can be
//! tested without a Python interpreter; the caller wraps the message.

use alphax_model::{
    AccountType, AccumulationMode, CashFlowKind, DeltaFormat, Direction, LeverageMode,
    LimitOrderPrice, MarginBasis, MarginMode, OrderType, PendingConflictMode, PriceAreaVioMode,
    PriceType, SizeType, StopEntryPrice, StopExitMode, StopExitPrice, StopLadderMode,
    StopUpdateMode, TradesType,
};

/// Every accepted spelling of a size type, for the error message.
const SIZE_TYPES: &str = "amount, value, percent, targetamount, targetvalue, targetpercent, \
                          valuepercent, percent100, valuepercent100, targetpercent100";

/// How a size is interpreted.
///
/// # Errors
///
/// A message naming the accepted values.
pub fn size_type(name: &str) -> Result<SizeType, String> {
    match name {
        "amount" => Ok(SizeType::Amount),
        "value" => Ok(SizeType::Value),
        "percent" => Ok(SizeType::Percent),
        "targetamount" => Ok(SizeType::TargetAmount),
        "targetvalue" => Ok(SizeType::TargetValue),
        "targetpercent" => Ok(SizeType::TargetPercent),
        "valuepercent" => Ok(SizeType::ValuePercent),
        "percent100" => Ok(SizeType::Percent100),
        "valuepercent100" => Ok(SizeType::ValuePercent100),
        "targetpercent100" => Ok(SizeType::TargetPercent100),
        other => Err(format!(
            "size_type must be one of {SIZE_TYPES}; got {other:?}"
        )),
    }
}

/// Which side of the market a column may take.
///
/// `"long"` and `"short"` are accepted alongside
/// `"longonly"`/`"shortonly"`, because the shorter spelling is what people
/// reach for and rejecting it teaches nothing.
///
/// # Errors
///
/// A message naming the accepted values.
pub fn direction(name: &str) -> Result<Direction, String> {
    match name {
        "longonly" | "long" => Ok(Direction::LongOnly),
        "shortonly" | "short" => Ok(Direction::ShortOnly),
        "both" => Ok(Direction::Both),
        other => Err(format!(
            "direction must be 'longonly', 'shortonly' or 'both'; got {other:?}"
        )),
    }
}

/// When a leveraged long borrows.
///
/// # Errors
///
/// A message naming the accepted values.
pub fn leverage_mode(name: &str) -> Result<LeverageMode, String> {
    match name {
        "lazy" => Ok(LeverageMode::Lazy),
        "eager" => Ok(LeverageMode::Eager),
        other => Err(format!(
            "leverage_mode must be 'lazy' or 'eager'; got {other:?}"
        )),
    }
}

/// Whether a column holds what it buys or trades on margin.
///
/// # Errors
///
/// A message naming the accepted values.
pub fn account_type(name: &str) -> Result<AccountType, String> {
    match name {
        "cash" => Ok(AccountType::Cash),
        "margin" => Ok(AccountType::Margin),
        other => Err(format!(
            "account_type must be 'cash' or 'margin'; got {other:?}"
        )),
    }
}

/// What a cash record was: the name `Portfolio.cash_flows` takes.
///
/// # Errors
///
/// A message naming the accepted values.
pub fn cash_flow_kind(name: &str) -> Result<CashFlowKind, String> {
    match name {
        "deposit" => Ok(CashFlowKind::Deposit),
        "earning" => Ok(CashFlowKind::Earning),
        "dividend" => Ok(CashFlowKind::Dividend),
        "funding" => Ok(CashFlowKind::Funding),
        "financing" => Ok(CashFlowKind::Financing),
        "interest" => Ok(CashFlowKind::Interest),
        "borrowfee" => Ok(CashFlowKind::BorrowFee),
        "liquidation" => Ok(CashFlowKind::Liquidation),
        "initialposition" => Ok(CashFlowKind::InitialPosition),
        other => Err(format!(
            "kind must be deposit, earning, dividend, funding, financing, interest, borrowfee, \
             liquidation or initialposition; got {other:?}"
        )),
    }
}

/// Which margin a margin column draws on.
///
/// # Errors
///
/// A message naming the accepted values.
pub fn margin_mode(name: &str) -> Result<MarginMode, String> {
    match name {
        "cross" => Ok(MarginMode::Cross),
        "isolated" => Ok(MarginMode::Isolated),
        other => Err(format!(
            "margin_mode must be 'cross' or 'isolated'; got {other:?}"
        )),
    }
}

/// What a margin column's `initial_margin` is measured in.
///
/// # Errors
///
/// A message naming the accepted values.
pub fn margin_basis(name: &str) -> Result<MarginBasis, String> {
    match name {
        "notional" => Ok(MarginBasis::Notional),
        "unit" => Ok(MarginBasis::Unit),
        other => Err(format!(
            "margin_basis must be 'notional' or 'unit'; got {other:?}"
        )),
    }
}

/// Whether a repeated signal changes an open position's size.
///
/// # Errors
///
/// A message naming the accepted values.
pub fn accumulate(name: &str) -> Result<AccumulationMode, String> {
    match name {
        "disabled" => Ok(AccumulationMode::Disabled),
        "both" => Ok(AccumulationMode::Both),
        "addonly" => Ok(AccumulationMode::AddOnly),
        "removeonly" => Ok(AccumulationMode::RemoveOnly),
        other => Err(format!(
            "accumulate must be 'disabled', 'both', 'addonly' or 'removeonly'; got {other:?}"
        )),
    }
}

/// Which trade view the statistics are scored over.
///
/// # Errors
///
/// A message naming the accepted values.
pub fn trades_type(name: &str) -> Result<TradesType, String> {
    match name {
        "exit" => Ok(TradesType::ExitTrades),
        "entry" => Ok(TradesType::EntryTrades),
        "position" | "positions" => Ok(TradesType::Positions),
        other => Err(format!(
            "trades must be 'exit', 'entry' or 'positions'; got {other:?}"
        )),
    }
}

/// Which price a stop is measured from when its position opens.
///
/// # Errors
///
/// A message naming the accepted values.
pub fn stop_entry_price(name: &str) -> Result<StopEntryPrice, String> {
    match name {
        "valprice" => Ok(StopEntryPrice::ValPrice),
        "price" => Ok(StopEntryPrice::Price),
        "fillprice" => Ok(StopEntryPrice::FillPrice),
        "close" => Ok(StopEntryPrice::Close),
        other => Err(format!(
            "stop_entry_price must be 'valprice', 'price', 'fillprice' or 'close'; got {other:?}"
        )),
    }
}

/// Which price a triggered stop fills at.
///
/// # Errors
///
/// A message naming the accepted values.
pub fn stop_exit_price(name: &str) -> Result<StopExitPrice, String> {
    match name {
        "stoplimit" => Ok(StopExitPrice::StopLimit),
        "stopmarket" => Ok(StopExitPrice::StopMarket),
        "price" => Ok(StopExitPrice::Price),
        "close" => Ok(StopExitPrice::Close),
        other => Err(format!(
            "stop_exit_price must be 'stoplimit', 'stopmarket', 'price' or 'close'; got {other:?}"
        )),
    }
}

/// What a triggered stop does to its position.
///
/// # Errors
///
/// A message naming the accepted values.
pub fn upon_stop_exit(name: &str) -> Result<StopExitMode, String> {
    match name {
        "close" => Ok(StopExitMode::Close),
        "closereduce" => Ok(StopExitMode::CloseReduce),
        "reverse" => Ok(StopExitMode::Reverse),
        "reversereduce" => Ok(StopExitMode::ReverseReduce),
        other => Err(format!(
            "upon_stop_exit must be 'close', 'closereduce', 'reverse' or 'reversereduce'; \
             got {other:?}"
        )),
    }
}

/// Whether a new stop value replaces the live one when a position is added to.
///
/// # Errors
///
/// A message naming the accepted values.
pub fn upon_stop_update(name: &str) -> Result<StopUpdateMode, String> {
    match name {
        "keep" => Ok(StopUpdateMode::Keep),
        "override" => Ok(StopUpdateMode::Override),
        "overridenan" => Ok(StopUpdateMode::OverrideNaN),
        other => Err(format!(
            "upon_stop_update must be 'keep', 'override' or 'overridenan'; got {other:?}"
        )),
    }
}

/// Whether a signal's order executes on its bar or rests as a limit order.
///
/// # Errors
///
/// A message naming the accepted values.
pub fn order_type(name: &str) -> Result<OrderType, String> {
    match name {
        "market" => Ok(OrderType::Market),
        "limit" => Ok(OrderType::Limit),
        other => Err(format!(
            "order_type must be 'market' or 'limit'; got {other:?}"
        )),
    }
}

/// What a limit order fills at.
///
/// # Errors
///
/// A message naming the accepted values.
pub fn limit_order_price(name: &str) -> Result<LimitOrderPrice, String> {
    match name {
        "limit" => Ok(LimitOrderPrice::Limit),
        "hardlimit" => Ok(LimitOrderPrice::HardLimit),
        "close" => Ok(LimitOrderPrice::Close),
        other => Err(format!(
            "limit_order_price must be 'limit', 'hardlimit' or 'close'; got {other:?}"
        )),
    }
}

/// How a limit order's delta is read.
///
/// # Errors
///
/// A message naming the accepted values.
pub fn delta_format(name: &str) -> Result<DeltaFormat, String> {
    match name {
        "percent" => Ok(DeltaFormat::Percent),
        "absolute" => Ok(DeltaFormat::Absolute),
        "target" => Ok(DeltaFormat::Target),
        "percent100" => Ok(DeltaFormat::Percent100),
        other => Err(format!(
            "delta_format must be 'percent', 'percent100', 'absolute' or 'target'; got {other:?}"
        )),
    }
}

/// What a signal does to a pending order it meets, and whether it executes.
///
/// # Errors
///
/// A message naming the accepted values.
pub fn pending_conflict(name: &str) -> Result<PendingConflictMode, String> {
    match name {
        "keepignore" => Ok(PendingConflictMode::KeepIgnore),
        "keepexecute" => Ok(PendingConflictMode::KeepExecute),
        "cancelignore" => Ok(PendingConflictMode::CancelIgnore),
        "cancelexecute" => Ok(PendingConflictMode::CancelExecute),
        other => Err(format!(
            "a conflict mode must be 'keepignore', 'keepexecute', 'cancelignore' or \
             'cancelexecute'; got {other:?}"
        )),
    }
}

/// Which of the bar's prices an order executes at.
///
/// # Errors
///
/// A message naming the accepted values.
pub fn price_type(name: &str) -> Result<PriceType, String> {
    match name {
        "price" => Ok(PriceType::Price),
        "open" => Ok(PriceType::Open),
        "close" => Ok(PriceType::Close),
        other => Err(format!(
            "price_type must be 'price', 'open' or 'close'; got {other:?}"
        )),
    }
}

/// What to do with an order price outside the bar.
///
/// # Errors
///
/// A message naming the accepted values.
pub fn price_area_vio_mode(name: &str) -> Result<PriceAreaVioMode, String> {
    match name {
        "ignore" => Ok(PriceAreaVioMode::Ignore),
        "cap" => Ok(PriceAreaVioMode::Cap),
        "error" => Ok(PriceAreaVioMode::Error),
        other => Err(format!(
            "price_area_vio_mode must be 'ignore', 'cap' or 'error'; got {other:?}"
        )),
    }
}

/// How a stop ladder splits a position.
///
/// # Errors
///
/// A message naming the accepted values.
pub fn stop_ladder(name: &str) -> Result<StopLadderMode, String> {
    match name {
        "disabled" => Ok(StopLadderMode::Disabled),
        "uniform" => Ok(StopLadderMode::Uniform),
        "weighted" => Ok(StopLadderMode::Weighted),
        "adaptuniform" => Ok(StopLadderMode::AdaptUniform),
        "adaptweighted" => Ok(StopLadderMode::AdaptWeighted),
        other => Err(format!(
            "stop_ladder must be 'disabled', 'uniform', 'weighted', 'adaptuniform' or \
             'adaptweighted'; got {other:?}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_price_and_ladder_option_maps_to_its_own_variant() {
        assert_eq!(price_type("price").unwrap(), PriceType::Price);
        assert_eq!(price_type("open").unwrap(), PriceType::Open);
        assert_eq!(price_type("close").unwrap(), PriceType::Close);
        assert_eq!(
            price_area_vio_mode("ignore").unwrap(),
            PriceAreaVioMode::Ignore
        );
        assert_eq!(price_area_vio_mode("cap").unwrap(), PriceAreaVioMode::Cap);
        assert_eq!(
            price_area_vio_mode("error").unwrap(),
            PriceAreaVioMode::Error
        );
        assert_eq!(stop_ladder("disabled").unwrap(), StopLadderMode::Disabled);
        assert_eq!(stop_ladder("uniform").unwrap(), StopLadderMode::Uniform);
        assert_eq!(
            stop_ladder("adaptweighted").unwrap(),
            StopLadderMode::AdaptWeighted
        );
        assert_eq!(stop_ladder("weighted").unwrap(), StopLadderMode::Weighted);
        assert!(price_type("nextopen").is_err());
        assert!(price_area_vio_mode("clip").is_err());
        assert!(stop_ladder("yes").is_err());
    }

    #[test]
    fn every_pending_conflict_mode_maps_to_its_own_variant() {
        assert_eq!(
            pending_conflict("keepignore").unwrap(),
            PendingConflictMode::KeepIgnore
        );
        assert_eq!(
            pending_conflict("keepexecute").unwrap(),
            PendingConflictMode::KeepExecute
        );
        assert_eq!(
            pending_conflict("cancelignore").unwrap(),
            PendingConflictMode::CancelIgnore
        );
        assert_eq!(
            pending_conflict("cancelexecute").unwrap(),
            PendingConflictMode::CancelExecute
        );
        assert!(pending_conflict("keep").is_err());
    }

    #[test]
    fn every_limit_option_maps_to_its_own_variant() {
        assert_eq!(order_type("market").unwrap(), OrderType::Market);
        assert_eq!(order_type("limit").unwrap(), OrderType::Limit);
        assert_eq!(limit_order_price("limit").unwrap(), LimitOrderPrice::Limit);
        assert_eq!(
            limit_order_price("hardlimit").unwrap(),
            LimitOrderPrice::HardLimit
        );
        assert_eq!(limit_order_price("close").unwrap(), LimitOrderPrice::Close);
        assert_eq!(delta_format("percent").unwrap(), DeltaFormat::Percent);
        assert_eq!(delta_format("absolute").unwrap(), DeltaFormat::Absolute);
        assert_eq!(delta_format("target").unwrap(), DeltaFormat::Target);
        assert!(
            order_type("stop")
                .unwrap_err()
                .contains("'market' or 'limit'")
        );
        assert!(limit_order_price("open").is_err());
        assert!(delta_format("pct").is_err());
    }

    #[test]
    fn every_size_type_maps_to_its_own_variant() {
        // Spelled out rather than looped, because the thing being tested is
        // precisely the pairing, and a loop over pairs would be the same table
        // asserting itself.
        assert_eq!(size_type("amount").unwrap(), SizeType::Amount);
        assert_eq!(size_type("value").unwrap(), SizeType::Value);
        assert_eq!(size_type("percent").unwrap(), SizeType::Percent);
        assert_eq!(size_type("targetamount").unwrap(), SizeType::TargetAmount);
        assert_eq!(size_type("targetvalue").unwrap(), SizeType::TargetValue);
        assert_eq!(size_type("targetpercent").unwrap(), SizeType::TargetPercent);
        assert_eq!(size_type("valuepercent").unwrap(), SizeType::ValuePercent);
        assert_eq!(size_type("percent100").unwrap(), SizeType::Percent100);
        assert_eq!(
            size_type("valuepercent100").unwrap(),
            SizeType::ValuePercent100
        );
        assert_eq!(
            size_type("targetpercent100").unwrap(),
            SizeType::TargetPercent100
        );
    }

    #[test]
    fn every_direction_maps_to_its_own_variant() {
        assert_eq!(direction("longonly").unwrap(), Direction::LongOnly);
        assert_eq!(direction("long").unwrap(), Direction::LongOnly);
        assert_eq!(direction("shortonly").unwrap(), Direction::ShortOnly);
        assert_eq!(direction("short").unwrap(), Direction::ShortOnly);
        assert_eq!(direction("both").unwrap(), Direction::Both);
    }

    #[test]
    fn every_cash_flow_kind_maps_to_its_own_variant() {
        assert_eq!(cash_flow_kind("deposit").unwrap(), CashFlowKind::Deposit);
        assert_eq!(cash_flow_kind("earning").unwrap(), CashFlowKind::Earning);
        assert_eq!(cash_flow_kind("dividend").unwrap(), CashFlowKind::Dividend);
        assert_eq!(cash_flow_kind("funding").unwrap(), CashFlowKind::Funding);
        assert_eq!(
            cash_flow_kind("financing").unwrap(),
            CashFlowKind::Financing
        );
        assert_eq!(cash_flow_kind("interest").unwrap(), CashFlowKind::Interest);
        assert_eq!(
            cash_flow_kind("borrowfee").unwrap(),
            CashFlowKind::BorrowFee
        );
        assert_eq!(
            cash_flow_kind("liquidation").unwrap(),
            CashFlowKind::Liquidation
        );
        assert_eq!(
            cash_flow_kind("initialposition").unwrap(),
            CashFlowKind::InitialPosition
        );
        assert!(cash_flow_kind("Funding").is_err());
    }

    #[test]
    fn every_margin_mode_maps_to_its_own_variant() {
        assert_eq!(margin_mode("cross").unwrap(), MarginMode::Cross);
        assert_eq!(margin_mode("isolated").unwrap(), MarginMode::Isolated);
        assert!(margin_mode("Isolated").is_err());
    }

    #[test]
    fn every_leverage_mode_maps_to_its_own_variant() {
        assert_eq!(leverage_mode("lazy").unwrap(), LeverageMode::Lazy);
        assert_eq!(leverage_mode("eager").unwrap(), LeverageMode::Eager);
        assert!(leverage_mode("Lazy").is_err());
    }

    #[test]
    fn every_account_type_maps_to_its_own_variant() {
        assert_eq!(account_type("cash").unwrap(), AccountType::Cash);
        assert_eq!(account_type("margin").unwrap(), AccountType::Margin);
        assert!(account_type("perp").is_err());
    }

    #[test]
    fn every_margin_basis_maps_to_its_own_variant() {
        assert_eq!(margin_basis("notional").unwrap(), MarginBasis::Notional);
        assert_eq!(margin_basis("unit").unwrap(), MarginBasis::Unit);
        assert!(margin_basis("Unit").is_err());
    }

    #[test]
    fn every_accumulation_mode_maps_to_its_own_variant() {
        assert_eq!(accumulate("disabled").unwrap(), AccumulationMode::Disabled);
        assert_eq!(accumulate("both").unwrap(), AccumulationMode::Both);
        assert_eq!(accumulate("addonly").unwrap(), AccumulationMode::AddOnly);
        assert_eq!(
            accumulate("removeonly").unwrap(),
            AccumulationMode::RemoveOnly
        );
    }

    #[test]
    fn every_trade_view_maps_to_its_own_variant() {
        assert_eq!(trades_type("exit").unwrap(), TradesType::ExitTrades);
        assert_eq!(trades_type("entry").unwrap(), TradesType::EntryTrades);
        assert_eq!(trades_type("position").unwrap(), TradesType::Positions);
        assert_eq!(trades_type("positions").unwrap(), TradesType::Positions);
    }

    #[test]
    fn every_stop_entry_price_maps_to_its_own_variant() {
        assert_eq!(
            stop_entry_price("valprice").unwrap(),
            StopEntryPrice::ValPrice
        );
        assert_eq!(stop_entry_price("price").unwrap(), StopEntryPrice::Price);
        assert_eq!(
            stop_entry_price("fillprice").unwrap(),
            StopEntryPrice::FillPrice
        );
        assert_eq!(stop_entry_price("close").unwrap(), StopEntryPrice::Close);
    }

    #[test]
    fn every_stop_exit_price_maps_to_its_own_variant() {
        assert_eq!(
            stop_exit_price("stoplimit").unwrap(),
            StopExitPrice::StopLimit
        );
        assert_eq!(
            stop_exit_price("stopmarket").unwrap(),
            StopExitPrice::StopMarket
        );
        assert_eq!(stop_exit_price("price").unwrap(), StopExitPrice::Price);
        assert_eq!(stop_exit_price("close").unwrap(), StopExitPrice::Close);
    }

    #[test]
    fn every_stop_exit_mode_maps_to_its_own_variant() {
        assert_eq!(upon_stop_exit("close").unwrap(), StopExitMode::Close);
        assert_eq!(
            upon_stop_exit("closereduce").unwrap(),
            StopExitMode::CloseReduce
        );
        assert_eq!(upon_stop_exit("reverse").unwrap(), StopExitMode::Reverse);
        assert_eq!(
            upon_stop_exit("reversereduce").unwrap(),
            StopExitMode::ReverseReduce
        );
    }

    #[test]
    fn every_stop_update_mode_maps_to_its_own_variant() {
        assert_eq!(upon_stop_update("keep").unwrap(), StopUpdateMode::Keep);
        assert_eq!(
            upon_stop_update("override").unwrap(),
            StopUpdateMode::Override
        );
        assert_eq!(
            upon_stop_update("overridenan").unwrap(),
            StopUpdateMode::OverrideNaN
        );
    }

    #[test]
    fn an_unknown_name_names_the_alternatives() {
        // The value the caller passed has to appear too: "must be one of ..."
        // without it leaves them hunting for which argument was wrong.
        let e = size_type("Amount").unwrap_err();
        assert!(e.contains("targetpercent"), "{e}");
        assert!(e.contains("\"Amount\""), "{e}");
        assert!(direction("").unwrap_err().contains("longonly"));
        assert!(accumulate("add").unwrap_err().contains("addonly"));
        assert!(trades_type("exits").unwrap_err().contains("'exit'"));
        assert!(
            stop_entry_price("fill")
                .unwrap_err()
                .contains("'fillprice'")
        );
        assert!(
            stop_exit_price("limit")
                .unwrap_err()
                .contains("'stoplimit'")
        );
        assert!(upon_stop_exit("flip").unwrap_err().contains("'reverse'"));
        assert!(
            upon_stop_update("nan")
                .unwrap_err()
                .contains("'overridenan'")
        );
    }

    #[test]
    fn matching_is_case_sensitive_and_says_so_by_failing() {
        // Accepting "Amount" as well would double the table for nothing; what
        // matters is that the rejection is loud rather than a silent fallback
        // to the default size type.
        assert!(size_type("AMOUNT").is_err());
        assert!(direction("LongOnly").is_err());
    }
}
