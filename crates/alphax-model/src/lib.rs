//! Domain model: value types, identifiers, market data, and the enums that
//! drive the execution state machine.
//!
//! # Two groups of types
//!
//! The *execution* enums ([`enums`]) drive the order state machine; their
//! discriminants are pinned because they are written into record arrays.
//!
//! The *value and identifier* types ([`types`], [`ids`]) are fixed-point raw
//! integers with per-instance precision, and opaque identifiers. Prices,
//! quantities and money are exact scaled integers, and a caller that holds an
//! identifier cannot depend on how it is stored.

pub mod bar;
pub mod enums;
pub mod ids;
pub mod types;

pub use bar::{Bar, BarType, OhlcvPanel};
pub use enums::{
    AccountType, AccumulationMode, CashFlowKind, ConflictMode, DeltaFormat, Direction,
    DirectionConflictMode, LeverageMode, LimitOrderPrice, MarginBasis, MarginMode,
    OppositeEntryMode, OrderSide, OrderStatus, OrderType, PendingConflictMode, PriceAreaVioMode,
    PriceType, SizeType, StatusInfo, StopEntryPrice, StopExitMode, StopExitPrice, StopLadderMode,
    StopUpdateMode, TradeDirection, TradeStatus, TradesType,
};
