//! State, request and result types for order execution.

use alphax_model::{
    AccountType, Direction, LeverageMode, MarginBasis, OrderSide, OrderStatus, SizeType, StatusInfo,
};

/// Account state that an order execution reads and rewrites.
///
/// `debt`, `locked_cash` and `free_cash` are the fields that are easy to get
/// wrong.
///
/// - `cash` is the account's own cash. Borrowed money is not in it: a
///   leveraged buy spends own cash only for the part it does not borrow.
/// - `debt` is what the column owes. On a short it is the proceeds of the
///   open short sales, held at their average entry price — what the short
///   "owes" back. On a leveraged long it is the money borrowed to buy it.
/// - `locked_cash` is the own cash the position ties up. On a long it is the
///   own part of what was paid for it, already gone from `cash`. On a short it
///   is the collateral, `value / leverage` at entry, still in `cash` but not
///   free.
/// - `free_cash` is own cash net of what open shorts tie up: their proceeds
///   and their collateral. Opening or extending a short moves it by
///   `proceeds - short_value - short_value / leverage`; at leverage 1 that is
///   `proceeds - 2 * short_value`, once for the position's own
///   value and once for the round-trip reservation. Closing unwinds it
///   through `(debt + locked_cash) / |position|`.
///
/// Without leverage a long has no debt and a short's `locked_cash` equals its
/// `debt`, which is the only state an unlevered cash account can describe.
///
/// A margin column ([`AccountType::Margin`]) reads the same fields
/// differently, on both sides alike: `debt` is the position's cost basis,
/// `|position|` times its average entry price, which the loop moves to the
/// valuation price as it settles each bar's profit into `cash`; `locked_cash`
/// is the initial margin posted, still in `cash` but not free. So `free_cash`
/// is `cash` less the margin of every margin position, and less the
/// reservations of any cash column's shorts in the same group.
///
/// Carrying only `cash` and `position` and deriving the rest does not work —
/// the reservation is path-dependent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExecuteOrderState {
    /// Own cash.
    pub cash: f64,
    /// Signed position size; negative is short.
    pub position: f64,
    /// What the column owes: short proceeds, or a long's borrowing.
    pub debt: f64,
    /// Own cash the position ties up: a long's own part, a short's collateral.
    pub locked_cash: f64,
    /// Own cash net of what open shorts tie up.
    pub free_cash: f64,
}

/// [`ExecuteOrderState`] plus the bookkeeping the simulation loop maintains.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProcessOrderState {
    /// Own cash.
    pub cash: f64,
    /// Signed position size.
    pub position: f64,
    /// What the column owes: short proceeds, or a long's borrowing.
    pub debt: f64,
    /// Own cash the position ties up.
    pub locked_cash: f64,
    /// Own cash net of what open shorts tie up.
    pub free_cash: f64,
    /// Price used to value the position.
    pub val_price: f64,
    /// Total portfolio value.
    pub value: f64,
    /// Index of the next order record slot.
    pub oidx: usize,
    /// Index of the next log record slot.
    pub lidx: usize,
}

impl ProcessOrderState {
    /// A flat state with the given cash, valued at `val_price`.
    #[must_use]
    pub const fn flat(cash: f64, val_price: f64) -> Self {
        Self {
            cash,
            position: 0.0,
            debt: 0.0,
            locked_cash: 0.0,
            free_cash: cash,
            val_price,
            value: cash,
            oidx: 0,
            lidx: 0,
        }
    }
}

/// A requested order.
///
/// Defaults: an infinite size means "as much as
/// possible" and an infinite price means "use the bar's close". Those
/// infinities are live sentinels, which is why [`alphax_core::math::is_close`]
/// refuses to call two infinities equal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Order {
    /// Requested size, interpreted per `size_type`.
    pub size: f64,
    /// Requested price; `inf` means the close.
    pub price: f64,
    /// How to interpret `size`.
    pub size_type: SizeType,
    /// Which sides this column may take.
    pub direction: Direction,
    /// Proportional fee rate.
    pub fees: f64,
    /// Flat fee per fill.
    pub fixed_fees: f64,
    /// Proportional slippage, applied against the trade direction.
    pub slippage: f64,
    /// Reject fills below this size.
    pub min_size: f64,
    /// Clamp fills to this size.
    pub max_size: f64,
    /// Round sizes down to a multiple of this.
    pub size_granularity: f64,
    /// How many times its own cash a buy may be worth, `1` or more.
    ///
    /// A buy needs own cash for `1 / leverage` of its notional plus the fees,
    /// and borrows the rest into `debt`. A short ties up its proceeds plus
    /// `1 / leverage` of its value. At `1`, the default, nothing is borrowed
    /// and the order behaves exactly as an unlevered one.
    pub leverage: f64,
    /// Whether a leveraged buy borrows only what own cash cannot cover, or
    /// always `1 - 1 / leverage` of it.
    pub leverage_mode: LeverageMode,
    /// Whether the column pays for its trades in full or posts margin.
    ///
    /// [`AccountType::Margin`] fills either side by posting `initial_margin`
    /// out of free cash, and ignores `leverage_mode` and `lock_cash`. Its
    /// state reads differently; see [`ExecuteOrderState`].
    pub account_type: AccountType,
    /// The margin a margin order posts per unit opened, in `margin_basis`.
    ///
    /// `NaN`, the default, is `1 / leverage` of the notional. Fixed at the
    /// fill: a later change applies to later fills only.
    pub initial_margin: f64,
    /// Whether `initial_margin` is a fraction of the notional or cash per unit.
    pub margin_basis: MarginBasis,
    /// The most cash one buy may spend, fees included. `inf` is no cap.
    ///
    /// Applied to the cash available before `percent` takes its share, so
    /// `Percent(0.5)` under a cap of 100 spends at most 50. Sells and shorts
    /// receive cash rather than spending it and are not affected. Not in the
    /// log schema either.
    pub cash_limit: f64,
    /// Probability of a random rejection, in `[0, 1]`.
    ///
    /// When non-zero the caller must supply the random draw; see
    /// [`crate::execute_order`].
    pub reject_prob: f64,
    /// Whether to reserve cash to cover an existing short.
    pub lock_cash: bool,
    /// Whether a partial fill is acceptable.
    pub allow_partial: bool,
    /// Whether a rejection should be an error instead of a status.
    pub raise_reject: bool,
    /// Whether to emit a log record.
    pub log: bool,
    /// Whether a price of 0 or below is a price rather than a bad input.
    ///
    /// Such an order must be sized in units. A cash column may buy, and sell
    /// what it holds; a margin column trades either side. Fees, slippage
    /// and margin are taken on the price's size, slippage always against
    /// the trade. Off by default.
    pub allow_negative_price: bool,
}

impl Default for Order {
    fn default() -> Self {
        Self {
            size: f64::INFINITY,
            price: f64::INFINITY,
            size_type: SizeType::Amount,
            direction: Direction::Both,
            fees: 0.0,
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
            log: false,
            allow_negative_price: false,
        }
    }
}

/// The outcome of an order request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrderResult {
    /// Filled size; `NaN` if nothing filled.
    pub size: f64,
    /// Fill price; `NaN` if nothing filled.
    pub price: f64,
    /// Fees charged; `NaN` if nothing filled.
    pub fees: f64,
    /// Which side filled.
    pub side: Option<OrderSide>,
    /// Filled, ignored or rejected.
    pub status: OrderStatus,
    /// Why, when not filled.
    pub status_info: Option<StatusInfo>,
}

impl OrderResult {
    /// An outcome where nothing filled.
    #[must_use]
    pub const fn not_filled(status: OrderStatus, info: StatusInfo) -> Self {
        Self {
            size: f64::NAN,
            price: f64::NAN,
            fees: f64::NAN,
            side: None,
            status,
            status_info: Some(info),
        }
    }

    /// A filled outcome.
    #[must_use]
    pub const fn filled(size: f64, price: f64, fees: f64, side: OrderSide) -> Self {
        Self {
            size,
            price,
            fees,
            side: Some(side),
            status: OrderStatus::Filled,
            status_info: None,
        }
    }

    /// Whether this order actually traded.
    #[must_use]
    pub const fn is_filled(&self) -> bool {
        matches!(self.status, OrderStatus::Filled)
    }
}
