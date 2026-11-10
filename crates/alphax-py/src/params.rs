//! The execution parameters both entry points share.
//!
//! `from_orders` and `from_signals` differ in what drives them — an order per
//! element against a signal pipeline — and agree on everything the execution
//! kernel reads: how big, at what price, what it costs, what is refused. That
//! shared middle is a dozen parameters, each of which can arrive as a number, a
//! value per column or a full matrix.
//!
//! Collected here for one reason: the borrows have to outlive the call that
//! builds the [`FlexArray`](alphax_core::flex::FlexArray)s from them, so they must
//! live in the caller's own frame. A struct is how two call sites can share
//! them without either one duplicating a dozen `let` bindings — and without a
//! helper that returns the arrays, which cannot be written, because the arrays
//! borrow from values the helper would drop on return.

use pyo3::prelude::*;

use crate::array::{Borrowed, BorrowedBool, borrow_opt_bool, borrow_opt_f64};

/// The borrowed execution parameters, held for the duration of a call.
///
/// Every field is `None` when Python did not supply it, which means "the
/// engine's own default" — see [`crate::array::borrow_opt_f64`] for why the
/// defaults are not written down here.
#[derive(Debug)]
pub struct Shared<'py> {
    /// Requested size.
    pub size: Option<Borrowed<'py>>,
    /// Order price. `inf` means the bar's close.
    pub price: Option<Borrowed<'py>>,
    /// Proportional fee rate.
    pub fees: Option<Borrowed<'py>>,
    /// Flat fee per fill.
    pub fixed_fees: Option<Borrowed<'py>>,
    /// Proportional slippage.
    pub slippage: Option<Borrowed<'py>>,
    /// Smallest acceptable fill.
    pub min_size: Option<Borrowed<'py>>,
    /// Largest acceptable fill.
    pub max_size: Option<Borrowed<'py>>,
    /// Lot size to round down to.
    pub size_granularity: Option<Borrowed<'py>>,
    /// Buying power per unit of own cash.
    pub leverage: Option<Borrowed<'py>>,
    /// Initial margin a margin column posts per unit it opens.
    pub initial_margin: Option<Borrowed<'py>>,
    /// The most cash one buy may spend.
    pub cash_limit: Option<Borrowed<'py>>,
    /// Probability of a random rejection.
    pub reject_prob: Option<Borrowed<'py>>,
    /// Whether to reserve cash to cover an open short.
    pub lock_cash: Option<BorrowedBool<'py>>,
    /// Whether a partial fill is acceptable.
    pub allow_partial: Option<BorrowedBool<'py>>,
    /// Whether a rejection aborts the run instead of being recorded.
    pub raise_reject: Option<BorrowedBool<'py>>,
    /// Whether each order writes a log row.
    pub log: Option<BorrowedBool<'py>>,
}

impl<'py> Shared<'py> {
    /// Reads them all, keeping each borrow alive in the returned value.
    ///
    /// # Errors
    ///
    /// A Python `TypeError` naming the first parameter whose type cannot be
    /// read — by name, because "expected float, got str" with twelve candidates
    /// is not a diagnosis.
    #[allow(
        clippy::too_many_arguments,
        reason = "one argument per parameter, which is what is being collected"
    )]
    pub fn borrow(
        size: Option<&Bound<'py, PyAny>>,
        price: Option<&Bound<'py, PyAny>>,
        fees: Option<&Bound<'py, PyAny>>,
        fixed_fees: Option<&Bound<'py, PyAny>>,
        slippage: Option<&Bound<'py, PyAny>>,
        min_size: Option<&Bound<'py, PyAny>>,
        max_size: Option<&Bound<'py, PyAny>>,
        size_granularity: Option<&Bound<'py, PyAny>>,
        leverage: Option<&Bound<'py, PyAny>>,
        initial_margin: Option<&Bound<'py, PyAny>>,
        cash_limit: Option<&Bound<'py, PyAny>>,
        reject_prob: Option<&Bound<'py, PyAny>>,
        lock_cash: Option<&Bound<'py, PyAny>>,
        allow_partial: Option<&Bound<'py, PyAny>>,
        raise_reject: Option<&Bound<'py, PyAny>>,
        log: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            size: borrow_opt_f64(size, "size")?,
            price: borrow_opt_f64(price, "price")?,
            fees: borrow_opt_f64(fees, "fees")?,
            fixed_fees: borrow_opt_f64(fixed_fees, "fixed_fees")?,
            slippage: borrow_opt_f64(slippage, "slippage")?,
            min_size: borrow_opt_f64(min_size, "min_size")?,
            max_size: borrow_opt_f64(max_size, "max_size")?,
            size_granularity: borrow_opt_f64(size_granularity, "size_granularity")?,
            leverage: borrow_opt_f64(leverage, "leverage")?,
            initial_margin: borrow_opt_f64(initial_margin, "initial_margin")?,
            cash_limit: borrow_opt_f64(cash_limit, "cash_limit")?,
            reject_prob: borrow_opt_f64(reject_prob, "reject_prob")?,
            lock_cash: borrow_opt_bool(lock_cash, "lock_cash")?,
            allow_partial: borrow_opt_bool(allow_partial, "allow_partial")?,
            raise_reject: borrow_opt_bool(raise_reject, "raise_reject")?,
            log: borrow_opt_bool(log, "log")?,
        })
    }

    /// Whether any order in this run asks for a log row.
    ///
    /// Decides which simulation entry point is called. A scalar `False` — the
    /// default — takes the unlogged path, so a run that is not investigating
    /// anything does not allocate a journal. An array is assumed to contain at
    /// least one `true` rather than scanned: a caller who passes a mask wants
    /// logging, and scanning to discover an all-`false` mask would cost a pass
    /// over the matrix to save allocating an empty vector.
    pub const fn wants_logs(&self) -> bool {
        match &self.log {
            None => false,
            Some(BorrowedBool::Scalar(on)) => *on,
            Some(BorrowedBool::Slice(_)) => true,
        }
    }
}

/// The borrowed liquidation parameters, which both entry points share: per
/// column, per bar or a matrix, like the other execution parameters.
#[derive(Debug)]
pub struct Liquidation<'py> {
    /// A margin column's maintenance margin, in `margin_basis`.
    pub maintenance_margin: Option<Borrowed<'py>>,
    /// What a liquidation fill pays, as a fraction of its notional.
    pub liquidation_fee: Option<Borrowed<'py>>,
    /// A cash column's maintenance margin.
    pub cash_maintenance_margin: Option<Borrowed<'py>>,
    /// The tier table's notional bounds and rates, rows the tiers.
    pub tier_notional: Option<Borrowed<'py>>,
    pub tier_rate: Option<Borrowed<'py>>,
    /// The fraction of a position each breach closes.
    pub liquidation_step: Option<Borrowed<'py>>,
}

impl<'py> Liquidation<'py> {
    /// Reads both, keeping each borrow alive in the returned value.
    ///
    /// # Errors
    ///
    /// A Python `TypeError` naming the parameter that cannot be read.
    pub fn borrow(
        maintenance_margin: Option<&Bound<'py, PyAny>>,
        liquidation_fee: Option<&Bound<'py, PyAny>>,
        cash_maintenance_margin: Option<&Bound<'py, PyAny>>,
        tier_notional: Option<&Bound<'py, PyAny>>,
        tier_rate: Option<&Bound<'py, PyAny>>,
        liquidation_step: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            maintenance_margin: borrow_opt_f64(maintenance_margin, "maintenance_margin")?,
            liquidation_fee: borrow_opt_f64(liquidation_fee, "liquidation_fee")?,
            cash_maintenance_margin: borrow_opt_f64(
                cash_maintenance_margin,
                "cash_maintenance_margin",
            )?,
            tier_notional: borrow_opt_f64(tier_notional, "margin_tier_notional")?,
            tier_rate: borrow_opt_f64(tier_rate, "margin_tier_rate")?,
            liquidation_step: borrow_opt_f64(liquidation_step, "liquidation_step")?,
        })
    }
}

/// The borrowed inputs of the holding charges, which both entry points share.
///
/// Every rate but `debt_rate` belongs to an instrument — a perpetual's funding
/// history, a broker's swap rates, a stock's borrow fee — and so is market
/// data, source-shaped through `column_map` like the prices; `mark_price` is
/// a price. `debt_rate` is the account's cost of borrowing cash, a parameter.
#[derive(Debug)]
pub struct Holding<'py> {
    /// A perpetual's funding rate per bar.
    pub funding_rate: Option<Borrowed<'py>>,
    /// Overnight financing per bar on a long.
    pub long_financing_rate: Option<Borrowed<'py>>,
    /// Overnight financing per bar on a short.
    pub short_financing_rate: Option<Borrowed<'py>>,
    /// Interest per bar on what a long borrowed.
    pub debt_rate: Option<Borrowed<'py>>,
    /// The fee per bar for the shares a short borrowed.
    pub borrow_rate: Option<Borrowed<'py>>,
    /// The price the charges are computed at.
    pub mark_price: Option<Borrowed<'py>>,
}

impl<'py> Holding<'py> {
    /// Reads them all, keeping each borrow alive in the returned value.
    ///
    /// # Errors
    ///
    /// A Python `TypeError` naming the first input that cannot be read.
    pub fn borrow(
        funding_rate: Option<&Bound<'py, PyAny>>,
        long_financing_rate: Option<&Bound<'py, PyAny>>,
        short_financing_rate: Option<&Bound<'py, PyAny>>,
        debt_rate: Option<&Bound<'py, PyAny>>,
        borrow_rate: Option<&Bound<'py, PyAny>>,
        mark_price: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            funding_rate: borrow_opt_f64(funding_rate, "funding_rate")?,
            long_financing_rate: borrow_opt_f64(long_financing_rate, "long_financing_rate")?,
            short_financing_rate: borrow_opt_f64(short_financing_rate, "short_financing_rate")?,
            debt_rate: borrow_opt_f64(debt_rate, "debt_rate")?,
            borrow_rate: borrow_opt_f64(borrow_rate, "borrow_rate")?,
            mark_price: borrow_opt_f64(mark_price, "mark_price")?,
        })
    }
}
