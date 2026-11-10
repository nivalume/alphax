//! Order prices beyond the `price` input, shared by both loops.
//!
//! - **`price_type`.** `Open` and `Close` execute at the bar's open or close
//!   without a price array; `Price`, the default, reads `price` with its
//!   `+inf` (close) and `-inf` conventions.
//! - **`from_ago`.** An order on bar `i` is the one decided `from_ago` bars
//!   earlier: its signals, size, size type, direction and an explicit price
//!   are read at `i − from_ago`, and nothing is ordered before that row
//!   exists. The price conventions — `±inf`, `Open`, `Close` — still resolve
//!   on bar `i`, the bar it executes on. `price_type = Open` with
//!   `from_ago = 1` is the next bar's open: a decision at one close
//!   executed at the following open, which reads nothing it could not have
//!   known.
//! - **Bid and ask.** Where given, a buy executes at `ask_price` and a sell at
//!   `bid_price`, in place of the order price; valuation still uses the
//!   valuation price. Slippage applies on top, as to any market order.
//! - **`price_area_vio_mode`.** An order price outside the bar's low–high
//!   range is executed at as it is (`Ignore`, the default), moved to the
//!   nearer bound (`Cap`), or refused with an error (`Error`). Only tested
//!   where the bar has a low and a high.

use alphax_core::error::{Error, Result};
use alphax_core::flex::FlexArray;
use alphax_model::{PriceAreaVioMode, PriceType, SizeType};

/// The row an order executing on bar `i` was decided on, or `None` before
/// there is one.
#[inline]
pub(crate) const fn source_row(i: usize, from_ago: usize) -> Option<usize> {
    i.checked_sub(from_ago)
}

/// The price inputs a loop reads beyond `price`.
#[derive(Clone, Copy)]
pub(crate) struct Quotes<'s, 'a> {
    pub(crate) price_type: &'s FlexArray<'a, PriceType>,
    pub(crate) bid: &'s FlexArray<'a, f64>,
    pub(crate) ask: &'s FlexArray<'a, f64>,
    pub(crate) vio: &'s FlexArray<'a, PriceAreaVioMode>,
    pub(crate) low: &'s FlexArray<'a, f64>,
    pub(crate) high: &'s FlexArray<'a, f64>,
}

impl Quotes<'_, '_> {
    /// Whether any order may execute at a price other than the one `price`
    /// gives, or have it checked: decided once, so a run without any reads
    /// none of these inputs.
    pub(crate) fn on(&self) -> bool {
        !matches!(self.price_type, FlexArray::Scalar(PriceType::Price))
            || !matches!(self.vio, FlexArray::Scalar(PriceAreaVioMode::Ignore))
            || !is_nan(self.bid)
            || !is_nan(self.ask)
    }

    /// The price an order on the given side executes at on bar `i`: its
    /// bid or ask where there is one, else `price`, then checked against
    /// the bar's range.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invariant`] for a price outside the bar under
    /// `PriceAreaVioMode::Error`.
    pub(crate) fn order_price(&self, i: usize, col: usize, price: f64, buy: bool) -> Result<f64> {
        let quote = if buy {
            self.ask.get(i, col)
        } else {
            self.bid.get(i, col)
        };
        let price = if quote.is_nan() { price } else { quote };
        let (low, high) = (self.low.get(i, col), self.high.get(i, col));
        if price.is_nan() || !(price < low || price > high) {
            return Ok(price);
        }
        match self.vio.get(i, col) {
            PriceAreaVioMode::Ignore => Ok(price),
            PriceAreaVioMode::Cap => Ok(price.clamp(low, high)),
            PriceAreaVioMode::Error => Err(Error::Invariant(format!(
                "the order price {price} at row {i}, column {col} is outside the bar's \
                 range [{low}, {high}]"
            ))),
        }
    }
}

/// Whether an order buys, before the kernel says so: `from_orders` needs the
/// side to pick between the bid and the ask. The same comparison the
/// kernel's size cascade makes, at the valuation price and value the order
/// is sized against.
pub(crate) fn buys(
    size: f64,
    size_type: SizeType,
    position: f64,
    val_price: f64,
    value: f64,
) -> bool {
    match size_type {
        SizeType::Amount
        | SizeType::Value
        | SizeType::Percent
        | SizeType::ValuePercent
        | SizeType::Percent100
        | SizeType::ValuePercent100 => size > 0.0,
        SizeType::TargetAmount => size > position,
        SizeType::TargetValue => size / val_price > position,
        SizeType::TargetPercent => size * value / val_price > position,
        SizeType::TargetPercent100 => size / 100.0 * value / val_price > position,
    }
}

/// Whether an input is the scalar `NaN`, which no read can turn into a price.
fn is_nan(array: &FlexArray<'_, f64>) -> bool {
    matches!(array, FlexArray::Scalar(v) if v.is_nan())
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "prices are copied or clamped, not computed"
)]
mod tests {
    use super::*;

    fn quotes<'s>(
        bid: &'s FlexArray<'static, f64>,
        ask: &'s FlexArray<'static, f64>,
        vio: &'s FlexArray<'static, PriceAreaVioMode>,
    ) -> Quotes<'s, 'static> {
        const TYPE: FlexArray<'static, PriceType> = FlexArray::Scalar(PriceType::Price);
        const LOW: FlexArray<'static, f64> = FlexArray::Scalar(90.0);
        const HIGH: FlexArray<'static, f64> = FlexArray::Scalar(110.0);
        Quotes {
            price_type: &TYPE,
            bid,
            ask,
            vio,
            low: &LOW,
            high: &HIGH,
        }
    }

    #[test]
    fn a_buy_takes_the_ask_and_a_sell_the_bid() {
        let (bid, ask) = (FlexArray::Scalar(99.0), FlexArray::Scalar(101.0));
        let vio = FlexArray::Scalar(PriceAreaVioMode::Ignore);
        let q = quotes(&bid, &ask, &vio);
        assert!(q.on());
        assert_eq!(q.order_price(0, 0, 100.0, true).unwrap(), 101.0);
        assert_eq!(q.order_price(0, 0, 100.0, false).unwrap(), 99.0);
        let none = FlexArray::Scalar(f64::NAN);
        let q = quotes(&none, &none, &vio);
        assert!(!q.on());
        assert_eq!(q.order_price(0, 0, 100.0, true).unwrap(), 100.0);
    }

    #[test]
    fn a_price_outside_the_bar_is_kept_capped_or_refused() {
        let none = FlexArray::Scalar(f64::NAN);
        let at = |mode, price| {
            let vio = FlexArray::Scalar(mode);
            quotes(&none, &none, &vio).order_price(3, 0, price, true)
        };
        assert_eq!(at(PriceAreaVioMode::Ignore, 120.0).unwrap(), 120.0);
        assert_eq!(at(PriceAreaVioMode::Cap, 120.0).unwrap(), 110.0);
        assert_eq!(at(PriceAreaVioMode::Cap, 80.0).unwrap(), 90.0);
        assert_eq!(at(PriceAreaVioMode::Cap, 100.0).unwrap(), 100.0);
        let err = at(PriceAreaVioMode::Error, 120.0).unwrap_err().to_string();
        assert!(err.contains("outside the bar's range [90, 110]"), "{err}");
    }

    #[test]
    fn an_order_has_a_source_row_once_it_exists() {
        assert_eq!(source_row(5, 0), Some(5));
        assert_eq!(source_row(5, 2), Some(3));
        assert_eq!(source_row(1, 2), None);
    }
}
