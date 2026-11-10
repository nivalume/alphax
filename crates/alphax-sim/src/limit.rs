//! Limit orders in `from_signals`: where one rests, when the market reaches
//! it and what it fills at.
//!
//! A signal whose `order_type` is `Limit` is sized as any other, then rests
//! at a limit price instead of executing:
//!
//! | `delta_format` | buy | sell |
//! |---|---|---|
//! | `Percent` (default) | `price × (1 − delta)` | `price × (1 + delta)` |
//! | `Absolute` | `price − delta` | `price + delta` |
//! | `Target` | `delta` | `delta` |
//!
//! `price` is the bar's order price and a `NaN` delta is none. A buy fills
//! once the market is at or below its limit, a sell at or above it.
//! `limit_reverse` turns both round — the delta is applied the other way and
//! a buy fills at or above, a sell at or below — which is a breakout (stop)
//! entry.
//!
//! # When it is tested
//!
//! At once, at the order price: a limit the price already reaches fills on
//! its own bar. An order placed at the open (`price = -inf`) is also tested
//! against the rest of that bar, its high and low; one placed at the close
//! or at any other price is not, because what the bar did after it is
//! unknown. `limit_delay` defers all of this to the next bar. From then on
//! each bar tests it from its open: a bar that opens beyond the limit has
//! gapped through it.
//!
//! # What it fills at
//!
//! | `limit_order_price` | reached at the start (the order price, or a gap at the open) | reached inside the bar |
//! |---|---|---|
//! | `Limit` (default) | that price, which is at least as good as the limit | the limit |
//! | `HardLimit` | the limit | the limit |
//! | `Close` | the close | the close |
//!
//! A limit order pays no slippage.
//!
//! # How long it rests
//!
//! `limit_tif` bars after the one it was placed on, and not on or after the
//! row `limit_expiry`; `NaN` is no limit. The engine has no timestamps, so
//! both count rows. And until anything else fills in its column, which
//! cancels it: it was sized against the position held when it was placed.

use alphax_core::error::{Error, Result};
use alphax_model::{DeltaFormat, LimitOrderPrice};

/// Whether a limit fills with the market at or below its price: a buy, or a
/// reversed sell.
const fn fills_below(buy: bool, reverse: bool) -> bool {
    buy != reverse
}

/// The limit price of an order placed at `price`, or `NaN` when `price` is.
///
/// # Errors
///
/// Returns [`Error::Invariant`] for a delta that is infinite or negative, or
/// a limit price that is not finite and above 0.
pub fn limit_price(
    price: f64,
    buy: bool,
    reverse: bool,
    delta: f64,
    format: DeltaFormat,
) -> Result<f64> {
    if price.is_nan() {
        return Ok(f64::NAN);
    }
    let level = if format == DeltaFormat::Target {
        if delta.is_nan() { price } else { delta }
    } else {
        let delta = if delta.is_nan() { 0.0 } else { delta };
        if !delta.is_finite() || delta < 0.0 {
            return Err(Error::Invariant(format!(
                "limit_delta must be NaN, or finite and 0 or greater; got {delta}"
            )));
        }
        let away = if fills_below(buy, reverse) {
            -delta
        } else {
            delta
        };
        match format {
            DeltaFormat::Percent => price * (1.0 + away),
            DeltaFormat::Percent100 => price * (1.0 + away / 100.0),
            _ => price + away,
        }
    };
    if !level.is_finite() || level <= 0.0 {
        return Err(Error::Invariant(format!(
            "a limit price must be finite and above 0; got {level} from the order price {price}"
        )));
    }
    Ok(level)
}

/// The last bar an order placed on `created` may fill on, or `None` when it
/// has expired before it could.
///
/// # Errors
///
/// Returns [`Error::Invariant`] for a negative `tif` or `expiry`.
pub fn last_row(created: usize, tif: f64, expiry: f64) -> Result<Option<usize>> {
    if tif < 0.0 || expiry < 0.0 {
        return Err(Error::Invariant(format!(
            "limit_tif and limit_expiry must be NaN or 0 or greater; got {tif} and {expiry}"
        )));
    }
    let by_tif = if tif.is_nan() || tif.is_infinite() {
        usize::MAX
    } else {
        created.saturating_add(tif.floor() as usize)
    };
    // Cancelled at the start of the row `expiry`.
    let by_expiry = if expiry.is_nan() || expiry.is_infinite() {
        usize::MAX
    } else {
        let first_gone = expiry.ceil() as usize;
        if first_gone <= created {
            return Ok(None);
        }
        first_gone - 1
    };
    Ok(Some(by_tif.min(by_expiry)))
}

/// What a limit at `level` fills at on a bar, or `NaN` if the bar does not
/// reach it.
///
/// `start` is where the test begins — the order price on the bar the order
/// is placed, the open on a later one — and is tested on its own first, so a
/// gap through the limit fills at it. `high` and `low` are the rest of the
/// bar, `NaN` when there is none to test.
#[allow(
    clippy::too_many_arguments,
    reason = "the order's four fields and the bar's four prices"
)]
pub fn fill_price(
    level: f64,
    buy: bool,
    reverse: bool,
    order_price: LimitOrderPrice,
    start: f64,
    high: f64,
    low: f64,
    close: f64,
) -> f64 {
    let below = fills_below(buy, reverse);
    let at_start = if below {
        start <= level
    } else {
        start >= level
    };
    if at_start {
        return match order_price {
            LimitOrderPrice::Limit => start,
            LimitOrderPrice::HardLimit => level,
            LimitOrderPrice::Close => close,
        };
    }
    let inside = if below { low <= level } else { high >= level };
    if inside {
        return match order_price {
            LimitOrderPrice::Limit | LimitOrderPrice::HardLimit => level,
            LimitOrderPrice::Close => close,
        };
    }
    f64::NAN
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "the prices are exact in binary or copied unchanged"
)]
mod tests {
    use super::*;

    const N: f64 = f64::NAN;

    #[test]
    fn the_delta_moves_the_limit_away_from_the_price_and_reverse_turns_it_round() {
        let p =
            |buy, reverse, delta, format| limit_price(64.0, buy, reverse, delta, format).unwrap();
        assert_eq!(p(true, false, 0.25, DeltaFormat::Percent), 48.0);
        assert_eq!(p(false, false, 0.25, DeltaFormat::Percent), 80.0);
        assert_eq!(p(true, true, 0.25, DeltaFormat::Percent), 80.0);
        assert_eq!(p(false, true, 0.25, DeltaFormat::Percent), 48.0);
        assert_eq!(p(true, false, 4.0, DeltaFormat::Absolute), 60.0);
        assert_eq!(p(false, false, 4.0, DeltaFormat::Absolute), 68.0);
        assert_eq!(p(true, false, 50.0, DeltaFormat::Target), 50.0);
        assert_eq!(p(true, false, N, DeltaFormat::Target), 64.0);
        assert_eq!(p(true, false, N, DeltaFormat::Percent), 64.0);
        assert!(
            limit_price(N, true, false, 0.1, DeltaFormat::Percent)
                .unwrap()
                .is_nan()
        );
    }

    #[test]
    fn a_limit_no_market_can_reach_is_refused() {
        for (delta, format) in [
            (-0.1, DeltaFormat::Percent),
            (f64::INFINITY, DeltaFormat::Absolute),
            (1.0, DeltaFormat::Percent),
            (64.0, DeltaFormat::Absolute),
            (-5.0, DeltaFormat::Target),
        ] {
            assert!(
                limit_price(64.0, true, false, delta, format).is_err(),
                "{delta}"
            );
        }
    }

    #[test]
    fn a_buy_fills_at_or_below_its_limit_and_a_gap_at_the_open() {
        let fill = |order_price, open, low| {
            fill_price(48.0, true, false, order_price, open, 70.0, low, 60.0)
        };
        // Not reached.
        assert!(fill(LimitOrderPrice::Limit, 64.0, 50.0).is_nan());
        // Reached inside the bar: the limit.
        assert_eq!(fill(LimitOrderPrice::Limit, 64.0, 40.0), 48.0);
        assert_eq!(fill(LimitOrderPrice::HardLimit, 64.0, 40.0), 48.0);
        assert_eq!(fill(LimitOrderPrice::Close, 64.0, 40.0), 60.0);
        // Opened below it: the open, which is better.
        assert_eq!(fill(LimitOrderPrice::Limit, 44.0, 40.0), 44.0);
        assert_eq!(fill(LimitOrderPrice::HardLimit, 44.0, 40.0), 48.0);
        assert_eq!(fill(LimitOrderPrice::Close, 44.0, 40.0), 60.0);
    }

    #[test]
    fn a_reversed_buy_is_a_breakout_and_fills_at_or_above() {
        // A buy stop at 80: reached at the high, or gapped through at 84.
        assert_eq!(
            fill_price(
                80.0,
                true,
                true,
                LimitOrderPrice::Limit,
                64.0,
                81.0,
                60.0,
                70.0
            ),
            80.0
        );
        assert_eq!(
            fill_price(
                80.0,
                true,
                true,
                LimitOrderPrice::Limit,
                84.0,
                90.0,
                82.0,
                85.0
            ),
            84.0
        );
        assert!(
            fill_price(
                80.0,
                true,
                true,
                LimitOrderPrice::Limit,
                64.0,
                79.0,
                60.0,
                70.0
            )
            .is_nan()
        );
        // And a sell limit, the mirror of a buy.
        assert_eq!(
            fill_price(
                80.0,
                false,
                false,
                LimitOrderPrice::Limit,
                64.0,
                81.0,
                60.0,
                70.0
            ),
            80.0
        );
    }

    #[test]
    fn without_a_range_only_the_start_is_tested() {
        assert!(fill_price(48.0, true, false, LimitOrderPrice::Limit, 50.0, N, N, 50.0).is_nan());
        assert_eq!(
            fill_price(48.0, true, false, LimitOrderPrice::Limit, 47.0, N, N, 47.0),
            47.0
        );
    }

    #[test]
    fn an_order_rests_for_its_bars_and_not_into_its_expiry() {
        assert_eq!(last_row(5, N, N).unwrap(), Some(usize::MAX));
        assert_eq!(last_row(5, 0.0, N).unwrap(), Some(5));
        assert_eq!(last_row(5, 3.0, N).unwrap(), Some(8));
        assert_eq!(last_row(5, 3.7, N).unwrap(), Some(8));
        assert_eq!(last_row(5, f64::INFINITY, 7.0).unwrap(), Some(6));
        assert_eq!(last_row(5, 1.0, 9.0).unwrap(), Some(6));
        assert_eq!(last_row(5, N, 5.0).unwrap(), None);
        assert_eq!(last_row(5, N, 5.5).unwrap(), Some(5));
        assert!(last_row(5, -1.0, N).is_err());
        assert!(last_row(5, N, -1.0).is_err());
    }
}
