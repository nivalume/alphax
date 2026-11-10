//! Fixed-point value types.
//!
//! Prices, quantities and money are stored as scaled integers, never as
//! `f64`. A backtest that sums a million fills in floating point accumulates
//! drift that is indistinguishable from alpha at the scale people care about.
//!
//! # Scale
//!
//! The raw representation is `value * 10^PRECISION`. The engine uses the
//! standard 9-decimal scale, which fits crypto and equities comfortably.
//! [`PriceRaw`] is a type alias so the high-precision variant (16 decimals on
//! `i128`, needed for 18-decimal on-chain quantities) is a type change rather
//! than a rewrite.
//!
//! # Sentinels
//!
//! [`Price::UNDEF`] uses `i64::MAX` rather than wrapping the type in
//! `Option`. The simulation kernels read prices in their innermost loop and
//! an `Option` discriminant there costs a branch per bar per column. The
//! trade-off is that arithmetic on an undefined price is nonsense rather than
//! a compile error, so constructors are the enforcement point.

/// Backing integer for prices. Widens to `i128` under high precision.
pub type PriceRaw = i64;

/// Backing integer for quantities. Unsigned: a quantity is never negative;
/// direction lives in [`crate::enums::OrderSide`].
pub type QuantityRaw = u64;

/// Decimal places in the raw representation.
pub const PRECISION: u8 = 9;

/// Scale factor implied by [`PRECISION`].
pub const SCALAR: f64 = 1_000_000_000.0;

/// A price, as a scaled integer with its own decimal precision.
///
/// `precision` is per-instance because instruments disagree: a JPY pair
/// quotes to 3 decimals, a BTC pair to 8. Arithmetic between two differently
/// scaled prices must reconcile them first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Price {
    raw: PriceRaw,
    precision: u8,
}

impl Price {
    /// Sentinel for "no price". Distinct from zero and from `NaN`.
    pub const UNDEF: Self = Self {
        raw: PriceRaw::MAX,
        precision: 0,
    };

    /// Sentinel for "price could not be computed".
    pub const ERROR: Self = Self {
        raw: PriceRaw::MIN,
        precision: 0,
    };

    /// Builds a price from a raw scaled integer.
    #[must_use]
    pub const fn from_raw(raw: PriceRaw, precision: u8) -> Self {
        Self { raw, precision }
    }

    /// The raw scaled integer.
    #[must_use]
    pub const fn raw(self) -> PriceRaw {
        self.raw
    }

    /// The decimal precision.
    #[must_use]
    pub const fn precision(self) -> u8 {
        self.precision
    }

    /// Whether this is one of the sentinel values.
    #[must_use]
    pub const fn is_sentinel(self) -> bool {
        self.raw == PriceRaw::MAX || self.raw == PriceRaw::MIN
    }
}

/// A quantity, as an unsigned scaled integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Quantity {
    raw: QuantityRaw,
    precision: u8,
}

impl Quantity {
    /// Builds a quantity from a raw scaled integer.
    #[must_use]
    pub const fn from_raw(raw: QuantityRaw, precision: u8) -> Self {
        Self { raw, precision }
    }

    /// The raw scaled integer.
    #[must_use]
    pub const fn raw(self) -> QuantityRaw {
        self.raw
    }

    /// The decimal precision.
    #[must_use]
    pub const fn precision(self) -> u8 {
        self.precision
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentinels_are_recognised_and_distinct_from_zero() {
        assert!(Price::UNDEF.is_sentinel());
        assert!(Price::ERROR.is_sentinel());
        assert_ne!(Price::UNDEF, Price::ERROR);
        assert!(!Price::from_raw(0, PRECISION).is_sentinel());
    }

    #[test]
    fn precision_is_carried_per_instance() {
        let jpy = Price::from_raw(150_500, 3);
        // 65_000.12345678 at 8 decimals.
        let btc = Price::from_raw(6_500_012_345_678, 8);
        assert_eq!(jpy.precision(), 3);
        assert_eq!(btc.precision(), 8);
        // Same nominal number at different scales is not the same price.
        assert_ne!(Price::from_raw(100, 2), Price::from_raw(100, 3));
    }
}
