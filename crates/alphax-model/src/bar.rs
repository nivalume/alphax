//! Bar / OHLCV data.
//!
//! The engine is bar-only by design. The
//! simulation loops read OHLCV as four parallel column-major `f64` arrays
//! rather than as a `Bar` struct per row — a struct-of-arrays layout, because
//! the stop engine touches `open`, `high` and `low` in separate passes and an
//! array-of-structs layout would pull the whole row into cache for each.
//!
//! The [`Bar`] struct is for code that handles one bar at a time.

use crate::ids::InstrumentId;
use crate::types::{Price, Quantity};

/// Nanoseconds since the Unix epoch.
pub type UnixNanos = u64;

/// A single OHLCV bar.
///
/// Carries two timestamps: `ts_event` is when the bar closed at the venue,
/// `ts_init` is when it became available to us. Replay orders by `ts_init`,
/// because that is the only one that cannot leak the future.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bar {
    /// Which instrument this bar belongs to.
    pub instrument_id: InstrumentId,
    /// Opening price.
    pub open: Price,
    /// Highest price.
    pub high: Price,
    /// Lowest price.
    pub low: Price,
    /// Closing price.
    pub close: Price,
    /// Traded volume.
    pub volume: Quantity,
    /// When the bar closed at the venue.
    pub ts_event: UnixNanos,
    /// When the bar became available to the engine.
    pub ts_init: UnixNanos,
}

/// Column-major OHLCV panel: the shape the simulation loops actually consume.
///
/// Each field is `n_rows * n_cols` elements, column-major, so one column's
/// full history is contiguous.
#[derive(Debug, Clone, Default)]
pub struct OhlcvPanel {
    /// Opening prices.
    pub open: Vec<f64>,
    /// Highest prices.
    pub high: Vec<f64>,
    /// Lowest prices.
    pub low: Vec<f64>,
    /// Closing prices.
    pub close: Vec<f64>,
    /// Traded volumes.
    pub volume: Vec<f64>,
    /// Number of bars.
    pub n_rows: usize,
    /// Number of columns.
    pub n_cols: usize,
}

/// Identity of a bar series, distinct from the instrument's identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BarType {
    pub instrument: InstrumentId,
    pub step: u64,
    pub aggregation: String,
    pub price_type: String,
    pub source: String,
}

impl std::str::FromStr for BarType {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let parts: Vec<_> = value.rsplitn(5, '-').collect();
        if parts.len() != 5 {
            return Err("expected SYMBOL.VENUE-STEP-AGGREGATION-PRICE-SOURCE".into());
        }
        let step = parts[3]
            .parse::<u64>()
            .map_err(|_| "bar step must be positive")?;
        let aggregation = parts[2];
        if step == 0
            || !matches!(parts[0], "INTERNAL" | "EXTERNAL")
            || !matches!(parts[1], "BID" | "ASK" | "MID" | "LAST")
            || !matches!(
                aggregation,
                "MILLISECOND"
                    | "SECOND"
                    | "MINUTE"
                    | "HOUR"
                    | "DAY"
                    | "WEEK"
                    | "MONTH"
                    | "TICK"
                    | "VOLUME"
                    | "VALUE"
                    | "TICK_IMBALANCE"
                    | "TICK_RUNS"
                    | "VOLUME_IMBALANCE"
                    | "VOLUME_RUNS"
                    | "VALUE_IMBALANCE"
                    | "VALUE_RUNS"
            )
        {
            return Err("invalid bar step, aggregation, price type or source".into());
        }
        Ok(Self {
            instrument: parts[4].parse()?,
            step,
            aggregation: aggregation.into(),
            price_type: parts[1].into(),
            source: parts[0].into(),
        })
    }
}

impl std::fmt::Display for BarType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}-{}-{}-{}-{}",
            self.instrument, self.step, self.aggregation, self.price_type, self.source
        )
    }
}

#[cfg(test)]
mod bar_type_tests {
    use super::*;
    #[test]
    fn parse_preserves_series_identity() {
        let text = "BTC-USDT.BINANCE-1-MINUTE-LAST-EXTERNAL";
        let b: BarType = text.parse().unwrap();
        assert_eq!(b.to_string(), text);
        assert_eq!(b.instrument.symbol.as_str(), "BTC-USDT");
        for invalid in [
            "BTC.BINANCE",
            "BTC.BINANCE-0-MINUTE-LAST-EXTERNAL",
            "BTC.BINANCE-1-WHATEVER-LAST-EXTERNAL",
        ] {
            assert!(invalid.parse::<BarType>().is_err());
        }
    }
}
