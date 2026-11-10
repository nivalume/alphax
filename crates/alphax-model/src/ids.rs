//! Identifiers.
//!
//! `Symbol` and `Venue` wrap an `Arc<str>`: a clone shares the string, and
//! comparing or hashing reads it. They are not interned. Interning would turn
//! each into a `Copy` handle that compares as an integer, which matters if
//! profiling shows identifier comparisons on the record path; the types are
//! opaque, so changing the store would not touch call sites.

use std::sync::Arc;

/// A tradeable symbol, e.g. `BTCUSDT`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Symbol(Arc<str>);

impl Symbol {
    /// A symbol from its name.
    pub fn new(s: impl AsRef<str>) -> Self {
        Self(Arc::from(s.as_ref()))
    }

    /// The symbol as a string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A trading venue, e.g. `BINANCE`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Venue(Arc<str>);

impl Venue {
    /// A venue from its name.
    pub fn new(s: impl AsRef<str>) -> Self {
        Self(Arc::from(s.as_ref()))
    }

    /// The venue as a string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A symbol qualified by its venue.
///
/// Rendered as `SYMBOL.VENUE`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InstrumentId {
    /// The symbol.
    pub symbol: Symbol,
    /// The venue.
    pub venue: Venue,
}

impl InstrumentId {
    /// Validate separate identity components without reinterpreting a dotted venue.
    pub fn from_parts(symbol: &str, venue: &str) -> Result<Self, String> {
        if symbol.is_empty() || venue.is_empty() || venue.contains('.') {
            return Err(
                "instrument requires a nonempty symbol and a nonempty venue without dots".into(),
            );
        }
        format!("{symbol}.{venue}").parse()
    }

    /// Builds an instrument id from its parts.
    #[must_use]
    pub const fn new(symbol: Symbol, venue: Venue) -> Self {
        Self { symbol, venue }
    }
}

impl std::fmt::Display for InstrumentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.symbol.as_str(), self.venue.as_str())
    }
}

impl std::str::FromStr for InstrumentId {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let (symbol, venue) = value.rsplit_once('.').ok_or("expected SYMBOL.VENUE")?;
        if symbol.is_empty()
            || venue.is_empty()
            || value
                .chars()
                .any(|c| c.is_whitespace() || c.is_control() || matches!(c, '/' | '\\'))
        {
            return Err("instrument requires nonempty symbol and venue, without whitespace or path separators".into());
        }
        Ok(Self::new(Symbol::new(symbol), Venue::new(venue)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_as_symbol_dot_venue() {
        let id = InstrumentId::new(Symbol::new("BTCUSDT"), Venue::new("BINANCE"));
        assert_eq!(id.to_string(), "BTCUSDT.BINANCE");
    }

    #[test]
    fn equal_names_compare_equal() {
        assert_eq!(Symbol::new("BTC"), Symbol::new("BTC"));
        assert_ne!(Symbol::new("BTC"), Symbol::new("ETH"));
    }
}
