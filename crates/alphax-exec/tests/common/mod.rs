//! Shared fixture plumbing for the golden tests.
//!
//! Both harnesses read the same TSV dialect and share the same notion of when
//! two floats agree, so that lives here rather than being duplicated and
//! allowed to drift apart.

#![allow(dead_code, reason = "each test binary uses a different subset")]

use std::collections::HashMap;
use std::path::PathBuf;

use alphax_core::math::is_close;
use alphax_exec::{
    AccountType, Direction, LeverageMode, MarginBasis, Order, OrderSide, OrderStatus, SizeType,
    StatusInfo,
};

/// Whether two floats agree for fixture purposes.
///
/// [`is_close`] alone is not enough: by design it reports `NaN` and infinities
/// as *not* close to anything, including themselves, because they are live
/// sentinels in the kernel. A fixture comparison has to treat matching
/// sentinels as agreement, so those two cases are handled before the tolerance
/// test.
pub fn agree(a: f64, b: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return a.is_nan() && b.is_nan();
    }
    if a.is_infinite() || b.is_infinite() {
        // Both infinite, and the same sign.
        return a.is_infinite() && b.is_infinite() && a.is_sign_positive() == b.is_sign_positive();
    }
    is_close(a, b)
}

/// A fixture file: a header of column names and the rows beneath it.
///
/// Columns are resolved by name, so reordering or renaming one in the
/// generator fails loudly here instead of silently shifting every value.
pub struct Table {
    cols: HashMap<String, usize>,
    rows: Vec<Vec<String>>,
}

impl Table {
    /// Loads a fixture from the repository's `tests/fixtures` directory.
    pub fn load(file_name: &str) -> Self {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures")
            .join(file_name);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));

        let mut lines = text
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty());

        let header: Vec<String> = lines
            .next()
            .unwrap_or_else(|| panic!("{} has no header", path.display()))
            .split('\t')
            .map(str::to_owned)
            .collect();

        let rows: Vec<Vec<String>> = lines
            .map(|line| {
                let fields: Vec<String> = line.split('\t').map(str::to_owned).collect();
                assert_eq!(
                    fields.len(),
                    header.len(),
                    "row {:?} has {} fields, header has {}",
                    fields.first().map(String::as_str).unwrap_or("?"),
                    fields.len(),
                    header.len()
                );
                fields
            })
            .collect();

        let cols = header
            .into_iter()
            .enumerate()
            .map(|(i, name)| (name, i))
            .collect();

        Self { cols, rows }
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the table has no rows.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Iterates the rows.
    pub fn iter(&self) -> impl Iterator<Item = Row<'_>> {
        self.rows.iter().map(move |fields| Row {
            cols: &self.cols,
            fields,
        })
    }
}

/// One row, addressed by column name.
#[derive(Clone, Copy)]
pub struct Row<'a> {
    cols: &'a HashMap<String, usize>,
    fields: &'a [String],
}

impl Row<'_> {
    /// The raw text of a column.
    pub fn str(&self, name: &str) -> &str {
        let idx = *self
            .cols
            .get(name)
            .unwrap_or_else(|| panic!("fixture has no column {name:?}"));
        &self.fields[idx]
    }

    /// A column parsed as `f64`. Accepts `nan`, `inf` and `-inf`.
    pub fn f64(&self, name: &str) -> f64 {
        let raw = self.str(name);
        raw.parse()
            .unwrap_or_else(|_| panic!("column {name:?}: cannot parse {raw:?} as f64"))
    }

    /// A column parsed as `usize`.
    pub fn usize(&self, name: &str) -> usize {
        let raw = self.str(name);
        raw.parse()
            .unwrap_or_else(|_| panic!("column {name:?}: cannot parse {raw:?} as usize"))
    }

    /// A column parsed as a bool.
    pub fn bool(&self, name: &str) -> bool {
        match self.str(name) {
            "true" => true,
            "false" => false,
            other => panic!("column {name:?}: cannot parse {other:?} as bool"),
        }
    }

    /// Builds the order described by this row's order columns.
    ///
    /// `reject_prob` is always zero in fixtures: it depends on a random
    /// draw, so it is covered by unit tests against injected draws instead.
    pub fn order(&self) -> Order {
        Order {
            size: self.f64("size"),
            price: self.f64("price"),
            size_type: parse_size_type(self.str("size_type")),
            direction: parse_direction(self.str("direction")),
            fees: self.f64("fees"),
            fixed_fees: self.f64("fixed_fees"),
            slippage: self.f64("slippage"),
            min_size: self.f64("min_size"),
            max_size: self.f64("max_size"),
            size_granularity: self.f64("size_granularity"),
            // Not exercised by the fixtures.
            leverage: 1.0,
            leverage_mode: LeverageMode::Lazy,
            account_type: AccountType::Cash,
            initial_margin: f64::NAN,
            margin_basis: MarginBasis::Notional,
            cash_limit: f64::INFINITY,
            reject_prob: 0.0,
            lock_cash: self.bool("lock_cash"),
            allow_partial: self.bool("allow_partial"),
            raise_reject: false,
            log: false,
            allow_negative_price: false,
        }
    }
}

pub fn parse_size_type(s: &str) -> SizeType {
    match s {
        "Amount" => SizeType::Amount,
        "Value" => SizeType::Value,
        "Percent" => SizeType::Percent,
        "TargetAmount" => SizeType::TargetAmount,
        "TargetValue" => SizeType::TargetValue,
        "TargetPercent" => SizeType::TargetPercent,
        other => panic!("unknown size type {other:?}"),
    }
}

pub fn parse_direction(s: &str) -> Direction {
    match s {
        "LongOnly" => Direction::LongOnly,
        "ShortOnly" => Direction::ShortOnly,
        "Both" => Direction::Both,
        other => panic!("unknown direction {other:?}"),
    }
}

pub fn parse_side(s: &str) -> Option<OrderSide> {
    match s {
        "Buy" => Some(OrderSide::Buy),
        "Sell" => Some(OrderSide::Sell),
        "none" => None,
        other => panic!("unknown side {other:?}"),
    }
}

pub fn parse_status(s: &str) -> OrderStatus {
    match s {
        "Filled" => OrderStatus::Filled,
        "Ignored" => OrderStatus::Ignored,
        "Rejected" => OrderStatus::Rejected,
        other => panic!("unknown status {other:?}"),
    }
}

pub fn parse_status_info(s: &str) -> Option<StatusInfo> {
    Some(match s {
        "none" => return None,
        "SizeNaN" => StatusInfo::SizeNaN,
        "PriceNaN" => StatusInfo::PriceNaN,
        "ValPriceNaN" => StatusInfo::ValPriceNaN,
        "ValueNaN" => StatusInfo::ValueNaN,
        "ValueZeroNeg" => StatusInfo::ValueZeroNeg,
        "SizeZero" => StatusInfo::SizeZero,
        "NoCashShort" => StatusInfo::NoCashShort,
        "NoCashLong" => StatusInfo::NoCashLong,
        "NoOpenPosition" => StatusInfo::NoOpenPosition,
        "MaxSizeExceeded" => StatusInfo::MaxSizeExceeded,
        "RandomEvent" => StatusInfo::RandomEvent,
        "CantCoverFees" => StatusInfo::CantCoverFees,
        "MinSizeNotReached" => StatusInfo::MinSizeNotReached,
        "PartialFill" => StatusInfo::PartialFill,
        other => panic!("unknown status info {other:?}"),
    })
}
