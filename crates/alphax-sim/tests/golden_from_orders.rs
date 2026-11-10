//! Golden test for the `from_orders` simulation loop.
//!
//! The kernel harnesses in `alphax-exec` verify one order and chains of orders.
//! This one verifies the loop that drives them: price-sentinel resolution,
//! valuation-price forward-fill, per-element parameter broadcasting, and the
//! resulting record arrays.
//!
//! Because the kernel is already pinned by its own golden tests, a failure here
//! is almost certainly the loop rather than the arithmetic — which is the
//! whole point of having built the harnesses in that order.
//!
//! The expected outputs are committed in `tests/fixtures/from_orders.tsv`, so this
//! needs no Python.

use std::collections::HashMap;
use std::path::PathBuf;

use alphax_core::flex::FlexArray;
use alphax_core::math::is_close;
use alphax_model::{Direction, OrderSide, SizeType};
use alphax_sim::groups::{CallSeq, Groups};
use alphax_sim::{FromOrders, simulate_from_orders, simulate_from_orders_parallel};

/// An array parameter as the fixture encodes it, owning its values so a
/// [`FlexArray`] can borrow from it.
enum Encoded<T> {
    Scalar(T),
    PerColumn(Vec<T>),
    PerRow(Vec<T>),
    Full(Vec<T>),
}

impl<T: Copy> Encoded<T> {
    /// Parses `"s:v"`, `"c:a;b"`, `"r:a;b"` or `"f:a;b;..."`.
    fn parse(text: &str, parse_value: impl Fn(&str) -> T) -> Self {
        let (kind, rest) = text
            .split_once(':')
            .unwrap_or_else(|| panic!("malformed encoded array {text:?}"));
        let values: Vec<T> = rest.split(';').map(&parse_value).collect();
        match kind {
            "s" => {
                assert_eq!(values.len(), 1, "scalar encoding must hold one value");
                Self::Scalar(values[0])
            }
            "c" => Self::PerColumn(values),
            "r" => Self::PerRow(values),
            "f" => Self::Full(values),
            other => panic!("unknown array encoding {other:?}"),
        }
    }

    fn as_flex(&self, n_rows: usize, n_cols: usize) -> FlexArray<'_, T> {
        match self {
            Self::Scalar(v) => FlexArray::Scalar(*v),
            Self::PerColumn(v) => FlexArray::per_column(v, n_cols).expect("per-column length"),
            Self::PerRow(v) => FlexArray::per_row(v, n_rows).expect("per-row length"),
            Self::Full(v) => FlexArray::full(v, n_rows, n_cols).expect("full length"),
        }
    }
}

fn parse_f64(s: &str) -> f64 {
    s.parse()
        .unwrap_or_else(|_| panic!("cannot parse {s:?} as f64"))
}

fn parse_bool(s: &str) -> bool {
    match s {
        "true" => true,
        "false" => false,
        other => panic!("cannot parse {other:?} as bool"),
    }
}

fn parse_size_type(s: &str) -> SizeType {
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

fn parse_direction(s: &str) -> Direction {
    match s {
        "LongOnly" => Direction::LongOnly,
        "ShortOnly" => Direction::ShortOnly,
        "Both" => Direction::Both,
        other => panic!("unknown direction {other:?}"),
    }
}

fn parse_side(s: &str) -> OrderSide {
    match s {
        "Buy" => OrderSide::Buy,
        "Sell" => OrderSide::Sell,
        other => panic!("unknown side {other:?}"),
    }
}

/// Splits a `;`-joined record column, where `-` means empty.
fn split_records<T>(text: &str, parse_value: impl Fn(&str) -> T) -> Vec<T> {
    if text == "-" {
        Vec::new()
    } else {
        text.split(';').map(parse_value).collect()
    }
}

/// See the note on `agree` in `alphax-exec`'s harness: `is_close` reports `NaN`
/// and infinities as close to nothing, including themselves, because they are
/// live sentinels. Fixture comparison has to treat matching sentinels as
/// agreement.
fn agree(a: f64, b: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return a.is_nan() && b.is_nan();
    }
    if a.is_infinite() || b.is_infinite() {
        return a.is_infinite() && b.is_infinite() && a.is_sign_positive() == b.is_sign_positive();
    }
    is_close(a, b)
}

struct Fixture {
    header: HashMap<String, usize>,
    rows: Vec<Vec<String>>,
}

impl Fixture {
    fn load() -> Self {
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/from_orders.tsv");
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let mut lines = text
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty());
        let names: Vec<String> = lines
            .next()
            .expect("fixture has no header")
            .split('\t')
            .map(str::to_owned)
            .collect();
        let rows = lines
            .map(|line| {
                let fields: Vec<String> = line.split('\t').map(str::to_owned).collect();
                assert_eq!(fields.len(), names.len(), "row width mismatch");
                fields
            })
            .collect();
        let header = names.into_iter().enumerate().map(|(i, n)| (n, i)).collect();
        Self { header, rows }
    }

    fn get<'a>(&self, row: &'a [String], name: &str) -> &'a str {
        let idx = *self
            .header
            .get(name)
            .unwrap_or_else(|| panic!("fixture has no column {name:?}"));
        &row[idx]
    }
}

#[test]
fn matches_the_golden_loop() {
    let fixture = Fixture::load();
    assert!(
        fixture.rows.len() > 200,
        "expected the full fixture set, got {}",
        fixture.rows.len()
    );

    let mut failures: Vec<String> = Vec::new();
    let mut total_fills = 0usize;

    for row in &fixture.rows {
        let g = |name: &str| fixture.get(row, name);
        let case = g("case");
        let n_rows: usize = g("n_rows").parse().unwrap();
        let n_cols: usize = g("n_cols").parse().unwrap();

        // Owned first, so the FlexArrays below can borrow.
        let init_cash = Encoded::parse(g("init_cash"), parse_f64);
        let close = Encoded::parse(g("close"), parse_f64);
        let size = Encoded::parse(g("size"), parse_f64);
        let price = Encoded::parse(g("price"), parse_f64);
        let size_type = Encoded::parse(g("size_type"), parse_size_type);
        let direction = Encoded::parse(g("direction"), parse_direction);
        let fees = Encoded::parse(g("fees"), parse_f64);
        let fixed_fees = Encoded::parse(g("fixed_fees"), parse_f64);
        let slippage = Encoded::parse(g("slippage"), parse_f64);
        let min_size = Encoded::parse(g("min_size"), parse_f64);
        let max_size = Encoded::parse(g("max_size"), parse_f64);
        let size_granularity = Encoded::parse(g("size_granularity"), parse_f64);
        let lock_cash = Encoded::parse(g("lock_cash"), parse_bool);
        let allow_partial = Encoded::parse(g("allow_partial"), parse_bool);
        let val_price = Encoded::parse(g("val_price"), parse_f64);

        let args = FromOrders {
            n_rows,
            n_cols,
            init_cash: init_cash.as_flex(n_rows, n_cols),
            init_position: FlexArray::Scalar(0.0),
            init_price: FlexArray::Scalar(f64::NAN),
            allow_negative_price: false,
            cash_maintenance_margin: FlexArray::Scalar(f64::NAN),
            margin_tier_notional: FlexArray::Scalar(f64::NAN),
            margin_tier_rate: FlexArray::Scalar(f64::NAN),
            liquidation_step: FlexArray::Scalar(f64::NAN),
            auto_deleverage: FlexArray::Scalar(false),
            resume: None,
            row_offset: 0,
            close: close.as_flex(n_rows, n_cols),
            size: size.as_flex(n_rows, n_cols),
            price: price.as_flex(n_rows, n_cols),
            size_type: size_type.as_flex(n_rows, n_cols),
            direction: direction.as_flex(n_rows, n_cols),
            fees: fees.as_flex(n_rows, n_cols),
            fixed_fees: fixed_fees.as_flex(n_rows, n_cols),
            slippage: slippage.as_flex(n_rows, n_cols),
            min_size: min_size.as_flex(n_rows, n_cols),
            max_size: max_size.as_flex(n_rows, n_cols),
            size_granularity: size_granularity.as_flex(n_rows, n_cols),
            lock_cash: lock_cash.as_flex(n_rows, n_cols),
            allow_partial: allow_partial.as_flex(n_rows, n_cols),
            raise_reject: FlexArray::Scalar(false),
            log: FlexArray::Scalar(false),
            val_price: val_price.as_flex(n_rows, n_cols),
            // No cash flows outside fills.
            cash_deposits: FlexArray::Scalar(0.0),
            cash_earnings: FlexArray::Scalar(0.0),
            cash_dividends: FlexArray::Scalar(0.0),
            // None of these is in play here.
            leverage: FlexArray::Scalar(1.0),
            leverage_mode: FlexArray::Scalar(alphax_model::LeverageMode::Lazy),
            account_type: FlexArray::Scalar(alphax_model::AccountType::Cash),
            initial_margin: FlexArray::Scalar(f64::NAN),
            margin_basis: FlexArray::Scalar(alphax_model::MarginBasis::Notional),
            multiplier: FlexArray::Scalar(1.0),
            price_type: FlexArray::Scalar(alphax_model::PriceType::Price),
            from_ago: FlexArray::Scalar(0),
            bid_price: FlexArray::Scalar(f64::NAN),
            ask_price: FlexArray::Scalar(f64::NAN),
            price_area_vio_mode: FlexArray::Scalar(alphax_model::PriceAreaVioMode::Ignore),
            force_exit: FlexArray::Scalar(false),
            cash_limit: FlexArray::Scalar(f64::INFINITY),
            reject_prob: FlexArray::Scalar(0.0),
            seed: 0,
            funding_rate: FlexArray::Scalar(0.0),
            long_financing_rate: FlexArray::Scalar(0.0),
            short_financing_rate: FlexArray::Scalar(0.0),
            debt_rate: FlexArray::Scalar(0.0),
            borrow_rate: FlexArray::Scalar(0.0),
            mark_price: FlexArray::Scalar(f64::NAN),
            maintenance_margin: FlexArray::Scalar(0.0),
            liquidation_fee: FlexArray::Scalar(0.0),
            margin_mode: FlexArray::Scalar(alphax_model::MarginMode::Cross),
            open: FlexArray::Scalar(f64::NAN),
            high: FlexArray::Scalar(f64::NAN),
            low: FlexArray::Scalar(f64::NAN),
            ffill_val_price: parse_bool(g("ffill_val_price")),
            update_value: parse_bool(g("update_value")),
            groups: Groups::ungrouped(n_cols),
            call_seq: CallSeq::Default,
        };

        let outcome = simulate_from_orders(&args);

        match (g("outcome"), &outcome) {
            ("err", Err(_)) => {}
            ("err", Ok(records)) => failures.push(format!(
                "{case}: expected an error ({}), we produced {} fills",
                g("note"),
                records.len()
            )),
            ("ok", Err(e)) => failures.push(format!("{case}: expected success, we errored: {e}")),
            ("ok", Ok(records)) => {
                let want_col = split_records(g("rec_col"), |s| s.parse::<i64>().unwrap());
                let want_idx = split_records(g("rec_idx"), |s| s.parse::<i64>().unwrap());
                let want_size = split_records(g("rec_size"), parse_f64);
                let want_price = split_records(g("rec_price"), parse_f64);
                let want_fees = split_records(g("rec_fees"), parse_f64);
                let want_side: Vec<i8> = split_records(g("rec_side"), |s| parse_side(s) as i8);
                total_fills += want_col.len();

                if records.len() != want_col.len() {
                    failures.push(format!(
                        "{case}: fill count: want {}, got {}",
                        want_col.len(),
                        records.len()
                    ));
                    continue;
                }

                let mut diffs: Vec<String> = Vec::new();
                // Discrete fields must match exactly. `col` and `idx` also
                // pin the record *ordering*, which is the contract that makes
                // a parallel run reproducible.
                for (k, (want, got)) in want_col.iter().zip(&records.col).enumerate() {
                    if want != got {
                        diffs.push(format!("[{k}].col: want {want}, got {got}"));
                    }
                }
                for (k, (want, got)) in want_idx.iter().zip(&records.idx).enumerate() {
                    if want != got {
                        diffs.push(format!("[{k}].idx: want {want}, got {got}"));
                    }
                }
                for (k, (want, got)) in want_side.iter().zip(&records.side).enumerate() {
                    if want != got {
                        diffs.push(format!("[{k}].side: want {want}, got {got}"));
                    }
                }
                for (k, (want, got)) in want_size.iter().zip(&records.size).enumerate() {
                    if !agree(*want, *got) {
                        diffs.push(format!("[{k}].size: want {want:?}, got {got:?}"));
                    }
                }
                for (k, (want, got)) in want_price.iter().zip(&records.price).enumerate() {
                    if !agree(*want, *got) {
                        diffs.push(format!("[{k}].price: want {want:?}, got {got:?}"));
                    }
                }
                for (k, (want, got)) in want_fees.iter().zip(&records.fees).enumerate() {
                    if !agree(*want, *got) {
                        diffs.push(format!("[{k}].fees: want {want:?}, got {got:?}"));
                    }
                }

                if !diffs.is_empty() {
                    // Only the first few, or a broad break buries the signal.
                    diffs.truncate(4);
                    failures.push(format!("{case}: {}", diffs.join("; ")));
                }

                // The same input through the parallel driver must produce
                // byte-identical records, whatever the chunking.
                for threads in [2, 4] {
                    let parallel = simulate_from_orders_parallel(&args, threads)
                        .expect("parallel run failed where serial succeeded");
                    if &parallel != records {
                        failures.push(format!(
                            "{case}: {threads} threads produced different records than 1"
                        ));
                    }
                }
            }
            (other, _) => panic!("{case}: unknown outcome {other:?}"),
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} cases disagree with the golden outputs ({total_fills} fills compared):\n  {}",
        failures.len(),
        fixture.rows.len(),
        failures.join("\n  ")
    );
}

#[test]
fn fixtures_exercise_every_broadcasting_path() {
    // The four FlexArray variants are the memory story behind large sweeps.
    // If the fixtures stopped covering one, a bug in it would go unnoticed.
    let fixture = Fixture::load();
    let mut kinds: Vec<char> = Vec::new();
    for row in &fixture.rows {
        for param in ["close", "size", "price", "fees", "direction", "val_price"] {
            if let Some(k) = fixture.get(row, param).chars().next()
                && !kinds.contains(&k)
            {
                kinds.push(k);
            }
        }
    }
    for want in ['s', 'c', 'r', 'f'] {
        assert!(
            kinds.contains(&want),
            "no fixture uses the {want:?} broadcasting encoding"
        );
    }
}
