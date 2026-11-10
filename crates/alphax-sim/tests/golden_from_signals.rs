//! Golden test for the `from_signals` simulation loop.
//!
//! The signal pipeline is pinned function by function in `golden_signals.rs`,
//! and the execution kernel before that. What this verifies is everything
//! between and around them — the stop engine, and the order the stages of a
//! bar run in.
//!
//! Sequencing is what unit tests cannot reach on their own and where an implementation
//! drifts: a stop is tested before it trails, the valuation price is resolved
//! before a stop can rewrite the order price, conflicts are resolved only when
//! an entry is involved, and stops are cleared on a close but re-anchored on a
//! reversal.
//!
//! The expected outputs are committed in `tests/fixtures/from_signals.tsv`, so this
//! needs no Python.

use std::collections::HashMap;
use std::path::PathBuf;

use alphax_core::flex::FlexArray;
use alphax_core::math::is_close;
use alphax_model::{
    AccumulationMode, ConflictMode, Direction, DirectionConflictMode, OppositeEntryMode, OrderSide,
    SizeType, StopEntryPrice, StopExitMode, StopExitPrice, StopUpdateMode,
};
use alphax_sim::{FromSignals, simulate_from_signals, simulate_from_signals_parallel};

/// An array parameter as the fixture encodes it, owning its values so a
/// [`FlexArray`] can borrow from it.
enum Encoded<T> {
    Scalar(T),
    PerColumn(Vec<T>),
    PerRow(Vec<T>),
    Full(Vec<T>),
}

impl<T: Copy> Encoded<T> {
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

fn number(s: &str) -> f64 {
    match s {
        "inf" => f64::INFINITY,
        "-inf" => f64::NEG_INFINITY,
        "nan" => f64::NAN,
        other => other
            .parse()
            .unwrap_or_else(|_| panic!("cannot parse {other:?} as f64")),
    }
}

fn boolean(s: &str) -> bool {
    match s {
        "true" => true,
        "false" => false,
        other => panic!("cannot parse {other:?} as bool"),
    }
}

fn direction(s: &str) -> Direction {
    match s {
        "LongOnly" => Direction::LongOnly,
        "ShortOnly" => Direction::ShortOnly,
        "Both" => Direction::Both,
        other => panic!("unknown direction {other:?}"),
    }
}

fn size_type(s: &str) -> SizeType {
    match s {
        "Amount" => SizeType::Amount,
        "Value" => SizeType::Value,
        "Percent" => SizeType::Percent,
        other => panic!("unsupported size type {other:?}"),
    }
}

fn accumulation(s: &str) -> AccumulationMode {
    match s {
        "Disabled" => AccumulationMode::Disabled,
        "Both" => AccumulationMode::Both,
        "AddOnly" => AccumulationMode::AddOnly,
        "RemoveOnly" => AccumulationMode::RemoveOnly,
        other => panic!("unknown accumulation mode {other:?}"),
    }
}

fn conflict(s: &str) -> ConflictMode {
    match s {
        "Ignore" => ConflictMode::Ignore,
        "Entry" => ConflictMode::Entry,
        "Exit" => ConflictMode::Exit,
        "Adjacent" => ConflictMode::Adjacent,
        "Opposite" => ConflictMode::Opposite,
        other => panic!("unknown conflict mode {other:?}"),
    }
}

fn dir_conflict(s: &str) -> DirectionConflictMode {
    match s {
        "Ignore" => DirectionConflictMode::Ignore,
        "Long" => DirectionConflictMode::Long,
        "Short" => DirectionConflictMode::Short,
        "Adjacent" => DirectionConflictMode::Adjacent,
        "Opposite" => DirectionConflictMode::Opposite,
        other => panic!("unknown direction conflict mode {other:?}"),
    }
}

fn opposite(s: &str) -> OppositeEntryMode {
    match s {
        "Ignore" => OppositeEntryMode::Ignore,
        "Close" => OppositeEntryMode::Close,
        "CloseReduce" => OppositeEntryMode::CloseReduce,
        "Reverse" => OppositeEntryMode::Reverse,
        "ReverseReduce" => OppositeEntryMode::ReverseReduce,
        other => panic!("unknown opposite entry mode {other:?}"),
    }
}

fn stop_entry(s: &str) -> StopEntryPrice {
    match s {
        "ValPrice" => StopEntryPrice::ValPrice,
        "Price" => StopEntryPrice::Price,
        "FillPrice" => StopEntryPrice::FillPrice,
        "Close" => StopEntryPrice::Close,
        other => panic!("unknown stop entry price {other:?}"),
    }
}

fn stop_exit_price(s: &str) -> StopExitPrice {
    match s {
        "StopLimit" => StopExitPrice::StopLimit,
        "StopMarket" => StopExitPrice::StopMarket,
        "Price" => StopExitPrice::Price,
        "Close" => StopExitPrice::Close,
        other => panic!("unknown stop exit price {other:?}"),
    }
}

fn stop_exit(s: &str) -> StopExitMode {
    match s {
        "Close" => StopExitMode::Close,
        "CloseReduce" => StopExitMode::CloseReduce,
        "Reverse" => StopExitMode::Reverse,
        "ReverseReduce" => StopExitMode::ReverseReduce,
        other => panic!("unknown stop exit mode {other:?}"),
    }
}

fn stop_update(s: &str) -> StopUpdateMode {
    match s {
        "Keep" => StopUpdateMode::Keep,
        "Override" => StopUpdateMode::Override,
        "OverrideNaN" => StopUpdateMode::OverrideNaN,
        other => panic!("unknown stop update mode {other:?}"),
    }
}

fn side(s: &str) -> OrderSide {
    match s {
        "Buy" => OrderSide::Buy,
        "Sell" => OrderSide::Sell,
        other => panic!("unknown side {other:?}"),
    }
}

/// Splits a `;`-joined record column, where `-` means empty.
fn split_records<T>(text: &str, parse_value: impl Fn(&str) -> T) -> Vec<T> {
    if text == "-" || text.is_empty() {
        Vec::new()
    } else {
        text.split(';').map(parse_value).collect()
    }
}

/// Sentinel-aware agreement; see the note in `alphax-exec`'s harness.
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
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/from_signals.tsv");
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

/// Runs one fixture row, returning the harness's verdict lines.
fn check_row(fixture: &Fixture, row: &[String], failures: &mut Vec<String>, fills: &mut usize) {
    let g = |name: &str| fixture.get(row, name);
    let case = g("case");
    let n_rows: usize = g("n_rows").parse().unwrap();
    let n_cols: usize = g("n_cols").parse().unwrap();

    // Owned first, so the FlexArrays below can borrow.
    let init_cash = Encoded::parse(g("init_cash"), number);
    let open = Encoded::parse(g("open"), number);
    let high = Encoded::parse(g("high"), number);
    let low = Encoded::parse(g("low"), number);
    let close = Encoded::parse(g("close"), number);
    let entries = Encoded::parse(g("entries"), boolean);
    let exits = Encoded::parse(g("exits"), boolean);
    let dir = Encoded::parse(g("direction"), direction);
    let long_entries = Encoded::parse(g("long_entries"), boolean);
    let long_exits = Encoded::parse(g("long_exits"), boolean);
    let short_entries = Encoded::parse(g("short_entries"), boolean);
    let short_exits = Encoded::parse(g("short_exits"), boolean);
    let size = Encoded::parse(g("size"), number);
    let st = Encoded::parse(g("size_type"), size_type);
    let price = Encoded::parse(g("price"), number);
    let fees = Encoded::parse(g("fees"), number);
    let fixed_fees = Encoded::parse(g("fixed_fees"), number);
    let slippage = Encoded::parse(g("slippage"), number);
    let min_size = Encoded::parse(g("min_size"), number);
    let max_size = Encoded::parse(g("max_size"), number);
    let size_granularity = Encoded::parse(g("size_granularity"), number);
    let lock_cash = Encoded::parse(g("lock_cash"), boolean);
    let allow_partial = Encoded::parse(g("allow_partial"), boolean);
    let accumulate = Encoded::parse(g("accumulate"), accumulation);
    let upon_long_conflict = Encoded::parse(g("upon_long_conflict"), conflict);
    let upon_short_conflict = Encoded::parse(g("upon_short_conflict"), conflict);
    let upon_dir_conflict = Encoded::parse(g("upon_dir_conflict"), dir_conflict);
    let upon_opposite_entry = Encoded::parse(g("upon_opposite_entry"), opposite);
    let sl_stop = Encoded::parse(g("sl_stop"), number);
    let sl_trail = Encoded::parse(g("sl_trail"), boolean);
    let tp_stop = Encoded::parse(g("tp_stop"), number);
    let stop_entry_price = Encoded::parse(g("stop_entry_price"), stop_entry);
    let sep = Encoded::parse(g("stop_exit_price"), stop_exit_price);
    let upon_stop_exit = Encoded::parse(g("upon_stop_exit"), stop_exit);
    let upon_stop_update = Encoded::parse(g("upon_stop_update"), stop_update);
    let val_price = Encoded::parse(g("val_price"), number);

    let args = FromSignals {
        groups: None,
        call_seq: alphax_sim::groups::CallSeq::Default,
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
        open: open.as_flex(n_rows, n_cols),
        high: high.as_flex(n_rows, n_cols),
        low: low.as_flex(n_rows, n_cols),
        close: close.as_flex(n_rows, n_cols),
        entries: entries.as_flex(n_rows, n_cols),
        exits: exits.as_flex(n_rows, n_cols),
        direction: dir.as_flex(n_rows, n_cols),
        long_entries: long_entries.as_flex(n_rows, n_cols),
        long_exits: long_exits.as_flex(n_rows, n_cols),
        short_entries: short_entries.as_flex(n_rows, n_cols),
        short_exits: short_exits.as_flex(n_rows, n_cols),
        size: size.as_flex(n_rows, n_cols),
        size_type: st.as_flex(n_rows, n_cols),
        price: price.as_flex(n_rows, n_cols),
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
        order_mode: false,
        accumulate: accumulate.as_flex(n_rows, n_cols),
        upon_long_conflict: upon_long_conflict.as_flex(n_rows, n_cols),
        upon_short_conflict: upon_short_conflict.as_flex(n_rows, n_cols),
        upon_dir_conflict: upon_dir_conflict.as_flex(n_rows, n_cols),
        upon_opposite_entry: upon_opposite_entry.as_flex(n_rows, n_cols),
        sl_stop: sl_stop.as_flex(n_rows, n_cols),
        sl_trail: sl_trail.as_flex(n_rows, n_cols),
        tp_stop: tp_stop.as_flex(n_rows, n_cols),
        stop_entry_price: stop_entry_price.as_flex(n_rows, n_cols),
        stop_exit_price: sep.as_flex(n_rows, n_cols),
        upon_stop_exit: upon_stop_exit.as_flex(n_rows, n_cols),
        upon_stop_update: upon_stop_update.as_flex(n_rows, n_cols),
        order_type: FlexArray::Scalar(alphax_model::OrderType::Market),
        limit_delta: FlexArray::Scalar(f64::NAN),
        delta_format: FlexArray::Scalar(alphax_model::DeltaFormat::Percent),
        limit_reverse: FlexArray::Scalar(false),
        limit_order_price: FlexArray::Scalar(alphax_model::LimitOrderPrice::Limit),
        limit_delay: FlexArray::Scalar(false),
        limit_tif: FlexArray::Scalar(f64::NAN),
        limit_expiry: FlexArray::Scalar(f64::NAN),
        upon_adj_limit_conflict: FlexArray::Scalar(alphax_model::PendingConflictMode::KeepIgnore),
        upon_opp_limit_conflict: FlexArray::Scalar(
            alphax_model::PendingConflictMode::CancelExecute,
        ),
        upon_adj_stop_conflict: FlexArray::Scalar(alphax_model::PendingConflictMode::KeepIgnore),
        upon_opp_stop_conflict: FlexArray::Scalar(alphax_model::PendingConflictMode::KeepIgnore),
        tsl_stop: FlexArray::Scalar(f64::NAN),
        tsl_th: FlexArray::Scalar(f64::NAN),
        td_stop: FlexArray::Scalar(f64::NAN),
        dt_stop: FlexArray::Scalar(f64::NAN),
        stop_ladder: alphax_model::StopLadderMode::Disabled,
        stop_order_type: FlexArray::Scalar(alphax_model::OrderType::Market),
        stop_limit_delta: FlexArray::Scalar(f64::NAN),
        use_stops: boolean(g("use_stops")),
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
        ffill_val_price: boolean(g("ffill_val_price")),
        update_value: boolean(g("update_value")),
    };

    let result = simulate_from_signals(&args);

    if g("outcome") == "error" {
        if result.is_ok() {
            failures.push(format!("{case}: expected a rejection, got records"));
        }
        return;
    }

    let records = match result {
        Ok(r) => r,
        Err(e) => {
            failures.push(format!("{case}: unexpected error: {e}"));
            return;
        }
    };

    let want_idx = split_records(g("rec_idx"), |s| s.parse::<i64>().expect("idx"));
    if records.len() != want_idx.len() {
        failures.push(format!(
            "{case}: expected {} fills, got {}",
            want_idx.len(),
            records.len()
        ));
        return;
    }
    *fills += records.len();

    let want_col = split_records(g("rec_col"), |s| s.parse::<i64>().expect("col"));
    let want_side = split_records(g("rec_side"), side);
    let want_size = split_records(g("rec_size"), number);
    let want_price = split_records(g("rec_price"), number);
    let want_fees = split_records(g("rec_fees"), number);

    for k in 0..records.len() {
        // Indices, columns and sides are compared exactly; a tolerance on
        // them would hide the off-by-one errors this exists to catch.
        if records.idx[k] != want_idx[k] {
            failures.push(format!(
                "{case}[{k}].idx: got {}, want {}",
                records.idx[k], want_idx[k]
            ));
        }
        if records.col[k] != want_col[k] {
            failures.push(format!(
                "{case}[{k}].col: got {}, want {}",
                records.col[k], want_col[k]
            ));
        }
        if records.side[k] != want_side[k] as i8 {
            failures.push(format!(
                "{case}[{k}].side: got {}, want {:?}",
                records.side[k], want_side[k]
            ));
        }
        for (name, got, want) in [
            ("size", records.size[k], want_size[k]),
            ("price", records.price[k], want_price[k]),
            ("fees", records.fees[k], want_fees[k]),
        ] {
            if !agree(got, want) {
                failures.push(format!("{case}[{k}].{name}: got {got}, want {want}"));
            }
        }
    }

    // The parallel path must be bit-identical, on every fixture rather than
    // on a chosen one: threading bugs show up on the awkward shapes.
    if n_cols > 1 {
        let parallel = simulate_from_signals_parallel(&args, 3).expect("parallel run");
        if parallel != records {
            failures.push(format!(
                "{case}: the parallel path disagrees with the serial one"
            ));
        }
    }
}

#[test]
fn matches_the_golden_loop() {
    let fixture = Fixture::load();
    assert!(
        fixture.rows.len() > 150,
        "expected the full fixture set, got {}",
        fixture.rows.len()
    );

    let mut failures: Vec<String> = Vec::new();
    let mut fills = 0usize;
    for row in &fixture.rows {
        check_row(&fixture, row, &mut failures, &mut fills);
    }

    assert!(
        fills > 500,
        "only {fills} fills across the set: it is not exercising much"
    );
    assert!(
        failures.is_empty(),
        "{} disagreements with the golden outputs:\n{}",
        failures.len(),
        failures
            .iter()
            .take(40)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// The fixture set must keep covering the stop engine's branches.
///
/// The gap branch is the one that matters most and the easiest to lose: a
/// fixture set where every stop level happens to sit inside its bar would
/// pass while the branch that fills at the open is never reached.
#[test]
fn the_fixtures_still_cover_the_stop_engine() {
    let fixture = Fixture::load();
    let mut with_stops = 0usize;
    let mut trailing = 0usize;
    let mut take_profit = 0usize;
    let mut rejections = 0usize;

    for row in &fixture.rows {
        let g = |name: &str| fixture.get(row, name);
        if g("outcome") == "error" {
            rejections += 1;
            continue;
        }
        if !boolean(g("use_stops")) {
            continue;
        }
        if g("sl_stop") != "s:nan" {
            with_stops += 1;
        }
        if g("sl_trail").contains("true") {
            trailing += 1;
        }
        if g("tp_stop") != "s:nan" {
            take_profit += 1;
        }
    }

    assert!(with_stops > 20, "only {with_stops} cases with a stop-loss");
    assert!(trailing > 5, "only {trailing} cases with a trailing stop");
    assert!(
        take_profit > 5,
        "only {take_profit} cases with a take-profit"
    );
    assert!(rejections >= 2, "only {rejections} rejection cases");
}
