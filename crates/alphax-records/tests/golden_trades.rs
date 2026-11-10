//! Golden test for the three trade views.
//!
//! The simulation harnesses verify that the right fills come out. This one
//! verifies what those fills *mean*: how an exit is attributed back to the
//! entries it closes, which is the part with three defensible answers and
//! therefore the part most easily replaced by a fourth, wrong one.
//!
//! The fixture feeds order records directly rather than running a simulation,
//! so a failure here is the derivation and not whatever produced the fills.
//! Some cases carry fills from a real `from_orders` run anyway, to confirm
//! the two agree on realistic input.
//!
//! The expected outputs are committed in `tests/fixtures/trades.tsv`, so this
//! needs no Python.

use std::collections::HashMap;
use std::path::PathBuf;

use alphax_core::flex::FlexArray;
use alphax_core::math::is_close;
use alphax_model::{OrderSide, TradeDirection, TradeStatus};
use alphax_records::{
    OrderRecords, RecordSink, TradeRecords, entry_trades, exit_trades, positions,
};

/// See the note in `alphax-exec`'s harness: `is_close` reports `NaN` and
/// infinities as close to nothing, including themselves, because they are
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
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/trades.tsv");
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

/// Parses `idx,col,size,price,fees,side | ...` into order records.
fn parse_orders(text: &str) -> OrderRecords {
    let mut out = OrderRecords::new();
    if text.is_empty() {
        return out;
    }
    for entry in text.split('|') {
        let f: Vec<&str> = entry.split(',').collect();
        assert_eq!(f.len(), 6, "malformed order {entry:?}");
        let side = match f[5] {
            "Buy" => OrderSide::Buy,
            "Sell" => OrderSide::Sell,
            other => panic!("unknown side {other:?}"),
        };
        out.push_order(
            f[0].parse().expect("idx"),
            f[1].parse().expect("col"),
            f[2].parse().expect("size"),
            f[3].parse().expect("price"),
            f[4].parse().expect("fees"),
            side,
        );
    }
    out
}

/// Splits a `;`-joined expected column; an empty field means no records.
fn split<T>(text: &str, parse_value: impl Fn(&str) -> T) -> Vec<T> {
    if text.is_empty() {
        Vec::new()
    } else {
        text.split(';').map(parse_value).collect()
    }
}

fn parse_direction(s: &str) -> i8 {
    match s {
        "Long" => TradeDirection::Long as i8,
        "Short" => TradeDirection::Short as i8,
        other => panic!("unknown trade direction {other:?}"),
    }
}

fn parse_status(s: &str) -> i8 {
    match s {
        "Open" => TradeStatus::Open as i8,
        "Closed" => TradeStatus::Closed as i8,
        other => panic!("unknown trade status {other:?}"),
    }
}

/// Compares one derived view against the fixture's expected columns.
fn compare(
    case: &str,
    view: &str,
    got: &TradeRecords,
    g: impl Fn(&str) -> String,
    failures: &mut Vec<String>,
) {
    let want_col = split(&g("t_col"), |s| s.parse::<i64>().expect("col"));
    if got.len() != want_col.len() {
        failures.push(format!(
            "{case}/{view}: expected {} trades, got {}",
            want_col.len(),
            got.len()
        ));
        return;
    }

    // Integer fields are compared exactly. A tolerance on an index or a
    // status code would hide exactly the off-by-one and mis-attribution
    // errors this harness exists to find.
    let ints: [(&str, &Vec<i64>); 4] = [
        ("col", &got.col),
        ("entry_idx", &got.entry_idx),
        ("exit_idx", &got.exit_idx),
        ("parent_id", &got.parent_id),
    ];
    for (name, values) in ints {
        let want = split(&g(&format!("t_{name}")), |s| {
            s.parse::<i64>().unwrap_or_else(|_| panic!("{name}"))
        });
        for (i, (&a, &b)) in values.iter().zip(&want).enumerate() {
            if a != b {
                failures.push(format!("{case}/{view}[{i}].{name}: got {a}, want {b}"));
            }
        }
    }

    let floats: [(&str, &Vec<f64>); 7] = [
        ("size", &got.size),
        ("entry_price", &got.entry_price),
        ("entry_fees", &got.entry_fees),
        ("exit_price", &got.exit_price),
        ("exit_fees", &got.exit_fees),
        ("pnl", &got.pnl),
        ("return", &got.ret),
    ];
    for (name, values) in floats {
        let want = split(&g(&format!("t_{name}")), |s| {
            s.parse::<f64>().unwrap_or_else(|_| panic!("{name}"))
        });
        for (i, (&a, &b)) in values.iter().zip(&want).enumerate() {
            if !agree(a, b) {
                failures.push(format!("{case}/{view}[{i}].{name}: got {a}, want {b}"));
            }
        }
    }

    for (name, values, parse) in [
        (
            "direction",
            &got.direction,
            parse_direction as fn(&str) -> i8,
        ),
        ("status", &got.status, parse_status as fn(&str) -> i8),
    ] {
        let want = split(&g(&format!("t_{name}")), parse);
        for (i, (&a, &b)) in values.iter().zip(&want).enumerate() {
            if a != b {
                failures.push(format!("{case}/{view}[{i}].{name}: got {a}, want {b}"));
            }
        }
    }
}

#[test]
fn matches_the_golden_views() {
    let fixture = Fixture::load();
    assert!(
        fixture.rows.len() > 200,
        "expected the full fixture set, got {}",
        fixture.rows.len()
    );

    let mut failures: Vec<String> = Vec::new();
    let mut checked = 0usize;
    let mut errors = 0usize;

    for row in &fixture.rows {
        let g = |name: &str| fixture.get(row, name);
        let case = g("case");
        let view = g("view");
        let n_rows: usize = g("n_rows").parse().unwrap();
        let n_cols: usize = g("n_cols").parse().unwrap();
        let close: Vec<f64> = g("close").split(';').map(|s| s.parse().unwrap()).collect();
        let close = FlexArray::per_column(&close, n_cols).expect("close width");
        let orders = parse_orders(g("orders"));

        let derived = match view {
            "EntryTrades" => entry_trades(&orders, &close, n_rows, n_cols),
            "ExitTrades" => exit_trades(&orders, &close, n_rows, n_cols),
            "Positions" => {
                exit_trades(&orders, &close, n_rows, n_cols).and_then(|t| positions(&t, n_cols))
            }
            other => panic!("unknown view {other:?}"),
        };

        if g("outcome") == "error" {
            // The fixture records an error; so must we. Which of the two checks fires
            // is not pinned — the fixture records only that it failed.
            errors += 1;
            if derived.is_ok() {
                failures.push(format!("{case}/{view}: expected an error, got records"));
            }
            continue;
        }

        match derived {
            Ok(got) => {
                checked += 1;
                compare(case, view, &got, |n| g(n).to_owned(), &mut failures);
            }
            Err(e) => failures.push(format!("{case}/{view}: unexpected error: {e}")),
        }
    }

    assert!(errors > 0, "fixture set has no error cases");
    assert!(
        failures.is_empty(),
        "{} of {checked} derivations disagree with the golden outputs:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Positions derived from either view must agree, on every fixture.
///
/// Positions are built from exit trades. The claim is that
/// entry trades give the same answer, and that claim is load-bearing for
/// anyone who reaches for the other view — so it is checked rather than
/// taken on faith.
#[test]
fn positions_agree_whichever_view_they_come_from() {
    let fixture = Fixture::load();
    let mut failures: Vec<String> = Vec::new();
    let mut checked = 0usize;

    for row in &fixture.rows {
        let g = |name: &str| fixture.get(row, name);
        if g("view") != "Positions" || g("outcome") != "ok" {
            continue;
        }
        let n_rows: usize = g("n_rows").parse().unwrap();
        let n_cols: usize = g("n_cols").parse().unwrap();
        let close: Vec<f64> = g("close").split(';').map(|s| s.parse().unwrap()).collect();
        let close = FlexArray::per_column(&close, n_cols).expect("close width");
        let orders = parse_orders(g("orders"));

        let from_exit = positions(
            &exit_trades(&orders, &close, n_rows, n_cols).expect("exit trades"),
            n_cols,
        )
        .expect("positions from exit trades");
        let from_entry = positions(
            &entry_trades(&orders, &close, n_rows, n_cols).expect("entry trades"),
            n_cols,
        )
        .expect("positions from entry trades");

        checked += 1;
        let case = g("case");
        if from_exit.len() != from_entry.len() {
            failures.push(format!(
                "{case}: {} positions from exits, {} from entries",
                from_exit.len(),
                from_entry.len()
            ));
            continue;
        }
        for i in 0..from_exit.len() {
            if !agree(from_exit.size[i], from_entry.size[i])
                || !agree(from_exit.pnl[i], from_entry.pnl[i])
                || from_exit.status[i] != from_entry.status[i]
                || from_exit.direction[i] != from_entry.direction[i]
            {
                failures.push(format!(
                    "{case}[{i}]: exits give size {} pnl {}, entries give size {} pnl {}",
                    from_exit.size[i], from_exit.pnl[i], from_entry.size[i], from_entry.pnl[i]
                ));
            }
        }
    }

    assert!(checked > 50, "only {checked} cases exercised");
    assert!(
        failures.is_empty(),
        "{} disagreements between the two routes to a position:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Total PnL is the one thing all three views must agree on.
///
/// They differ on how many trades there were and what each returned; if they
/// also differed on the total, one of them would be losing or inventing
/// money.
#[test]
fn the_views_agree_on_total_pnl() {
    let fixture = Fixture::load();
    let mut failures: Vec<String> = Vec::new();

    for row in &fixture.rows {
        let g = |name: &str| fixture.get(row, name);
        if g("view") != "Positions" || g("outcome") != "ok" {
            continue;
        }
        let n_rows: usize = g("n_rows").parse().unwrap();
        let n_cols: usize = g("n_cols").parse().unwrap();
        let close: Vec<f64> = g("close").split(';').map(|s| s.parse().unwrap()).collect();
        let close = FlexArray::per_column(&close, n_cols).expect("close width");
        let orders = parse_orders(g("orders"));

        let sum = |t: &TradeRecords| t.pnl.iter().sum::<f64>();
        let entry = sum(&entry_trades(&orders, &close, n_rows, n_cols).expect("entry"));
        let exit = sum(&exit_trades(&orders, &close, n_rows, n_cols).expect("exit"));
        let pos = sum(&positions(
            &exit_trades(&orders, &close, n_rows, n_cols).expect("exit"),
            n_cols,
        )
        .expect("positions"));

        if !agree(entry, exit) || !agree(exit, pos) {
            failures.push(format!(
                "{}: entry {entry}, exit {exit}, positions {pos}",
                g("case")
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "{} cases where the views disagree on total PnL:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
