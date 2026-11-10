//! Golden test for cash sharing and the call order.
//!
//! This is the one feature that breaks the assumption the rest of the engine
//! is built on. Everywhere else a column is a self-contained simulation; in a
//! shared group each fill changes what the next column can afford, so the
//! order columns execute in is part of the result rather than an
//! implementation detail.
//!
//! Every case is replayed both ways — layout order and sorted by order value —
//! and a separate test asserts that at least one case actually differs between
//! them. Without that, an implementation that ignored the sort entirely would pass.
//!
//! The expected outputs are committed in `tests/fixtures/cash_sharing.tsv`, so this
//! needs no Python.

use std::collections::HashMap;
use std::path::PathBuf;

use alphax_core::flex::FlexArray;
use alphax_model::{Direction, SizeType};
use alphax_records::OrderRecords;
use alphax_sim::from_orders::{FromOrders, simulate_from_orders, simulate_from_orders_parallel};
use alphax_sim::groups::{CallSeq, Groups};

fn parse_floats(s: &str) -> Vec<f64> {
    if s.is_empty() {
        Vec::new()
    } else {
        s.split(';')
            .map(|v| match v {
                "nan" => f64::NAN,
                "inf" => f64::INFINITY,
                "-inf" => f64::NEG_INFINITY,
                other => other.parse().expect("float"),
            })
            .collect()
    }
}

fn parse_ints(s: &str) -> Vec<i64> {
    if s.is_empty() {
        Vec::new()
    } else {
        s.split(';').map(|v| v.parse().expect("int")).collect()
    }
}

fn size_type(code: i64) -> SizeType {
    match code {
        0 => SizeType::Amount,
        1 => SizeType::Value,
        2 => SizeType::Percent,
        3 => SizeType::TargetAmount,
        4 => SizeType::TargetValue,
        5 => SizeType::TargetPercent,
        other => panic!("unknown size type {other}"),
    }
}

fn direction(code: i64) -> Direction {
    match code {
        0 => Direction::LongOnly,
        1 => Direction::ShortOnly,
        2 => Direction::Both,
        other => panic!("unknown direction {other}"),
    }
}

fn agree(a: f64, b: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return a.is_nan() && b.is_nan();
    }
    (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
}

struct Fixture {
    header: HashMap<String, usize>,
    rows: Vec<Vec<String>>,
}

impl Fixture {
    fn load() -> Self {
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/cash_sharing.tsv");
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
                let mut fields: Vec<String> = line.split('\t').map(str::to_owned).collect();
                while fields.len() < names.len() {
                    fields.push(String::new());
                }
                assert_eq!(fields.len(), names.len(), "row wider than the header");
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

/// Everything one fixture row needs to run, with the borrows kept alive.
struct Workload {
    n_rows: usize,
    n_cols: usize,
    group_lens: Vec<usize>,
    per_col_cash: Vec<f64>,
    close: Vec<f64>,
    size: Vec<f64>,
    size_type: SizeType,
    direction: Direction,
    fees: f64,
    auto: bool,
    update_value: bool,
}

impl Workload {
    fn from_row(f: &Fixture, row: &[String]) -> Self {
        let g = |name: &str| f.get(row, name);
        let n_rows: usize = g("n_rows").parse().unwrap();
        let n_cols: usize = g("n_cols").parse().unwrap();
        let group_lens: Vec<usize> = parse_ints(g("group_lens"))
            .into_iter()
            .map(|v| v as usize)
            .collect();
        // The fixture carries one starting balance per *group*; this engine
        // reads it from the group's first column, so it is expanded here.
        let group_cash = parse_floats(g("init_cash"));
        let mut per_col_cash = vec![f64::NAN; n_cols];
        let mut start = 0;
        for (gi, &len) in group_lens.iter().enumerate() {
            per_col_cash[start..start + len].fill(group_cash[gi]);
            start += len;
        }
        Self {
            n_rows,
            n_cols,
            group_lens,
            per_col_cash,
            close: parse_floats(g("close")),
            size: parse_floats(g("size")),
            size_type: size_type(g("size_type").parse().unwrap()),
            direction: direction(g("direction").parse().unwrap()),
            fees: g("fees").parse().unwrap(),
            auto: g("auto_call_seq") == "True",
            update_value: g("update_value") == "True",
        }
    }

    fn args(&self) -> FromOrders<'_> {
        let close = FlexArray::full(&self.close, self.n_rows, self.n_cols).unwrap();
        FromOrders {
            n_rows: self.n_rows,
            n_cols: self.n_cols,
            groups: Groups::from_lens(&self.group_lens, self.n_cols).unwrap(),
            call_seq: if self.auto {
                CallSeq::Auto
            } else {
                CallSeq::Default
            },
            init_cash: FlexArray::per_column(&self.per_col_cash, self.n_cols).unwrap(),
            close,
            size: FlexArray::full(&self.size, self.n_rows, self.n_cols).unwrap(),
            // The fixture's order price is `close`.
            price: close,
            size_type: FlexArray::Scalar(self.size_type),
            direction: FlexArray::Scalar(self.direction),
            fees: FlexArray::Scalar(self.fees),
            update_value: self.update_value,
            ..FromOrders::default()
        }
    }
}

/// Records sorted by column then row, which is how the fixture stores them.
fn sorted_by_column(r: &OrderRecords) -> Vec<usize> {
    let mut idxs: Vec<usize> = (0..r.len()).collect();
    idxs.sort_by_key(|&i| (r.col[i], r.idx[i]));
    idxs
}

#[test]
fn matches_the_golden_with_cash_sharing() {
    let fixture = Fixture::load();
    assert!(
        fixture.rows.len() >= 20,
        "expected the full fixture set, got {}",
        fixture.rows.len()
    );

    let mut failures: Vec<String> = Vec::new();

    for (n, row) in fixture.rows.iter().enumerate() {
        let g = |name: &str| fixture.get(row, name);
        let w = Workload::from_row(&fixture, row);
        let args = w.args();
        let records = match simulate_from_orders(&args) {
            Ok(r) => r,
            Err(e) => {
                failures.push(format!("row {n} {}: simulation failed: {e}", g("case")));
                continue;
            }
        };

        let order = sorted_by_column(&records);
        let want_col = parse_ints(g("fill_col"));
        let want_idx = parse_ints(g("fill_idx"));
        let want_size = parse_floats(g("fill_size"));
        let want_price = parse_floats(g("fill_price"));
        let want_fees = parse_floats(g("fill_fees"));
        let want_side = parse_ints(g("fill_side"));

        let label = format!("row {n} {} auto={} [{}]", g("case"), w.auto, g("note"));
        if order.len() != want_col.len() {
            failures.push(format!(
                "{label}: {} fills, expected {}",
                order.len(),
                want_col.len()
            ));
            continue;
        }
        for (k, &i) in order.iter().enumerate() {
            // Integer fields must match exactly; a differing column or bar is
            // a different trade, not a rounding difference.
            if records.col[i] != want_col[k]
                || records.idx[i] != want_idx[k]
                || records.side[i] as i64 != want_side[k]
            {
                failures.push(format!(
                    "{label}: fill {k} is col {} row {} side {}, expected col {} row {} side {}",
                    records.col[i],
                    records.idx[i],
                    records.side[i],
                    want_col[k],
                    want_idx[k],
                    want_side[k]
                ));
                continue;
            }
            if !agree(records.size[i], want_size[k])
                || !agree(records.price[i], want_price[k])
                || !agree(records.fees[i], want_fees[k])
            {
                failures.push(format!(
                    "{label}: fill {k} is {}@{} fee {}, expected {}@{} fee {}",
                    records.size[i],
                    records.price[i],
                    records.fees[i],
                    want_size[k],
                    want_price[k],
                    want_fees[k]
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} rows disagree with the golden outputs:\n{}",
        failures.len(),
        fixture.rows.len(),
        failures
            .iter()
            .take(25)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// The call order is load-bearing, and the fixtures prove it.
///
/// Without this, an implementation that ignored `CallSeq::Auto` entirely would pass every
/// row above — the two settings would simply produce the same records and the
/// harness would confirm both.
#[test]
fn sorting_the_call_order_changes_the_fills() {
    let fixture = Fixture::load();
    let mut by_case: HashMap<&str, Vec<&Vec<String>>> = HashMap::new();
    for row in &fixture.rows {
        by_case
            .entry(fixture.get(row, "case"))
            .or_default()
            .push(row);
    }

    let mut differing = 0_usize;
    for rows in by_case.values() {
        assert_eq!(rows.len(), 2, "each case should be generated both ways");
        if fixture.get(rows[0], "fill_size") != fixture.get(rows[1], "fill_size") {
            differing += 1;
        }
    }
    assert!(
        differing > 0,
        "no case's fills change with the call order, so an implementation that ignored \
         the sort would pass the whole harness"
    );
}

/// Groups of one reproduce the ungrouped simulation exactly.
///
/// The regression that matters most here: cash sharing is new machinery on the
/// path every existing run takes, and a group of one has to come out
/// bit-identical to no grouping at all.
#[test]
fn a_group_of_one_is_an_ungrouped_column() {
    let fixture = Fixture::load();
    let mut checked = 0_usize;
    for row in &fixture.rows {
        let w = Workload::from_row(&fixture, row);
        if w.group_lens.iter().any(|&l| l != 1) {
            continue;
        }
        let grouped = simulate_from_orders(&w.args()).unwrap();
        let ungrouped = simulate_from_orders(&FromOrders {
            groups: Groups::ungrouped(w.n_cols),
            call_seq: CallSeq::Default,
            ..w.args()
        })
        .unwrap();
        assert_eq!(grouped, ungrouped, "a group of one changed the records");
        checked += 1;
    }
    assert!(checked > 0, "no all-singleton cases in the fixtures");
}

/// Parallelism splits by group, and still produces identical records.
///
/// The property the whole parallel design rests on, restated for the grouped
/// case: the unit of work changed from a column to a group, and nothing about
/// determinism did.
#[test]
fn splitting_by_group_does_not_change_the_records() {
    let fixture = Fixture::load();
    let mut with_several_groups = 0_usize;
    for row in &fixture.rows {
        let w = Workload::from_row(&fixture, row);
        let serial = simulate_from_orders(&w.args()).unwrap();
        for threads in [2, 3, 8] {
            let parallel = simulate_from_orders_parallel(&w.args(), threads).unwrap();
            assert_eq!(
                serial,
                parallel,
                "thread count {threads} changed the records for {}",
                fixture.get(row, "case")
            );
        }
        if w.group_lens.len() > 1 {
            with_several_groups += 1;
        }
    }
    assert!(
        with_several_groups > 0,
        "every case is a single group, so the split is never exercised"
    );
}

/// A grouping that does not cover the shape is rejected.
#[test]
fn a_mismatched_grouping_is_rejected() {
    let args = FromOrders {
        n_rows: 2,
        n_cols: 3,
        groups: Groups::from_lens(&[2], 2).unwrap(),
        close: FlexArray::Scalar(10.0),
        ..FromOrders::default()
    };
    assert!(simulate_from_orders(&args).is_err());
    assert!(simulate_from_orders_parallel(&args, 4).is_err());
}
