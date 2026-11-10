//! Golden test for the statistics table.
//!
//! The widest harness in the project: one row runs a simulation, derives the
//! value curve and returns from its records, derives trades and drawdowns, and
//! reduces all of it to twenty-eight labelled rows. A failure here is
//! diagnosed by the narrower harnesses — which is why they were built first.
//!
//! It is also the only one whose expected outputs are the *user-facing* table
//! rather than a kernel's, because the row titles, the row order, the percent
//! scalings and the open-trade default are not visible from the kernel tier.
//! An implementation can match every kernel and still print a table nobody can
//! line up against the documented one.
//!
//! The expected outputs are committed in `tests/fixtures/stats.tsv`, so this
//! needs no Python.

use std::collections::HashMap;
use std::path::PathBuf;

use alphax_analysis::metrics::DAILY;
use alphax_analysis::stats::{Select, Value};
use alphax_analysis::{Portfolio, Scoring};
use alphax_core::flex::FlexArray;
use alphax_model::{Direction, SizeType};
use alphax_sim::from_orders::FromOrders;
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

/// Agreement, with `NaN` and the infinities exact.
fn agree(a: f64, b: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return a.is_nan() && b.is_nan();
    }
    if a.is_infinite() || b.is_infinite() {
        #[allow(clippy::float_cmp, reason = "infinities agree exactly or not at all")]
        let same = a == b;
        return same;
    }
    (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
}

struct Fixture {
    header: HashMap<String, usize>,
    rows: Vec<Vec<String>>,
}

impl Fixture {
    fn load() -> Self {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/stats.tsv");
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

#[test]
fn matches_the_golden_stats_table() {
    let fixture = Fixture::load();
    assert!(
        fixture.rows.len() >= 72,
        "expected the full fixture set, got {}",
        fixture.rows.len()
    );

    let mut failures: Vec<String> = Vec::new();
    let mut cases_seen: std::collections::HashSet<&str> = std::collections::HashSet::new();

    for (n, row) in fixture.rows.iter().enumerate() {
        let g = |name: &str| fixture.get(row, name);
        cases_seen.insert(g("case"));

        let n_rows: usize = g("n_rows").parse().unwrap();
        let n_cols: usize = g("n_cols").parse().unwrap();
        let col: usize = g("col").parse().unwrap();
        let init_cash: f64 = g("init_cash").parse().unwrap();
        let close_vals = parse_floats(g("close"));
        let size_vals = parse_floats(g("size"));
        let incl_open = g("incl_open") == "True";

        // Run the whole stack, exactly as a caller would.
        let close = FlexArray::full(&close_vals, n_rows, n_cols).unwrap();
        let size = FlexArray::full(&size_vals, n_rows, n_cols).unwrap();
        let cash = FlexArray::scalar(init_cash);
        let args = FromOrders {
            n_rows,
            n_cols,
            init_cash: cash,
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
            close,
            size,
            price: close,
            size_type: FlexArray::scalar(SizeType::Amount),
            direction: FlexArray::scalar(Direction::Both),
            fees: FlexArray::scalar(g("fees").parse::<f64>().unwrap()),
            fixed_fees: FlexArray::scalar(g("fixed_fees").parse::<f64>().unwrap()),
            slippage: FlexArray::scalar(g("slippage").parse::<f64>().unwrap()),
            min_size: FlexArray::scalar(0.0),
            max_size: FlexArray::scalar(f64::INFINITY),
            size_granularity: FlexArray::scalar(f64::NAN),
            lock_cash: FlexArray::scalar(false),
            allow_partial: FlexArray::scalar(true),
            raise_reject: FlexArray::scalar(false),
            log: FlexArray::scalar(false),
            // `-inf` resolves to the order price, which is the close here.
            val_price: FlexArray::scalar(f64::NEG_INFINITY),
            // No cash flows outside fills.
            cash_deposits: FlexArray::scalar(0.0),
            cash_earnings: FlexArray::scalar(0.0),
            cash_dividends: FlexArray::scalar(0.0),
            // None of these is in play here.
            leverage: FlexArray::scalar(1.0),
            leverage_mode: FlexArray::scalar(alphax_model::LeverageMode::Lazy),
            account_type: FlexArray::scalar(alphax_model::AccountType::Cash),
            initial_margin: FlexArray::scalar(f64::NAN),
            margin_basis: FlexArray::scalar(alphax_model::MarginBasis::Notional),
            multiplier: FlexArray::scalar(1.0),
            price_type: FlexArray::scalar(alphax_model::PriceType::Price),
            from_ago: FlexArray::Scalar(0),
            bid_price: FlexArray::scalar(f64::NAN),
            ask_price: FlexArray::scalar(f64::NAN),
            price_area_vio_mode: FlexArray::scalar(alphax_model::PriceAreaVioMode::Ignore),
            force_exit: FlexArray::scalar(false),
            cash_limit: FlexArray::scalar(f64::INFINITY),
            reject_prob: FlexArray::scalar(0.0),
            seed: 0,
            funding_rate: FlexArray::scalar(0.0),
            long_financing_rate: FlexArray::scalar(0.0),
            short_financing_rate: FlexArray::scalar(0.0),
            debt_rate: FlexArray::scalar(0.0),
            borrow_rate: FlexArray::scalar(0.0),
            mark_price: FlexArray::scalar(f64::NAN),
            maintenance_margin: FlexArray::scalar(0.0),
            liquidation_fee: FlexArray::scalar(0.0),
            margin_mode: FlexArray::scalar(alphax_model::MarginMode::Cross),
            open: FlexArray::scalar(f64::NAN),
            high: FlexArray::scalar(f64::NAN),
            low: FlexArray::scalar(f64::NAN),
            ffill_val_price: true,
            update_value: false,
            groups: Groups::ungrouped(n_cols),
            call_seq: CallSeq::Default,
        };
        // Through `Portfolio`, so the expected tables test the type's own
        // assembly of the rebuild chain — which is the part a caller used to
        // have to get right by hand.
        let pf = Portfolio::from_orders(&args, 1)
            .expect("the fixture workload is valid")
            .with_scoring(Scoring {
                ann_factor: DAILY,
                risk_free: 0.0,
                trades_type: alphax_model::TradesType::ExitTrades,
                incl_open,
            });
        let benchmark = pf.buy_and_hold();
        let pf = pf.with_benchmark(benchmark).expect("shapes match");

        let got = pf.stats(col, &Select::all()).expect("the column exists");
        let want_titles: Vec<&str> = g("titles").split('|').collect();
        let want_values = parse_floats(g("values"));
        assert_eq!(want_titles.len(), want_values.len());

        if got.len() != want_titles.len() {
            failures.push(format!(
                "row {n} {}: table has {} rows, expected {}",
                g("case"),
                got.len(),
                want_titles.len()
            ));
            continue;
        }

        for (k, ((title, value), (wt, wv))) in got
            .iter()
            .zip(want_titles.iter().zip(&want_values))
            .enumerate()
        {
            if title != wt {
                failures.push(format!(
                    "row {n} {}: row {k} is {title:?}, expected {wt:?}",
                    g("case")
                ));
                continue;
            }
            let as_f64 = match value {
                Value::Count(c) => *c as f64,
                Value::Float(v) => *v,
                Value::Missing => f64::NAN,
            };
            if !agree(as_f64, *wv) {
                failures.push(format!(
                    "row {n} {} [{}] incl_open={incl_open}: {title} is {as_f64}, expected {wv}",
                    g("case"),
                    g("note")
                ));
            }
        }
    }

    for case in [
        "closed out",
        "still holding",
        "never traded",
        "all losses",
        "all wins",
        "shorting",
        "flat price",
        "monotone up",
    ] {
        assert!(cases_seen.contains(case), "no fixtures for {case:?}");
    }
    assert!(
        failures.is_empty(),
        "{} of {} tables disagree with the golden outputs:\n{}",
        failures.len(),
        fixture.rows.len(),
        failures
            .iter()
            .take(30)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// The fixtures still cover both settings of the open-trade default.
///
/// That default is the one thing in this table a reader is most likely to get
/// wrong, and a fixture set that only exercised one side of it would pass
/// against an implementation that ignored the flag entirely.
#[test]
fn the_fixtures_still_exercise_both_open_trade_settings() {
    let fixture = Fixture::load();
    let mut with = 0_usize;
    let mut without = 0_usize;
    let mut differing = 0_usize;
    let mut by_case: HashMap<(&str, &str, &str, &str), Vec<&str>> = HashMap::new();
    for row in &fixture.rows {
        let g = |name: &str| fixture.get(row, name);
        if g("incl_open") == "True" {
            with += 1;
        } else {
            without += 1;
        }
        by_case
            .entry((g("case"), g("fees"), g("fixed_fees"), g("slippage")))
            .or_default()
            .push(g("values"));
    }
    for values in by_case.values() {
        if values.len() == 2 && values[0] != values[1] {
            differing += 1;
        }
    }
    assert!(with > 0 && without > 0, "only one setting is covered");
    assert!(
        differing > 0,
        "no case where the open-trade setting changes the table, so an implementation \
         that ignored the flag would pass"
    );
}
