//! End-to-end golden test: simulation through to metrics.
//!
//! The widest harness in the project. One case runs `from_orders`, derives
//! the value curve from the records it produced, and reduces that to a dozen
//! scalars — so a single row exercises the execution kernel, the loop, the
//! record layer, the rebuild chain, the trade views and the metrics at once.
//!
//! That width is why it is the *last* harness rather than the first: a
//! failure here is diagnosed by the narrower ones. On its own it can only say
//! that something in the pipeline drifted.
//!
//! The series are compared as well as the scalars. A Sharpe ratio that agrees
//! over a value curve that does not is a coincidence, not a result — the
//! same reason the golden tests compare record arrays rather than
//! aggregates.
//!
//! The expected outputs are committed in `tests/fixtures/analysis.tsv`, so this
//! needs no Python.

use std::collections::HashMap;
use std::path::PathBuf;

use alphax_analysis::metrics::{
    DAILY, annualized_return, annualized_volatility, calmar_ratio, downside_risk, max_drawdown,
    sharpe_ratio, sortino_ratio, total_return,
};
use alphax_analysis::series::{asset_value, assets, cash, cash_flow, returns, value};
use alphax_analysis::trade_stats;
use alphax_core::flex::FlexArray;
use alphax_core::math::is_close;
use alphax_core::matrix::Matrix;
use alphax_model::{Direction, SizeType};
use alphax_records::exit_trades;
use alphax_sim::groups::Groups;
use alphax_sim::{FromOrders, simulate_from_orders};

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
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/analysis.tsv");
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

fn floats(text: &str) -> Vec<f64> {
    if text.is_empty() {
        return Vec::new();
    }
    text.split(';')
        .map(|s| s.parse().unwrap_or_else(|_| panic!("cannot parse {s:?}")))
        .collect()
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

/// Compares a full matrix against its column-major expected values.
fn compare_matrix(case: &str, name: &str, got: &Matrix, want: &[f64], failures: &mut Vec<String>) {
    if got.as_slice().len() != want.len() {
        failures.push(format!(
            "{case}.{name}: expected {} values, got {}",
            want.len(),
            got.as_slice().len()
        ));
        return;
    }
    for (i, (&a, &b)) in got.as_slice().iter().zip(want).enumerate() {
        if !agree(a, b) {
            let (row, col) = (i % got.n_rows(), i / got.n_rows());
            failures.push(format!(
                "{case}.{name}[row {row}, col {col}]: got {a}, want {b}"
            ));
            return; // one report per series is enough to localize it
        }
    }
}

fn compare_scalars(case: &str, name: &str, got: &[f64], want: &[f64], failures: &mut Vec<String>) {
    if got.len() != want.len() {
        failures.push(format!(
            "{case}.{name}: expected {} values, got {}",
            want.len(),
            got.len()
        ));
        return;
    }
    for (col, (&a, &b)) in got.iter().zip(want).enumerate() {
        if !agree(a, b) {
            failures.push(format!("{case}.{name}[col {col}]: got {a}, want {b}"));
        }
    }
}

#[test]
fn matches_the_golden_end_to_end() {
    let fixture = Fixture::load();
    assert!(
        fixture.rows.len() > 30,
        "expected the full fixture set, got {}",
        fixture.rows.len()
    );

    let mut failures: Vec<String> = Vec::new();

    for row in &fixture.rows {
        let g = |name: &str| fixture.get(row, name);
        let case = g("case");
        let n_rows: usize = g("n_rows").parse().unwrap();
        let n_cols: usize = g("n_cols").parse().unwrap();

        let close_values = floats(g("close"));
        let size_values = floats(g("size"));
        let close = FlexArray::full(&close_values, n_rows, n_cols).expect("close shape");
        let size = FlexArray::full(&size_values, n_rows, n_cols).expect("size shape");
        let init_cash_value: f64 = g("init_cash").parse().unwrap();
        let init_cash = FlexArray::scalar(init_cash_value);

        let args = FromOrders {
            n_rows,
            n_cols,
            init_cash: FlexArray::scalar(init_cash_value),
            close,
            size,
            // The fixture's order price is `close`, so `+inf`
            // resolves to exactly the same series.
            price: FlexArray::Scalar(f64::INFINITY),
            size_type: FlexArray::scalar(parse_size_type(g("size_type"))),
            direction: FlexArray::scalar(parse_direction(g("direction"))),
            fees: FlexArray::scalar(g("fees").parse().unwrap()),
            fixed_fees: FlexArray::scalar(g("fixed_fees").parse().unwrap()),
            slippage: FlexArray::scalar(g("slippage").parse().unwrap()),
            groups: Groups::ungrouped(n_cols),
            ..FromOrders::default()
        };

        let orders = match simulate_from_orders(&args) {
            Ok(o) => o,
            Err(e) => {
                failures.push(format!("{case}: simulation failed: {e}"));
                continue;
            }
        };

        let held = assets(&orders, n_rows, n_cols);
        let curve = value(
            &cash(&cash_flow(&orders, n_rows, n_cols), &init_cash),
            &asset_value(&held, &close, None),
        );
        let rets = returns(&curve, &init_cash);

        compare_matrix(case, "value", &curve, &floats(g("value")), &mut failures);
        compare_matrix(case, "returns", &rets, &floats(g("returns")), &mut failures);

        let scalar_metrics: [(&str, Vec<f64>); 8] = [
            ("total_return", rets.map_columns(total_return)),
            (
                "annualized_return",
                rets.map_columns(|c| annualized_return(c, DAILY)),
            ),
            (
                "annualized_volatility",
                rets.map_columns(|c| annualized_volatility(c, DAILY)),
            ),
            ("max_drawdown", rets.map_columns(max_drawdown)),
            (
                "sharpe_ratio",
                rets.map_columns(|c| sharpe_ratio(c, DAILY, 0.0)),
            ),
            (
                "sortino_ratio",
                rets.map_columns(|c| sortino_ratio(c, DAILY, 0.0)),
            ),
            ("calmar_ratio", rets.map_columns(|c| calmar_ratio(c, DAILY))),
            (
                "downside_risk",
                rets.map_columns(|c| downside_risk(c, DAILY, 0.0)),
            ),
        ];
        for (name, got) in scalar_metrics {
            compare_scalars(case, name, &got, &floats(g(name)), &mut failures);
        }

        let trades = exit_trades(&orders, &close, n_rows, n_cols).expect("exit trades");
        let stats = trade_stats::per_column(&trades, n_cols).expect("trade stats");
        let pick =
            |f: fn(&trade_stats::TradeStats) -> f64| -> Vec<f64> { stats.iter().map(f).collect() };
        compare_scalars(
            case,
            "trade_count",
            &stats.iter().map(|s| s.count as f64).collect::<Vec<_>>(),
            &floats(g("trade_count")),
            &mut failures,
        );
        for (name, got) in [
            ("win_rate", pick(|s| s.win_rate)),
            ("profit_factor", pick(|s| s.profit_factor)),
            ("expectancy", pick(|s| s.expectancy)),
            ("total_pnl", pick(|s| s.total_pnl)),
        ] {
            compare_scalars(case, name, &got, &floats(g(name)), &mut failures);
        }
    }

    assert!(
        failures.is_empty(),
        "{} disagreements with the golden outputs:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The fixture set must keep covering the degenerate cases.
///
/// Every metric here has a branch for "nothing happened", and those branches
/// are where an implementation silently drifts — an untraded column that reads 0.0
/// instead of `NaN` sorts as mediocre rather than absent. A fixture set that
/// drifts to only well-behaved cases would still pass the test above.
#[test]
fn the_fixtures_still_cover_the_degenerate_cases() {
    let fixture = Fixture::load();
    let mut saw_infinite_sharpe = false;
    let mut saw_nan_calmar = false;
    let mut saw_untraded_column = false;
    let mut saw_zero_drawdown = false;

    for row in &fixture.rows {
        let g = |name: &str| fixture.get(row, name);
        saw_infinite_sharpe |= floats(g("sharpe_ratio")).iter().any(|v| v.is_infinite());
        saw_nan_calmar |= floats(g("calmar_ratio")).iter().any(|v| v.is_nan());
        saw_untraded_column |= floats(g("trade_count")).contains(&0.0);
        saw_zero_drawdown |= floats(g("max_drawdown")).contains(&0.0);
    }

    assert!(saw_infinite_sharpe, "no column with zero volatility");
    assert!(saw_nan_calmar, "no column without a drawdown");
    assert!(saw_untraded_column, "no column that never traded");
    assert!(saw_zero_drawdown, "no column whose curve only rose");
}
