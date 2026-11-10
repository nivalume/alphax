//! Golden test for the eight indicators.
//!
//! The array layer underneath is pinned function by function in
//! `alphax-generic`, so what this verifies is the composition: which window
//! feeds which average, and the readings that are easy to special-case
//! wrongly.
//!
//! Those degenerate readings are most of the value. An RSI of exactly 100
//! arrives through a division by zero rather than a branch; a stochastic on a
//! flat range divides zero by zero; the first bar's true range has no previous
//! close to reach for; on-balance volume's first bar compares against a `NaN`
//! and therefore counts positive, which follows from `NaN` comparing false and
//! is not a rule anyone wrote down.
//!
//! Each fixture holds one parameter combination, so a disagreement names the
//! combination rather than a column index inside a sweep. A separate test
//! covers the sweep layout itself.
//!
//! The expected outputs are committed in `tests/fixtures/indicators.tsv`, so this
//! needs no Python.

use std::collections::HashMap;
use std::path::PathBuf;

use alphax_core::math::is_close;
use alphax_core::matrix::Matrix;
use alphax_indicators::{
    BbParams, Indicator, MaParams, MacdParams, StochParams, atr, bbands, ma, macd, mstd, obv, rsi,
    stoch,
};

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

fn numbers(s: &str) -> Vec<f64> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(';')
        .map(|v| match v {
            "inf" => f64::INFINITY,
            "-inf" => f64::NEG_INFINITY,
            "nan" => f64::NAN,
            other => other
                .parse()
                .unwrap_or_else(|_| panic!("cannot parse {other:?} as f64")),
        })
        .collect()
}

struct Fixture {
    header: HashMap<String, usize>,
    rows: Vec<Vec<String>>,
}

impl Fixture {
    fn load() -> Self {
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/indicators.tsv");
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

/// Parses `a=1,b=true` into a lookup.
fn params(text: &str) -> HashMap<&str, &str> {
    if text.is_empty() {
        return HashMap::new();
    }
    text.split(',')
        .map(|kv| {
            let (k, v) = kv
                .split_once('=')
                .unwrap_or_else(|| panic!("bad param {kv:?}"));
            (k, v)
        })
        .collect()
}

fn usize_param(p: &HashMap<&str, &str>, name: &str) -> usize {
    p[name].parse().unwrap_or_else(|_| panic!("bad {name}"))
}

fn bool_param(p: &HashMap<&str, &str>, name: &str) -> bool {
    // The generator writes Python booleans.
    p[name] == "True"
}

fn f64_param(p: &HashMap<&str, &str>, name: &str) -> f64 {
    p[name].parse().unwrap_or_else(|_| panic!("bad {name}"))
}

#[test]
fn matches_the_golden_indicators() {
    let fixture = Fixture::load();
    assert!(
        fixture.rows.len() > 400,
        "expected the full fixture set, got {}",
        fixture.rows.len()
    );

    let mut failures: Vec<String> = Vec::new();
    let mut seen: HashMap<String, usize> = HashMap::new();

    for row in &fixture.rows {
        let g = |name: &str| fixture.get(row, name);
        let name = g("indicator");
        let case = g("case");
        *seen.entry(name.to_owned()).or_default() += 1;

        let n_rows: usize = g("n_rows").parse().unwrap();
        let n_cols: usize = g("n_cols").parse().unwrap();
        let build = |field: &str| {
            Matrix::from_column_major(numbers(g(field)), n_rows, n_cols)
                .unwrap_or_else(|e| panic!("{case}: bad {field}: {e}"))
        };
        let (high, low, close, volume) =
            (build("high"), build("low"), build("close"), build("volume"));

        let p = params(g("params"));
        let computed: Result<Indicator, String> = match name {
            "ma" => {
                let mp = [MaParams {
                    window: usize_param(&p, "window"),
                    ewm: bool_param(&p, "ewm"),
                }];
                ma(&close, &mp, false, None).map_err(|e| e.to_string())
            }
            "mstd" => {
                let mp = [MaParams {
                    window: usize_param(&p, "window"),
                    ewm: bool_param(&p, "ewm"),
                }];
                mstd(&close, &mp, false, 0, None).map_err(|e| e.to_string())
            }
            "rsi" => {
                let mp = [MaParams {
                    window: usize_param(&p, "window"),
                    ewm: bool_param(&p, "ewm"),
                }];
                rsi(&close, &mp, false, None).map_err(|e| e.to_string())
            }
            "atr" => {
                let mp = [MaParams {
                    window: usize_param(&p, "window"),
                    ewm: bool_param(&p, "ewm"),
                }];
                atr(&high, &low, &close, &mp, false, None).map_err(|e| e.to_string())
            }
            "bbands" => {
                let bp = [BbParams {
                    window: usize_param(&p, "window"),
                    ewm: bool_param(&p, "ewm"),
                    alpha: f64_param(&p, "alpha"),
                }];
                bbands(&close, &bp, false, 0, None).map_err(|e| e.to_string())
            }
            "stoch" => {
                let sp = [StochParams {
                    k_window: usize_param(&p, "k_window"),
                    d_window: usize_param(&p, "d_window"),
                    d_ewm: bool_param(&p, "d_ewm"),
                }];
                stoch(&high, &low, &close, &sp, false, None).map_err(|e| e.to_string())
            }
            "macd" => {
                let mp = [MacdParams {
                    fast_window: usize_param(&p, "fast_window"),
                    slow_window: usize_param(&p, "slow_window"),
                    signal_window: usize_param(&p, "signal_window"),
                    macd_ewm: bool_param(&p, "macd_ewm"),
                    signal_ewm: bool_param(&p, "signal_ewm"),
                }];
                macd(&close, &mp, false, None).map_err(|e| e.to_string())
            }
            "obv" => obv(&close, &volume, None).map_err(|e| e.to_string()),
            other => panic!("unknown indicator {other:?}"),
        };

        let indicator = match computed {
            Ok(v) => v,
            Err(e) => {
                failures.push(format!("{name}/{case} {}: rejected: {e}", g("params")));
                continue;
            }
        };

        let want_outputs: Vec<&str> = g("outputs").split(',').collect();
        let want_values: Vec<Vec<f64>> = g("result").split('|').map(numbers).collect();
        assert_eq!(want_outputs.len(), want_values.len(), "malformed fixture");

        for (out_name, want) in want_outputs.iter().zip(&want_values) {
            let got = match indicator.get(out_name) {
                Ok(m) => m,
                Err(e) => {
                    failures.push(format!("{name}/{case}: {e}"));
                    continue;
                }
            };
            if got.as_slice().len() != want.len() {
                failures.push(format!(
                    "{name}/{case}.{out_name}: expected {} values, got {}",
                    want.len(),
                    got.as_slice().len()
                ));
                continue;
            }
            for (i, (&a, &b)) in got.as_slice().iter().zip(want).enumerate() {
                if !agree(a, b) {
                    let (r, c) = (i % n_rows, i / n_rows);
                    failures.push(format!(
                        "{name}/{case} [{}] {out_name}[row {r}, col {c}]: got {a}, want {b}",
                        g("params")
                    ));
                    break;
                }
            }
        }
    }

    for name in ["ma", "mstd", "bbands", "rsi", "stoch", "macd", "atr", "obv"] {
        assert!(seen.contains_key(name), "no fixtures for {name}");
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

/// A swept run must equal the one-at-a-time runs, column for column.
///
/// The fixtures pin one combination each, which says nothing about how a
/// sweep arranges them. This does: running three windows at once must put
/// each window's result exactly where [`alphax_indicators::column_of`] says, and
/// the caching that makes a sweep cheap must not change any value.
#[test]
fn a_sweep_equals_its_parts() {
    let fixture = Fixture::load();
    let row = fixture
        .rows
        .iter()
        .find(|r| fixture.get(r, "case") == "random" && fixture.get(r, "indicator") == "ma")
        .expect("the random case is gone from the fixture set");

    let n_rows: usize = fixture.get(row, "n_rows").parse().unwrap();
    let n_cols: usize = fixture.get(row, "n_cols").parse().unwrap();
    let close =
        Matrix::from_column_major(numbers(fixture.get(row, "close")), n_rows, n_cols).unwrap();

    let windows = [2_usize, 3, 5, 2];
    let grid: Vec<MaParams> = windows
        .iter()
        .map(|&window| MaParams { window, ewm: false })
        .collect();
    let swept = ma(&close, &grid, false, None).unwrap();
    let swept_ma = swept.get("ma").unwrap();
    assert_eq!(swept_ma.n_cols(), windows.len() * n_cols);

    for (comb, &window) in windows.iter().enumerate() {
        let single = ma(&close, &[MaParams { window, ewm: false }], false, None).unwrap();
        let single_ma = single.get("ma").unwrap();
        for s in 0..n_cols {
            let col = alphax_indicators::column_of(comb, s, n_cols);
            assert!(
                swept_ma
                    .column(col)
                    .iter()
                    .zip(single_ma.column(s))
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "combination {comb} (window {window}), symbol {s} lands in the wrong column"
            );
        }
    }

    // The repeated window is a distinct combination with identical values —
    // the cache serves it, and the layout still gives it its own columns.
    for s in 0..n_cols {
        let first = swept_ma.column(alphax_indicators::column_of(0, s, n_cols));
        let repeat = swept_ma.column(alphax_indicators::column_of(3, s, n_cols));
        assert!(
            first
                .iter()
                .zip(repeat)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }

    // And the labels say which is which.
    assert_eq!(swept.columns.n_cols(), windows.len() * n_cols);
    assert!(swept.columns.describe(0).contains("window=2"));
    assert!(swept.columns.describe(2 * n_cols).contains("window=5"));
}

/// The fixture set must keep the cases that make the degenerate readings
/// reachable.
#[test]
fn the_fixtures_still_cover_the_degenerate_readings() {
    let fixture = Fixture::load();
    let mut saw_rsi_100 = false;
    let mut saw_rsi_0 = false;
    let mut saw_nan_from_flat_range = false;
    let mut cases: HashMap<&str, usize> = HashMap::new();

    for row in &fixture.rows {
        let g = |name: &str| fixture.get(row, name);
        *cases.entry(g("case")).or_default() += 1;
        let values = numbers(g("result").split('|').next().unwrap_or(""));
        match g("indicator") {
            "rsi" => {
                // Exact: 100 and 0 are the readings that arrive through a
                // division by zero, and an approximate match would also
                // accept a nearby ordinary value.
                saw_rsi_100 |= values.contains(&100.0);
                saw_rsi_0 |= values.contains(&0.0);
            }
            "stoch" if g("case") == "flat" => {
                saw_nan_from_flat_range |= values.iter().any(|v| v.is_nan());
            }
            _ => {}
        }
    }

    assert!(saw_rsi_100, "no RSI reading of 100 (nothing ever fell)");
    assert!(saw_rsi_0, "no RSI reading of 0 (nothing ever rose)");
    assert!(
        saw_nan_from_flat_range,
        "no stochastic on a flat range, where the denominator is zero"
    );
    for name in ["flat", "with_gaps", "gapping_open", "two_symbols", "short"] {
        assert!(cases.contains_key(name), "the {name} case is gone");
    }
}
