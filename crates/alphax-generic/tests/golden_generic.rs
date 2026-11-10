//! Golden test for the rolling and elementwise array layer.
//!
//! The arithmetic here is trivial and the edge cases are not, so the fixture
//! series are awkward on purpose: leading and trailing gaps, interior gaps,
//! all-`NaN`, constant, shorter than its own window, and values spanning
//! sixteen orders of magnitude.
//!
//! That last one is the one that earns its place. [`rolling_mean`] keeps a
//! running sum and subtracts what leaves the window rather than re-summing
//! it — `O(1)` per bar, and a different answer in the last bits. A fixture set
//! of well-scaled numbers would never tell the two apart, and the difference
//! is enough to move a fill across a threshold downstream.
//!
//! The expected outputs are committed in `tests/fixtures/generic.tsv`, so this
//! needs no Python.

use std::collections::HashMap;
use std::path::PathBuf;

use alphax_core::math::is_close;
use alphax_generic::elementwise::{
    bfill_1d, bshift_1d, crossed_above_1d, diff_1d, ffill_1d, fillna_1d, fshift_1d, pct_change_1d,
};
use alphax_generic::rolling::{
    ewm_mean_1d, ewm_std_1d, expanding_max_1d, expanding_mean_1d, expanding_min_1d,
    expanding_std_1d, rolling_max_1d, rolling_mean_1d, rolling_min_1d, rolling_std_1d,
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

fn numbers(s: &str) -> Vec<f64> {
    if s.is_empty() {
        Vec::new()
    } else {
        s.split(';').map(number).collect()
    }
}

struct Fixture {
    header: HashMap<String, usize>,
    rows: Vec<Vec<String>>,
    /// Named series, defined in the file's own preamble so that the two sides
    /// cannot drift apart.
    series: HashMap<String, Vec<f64>>,
}

impl Fixture {
    fn load() -> Self {
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/generic.tsv");
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));

        let mut series = HashMap::new();
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("#series\t") {
                let mut parts = rest.split('\t');
                let name = parts.next().expect("series name").to_owned();
                let values = numbers(parts.next().expect("series values"));
                series.insert(name, values);
            }
        }

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
        Self {
            header,
            rows,
            series,
        }
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
fn matches_the_golden_array_layer() {
    let fixture = Fixture::load();
    assert!(
        fixture.rows.len() > 1500,
        "expected the full fixture set, got {}",
        fixture.rows.len()
    );
    assert!(!fixture.series.is_empty(), "no series definitions found");

    let mut failures: Vec<String> = Vec::new();
    let mut seen: HashMap<String, usize> = HashMap::new();
    let mut errors = 0usize;

    for (n, row) in fixture.rows.iter().enumerate() {
        let g = |name: &str| fixture.get(row, name);
        let kind = g("fn");
        *seen.entry(kind.to_owned()).or_default() += 1;

        // Crossovers carry their two series inline; everything else names one
        // from the preamble.
        let a: Vec<f64> = if kind == "crossed_above" {
            numbers(g("series"))
        } else {
            fixture
                .series
                .get(g("series"))
                .unwrap_or_else(|| panic!("unknown series {:?}", g("series")))
                .clone()
        };

        let window: Option<usize> = (!g("window").is_empty()).then(|| g("window").parse().unwrap());
        let minp: Option<usize> = (!g("minp").is_empty()).then(|| g("minp").parse().unwrap());
        let ddof: usize = if g("ddof").is_empty() {
            0
        } else {
            g("ddof").parse().unwrap()
        };
        let shift: usize = if g("n").is_empty() {
            1
        } else {
            g("n").parse().unwrap()
        };
        let adjust = g("adjust") == "true";

        // Boolean-valued functions are compared separately.
        if kind == "crossed_above" {
            let b = numbers(g("series2"));
            let wait: usize = g("wait").parse().unwrap();
            let got = crossed_above_1d(&a, &b, wait);
            let want: Vec<bool> = g("result").split(';').map(|s| s == "true").collect();
            if got != want {
                failures.push(format!(
                    "row {n} crossed_above ({}) wait {wait}: got {got:?}, want {want:?}",
                    g("note")
                ));
            }
            continue;
        }

        let computed: Result<Vec<f64>, String> = match kind {
            "rolling_mean" => rolling_mean_1d(&a, window.unwrap(), minp).map_err(|e| e.to_string()),
            "rolling_std" => {
                rolling_std_1d(&a, window.unwrap(), minp, ddof).map_err(|e| e.to_string())
            }
            "rolling_min" => rolling_min_1d(&a, window.unwrap(), minp).map_err(|e| e.to_string()),
            "rolling_max" => rolling_max_1d(&a, window.unwrap(), minp).map_err(|e| e.to_string()),
            "expanding_mean" => expanding_mean_1d(&a, minp.unwrap()).map_err(|e| e.to_string()),
            "expanding_std" => expanding_std_1d(&a, minp.unwrap(), ddof).map_err(|e| e.to_string()),
            "expanding_min" => Ok(expanding_min_1d(&a, minp.unwrap())),
            "expanding_max" => Ok(expanding_max_1d(&a, minp.unwrap())),
            "ewm_mean" => ewm_mean_1d(&a, window.unwrap(), minp.unwrap_or(0), adjust)
                .map_err(|e| e.to_string()),
            "ewm_std" => ewm_std_1d(&a, window.unwrap(), minp.unwrap_or(0), adjust)
                .map_err(|e| e.to_string()),
            "fshift" => Ok(fshift_1d(&a, shift, number(g("fill")))),
            "bshift" => Ok(bshift_1d(&a, shift, number(g("fill")))),
            "diff" => Ok(diff_1d(&a, shift)),
            "pct_change" => Ok(pct_change_1d(&a, shift)),
            "ffill" => Ok(ffill_1d(&a)),
            "bfill" => Ok(bfill_1d(&a)),
            "fillna" => Ok(fillna_1d(&a, number(g("fill")))),
            other => panic!("unknown fixture function {other:?}"),
        };

        if g("outcome") == "error" {
            errors += 1;
            if computed.is_ok() {
                failures.push(format!("row {n} {kind}: expected a rejection, got values"));
            }
            continue;
        }

        let got = match computed {
            Ok(v) => v,
            Err(e) => {
                failures.push(format!("row {n} {kind}: unexpected rejection: {e}"));
                continue;
            }
        };
        let want = numbers(g("result"));
        if got.len() != want.len() {
            failures.push(format!(
                "row {n} {kind}: expected {} values, got {}",
                want.len(),
                got.len()
            ));
            continue;
        }
        for (i, (&x, &y)) in got.iter().zip(&want).enumerate() {
            if !agree(x, y) {
                failures.push(format!(
                    "row {n} {kind} on {:?} [{i}]: got {x}, want {y}",
                    g("series")
                ));
                break;
            }
        }
    }

    assert!(errors >= 2, "only {errors} rejection cases in the set");
    for name in [
        "rolling_mean",
        "rolling_std",
        "rolling_min",
        "rolling_max",
        "expanding_mean",
        "expanding_std",
        "ewm_mean",
        "ewm_std",
        "fshift",
        "bshift",
        "diff",
        "pct_change",
        "ffill",
        "bfill",
        "crossed_above",
    ] {
        assert!(seen.contains_key(name), "no fixtures for {name}");
    }
    assert!(
        failures.is_empty(),
        "{} of {} rows disagree with the golden outputs:\n{}",
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

/// The fixture set must keep the series that make the edge cases reachable.
///
/// Every one of these exists because a plausible implementation gets it wrong
/// and a well-behaved series would not notice.
#[test]
fn the_fixtures_still_cover_the_awkward_series() {
    let fixture = Fixture::load();
    for name in [
        "leading_gap",
        "trailing_gap",
        "interior_gaps",
        "all_nan",
        "constant",
        "single",
        "wide_scale",
    ] {
        assert!(
            fixture.series.contains_key(name),
            "the {name} series is gone from the fixture set"
        );
    }

    // And the wide-scale one has to actually span the range that separates an
    // accumulating mean from a re-summing one.
    let wide = &fixture.series["wide_scale"];
    let max = wide.iter().copied().fold(0.0_f64, f64::max);
    let min = wide
        .iter()
        .copied()
        .filter(|v| *v > 0.0)
        .fold(f64::INFINITY, f64::min);
    assert!(
        max / min > 1e12,
        "the wide_scale series spans only {:e}",
        max / min
    );
}
