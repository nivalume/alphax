//! Golden test for the return metrics, scalar and rolling.
//!
//! Where `golden_analysis.rs` runs a whole simulation and checks the metrics
//! at the end, this feeds return series straight in — which is the only way to
//! reach the degenerate cases on purpose. A simulation will not readily
//! produce a column that is entirely missing, one that never moved, or one
//! whose benchmark never rose, and those are where the metrics drift.
//!
//! The expected outputs are committed in `tests/fixtures/metrics.tsv`, so this
//! needs no Python.

use std::collections::HashMap;
use std::path::PathBuf;

use alphax_analysis::benchmark;
use alphax_analysis::metrics;
use alphax_analysis::rolling;

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

fn parse_one(s: &str, default: f64) -> f64 {
    if s.is_empty() {
        default
    } else {
        parse_floats(s)[0]
    }
}

/// Agreement to the last few bits, with `NaN` and the infinities exact.
///
/// A metric that answered `+inf` where the expected output is a large finite
/// number is not "close" — it is the difference between a column that sorts
/// first in a sweep and one that does not.
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
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/metrics.tsv");
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
fn matches_the_golden_metrics() {
    let fixture = Fixture::load();
    assert!(
        fixture.rows.len() > 1500,
        "expected the full fixture set, got {}",
        fixture.rows.len()
    );

    let mut failures: Vec<String> = Vec::new();
    let mut seen: HashMap<String, usize> = HashMap::new();

    for (n, row) in fixture.rows.iter().enumerate() {
        let g = |name: &str| fixture.get(row, name);
        let kind = g("fn");
        *seen.entry(kind.to_owned()).or_default() += 1;

        let r = parse_floats(g("returns"));
        let b = parse_floats(g("benchmark"));
        let ann = parse_one(g("ann_factor"), 365.0);
        let rf = parse_one(g("risk_free"), 0.0);
        let req = parse_one(g("required_return"), 0.0);
        let cutoff = parse_one(g("cutoff"), 0.05);
        let ddof: usize = if g("ddof").is_empty() {
            1
        } else {
            g("ddof").parse().unwrap()
        };
        let window: usize = if g("window").is_empty() {
            0
        } else {
            g("window").parse().unwrap()
        };
        let minp: Option<usize> = if g("minp").is_empty() {
            None
        } else {
            Some(g("minp").parse().unwrap())
        };

        let mut fail = |msg: String| {
            failures.push(format!("row {n} {kind} [{}]: {msg}", g("note")));
        };

        // The rolling forms come back as a series; the scalars as one value.
        if let Some(name) = kind.strip_prefix("rolling_") {
            let got = match name {
                "total_return" => rolling::rolling_total_return(&r, window, minp),
                "annualized_return" => rolling::rolling_annualized_return(&r, window, minp, ann),
                "annualized_volatility" => {
                    rolling::rolling_annualized_volatility(&r, window, minp, ann)
                }
                "max_drawdown" => rolling::rolling_max_drawdown(&r, window, minp),
                "calmar_ratio" => rolling::rolling_calmar_ratio(&r, window, minp, ann),
                "omega_ratio" => rolling::rolling_omega_ratio(&r, window, minp, ann, rf, req),
                "sharpe_ratio" => rolling::rolling_sharpe_ratio(&r, window, minp, ann, rf),
                "downside_risk" => rolling::rolling_downside_risk(&r, window, minp, ann, req),
                "sortino_ratio" => rolling::rolling_sortino_ratio(&r, window, minp, ann, req),
                "tail_ratio" => rolling::rolling_tail_ratio(&r, window, minp),
                "value_at_risk" => rolling::rolling_value_at_risk(&r, window, minp, cutoff),
                "cond_value_at_risk" => {
                    rolling::rolling_cond_value_at_risk(&r, window, minp, cutoff)
                }
                "information_ratio" => {
                    rolling::rolling_information_ratio(&r, &b, window, minp, ddof)
                }
                "beta" => rolling::rolling_beta(&r, &b, window, minp),
                "alpha" => rolling::rolling_alpha(&r, &b, window, minp, ann, rf),
                "capture" => rolling::rolling_capture(&r, &b, window, minp, ann),
                "up_capture" => rolling::rolling_up_capture(&r, &b, window, minp, ann),
                "down_capture" => rolling::rolling_down_capture(&r, &b, window, minp, ann),
                other => panic!("unknown rolling fixture {other:?}"),
            }
            .expect("the fixture stays within the shape rules");
            let want = parse_floats(g("result"));
            if got.len() != want.len() || !got.iter().zip(&want).all(|(a, b)| agree(*a, *b)) {
                fail(format!(
                    "window {window} minp {minp:?}: got {got:?}, want {want:?}"
                ));
            }
            continue;
        }

        let got = match kind {
            "total_return" => metrics::total_return(&r),
            "annualized_return" => metrics::annualized_return(&r, ann),
            "annualized_volatility" => metrics::annualized_volatility(&r, ann),
            "max_drawdown" => metrics::max_drawdown(&r),
            "calmar_ratio" => metrics::calmar_ratio(&r, ann),
            "omega_ratio" => metrics::omega_ratio(&r, ann, rf, req),
            "sharpe_ratio" => metrics::sharpe_ratio(&r, ann, rf),
            "downside_risk" => metrics::downside_risk(&r, ann, req),
            "sortino_ratio" => metrics::sortino_ratio(&r, ann, req),
            "tail_ratio" => metrics::tail_ratio(&r),
            "value_at_risk" => metrics::value_at_risk(&r, cutoff),
            "cond_value_at_risk" => metrics::cond_value_at_risk(&r, cutoff),
            "information_ratio" => benchmark::information_ratio(&r, &b, ddof),
            "beta" => benchmark::beta(&r, &b),
            "alpha" => benchmark::alpha(&r, &b, ann, rf),
            "capture" => benchmark::capture(&r, &b, ann),
            "up_capture" => benchmark::up_capture(&r, &b, ann),
            "down_capture" => benchmark::down_capture(&r, &b, ann),
            other => panic!("unknown fixture metric {other:?}"),
        };
        let want = parse_floats(g("result"))[0];
        if !agree(got, want) {
            fail(format!(
                "returns {:?} ann {ann} rf {rf} req {req} cutoff {cutoff}: got {got}, want {want}",
                g("returns")
            ));
        }
    }

    for name in [
        "omega_ratio",
        "tail_ratio",
        "value_at_risk",
        "cond_value_at_risk",
        "information_ratio",
        "beta",
        "alpha",
        "capture",
        "up_capture",
        "down_capture",
        "rolling_sharpe_ratio",
        "rolling_beta",
        "rolling_cond_value_at_risk",
    ] {
        assert!(seen.contains_key(name), "no fixtures for {name}");
    }
    assert_eq!(
        seen.keys().filter(|k| k.starts_with("rolling_")).count(),
        18,
        "every scalar metric should have a rolling counterpart under test"
    );
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

/// The fixtures still reach the degenerate branches, not just the interior.
///
/// Every metric here has a branch that only a pathological series enters, and
/// those branches are where an implementation silently drifts. A fixture set that
/// drifted toward well-behaved random returns would pass while testing none of
/// them.
#[test]
fn the_fixtures_still_reach_the_degenerate_cases() {
    let fixture = Fixture::load();
    let mut has_empty = false;
    let mut has_all_nan = false;
    let mut has_infinity = false;
    let mut has_flat_benchmark = false;
    let mut has_one_ann_factor = false;
    let mut has_negative_ann_factor = false;
    let mut has_extreme_cutoff = false;
    let mut has_partial_window = false;

    for row in &fixture.rows {
        let g = |name: &str| fixture.get(row, name);
        let note = g("note");
        has_empty |= note == "empty";
        has_all_nan |= note.contains("nothing but gaps") || note.contains("all gaps");
        has_infinity |= note.contains("infinity");
        has_flat_benchmark |= note.contains("flat benchmark");
        has_one_ann_factor |= g("ann_factor") == "1.0";
        has_negative_ann_factor |= g("ann_factor").starts_with('-');
        has_extreme_cutoff |= g("cutoff") == "0.0" || g("cutoff") == "1.0";
        has_partial_window |= !g("minp").is_empty() && g("minp") != g("window");
    }

    assert!(has_empty, "no empty series");
    assert!(has_all_nan, "no column of nothing but gaps");
    assert!(has_infinity, "no infinite return");
    assert!(has_flat_benchmark, "no benchmark that never moved");
    assert!(
        has_one_ann_factor,
        "no annualization factor of exactly 1, which skips omega's threshold \
         conversion"
    );
    assert!(
        has_negative_ann_factor,
        "no negative annualization factor, which omega has no real root for"
    );
    assert!(has_extreme_cutoff, "no cutoff at 0 or 1");
    assert!(
        has_partial_window,
        "no min_periods below the window, so the short leading windows are \
         never exercised"
    );
}
