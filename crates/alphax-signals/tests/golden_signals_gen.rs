//! Golden test for signal generation, cleaning and ranking.
//!
//! The inputs are boolean arrays, so the awkward cases are about *shape*
//! rather than magnitude — and at these sizes the shapes can be enumerated
//! rather than sampled. Every array of up to six bars is covered for the
//! single-column functions, which leaves nothing to chance about leading
//! runs, trailing runs, alternating bars or empty columns.
//!
//! The expected outputs are committed in `tests/fixtures/signals_gen.tsv`, so this
//! needs no Python.

use std::collections::HashMap;
use std::path::PathBuf;

use alphax_signals::clean::{FirstSignal, clean_enex};
use alphax_signals::generate::{ExitWindow, exit_after};
use alphax_signals::ranges::{
    RangeRecords, between_partition_ranges, between_ranges, partition_ranges,
};
use alphax_signals::rank::{
    Gaps, LeadingRun, normalized_average_index, nth_index, partition_rank, signal_rank,
};

fn parse_bits(s: &str) -> Vec<bool> {
    s.chars().map(|c| c == 'X').collect()
}

fn render(v: &[bool]) -> String {
    v.iter().map(|b| if *b { 'X' } else { '.' }).collect()
}

fn parse_ints(s: &str) -> Vec<i64> {
    if s.is_empty() {
        Vec::new()
    } else {
        s.split(';').map(|v| v.parse().expect("int")).collect()
    }
}

fn parse_floats(s: &str) -> Vec<f64> {
    if s.is_empty() {
        Vec::new()
    } else {
        s.split(';')
            .map(|v| {
                if v == "nan" {
                    f64::NAN
                } else {
                    v.parse().expect("float")
                }
            })
            .collect()
    }
}

/// Renders ranges as the generator writes them: `col,start,end,status` groups.
fn render_ranges(r: &RangeRecords) -> String {
    if r.is_empty() {
        return "-".to_owned();
    }
    (0..r.len())
        .map(|i| {
            format!(
                "{},{},{},{}",
                r.col[i], r.start_idx[i], r.end_idx[i], r.status[i]
            )
        })
        .collect::<Vec<_>>()
        .join("|")
}

struct Fixture {
    header: HashMap<String, usize>,
    rows: Vec<Vec<String>>,
}

impl Fixture {
    fn load() -> Self {
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/signals_gen.tsv");
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
                // Trailing empty fields survive the split, but a trailing tab
                // can still be dropped by an editor; pad rather than panic.
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

/// The generator writes Python booleans.
fn flag(s: &str) -> bool {
    s == "True"
}

#[test]
fn matches_the_golden_signal_layer() {
    let fixture = Fixture::load();
    assert!(
        fixture.rows.len() > 5000,
        "expected the full fixture set, got {}",
        fixture.rows.len()
    );

    let mut failures: Vec<String> = Vec::new();
    let mut seen: HashMap<String, usize> = HashMap::new();

    for (n, row) in fixture.rows.iter().enumerate() {
        let g = |name: &str| fixture.get(row, name);
        let kind = g("fn");
        *seen.entry(kind.to_owned()).or_default() += 1;

        let n_rows: usize = g("n_rows").parse().unwrap();
        let n_cols: usize = g("n_cols").parse().unwrap();
        let a = parse_bits(g("a"));
        let want = g("result");
        let mut fail =
            |msg: String| failures.push(format!("row {n} {kind} on {:?}: {msg}", g("a")));

        match kind {
            "signal_rank" | "partition_rank" => {
                let reset = (!g("reset").is_empty()).then(|| parse_bits(g("reset")));
                let leading = if flag(g("after_false")) {
                    LeadingRun::Skip
                } else {
                    LeadingRun::Rank
                };
                let got = if kind == "signal_rank" {
                    let gaps = if flag(g("allow_gaps")) {
                        Gaps::Allow
                    } else {
                        Gaps::ResetEachRun
                    };
                    signal_rank(&a, reset.as_deref(), n_rows, n_cols, gaps, leading)
                } else {
                    partition_rank(&a, reset.as_deref(), n_rows, n_cols, leading)
                };
                let expected = parse_ints(want);
                if got != expected {
                    fail(format!("got {got:?}, want {expected:?}"));
                }
            }
            "partition_ranges" | "between_ranges" | "between_partition_ranges" => {
                let got = match kind {
                    "partition_ranges" => partition_ranges(&a, n_rows, n_cols),
                    "between_ranges" => between_ranges(&a, n_rows, n_cols),
                    _ => between_partition_ranges(&a, n_rows, n_cols),
                }
                .expect("shapes match");
                let rendered = render_ranges(&got);
                if rendered != want {
                    fail(format!("got {rendered}, want {want}"));
                }
            }
            "nth_index" => {
                let nth: i64 = g("n").parse().unwrap();
                let got = nth_index(&a, n_rows, n_cols, nth);
                let expected = parse_ints(want);
                if got != expected {
                    fail(format!("n={nth}: got {got:?}, want {expected:?}"));
                }
            }
            "norm_avg_index" => {
                let got = normalized_average_index(&a, n_rows, n_cols);
                let expected = parse_floats(want);
                for (x, y) in got.iter().zip(&expected) {
                    let agree = if x.is_nan() || y.is_nan() {
                        x.is_nan() && y.is_nan()
                    } else {
                        (x - y).abs() < 1e-12
                    };
                    if !agree {
                        fail(format!("got {got:?}, want {expected:?}"));
                        break;
                    }
                }
            }
            "clean_enex" => {
                let b = parse_bits(g("b"));
                let first = if flag(g("entry_first")) {
                    FirstSignal::EntryFirst
                } else {
                    FirstSignal::Either
                };
                let (e, x) = clean_enex(&a, &b, n_rows, n_cols, first).expect("shapes match");
                if render(&e) != want || render(&x) != g("result2") {
                    fail(format!(
                        "exits {:?}: got ({}, {}), want ({want}, {})",
                        g("b"),
                        render(&e),
                        render(&x),
                        g("result2")
                    ));
                }
            }
            "exit_after" => {
                let wait: usize = g("wait").parse().unwrap();
                let window = if flag(g("until_next")) {
                    ExitWindow::UntilNextEntry
                } else if flag(g("skip_until_exit")) {
                    ExitWindow::ToEndSkippingResolved
                } else {
                    ExitWindow::ToEnd
                };
                let got = exit_after(&a, n_rows, n_cols, wait, window).expect("shapes match");
                if render(&got) != want {
                    fail(format!(
                        "wait {wait} {window:?}: got {}, want {want}",
                        render(&got)
                    ));
                }
            }
            other => panic!("unknown fixture function {other:?}"),
        }
    }

    for name in [
        "signal_rank",
        "partition_rank",
        "partition_ranges",
        "between_ranges",
        "between_partition_ranges",
        "nth_index",
        "norm_avg_index",
        "clean_enex",
        "exit_after",
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
            .take(25)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// The fixture set is exhaustive at these sizes, and must stay that way.
///
/// A sampled set would sooner or later stop containing a column that begins
/// mid-run, or one whose run reaches the last bar, and those are exactly the
/// cases the ranking and range code exists to get right.
#[test]
fn the_fixtures_still_enumerate_every_short_array() {
    let fixture = Fixture::load();
    let mut ranked: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut with_reset = 0usize;
    for row in &fixture.rows {
        let g = |name: &str| fixture.get(row, name);
        if g("fn") == "partition_ranges" && g("n_cols") == "1" {
            ranked.insert(g("a"));
        }
        if !g("reset").is_empty() {
            with_reset += 1;
        }
    }
    // 2 + 4 + 8 + 16 + 32 + 64 arrays of length 1 through 6.
    assert_eq!(
        ranked.len(),
        126,
        "the enumeration is no longer exhaustive over arrays of up to six bars"
    );
    assert!(
        ranked.contains("XXXXXX") && ranked.contains("......") && ranked.contains("X....."),
        "the all-true, all-false and leading-run arrays are gone"
    );
    assert!(with_reset > 20, "only {with_reset} cases exercise a reset");
}
