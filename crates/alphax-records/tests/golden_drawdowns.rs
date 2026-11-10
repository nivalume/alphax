//! Golden test for the drawdown record layer.
//!
//! Where one episode ends and the next begins is the whole difficulty here,
//! and the boundaries are enumerable at small sizes: every path of up to five
//! bars over a three-value alphabet, plus every path of up to four bars over
//! two values and a gap. That makes the flat-stretch, exact-recovery,
//! V-shaped-rebound and trailing-gap cases certain rather than likely.
//!
//! The expected outputs are committed in `tests/fixtures/drawdowns.tsv`, so this
//! needs no Python.

use std::collections::HashMap;
use std::path::PathBuf;

use alphax_records::drawdowns::{DrawdownRecords, find_drawdowns};

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

fn fmt(v: f64) -> String {
    if v.is_nan() {
        "nan".to_owned()
    } else if v.is_infinite() {
        if v > 0.0 { "inf" } else { "-inf" }.to_owned()
    } else {
        format!("{v:?}")
    }
}

/// Records as the generator writes them.
fn render(d: &DrawdownRecords) -> String {
    if d.is_empty() {
        return "-".to_owned();
    }
    (0..d.len())
        .map(|i| {
            format!(
                "{},{},{},{},{},{},{},{},{}",
                d.col[i],
                d.peak_idx[i],
                d.start_idx[i],
                d.valley_idx[i],
                d.end_idx[i],
                fmt(d.peak_val[i]),
                fmt(d.valley_val[i]),
                fmt(d.end_val[i]),
                d.status[i]
            )
        })
        .collect::<Vec<_>>()
        .join("|")
}

fn agree(a: f64, b: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return a.is_nan() && b.is_nan();
    }
    if a.is_infinite() || b.is_infinite() {
        #[allow(clippy::float_cmp, reason = "infinities agree exactly or not at all")]
        let same = a == b;
        return same;
    }
    (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0)
}

struct Fixture {
    header: HashMap<String, usize>,
    rows: Vec<Vec<String>>,
}

impl Fixture {
    fn load() -> Self {
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/drawdowns.tsv");
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
fn matches_the_golden_drawdowns() {
    let fixture = Fixture::load();
    assert!(
        fixture.rows.len() > 900,
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
        let ts = parse_floats(g("ts"));
        let want = g("result");
        let d = find_drawdowns(&ts, n_rows, n_cols).expect("the generator matches shapes");

        match kind {
            "drawdowns" => {
                let got = render(&d);
                if got != want {
                    failures.push(format!(
                        "row {n} on {:?} [{}]: got {got}, want {want}",
                        g("ts"),
                        g("note")
                    ));
                }
            }
            "derived" => {
                // The five derived quantities, grouped per record.
                let dd = d.drawdown();
                let decline = d.decline_duration();
                let recovery = d.recovery_duration();
                let ratio = d.recovery_duration_ratio();
                let ret = d.recovery_return();
                let got = if d.is_empty() {
                    "-".to_owned()
                } else {
                    (0..d.len())
                        .map(|i| {
                            [dd[i], decline[i], recovery[i], ratio[i], ret[i]]
                                .iter()
                                .map(|v| fmt(*v))
                                .collect::<Vec<_>>()
                                .join(",")
                        })
                        .collect::<Vec<_>>()
                        .join("|")
                };
                let same = got == want
                    || (want != "-"
                        && got != "-"
                        && got.split(['|', ',']).count() == want.split(['|', ',']).count()
                        && got
                            .split(['|', ','])
                            .zip(want.split(['|', ',']))
                            .all(|(a, b)| agree(parse_floats(a)[0], parse_floats(b)[0])));
                if !same {
                    failures.push(format!(
                        "row {n} derived on {:?} [{}]: got {got}, want {want}",
                        g("ts"),
                        g("note")
                    ));
                }
            }
            other => panic!("unknown fixture function {other:?}"),
        }
    }

    for name in ["drawdowns", "derived"] {
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

/// The fixtures still enumerate the short paths, and still reach the shapes
/// that decide where one episode ends and the next begins.
#[test]
fn the_fixtures_still_enumerate_the_short_paths() {
    let fixture = Fixture::load();
    let mut paths: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut with_gap = 0_usize;
    let mut trailing_gap = 0_usize;
    for row in &fixture.rows {
        let g = |name: &str| fixture.get(row, name);
        if g("fn") != "drawdowns" || g("n_cols") != "1" {
            continue;
        }
        paths.insert(g("ts"));
        if g("ts").contains("nan") {
            with_gap += 1;
            if g("ts").ends_with("nan") {
                trailing_gap += 1;
            }
        }
    }
    // 3 + 9 + 27 + 81 + 243 distinct paths over three values, up to five bars.
    assert!(
        paths.len() >= 363,
        "the enumeration is no longer exhaustive over short paths: {} left",
        paths.len()
    );
    assert!(with_gap > 40, "only {with_gap} paths contain a gap");
    assert!(
        trailing_gap > 10,
        "only {trailing_gap} paths end on a gap, which is the case that cannot \
         close an episode"
    );
}
