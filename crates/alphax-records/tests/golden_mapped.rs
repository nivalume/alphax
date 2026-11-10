//! Golden test for the mapped-array layer.
//!
//! The input space is small enough to enumerate: every assignment of up to
//! four records to up to three columns, crossed with a handful of value
//! vectors chosen for the branches they reach. That leaves nothing to chance
//! about empty columns, interleaved columns, ties, or `NaN` records.
//!
//! The expected outputs are committed in `tests/fixtures/mapped.tsv`, so this
//! needs no Python.

use std::collections::HashMap;
use std::path::PathBuf;

use alphax_records::mapped::{ColMap, MappedArray};

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
            .map(|v| match v {
                "nan" => f64::NAN,
                "inf" => f64::INFINITY,
                "-inf" => f64::NEG_INFINITY,
                other => other.parse().expect("float"),
            })
            .collect()
    }
}

fn parse_bits(s: &str) -> Vec<bool> {
    s.chars().map(|c| c == 'X').collect()
}

fn render_bits(v: &[bool]) -> String {
    v.iter().map(|b| if *b { 'X' } else { '.' }).collect()
}

/// `NaN == NaN` here: the fixtures record which columns had no answer, and
/// that is exactly what must agree.
fn agree(a: f64, b: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        a.is_nan() && b.is_nan()
    } else if a.is_infinite() || b.is_infinite() {
        // No tolerance applies to an infinity, and the sign is the whole
        // content of the answer: a column whose Sharpe ran away upward is not
        // the same result as one that ran away downward.
        #[allow(clippy::float_cmp, reason = "infinities agree exactly or not at all")]
        let same = a == b;
        same
    } else {
        (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.0)
    }
}

struct Fixture {
    header: HashMap<String, usize>,
    rows: Vec<Vec<String>>,
}

impl Fixture {
    fn load() -> Self {
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/mapped.tsv");
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
fn matches_the_golden_mapped_layer() {
    let fixture = Fixture::load();
    assert!(
        fixture.rows.len() > 20_000,
        "expected the full fixture set, got {}",
        fixture.rows.len()
    );

    let mut failures: Vec<String> = Vec::new();
    let mut seen: HashMap<String, usize> = HashMap::new();

    for (n, row) in fixture.rows.iter().enumerate() {
        let g = |name: &str| fixture.get(row, name);
        let kind = g("fn");
        *seen.entry(kind.to_owned()).or_default() += 1;

        let n_cols: usize = g("n_cols").parse().unwrap();
        let col_arr = parse_ints(g("col_arr"));
        let values = parse_floats(g("values"));
        let want = g("result");
        let map = ColMap::build(&col_arr, n_cols).expect("the generator stays in range");
        let mut fail = |msg: String| {
            failures.push(format!(
                "row {n} {kind} cols={:?} vals={:?} [{}]: {msg}",
                g("col_arr"),
                g("values"),
                g("note")
            ));
        };

        if kind == "col_map" {
            let idxs: Vec<i64> = (0..n_cols)
                .flat_map(|c| map.column(c).iter().map(|&i| i as i64))
                .collect();
            let lens: Vec<i64> = map.lens().iter().map(|&l| l as i64).collect();
            let (want_idxs, want_lens) = want.split_once('|').expect("col_map has two halves");
            if idxs != parse_ints(want_idxs) || lens != parse_ints(want_lens) {
                fail(format!("got {idxs:?}|{lens:?}, want {want}"));
            }
            continue;
        }

        if kind == "value_counts" {
            let codes = parse_ints(g("codes"));
            let n_uniques: usize = g("n_uniques").parse().unwrap();
            let zeros = vec![0.0; map.len()];
            let m = MappedArray::new(&zeros, &map).unwrap();
            let got = m.value_counts(&codes, n_uniques).unwrap();
            let expected = parse_ints(want);
            let got_ints: Vec<i64> = got.as_slice().iter().map(|&v| v as i64).collect();
            if got_ints != expected {
                fail(format!("got {got_ints:?}, want {expected:?}"));
            }
            continue;
        }

        let m = MappedArray::new(&values, &map).expect("value count matches the map");

        match kind {
            "reduce_sum" | "reduce_mean" | "reduce_min" | "reduce_max" | "reduce_count"
            | "reduce_std0" | "reduce_std1" => {
                let got = match kind {
                    "reduce_sum" => m.sum(),
                    "reduce_mean" => m.mean(),
                    "reduce_min" => m.min(),
                    "reduce_max" => m.max(),
                    "reduce_count" => m.count(),
                    "reduce_std0" => m.std(0),
                    _ => m.std(1),
                };
                let expected = parse_floats(want);
                if got.len() != expected.len()
                    || !got.iter().zip(&expected).all(|(a, b)| agree(*a, *b))
                {
                    fail(format!("got {got:?}, want {expected:?}"));
                }
            }
            "stack_expand" => {
                let fill = parse_floats(g("fill"))[0];
                let got = m.stack_expand(fill);
                let expected = parse_floats(want);
                if got.as_slice().len() != expected.len()
                    || !got
                        .as_slice()
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| agree(*a, *b))
                {
                    fail(format!(
                        "got {:?}, want {expected:?}",
                        got.as_slice().to_vec()
                    ));
                }
            }
            "top_n" | "bottom_n" => {
                let k: usize = g("n").parse().unwrap();
                let got = if kind == "top_n" {
                    m.top_n_mask(k)
                } else {
                    m.bottom_n_mask(k)
                };
                if got != parse_bits(want) {
                    fail(format!("n={k}: got {}, want {want}", render_bits(&got)));
                }
            }
            "idxmax" | "idxmin" => {
                let idx_arr = parse_ints(g("idx_arr"));
                let got = if kind == "idxmax" {
                    m.idxmax(&idx_arr, -1)
                } else {
                    m.idxmin(&idx_arr, -1)
                }
                .expect("the generator stays in range");
                let expected = parse_ints(want);
                if got != expected {
                    fail(format!("got {got:?}, want {expected:?}"));
                }
            }
            other => panic!("unknown fixture function {other:?}"),
        }
    }

    for name in [
        "col_map",
        "reduce_sum",
        "reduce_mean",
        "reduce_std1",
        "stack_expand",
        "top_n",
        "bottom_n",
        "idxmax",
        "idxmin",
        "value_counts",
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

/// The fixture set enumerates every short assignment, and must stay that way.
///
/// A sampled set would sooner or later stop containing the column that gets no
/// records, or the interleaved assignment that forces the gathering path —
/// which are the two cases this layer exists to get right.
#[test]
fn the_fixtures_still_enumerate_every_short_assignment() {
    let fixture = Fixture::load();
    let mut assignments: std::collections::HashSet<(String, String)> =
        std::collections::HashSet::new();
    let mut interleaved = 0_usize;
    let mut empty_columns = 0_usize;
    for row in &fixture.rows {
        let g = |name: &str| fixture.get(row, name);
        if g("fn") != "col_map" {
            continue;
        }
        assignments.insert((g("n_cols").to_owned(), g("col_arr").to_owned()));
        let cols = parse_ints(g("col_arr"));
        if cols.windows(2).any(|w| w[1] < w[0]) {
            interleaved += 1;
        }
        let n_cols: usize = g("n_cols").parse().unwrap();
        if (0..n_cols as i64).any(|c| !cols.contains(&c)) {
            empty_columns += 1;
        }
    }
    // Sum over n_cols in 1..=3 and n in 0..=4 of n_cols^n.
    let expected: usize = (1..=3)
        .map(|c: usize| (0..=4).map(|n| c.pow(n)).sum::<usize>())
        .sum();
    assert_eq!(
        assignments.len(),
        expected,
        "the enumeration is no longer exhaustive over short column assignments"
    );
    assert!(
        interleaved > 30,
        "only {interleaved} assignments force the gathering path"
    );
    assert!(
        empty_columns > 30,
        "only {empty_columns} assignments leave a column with no records"
    );
}

/// Records this engine produces are column-major, so they take the fast path.
///
/// Not a performance assertion — a structural one. If simulation ever started
/// emitting records out of column order, every reduction would silently fall
/// back to gathering and trade derivation would fail outright.
#[test]
fn a_column_major_assignment_needs_no_gather() {
    let mut any_contiguous = 0_usize;
    for cols in [vec![], vec![0_i64], vec![0, 0, 1], vec![0, 1, 1, 2]] {
        let map = ColMap::build(&cols, 3).unwrap();
        assert!(map.is_contiguous(), "{cols:?} should need no gather");
        any_contiguous += 1;
    }
    assert_eq!(any_contiguous, 4);
    assert!(!ColMap::build(&[1_i64, 0], 3).unwrap().is_contiguous());
}
