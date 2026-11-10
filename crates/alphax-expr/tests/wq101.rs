//! WorldQuant 101 against `tests/fixtures/wq101.tsv`: the paper's formulas
//! evaluated by an independent NumPy interpreter
//! (`tests/generators/generate_wq101.py`). Cells the fixture marks with `~` may
//! depend on how a tie in exact arithmetic rounds (a two-point correlation
//! of exactly 1, ranked against another 1) and are not compared; the
//! generator's docstring says how they are found.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use alphax_expr::{FactorInputs, wq101, wq101_all, wq101_formula};

struct Fixture {
    n_rows: usize,
    n_cols: usize,
    fields: HashMap<String, Vec<f64>>,
    kept: Vec<usize>,
    expected: HashMap<String, Vec<f64>>,
}

fn fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/wq101.tsv");
    let text = fs::read_to_string(path).unwrap();
    let mut lines = text.lines().filter(|line| !line.starts_with('#'));
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    let mut fields: HashMap<String, Vec<f64>> = HashMap::new();
    let mut expected: HashMap<String, Vec<f64>> = HashMap::new();
    let (mut n_rows, mut n_cols, mut kept) = (0, 0, Vec::new());
    for line in lines {
        let cells: Vec<&str> = line.split('\t').collect();
        let (col, row): (usize, usize) = (cells[0].parse().unwrap(), cells[1].parse().unwrap());
        n_rows = n_rows.max(row + 1);
        n_cols = n_cols.max(col + 1);
        let keep = cells[2] == "1";
        if keep && col == 0 {
            kept.push(row);
        }
        for (name, cell) in header.iter().zip(&cells).skip(3) {
            if cell.is_empty() {
                continue;
            }
            // A tie-dependent cell reads as an infinity of either sign:
            // never a real alpha value here, and skipped below.
            let value = match cell.strip_prefix('~') {
                Some(_) => f64::INFINITY,
                None => cell.parse::<f64>().unwrap(),
            };
            let target = if name.starts_with("alpha") {
                &mut expected
            } else {
                &mut fields
            };
            target.entry((*name).to_owned()).or_default().push(value);
        }
    }
    Fixture {
        n_rows,
        n_cols,
        fields,
        kept,
        expected,
    }
}

#[test]
fn every_alpha_matches_the_fixture() {
    let f = fixture();
    let mut inputs = FactorInputs::new(f.n_rows, f.n_cols).unwrap();
    for (name, values) in &f.fields {
        inputs = inputs.with_field(name, values).unwrap();
    }
    let set = wq101_all().unwrap();
    assert_eq!(set.names().len(), 101);
    let values = set.evaluate(&inputs).unwrap();
    let mut failures = Vec::new();
    let mut ties = 0;
    for (name, matrix) in values.iter() {
        let want = &f.expected[name];
        let mut worst = 0.0_f64;
        let mut at = None;
        for col in 0..f.n_cols {
            for (k, &row) in f.kept.iter().enumerate() {
                let (got, want) = (matrix.get(row, col), want[col * f.kept.len() + k]);
                if want.is_infinite() {
                    ties += 1;
                    continue;
                }
                let error = if got.is_nan() && want.is_nan() {
                    0.0
                } else {
                    // One side NaN is the worst error there is.
                    let e = (got - want).abs() / want.abs().max(1.0);
                    if e.is_nan() { f64::INFINITY } else { e }
                };
                if error > worst {
                    worst = error;
                    at = Some((row, col, got, want));
                }
            }
        }
        if worst > 1e-9 {
            failures.push(format!("{name}: worst {worst:e} at {at:?}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    // A small share of cells, or the comparison has stopped meaning much.
    // Alpha#29, #45, #80 and #96 are tie-dominated on this panel (#45 ranks
    // two-point correlations, which are all ±1).
    assert!(
        ties * 10 < 101 * f.n_cols * f.kept.len(),
        "{ties} tie cells"
    );
}

#[test]
fn the_default_set_reads_only_prices_and_volume() {
    let set = wq101().unwrap();
    assert_eq!(set.names().len(), 82);
    assert_eq!(
        set.required_fields().into_iter().collect::<Vec<_>>(),
        ["close", "high", "low", "open", "volume", "vwap"]
    );
    assert!(!set.uses_future());
    assert!(set.names().all(|n| n != "alpha056" && n != "alpha048"));
    assert_eq!(
        wq101_formula(101),
        Some("((close - open) / ((high - low) + .001))")
    );
    assert_eq!(wq101_formula(0), None);
}
