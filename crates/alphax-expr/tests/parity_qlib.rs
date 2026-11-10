use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

#[cfg(feature = "polars")]
use alphax_expr::PolarsBackend;
use alphax_expr::{
    ComputeBackend, FactorInputs, FactorSet, FactorValues, NativeBackend, alpha158, alpha360,
};
use alphax_model::OhlcvPanel;

struct Fixture {
    panel: OhlcvPanel,
    vwap: Vec<f64>,
    expected: HashMap<String, Vec<f64>>,
}

fn fixture(file_name: &str, factors: &FactorSet) -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(file_name);
    let text = fs::read_to_string(path).unwrap();
    let mut lines = text.lines().filter(|line| !line.starts_with('#'));
    let header = lines.next().unwrap().split('\t').collect::<Vec<_>>();
    let positions = header
        .iter()
        .enumerate()
        .map(|(index, name)| (*name, index))
        .collect::<HashMap<_, _>>();
    let factor_names = factors.names().map(str::to_owned).collect::<Vec<_>>();
    let mut fields = ["open", "high", "low", "close", "volume", "vwap"]
        .map(|name| (name, Vec::new()))
        .into_iter()
        .collect::<HashMap<_, _>>();
    let mut expected = factor_names
        .iter()
        .map(|name| (name.clone(), Vec::new()))
        .collect::<HashMap<_, _>>();
    let mut symbols = Vec::new();

    for line in lines {
        let cells = line.split('\t').collect::<Vec<_>>();
        symbols.push(cells[positions["symbol"]]);
        for (name, values) in &mut fields {
            values.push(cells[positions[name]].parse().unwrap());
        }
        for (name, values) in &mut expected {
            values.push(cells[positions[name.as_str()]].parse().unwrap());
        }
    }

    let n_rows = symbols
        .iter()
        .take_while(|symbol| **symbol == symbols[0])
        .count();
    let n_cols = symbols.len() / n_rows;
    assert_eq!(symbols.len(), n_rows * n_cols);
    for column in 0..n_cols {
        let start = column * n_rows;
        assert!(
            symbols[start..start + n_rows]
                .iter()
                .all(|symbol| *symbol == symbols[start])
        );
    }

    Fixture {
        panel: OhlcvPanel {
            open: fields.remove("open").unwrap(),
            high: fields.remove("high").unwrap(),
            low: fields.remove("low").unwrap(),
            close: fields.remove("close").unwrap(),
            volume: fields.remove("volume").unwrap(),
            n_rows,
            n_cols,
        },
        vwap: fields.remove("vwap").unwrap(),
        expected,
    }
}

fn assert_matches_qlib(backend: &dyn ComputeBackend, fixture: &Fixture, actual: &FactorValues) {
    assert_eq!(actual.iter().count(), fixture.expected.len());
    let mut mismatches = Vec::new();
    for (name, matrix) in actual.iter() {
        let expected = &fixture.expected[name];
        assert_eq!(matrix.as_slice().len(), expected.len());
        for (index, (&actual, &expected)) in matrix.as_slice().iter().zip(expected).enumerate() {
            if expected.is_nan() {
                if !actual.is_nan() {
                    mismatches.push(format!(
                        "{} {name}[{index}]: expected NaN, got {actual:?}",
                        backend.name()
                    ));
                }
                continue;
            }
            if actual.is_nan() {
                mismatches.push(format!(
                    "{} {name}[{index}]: expected {expected:?}, got NaN",
                    backend.name()
                ));
                continue;
            }
            // Qlib's rolling regression and correlation use incremental sums;
            // the stable backends recompute each window in f64.
            let epsilon = if name.starts_with("CORR") || name.starts_with("CORD") {
                1e-4
            } else {
                1e-6
            };
            let tolerance = epsilon * expected.abs().max(1.0);
            if (actual - expected).abs() > tolerance {
                mismatches.push(format!(
                    "{} {name}[{index}]: qlib={expected:?}, actual={actual:?}, tolerance={tolerance}",
                    backend.name()
                ));
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} mismatches (showing up to 20):\n{}",
        mismatches.len(),
        mismatches
            .iter()
            .take(20)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn native_matches_qlib_alpha158_fixture() {
    let factors = alpha158().unwrap();
    let fixture = fixture("qlib_alpha158.tsv", &factors);
    let inputs = FactorInputs::from_ohlcv(&fixture.panel)
        .unwrap()
        .with_field("vwap", &fixture.vwap)
        .unwrap();
    let actual = factors.evaluate_with(&inputs, &NativeBackend).unwrap();
    assert_matches_qlib(&NativeBackend, &fixture, &actual);
}

#[cfg(feature = "polars")]
#[test]
fn polars_matches_qlib_alpha158_fixture() {
    let factors = alpha158().unwrap();
    let fixture = fixture("qlib_alpha158.tsv", &factors);
    let inputs = FactorInputs::from_ohlcv(&fixture.panel)
        .unwrap()
        .with_field("vwap", &fixture.vwap)
        .unwrap();
    let actual = factors.evaluate_with(&inputs, &PolarsBackend).unwrap();
    assert_matches_qlib(&PolarsBackend, &fixture, &actual);
}

#[test]
fn native_matches_qlib_alpha360_fixture() {
    let factors = alpha360().unwrap();
    let fixture = fixture("qlib_alpha360.tsv", &factors);
    let inputs = FactorInputs::from_ohlcv(&fixture.panel)
        .unwrap()
        .with_field("vwap", &fixture.vwap)
        .unwrap();
    let actual = factors.evaluate_with(&inputs, &NativeBackend).unwrap();
    assert_matches_qlib(&NativeBackend, &fixture, &actual);
}

#[cfg(feature = "polars")]
#[test]
fn polars_matches_qlib_alpha360_fixture() {
    let factors = alpha360().unwrap();
    let fixture = fixture("qlib_alpha360.tsv", &factors);
    let inputs = FactorInputs::from_ohlcv(&fixture.panel)
        .unwrap()
        .with_field("vwap", &fixture.vwap)
        .unwrap();
    let actual = factors.evaluate_with(&inputs, &PolarsBackend).unwrap();
    assert_matches_qlib(&PolarsBackend, &fixture, &actual);
}
