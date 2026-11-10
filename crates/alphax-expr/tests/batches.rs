//! Batched output and the narrow-panel split give the same values as one
//! whole evaluation.

use alphax_expr::{FactorInputs, FactorSet, alpha158};

fn panel(n_rows: usize, n_cols: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let mut x = 100.0;
    (0..n_rows * n_cols)
        .map(|i| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            if i % n_rows == 0 {
                x = 100.0;
            }
            x *= 1.0 + ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.02;
            if i % 41 == 3 { f64::NAN } else { x }
        })
        .collect()
}

fn bits(m: &alphax_core::matrix::Matrix) -> Vec<u64> {
    m.as_slice().iter().map(|v| v.to_bits()).collect()
}

fn fields(n_rows: usize, n_cols: usize) -> Vec<(&'static str, Vec<f64>)> {
    ["open", "high", "low", "close", "volume", "vwap"]
        .iter()
        .enumerate()
        .map(|(k, name)| (*name, panel(n_rows, n_cols, k as u64 + 1)))
        .collect()
}

fn inputs<'a>(n_rows: usize, n_cols: usize, f: &'a [(&str, Vec<f64>)]) -> FactorInputs<'a> {
    let mut inputs = FactorInputs::new(n_rows, n_cols).unwrap();
    for (name, values) in f {
        inputs = inputs.with_field(name, values).unwrap();
    }
    inputs
}

#[test]
fn batches_cover_every_symbol_once_and_match() {
    let (n_rows, n_cols) = (120, 7);
    let f = fields(n_rows, n_cols);
    let inputs = inputs(n_rows, n_cols, &f);
    for set in [
        alpha158().unwrap(),
        FactorSet::compile([
            ("cs", "CSRank(Mean($close, 5)) - $open"),
            ("ts", "Std($close, 10)"),
        ])
        .unwrap(),
    ] {
        let whole = set.evaluate(&inputs).unwrap();
        let mut seen = Vec::new();
        set.evaluate_batches(&inputs, 3, |cols, values| {
            let expected = whole.columns(cols.clone()).unwrap();
            for ((name, got), (_, want)) in values.iter().zip(expected.iter()) {
                assert_eq!(bits(got), bits(want), "{name} {cols:?}");
            }
            seen.push(cols);
            Ok(())
        })
        .unwrap();
        assert_eq!(seen, [0..3, 3..6, 6..7]);
    }
    let set = alpha158().unwrap();
    assert!(set.evaluate_batches(&inputs, 0, |_, _| Ok(())).is_err());
    let stop = set.evaluate_batches(&inputs, 2, |cols, _| {
        if cols.start == 2 {
            Err(alphax_core::Error::Invariant("sink full".into()))
        } else {
            Ok(())
        }
    });
    assert!(stop.unwrap_err().to_string().contains("sink full"));
}

#[test]
fn a_narrow_panel_split_by_factor_matches_the_whole_graph() {
    // One thread never splits; four threads split a one- or two-symbol panel.
    let set = alpha158().unwrap();
    for n_cols in [1, 2] {
        let n_rows = 200;
        let f = fields(n_rows, n_cols);
        let inputs = inputs(n_rows, n_cols, &f);
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| set.evaluate(&inputs).unwrap())
        };
        let (whole, split) = (run(1), run(4));
        for ((name, a), (_, b)) in whole.iter().zip(split.iter()) {
            assert_eq!(bits(a), bits(b), "{name}");
        }
    }
}
