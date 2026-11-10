//! A parameter grid is every case compiled into one set: its values are the
//! values of each case compiled alone, laid out case by case.

use alphax_expr::{FactorGrid, FactorInputs, FactorSet, GridCase};

fn panel(n_rows: usize, n_cols: usize, seed: f64) -> Vec<f64> {
    (0..n_rows * n_cols)
        .map(|k| 100.0 + ((k as f64) * 0.37 + seed).sin() * 5.0 + (k % n_rows) as f64 * 0.02)
        .collect()
}

fn case(pairs: &[(&str, &str)]) -> GridCase {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect()
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn a_grid_is_each_case_compiled_alone_side_by_side() {
    let (n_rows, n_cols) = (120, 3);
    let close = panel(n_rows, n_cols, 0.0);
    let volume = panel(n_rows, n_cols, 3.0);
    let inputs = FactorInputs::new(n_rows, n_cols)
        .unwrap()
        .with_field("close", &close)
        .unwrap()
        .with_field("volume", &volume)
        .unwrap();
    let factors = [
        ("z", "($close - Mean($close, {n})) / Std($close, {n})"),
        ("flow", "Corr(${field}, $volume, {n}) > {level}"),
        ("rank", "CSRank(Mean($close, {n}))"),
        ("plain", "$close / Ref($close, 1) - 1"),
    ];
    let cases = vec![
        case(&[("n", "5"), ("field", "close"), ("level", "0.2")]),
        case(&[("n", "20"), ("field", "close"), ("level", "-0.5")]),
        case(&[("n", "20"), ("field", "volume"), ("level", "0")]),
    ];
    let grid = FactorGrid::compile(factors, cases.clone()).unwrap();
    assert_eq!(
        grid.names().collect::<Vec<_>>(),
        ["z", "flow", "rank", "plain"]
    );
    let values = grid.evaluate(&inputs).unwrap();
    for (name, source) in factors {
        let wide = values.get(name).unwrap();
        assert_eq!(
            (wide.n_rows(), wide.n_cols()),
            (n_rows, n_cols * cases.len())
        );
        for (k, case) in cases.iter().enumerate() {
            let mut text = source.to_owned();
            for (param, value) in case {
                text = text.replace(&format!("{{{param}}}"), value);
            }
            let alone = FactorSet::compile([("f", text)])
                .unwrap()
                .evaluate(&inputs)
                .unwrap();
            let alone = alone.get("f").unwrap().as_slice();
            let part = &wide.as_slice()[k * n_rows * n_cols..(k + 1) * n_rows * n_cols];
            assert_eq!(bits(part), bits(alone), "{name} case {k}");
        }
    }
    assert_eq!(grid.lookbacks()[0], [4, 19, 19]);
}

#[test]
fn warmup_follows_each_cases_own_lookback() {
    let (n_rows, n_cols) = (40, 2);
    let close = panel(n_rows, n_cols, 1.0);
    let inputs = FactorInputs::new(n_rows, n_cols)
        .unwrap()
        .with_field("close", &close)
        .unwrap();
    let cases = vec![case(&[("n", "3")]), case(&[("n", "10")])];
    let grid = FactorGrid::compile([("ma", "Mean($close, {n})")], cases)
        .unwrap()
        .with_warmup(true);
    let wide = grid.evaluate(&inputs).unwrap();
    let wide = wide.get("ma").unwrap();
    for (col, blank) in [(0, 2), (1, 2), (2, 9), (3, 9)] {
        let column = wide.column(col);
        assert!(column[..blank].iter().all(|v| v.is_nan()), "column {col}");
        assert!(!column[blank..].iter().any(|v| v.is_nan()), "column {col}");
    }
}

#[test]
fn a_grid_without_parameters_is_one_case() {
    let grid = FactorGrid::compile([("c", "$close * 2")], vec![GridCase::new()]).unwrap();
    let close = [1.0, 2.0];
    let inputs = FactorInputs::new(2, 1)
        .unwrap()
        .with_field("close", &close)
        .unwrap();
    assert_eq!(
        grid.evaluate(&inputs).unwrap().get("c").unwrap().as_slice(),
        [2.0, 4.0]
    );
    assert!(FactorGrid::compile([("c", "$close")], Vec::new()).is_err());
}
