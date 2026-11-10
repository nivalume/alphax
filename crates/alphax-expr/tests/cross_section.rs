//! Cross-sectional operators and staged evaluation.

use alphax_expr::{FactorInputs, FactorSet};

const N: f64 = f64::NAN;

fn panel(n_rows: usize, n_cols: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n_rows * n_cols)
        .map(|i| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            if i % 23 == 7 {
                N
            } else {
                // Coarse values so ties happen.
                ((state >> 33) % 40) as f64 / 4.0 + 1.0
            }
        })
        .collect()
}

fn eval(
    sources: &[(&str, &str)],
    fields: &[(&str, &[f64])],
    shape: (usize, usize),
) -> Vec<Vec<f64>> {
    let mut inputs = FactorInputs::new(shape.0, shape.1).unwrap();
    for (name, values) in fields {
        inputs = inputs.with_field(name, values).unwrap();
    }
    FactorSet::compile(sources.iter().copied())
        .unwrap()
        .evaluate(&inputs)
        .unwrap()
        .into_matrices()
        .map(|(_, m)| m.into_vec())
        .collect()
}

fn bits(a: &[f64]) -> Vec<u64> {
    a.iter()
        .map(|v| if v.is_nan() { u64::MAX } else { v.to_bits() })
        .collect()
}

/// pandas' `rank(pct=True)` of one row, written the slow way.
#[allow(clippy::float_cmp, reason = "ties are exact equality, as in pandas")]
fn naive_rank(row: &[f64]) -> Vec<f64> {
    let present: Vec<f64> = row.iter().copied().filter(|v| !v.is_nan()).collect();
    let n = present.len() as f64;
    row.iter()
        .map(|&v| {
            if v.is_nan() {
                return N;
            }
            let below = present.iter().filter(|&&p| p < v).count() as f64;
            let equal = present.iter().filter(|&&p| p == v).count() as f64;
            (below + (equal + 1.0) / 2.0) / n
        })
        .collect()
}

fn rows_of(values: &[f64], n_rows: usize, n_cols: usize) -> Vec<Vec<f64>> {
    (0..n_rows)
        .map(|r| (0..n_cols).map(|c| values[c * n_rows + r]).collect())
        .collect()
}

#[test]
fn cs_rank_matches_a_row_by_row_rank() {
    let (n_rows, n_cols) = (40, 9);
    let close = panel(n_rows, n_cols, 1);
    let out = eval(
        &[("r", "CSRank($close)")],
        &[("close", &close)],
        (n_rows, n_cols),
    );
    for (r, row) in rows_of(&close, n_rows, n_cols).iter().enumerate() {
        let expected = naive_rank(row);
        for c in 0..n_cols {
            let got = out[0][c * n_rows + r];
            assert!(
                (got.is_nan() && expected[c].is_nan()) || (got - expected[c]).abs() < 1e-15,
                "row {r} col {c}: {got} vs {}",
                expected[c]
            );
        }
    }
}

#[test]
fn stages_compose_and_match_the_factors_computed_one_at_a_time() {
    let (n_rows, n_cols) = (60, 7);
    let close = panel(n_rows, n_cols, 2);
    let volume = panel(n_rows, n_cols, 3);
    let group: Vec<f64> = (0..n_rows * n_cols)
        .map(|i| ((i / n_rows) % 3) as f64)
        .collect();
    let fields: [(&str, &[f64]); 3] = [("close", &close), ("volume", &volume), ("group", &group)];
    let sources = [
        ("mixed", "CSRank(Mean($close, 3)) + Mean($close, 3)"),
        ("lagged", "Ref(CSRank($close), 1)"),
        ("twice", "CSZScore(CSRank($close) * $volume)"),
        ("rolled", "Corr(CSRank($close), CSRank($volume), 5)"),
        ("neutral", "IndNeutralize(Delta($close, 2), $group)"),
        ("resi", "CSResi($close, $volume)"),
        ("scaled", "Scale(Sign($close - Ref($close, 1)), 2)"),
        ("demean", "CSDemean(Ref($close, 2))"),
        ("bare", "$close"),
        ("literal", "1 + 1"),
        ("plain", "Mean($volume, 4)"),
    ];
    let together = eval(&sources, &fields, (n_rows, n_cols));
    for (k, source) in sources.iter().enumerate() {
        let alone = eval(&[*source], &fields, (n_rows, n_cols));
        assert_eq!(bits(&together[k]), bits(&alone[0]), "{}", source.0);
    }
    // Spot checks against hand computation.
    let rank_close: Vec<Vec<f64>> = rows_of(&close, n_rows, n_cols)
        .iter()
        .map(|row| naive_rank(row))
        .collect();
    for c in 0..n_cols {
        assert!(together[1][c * n_rows].is_nan());
        for r in 1..n_rows {
            let (got, want) = (together[1][c * n_rows + r], rank_close[r - 1][c]);
            assert!((got.is_nan() && want.is_nan()) || got.to_bits() == want.to_bits());
        }
    }
    // Scaled rows sum to 2 in absolute value where any value is present.
    for r in 1..n_rows {
        let total: f64 = (0..n_cols)
            .map(|c| together[6][c * n_rows + r])
            .filter(|v| !v.is_nan())
            .map(f64::abs)
            .sum();
        assert!(
            total == 0.0 || (total - 2.0).abs() < 1e-12 || total.is_nan(),
            "{total}"
        );
    }
    // Group neutralized values sum to zero inside each group on a row.
    for r in 2..n_rows {
        for g in 0..3 {
            let s: f64 = (0..n_cols)
                .filter(|c| c % 3 == g)
                .map(|c| together[4][c * n_rows + r])
                .filter(|v| !v.is_nan())
                .sum();
            assert!(s.abs() < 1e-9, "row {r} group {g}: {s}");
        }
    }
    assert_eq!(bits(&together[8]), bits(&close));
    assert!(together[9].iter().all(|v| v.to_bits() == 2.0_f64.to_bits()));
}

#[test]
fn thread_count_does_not_change_staged_results() {
    let (n_rows, n_cols) = (300, 13);
    let close = panel(n_rows, n_cols, 4);
    let volume = panel(n_rows, n_cols, 5);
    let fields: [(&str, &[f64]); 2] = [("close", &close), ("volume", &volume)];
    let sources = [
        ("a", "CSRank(Corr($close, $volume, 10))"),
        (
            "b",
            "-1 * Corr(CSRank(Delta(Log($volume), 2)), CSRank(($close - $close * 0.9) / $close), 6)",
        ),
        ("c", "CSZScore(Mean($close, 5)) - CSRank($volume)"),
    ];
    let run = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| eval(&sources, &fields, (n_rows, n_cols)))
    };
    let one = run(1);
    for threads in [2, 4, 7] {
        let many = run(threads);
        for k in 0..sources.len() {
            assert_eq!(
                bits(&one[k]),
                bits(&many[k]),
                "{threads} threads, factor {k}"
            );
        }
    }
}

#[test]
fn new_elementwise_operators() {
    let x = [-4.0, 0.0, 2.25, N];
    let m = [1.0, 0.0, 1.0, 1.0];
    let out = eval(
        &[
            ("exp", "Exp($x)"),
            ("sqrt", "Sqrt(Abs($x))"),
            ("clip", "Clip($x, -1, 2)"),
            ("fill", "FillNa($x, 7)"),
            ("isnan", "IsNan($x)"),
            ("mask", "Mask($x, $m)"),
            ("spow", "SignedPower($x, 2)"),
            ("decay", "DecayLinear($m, 3)"),
            ("prod", "Prod($x + 5, 2)"),
        ],
        &[("x", &x), ("m", &m)],
        (4, 1),
    );
    let same = |a: &[f64], b: &[f64]| {
        assert_eq!(bits(a), bits(b), "{a:?} vs {b:?}");
    };
    same(&out[0], &[(-4.0f64).exp(), 1.0, 2.25f64.exp(), N]);
    same(&out[1], &[2.0, 0.0, 1.5, N]);
    same(&out[2], &[-1.0, 0.0, 2.0, N]);
    same(&out[3], &[-4.0, 0.0, 2.25, 7.0]);
    same(&out[4], &[0.0, 0.0, 0.0, 1.0]);
    same(&out[5], &[-4.0, N, 2.25, N]);
    same(&out[6], &[-16.0, 0.0, 5.0625, N]);
    // Weights 1, 2, 3 newest heaviest: row 2 is (1·1 + 2·0 + 3·1) / 6.
    same(
        &out[7],
        &[1.0, 1.0 / 3.0, 4.0 / 6.0, (1.0 * 0.0 + 2.0 + 3.0) / 6.0],
    );
    same(&out[8], &[1.0, 5.0, 5.0 * 7.25, 7.25]);
}
