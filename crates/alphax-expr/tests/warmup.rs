//! Lookbacks and warm-up: a factor is blank on each symbol's rows before its
//! windows have filled, and a cross-section skips symbols still warming up.

use alphax_expr::{FactorInputs, FactorSet, FactorValues, alpha158, alpha360};

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// A panel whose column `c` starts (has values) from row `starts[c]`.
fn panel(n_rows: usize, starts: &[usize], seed: f64) -> Vec<f64> {
    let mut out = Vec::new();
    for (col, &start) in starts.iter().enumerate() {
        for row in 0..n_rows {
            out.push(if row < start {
                f64::NAN
            } else {
                100.0 + ((row as f64 * 0.37 + col as f64 * 1.9 + seed).sin() * 7.0)
            });
        }
    }
    out
}

#[test]
fn lookbacks_count_windows_and_lags() {
    let cases = [
        ("$close", 0),
        ("Mean($close, 20)", 19),
        ("Ref(Mean($close, 5), 2)", 6),
        ("Delta($close, 3)", 3),
        ("Corr(Ref($close, 1), $volume, 10)", 10),
        ("EMA($close, 20)", 19),
        ("EMA($close, 0.5)", 2),
        ("Mean($close, 0)", 0),
        ("Ref($close, -2)", 0),
        ("Max(Ref($close, 1), 3) + Std($close, 10)", 9),
        ("CSRank(Mean($close, 5))", 4),
        ("Ref(CSRank(Mean($close, 5)), 1)", 5),
        ("ChangeInstrument('X', Mean($close, 10))", 9),
        ("P($$roe_q)", 0),
        ("DayLast(Mean($close, 4))", 3),
    ];
    let set = FactorSet::compile(cases.iter().map(|(source, _)| (*source, *source))).unwrap();
    for ((source, want), got) in cases.iter().zip(set.lookbacks()) {
        assert_eq!(got, *want, "{source}");
    }
    assert_eq!(alpha158().unwrap().lookbacks().into_iter().max(), Some(60));
    assert_eq!(alpha360().unwrap().lookbacks().into_iter().max(), Some(59));
}

#[test]
fn warmup_blanks_each_symbols_first_rows_and_nothing_else() {
    let n_rows = 30;
    let close = panel(n_rows, &[0, 0, 8], 0.0);
    let volume = panel(n_rows, &[0, 3, 0], 1.0);
    let inputs = FactorInputs::new(n_rows, 3)
        .unwrap()
        .with_field("close", &close)
        .unwrap()
        .with_field("volume", &volume)
        .unwrap();
    let sources = [
        ("mean", "Mean($close, 5)"),
        ("lagged", "Ref($close, 2) + $volume"),
        ("corr", "Corr($close, $volume, 6)"),
        ("expanding", "Mean($volume, 0)"),
    ];
    let plain = FactorSet::compile(sources).unwrap();
    let warm = plain.clone().with_warmup(true);
    assert!(warm.warmup() && !plain.warmup());
    let lookbacks = warm.lookbacks();
    let raw = plain.evaluate(&inputs).unwrap();
    let pool = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
    };
    let blanked = pool(1).install(|| warm.evaluate(&inputs)).unwrap();
    let blanked4 = pool(4).install(|| warm.evaluate(&inputs)).unwrap();
    // Each symbol's inputs begin: close at 0, 0, 8 and volume at 0, 3, 0.
    let starts = |name: &str, col: usize| match name {
        "mean" => [0, 0, 8][col],
        "expanding" => [0, 3, 0][col],
        _ => [0, 3, 8][col],
    };
    for (k, (name, _)) in sources.iter().enumerate() {
        let (raw, warm) = (raw.get(name).unwrap(), blanked.get(name).unwrap());
        assert_eq!(
            bits(warm.as_slice()),
            bits(blanked4.get(name).unwrap().as_slice())
        );
        for col in 0..3 {
            let end = starts(name, col) + lookbacks[k];
            assert!(
                warm.column(col)[..end].iter().all(|v| v.is_nan()),
                "{name}[{col}]"
            );
            assert_eq!(
                bits(&warm.column(col)[end..]),
                bits(&raw.column(col)[end..]),
                "{name}[{col}]"
            );
        }
    }
    let mut batched = Vec::new();
    warm.evaluate_batches(&inputs, 1, |_, values| {
        batched.push(values);
        Ok(())
    })
    .unwrap();
    for (name, _) in sources {
        let joined: Vec<f64> = batched
            .iter()
            .flat_map(|v: &FactorValues| v.get(name).unwrap().as_slice().to_vec())
            .collect();
        assert_eq!(
            bits(&joined),
            bits(blanked.get(name).unwrap().as_slice()),
            "{name}"
        );
    }
}

#[test]
fn a_symbol_warming_up_is_left_out_of_cross_sections() {
    let n_rows = 20;
    let close = panel(n_rows, &[0, 0, 6, 0], 2.0);
    let inputs = FactorInputs::new(n_rows, 4)
        .unwrap()
        .with_field("close", &close)
        .unwrap();
    let warm = FactorSet::compile([("rank", "CSRank(Mean($close, 4))")])
        .unwrap()
        .with_warmup(true);
    let values = warm.evaluate(&inputs).unwrap();
    // The ranks a cross-section gives the warmed means.
    let means = FactorSet::compile([("mean", "Mean($close, 4)")])
        .unwrap()
        .with_warmup(true)
        .evaluate(&inputs)
        .unwrap();
    let means = means.get("mean").unwrap().as_slice().to_vec();
    let ranked = FactorSet::compile([("rank", "CSRank($mean)")])
        .unwrap()
        .evaluate(
            &FactorInputs::new(n_rows, 4)
                .unwrap()
                .with_field("mean", &means)
                .unwrap(),
        )
        .unwrap();
    assert_eq!(
        bits(values.get("rank").unwrap().as_slice()),
        bits(ranked.get("rank").unwrap().as_slice())
    );
    // Symbol 2 starts at row 6 and its mean fills at row 9: in between the
    // other three rank among themselves.
    let rank = values.get("rank").unwrap();
    assert!(rank.column(2)[..9].iter().all(|v| v.is_nan()));
    let row = 7;
    let mut present: Vec<f64> = [0, 1, 3].iter().map(|&c| rank.column(c)[row]).collect();
    present.sort_by(f64::total_cmp);
    assert_eq!(present, [1.0 / 3.0, 2.0 / 3.0, 1.0]);
    // Without warm-up the partial mean of symbol 2 is ranked too.
    let plain = FactorSet::compile([("rank", "CSRank(Mean($close, 4))")])
        .unwrap()
        .evaluate(&inputs)
        .unwrap();
    assert!(!plain.get("rank").unwrap().column(2)[row].is_nan());
}
