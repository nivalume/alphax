use alphax::prelude::*;

fn matrix(columns: &[&[f64]]) -> Matrix {
    let n_rows = columns[0].len();
    Matrix::from_column_major(
        columns
            .iter()
            .flat_map(|column| column.iter().copied())
            .collect(),
        n_rows,
        columns.len(),
    )
    .unwrap()
}

#[test]
fn swept_columns_match_individual_runs_and_parallel_execution() -> Result<()> {
    let factor = matrix(&[&[-1.0, 1.0, 1.0, -1.0], &[-2.0, 2.0, 2.0, -2.0]]);
    let close = matrix(&[&[10.0, 11.0, 12.0, 13.0], &[20.0, 21.0, 22.0, 23.0]]);
    let symbols = [Label::from("BTC"), Label::from("ETH")];
    let cases = [(Label::from("zero"), 0.0), (Label::from("wide"), 1.5)];
    let sweep = SignalSweep::from_factor(
        ("momentum", &factor),
        &symbols,
        &cases,
        0,
        |score, threshold| (score > *threshold, score < -*threshold),
    )?;
    let swept_close = sweep.expand(&close)?;
    let args = FromSignals {
        size: FlexArray::scalar(1.0),
        use_stops: false,
        ..sweep.args(&swept_close)?
    };
    let serial = simulate_from_signals(&args)?;
    let parallel = simulate_from_signals_parallel(&args, 4)?;
    assert_eq!(serial, parallel);

    for (case_idx, case) in cases.iter().enumerate() {
        for symbol in 0..symbols.len() {
            let one_factor =
                Matrix::from_column_major(factor.column(symbol).to_vec(), factor.n_rows(), 1)?;
            let one_close =
                Matrix::from_column_major(close.column(symbol).to_vec(), close.n_rows(), 1)?;
            let one_symbols = [symbols[symbol].clone()];
            let one_cases = [case.clone()];
            let one = SignalSweep::from_factor(
                ("momentum", &one_factor),
                &one_symbols,
                &one_cases,
                0,
                |score, threshold| (score > *threshold, score < -*threshold),
            )?;
            let one_expanded = one.expand(&one_close)?;
            let one_args = FromSignals {
                size: FlexArray::scalar(1.0),
                use_stops: false,
                ..one.args(&one_expanded)?
            };
            let expected = simulate_from_signals(&one_args)?;
            let col = (case_idx * symbols.len() + symbol) as i64;
            let rows: Vec<usize> = serial
                .col
                .iter()
                .enumerate()
                .filter_map(|(row, &record_col)| (record_col == col).then_some(row))
                .collect();
            assert_eq!(
                rows.iter().map(|&row| serial.idx[row]).collect::<Vec<_>>(),
                expected.idx
            );
            assert_eq!(
                rows.iter().map(|&row| serial.size[row]).collect::<Vec<_>>(),
                expected.size
            );
            assert_eq!(
                rows.iter()
                    .map(|&row| serial.price[row])
                    .collect::<Vec<_>>(),
                expected.price
            );
            assert_eq!(
                rows.iter().map(|&row| serial.side[row]).collect::<Vec<_>>(),
                expected.side
            );
        }
    }
    Ok(())
}

#[test]
fn continuous_entry_condition_can_reenter_after_a_stop() -> Result<()> {
    let factor = matrix(&[&[1.0, 1.0, 1.0]]);
    let close = matrix(&[&[10.0, 8.0, 8.0]]);
    let symbols = [Label::from("BTC")];
    let cases = [(Label::from("always"), ())];
    let sweep = SignalSweep::from_factor(("score", &factor), &symbols, &cases, 0, |score, _| {
        (score > 0.0, false)
    })?;
    let swept_close = sweep.expand(&close)?;
    let args = FromSignals {
        size: FlexArray::scalar(1.0),
        sl_stop: FlexArray::scalar(0.1),
        ..sweep.args(&swept_close)?
    };
    let records = simulate_from_signals(&args)?;
    assert_eq!(records.idx, [0, 1, 2], "entry, stop, then re-entry");
    Ok(())
}
