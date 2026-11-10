//! A factor evaluated once, then swept across signal rules and symbols.

use alphax::prelude::*;
use alphax_model::OhlcvPanel;

fn prices(n_rows: usize, n_symbols: usize) -> Matrix {
    let mut data = Vec::with_capacity(n_rows * n_symbols);
    for symbol in 0..n_symbols {
        let mut price = 100.0 + symbol as f64 * 20.0;
        for row in 0..n_rows {
            let cycle = ((row as f64 + symbol as f64 * 5.0) / 11.0).sin();
            price *= 1.0 + 0.001 + cycle * 0.012;
            data.push(price);
        }
    }
    Matrix::from_column_major(data, n_rows, n_symbols).expect("price shape")
}

fn main() -> Result<()> {
    let symbols = [Label::from("BTC"), Label::from("ETH"), Label::from("SOL")];
    let close = prices(240, symbols.len());
    let panel = OhlcvPanel {
        close: close.as_slice().to_vec(),
        n_rows: close.n_rows(),
        n_cols: close.n_cols(),
        ..OhlcvPanel::default()
    };

    let factors = FactorSet::compile([("momentum", "Mean($close / Ref($close, 20) - 1, 5)")])?;
    let values = factors.evaluate(&FactorInputs::from_ohlcv(&panel)?)?;
    let momentum = values
        .get("momentum")
        .expect("the compiled factor is present");

    let cases = [
        (Label::from("loose"), (0.02, 0.00)),
        (Label::from("strict"), (0.05, 0.01)),
    ];
    let sweep = SignalSweep::from_factor(
        ("momentum", momentum),
        &symbols,
        &cases,
        1,
        |score, &(entry, exit)| (score > entry, score < exit),
    )?;
    let swept_close = sweep.expand(&close)?;
    let args = FromSignals {
        init_cash: FlexArray::scalar(10_000.0),
        fees: FlexArray::scalar(0.001),
        ..sweep.args(&swept_close)?
    };
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let portfolio = Portfolio::from_signals(&args, threads)?;

    println!(
        "one factor, {} cases, {} symbols -> {} parallel columns, {} fills",
        cases.len(),
        symbols.len(),
        sweep.columns().n_cols(),
        portfolio.orders().len()
    );
    for (col, value) in portfolio.total_return().iter().enumerate() {
        println!(
            "  {:<48} {:>8.2}%",
            sweep.columns().describe(col),
            value * 100.0
        );
    }
    Ok(())
}
