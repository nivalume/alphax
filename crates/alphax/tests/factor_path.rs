//! Factor expression -> delayed signal -> portfolio, end to end.

use alphax::prelude::*;
use alphax_analysis::Portfolio;
use alphax_expr::{FactorInputs, FactorSet};
use alphax_model::OhlcvPanel;

#[test]
fn a_factor_drives_the_existing_signal_simulator() -> Result<()> {
    let close = vec![10.0, 9.0, 8.0, 9.0, 10.0, 11.0, 10.0, 9.0, 8.0, 9.0];
    let panel = OhlcvPanel {
        close: close.clone(),
        n_rows: close.len(),
        n_cols: 1,
        ..OhlcvPanel::default()
    };
    let factors = FactorSet::compile([("trend", "$close / Mean($close, 3) - 1")])?;
    let values = factors.evaluate(&FactorInputs::from_ohlcv(&panel)?)?;
    let trend = values
        .get("trend")
        .expect("compiled factor is present")
        .column(0);

    // A close-derived factor becomes tradable on the next bar. Delaying here
    // keeps that timing decision in the strategy instead of hiding it in the
    // expression engine.
    let mut raw_entries = vec![false; close.len()];
    let mut raw_exits = vec![false; close.len()];
    for row in 1..close.len() {
        raw_entries[row] = trend[row - 1] > 0.05;
        raw_exits[row] = trend[row - 1] < -0.05;
    }
    let (entries, exits) = clean_enex(
        &raw_entries,
        &raw_exits,
        close.len(),
        1,
        FirstSignal::EntryFirst,
    )?;

    let args = FromSignals {
        n_rows: close.len(),
        n_cols: 1,
        close: FlexArray::full(&close, close.len(), 1)?,
        entries: FlexArray::full(&entries, close.len(), 1)?,
        exits: FlexArray::full(&exits, close.len(), 1)?,
        ..FromSignals::default()
    };
    let portfolio = Portfolio::from_signals(&args, 1)?;

    assert_eq!(portfolio.n_rows(), close.len());
    assert_eq!(portfolio.n_cols(), 1);
    assert!(
        portfolio.orders().len() >= 2,
        "factor never completed a trade"
    );
    Ok(())
}
