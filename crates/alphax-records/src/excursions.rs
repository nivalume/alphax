//! Maximum adverse and favorable excursion per trade.
//!
//! A trade's PnL says where it ended; its excursions say how far it went
//! along the way. MFE is the best the trade ever looked, MAE the worst — both
//! measured from the entry price over the bars the trade was held. Together
//! they answer questions PnL cannot: whether stops are placed inside the noise,
//! and whether exits leave most of the move on the table.
//!
//! # Conventions
//!
//! - **Per unit of size.** Values are price distances, not money: multiply by
//!   `size` for the trade's own scale. The `*_ret` fields divide by the entry
//!   price instead, so trades at different price levels are comparable.
//! - **Signed.** MFE is never negative and MAE never positive, for both
//!   directions. A trade that never moved in its favor has an MFE of exactly
//!   zero rather than a negative one.
//! - **Window.** From the entry bar to the exit bar, both inclusive. An open
//!   trade runs to the last bar, `n_rows - 1`.
//! - **Missing bars.** Where `high` or `low` is NaN on a bar, that bar's
//!   `close` stands in for it, so close-only data still yields excursions.
//!   A bar where all three are NaN contributes nothing.
//!
//! # The entry bar
//!
//! **The entry bar's whole range is counted**, although the fill may have
//! happened at that bar's close, after its high and low were already printed.
//! There is no intrabar path to tell the two apart, so this errs on the side
//! of reporting a larger excursion than the trade may actually have lived
//! through — conservative for MAE, which is the side stop placement reads.
//! The exit bar is treated the same way, for the same reason.

use alphax_core::error::{Error, Result};
use alphax_core::flex::FlexArray;
use alphax_model::{TradeDirection, TradeStatus};

use crate::TradeRecords;

/// Excursions per trade, struct-of-arrays, in the order of the trades.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Excursions {
    /// Maximum adverse excursion, as a price distance; never positive.
    pub mae: Vec<f64>,
    /// Maximum favorable excursion, as a price distance; never negative.
    pub mfe: Vec<f64>,
    /// `mae` over the entry price.
    pub mae_ret: Vec<f64>,
    /// `mfe` over the entry price.
    pub mfe_ret: Vec<f64>,
}

/// A bar's value, with `close` standing in where `value` is missing.
#[inline]
fn or_close(value: f64, close: f64) -> f64 {
    if value.is_nan() { close } else { value }
}

/// Computes MAE and MFE for every trade.
///
/// `high`, `low` and `close` are read at the trade's own column. See the
/// module docs for the window, the NaN fallback and the entry-bar caveat.
///
/// # Errors
///
/// Returns [`Error::Invariant`] if a trade's entry or exit lies outside
/// `0..n_rows`, its exit precedes its entry, or its entry price is not
/// positive.
pub fn excursions(
    trades: &TradeRecords,
    high: &FlexArray<'_, f64>,
    low: &FlexArray<'_, f64>,
    close: &FlexArray<'_, f64>,
    n_rows: usize,
) -> Result<Excursions> {
    let n = trades.len();
    let mut out = Excursions {
        mae: Vec::with_capacity(n),
        mfe: Vec::with_capacity(n),
        mae_ret: Vec::with_capacity(n),
        mfe_ret: Vec::with_capacity(n),
    };
    let last = n_rows as i64 - 1;

    for i in 0..n {
        let col = usize::try_from(trades.col[i])
            .map_err(|_| Error::Invariant("found a trade with a negative column".into()))?;
        let entry = trades.entry_idx[i];
        // An open trade's recorded exit is usually the last bar already, but
        // the window is defined by status, not by whatever the producer wrote.
        let exit = if trades.status[i] == TradeStatus::Open as i8 {
            last
        } else {
            trades.exit_idx[i]
        };
        if entry < 0 || exit > last || exit < entry {
            return Err(Error::Invariant(format!(
                "trade {i} spans rows {entry}..={exit}, outside 0..{n_rows}"
            )));
        }
        let entry_price = trades.entry_price[i];
        if entry_price <= 0.0 || entry_price.is_nan() {
            return Err(Error::Invariant(
                "found a trade with entry price 0 or less".into(),
            ));
        }

        // Seeding both extremes with the entry price is what clamps MFE to
        // at least zero and MAE to at most zero. `f64::max`/`min` skip NaN,
        // so a bar with nothing usable leaves them where they were.
        let (mut hi, mut lo) = (entry_price, entry_price);
        for row in entry as usize..=exit as usize {
            let c = close.get(row, col);
            hi = hi.max(or_close(high.get(row, col), c));
            lo = lo.min(or_close(low.get(row, col), c));
        }

        let (mfe, mae) = if trades.direction[i] == TradeDirection::Short as i8 {
            (entry_price - lo, entry_price - hi)
        } else {
            (hi - entry_price, lo - entry_price)
        };
        out.mfe.push(mfe);
        out.mae.push(mae);
        out.mfe_ret.push(mfe / entry_price);
        out.mae_ret.push(mae / entry_price);
    }
    Ok(out)
}

/// Mean favorable excursion over mean adverse excursion, both as returns.
///
/// Above one, trades on average went further in their favor than against
/// it — an entry edge measured before any exit rule has had a say. NaN when
/// there are no trades or no trade ever moved against its entry.
#[must_use]
pub fn edge_ratio(ex: &Excursions) -> f64 {
    ratio_of_means(ex.mfe_ret.iter().copied(), ex.mae_ret.iter().copied())
}

/// [`edge_ratio`] per column: column `c` is scored over the trades whose
/// `cols` entry is `c`, in their order, so it equals [`edge_ratio`] over that
/// subset bit for bit. A column with no trades is NaN.
///
/// `cols` is the trades' own column, [`TradeRecords::col`], aligned with `ex`.
///
/// # Errors
///
/// Returns [`Error::Invariant`] if `cols` and `ex` differ in length or a
/// column lies outside `0..n_cols`.
pub fn edge_ratio_by_column(ex: &Excursions, cols: &[i64], n_cols: usize) -> Result<Vec<f64>> {
    if cols.len() != ex.mfe_ret.len() || cols.len() != ex.mae_ret.len() {
        return Err(Error::Invariant(format!(
            "{} trade columns for {} excursions",
            cols.len(),
            ex.mfe_ret.len()
        )));
    }
    let mut trades_of = vec![Vec::new(); n_cols];
    for (i, &col) in cols.iter().enumerate() {
        let c = usize::try_from(col)
            .ok()
            .filter(|&c| c < n_cols)
            .ok_or_else(|| {
                Error::Invariant(format!("trade {i} is in column {col}, outside 0..{n_cols}"))
            })?;
        trades_of[c].push(i);
    }
    Ok(trades_of
        .iter()
        .map(|trades| {
            ratio_of_means(
                trades.iter().map(|&i| ex.mfe_ret[i]),
                trades.iter().map(|&i| ex.mae_ret[i]),
            )
        })
        .collect())
}

/// Mean MFE over mean absolute MAE, the reduction both edge ratios share.
fn ratio_of_means(mfe_ret: impl Iterator<Item = f64>, mae_ret: impl Iterator<Item = f64>) -> f64 {
    let mut n = 0_usize;
    let sum_mfe: f64 = mfe_ret.inspect(|_| n += 1).sum();
    if n == 0 {
        return f64::NAN;
    }
    let mean_mfe = sum_mfe / n as f64;
    let mean_mae = mae_ret.map(f64::abs).sum::<f64>() / n as f64;
    if mean_mae <= 0.0 {
        return f64::NAN;
    }
    mean_mfe / mean_mae
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "every expected value is a difference of short binary fractions, \
              exact in f64"
)]
mod tests {
    use super::*;

    /// One trade per call, appended to `t`. Fields excursions does not read
    /// are left at zero.
    fn push(
        t: &mut TradeRecords,
        col: i64,
        entry: i64,
        exit: i64,
        price: f64,
        dir: TradeDirection,
        status: TradeStatus,
    ) {
        t.col.push(col);
        t.size.push(1.0);
        t.entry_idx.push(entry);
        t.entry_price.push(price);
        t.entry_fees.push(0.0);
        t.exit_idx.push(exit);
        t.exit_price.push(price);
        t.exit_fees.push(0.0);
        t.pnl.push(0.0);
        t.ret.push(0.0);
        t.direction.push(dir as i8);
        t.status.push(status as i8);
        t.parent_id.push(t.len() as i64 - 1);
    }

    const HIGH: [f64; 5] = [10.5, 12.0, 11.0, 13.0, 9.5];
    const LOW: [f64; 5] = [9.5, 9.0, 10.0, 11.5, 8.0];
    const CLOSE: [f64; 5] = [10.0, 11.0, 10.5, 12.0, 9.0];

    fn run(t: &TradeRecords, high: &[f64], low: &[f64], close: &[f64]) -> Excursions {
        let n = close.len();
        excursions(
            t,
            &FlexArray::per_row(high, n).unwrap(),
            &FlexArray::per_row(low, n).unwrap(),
            &FlexArray::per_row(close, n).unwrap(),
            n,
        )
        .unwrap()
    }

    #[test]
    fn a_long_measures_up_from_the_high_and_down_from_the_low() {
        let mut t = TradeRecords::new();
        push(
            &mut t,
            0,
            0,
            2,
            10.0,
            TradeDirection::Long,
            TradeStatus::Closed,
        );
        let ex = run(&t, &HIGH, &LOW, &CLOSE);
        // Bars 0..=2: max high 12, min low 9.
        assert_eq!(ex.mfe, [2.0]);
        assert_eq!(ex.mae, [-1.0]);
        assert_eq!(ex.mfe_ret, [0.2]);
        assert_eq!(ex.mae_ret, [-0.1]);
    }

    #[test]
    fn a_short_measures_down_from_the_low_and_up_from_the_high() {
        let mut t = TradeRecords::new();
        push(
            &mut t,
            0,
            1,
            2,
            11.0,
            TradeDirection::Short,
            TradeStatus::Closed,
        );
        let ex = run(&t, &HIGH, &LOW, &CLOSE);
        // Bars 1..=2: min low 9, max high 12.
        assert_eq!(ex.mfe, [2.0]);
        assert_eq!(ex.mae, [-1.0]);
    }

    #[test]
    fn an_open_trade_runs_to_the_last_bar() {
        let mut t = TradeRecords::new();
        // The recorded exit is deliberately stale: status decides the window.
        push(
            &mut t,
            0,
            2,
            2,
            10.0,
            TradeDirection::Long,
            TradeStatus::Open,
        );
        let ex = run(&t, &HIGH, &LOW, &CLOSE);
        // Bars 2..=4: max high 13, min low 8.
        assert_eq!(ex.mfe, [3.0]);
        assert_eq!(ex.mae, [-2.0]);
    }

    #[test]
    fn excursions_are_clamped_at_zero() {
        let mut t = TradeRecords::new();
        // A long entered above every high and below no low in its window.
        push(
            &mut t,
            0,
            2,
            2,
            11.5,
            TradeDirection::Long,
            TradeStatus::Closed,
        );
        // A short entered below every low in its window.
        push(
            &mut t,
            0,
            2,
            2,
            9.0,
            TradeDirection::Short,
            TradeStatus::Closed,
        );
        let ex = run(&t, &HIGH, &LOW, &CLOSE);
        assert_eq!(ex.mfe, [0.0, 0.0]);
        assert_eq!(ex.mae, [-1.5, -2.0]);
    }

    #[test]
    fn a_missing_high_or_low_falls_back_to_the_close() {
        let high = [10.5, f64::NAN, 11.0];
        let low = [9.5, f64::NAN, 10.0];
        let close = [10.0, 14.0, 10.5];
        let mut t = TradeRecords::new();
        push(
            &mut t,
            0,
            0,
            2,
            10.0,
            TradeDirection::Long,
            TradeStatus::Closed,
        );
        let ex = run(&t, &high, &low, &close);
        // Bar 1 contributes its close, 14, to both sides.
        assert_eq!(ex.mfe, [4.0]);
        assert_eq!(ex.mae, [-0.5]);
    }

    #[test]
    fn columns_are_read_at_the_trades_own_column() {
        // Two columns, column-major: column 1 is column 0 shifted up by 10.
        let high = [11.0, 12.0, 21.0, 22.0];
        let low = [9.0, 10.0, 19.0, 20.0];
        let close = [10.0, 11.0, 20.0, 21.0];
        let mut t = TradeRecords::new();
        push(
            &mut t,
            1,
            0,
            1,
            20.0,
            TradeDirection::Long,
            TradeStatus::Closed,
        );
        let ex = excursions(
            &t,
            &FlexArray::full(&high, 2, 2).unwrap(),
            &FlexArray::full(&low, 2, 2).unwrap(),
            &FlexArray::full(&close, 2, 2).unwrap(),
            2,
        )
        .unwrap();
        assert_eq!(ex.mfe, [2.0]);
        assert_eq!(ex.mae, [-1.0]);
    }

    #[test]
    fn a_trade_outside_the_rows_is_rejected() {
        let mut t = TradeRecords::new();
        push(
            &mut t,
            0,
            0,
            5,
            10.0,
            TradeDirection::Long,
            TradeStatus::Closed,
        );
        let c = FlexArray::scalar(10.0);
        assert!(matches!(
            excursions(&t, &c, &c, &c, 5),
            Err(Error::Invariant(_))
        ));
    }

    #[test]
    fn edge_ratio_is_mean_mfe_over_mean_mae() {
        let ex = Excursions {
            mae: vec![],
            mfe: vec![],
            mae_ret: vec![-0.25, -0.75],
            mfe_ret: vec![0.5, 1.5],
        };
        // Means 1.0 and 0.5.
        assert_eq!(edge_ratio(&ex), 2.0);
    }

    #[test]
    fn edge_ratio_is_nan_without_trades_or_without_adversity() {
        assert!(edge_ratio(&Excursions::default()).is_nan());
        let ex = Excursions {
            mae: vec![0.0],
            mfe: vec![1.0],
            mae_ret: vec![0.0],
            mfe_ret: vec![0.1],
        };
        assert!(edge_ratio(&ex).is_nan());
    }

    #[test]
    fn edge_ratio_by_column_scores_each_columns_own_trades() {
        // Columns interleaved on purpose: grouping must not assume sorted input.
        let ex = Excursions {
            mae: vec![],
            mfe: vec![],
            mae_ret: vec![-0.25, -0.1, -0.75, 0.0],
            mfe_ret: vec![0.5, 0.3, 1.5, 0.2],
        };
        let cols = [0, 2, 0, 2];
        let by_col = edge_ratio_by_column(&ex, &cols, 3).unwrap();
        let column_zero = Excursions {
            mae: vec![],
            mfe: vec![],
            mae_ret: vec![-0.25, -0.75],
            mfe_ret: vec![0.5, 1.5],
        };
        assert_eq!(by_col[0], edge_ratio(&column_zero));
        assert!(by_col[1].is_nan(), "no trades");
        // Means 0.25 and 0.05.
        assert!((by_col[2] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn edge_ratio_by_column_rejects_a_column_outside_the_run() {
        let ex = Excursions {
            mae: vec![],
            mfe: vec![],
            mae_ret: vec![-0.1],
            mfe_ret: vec![0.1],
        };
        assert!(edge_ratio_by_column(&ex, &[1], 1).is_err());
        assert!(edge_ratio_by_column(&ex, &[-1], 1).is_err());
        assert!(edge_ratio_by_column(&ex, &[0, 0], 1).is_err());
    }
}
