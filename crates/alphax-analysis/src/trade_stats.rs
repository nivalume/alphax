//! Per-column statistics over trade records.
//!
//! Every function here reduces one column's trades to one number, so which
//! trade view was used decides what the number means. A win rate over
//! [`entry_trades`](alphax_records::entry_trades) answers "how often did an
//! entry work out"; over [`exit_trades`](alphax_records::exit_trades), "how
//! often did an exit book a profit". Neither is wrong and they are not
//! interchangeable.
//!
//! # A break-even trade is neither a win nor a loss
//!
//! Winners are `pnl > 0` and losers are `pnl < 0`, so a trade that came out
//! exactly flat is counted in the denominator and in neither bucket. That
//! makes exact zero load-bearing: this is the predicate the tolerant `add` in
//! the trade statistics exists to protect, and the reason a residue of 1e-14
//! there would show up here as a phantom win.

use alphax_core::error::Result;
use alphax_records::TradeRecords;

use crate::metrics::{nanmean, nanstd};

/// One column's trade statistics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TradeStats {
    /// Number of trades.
    pub count: usize,
    /// Trades with a positive PnL.
    pub winning: usize,
    /// Trades with a negative PnL.
    pub losing: usize,
    /// Winners over all trades; `NaN` when there were none.
    pub win_rate: f64,
    /// Gross profit over gross loss.
    pub profit_factor: f64,
    /// Expected PnL per trade, weighted by win rate.
    pub expectancy: f64,
    /// System Quality Number: `sqrt(count) * mean(pnl) / std(pnl)`.
    pub sqn: f64,
    /// Sum of every trade's PnL.
    pub total_pnl: f64,
    /// Best single trade.
    pub best: f64,
    /// Worst single trade.
    pub worst: f64,
}

impl TradeStats {
    /// Statistics over one column's PnL values.
    #[must_use]
    pub fn from_pnl(pnl: &[f64]) -> Self {
        let count = pnl.len();
        if count == 0 {
            return Self {
                count: 0,
                winning: 0,
                losing: 0,
                win_rate: f64::NAN,
                profit_factor: f64::NAN,
                expectancy: f64::NAN,
                sqn: f64::NAN,
                total_pnl: 0.0,
                best: f64::NAN,
                worst: f64::NAN,
            };
        }

        let wins: Vec<f64> = pnl.iter().copied().filter(|p| *p > 0.0).collect();
        let losses: Vec<f64> = pnl.iter().copied().filter(|p| *p < 0.0).collect();
        let total_win: f64 = wins.iter().sum();
        let total_loss: f64 = losses.iter().sum();
        let win_rate = wins.len() as f64 / count as f64;

        // A column of only wins has no losing trades to average. The
        // zero is substituted rather than letting the whole statistic
        // go missing, because "every trade won" is a result, not an absence
        // of one.
        let avg_win = if wins.is_empty() { 0.0 } else { nanmean(&wins) };
        let avg_loss = if losses.is_empty() {
            0.0
        } else {
            nanmean(&losses)
        };

        let std = nanstd(pnl, 1);
        Self {
            count,
            winning: wins.len(),
            losing: losses.len(),
            win_rate,
            profit_factor: total_win / total_loss.abs(),
            expectancy: win_rate * avg_win - (1.0 - win_rate) * avg_loss.abs(),
            sqn: (count as f64).sqrt() * nanmean(pnl) / std,
            total_pnl: pnl.iter().sum(),
            best: pnl.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            worst: pnl.iter().copied().fold(f64::INFINITY, f64::min),
        }
    }
}

/// Trade statistics for every column, in column order.
///
/// Columns with no trades still get an entry, so the result lines up with the
/// column labels and with every other per-column metric.
///
/// # Errors
///
/// Returns [`alphax_core::Error::Invariant`] if the records are not grouped by
/// column.
pub fn per_column(trades: &TradeRecords, n_cols: usize) -> Result<Vec<TradeStats>> {
    let map = trades.col_map(n_cols)?;
    let pnl = trades.mapped(&trades.pnl, &map)?;
    // The empty statistics are the fill: a column that traded nothing still
    // gets an entry, so the result lines up with the column labels.
    let mut out = vec![TradeStats::from_pnl(&[]); n_cols];
    pnl.for_each_column(|col, values| out[col] = TradeStats::from_pnl(values));
    Ok(out)
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "the degenerate results and exact rates are the contract here"
)]
mod tests {
    use super::*;

    #[test]
    fn win_rate_counts_winners_over_all_trades() {
        let s = TradeStats::from_pnl(&[1.0, -1.0, 2.0, -0.5]);
        assert_eq!(s.count, 4);
        assert_eq!(s.winning, 2);
        assert_eq!(s.losing, 2);
        assert_eq!(s.win_rate, 0.5);
    }

    #[test]
    fn a_break_even_trade_is_neither_a_win_nor_a_loss() {
        // It still counts in the denominator, so the win rate drops. This is
        // the predicate that makes the tolerant `add` in trade statistics
        // matter: a residue of 1e-14 here would read as a third winner.
        let s = TradeStats::from_pnl(&[1.0, 1.0, 0.0]);
        assert_eq!(s.count, 3);
        assert_eq!(s.winning, 2);
        assert_eq!(s.losing, 0);
        assert_eq!(s.win_rate, 2.0 / 3.0);
    }

    #[test]
    fn profit_factor_is_gross_profit_over_gross_loss() {
        let s = TradeStats::from_pnl(&[3.0, 1.0, -2.0]);
        assert_eq!(s.profit_factor, 2.0);
    }

    #[test]
    fn a_column_of_only_wins_has_an_infinite_profit_factor() {
        let s = TradeStats::from_pnl(&[1.0, 2.0]);
        assert_eq!(s.profit_factor, f64::INFINITY);
        // And its expectancy is its average win, with nothing subtracted.
        assert_eq!(s.expectancy, 1.5);
    }

    #[test]
    fn an_empty_column_has_no_statistics_but_still_has_an_entry() {
        let s = TradeStats::from_pnl(&[]);
        assert_eq!(s.count, 0);
        assert!(s.win_rate.is_nan());
        assert_eq!(s.total_pnl, 0.0);
    }

    #[test]
    fn expectancy_nets_the_average_win_against_the_average_loss() {
        // Half win 2, half lose 1: 0.5 * 2 - 0.5 * 1.
        let s = TradeStats::from_pnl(&[2.0, -1.0]);
        assert_eq!(s.expectancy, 0.5);
    }

    #[test]
    fn best_and_worst_bracket_the_column() {
        let s = TradeStats::from_pnl(&[1.0, -3.0, 2.0]);
        assert_eq!(s.best, 2.0);
        assert_eq!(s.worst, -3.0);
        assert_eq!(s.total_pnl, 0.0);
    }
}
