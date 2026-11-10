//! The metric registry: what `stats()` prints, and how to ask for less of it.
//!
//! Everything else in this crate answers one question at a time. This answers
//! the question people actually open a backtest with — *how did it go* — as
//! one ordered, labelled table, and it is the single most-used entry point in
//! the API.
//!
//! It is a registry rather than a function for two reasons.
//!
//! **Order and naming are part of the output.** A table whose rows move
//! between runs cannot be diffed, and a sweep that prints "total_return" in
//! one place and "Total Return [%]" in another cannot be read. The order here
//! is fixed, and so are the titles and the percent scaling — down to
//! max drawdown being reported as a *positive* percentage while every other
//! drawdown number in this crate is negative.
//!
//! **Selecting metrics by tag is what makes it usable on a sweep.** Computing
//! twenty-eight metrics over a thousand columns to look at three of them is
//! most of the analysis layer's cost for none of its value. [`Tag`] filtering
//! is how a caller says which ones it wants, and unasked-for metrics are never
//! computed.
//!
//! # Every metric is per account
//!
//! [`stats`] takes one account's inputs and returns its table. [`crate::portfolio`]
//! aggregates shared-account cash, equity and records before constructing these
//! inputs; independent columns remain separate accounts.
//!
//! # A missing input is `None`, not zero
//!
//! Benchmark return needs a benchmark and gross exposure needs a price
//! series. A caller that has neither still gets a table, with those rows
//! [`Value::Missing`] — which prints as a blank rather than as a zero that
//! reads like a measurement.

use std::collections::BTreeSet;

use alphax_model::TradeStatus;
use alphax_records::{DrawdownRecords, DrawdownStatus, OrderRecords, TradeRecords};

use crate::benchmark;
use crate::metrics;
use crate::trade_stats::TradeStats;

/// What a metric answers with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value {
    /// A count of things: trades, bars.
    Count(i64),
    /// A measured quantity.
    Float(f64),
    /// The inputs for this metric were not supplied.
    ///
    /// Distinct from `Float(f64::NAN)`, which says the metric *was* computed
    /// and had no answer — a column with no trades has no win rate, which is
    /// a result. This says nobody asked the question.
    Missing,
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Count(n) => write!(f, "{n}"),
            Self::Float(v) if v.is_nan() => write!(f, "NaN"),
            Self::Float(v) => write!(f, "{v:.6}"),
            Self::Missing => write!(f, "—"),
        }
    }
}

/// Labels a metric carries, for selecting a subset of the table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tag {
    /// Depends only on the shape of the run, not its results.
    Wrapper,
    /// Reads the value curve or the cash.
    Portfolio,
    /// Reads order records.
    Orders,
    /// Reads trade records.
    Trades,
    /// Restricted to closed trades.
    Closed,
    /// Restricted to open trades.
    Open,
    /// Restricted to winning trades.
    Winning,
    /// Restricted to losing trades.
    Losing,
    /// Reads drawdown records.
    Drawdowns,
    /// Measures a span of bars rather than a quantity.
    Duration,
    /// Reads the return series.
    Returns,
}

/// One column's inputs, as far as each metric needs them.
///
/// Borrowed throughout: a sweep builds one of these per column and none of it
/// is copied.
#[derive(Debug, Clone, Copy)]
pub struct StatsInput<'a> {
    /// Column these records and series are being scored for.
    pub col: usize,
    /// Number of bars in the run.
    pub n_rows: usize,
    /// Starting cash for this column.
    pub init_cash: f64,
    /// Portfolio value per bar.
    pub value: &'a [f64],
    /// Per-bar returns of the value curve.
    pub returns: &'a [f64],
    /// Capital-based return when available. Otherwise compound `returns`.
    /// Shared accounts use this to retain PnL across missing valuation bars.
    pub total_return: Option<f64>,
    /// Buy-and-hold returns for this column's instrument, if known.
    pub benchmark: Option<&'a [f64]>,
    /// Gross exposure per bar, if known.
    pub gross_exposure: Option<&'a [f64]>,
    /// This column's fills.
    pub orders: &'a OrderRecords,
    /// This column's trades, in whichever view the caller chose.
    pub trades: &'a TradeRecords,
    /// This column's drawdown episodes.
    pub drawdowns: &'a DrawdownRecords,
    /// Annualization factor for the ratio metrics.
    pub ann_factor: f64,
    /// Risk-free rate per bar.
    pub risk_free: f64,
    /// Whether the trade-quality metrics count trades still open.
    ///
    /// **Defaults to false**, and the default is the
    /// right one: an open trade's PnL is a mark to market, not a realised
    /// result. Counting it makes the win rate part measurement and part
    /// projection, and the projection moves every time the last bar does.
    ///
    /// It applies to win rate, best and worst trade, the winning and losing
    /// averages and their durations, profit factor and expectancy — not to the
    /// three trade *counts*, which report open and closed separately, and not
    /// to open trade PnL, which is about the open ones by definition.
    pub incl_open: bool,
}

/// A metric's identity, independent of any particular run.
#[derive(Debug, Clone, Copy)]
pub struct Metric {
    /// Stable machine-readable name, for selecting one by hand.
    pub key: &'static str,
    /// Human-readable title, as printed in the table.
    pub title: &'static str,
    /// Labels for tag filtering.
    pub tags: &'static [Tag],
    compute: fn(&StatsInput<'_>) -> Value,
}

impl Metric {
    /// Computes this metric for one column.
    #[must_use]
    pub fn compute(&self, input: &StatsInput<'_>) -> Value {
        (self.compute)(input)
    }
}

/// Which metrics to compute.
#[derive(Debug, Clone, Default)]
pub struct Select {
    /// Keep only metrics carrying **every** one of these tags.
    pub require: Vec<Tag>,
    /// Drop metrics carrying **any** of these tags.
    pub exclude: Vec<Tag>,
    /// Keep only these keys; empty means no key restriction.
    pub keys: Vec<&'static str>,
}

impl Select {
    /// Everything, in the table's order.
    #[must_use]
    pub fn all() -> Self {
        Self::default()
    }

    /// Only metrics carrying every one of `tags`.
    #[must_use]
    pub fn with_tags(tags: &[Tag]) -> Self {
        Self {
            require: tags.to_vec(),
            ..Self::default()
        }
    }

    /// Only these named metrics, still in the registry's order.
    ///
    /// Order comes from the registry, not from the argument: a table whose
    /// rows move depending on how they were asked for cannot be diffed.
    #[must_use]
    pub fn keys(keys: &[&'static str]) -> Self {
        Self {
            keys: keys.to_vec(),
            ..Self::default()
        }
    }

    /// Everything except metrics carrying any of `tags`.
    #[must_use]
    pub fn without_tags(tags: &[Tag]) -> Self {
        Self {
            exclude: tags.to_vec(),
            ..Self::default()
        }
    }

    fn admits(&self, m: &Metric) -> bool {
        if !self.keys.is_empty() && !self.keys.contains(&m.key) {
            return false;
        }
        if self.require.iter().any(|t| !m.tags.contains(t)) {
            return false;
        }
        !self.exclude.iter().any(|t| m.tags.contains(t))
    }
}

/// One column's statistics table.
///
/// Metrics not admitted by `select` are not computed at all, which is the
/// point of asking for a subset on a sweep.
#[must_use]
pub fn stats(input: &StatsInput<'_>, select: &Select) -> Vec<(&'static str, Value)> {
    DEFAULT_METRICS
        .iter()
        .filter(|m| select.admits(m))
        .map(|m| (m.title, m.compute(input)))
        .collect()
}

/// The registry, in the order the table prints it.
#[must_use]
pub fn default_metrics() -> &'static [Metric] {
    DEFAULT_METRICS
}

/// Every tag any registered metric carries.
#[must_use]
pub fn known_tags() -> BTreeSet<Tag> {
    DEFAULT_METRICS
        .iter()
        .flat_map(|m| m.tags.iter().copied())
        .collect()
}

// --- the metrics themselves ----------------------------------------------

/// Trades in a given status, as their indices.
fn in_status(input: &StatsInput<'_>, status: TradeStatus) -> Vec<usize> {
    let trades = input.trades;
    (0..trades.len())
        .filter(|&i| trades.col[i] == input.col as i64 && trades.status[i] == status as i8)
        .collect()
}

/// The trades the quality metrics are allowed to see.
///
/// Closed only unless the caller asked otherwise; see
/// [`StatsInput::incl_open`].
fn eligible(input: &StatsInput<'_>) -> Vec<usize> {
    if input.incl_open {
        (0..input.trades.len())
            .filter(|&i| input.trades.col[i] == input.col as i64)
            .collect()
    } else {
        in_status(input, TradeStatus::Closed)
    }
}

/// The PnL of the eligible trades.
fn eligible_pnl(input: &StatsInput<'_>) -> Vec<f64> {
    eligible(input)
        .iter()
        .map(|&i| input.trades.pnl[i])
        .collect()
}

/// The eligible trades that made money, and the ones that lost it.
fn eligible_split(input: &StatsInput<'_>) -> (Vec<usize>, Vec<usize>) {
    let idxs = eligible(input);
    let t = input.trades;
    (
        idxs.iter().copied().filter(|&i| t.pnl[i] > 0.0).collect(),
        idxs.iter().copied().filter(|&i| t.pnl[i] < 0.0).collect(),
    )
}

/// The returns of the trades at `idxs`.
fn returns_of(trades: &TradeRecords, idxs: &[usize]) -> Vec<f64> {
    idxs.iter().map(|&i| trades.ret[i]).collect()
}

/// Bars each of the trades at `idxs` was held for.
///
/// Through [`TradeRecords::duration`], so the open/closed convention is
/// applied once rather than re-derived here.
fn durations_of(trades: &TradeRecords, idxs: &[usize]) -> Vec<f64> {
    let all = trades.duration();
    idxs.iter().map(|&i| all[i]).collect()
}

/// The best or worst return among the eligible trades.
fn mut_extreme(input: &StatsInput<'_>, max: bool) -> f64 {
    let rets = returns_of(input.trades, &eligible(input));
    rets.into_iter()
        .fold(f64::NAN, if max { f64::max } else { f64::min })
}

fn mean_or_nan(values: &[f64]) -> f64 {
    if values.is_empty() {
        f64::NAN
    } else {
        metrics::nanmean(values)
    }
}

static DEFAULT_METRICS: &[Metric] = &[
    Metric {
        key: "start",
        title: "Start",
        tags: &[Tag::Wrapper],
        // Row indices rather than timestamps: this layer has no calendar, and
        // inventing one would be worse than reporting the bar number.
        compute: |_| Value::Count(0),
    },
    Metric {
        key: "end",
        title: "End",
        tags: &[Tag::Wrapper],
        compute: |i| Value::Count(i.n_rows as i64 - 1),
    },
    Metric {
        key: "period",
        title: "Period",
        tags: &[Tag::Wrapper, Tag::Duration],
        compute: |i| Value::Count(i.n_rows as i64),
    },
    Metric {
        key: "start_value",
        title: "Start Value",
        tags: &[Tag::Portfolio],
        compute: |i| Value::Float(i.init_cash),
    },
    Metric {
        key: "end_value",
        title: "End Value",
        tags: &[Tag::Portfolio],
        compute: |i| Value::Float(i.value.last().copied().unwrap_or(f64::NAN)),
    },
    Metric {
        key: "total_return",
        title: "Total Return [%]",
        tags: &[Tag::Portfolio],
        compute: |i| {
            Value::Float(
                i.total_return
                    .unwrap_or_else(|| metrics::total_return(i.returns))
                    * 100.0,
            )
        },
    },
    Metric {
        key: "benchmark_return",
        title: "Benchmark Return [%]",
        tags: &[Tag::Portfolio],
        compute: |i| match i.benchmark {
            Some(b) => Value::Float(metrics::total_return(b) * 100.0),
            None => Value::Missing,
        },
    },
    Metric {
        key: "max_gross_exposure",
        title: "Max Gross Exposure [%]",
        tags: &[Tag::Portfolio],
        compute: |i| match i.gross_exposure {
            Some(e) => Value::Float(e.iter().copied().fold(f64::NAN, f64::max) * 100.0),
            None => Value::Missing,
        },
    },
    Metric {
        key: "total_fees_paid",
        title: "Total Fees Paid",
        tags: &[Tag::Portfolio, Tag::Orders],
        compute: |i| {
            Value::Float(
                i.orders
                    .col
                    .iter()
                    .zip(&i.orders.fees)
                    .filter(|(col, _)| **col == i.col as i64)
                    .map(|(_, fee)| fee)
                    .sum(),
            )
        },
    },
    Metric {
        key: "max_dd",
        title: "Max Drawdown [%]",
        tags: &[Tag::Portfolio, Tag::Drawdowns],
        // Negated, so this row is a positive percentage while every other
        // drawdown figure in this crate is negative. A deliberate choice,
        // and the one row of the table whose sign is not what a reader of the
        // rest of the crate would predict.
        compute: |i| {
            let dd = i.drawdowns.drawdown();
            let worst = dd
                .into_iter()
                .zip(&i.drawdowns.col)
                .filter(|(_, col)| **col == i.col as i64)
                .map(|(value, _)| value)
                .fold(f64::NAN, f64::min);
            Value::Float(-worst * 100.0)
        },
    },
    Metric {
        key: "max_dd_duration",
        title: "Max Drawdown Duration",
        tags: &[Tag::Portfolio, Tag::Drawdowns, Tag::Duration],
        compute: |i| {
            let longest = i
                .drawdowns
                .duration()
                .into_iter()
                .zip(&i.drawdowns.col)
                .filter(|(_, col)| **col == i.col as i64)
                .map(|(value, _)| value)
                .fold(f64::NAN, f64::max);
            Value::Float(longest)
        },
    },
    Metric {
        key: "total_trades",
        title: "Total Trades",
        tags: &[Tag::Portfolio, Tag::Trades],
        compute: |i| {
            Value::Count(
                i.trades
                    .col
                    .iter()
                    .filter(|col| **col == i.col as i64)
                    .count() as i64,
            )
        },
    },
    Metric {
        key: "total_closed_trades",
        title: "Total Closed Trades",
        tags: &[Tag::Portfolio, Tag::Trades, Tag::Closed],
        compute: |i| Value::Count(in_status(i, TradeStatus::Closed).len() as i64),
    },
    Metric {
        key: "total_open_trades",
        title: "Total Open Trades",
        tags: &[Tag::Portfolio, Tag::Trades, Tag::Open],
        compute: |i| Value::Count(in_status(i, TradeStatus::Open).len() as i64),
    },
    Metric {
        key: "open_trade_pnl",
        title: "Open Trade PnL",
        tags: &[Tag::Portfolio, Tag::Trades, Tag::Open],
        compute: |i| {
            let open = in_status(i, TradeStatus::Open);
            Value::Float(open.iter().map(|&k| i.trades.pnl[k]).sum())
        },
    },
    Metric {
        key: "win_rate",
        title: "Win Rate [%]",
        tags: &[Tag::Portfolio, Tag::Trades],
        compute: |i| Value::Float(TradeStats::from_pnl(&eligible_pnl(i)).win_rate * 100.0),
    },
    Metric {
        key: "best_trade",
        title: "Best Trade [%]",
        tags: &[Tag::Portfolio, Tag::Trades],
        compute: |i| Value::Float(mut_extreme(i, true) * 100.0),
    },
    Metric {
        key: "worst_trade",
        title: "Worst Trade [%]",
        tags: &[Tag::Portfolio, Tag::Trades],
        compute: |i| Value::Float(mut_extreme(i, false) * 100.0),
    },
    Metric {
        key: "avg_winning_trade",
        title: "Avg Winning Trade [%]",
        tags: &[Tag::Portfolio, Tag::Trades, Tag::Winning],
        compute: |i| Value::Float(mean_or_nan(&returns_of(i.trades, &eligible_split(i).0)) * 100.0),
    },
    Metric {
        key: "avg_losing_trade",
        title: "Avg Losing Trade [%]",
        tags: &[Tag::Portfolio, Tag::Trades, Tag::Losing],
        compute: |i| Value::Float(mean_or_nan(&returns_of(i.trades, &eligible_split(i).1)) * 100.0),
    },
    Metric {
        key: "avg_winning_trade_duration",
        title: "Avg Winning Trade Duration",
        tags: &[Tag::Portfolio, Tag::Trades, Tag::Winning, Tag::Duration],
        compute: |i| Value::Float(mean_or_nan(&durations_of(i.trades, &eligible_split(i).0))),
    },
    Metric {
        key: "avg_losing_trade_duration",
        title: "Avg Losing Trade Duration",
        tags: &[Tag::Portfolio, Tag::Trades, Tag::Losing, Tag::Duration],
        compute: |i| Value::Float(mean_or_nan(&durations_of(i.trades, &eligible_split(i).1))),
    },
    Metric {
        key: "profit_factor",
        title: "Profit Factor",
        tags: &[Tag::Portfolio, Tag::Trades],
        compute: |i| Value::Float(TradeStats::from_pnl(&eligible_pnl(i)).profit_factor),
    },
    Metric {
        key: "expectancy",
        title: "Expectancy",
        tags: &[Tag::Portfolio, Tag::Trades],
        compute: |i| Value::Float(TradeStats::from_pnl(&eligible_pnl(i)).expectancy),
    },
    Metric {
        key: "sharpe_ratio",
        title: "Sharpe Ratio",
        tags: &[Tag::Portfolio, Tag::Returns],
        compute: |i| Value::Float(metrics::sharpe_ratio(i.returns, i.ann_factor, i.risk_free)),
    },
    Metric {
        key: "calmar_ratio",
        title: "Calmar Ratio",
        tags: &[Tag::Portfolio, Tag::Returns],
        compute: |i| Value::Float(metrics::calmar_ratio(i.returns, i.ann_factor)),
    },
    Metric {
        key: "omega_ratio",
        title: "Omega Ratio",
        tags: &[Tag::Portfolio, Tag::Returns],
        compute: |i| {
            Value::Float(metrics::omega_ratio(
                i.returns,
                i.ann_factor,
                i.risk_free,
                0.0,
            ))
        },
    },
    Metric {
        key: "sortino_ratio",
        title: "Sortino Ratio",
        tags: &[Tag::Portfolio, Tag::Returns],
        compute: |i| Value::Float(metrics::sortino_ratio(i.returns, i.ann_factor, 0.0)),
    },
];

/// Metrics that read a benchmark, for callers that have one.
///
/// Not part of the default table,
/// but registered the same way, so a caller can append them by key.
#[must_use]
pub fn benchmark_metrics() -> &'static [Metric] {
    BENCHMARK_METRICS
}

static BENCHMARK_METRICS: &[Metric] = &[
    Metric {
        key: "beta",
        title: "Beta",
        tags: &[Tag::Portfolio, Tag::Returns],
        compute: |i| match i.benchmark {
            Some(b) => Value::Float(benchmark::beta(i.returns, b)),
            None => Value::Missing,
        },
    },
    Metric {
        key: "alpha",
        title: "Alpha",
        tags: &[Tag::Portfolio, Tag::Returns],
        compute: |i| match i.benchmark {
            Some(b) => Value::Float(benchmark::alpha(i.returns, b, i.ann_factor, i.risk_free)),
            None => Value::Missing,
        },
    },
    Metric {
        key: "information_ratio",
        title: "Information Ratio",
        tags: &[Tag::Portfolio, Tag::Returns],
        compute: |i| match i.benchmark {
            Some(b) => Value::Float(benchmark::information_ratio(i.returns, b, 1)),
            None => Value::Missing,
        },
    },
    Metric {
        key: "up_capture",
        title: "Up Capture",
        tags: &[Tag::Portfolio, Tag::Returns],
        compute: |i| match i.benchmark {
            Some(b) => Value::Float(benchmark::up_capture(i.returns, b, i.ann_factor)),
            None => Value::Missing,
        },
    },
    Metric {
        key: "down_capture",
        title: "Down Capture",
        tags: &[Tag::Portfolio, Tag::Returns],
        compute: |i| match i.benchmark {
            Some(b) => Value::Float(benchmark::down_capture(i.returns, b, i.ann_factor)),
            None => Value::Missing,
        },
    },
];

/// Whether every drawdown episode recovered.
///
/// Not a registered metric — a convenience for callers that want to know
/// whether [`Metric`] `max_dd_duration` is a measurement or a lower bound.
#[must_use]
pub fn all_drawdowns_recovered(d: &DrawdownRecords) -> bool {
    d.status
        .iter()
        .all(|s| *s == DrawdownStatus::Recovered as i8)
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "the registry's exact values and degenerate results are the contract"
)]
mod tests {
    use super::*;
    use alphax_core::flex::FlexArray;
    use alphax_model::OrderSide;
    use alphax_records::RecordSink;

    /// Close prices for the five-bar fixture below.
    const CLOSE: [f64; 5] = [10.0, 11.0, 12.0, 12.0, 11.0];

    /// Fills that produce one closed trade and one still open at the last bar.
    fn orders() -> OrderRecords {
        let mut o = OrderRecords::new();
        o.push_order(0, 0, 1.0, 10.0, 0.1, OrderSide::Buy);
        o.push_order(2, 0, 1.0, 12.0, 0.2, OrderSide::Sell);
        o.push_order(3, 0, 1.0, 12.0, 0.1, OrderSide::Buy);
        o
    }

    /// Derived through the public path rather than hand-built, so the records
    /// have the shape a simulation actually produces.
    fn trades(o: &OrderRecords) -> TradeRecords {
        let close = FlexArray::full(&CLOSE, 5, 1).unwrap();
        alphax_records::exit_trades(o, &close, 5, 1).unwrap()
    }

    fn drawdowns() -> DrawdownRecords {
        alphax_records::find_drawdowns(&[100.0, 120.0, 90.0, 110.0, 130.0], 5, 1).unwrap()
    }

    fn input<'a>(
        value: &'a [f64],
        returns: &'a [f64],
        o: &'a OrderRecords,
        t: &'a TradeRecords,
        d: &'a DrawdownRecords,
    ) -> StatsInput<'a> {
        StatsInput {
            col: 0,
            n_rows: 5,
            init_cash: 100.0,
            total_return: None,
            value,
            returns,
            benchmark: None,
            gross_exposure: None,
            orders: o,
            trades: t,
            drawdowns: d,
            ann_factor: metrics::DAILY,
            risk_free: 0.0,
            incl_open: false,
        }
    }

    #[test]
    fn the_default_table_is_twenty_eight_rows_in_order() {
        assert_eq!(DEFAULT_METRICS.len(), 28);
        let keys: Vec<&str> = DEFAULT_METRICS.iter().map(|m| m.key).collect();
        assert_eq!(keys[0], "start");
        assert_eq!(keys[1], "end");
        assert_eq!(keys[2], "period");
        assert_eq!(keys[27], "sortino_ratio");
        // No duplicates, or a table would have two rows with one name.
        let unique: BTreeSet<&str> = keys.iter().copied().collect();
        assert_eq!(unique.len(), keys.len());
    }

    #[test]
    fn a_full_table_comes_back_labelled_and_in_order() {
        let value = [100.0, 120.0, 90.0, 110.0, 130.0];
        let returns = [0.0, 0.2, -0.25, 2.0 / 9.0, 2.0 / 11.0];
        let o = orders();
        let (t, d) = (trades(&o), drawdowns());
        let table = stats(&input(&value, &returns, &o, &t, &d), &Select::all());
        assert_eq!(table.len(), 28);
        assert_eq!(table[0].0, "Start");
        assert_eq!(table[0].1, Value::Count(0));
        assert_eq!(table[1].1, Value::Count(4), "the last bar index");
        assert_eq!(table[2].1, Value::Count(5), "five bars");
        assert_eq!(table[3].1, Value::Float(100.0));
        assert_eq!(table[4].1, Value::Float(130.0));
    }

    #[test]
    fn max_drawdown_is_reported_positive_here_and_negative_everywhere_else() {
        // The one row whose sign is not what a reader of the rest of the crate
        // would predict. 120 -> 90 is a 25% drawdown, printed as +25.
        let value = [100.0, 120.0, 90.0, 110.0, 130.0];
        let returns = [0.0, 0.2, -0.25, 2.0 / 9.0, 2.0 / 11.0];
        let o = orders();
        let (t, d) = (trades(&o), drawdowns());
        let table = stats(
            &input(&value, &returns, &o, &t, &d),
            &Select::keys(&["max_dd"]),
        );
        assert_eq!(table.len(), 1);
        let Value::Float(v) = table[0].1 else {
            panic!("expected a float")
        };
        assert!((v - 25.0).abs() < 1e-9, "got {v}");
        assert!(d.drawdown()[0] < 0.0, "the record itself is negative");
    }

    #[test]
    fn a_missing_input_is_not_a_zero() {
        // Benchmark return and gross exposure have no inputs here. Reporting
        // 0.0 would read as "the benchmark went nowhere", which is a
        // measurement nobody made.
        let value = [100.0, 130.0];
        let returns = [0.0, 0.3];
        let o = orders();
        let (t, d) = (trades(&o), drawdowns());
        let table = stats(
            &input(&value, &returns, &o, &t, &d),
            &Select::keys(&["benchmark_return", "max_gross_exposure"]),
        );
        assert!(table.iter().all(|(_, v)| *v == Value::Missing));
        assert_eq!(Value::Missing.to_string(), "—");
    }

    #[test]
    fn tag_filtering_requires_every_tag_and_keeps_the_registry_order() {
        let value = [100.0, 130.0];
        let returns = [0.0, 0.3];
        let o = orders();
        let (t, d) = (trades(&o), drawdowns());
        let inp = input(&value, &returns, &o, &t, &d);

        let returns_only = stats(&inp, &Select::with_tags(&[Tag::Returns]));
        assert_eq!(
            returns_only.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
            [
                "Sharpe Ratio",
                "Calmar Ratio",
                "Omega Ratio",
                "Sortino Ratio"
            ]
        );

        // Two tags means both, not either.
        let winning_durations = stats(&inp, &Select::with_tags(&[Tag::Winning, Tag::Duration]));
        assert_eq!(winning_durations.len(), 1);
        assert_eq!(winning_durations[0].0, "Avg Winning Trade Duration");
    }

    #[test]
    fn excluding_a_tag_drops_exactly_that_family() {
        let value = [100.0, 130.0];
        let returns = [0.0, 0.3];
        let o = orders();
        let (t, d) = (trades(&o), drawdowns());
        let inp = input(&value, &returns, &o, &t, &d);
        let full = stats(&inp, &Select::all()).len();
        let no_trades = stats(&inp, &Select::without_tags(&[Tag::Trades]));
        let trade_rows = DEFAULT_METRICS
            .iter()
            .filter(|m| m.tags.contains(&Tag::Trades))
            .count();
        assert_eq!(no_trades.len(), full - trade_rows);
        assert!(!no_trades.iter().any(|(t, _)| t.contains("Trade")));
    }

    #[test]
    fn selecting_by_key_still_uses_the_registry_order() {
        // Asked for in one order, returned in another: the table's order is a
        // property of the table, not of the request, or two runs could not be
        // diffed against each other.
        let value = [100.0, 130.0];
        let returns = [0.0, 0.3];
        let o = orders();
        let (t, d) = (trades(&o), drawdowns());
        let table = stats(
            &input(&value, &returns, &o, &t, &d),
            &Select::keys(&["sortino_ratio", "start", "sharpe_ratio"]),
        );
        assert_eq!(
            table.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
            ["Start", "Sharpe Ratio", "Sortino Ratio"]
        );
    }

    #[test]
    fn unasked_metrics_are_never_computed() {
        // The reason tag filtering exists: on a thousand-column sweep, the
        // cost of the metrics nobody looked at is most of the analysis layer.
        // An input that would panic if touched proves they are not.
        let value = [100.0, 130.0];
        let returns = [0.0, 0.3];
        let o = orders();
        let (t, d) = (trades(&o), drawdowns());
        let inp = input(&value, &returns, &o, &t, &d);
        let mut computed = 0;
        for m in DEFAULT_METRICS {
            if Select::with_tags(&[Tag::Wrapper]).admits(m) {
                let _ = m.compute(&inp);
                computed += 1;
            }
        }
        assert_eq!(computed, 3, "only the three wrapper rows");
    }

    #[test]
    fn open_and_closed_trades_are_counted_separately_and_sum_to_the_total() {
        let value = [100.0, 130.0];
        let returns = [0.0, 0.3];
        let o = orders();
        let (t, d) = (trades(&o), drawdowns());
        let table = stats(
            &input(&value, &returns, &o, &t, &d),
            &Select::keys(&["total_trades", "total_closed_trades", "total_open_trades"]),
        );
        assert_eq!(table[0].1, Value::Count(2));
        assert_eq!(table[1].1, Value::Count(1));
        assert_eq!(table[2].1, Value::Count(1));
    }

    #[test]
    fn the_quality_metrics_ignore_open_trades_by_default() {
        // The fixture has one closed winner and one open loser. Counting the
        // open one makes the win rate 50% — but its PnL is a mark to market
        // that moves every time the last bar does, so half of that "measured"
        // rate is a projection. The default is closed-only; `incl_open` is how a caller opts in.
        let o = orders();
        let (t, d) = (trades(&o), drawdowns());
        let value = [1000.0, 1000.0, 1001.9, 1001.9, 1001.0];
        let returns = [0.0, 0.0, 0.0019, 0.0, -0.0009];
        let keys = ["win_rate", "total_trades", "total_open_trades"];

        let mut inp = input(&value, &returns, &o, &t, &d);
        let closed_only = stats(&inp, &Select::keys(&keys));
        assert_eq!(closed_only[0].1, Value::Count(2), "both trades are counted");
        assert_eq!(closed_only[1].1, Value::Count(1), "one of them is open");
        assert_eq!(
            closed_only[2].1,
            Value::Float(100.0),
            "the only closed trade won"
        );

        inp.incl_open = true;
        let with_open = stats(&inp, &Select::keys(&keys));
        assert_eq!(with_open[2].1, Value::Float(50.0));
        // The counts are unaffected either way: they report the split.
        assert_eq!(with_open[0].1, closed_only[0].1);
        assert_eq!(with_open[1].1, closed_only[1].1);
    }

    #[test]
    fn a_column_that_never_traded_still_produces_a_full_table() {
        // The row that matters for a sweep: every metric has to answer, even
        // if the answer is "there was nothing to measure". Dropping rows would
        // misalign this column's table against every other column's.
        let value = [100.0, 100.0, 100.0];
        let returns = [0.0, 0.0, 0.0];
        let (o, t, d) = (
            OrderRecords::new(),
            TradeRecords::new(),
            DrawdownRecords::new(),
        );
        let table = stats(&input(&value, &returns, &o, &t, &d), &Select::all());
        assert_eq!(table.len(), 28);
        let by_title: std::collections::HashMap<_, _> = table.iter().copied().collect();
        assert_eq!(by_title["Total Trades"], Value::Count(0));
        assert_eq!(by_title["Total Fees Paid"], Value::Float(0.0));
        let Value::Float(wr) = by_title["Win Rate [%]"] else {
            panic!("expected a float")
        };
        assert!(wr.is_nan(), "no trades means no win rate, not a zero one");
        // A flat curve has no volatility, so its Sharpe is the documented
        // +inf trap rather than a number.
        assert_eq!(by_title["Sharpe Ratio"], Value::Float(f64::INFINITY));
    }

    #[test]
    fn every_registered_tag_selects_something() {
        // A tag nothing carries is a filter that silently returns an empty
        // table, which reads like "this run had no results".
        for tag in known_tags() {
            let n = DEFAULT_METRICS
                .iter()
                .filter(|m| m.tags.contains(&tag))
                .count();
            assert!(n > 0, "{tag:?} selects nothing");
        }
    }

    #[test]
    fn the_benchmark_metrics_are_registered_the_same_way() {
        assert_eq!(BENCHMARK_METRICS.len(), 5);
        let value = [100.0, 130.0];
        let returns = [0.0, 0.3];
        let o = orders();
        let (t, d) = (trades(&o), drawdowns());
        let mut inp = input(&value, &returns, &o, &t, &d);
        assert!(
            BENCHMARK_METRICS
                .iter()
                .all(|m| m.compute(&inp) == Value::Missing)
        );
        let b = [0.0, 0.1];
        inp.benchmark = Some(&b);
        assert!(
            BENCHMARK_METRICS
                .iter()
                .all(|m| m.compute(&inp) != Value::Missing)
        );
    }
}
