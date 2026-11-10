//! The result of a run, and everything derivable from it.
//!
//! Before this type, a caller who wanted a statistics table had to assemble
//! the rebuild chain by hand: assets, then cash flow, then cash, then asset
//! value, then value, then returns, then trades, then drawdowns, then one
//! `StatsInput` per column. That was eighty-odd lines, it appeared verbatim in
//! three places, and nothing checked the order or the arguments — passing the
//! order price where the close belongs produces a plausible equity curve that
//! is wrong.
//!
//! # Only four things are stored
//!
//! Order records, the close prices, the starting cash and the shape.
//! Everything else is derived on demand and cached. For a sweep that is the
//! difference between holding one array per column and holding a dozen, and
//! most columns in a sweep are never asked for more than a single scalar.
//!
//! The cache is [`OnceLock`], not a `RefCell`, so a `Portfolio` stays `Sync`
//! and several threads can compute different columns' metrics from one
//! portfolio. That matters for the same reason the simulator's state is
//! thread-local: the whole design is built on being able to fan out.
//!
//! # What it borrows
//!
//! Prices are borrowed, not copied. A sweep's close matrix is the largest
//! thing in the system and there is no reason for the result of a run to own
//! a second copy of it.
//!
//! # Account groups
//!
//! Positions and trades remain per symbol. Cash, value and return metrics
//! follow cash-sharing groups, with initial capital counted once per group.

use std::borrow::Cow;
use std::ops::Deref;
use std::sync::OnceLock;

use alphax_core::error::{Error, Result};
use alphax_core::flex::FlexArray;
use alphax_core::math::add;
use alphax_core::matrix::Matrix;
use alphax_model::TradesType;
use alphax_records::{DrawdownRecords, Excursions, OrderRecords, TradeRecords, find_drawdowns};
use alphax_sim::groups::Groups;

use crate::metrics::{self, DAILY};
use crate::series;
use crate::stats::{self, Select, StatsInput, Value};
use crate::trade_stats::TradeStats;

/// How a run is scored, separately from what it did.
///
/// Held apart from the records because the same run is often looked at more
/// than one way — closed trades or all of them, one annualization or another —
/// and re-running the simulation to change a reporting choice would be absurd.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scoring {
    /// Bars per year, for the annualized metrics.
    ///
    /// [`DAILY`] is 365. For crypto that is right; for daily equity bars it
    /// should be about 252, and using the wrong one inflates or deflates every
    /// annualized figure by around 20%. The data boundary should derive it
    /// from the bar length.
    pub ann_factor: f64,
    /// Risk-free rate per bar.
    pub risk_free: f64,
    /// Which trade view the trade statistics use.
    ///
    /// Not a formatting choice: a win rate over entry trades answers "how
    /// often did an entry work out" and over exit trades "how often did an
    /// exit book a profit". Neither is wrong and they are different numbers.
    pub trades_type: TradesType,
    /// Whether the trade-quality metrics count trades still open.
    ///
    /// False by default: an open trade's PnL is a mark
    /// to market, not a realised result.
    pub incl_open: bool,
}

impl Default for Scoring {
    fn default() -> Self {
        Self {
            ann_factor: DAILY,
            risk_free: 0.0,
            trades_type: TradesType::ExitTrades,
            incl_open: false,
        }
    }
}

/// Everything derived from a run, computed once and kept.
///
/// Normally owned by the [`Portfolio`] that fills it. A caller that has to
/// rebuild a borrowing `Portfolio` per call — the Python layer, whose
/// object owns the prices the engine only borrows — keeps one of these
/// alongside and lends it through [`Portfolio::from_borrowed`], so a value
/// curve computed for one call is still there for the next.
#[derive(Debug, Default)]
pub struct Cache {
    assets: OnceLock<Matrix>,
    cash: OnceLock<Matrix>,
    free_cash: OnceLock<Matrix>,
    asset_value: OnceLock<Matrix>,
    column_asset_value: OnceLock<Matrix>,
    account_orders: OnceLock<OrderRecords>,
    account_trades: OnceLock<TradeRecords>,
    value: OnceLock<Matrix>,
    returns: OnceLock<Matrix>,
    deposits: OnceLock<Option<Matrix>>,
    gross_exposure: OnceLock<Matrix>,
    trades: OnceLock<TradeRecords>,
    drawdowns: OnceLock<DrawdownRecords>,
}

impl Cache {
    /// Drops every cached derivation, releasing the matrices it held.
    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

/// The cache a portfolio reads: its own, or one lent by whoever owns the run.
#[derive(Debug)]
enum CacheRef<'a> {
    /// Boxed so the enum is one pointer either way; a `Cache` is a dozen
    /// `OnceLock`s.
    Owned(Box<Cache>),
    Borrowed(&'a Cache),
}

impl Deref for CacheRef<'_> {
    type Target = Cache;
    fn deref(&self) -> &Cache {
        match self {
            Self::Owned(cache) => cache,
            Self::Borrowed(cache) => cache,
        }
    }
}

/// A completed run.
///
/// Built by [`Portfolio::from_orders`] or [`Portfolio::from_signals`], which
/// run the simulation, or by [`Portfolio::from_records`] when the records came
/// from somewhere else, such as a file.
#[derive(Debug)]
pub struct Portfolio<'a> {
    orders: Cow<'a, OrderRecords>,
    close: FlexArray<'a, f64>,
    n_rows: usize,
    n_cols: usize,
    groups: Groups,
    group_init_cash: Vec<f64>,
    /// Benchmark returns per account, if supplied.
    benchmark: Option<Cow<'a, Matrix>>,
    scoring: Scoring,
    cache: CacheRef<'a>,
}

impl<'a> Portfolio<'a> {
    /// Wraps records that were produced elsewhere.
    ///
    /// The engine-agnostic entry point: any engine that produces the same
    /// records gets the same analysis layer for free. That is
    /// the whole reason the record schema is the contract rather than the
    /// simulation loop.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invariant`] if a record names a column outside the
    /// shape, or is out of the column-major order the record layer
    /// guarantees.
    pub fn from_records(
        orders: OrderRecords,
        close: FlexArray<'a, f64>,
        init_cash: FlexArray<'a, f64>,
        n_rows: usize,
        n_cols: usize,
        scoring: Scoring,
    ) -> Result<Self> {
        Self::build(
            Cow::Owned(orders),
            close,
            init_cash,
            n_rows,
            n_cols,
            scoring,
            Groups::ungrouped(n_cols),
            CacheRef::Owned(Box::default()),
        )
    }

    /// Wraps group-major simulation fills for account-level analysis.
    ///
    /// Initial cash uses the simulator convention: the first column of each
    /// group supplies its balance. Records are sorted for per-symbol derivation;
    /// the simulation's original records/log identities are not modified.
    ///
    /// # Errors
    /// Rejects invalid grouping, record columns or rows.
    pub fn from_grouped_records(
        mut orders: OrderRecords,
        close: FlexArray<'a, f64>,
        init_cash: FlexArray<'a, f64>,
        n_rows: usize,
        n_cols: usize,
        scoring: Scoring,
        groups: Groups,
    ) -> Result<Self> {
        if groups.is_shared() {
            orders.sort_by_column();
        }
        Self::build(
            Cow::Owned(orders),
            close,
            init_cash,
            n_rows,
            n_cols,
            scoring,
            groups,
            CacheRef::Owned(Box::default()),
        )
    }

    /// Wraps records and a cache that outlive this portfolio.
    ///
    /// For an owner that has to rebuild a borrowing `Portfolio` per call: the
    /// records are not cloned, and whatever this call derives stays in
    /// `cache` for the next one. `orders` must already be in column-major
    /// order (see [`OrderRecords::sort_by_column`]); a shared-cash run's
    /// execution-order records are rejected rather than sorted, because the
    /// sort would have to be repeated per call.
    ///
    /// # Errors
    ///
    /// As [`Self::from_grouped_records`].
    #[allow(
        clippy::too_many_arguments,
        reason = "one argument per input the borrowing constructor cannot own"
    )]
    pub fn from_borrowed(
        orders: &'a OrderRecords,
        close: FlexArray<'a, f64>,
        init_cash: FlexArray<'a, f64>,
        n_rows: usize,
        n_cols: usize,
        scoring: Scoring,
        groups: Groups,
        cache: &'a Cache,
    ) -> Result<Self> {
        Self::build(
            Cow::Borrowed(orders),
            close,
            init_cash,
            n_rows,
            n_cols,
            scoring,
            groups,
            CacheRef::Borrowed(cache),
        )
    }

    #[allow(clippy::too_many_arguments, reason = "the constructors' shared body")]
    fn build(
        orders: Cow<'a, OrderRecords>,
        close: FlexArray<'a, f64>,
        init_cash: FlexArray<'a, f64>,
        n_rows: usize,
        n_cols: usize,
        scoring: Scoring,
        groups: Groups,
        cache: CacheRef<'a>,
    ) -> Result<Self> {
        let covered: usize = (0..groups.len()).map(|g| groups.len_of(g)).sum();
        if covered != n_cols {
            return Err(Error::Invariant(
                "grouping does not cover the price columns".into(),
            ));
        }
        // Checked once here rather than in every derivation: a record naming
        // a column that does not exist would otherwise surface as a panic
        // deep inside whichever series was asked for first.
        alphax_records::ColMap::build(&orders.col, n_cols)?;
        for (i, &row) in orders.idx.iter().enumerate() {
            if row < 0 || row as usize >= n_rows {
                return Err(Error::Invariant(format!(
                    "fill {i} is on row {row}, outside 0..{n_rows}"
                )));
            }
        }
        let cash = &orders.cash;
        for i in 0..cash.len() {
            let (col, row) = (cash.col[i], cash.idx[i]);
            if col < 0 || col as usize >= n_cols || row < 0 || row as usize >= n_rows {
                return Err(Error::Invariant(format!(
                    "cash flow {i} is at row {row}, column {col}, outside the \
                     {n_rows} x {n_cols} shape"
                )));
            }
        }
        let group_init_cash = (0..groups.len())
            .map(|g| init_cash.get(0, groups.range(g).start))
            .collect();
        Ok(Self {
            orders,
            close,
            n_rows,
            n_cols,
            groups,
            group_init_cash,
            benchmark: None,
            scoring,
            cache,
        })
    }

    /// Number of independent accounts.
    #[must_use]
    pub fn n_groups(&self) -> usize {
        self.groups.len()
    }

    /// The account membership of each symbol column.
    #[must_use]
    pub fn groups(&self) -> &Groups {
        &self.groups
    }

    fn account_init_cash(&self) -> FlexArray<'_, f64> {
        FlexArray::per_column(&self.group_init_cash, self.n_groups())
            .expect("one balance per group")
    }

    /// Sums per-symbol series without mixing independent accounts.
    fn sum_groups(&self, columns: &Matrix) -> Matrix {
        let mut out = Matrix::zeros(self.n_rows, self.n_groups());
        for group in 0..self.n_groups() {
            for col in self.groups.range(group) {
                for row in 0..self.n_rows {
                    out.column_mut(group)[row] += columns.get(row, col);
                }
            }
        }
        out
    }

    fn account_orders(&self) -> &OrderRecords {
        if !self.groups.is_shared() {
            return &self.orders;
        }
        self.cache.account_orders.get_or_init(|| {
            let mut records = OrderRecords::clone(&self.orders);
            for col in &mut records.col {
                *col = self.groups.of_column(*col as usize) as i64;
            }
            records
        })
    }

    fn account_trades(&self) -> Result<&TradeRecords> {
        if !self.groups.is_shared() {
            return self.trades();
        }
        if self.cache.account_trades.get().is_none() {
            // Pair fills per symbol first; only the finished trades are grouped.
            let mut records = self.trades()?.clone();
            for col in &mut records.col {
                *col = self.groups.of_column(*col as usize) as i64;
            }
            let _ = self.cache.account_trades.set(records);
        }
        Ok(self.cache.account_trades.get().expect("just set"))
    }

    /// Attaches buy-and-hold returns to compare against.
    ///
    /// One return series per account group, not per symbol. Shared accounts
    /// require an explicit benchmark; no allocation policy is inferred.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invariant`] if the shape does not match the run's.
    pub fn with_benchmark(mut self, benchmark: Matrix) -> Result<Self> {
        if benchmark.n_rows() != self.n_rows || benchmark.n_cols() != self.n_groups() {
            return Err(Error::Invariant(format!(
                "a benchmark of {}x{} against a run of {}x{}",
                benchmark.n_rows(),
                benchmark.n_cols(),
                self.n_rows,
                self.n_groups()
            )));
        }
        self.benchmark = Some(Cow::Owned(benchmark));
        Ok(self)
    }

    /// [`Self::with_benchmark`] over a benchmark someone else keeps, so a
    /// caller rebuilding the portfolio per call does not copy it each time.
    ///
    /// # Errors
    ///
    /// As [`Self::with_benchmark`].
    pub fn with_benchmark_ref(mut self, benchmark: &'a Matrix) -> Result<Self> {
        if benchmark.n_rows() != self.n_rows || benchmark.n_cols() != self.n_groups() {
            return Err(Error::Invariant(format!(
                "a benchmark of {}x{} against a run of {}x{}",
                benchmark.n_rows(),
                benchmark.n_cols(),
                self.n_rows,
                self.n_groups()
            )));
        }
        self.benchmark = Some(Cow::Borrowed(benchmark));
        Ok(self)
    }

    /// Buy-and-hold returns per column, derived from the close prices.
    ///
    /// The obvious benchmark for a long-only run, and the one a caller almost
    /// always means. Supply a different one through [`Self::with_benchmark`].
    #[must_use]
    pub fn buy_and_hold(&self) -> Matrix {
        let mut out = Matrix::zeros(self.n_rows, self.n_cols);
        for col in 0..self.n_cols {
            let returns: Vec<f64> = (1..self.n_rows)
                .map(|row| {
                    series::period_return(self.close.get(row - 1, col), self.close.get(row, col))
                })
                .collect();
            out.column_mut(col)[1..].copy_from_slice(&returns);
        }
        out
    }

    /// Runs a `from_orders` simulation and wraps the result.
    ///
    /// # Errors
    /// Propagates simulation and grouping failures.
    pub fn from_orders(
        args: &alphax_sim::from_orders::FromOrders<'a>,
        threads: usize,
    ) -> Result<Self> {
        let orders = if threads > 1 {
            alphax_sim::from_orders::simulate_from_orders_parallel(args, threads)?
        } else {
            alphax_sim::from_orders::simulate_from_orders(args)?
        };
        Self::from_grouped_records(
            orders,
            args.close,
            args.init_cash,
            args.n_rows,
            args.n_cols,
            Scoring::default(),
            args.groups.clone(),
        )
    }

    /// Runs a `from_signals` simulation and wraps the result.
    ///
    /// # Errors
    ///
    /// Propagates the simulation's failures.
    pub fn from_signals(
        args: &alphax_sim::from_signals::FromSignals<'a>,
        threads: usize,
    ) -> Result<Self> {
        let orders = if threads > 1 {
            alphax_sim::simulate_from_signals_parallel(args, threads)?
        } else {
            alphax_sim::simulate_from_signals(args)?
        };
        Self::from_grouped_records(
            orders,
            args.close,
            args.init_cash,
            args.n_rows,
            args.n_cols,
            Scoring::default(),
            args.resolved_groups()?,
        )
    }

    /// Number of bars.
    #[must_use]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of columns.
    #[must_use]
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// The fills.
    #[must_use]
    pub fn orders(&self) -> &OrderRecords {
        &self.orders
    }

    /// How this run is being scored.
    #[must_use]
    pub fn scoring(&self) -> Scoring {
        self.scoring
    }

    /// The same run, scored differently.
    ///
    /// Drops the cached derivations that depend on the scoring and keeps the
    /// ones that do not — the value curve does not change because the
    /// annualization factor did.
    #[must_use]
    pub fn with_scoring(mut self, scoring: Scoring) -> Self {
        // Only the trade view changes what is derived; the rest are read at
        // the point a metric is computed, so their caches stay valid.
        if scoring.trades_type != self.scoring.trades_type {
            match &mut self.cache {
                CacheRef::Owned(cache) => {
                    cache.trades = OnceLock::new();
                    cache.account_trades = OnceLock::new();
                }
                // A lent cache is someone else's to keep; start a private one.
                CacheRef::Borrowed(_) => self.cache = CacheRef::Owned(Box::default()),
            }
        }
        self.scoring = scoring;
        self
    }

    // --- the rebuild chain, in order --------------------------------------

    /// Position held at the end of each bar.
    #[must_use]
    pub fn assets(&self) -> &Matrix {
        self.cache
            .assets
            .get_or_init(|| series::assets(&self.orders, self.n_rows, self.n_cols))
    }

    /// Cash held at the end of each bar.
    ///
    /// Every fill is charged its whole price, so under leverage this is own
    /// cash less what the longs borrowed, and goes negative once more was
    /// borrowed than is held. That is what keeps [`Self::value`] equal to the
    /// account's equity without a separate debt series.
    #[must_use]
    pub fn cash(&self) -> &Matrix {
        self.cache.cash.get_or_init(|| {
            let flow = series::cash_flow(&self.orders, self.n_rows, self.n_cols);
            series::cash(&self.sum_groups(&flow), &self.account_init_cash())
        })
    }

    /// Cash net of the margin an open short reserves.
    ///
    /// Identical to [`Self::cash`] on a long-only run, and lower wherever
    /// something is short. A short reserves its proceeds plus `value /
    /// leverage`: twice its value without leverage.
    ///
    /// A margin column's position counts at its profit, as the simulation
    /// settles it, and reserves its initial margin: what it leaves free is
    /// its share of the equity less the margin posted.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invariant`] if the records are not grouped by column,
    /// which [`Self::from_records`] has already checked — so in practice this
    /// does not fail.
    pub fn free_cash(&self) -> Result<&Matrix> {
        if self.cache.free_cash.get().is_none() {
            let flow = series::free_cash_flow(&self.orders, self.n_rows, self.n_cols)?;
            let mut free = series::cash(&self.sum_groups(&flow), &self.account_init_cash());
            self.add_margin_positions(&mut free);
            let _ = self.cache.free_cash.set(free);
        }
        Ok(self.cache.free_cash.get().expect("just set"))
    }

    /// Adds each margin column's `position × close` to its account in `free`.
    ///
    /// A margin column's fills are charged their whole price, so its cash
    /// alone is its equity less the position's value, which the account never
    /// paid. Nothing to do on a run without margin columns.
    ///
    /// An isolated column's profit is not settled into the account's cash, so
    /// it is not free: what is added back is its cost basis, the signed `debt`
    /// its last fill left, which leaves the margin it posted reserved and its
    /// unrealized profit or loss out.
    fn add_margin_positions(&self, free: &mut Matrix) {
        if !self.orders.margin.contains(&true) {
            return;
        }
        let mut margin = vec![false; self.n_cols];
        let mut isolated = vec![false; self.n_cols];
        for (r, &on) in self.orders.margin.iter().enumerate() {
            let col = self.orders.col[r] as usize;
            margin[col] |= on;
            isolated[col] |= self.orders.isolated.get(r) == Some(&true);
        }
        let assets = self.assets();
        let multiplier = series::multipliers(&self.orders, self.n_cols);
        for col in (0..self.n_cols).filter(|&col| margin[col] && !isolated[col]) {
            let group = self.groups.of_column(col);
            let m = multiplier.as_ref().map_or(1.0, |m| m[col]);
            for row in 0..self.n_rows {
                let position = assets.get(row, col);
                if position != 0.0 {
                    let cell = &mut free.column_mut(group)[row];
                    *cell = add(*cell, position * (self.close.get(row, col) * m));
                }
            }
        }
        if isolated.contains(&true) {
            self.add_isolated_bases(free, &isolated);
        }
    }

    /// Adds each isolated column's signed cost basis to its account in `free`,
    /// carried from each fill to the next.
    fn add_isolated_bases(&self, free: &mut Matrix, isolated: &[bool]) {
        let orders = &self.orders;
        let assets = self.assets();
        let mut r = 0;
        while r < orders.len() {
            let col = orders.col[r] as usize;
            let end = (r..orders.len())
                .find(|&k| orders.col[k] as usize != col)
                .unwrap_or(orders.len());
            if isolated[col] {
                let group = self.groups.of_column(col);
                for k in r..end {
                    let from = orders.idx[k] as usize;
                    let to = if k + 1 < end {
                        orders.idx[k + 1] as usize
                    } else {
                        self.n_rows
                    };
                    for row in from..to {
                        let position = assets.get(row, col);
                        let owed = if position > 0.0 {
                            orders.debt[k]
                        } else if position < 0.0 {
                            -orders.debt[k]
                        } else {
                            0.0
                        };
                        if owed != 0.0 {
                            let cell = &mut free.column_mut(group)[row];
                            *cell = add(*cell, owed);
                        }
                    }
                }
            }
            r = end;
        }
    }

    /// The position's market value at each bar's close.
    #[must_use]
    pub fn asset_value(&self) -> &Matrix {
        if !self.groups.is_shared() {
            return self.column_asset_value();
        }
        self.cache
            .asset_value
            .get_or_init(|| self.sum_groups(self.column_asset_value()))
    }

    /// Signed marked-to-market value per symbol column.
    #[must_use]
    pub fn column_asset_value(&self) -> &Matrix {
        self.cache.column_asset_value.get_or_init(|| {
            let multiplier = series::multipliers(&self.orders, self.n_cols);
            let mut value = series::asset_value(self.assets(), &self.close, multiplier.as_deref());
            if self.groups.is_shared() {
                for col in 0..self.n_cols {
                    for row in 0..self.n_rows {
                        if self.assets().get(row, col) == 0.0 {
                            value.column_mut(col)[row] = 0.0;
                        }
                    }
                }
            }
            value
        })
    }

    /// Total account value at each bar: cash plus the position.
    #[must_use]
    pub fn value(&self) -> &Matrix {
        self.cache
            .value
            .get_or_init(|| series::value(self.cash(), self.asset_value()))
    }

    /// Per-bar return of the value curve, with deposits taken out.
    ///
    /// A deposit raises the value without being earned, so each bar's return
    /// is measured from the previous value plus that bar's deposit. A run
    /// without deposits gets the plain chain, bit for bit.
    #[must_use]
    pub fn returns(&self) -> &Matrix {
        self.cache.returns.get_or_init(|| {
            series::returns_with_deposits(
                self.value(),
                &self.account_init_cash(),
                self.deposits().as_ref(),
            )
        })
    }

    /// Deposits and withdrawals per bar and account, or `None` if there were
    /// none.
    #[must_use]
    pub fn deposits(&self) -> &Option<Matrix> {
        self.cache.deposits.get_or_init(|| {
            series::deposits(&self.orders, self.n_rows, self.n_cols)
                .map(|columns| self.sum_groups(&columns))
        })
    }

    /// The cash flows that were not fills, in column-major order.
    #[must_use]
    pub fn cash_records(&self) -> &alphax_records::CashRecords {
        &self.orders.cash
    }

    /// The cash flows of one kind per bar and column: the funding a
    /// perpetual paid or received, the interest a leveraged long paid.
    ///
    /// Per column, as booked, even in a shared account; not cached, since it
    /// is one pass over the cash records.
    #[must_use]
    pub fn cash_flow_by_kind(&self, kind: alphax_model::CashFlowKind) -> Matrix {
        series::cash_flow_by_kind(&self.orders, kind, self.n_rows, self.n_cols)
    }

    /// How much of the account is in the position, per bar.
    ///
    /// Through free cash, not total value — the two agree on a long-only run
    /// and part company on a short.
    ///
    /// # Errors
    ///
    /// As [`Self::free_cash`].
    pub fn gross_exposure(&self) -> Result<&Matrix> {
        if self.cache.gross_exposure.get().is_none() {
            let free = self.free_cash()?;
            let mut absolute = self.column_asset_value().clone();
            for col in 0..self.n_cols {
                for value in absolute.column_mut(col) {
                    *value = value.abs();
                }
            }
            let _ = self
                .cache
                .gross_exposure
                .set(series::gross_exposure(&self.sum_groups(&absolute), free));
        }
        Ok(self.cache.gross_exposure.get().expect("just set"))
    }

    // --- exposure against total account value ------------------------------
    //
    // Unlike [`Self::gross_exposure`], which divides by free cash plus the
    // position, everything in this section divides by the account's total
    // value ([`Self::value`]). That is the denominator a weight is quoted in:
    // a 0.5 allocation is half the account, whatever margin a short reserves.
    // On a long-only run the two families agree; with shorts they differ.

    /// Each symbol column's signed position value as a fraction of its
    /// account's total value.
    ///
    /// The weight a column actually carries, as opposed to the weight it was
    /// ordered at — drift between rebalances shows up here. Shorts are
    /// negative. Within an account the columns sum to `(value - cash) / value`.
    /// NaN wherever the account value is zero or NaN, since a weight of a
    /// worthless account means nothing.
    #[must_use]
    pub fn allocations(&self) -> Matrix {
        let column_value = self.column_asset_value();
        let value = self.value();
        let mut out = Matrix::zeros(self.n_rows, self.n_cols);
        for col in 0..self.n_cols {
            let group = self.groups.of_column(col);
            for row in 0..self.n_rows {
                out.column_mut(col)[row] = ratio(column_value.get(row, col), value.get(row, group));
            }
        }
        out
    }

    /// Sums the positive (`long`) or absolute negative position values per
    /// account and divides by the account's value.
    fn side_exposure(&self, long: bool) -> Matrix {
        let column_value = self.column_asset_value();
        let mut side = Matrix::zeros(self.n_rows, self.n_cols);
        for col in 0..self.n_cols {
            for (row, out) in side.column_mut(col).iter_mut().enumerate() {
                let v = column_value.get(row, col);
                *out = if long { v.max(0.0) } else { (-v).max(0.0) };
            }
        }
        let mut summed = self.sum_groups(&side);
        let value = self.value();
        for group in 0..self.n_groups() {
            for (row, out) in summed.column_mut(group).iter_mut().enumerate() {
                *out = ratio(*out, value.get(row, group));
            }
        }
        summed
    }

    /// Long position value per account as a fraction of its total value.
    ///
    /// Only the long legs: a hedged book can be 100% long and 100% short at
    /// once, and netting would hide both.
    #[must_use]
    pub fn long_exposure(&self) -> Matrix {
        self.side_exposure(true)
    }

    /// Absolute short position value per account as a fraction of its total
    /// value. Non-negative; see [`Self::long_exposure`].
    #[must_use]
    pub fn short_exposure(&self) -> Matrix {
        self.side_exposure(false)
    }

    /// Long minus short exposure per account: the directional bet.
    ///
    /// Zero for a dollar-neutral book, however large its legs.
    #[must_use]
    pub fn net_exposure(&self) -> Matrix {
        combine(&self.long_exposure(), &self.short_exposure(), |l, s| l - s)
    }

    /// Long plus short exposure per account: how much is at work.
    ///
    /// Named apart from [`Self::gross_exposure`] because the denominator
    /// differs: this divides by total account value, that one by free cash
    /// plus the position. Use this one to compare leverage across books.
    #[must_use]
    pub fn gross_long_short(&self) -> Matrix {
        combine(&self.long_exposure(), &self.short_exposure(), |l, s| l + s)
    }

    /// Fraction of bars on which each symbol column holds a non-zero
    /// position.
    ///
    /// A low coverage means a column's return is earned on few bars, so its
    /// per-bar statistics rest on a thin sample. NaN for a run with no bars.
    #[must_use]
    pub fn position_coverage(&self) -> Vec<f64> {
        #[allow(clippy::cast_precision_loss, reason = "bar counts are far below 2^52")]
        self.assets().map_columns(|column| {
            if column.is_empty() {
                return f64::NAN;
            }
            let held = column.iter().filter(|&&a| a != 0.0).count();
            held as f64 / column.len() as f64
        })
    }

    /// Trades, in the view [`Scoring::trades_type`] names.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invariant`] if a fill has a non-positive price, which
    /// the execution kernel does not produce but a foreign record source
    /// might.
    pub fn trades(&self) -> Result<&TradeRecords> {
        if self.cache.trades.get().is_none() {
            let derived = match self.scoring.trades_type {
                TradesType::EntryTrades => alphax_records::entry_trades(
                    &self.orders,
                    &self.close,
                    self.n_rows,
                    self.n_cols,
                )?,
                TradesType::ExitTrades => alphax_records::exit_trades(
                    &self.orders,
                    &self.close,
                    self.n_rows,
                    self.n_cols,
                )?,
                TradesType::Positions => {
                    let exits = alphax_records::exit_trades(
                        &self.orders,
                        &self.close,
                        self.n_rows,
                        self.n_cols,
                    )?;
                    alphax_records::positions(&exits, self.n_cols)?
                }
            };
            let _ = self.cache.trades.set(derived);
        }
        Ok(self.cache.trades.get().expect("just set"))
    }

    /// Drawdown episodes of the value curve.
    #[must_use]
    pub fn drawdowns(&self) -> &DrawdownRecords {
        self.cache.drawdowns.get_or_init(|| {
            // Money moved in or out is neither a loss nor a recovery, so with
            // deposits the episodes are found on the compounded returns.
            let compounded;
            let curve = if self.deposits().is_some() {
                compounded = series::compounded_value(self.returns(), &self.account_init_cash());
                &compounded
            } else {
                self.value()
            };
            find_drawdowns(curve.as_slice(), self.n_rows, self.n_groups())
                .expect("the value curve has the portfolio's shape")
        })
    }

    /// Maximum adverse and favorable excursion of every trade, in the order of
    /// [`Self::trades`].
    ///
    /// A portfolio keeps only its closes, so the bars' range is passed in:
    /// `high` and `low` laid out column for column like the close the
    /// portfolio was built with. `FlexArray::Scalar(f64::NAN)` measures that
    /// side on closes alone, which can only understate it. The window, the
    /// fallback to the close and the entry-bar caveat are
    /// [`alphax_records::excursions()`]'s.
    ///
    /// # Errors
    ///
    /// As [`Self::trades`] and [`alphax_records::excursions()`].
    pub fn excursions(
        &self,
        high: &FlexArray<'_, f64>,
        low: &FlexArray<'_, f64>,
    ) -> Result<Excursions> {
        alphax_records::excursions(self.trades()?, high, low, &self.close, self.n_rows)
    }

    /// Edge ratio per account: mean MFE over mean MAE of the account's trades,
    /// both as returns. See [`alphax_records::edge_ratio`].
    ///
    /// Per account, like [`Self::trade_stats`]. NaN for an account without
    /// trades, or whose trades never moved against their entry.
    ///
    /// # Errors
    ///
    /// As [`Self::excursions`].
    pub fn edge_ratio(
        &self,
        high: &FlexArray<'_, f64>,
        low: &FlexArray<'_, f64>,
    ) -> Result<Vec<f64>> {
        let ex = self.excursions(high, low)?;
        alphax_records::edge_ratio_by_column(&ex, &self.account_trades()?.col, self.n_groups())
    }

    /// MAE and MFE by side for account `account`: the mean of each, as
    /// returns, and the edge ratio, for its long trades and its short ones.
    /// NaN for a side without trades. The keys are the stats table's.
    ///
    /// # Errors
    ///
    /// As [`Self::excursions`], and an account out of range.
    pub fn excursion_stats(
        &self,
        high: &FlexArray<'_, f64>,
        low: &FlexArray<'_, f64>,
        account: usize,
    ) -> Result<Vec<(&'static str, f64)>> {
        if account >= self.n_groups() {
            return Err(Error::Invariant(format!(
                "account {account} out of range 0..{}",
                self.n_groups()
            )));
        }
        let ex = self.excursions(high, low)?;
        let trades = self.account_trades()?;
        let mut out = Vec::with_capacity(6);
        for (side, names) in [
            (0_i8, ["avg_mae_long", "avg_mfe_long", "edge_ratio_long"]),
            (1_i8, ["avg_mae_short", "avg_mfe_short", "edge_ratio_short"]),
        ] {
            let mine: Vec<usize> = (0..trades.col.len())
                .filter(|&i| trades.col[i] == account as i64 && trades.direction[i] == side)
                .collect();
            let mean = |v: &[f64]| {
                if mine.is_empty() {
                    f64::NAN
                } else {
                    mine.iter().map(|&i| v[i]).sum::<f64>() / mine.len() as f64
                }
            };
            let (mae, mfe) = (mean(&ex.mae_ret), mean(&ex.mfe_ret));
            let edge = if mae < 0.0 { mfe / -mae } else { f64::NAN };
            out.extend([(names[0], mae), (names[1], mfe), (names[2], edge)]);
        }
        Ok(out)
    }

    // --- per-account metrics ------------------------------------------------

    /// Applies a reduction to every column's returns.
    ///
    /// The escape hatch for a metric this type does not name.
    #[must_use]
    pub fn map_returns(&self, f: impl Fn(&[f64]) -> f64) -> Vec<f64> {
        self.returns().map_columns(f)
    }

    /// Total return per account.
    #[must_use]
    pub fn total_return(&self) -> Vec<f64> {
        if self.groups.is_shared() {
            return (0..self.n_groups())
                .map(|group| self.account_total_return(group))
                .collect();
        }
        self.map_returns(metrics::total_return)
    }

    fn account_total_return(&self, group: usize) -> f64 {
        // With deposits, end over start is not a return: compound the bars.
        if self.deposits().is_some() {
            return metrics::total_return(self.returns().column(group));
        }
        let end = self
            .value()
            .column(group)
            .last()
            .copied()
            .unwrap_or(f64::NAN);
        (end - self.group_init_cash[group]) / self.group_init_cash[group]
    }

    /// Annualized return per account.
    #[must_use]
    pub fn annualized_return(&self) -> Vec<f64> {
        self.map_returns(|r| metrics::annualized_return(r, self.scoring.ann_factor))
    }

    /// Sharpe ratio per account.
    ///
    /// A column that never traded has a flat curve — zero return over zero
    /// volatility — and answers `+inf`, which **sorts as the best column in a
    /// sweep**. [`Self::traded_only`] is how a ranking says what it is
    /// ranking.
    #[must_use]
    pub fn sharpe_ratio(&self) -> Vec<f64> {
        self.map_returns(|r| {
            metrics::sharpe_ratio(r, self.scoring.ann_factor, self.scoring.risk_free)
        })
    }

    /// Sortino ratio per account.
    #[must_use]
    pub fn sortino_ratio(&self) -> Vec<f64> {
        self.map_returns(|r| metrics::sortino_ratio(r, self.scoring.ann_factor, 0.0))
    }

    /// Calmar ratio per account.
    #[must_use]
    pub fn calmar_ratio(&self) -> Vec<f64> {
        self.map_returns(|r| metrics::calmar_ratio(r, self.scoring.ann_factor))
    }

    /// Worst drawdown per account, as a negative fraction.
    #[must_use]
    pub fn max_drawdown(&self) -> Vec<f64> {
        self.map_returns(metrics::max_drawdown)
    }

    /// Trade statistics per account.
    ///
    /// # Errors
    ///
    /// As [`Self::trades`].
    pub fn trade_stats(&self) -> Result<Vec<TradeStats>> {
        crate::trade_stats::per_column(self.account_trades()?, self.n_groups())
    }

    /// Number of trades per account.
    ///
    /// # Errors
    ///
    /// As [`Self::trades`].
    pub fn trade_counts(&self) -> Result<Vec<usize>> {
        let trades = self.account_trades()?;
        let map = trades.col_map(self.n_groups())?;
        Ok(map.lens().to_vec())
    }

    /// A metric with the untraded columns masked to `NaN`.
    ///
    /// The trap this exists for: a column whose signals never fired has zero
    /// return and zero volatility, so its Sharpe is `+inf` and it tops a naive
    /// ranking. Every reduction in [`crate::aggregate`] skips `NaN`, so
    /// masking is how a caller says "rank the columns that actually did
    /// something" — which is almost always what was meant.
    ///
    /// # Errors
    ///
    /// As [`Self::trades`], or [`Error::Invariant`] if `values` is not one per
    /// column.
    pub fn traded_only(&self, values: &[f64]) -> Result<Vec<f64>> {
        if values.len() != self.n_groups() {
            return Err(Error::Invariant(format!(
                "expected one value per account: got {} for {} groups",
                values.len(),
                self.n_groups()
            )));
        }
        let counts = self.trade_counts()?;
        Ok(values
            .iter()
            .zip(counts)
            .map(|(v, n)| if n == 0 { f64::NAN } else { *v })
            .collect())
    }

    // --- the statistics table ----------------------------------------------

    /// One account's inputs, for a caller that wants the registry directly.
    ///
    /// # Errors
    ///
    /// As [`Self::trades`], or [`Error::Invariant`] if `col` is out of range.
    pub fn stats_input(&self, col: usize) -> Result<StatsInput<'_>> {
        if col >= self.n_groups() {
            return Err(Error::Invariant(format!(
                "account {col} is outside 0..{}",
                self.n_groups()
            )));
        }
        Ok(StatsInput {
            col,
            n_rows: self.n_rows,
            init_cash: self.group_init_cash[col],
            value: self.value().column(col),
            returns: self.returns().column(col),
            total_return: self
                .groups
                .is_shared()
                .then(|| self.account_total_return(col)),
            benchmark: self.benchmark.as_deref().map(|b| b.column(col)),
            gross_exposure: self.gross_exposure().ok().map(|e| e.column(col)),
            orders: self.account_orders(),
            trades: self.account_trades()?,
            drawdowns: self.drawdowns(),
            ann_factor: self.scoring.ann_factor,
            risk_free: self.scoring.risk_free,
            incl_open: self.scoring.incl_open,
        })
    }

    /// One account's statistics table.
    ///
    /// # Errors
    ///
    /// As [`Self::stats_input`].
    pub fn stats(&self, col: usize, select: &Select) -> Result<Vec<(&'static str, Value)>> {
        Ok(stats::stats(&self.stats_input(col)?, select))
    }

    /// Every account's table, in column order.
    ///
    /// # Errors
    ///
    /// As [`Self::stats_input`].
    pub fn stats_all(&self, select: &Select) -> Result<Vec<Vec<(&'static str, Value)>>> {
        (0..self.n_groups())
            .map(|c| self.stats(c, select))
            .collect()
    }
}

/// `part / whole`, NaN when the whole is zero or NaN.
fn ratio(part: f64, whole: f64) -> f64 {
    if whole == 0.0 || whole.is_nan() {
        f64::NAN
    } else {
        part / whole
    }
}

/// Elementwise combination of two same-shaped matrices.
fn combine(a: &Matrix, b: &Matrix, f: impl Fn(f64, f64) -> f64) -> Matrix {
    let data = a
        .as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(&x, &y)| f(x, y))
        .collect();
    Matrix::from_column_major(data, a.n_rows(), a.n_cols()).expect("same shape in, same shape out")
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "the derived series of an exact workload have exact values"
)]
mod tests {
    use super::*;
    use alphax_model::OrderSide;
    use alphax_records::RecordSink;

    const CLOSE: [f64; 5] = [10.0, 11.0, 12.0, 11.0, 13.0];

    /// One closed round trip in column 0, nothing in column 1.
    fn records() -> OrderRecords {
        let mut o = OrderRecords::new();
        o.push_order(0, 0, 10.0, 10.0, 0.0, OrderSide::Buy);
        o.push_order(2, 0, 10.0, 12.0, 0.0, OrderSide::Sell);
        o
    }

    fn portfolio<'a>(close: &'a [f64], cash: &'a FlexArray<'a, f64>) -> Portfolio<'a> {
        Portfolio::from_records(
            records(),
            FlexArray::full(close, 5, 2).unwrap(),
            *cash,
            5,
            2,
            Scoring::default(),
        )
        .unwrap()
    }

    fn two_columns() -> Vec<f64> {
        let mut v = CLOSE.to_vec();
        v.extend_from_slice(&CLOSE);
        v
    }

    #[test]
    fn the_rebuild_chain_comes_out_in_one_call_each() {
        // The point of the type: what used to be eighty lines of ordered
        // assembly is now a method per quantity, and the order is the type's
        // problem rather than the caller's.
        let close = two_columns();
        let cash = FlexArray::scalar(1000.0);
        let pf = portfolio(&close, &cash);

        assert_eq!(pf.assets().column(0), [10.0, 10.0, 0.0, 0.0, 0.0]);
        assert_eq!(pf.cash().column(0), [900.0, 900.0, 1020.0, 1020.0, 1020.0]);
        assert_eq!(pf.asset_value().column(0), [100.0, 110.0, 0.0, 0.0, 0.0]);
        assert_eq!(
            pf.value().column(0),
            [1000.0, 1010.0, 1020.0, 1020.0, 1020.0]
        );
        // Compounded bar by bar rather than end over start, so it lands a
        // couple of ULPs off the exact figure — which is the
        // intended arithmetic, not a rounding to tidy up.
        assert!((pf.total_return()[0] - 0.02).abs() < 1e-15);
        // The untraded column never moved.
        assert_eq!(pf.value().column(1), [1000.0; 5]);
        assert_eq!(pf.total_return()[1], 0.0);
    }

    #[test]
    fn a_derivation_is_computed_once() {
        // Not a micro-optimization: for a sweep this is the difference
        // between holding one array per column and holding a dozen, and most
        // columns are never asked for more than a scalar.
        let close = two_columns();
        let cash = FlexArray::scalar(1000.0);
        let pf = portfolio(&close, &cash);
        let first = std::ptr::from_ref(pf.value());
        let second = std::ptr::from_ref(pf.value());
        assert_eq!(first, second, "the value curve was rebuilt");
        // And the chain below it is shared too.
        let _ = pf.returns();
        assert_eq!(std::ptr::from_ref(pf.value()), first);
    }

    #[test]
    fn the_stats_table_needs_no_assembly() {
        let close = two_columns();
        let cash = FlexArray::scalar(1000.0);
        let pf = portfolio(&close, &cash);
        let table = pf.stats(0, &Select::all()).unwrap();
        assert_eq!(table.len(), 28);
        let by_title: std::collections::HashMap<_, _> = table.into_iter().collect();
        assert_eq!(by_title["Start Value"], Value::Float(1000.0));
        assert_eq!(by_title["End Value"], Value::Float(1020.0));
        assert_eq!(by_title["Total Trades"], Value::Count(1));
        // Gross exposure needs free cash, which the portfolio derives itself
        // rather than making the caller supply it.
        assert!(matches!(
            by_title["Max Gross Exposure [%]"],
            Value::Float(_)
        ));
        assert_eq!(pf.stats_all(&Select::all()).unwrap().len(), 2);

        let untraded: std::collections::HashMap<_, _> = pf
            .stats(1, &Select::keys(&["total_fees_paid", "total_trades"]))
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(untraded["Total Fees Paid"], Value::Float(0.0));
        assert_eq!(untraded["Total Trades"], Value::Count(0));
    }

    #[test]
    fn the_untraded_column_is_the_trap_and_masking_is_the_answer() {
        // Column 1 never traded: flat curve, zero volatility, `+inf` Sharpe,
        // and it tops a naive ranking. This is the single easiest way to read
        // a sweep wrong, so the masking helper is a method rather than a note.
        let close = two_columns();
        let cash = FlexArray::scalar(1000.0);
        let pf = portfolio(&close, &cash);

        let raw = pf.sharpe_ratio();
        assert_eq!(raw[1], f64::INFINITY);
        assert!(
            raw[1] > raw[0],
            "the untraded column outranks the traded one"
        );

        let masked = pf.traded_only(&raw).unwrap();
        assert!(masked[1].is_nan());
        assert_eq!(masked[0], raw[0]);
        assert!(pf.traded_only(&[1.0]).is_err(), "width is checked");
    }

    #[test]
    fn free_cash_and_plain_cash_agree_while_nothing_is_short() {
        let close = two_columns();
        let cash = FlexArray::scalar(1000.0);
        let pf = portfolio(&close, &cash);
        assert_eq!(pf.cash().as_slice(), pf.free_cash().unwrap().as_slice());
    }

    #[test]
    fn the_trade_view_is_a_scoring_choice_not_a_rerun() {
        // A win rate over entry trades and over exit trades are different
        // numbers answering different questions. Changing which one is
        // reported must not mean simulating again.
        let close = two_columns();
        let cash = FlexArray::scalar(1000.0);
        let pf = portfolio(&close, &cash);
        assert_eq!(pf.trades().unwrap().len(), 1);

        let pf = pf.with_scoring(Scoring {
            trades_type: TradesType::Positions,
            ..Scoring::default()
        });
        assert_eq!(pf.scoring().trades_type, TradesType::Positions);
        assert_eq!(pf.trades().unwrap().len(), 1);
        // The records themselves are untouched: nothing was re-simulated.
        assert_eq!(pf.orders().len(), 2);
    }

    #[test]
    fn a_record_outside_the_shape_is_caught_at_construction() {
        // Otherwise it surfaces as a panic deep inside whichever series was
        // asked for first, which says nothing about where the bad record came
        // from.
        let close = two_columns();
        let mut bad = OrderRecords::new();
        bad.push_order(0, 5, 1.0, 10.0, 0.0, OrderSide::Buy);
        assert!(
            Portfolio::from_records(
                bad,
                FlexArray::full(&close, 5, 2).unwrap(),
                FlexArray::scalar(1000.0),
                5,
                2,
                Scoring::default(),
            )
            .is_err()
        );

        let mut late = OrderRecords::new();
        late.push_order(99, 0, 1.0, 10.0, 0.0, OrderSide::Buy);
        let e = Portfolio::from_records(
            late,
            FlexArray::full(&close, 5, 2).unwrap(),
            FlexArray::scalar(1000.0),
            5,
            2,
            Scoring::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("row 99"), "got {e}");
    }

    #[test]
    fn the_annualization_factor_is_a_setting_not_a_constant() {
        // 365 for crypto, about 252 for daily equity bars. Using the wrong one
        // moves every annualized figure by around 20%, silently.
        let close = two_columns();
        let cash = FlexArray::scalar(1000.0);
        let pf = portfolio(&close, &cash);
        let crypto = pf.annualized_return()[0];
        let pf = pf.with_scoring(Scoring {
            ann_factor: 252.0,
            ..Scoring::default()
        });
        let equity = pf.annualized_return()[0];
        assert!(crypto > equity, "{crypto} should exceed {equity}");
    }

    #[test]
    fn an_empty_run_still_answers() {
        // Every column reports, including the ones that did nothing: dropping
        // them would misalign the result against the column labels.
        let close = two_columns();
        let pf = Portfolio::from_records(
            OrderRecords::new(),
            FlexArray::full(&close, 5, 2).unwrap(),
            FlexArray::scalar(1000.0),
            5,
            2,
            Scoring::default(),
        )
        .unwrap();
        assert_eq!(pf.total_return(), [0.0, 0.0]);
        assert_eq!(pf.trade_counts().unwrap(), [0, 0]);
        assert_eq!(pf.stats_all(&Select::all()).unwrap().len(), 2);
        assert!(pf.traded_only(&pf.sharpe_ratio()).unwrap()[0].is_nan());
    }

    #[test]
    fn excursions_follow_the_trades_and_the_range_passed_in() {
        let close = two_columns();
        let cash = FlexArray::scalar(1000.0);
        let pf = portfolio(&close, &cash);
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        let high = FlexArray::full(&high, 5, 2).unwrap();
        let low = FlexArray::full(&low, 5, 2).unwrap();

        // The round trip in column 0 holds bars 0..=2 from 10: high 13, low 9.
        let ex = pf.excursions(&high, &low).unwrap();
        assert_eq!(ex.mfe, [3.0]);
        assert_eq!(ex.mae, [-1.0]);
        let edge = pf.edge_ratio(&high, &low).unwrap();
        assert!((edge[0] - 3.0).abs() < 1e-12);
        assert!(edge[1].is_nan(), "column 1 never traded");

        // On closes alone the trade never went below its entry.
        let none = FlexArray::scalar(f64::NAN);
        let ex = pf.excursions(&none, &none).unwrap();
        assert_eq!((ex.mfe[0], ex.mae[0]), (2.0, 0.0));
        assert!(pf.edge_ratio(&none, &none).unwrap()[0].is_nan());
    }

    #[test]
    fn asking_for_a_column_that_does_not_exist_is_an_error() {
        let close = two_columns();
        let cash = FlexArray::scalar(1000.0);
        let pf = portfolio(&close, &cash);
        assert!(pf.stats(2, &Select::all()).is_err());
    }
}
