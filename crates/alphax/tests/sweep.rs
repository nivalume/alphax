//! The end-to-end sweep, as assertions.
//!
//! `examples/sweep.rs` is the readable version; this is the one that fails
//! when something drifts. It asserts the structural properties rather than
//! the numbers, because the numbers describe a synthetic generator and
//! pinning them would only make the test brittle:
//!
//! - a sweep is one run of `params × symbols` columns, not `params` runs;
//! - the threaded run is bit-identical to the serial one;
//! - the labels lead back from a column to the parameters that made it;
//! - a column that never traded scores `+inf` and tops a naive ranking.
//!
//! That last one is a property worth pinning precisely because it is a trap.

use alphax::prelude::*;
use alphax_generic::elementwise::crossed_above_1d;

/// A deterministic pseudo-random walk, matching the example's generator.
struct Rng(u64);

impl Rng {
    const fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let bits = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (bits >> 11) as f64 / (1_u64 << 53) as f64 - 0.5
    }
}

fn prices(n_rows: usize, n_symbols: usize) -> Matrix {
    let mut data = Vec::with_capacity(n_rows * n_symbols);
    for s in 0..n_symbols {
        let mut rng = Rng::new(0x5DEE_CE66_D000_0000 ^ (s as u64 + 1).wrapping_mul(0x9E37_79B9));
        let mut level = 100.0;
        for _ in 0..n_rows {
            level *= 1.0 + rng.next() * 0.022;
            data.push(level);
        }
    }
    Matrix::from_column_major(data, n_rows, n_symbols).expect("price shape")
}

/// Everything the example does, returning what the assertions need.
struct Sweep {
    n_cols: usize,
    columns: Columns,
    orders: OrderRecords,
    sharpe: Vec<f64>,
    trade_counts: Vec<usize>,
    args_close: Vec<f64>,
}

fn run(n_rows: usize, fast: &[usize], slow: &[usize], n_symbols: usize) -> Result<Sweep> {
    let close = prices(n_rows, n_symbols);
    let labels: Vec<Label> = (0..n_symbols).map(Label::from).collect();

    let fp: Vec<MaParams> = fast
        .iter()
        .map(|&window| MaParams { window, ewm: false })
        .collect();
    let sp: Vec<MaParams> = slow
        .iter()
        .map(|&window| MaParams { window, ewm: false })
        .collect();
    let fast_ind = ma(&close, &fp, false, Some(&labels))?;
    let slow_ind = ma(&close, &sp, false, Some(&labels))?;
    let (fast_ma, slow_ma) = (fast_ind.get("ma")?, slow_ind.get("ma")?);
    let n_cols = fast_ma.n_cols();

    let mut entries = Vec::with_capacity(n_rows * n_cols);
    let mut exits = Vec::with_capacity(n_rows * n_cols);
    let mut swept_close = Vec::with_capacity(n_rows * n_cols);
    for col in 0..n_cols {
        entries.extend(crossed_above_1d(
            fast_ma.column(col),
            slow_ma.column(col),
            0,
        ));
        exits.extend(crossed_above_1d(
            slow_ma.column(col),
            fast_ma.column(col),
            0,
        ));
        swept_close.extend_from_slice(close.column(col % n_symbols));
    }

    let args = FromSignals {
        n_rows,
        n_cols,
        init_cash: FlexArray::scalar(10_000.0),
        close: FlexArray::full(&swept_close, n_rows, n_cols)?,
        entries: FlexArray::full(&entries, n_rows, n_cols)?,
        exits: FlexArray::full(&exits, n_rows, n_cols)?,
        direction: FlexArray::scalar(Direction::LongOnly),
        size: FlexArray::scalar(f64::INFINITY),
        fees: FlexArray::scalar(0.001),
        slippage: FlexArray::scalar(0.0005),
        sl_stop: FlexArray::scalar(0.15),
        ..FromSignals::default()
    };
    let orders = simulate_from_signals(&args)?;

    let init_cash = FlexArray::scalar(10_000.0);
    let close_flex = FlexArray::full(&swept_close, n_rows, n_cols)?;
    let returns =
        alphax::analysis::returns_from_orders(&orders, &close_flex, &init_cash, n_rows, n_cols)?;
    let sharpe = returns.map_columns(|c| sharpe_ratio(c, DAILY, 0.0));

    let trades = exit_trades(&orders, &close_flex, n_rows, n_cols)?;
    let stats = alphax::analysis::trade_stats::per_column(&trades, n_cols)?;
    let trade_counts = stats.iter().map(|s| s.count).collect();

    let fast_labels: Vec<Label> = fast.iter().map(|w| Label::from(*w)).collect();
    let slow_labels: Vec<Label> = slow.iter().map(|w| Label::from(*w)).collect();
    let columns = Columns::new(vec![
        Level::repeat("fast", &fast_labels, n_symbols),
        Level::repeat("slow", &slow_labels, n_symbols),
        Level::tile("symbol", &labels, fast.len()),
    ])?;

    Ok(Sweep {
        n_cols,
        columns,
        orders,
        sharpe,
        trade_counts,
        args_close: swept_close,
    })
}

#[test]
fn a_sweep_is_one_run_of_every_combination() {
    let fast = [5_usize, 10, 20];
    let slow = [20_usize, 50, 100];
    let s = run(800, &fast, &slow, 4).expect("the sweep runs");

    assert_eq!(s.n_cols, fast.len() * 4, "three pairs across four symbols");
    assert_eq!(s.columns.n_cols(), s.n_cols);
    assert!(!s.orders.is_empty(), "the workload must actually trade");

    // A column's labels lead back to the parameters that produced it.
    assert!(s.columns.describe(0).contains("fast=5"));
    assert!(s.columns.describe(0).contains("slow=20"));
    assert_eq!(s.columns.label(0, "symbol").unwrap(), &Label::from(0_usize));
    assert_eq!(s.columns.label(3, "symbol").unwrap(), &Label::from(3_usize));
    // The fourth column starts the second combination.
    assert!(s.columns.describe(4).contains("fast=10"));
}

#[test]
fn threading_does_not_change_the_answer() {
    let n_rows = 800;
    let fast = [5_usize, 10, 20];
    let slow = [20_usize, 50, 100];
    let s = run(n_rows, &fast, &slow, 4).expect("the sweep runs");

    // Rebuild the arguments and run them threaded; the records must be
    // bit-identical, not merely equivalent. Ids depend on column order, so a
    // difference here would mean a threading bug rather than a rounding one.
    let close = prices(n_rows, 4);
    let labels: Vec<Label> = (0..4_usize).map(Label::from).collect();
    let fp: Vec<MaParams> = fast
        .iter()
        .map(|&window| MaParams { window, ewm: false })
        .collect();
    let sp: Vec<MaParams> = slow
        .iter()
        .map(|&window| MaParams { window, ewm: false })
        .collect();
    let fast_ind = ma(&close, &fp, false, Some(&labels)).unwrap();
    let slow_ind = ma(&close, &sp, false, Some(&labels)).unwrap();
    let (fast_ma, slow_ma) = (fast_ind.get("ma").unwrap(), slow_ind.get("ma").unwrap());

    let mut entries = Vec::new();
    let mut exits = Vec::new();
    for col in 0..s.n_cols {
        entries.extend(crossed_above_1d(
            fast_ma.column(col),
            slow_ma.column(col),
            0,
        ));
        exits.extend(crossed_above_1d(
            slow_ma.column(col),
            fast_ma.column(col),
            0,
        ));
    }

    let args = FromSignals {
        n_rows,
        n_cols: s.n_cols,
        init_cash: FlexArray::scalar(10_000.0),
        close: FlexArray::full(&s.args_close, n_rows, s.n_cols).unwrap(),
        entries: FlexArray::full(&entries, n_rows, s.n_cols).unwrap(),
        exits: FlexArray::full(&exits, n_rows, s.n_cols).unwrap(),
        direction: FlexArray::scalar(Direction::LongOnly),
        size: FlexArray::scalar(f64::INFINITY),
        fees: FlexArray::scalar(0.001),
        slippage: FlexArray::scalar(0.0005),
        sl_stop: FlexArray::scalar(0.15),
        ..FromSignals::default()
    };

    for threads in [2, 3, 4, 8] {
        let parallel = simulate_from_signals_parallel(&args, threads).unwrap();
        assert_eq!(parallel, s.orders, "with {threads} threads");
    }
}

#[test]
fn a_column_that_never_traded_tops_a_naive_ranking() {
    // A slow window nearly as long as the sample cannot produce a crossover,
    // so that column's equity curve is flat: zero return and zero volatility.
    // Sharpe divides one by the other and answers `+inf`, which sorts first.
    //
    // This is the intended behaviour and it is pinned deliberately: a
    // sweep that ranks the raw vector finds the column that did nothing.
    let n_rows = 800;
    let fast = [5_usize, 10, 20];
    let slow = [20_usize, 50, 780];
    let s = run(n_rows, &fast, &slow, 4).expect("the sweep runs");

    let untraded: Vec<usize> = (0..s.n_cols).filter(|c| s.trade_counts[*c] == 0).collect();
    assert!(
        !untraded.is_empty(),
        "the over-long window should have produced an untraded column"
    );
    for &col in &untraded {
        assert!(
            s.sharpe[col].is_infinite() && s.sharpe[col].is_sign_positive(),
            "column {col} never traded but has sharpe {}",
            s.sharpe[col]
        );
    }

    let (naive_col, _, naive) = best_column(&s.columns, &s.sharpe).unwrap().unwrap();
    assert!(
        naive.is_infinite(),
        "the naive ranking picks an untraded column"
    );
    assert!(untraded.contains(&naive_col));

    // Masking the untraded columns to `NaN` is how a ranking says what it is
    // ranking: every reduction here skips `NaN` and none skips infinity.
    let traded: Vec<f64> = s
        .sharpe
        .iter()
        .zip(&s.trade_counts)
        .map(|(v, n)| if *n == 0 { f64::NAN } else { *v })
        .collect();
    let (col, _, value) = best_column(&s.columns, &traded).unwrap().unwrap();
    assert!(
        value.is_finite(),
        "the filtered ranking picks a real column"
    );
    assert!(s.trade_counts[col] > 0);

    // And the group means go from useless to usable for the same reason.
    let raw = group_by(&s.columns, &s.sharpe, "fast", Reduce::Mean).unwrap();
    assert!(
        raw.iter().any(|(_, v)| v.is_infinite()),
        "one infinite column makes its whole group infinite"
    );
    let filtered = group_by(&s.columns, &traded, "fast", Reduce::Mean).unwrap();
    assert!(filtered.iter().all(|(_, v)| v.is_finite() || v.is_nan()));
}
