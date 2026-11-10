//! Liquidations and isolated margin once a run is rebuilt from its records.
//!
//! A liquidation is a fill and a write-off is a cash record, so the rebuild
//! counts both unchanged. What is new is an isolated column's free cash,
//! which leaves out its unrealized profit. These check both by hand, and
//! check that the rebuilt equity is the simulation's on a run that
//! liquidates and writes off in separate and shared accounts.
use alphax_analysis::Scoring;
use alphax_analysis::portfolio::Portfolio;
use alphax_core::flex::FlexArray;
use alphax_model::{AccountType, CashFlowKind, Direction, MarginMode};
use alphax_sim::from_orders::FromOrders;
use alphax_sim::groups::Groups;

const N: f64 = f64::NAN;

fn close_to(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < 1e-9, "{actual:?} vs {expected:?}");
    }
}

fn perpetual<'a>(close: &'a [f64], size: &'a [f64], mode: MarginMode) -> FromOrders<'a> {
    FromOrders {
        n_rows: close.len(),
        n_cols: 1,
        init_cash: FlexArray::scalar(2000.),
        close: FlexArray::per_row(close, close.len()).unwrap(),
        size: FlexArray::per_row(size, close.len()).unwrap(),
        account_type: FlexArray::scalar(AccountType::Margin),
        margin_mode: FlexArray::scalar(mode),
        initial_margin: FlexArray::scalar(0.1),
        groups: Groups::ungrouped(1),
        ..FromOrders::default()
    }
}

#[test]
fn an_isolated_columns_profit_is_equity_and_not_free_cash() {
    // 100 long at 100 posts 1000 of 2000. Cross, the profit is free to use;
    // isolated, it stays with the position.
    let close = [100., 95., 105.];
    let size = [100., N, N];
    let isolated =
        Portfolio::from_orders(&perpetual(&close, &size, MarginMode::Isolated), 1).unwrap();
    close_to(isolated.value().column(0), &[2000., 1500., 2500.]);
    close_to(
        isolated.free_cash().unwrap().column(0),
        &[1000., 1000., 1000.],
    );
    assert_eq!(isolated.orders().isolated, [true]);
    let cross = Portfolio::from_orders(&perpetual(&close, &size, MarginMode::Cross), 1).unwrap();
    close_to(cross.value().column(0), &[2000., 1500., 2500.]);
    close_to(cross.free_cash().unwrap().column(0), &[1000., 500., 1500.]);
}

#[test]
fn a_written_off_account_has_lost_everything_and_no_more() {
    // 1000 on 10% margin: a gap to 80 loses 2000; the 1000 the account
    // cannot pay is written off and the value stops at zero.
    let open = [100., 80.];
    let close = [100., 80.];
    let size = [100., N];
    let args = FromOrders {
        init_cash: FlexArray::scalar(1000.),
        open: FlexArray::per_row(&open, 2).unwrap(),
        maintenance_margin: FlexArray::scalar(0.05),
        ..perpetual(&close, &size, MarginMode::Cross)
    };
    let pf = Portfolio::from_orders(&args, 1).unwrap();
    assert_eq!(pf.orders().liquidation, [false, true]);
    close_to(
        pf.cash_flow_by_kind(CashFlowKind::Liquidation).column(0),
        &[0., 1000.],
    );
    close_to(pf.value().column(0), &[1000., 0.]);
    close_to(pf.returns().column(0), &[0., -1.]);
    assert!(pf.deposits().is_none(), "a write-off is not a deposit");
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1_u64 << 53) as f64
    }
}

/// Checks the rebuilt equity against what the simulation valued each account
/// at after the bar's liquidations: the bar's orders fill at its close, so
/// they change the equity only by their fees, and its charges land after.
fn check(groups: &Groups, seed: u64, partial: bool) {
    let (n_rows, n_cols) = (120, 4);
    let mut rng = Lcg(seed);
    let (mut open, mut high, mut low, mut close) = (vec![], vec![], vec![], vec![]);
    for _ in 0..n_cols {
        let mut p = 20.0 + 10.0 * rng.next();
        for _ in 0..n_rows {
            // An occasional gap, past where a leveraged position can pay.
            let gap = if rng.next() < 0.1 { 0.3 } else { 0.01 };
            let o = p * (1.0 + gap * (rng.next() - 0.5) * 2.0);
            p = o * (1.0 + 0.06 * (rng.next() - 0.5));
            open.push(o);
            close.push(p);
            high.push(o.max(p) * (1.0 + 0.02 * rng.next()));
            low.push(o.min(p) * (1.0 - 0.02 * rng.next()));
        }
    }
    let size: Vec<f64> = (0..n_rows * n_cols)
        .map(|_| {
            let u = rng.next();
            if u < 0.6 { N } else { (u - 0.8) * 200.0 }
        })
        .collect();
    let types = [
        AccountType::Margin,
        AccountType::Cash,
        AccountType::Margin,
        AccountType::Margin,
    ];
    let modes = [
        MarginMode::Cross,
        MarginMode::Cross,
        MarginMode::Isolated,
        MarginMode::Cross,
    ];
    let args = FromOrders {
        n_rows,
        n_cols,
        open: FlexArray::full(&open, n_rows, n_cols).unwrap(),
        high: FlexArray::full(&high, n_rows, n_cols).unwrap(),
        low: FlexArray::full(&low, n_rows, n_cols).unwrap(),
        close: FlexArray::full(&close, n_rows, n_cols).unwrap(),
        size: FlexArray::full(&size, n_rows, n_cols).unwrap(),
        init_cash: FlexArray::scalar(300.),
        fees: FlexArray::scalar(0.001),
        direction: FlexArray::scalar(Direction::Both),
        account_type: FlexArray::per_column(&types, n_cols).unwrap(),
        margin_mode: FlexArray::per_column(&modes, n_cols).unwrap(),
        initial_margin: FlexArray::scalar(0.1),
        maintenance_margin: FlexArray::scalar(0.05),
        cash_maintenance_margin: FlexArray::scalar(if partial { 0.1 } else { f64::NAN }),
        margin_tier_notional: if partial {
            FlexArray::PerRow(&TIER_NOTIONAL)
        } else {
            FlexArray::scalar(N)
        },
        margin_tier_rate: if partial {
            FlexArray::PerRow(&TIER_RATE)
        } else {
            FlexArray::scalar(N)
        },
        liquidation_step: FlexArray::scalar(if partial { 0.5 } else { N }),
        leverage: FlexArray::scalar(if partial { 3.0 } else { 1.0 }),
        liquidation_fee: FlexArray::scalar(0.01),
        funding_rate: FlexArray::scalar(1e-4),
        cash_deposits: FlexArray::scalar(3.),
        log: FlexArray::scalar(true),
        groups: groups.clone(),
        ..FromOrders::default()
    };
    let journal = alphax_sim::simulate_from_orders_logged(&args).unwrap();
    assert!(
        journal.orders.liquidation.contains(&true),
        "nothing liquidated"
    );
    if partial {
        let o = &journal.orders;
        let liquidated = |r: usize| o.liquidation[r];
        assert!(
            (0..o.len()).any(|r| liquidated(r) && !o.margin[r]),
            "no cash column liquidated"
        );
        // A partial step leaves a position, so a column liquidates twice on
        // one bar somewhere.
        assert!(
            (1..o.len()).any(|r| liquidated(r)
                && liquidated(r - 1)
                && o.col[r] == o.col[r - 1]
                && o.idx[r] == o.idx[r - 1]),
            "no partial steps"
        );
    }
    assert!(
        journal
            .orders
            .cash
            .kind
            .contains(&(CashFlowKind::Liquidation as i8)),
        "nothing written off"
    );
    let pf = Portfolio::from_grouped_records(
        journal.orders.clone(),
        args.close,
        args.init_cash,
        n_rows,
        n_cols,
        Scoring::default(),
        groups.clone(),
    )
    .unwrap();

    let n_groups = groups.len();
    let mut opened = vec![N; n_rows * n_groups];
    let mut fees = vec![0.0; n_rows * n_groups];
    let logs = &journal.logs;
    for r in 0..logs.len() {
        if logs.liquidation[r] {
            // Before the bar's orders: its fee and fill are in the value the
            // orders were sized against.
            continue;
        }
        let k = groups.of_column(logs.col[r] as usize) * n_rows + logs.idx[r] as usize;
        opened[k] = logs.value[r];
        if !logs.res_fees[r].is_nan() {
            fees[k] += logs.res_fees[r];
        }
    }
    let mut closing = vec![0.0; n_rows * n_groups];
    let cash = &journal.orders.cash;
    for i in 0..cash.len() {
        let kind = cash.kind[i];
        // The deposit and the write-off are booked before the orders.
        if kind == CashFlowKind::Deposit as i8 || kind == CashFlowKind::Liquidation as i8 {
            continue;
        }
        let k = groups.of_column(cash.col[i] as usize) * n_rows + cash.idx[i] as usize;
        closing[k] += cash.amount[i];
    }
    let value = pf.value();
    for g in 0..n_groups {
        for row in 0..n_rows {
            let k = g * n_rows + row;
            let expected = opened[k] - fees[k] + closing[k];
            let rebuilt = value.get(row, g);
            assert!(
                (rebuilt - expected).abs() < 1e-9 * expected.abs().max(1.0),
                "account {g}, bar {row}: rebuilt {rebuilt}, simulated {expected}"
            );
        }
    }
}

/// A tier table for the partial runs: 4% to a notional of 200, 8% beyond.
static TIER_NOTIONAL: [f64; 2] = [200.0, N];
static TIER_RATE: [f64; 2] = [0.04, 0.08];

#[test]
fn the_rebuilt_equity_is_the_simulations_in_separate_accounts() {
    check(&Groups::ungrouped(4), 3, false);
}

#[test]
fn the_rebuilt_equity_is_the_simulations_in_shared_accounts() {
    check(&Groups::from_lens(&[2, 2], 4).unwrap(), 17, false);
}

#[test]
fn the_rebuilt_equity_survives_partial_tiered_and_cash_liquidations() {
    // Partial steps, a tier table, leveraged cash columns with a
    // maintenance margin of their own.
    check(&Groups::ungrouped(4), 5, true);
    check(&Groups::from_lens(&[2, 2], 4).unwrap(), 23, true);
}

/// The free cash an isolated run rebuilds is the simulation's: at each
/// bar's first order, what the loop had free is the rebuilt free cash of
/// the bar before. Funding comes out of each column's margin, not the free
/// cash, and a column that is never settled moves it only by its fills.
fn check_isolated_free_cash(groups: &Groups, seed: u64) {
    let (n_rows, n_cols) = (120, 4);
    let mut rng = Lcg(seed);
    let close: Vec<f64> = (0..n_cols)
        .flat_map(|_| {
            let mut p = 50.0;
            (0..n_rows)
                .map(|_| {
                    p *= 1.0 + 0.04 * (rng.next() - 0.5);
                    p
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let size: Vec<f64> = (0..n_rows * n_cols)
        .map(|_| {
            let u = rng.next();
            if u < 0.7 { N } else { (u - 0.85) * 40.0 }
        })
        .collect();
    let args = FromOrders {
        n_rows,
        n_cols,
        close: FlexArray::full(&close, n_rows, n_cols).unwrap(),
        size: FlexArray::full(&size, n_rows, n_cols).unwrap(),
        init_cash: FlexArray::scalar(5000.),
        fees: FlexArray::scalar(0.001),
        direction: FlexArray::scalar(Direction::Both),
        account_type: FlexArray::scalar(AccountType::Margin),
        margin_mode: FlexArray::scalar(MarginMode::Isolated),
        initial_margin: FlexArray::scalar(0.2),
        funding_rate: FlexArray::scalar(3e-4),
        log: FlexArray::scalar(true),
        groups: groups.clone(),
        ..FromOrders::default()
    };
    let journal = alphax_sim::simulate_from_orders_logged(&args).unwrap();
    assert!(
        journal
            .orders
            .cash
            .kind
            .contains(&(CashFlowKind::Funding as i8)),
        "no funding"
    );
    let pf = Portfolio::from_grouped_records(
        journal.orders.clone(),
        args.close,
        args.init_cash,
        n_rows,
        n_cols,
        Scoring::default(),
        groups.clone(),
    )
    .unwrap();
    let free = pf.free_cash().unwrap();
    let logs = &journal.logs;
    let mut seen = vec![false; n_rows * groups.len()];
    for r in 0..logs.len() {
        let (g, row) = (groups.of_column(logs.col[r] as usize), logs.idx[r] as usize);
        if row == 0 || std::mem::replace(&mut seen[g * n_rows + row], true) {
            continue;
        }
        let rebuilt = free.get(row - 1, g);
        assert!(
            (logs.free_cash[r] - rebuilt).abs() < 1e-9,
            "account {g}, bar {row}: simulated {}, rebuilt {rebuilt}",
            logs.free_cash[r]
        );
    }
}

#[test]
fn the_rebuilt_free_cash_of_isolated_columns_is_the_simulations() {
    check_isolated_free_cash(&Groups::ungrouped(4), 7);
    check_isolated_free_cash(&Groups::from_lens(&[2, 2], 4).unwrap(), 11);
}
