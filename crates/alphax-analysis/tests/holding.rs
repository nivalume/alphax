//! Holding charges once a run is rebuilt from its records.
//!
//! A charge is a cash record like an earning, so the rebuild needs nothing new
//! to count it. These check that it does: by hand on one perpetual, and
//! against the simulation's own account values on a run mixing every charge
//! with leverage, margin columns and shared accounts.
use alphax_analysis::Scoring;
use alphax_analysis::portfolio::Portfolio;
use alphax_core::flex::FlexArray;
use alphax_model::{AccountType, CashFlowKind, Direction, LeverageMode};
use alphax_sim::from_orders::FromOrders;
use alphax_sim::groups::Groups;

const N: f64 = f64::NAN;
const R: f64 = 1.0 / 1024.0;

fn close_to(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < 1e-9, "{actual:?} vs {expected:?}");
    }
}

#[test]
fn funding_is_part_of_the_equity_and_the_return() {
    // Long 8 at 128 on half margin, funded on bars 0 and 1, sold at 144.
    let close = [128., 160., 160., 144.];
    let size = [8., N, N, -8.];
    let funding = [R, R, N, N];
    let args = FromOrders {
        n_rows: 4,
        n_cols: 1,
        close: FlexArray::per_row(&close, 4).unwrap(),
        size: FlexArray::per_row(&size, 4).unwrap(),
        init_cash: FlexArray::scalar(1000.),
        account_type: FlexArray::scalar(AccountType::Margin),
        initial_margin: FlexArray::scalar(0.5),
        funding_rate: FlexArray::per_row(&funding, 4).unwrap(),
        groups: Groups::ungrouped(1),
        ..FromOrders::default()
    };
    let pf = Portfolio::from_orders(&args, 1).unwrap();
    close_to(
        pf.cash_flow_by_kind(CashFlowKind::Funding).column(0),
        &[-1., -1.25, 0., 0.],
    );
    assert!(
        pf.cash_flow_by_kind(CashFlowKind::Financing)
            .as_slice()
            .iter()
            .all(|v| *v == 0.)
    );
    close_to(pf.value().column(0), &[999., 1253.75, 1253.75, 1125.75]);
    // The simulation's free cash: equity less the 512 posted.
    close_to(
        pf.free_cash().unwrap().column(0),
        &[487., 741.75, 741.75, 1125.75],
    );
    let r = pf.returns().column(0);
    assert!(
        (r[0] - (999. / 1000. - 1.)).abs() < 1e-15,
        "a charge is a return"
    );
    assert!(pf.deposits().is_none(), "and not a deposit");
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
/// at: a bar's orders fill at its close, so they change the equity only by
/// their fees, and the bar's charges land after them.
fn check(groups: &Groups, seed: u64) {
    let (n_rows, n_cols) = (80, 4);
    let mut rng = Lcg(seed);
    let mut close = Vec::with_capacity(n_rows * n_cols);
    for _ in 0..n_cols {
        let mut p = 20.0 + 10.0 * rng.next();
        for _ in 0..n_rows {
            p *= 1.0 + 0.08 * (rng.next() - 0.5);
            close.push(p);
        }
    }
    let size: Vec<f64> = (0..n_rows * n_cols)
        .map(|_| {
            let u = rng.next();
            if u < 0.5 { N } else { (u - 0.75) * 40.0 }
        })
        .collect();
    let funding: Vec<f64> = (0..n_rows * n_cols)
        .map(|_| (rng.next() - 0.4) * 1e-3)
        .collect();
    let mark: Vec<f64> = close
        .iter()
        .map(|c| c * (1.0 + 0.002 * rng.next()))
        .collect();
    let types = [
        AccountType::Margin,
        AccountType::Cash,
        AccountType::Margin,
        AccountType::Cash,
    ];
    let leverage = [1.0, 3.0, 1.0, 1.0];
    let args = FromOrders {
        n_rows,
        n_cols,
        close: FlexArray::full(&close, n_rows, n_cols).unwrap(),
        size: FlexArray::full(&size, n_rows, n_cols).unwrap(),
        init_cash: FlexArray::scalar(1000.),
        fees: FlexArray::scalar(0.001),
        direction: FlexArray::scalar(Direction::Both),
        leverage: FlexArray::per_column(&leverage, n_cols).unwrap(),
        // Every buy borrows, so the leveraged column pays interest.
        leverage_mode: FlexArray::scalar(LeverageMode::Eager),
        account_type: FlexArray::per_column(&types, n_cols).unwrap(),
        initial_margin: FlexArray::scalar(0.25),
        funding_rate: FlexArray::full(&funding, n_rows, n_cols).unwrap(),
        long_financing_rate: FlexArray::scalar(2e-4),
        short_financing_rate: FlexArray::scalar(1e-4),
        debt_rate: FlexArray::scalar(3e-4),
        borrow_rate: FlexArray::scalar(5e-4),
        mark_price: FlexArray::full(&mark, n_rows, n_cols).unwrap(),
        cash_earnings: FlexArray::scalar(0.01),
        log: FlexArray::scalar(true),
        groups: groups.clone(),
        ..FromOrders::default()
    };
    let journal = alphax_sim::simulate_from_orders_logged(&args).unwrap();
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
    for kind in [
        CashFlowKind::Funding,
        CashFlowKind::Financing,
        CashFlowKind::Interest,
        CashFlowKind::BorrowFee,
    ] {
        assert!(
            journal.orders.cash.kind.contains(&(kind as i8)),
            "the workload books no {kind:?}"
        );
    }

    let n_groups = groups.len();
    let mut opened = vec![N; n_rows * n_groups];
    let mut fees = vec![0.0; n_rows * n_groups];
    let logs = &journal.logs;
    for r in 0..logs.len() {
        let (col, row) = (logs.col[r] as usize, logs.idx[r] as usize);
        let g = groups.of_column(col);
        // Every order of a bar is sized against the value the bar opened with.
        opened[g * n_rows + row] = logs.value[r];
        if !logs.res_fees[r].is_nan() {
            fees[g * n_rows + row] += logs.res_fees[r];
        }
    }
    let mut charged = vec![0.0; n_rows * n_groups];
    let cash = &journal.orders.cash;
    for i in 0..cash.len() {
        let g = groups.of_column(cash.col[i] as usize);
        charged[g * n_rows + cash.idx[i] as usize] += cash.amount[i];
    }
    let value = pf.value();
    for g in 0..n_groups {
        for row in 0..n_rows {
            let k = g * n_rows + row;
            let expected = opened[k] - fees[k] + charged[k];
            let rebuilt = value.get(row, g);
            assert!(
                (rebuilt - expected).abs() < 1e-9 * expected.abs().max(1.0),
                "account {g}, bar {row}: rebuilt {rebuilt}, simulated {expected}"
            );
        }
    }
}

#[test]
fn the_rebuilt_equity_is_the_simulations_in_separate_accounts() {
    check(&Groups::ungrouped(4), 5);
}

#[test]
fn the_rebuilt_equity_is_the_simulations_in_shared_accounts() {
    check(&Groups::from_lens(&[2, 2], 4).unwrap(), 11);
}
