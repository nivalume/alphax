//! Deposits and earnings: hand-calculated balances, and the simulation's own
//! cash state checked against the balance the analysis layer rebuilds.
//!
//! There is no golden fixture for these inputs, so two checks stand in for
//! one: small accounts whose every balance is worked out by hand, and a
//! randomized run where the cash each logged order saw *inside* the loop has to
//! equal the cash rebuilt from records alone *outside* it. The second is the check that matters — the
//! rebuild chain is the only way anything downstream learns about a balance.
#![allow(
    clippy::float_cmp,
    reason = "the hand-calculated accounts have exact balances"
)]

use alphax_analysis::portfolio::{Portfolio, Scoring};
use alphax_core::flex::FlexArray;
use alphax_model::{CashFlowKind, Direction, SizeType};
use alphax_sim::from_orders::FromOrders;
use alphax_sim::from_signals::FromSignals;
use alphax_sim::groups::{CallSeq, Groups};

const DEPOSIT: i8 = CashFlowKind::Deposit as i8;
const EARNING: i8 = CashFlowKind::Earning as i8;
const DIVIDEND: i8 = CashFlowKind::Dividend as i8;

#[test]
fn a_deposit_is_not_a_return_and_a_withdrawal_is_capped() {
    // Bar 0: buy 5 at 10.          cash  50, value 100.
    // Bar 1: deposit 100.          cash 150, value 200, return 0.
    // Bar 2: price doubles, earn 1. cash 151, value 251, return 51/200.
    // Bar 3: withdraw 200, capped at the 151 free.
    //                              cash   0, value 100, return 0.
    let close = [10., 10., 20., 20.];
    let size = [5., f64::NAN, f64::NAN, f64::NAN];
    let deposits = [0., 100., 0., -200.];
    let earnings = [0., 0., 1., 0.];
    let args = FromOrders {
        n_rows: 4,
        n_cols: 1,
        close: FlexArray::full(&close, 4, 1).unwrap(),
        size: FlexArray::full(&size, 4, 1).unwrap(),
        cash_deposits: FlexArray::per_row(&deposits, 4).unwrap(),
        cash_earnings: FlexArray::per_row(&earnings, 4).unwrap(),
        groups: Groups::ungrouped(1),
        ..FromOrders::default()
    };
    let pf = Portfolio::from_orders(&args, 1).unwrap();

    assert_eq!(pf.cash().column(0), [50., 150., 151., 0.]);
    assert_eq!(pf.value().column(0), [100., 200., 251., 100.]);
    let returns = pf.returns().column(0);
    assert_eq!(returns[0], 0.0);
    assert_eq!(returns[1], 0.0, "a deposit is not a gain");
    assert!((returns[2] - 51.0 / 200.0).abs() < 1e-15);
    assert_eq!(returns[3], 0.0, "a withdrawal is not a loss");
    assert!((pf.total_return()[0] - 0.255).abs() < 1e-12);

    let cash = pf.cash_records();
    assert_eq!(cash.idx, [1, 2, 3]);
    assert_eq!(
        cash.amount,
        [100., 1., -151.],
        "the record holds what was applied"
    );
    assert_eq!(cash.kind, [DEPOSIT, EARNING, DEPOSIT]);

    // The value fell from 251 to 100, but by withdrawal, not by loss.
    assert_eq!(pf.drawdowns().len(), 0);
    assert_eq!(pf.max_drawdown(), [0.0]);
}

#[test]
fn a_deposit_is_invested_by_a_target_order_on_the_same_bar() {
    // Booked at the open, so the bar's target-percent rebalance sees it.
    let close = [10., 10.];
    let deposits = [0., 100.];
    let args = FromOrders {
        n_rows: 2,
        n_cols: 1,
        close: FlexArray::full(&close, 2, 1).unwrap(),
        size: FlexArray::scalar(1.0),
        size_type: FlexArray::scalar(SizeType::TargetPercent),
        cash_deposits: FlexArray::per_row(&deposits, 2).unwrap(),
        groups: Groups::ungrouped(1),
        ..FromOrders::default()
    };
    let pf = Portfolio::from_orders(&args, 1).unwrap();
    assert_eq!(pf.assets().column(0), [10., 20.]);
    assert_eq!(pf.cash().column(0), [0., 0.]);
    assert_eq!(pf.returns().column(0), [0., 0.]);
}

#[test]
fn earnings_are_booked_on_bars_without_a_signal() {
    let close = [10.; 5];
    let none = [false; 5];
    let args = FromSignals {
        n_rows: 5,
        n_cols: 1,
        close: FlexArray::full(&close, 5, 1).unwrap(),
        entries: FlexArray::full(&none, 5, 1).unwrap(),
        exits: FlexArray::full(&none, 5, 1).unwrap(),
        cash_earnings: FlexArray::scalar(1.0),
        ..FromSignals::default()
    };
    let pf = Portfolio::from_signals(&args, 1).unwrap();
    assert!(pf.orders().is_empty());
    assert_eq!(pf.cash_records().len(), 5);
    assert_eq!(pf.cash().column(0), [101., 102., 103., 104., 105.]);
    assert!((pf.total_return()[0] - 0.05).abs() < 1e-12);
}

#[test]
fn a_shared_account_books_its_deposit_once() {
    // Two columns, one account. The deposit is read from the first column
    // only; the second column's value is never consulted.
    let close = [10., 10., 20., 20.];
    let deposits = [0., 50., 999., 999.];
    let args = FromOrders {
        n_rows: 2,
        n_cols: 2,
        close: FlexArray::full(&close, 2, 2).unwrap(),
        size: FlexArray::scalar(f64::NAN),
        cash_deposits: FlexArray::full(&deposits, 2, 2).unwrap(),
        groups: Groups::from_lens(&[2], 2).unwrap(),
        ..FromOrders::default()
    };
    let pf = Portfolio::from_orders(&args, 1).unwrap();
    assert_eq!(pf.n_groups(), 1);
    assert_eq!(pf.cash().column(0), [100., 150.]);
    assert_eq!(pf.cash_records().col, [0]);
    assert_eq!(pf.returns().column(0), [0., 0.]);
}

#[test]
fn an_infinite_amount_is_refused() {
    let close = [10.; 2];
    let args = FromOrders {
        n_rows: 2,
        n_cols: 1,
        close: FlexArray::full(&close, 2, 1).unwrap(),
        cash_earnings: FlexArray::scalar(f64::INFINITY),
        groups: Groups::ungrouped(1),
        ..FromOrders::default()
    };
    assert!(Portfolio::from_orders(&args, 1).is_err());
}

/// A small deterministic generator, so the randomized case needs no crate.
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

/// A run that trades both ways on every bar, with deposits, withdrawals,
/// dividends, earnings and charges large enough to hit the cap.
struct Workload {
    n_rows: usize,
    n_cols: usize,
    close: Vec<f64>,
    size: Vec<f64>,
    deposits: Vec<f64>,
    dividends: Vec<f64>,
    earnings: Vec<f64>,
}

impl Workload {
    fn new(n_rows: usize, n_cols: usize, seed: u64) -> Self {
        let mut rng = Lcg(seed);
        let n = n_rows * n_cols;
        let mut close = Vec::with_capacity(n);
        for _ in 0..n_cols {
            let mut p = 10.0 + 10.0 * rng.next();
            for _ in 0..n_rows {
                p *= 1.0 + 0.1 * (rng.next() - 0.5);
                close.push(p);
            }
        }
        let size = (0..n).map(|_| (rng.next() - 0.5) * 6.0).collect();
        let flow = |rng: &mut Lcg, scale: f64| -> f64 {
            let u = rng.next();
            if u < 0.6 {
                0.0
            } else if u < 0.8 {
                scale * rng.next()
            } else {
                // Often more than the account holds, so the cap is exercised.
                -scale * 3.0 * rng.next()
            }
        };
        let deposits = (0..n).map(|_| flow(&mut rng, 40.0)).collect();
        let earnings = (0..n).map(|_| flow(&mut rng, 5.0)).collect();
        // Per share, on about one bar in ten; shorts pay them.
        let dividends = (0..n)
            .map(|_| if rng.next() < 0.1 { rng.next() } else { 0.0 })
            .collect();
        Self {
            n_rows,
            n_cols,
            close,
            size,
            deposits,
            dividends,
            earnings,
        }
    }

    fn args(&self, groups: Groups) -> FromOrders<'_> {
        FromOrders {
            n_rows: self.n_rows,
            n_cols: self.n_cols,
            close: FlexArray::full(&self.close, self.n_rows, self.n_cols).unwrap(),
            size: FlexArray::full(&self.size, self.n_rows, self.n_cols).unwrap(),
            direction: FlexArray::scalar(Direction::Both),
            fees: FlexArray::scalar(0.002),
            cash_deposits: FlexArray::full(&self.deposits, self.n_rows, self.n_cols).unwrap(),
            cash_dividends: FlexArray::full(&self.dividends, self.n_rows, self.n_cols).unwrap(),
            cash_earnings: FlexArray::full(&self.earnings, self.n_rows, self.n_cols).unwrap(),
            log: FlexArray::scalar(true),
            groups,
            call_seq: CallSeq::Default,
            ..FromOrders::default()
        }
    }
}

/// Checks every logged order against the rebuilt balances.
///
/// The first order an account executes on a bar saw the cash the account
/// closed the previous bar with, plus what the bar's open booked: its deposit
/// and its dividends. The rebuilt cash at the previous bar's close already
/// includes that bar's earnings, so this checks every kind of flow and both
/// balance series in one comparison.
fn check_against_logs(work: &Workload, groups: &Groups) {
    let args = work.args(groups.clone());
    let journal = alphax_sim::simulate_from_orders_logged(&args).unwrap();
    let pf = Portfolio::from_grouped_records(
        journal.orders.clone(),
        args.close,
        args.init_cash,
        work.n_rows,
        work.n_cols,
        Scoring::default(),
        groups.clone(),
    )
    .unwrap();

    let records = pf.cash_records();
    for kind in [DEPOSIT, DIVIDEND, EARNING] {
        assert!(
            records.kind.contains(&kind),
            "the workload books no kind {kind}"
        );
    }
    assert!(
        (0..records.len()).any(|i| records.kind[i] == DIVIDEND && records.amount[i] < 0.0),
        "the workload should have a short pay a dividend"
    );
    assert!(
        (0..records.len()).any(|i| records.amount[i] < 0.0),
        "the workload should book an outflow"
    );

    let cash = pf.cash();
    let free_cash = pf.free_cash().unwrap();
    // What each account's bar opened with, besides the previous close.
    let mut at_open = vec![0.0; work.n_rows * groups.len()];
    for i in 0..records.len() {
        if records.kind[i] != EARNING {
            let group = groups.of_column(records.col[i] as usize);
            at_open[group * work.n_rows + records.idx[i] as usize] += records.amount[i];
        }
    }
    let logs = &journal.logs;
    let mut checked = 0;
    for r in 0..logs.len() {
        let (col, row) = (logs.col[r] as usize, logs.idx[r] as usize);
        let group = groups.of_column(col);
        // Only the account's first order on the bar; later ones see the
        // fills before them.
        if col != groups.range(group).start {
            continue;
        }
        let deposit = at_open[group * work.n_rows + row];
        let (expected_cash, expected_free) = if row == 0 {
            let init = args.init_cash.get(0, col);
            (init + deposit, init + deposit)
        } else {
            (
                cash.get(row - 1, group) + deposit,
                free_cash.get(row - 1, group) + deposit,
            )
        };
        assert!(
            (logs.cash[r] - expected_cash).abs() < 1e-9,
            "row {row}, col {col}: the loop saw cash {}, the rebuild says {expected_cash}",
            logs.cash[r]
        );
        assert!(
            (logs.free_cash[r] - expected_free).abs() < 1e-9,
            "row {row}, col {col}: the loop saw free cash {}, the rebuild says {expected_free}",
            logs.free_cash[r]
        );
        // Free cash may be negative — buying back a short above its entry
        // does that without any cash flow — but cash never is.
        assert!(logs.cash[r] >= 0.0, "row {row}, col {col}: negative cash");
        checked += 1;
    }
    assert_eq!(checked, work.n_rows * groups.len());
}

#[test]
fn the_loop_and_the_rebuild_agree_on_every_balance() {
    let work = Workload::new(60, 4, 20_261_011);
    check_against_logs(&work, &Groups::ungrouped(4));
}

#[test]
fn the_loop_and_the_rebuild_agree_in_shared_accounts() {
    let work = Workload::new(60, 5, 7);
    check_against_logs(&work, &Groups::from_lens(&[2, 3], 5).unwrap());
}

#[test]
fn threads_do_not_change_the_cash_records() {
    let work = Workload::new(50, 9, 99);
    for groups in [
        Groups::ungrouped(9),
        Groups::from_lens(&[3, 1, 5], 9).unwrap(),
    ] {
        let args = work.args(groups);
        let serial = alphax_sim::simulate_from_orders(&args).unwrap();
        assert!(!serial.cash.is_empty());
        for threads in [2, 3, 8] {
            let parallel = alphax_sim::simulate_from_orders_parallel(&args, threads).unwrap();
            assert_eq!(serial, parallel, "{threads} threads");
        }
        let logged = alphax_sim::simulate_from_orders_logged_parallel(&args, 3).unwrap();
        assert_eq!(logged.orders, serial, "logging changed the records");
    }
}

#[test]
fn signal_runs_book_the_same_flows_on_every_thread_count() {
    let work = Workload::new(40, 6, 3);
    let entries: Vec<bool> = work.size.iter().map(|s| *s > 1.5).collect();
    let exits: Vec<bool> = work.size.iter().map(|s| *s < -1.5).collect();
    let args = FromSignals {
        n_rows: work.n_rows,
        n_cols: work.n_cols,
        close: FlexArray::full(&work.close, work.n_rows, work.n_cols).unwrap(),
        entries: FlexArray::full(&entries, work.n_rows, work.n_cols).unwrap(),
        exits: FlexArray::full(&exits, work.n_rows, work.n_cols).unwrap(),
        direction: FlexArray::scalar(Direction::Both),
        size: FlexArray::scalar(2.0),
        cash_deposits: FlexArray::full(&work.deposits, work.n_rows, work.n_cols).unwrap(),
        cash_earnings: FlexArray::full(&work.earnings, work.n_rows, work.n_cols).unwrap(),
        groups: Some(Groups::from_lens(&[2, 4], 6).unwrap()),
        ..FromSignals::default()
    };
    let serial = alphax_sim::simulate_from_signals(&args).unwrap();
    assert!(!serial.cash.is_empty());
    for threads in [2, 4] {
        assert_eq!(
            serial,
            alphax_sim::simulate_from_signals_parallel(&args, threads).unwrap()
        );
    }
    // Every account stays solvent: the caps keep the cash rebuilt from
    // records at or above zero on every bar.
    let pf = Portfolio::from_signals(&args, 1).unwrap();
    assert!(pf.cash().as_slice().iter().all(|c| *c >= -1e-9));
}

#[test]
fn a_dividend_is_owed_to_the_position_held_at_the_open() {
    // Bar 0: buy 10 at 10.                         cash   0.
    // Bar 1: dividend 1 on the 10 held.            cash  10, return 10 %.
    // Bar 2: dividend 1, then sell all 10 at 10.   cash 120: the sale on the
    //        ex-date does not forfeit the dividend.
    // Bar 3: dividend 1, nothing held.             cash 120.
    // Bar 4: buy 10 at 10 on a dividend bar: the buyer is not owed it.
    let close = [10.0; 5];
    let size = [10., f64::NAN, -10., f64::NAN, 10.];
    let dividends = [0., 1., 1., 1., 1.];
    let args = FromOrders {
        n_rows: 5,
        n_cols: 1,
        close: FlexArray::full(&close, 5, 1).unwrap(),
        size: FlexArray::full(&size, 5, 1).unwrap(),
        cash_dividends: FlexArray::full(&dividends, 5, 1).unwrap(),
        groups: Groups::ungrouped(1),
        ..FromOrders::default()
    };
    let pf = Portfolio::from_orders(&args, 1).unwrap();
    assert_eq!(pf.cash().column(0), [0., 10., 120., 120., 20.]);
    assert_eq!(pf.cash_records().idx, [1, 2]);
    assert_eq!(pf.cash_records().kind, [DIVIDEND, DIVIDEND]);
    assert_eq!(pf.returns().get(1, 0), 0.1, "a dividend is a return");
    assert!(pf.deposits().is_none(), "and not a deposit");
}

#[test]
fn a_short_pays_the_dividend() {
    let close = [10.0; 2];
    let size = [-5., f64::NAN];
    let dividends = [0., 2.];
    let args = FromOrders {
        n_rows: 2,
        n_cols: 1,
        close: FlexArray::full(&close, 2, 1).unwrap(),
        size: FlexArray::full(&size, 2, 1).unwrap(),
        cash_dividends: FlexArray::full(&dividends, 2, 1).unwrap(),
        groups: Groups::ungrouped(1),
        ..FromOrders::default()
    };
    let pf = Portfolio::from_orders(&args, 1).unwrap();
    assert_eq!(pf.cash_records().amount, [-10.0]);
    assert_eq!(pf.cash().column(0), [150., 140.]);
    assert_eq!(pf.value().column(0), [100., 90.]);
}
