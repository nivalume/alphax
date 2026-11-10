//! `cash_limit` and `reject_prob` through both loops.
//!
//! Neither has a golden fixture — `cash_limit` has no recorded expectations,
//! and `reject_prob` draws from a generator keyed on `(seed, row, col)` that no
//! file can pin — so these are hand-calculated cases and
//! properties: the cap's arithmetic, the rejection rate, and that the draws
//! depend on the seed and on nothing else.
#![allow(clippy::float_cmp, reason = "the hand-calculated fills are exact")]

use alphax_core::flex::FlexArray;
use alphax_model::{Direction, SizeType, StatusInfo};
use alphax_sim::groups::{CallSeq, Groups};
use alphax_sim::{
    FromOrders, FromSignals, simulate_from_orders, simulate_from_orders_logged,
    simulate_from_orders_parallel, simulate_from_signals, simulate_from_signals_parallel,
};

fn one_column<'a>(close: &'a [f64], size: &'a [f64]) -> FromOrders<'a> {
    FromOrders {
        n_rows: close.len(),
        n_cols: 1,
        close: FlexArray::full(close, close.len(), 1).unwrap(),
        size: FlexArray::full(size, close.len(), 1).unwrap(),
        groups: Groups::ungrouped(1),
        ..FromOrders::default()
    }
}

#[test]
fn a_cash_limit_caps_what_one_buy_spends_fees_included() {
    let close = [10.0];
    let size = [f64::INFINITY];
    let args = FromOrders {
        cash_limit: FlexArray::scalar(30.0),
        fees: FlexArray::scalar(0.5),
        ..one_column(&close, &size)
    };
    let fills = simulate_from_orders(&args).unwrap();
    // 30 of cash at 50% fees buys 20 of stock: 2 units.
    assert_eq!(fills.size, [2.0]);
    assert_eq!(fills.fees, [10.0]);
}

#[test]
fn a_cash_limit_is_applied_before_percent() {
    let close = [10.0];
    let size = [0.5];
    let args = FromOrders {
        size_type: FlexArray::scalar(SizeType::Percent),
        cash_limit: FlexArray::scalar(30.0),
        ..one_column(&close, &size)
    };
    // Half of the 30 the order may use, not half of the 100 held.
    assert_eq!(simulate_from_orders(&args).unwrap().size, [1.5]);
}

#[test]
fn a_cash_limit_does_not_touch_sells_or_shorts() {
    let close = [10.0, 10.0];
    let size = [5.0, -8.0];
    let args = FromOrders {
        cash_limit: FlexArray::scalar(50.0),
        ..one_column(&close, &size)
    };
    // The buy uses the whole 50; the sale of 8 closes 5 and shorts 3, uncapped.
    assert_eq!(simulate_from_orders(&args).unwrap().size, [5.0, 8.0]);
}

#[test]
fn an_invalid_cash_limit_is_an_error() {
    let close = [10.0];
    let size = [1.0];
    for bad in [f64::NAN, -1.0] {
        let args = FromOrders {
            cash_limit: FlexArray::scalar(bad),
            ..one_column(&close, &size)
        };
        assert!(simulate_from_orders(&args).is_err(), "{bad} was accepted");
    }
}

#[test]
fn a_certain_rejection_rejects_every_order_and_says_why() {
    let close = [10.0; 4];
    let size = [1.0; 4];
    let args = FromOrders {
        reject_prob: FlexArray::scalar(1.0),
        log: FlexArray::scalar(true),
        ..one_column(&close, &size)
    };
    let journal = simulate_from_orders_logged(&args).unwrap();
    assert!(journal.orders.is_empty());
    assert!(
        journal
            .logs
            .reasons()
            .iter()
            .all(|r| *r == StatusInfo::RandomEvent.describe())
    );
}

/// Many independent single-unit buys, so each one is a Bernoulli trial.
fn many_orders(n_rows: usize, n_cols: usize) -> (Vec<f64>, Vec<f64>) {
    (vec![1.0; n_rows * n_cols], vec![1.0; n_rows * n_cols])
}

fn sweep<'a>(close: &'a [f64], size: &'a [f64], n_rows: usize, n_cols: usize) -> FromOrders<'a> {
    FromOrders {
        n_rows,
        n_cols,
        init_cash: FlexArray::scalar(1e9),
        close: FlexArray::full(close, n_rows, n_cols).unwrap(),
        size: FlexArray::full(size, n_rows, n_cols).unwrap(),
        reject_prob: FlexArray::scalar(0.3),
        seed: 42,
        groups: Groups::ungrouped(n_cols),
        ..FromOrders::default()
    }
}

#[test]
fn the_rejection_rate_is_the_probability_asked_for() {
    let (n_rows, n_cols) = (2_000, 10);
    let (close, size) = many_orders(n_rows, n_cols);
    let fills = simulate_from_orders(&sweep(&close, &size, n_rows, n_cols)).unwrap();
    let rate = 1.0 - fills.len() as f64 / (n_rows * n_cols) as f64;
    // Twenty thousand trials: the standard error is about 0.003.
    assert!((rate - 0.3).abs() < 0.015, "rejected {rate}");
}

#[test]
fn draws_depend_on_the_seed_and_not_on_threads_or_grouping() {
    let (n_rows, n_cols) = (300, 12);
    let (close, size) = many_orders(n_rows, n_cols);
    let args = sweep(&close, &size, n_rows, n_cols);
    let serial = simulate_from_orders(&args).unwrap();
    assert_eq!(serial, simulate_from_orders(&args).unwrap());
    for threads in [2, 5] {
        assert_eq!(
            serial,
            simulate_from_orders_parallel(&args, threads).unwrap()
        );
    }
    let reseeded = FromOrders {
        seed: 43,
        ..args.clone()
    };
    assert_ne!(serial, simulate_from_orders(&reseeded).unwrap());

    // Sharing cash changes which account an order belongs to, not its draw:
    // with cash to spare, the same orders survive.
    let shared = FromOrders {
        groups: Groups::from_lens(&[4, 8], n_cols).unwrap(),
        call_seq: CallSeq::Default,
        ..args
    };
    let mut grouped = simulate_from_orders(&shared).unwrap();
    grouped.sort_by_column();
    assert_eq!(serial.col, grouped.col);
    assert_eq!(serial.idx, grouped.idx);
}

#[test]
fn signal_runs_reject_at_random_too() {
    let (n_rows, n_cols) = (400, 6);
    let close = vec![10.0; n_rows * n_cols];
    // Enter on even bars, exit on odd ones: every bar is an order.
    let entries: Vec<bool> = (0..n_rows * n_cols)
        .map(|k| (k % n_rows) % 2 == 0)
        .collect();
    let exits: Vec<bool> = entries.iter().map(|e| !e).collect();
    let args = FromSignals {
        n_rows,
        n_cols,
        close: FlexArray::full(&close, n_rows, n_cols).unwrap(),
        entries: FlexArray::full(&entries, n_rows, n_cols).unwrap(),
        exits: FlexArray::full(&exits, n_rows, n_cols).unwrap(),
        direction: FlexArray::scalar(Direction::LongOnly),
        size: FlexArray::scalar(1.0),
        reject_prob: FlexArray::scalar(0.5),
        seed: 7,
        ..FromSignals::default()
    };
    let baseline = simulate_from_signals(&FromSignals {
        reject_prob: FlexArray::scalar(0.0),
        ..args.clone()
    })
    .unwrap();
    let rejected = simulate_from_signals(&args).unwrap();
    assert!(rejected.len() < baseline.len() * 3 / 4);
    assert!(!rejected.is_empty());
    assert_eq!(rejected, simulate_from_signals_parallel(&args, 3).unwrap());
}
