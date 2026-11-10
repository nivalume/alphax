//! `leverage` and `leverage_mode` through both loops.
//!
//! The golden fixtures only pin down `leverage = 1` (the kernel's golden
//! tests). What they cannot see is the loop's side: that
//! a levered run values longs net of their debt, records each fill's balances,
//! and leaves a `leverage = 1` column exactly as it was. These are
//! hand-calculated cases and properties of that.
#![allow(clippy::float_cmp, reason = "the hand-calculated fills are exact")]

use alphax_core::flex::FlexArray;
use alphax_model::{Direction, LeverageMode, StatusInfo};
use alphax_records::OrderRecords;
use alphax_sim::groups::Groups;
use alphax_sim::{
    FromOrders, FromSignals, simulate_from_orders, simulate_from_orders_logged,
    simulate_from_orders_logged_parallel, simulate_from_orders_parallel, simulate_from_signals,
    simulate_from_signals_parallel,
};

fn one_column<'a>(close: &'a [f64], size: &'a [f64], leverage: f64) -> FromOrders<'a> {
    FromOrders {
        n_rows: close.len(),
        n_cols: 1,
        init_cash: FlexArray::scalar(1000.0),
        close: FlexArray::full(close, close.len(), 1).unwrap(),
        size: FlexArray::full(size, close.len(), 1).unwrap(),
        leverage: FlexArray::scalar(leverage),
        groups: Groups::ungrouped(1),
        ..FromOrders::default()
    }
}

/// Prices that wander enough to make buys, sells and shorts all fail sometimes.
fn prices(n_rows: usize, n_cols: usize) -> Vec<f64> {
    let mut close = Vec::with_capacity(n_rows * n_cols);
    for col in 0..n_cols {
        let mut price = 10.0 + col as f64;
        for i in 0..n_rows {
            let wiggle = ((i * 2_654_435_761 + col * 40_503) % 1000) as f64 / 1000.0 - 0.5;
            price = (price * (1.0 + wiggle * 0.04)).max(0.5);
            close.push(price);
        }
    }
    close
}

/// Buys of 30, sales of 45 and nothing, in a pattern that differs by column.
fn sizes(n_rows: usize, n_cols: usize) -> Vec<f64> {
    let mut size = Vec::with_capacity(n_rows * n_cols);
    for col in 0..n_cols {
        for i in 0..n_rows {
            size.push(match (i + col) % 5 {
                0 | 1 => 30.0,
                3 => -45.0,
                _ => 0.0,
            });
        }
    }
    size
}

/// One column's fills as raw bits, so that equality means bit-identical.
fn column_bits(records: &OrderRecords, col: usize) -> Vec<[u64; 5]> {
    (0..records.len())
        .filter(|&k| records.col[k] == col as i64)
        .map(|k| {
            [
                records.idx[k] as u64,
                records.size[k].to_bits(),
                records.price[k].to_bits(),
                records.fees[k].to_bits(),
                records.side[k] as u64,
            ]
        })
        .collect()
}

fn all_bits(records: &OrderRecords) -> Vec<u64> {
    let mut bits: Vec<u64> = Vec::new();
    for k in 0..records.len() {
        bits.extend([
            records.col[k] as u64,
            records.idx[k] as u64,
            records.size[k].to_bits(),
            records.price[k].to_bits(),
            records.fees[k].to_bits(),
            records.side[k] as u64,
        ]);
    }
    bits.extend(records.debt.iter().map(|x| x.to_bits()));
    bits.extend(records.locked_cash.iter().map(|x| x.to_bits()));
    bits
}

#[test]
fn a_levered_long_borrows_what_own_cash_cannot_cover_and_records_it() {
    let close = [100.0, 120.0, 110.0];
    let size = [15.0, f64::NAN, -15.0];
    let journal = simulate_from_orders_logged(&FromOrders {
        log: FlexArray::scalar(true),
        ..one_column(&close, &size, 2.0)
    })
    .unwrap();
    let (fills, logs) = (&journal.orders, &journal.logs);
    // 1500 of stock on 1000 of own cash: 500 borrowed, all 1000 put up.
    assert_eq!(fills.size, [15.0, 15.0]);
    assert_eq!(fills.debt, [500.0, 0.0]);
    assert_eq!(fills.locked_cash, [1000.0, 0.0]);
    assert_eq!(logs.new_cash, [0.0, 0.0, 1150.0]);
    // Each bar is valued at its own price, net of the debt: 1000 before the
    // buy, then 15 * 120 - 500 and 15 * 110 - 500.
    assert_eq!(logs.value, [1000.0, 1300.0, 1150.0]);
    assert_eq!(logs.req_leverage, [2.0, 2.0, 2.0]);
}

#[test]
fn eager_puts_up_one_over_leverage_even_with_cash_to_spare() {
    let close = [100.0];
    let size = [5.0];
    let fills = simulate_from_orders(&FromOrders {
        leverage_mode: FlexArray::scalar(LeverageMode::Eager),
        ..one_column(&close, &size, 2.0)
    })
    .unwrap();
    assert_eq!(fills.debt, [250.0]);
    assert_eq!(fills.locked_cash, [250.0]);
}

#[test]
fn a_levered_short_ties_up_its_proceeds_plus_value_over_leverage() {
    let close = [100.0, 80.0];
    let size = [-10.0, 10.0];
    let journal = simulate_from_orders_logged(&FromOrders {
        log: FlexArray::scalar(true),
        ..one_column(&close, &size, 2.0)
    })
    .unwrap();
    assert_eq!(journal.orders.debt, [1000.0, 0.0]);
    assert_eq!(journal.orders.locked_cash, [500.0, 0.0]);
    // 2000 held, 1500 of it tied up; covering at 80 frees all of it.
    assert_eq!(journal.logs.new_free_cash, [500.0, 1200.0]);
}

#[test]
fn an_account_in_debt_cannot_buy_and_says_why() {
    // Four times the cash, then the price halves: the sale does not repay
    // the 3000 borrowed, and the account is left owing 1000.
    let close = [100.0, 50.0, 50.0];
    let size = [40.0, -40.0, 1.0];
    let journal = simulate_from_orders_logged(&FromOrders {
        log: FlexArray::scalar(true),
        ..one_column(&close, &size, 4.0)
    })
    .unwrap();
    assert_eq!(journal.logs.new_cash, [0.0, -1000.0, -1000.0]);
    assert_eq!(
        journal.logs.res_status_info[2],
        StatusInfo::NoCashLeverage as i64
    );
}

#[test]
fn an_unlevered_run_records_no_balances() {
    let close = [100.0, 80.0];
    let size = [-10.0, 10.0];
    let fills = simulate_from_orders(&one_column(&close, &size, 1.0)).unwrap();
    assert_eq!(fills.len(), 2);
    assert!(fills.debt.is_empty() && fills.locked_cash.is_empty());
}

#[test]
fn a_unit_leverage_column_is_unchanged_inside_a_levered_run() {
    let (n_rows, n_cols) = (200, 4);
    let close = prices(n_rows, n_cols);
    let size = sizes(n_rows, n_cols);
    let levered = [1.0, 3.0, 1.0, 2.0];
    let base = || FromOrders {
        n_rows,
        n_cols,
        init_cash: FlexArray::scalar(500.0),
        close: FlexArray::full(&close, n_rows, n_cols).unwrap(),
        size: FlexArray::full(&size, n_rows, n_cols).unwrap(),
        fees: FlexArray::scalar(0.002),
        direction: FlexArray::scalar(Direction::Both),
        groups: Groups::ungrouped(n_cols),
        ..FromOrders::default()
    };
    let plain = simulate_from_orders(&base()).unwrap();
    let mixed = simulate_from_orders(&FromOrders {
        leverage: FlexArray::per_column(&levered, n_cols).unwrap(),
        ..base()
    })
    .unwrap();
    assert_eq!(mixed.debt.len(), mixed.len());
    for col in [0, 2] {
        assert_eq!(
            column_bits(&plain, col),
            column_bits(&mixed, col),
            "column {col}"
        );
    }
    // And the levered columns did trade differently, or this proves nothing.
    assert_ne!(column_bits(&plain, 1), column_bits(&mixed, 1));
}

#[test]
fn a_unit_leverage_signal_column_is_unchanged_inside_a_levered_run() {
    let (n_rows, n_cols) = (200, 3);
    let close = prices(n_rows, n_cols);
    let entries: Vec<bool> = (0..n_rows * n_cols).map(|k| k % 7 == 0).collect();
    let exits: Vec<bool> = (0..n_rows * n_cols).map(|k| k % 7 == 4).collect();
    let levered = [1.0, 2.5, 1.0];
    let base = || FromSignals {
        n_rows,
        n_cols,
        close: FlexArray::full(&close, n_rows, n_cols).unwrap(),
        entries: FlexArray::full(&entries, n_rows, n_cols).unwrap(),
        exits: FlexArray::full(&exits, n_rows, n_cols).unwrap(),
        direction: FlexArray::scalar(Direction::Both),
        fees: FlexArray::scalar(0.001),
        ..FromSignals::default()
    };
    let plain = simulate_from_signals(&base()).unwrap();
    let mixed = simulate_from_signals(&FromSignals {
        leverage: FlexArray::per_column(&levered, n_cols).unwrap(),
        ..base()
    })
    .unwrap();
    for col in [0, 2] {
        assert_eq!(
            column_bits(&plain, col),
            column_bits(&mixed, col),
            "column {col}"
        );
    }
    assert_ne!(column_bits(&plain, 1), column_bits(&mixed, 1));
}

#[test]
fn a_levered_signal_entry_buys_leverage_times_its_cash() {
    let close = [100.0, 110.0];
    let args = FromSignals {
        n_rows: 2,
        n_cols: 1,
        init_cash: FlexArray::scalar(1000.0),
        close: FlexArray::full(&close, 2, 1).unwrap(),
        entries: FlexArray::full(&[true, false], 2, 1).unwrap(),
        exits: FlexArray::full(&[false, true], 2, 1).unwrap(),
        leverage: FlexArray::scalar(2.0),
        ..FromSignals::default()
    };
    let fills = simulate_from_signals(&args).unwrap();
    assert_eq!(fills.size, [20.0, 20.0]);
    assert_eq!(fills.debt, [1000.0, 0.0]);
}

#[test]
fn serial_and_parallel_levered_runs_agree_bit_for_bit() {
    let (n_rows, n_cols) = (150, 6);
    let close = prices(n_rows, n_cols);
    let size = sizes(n_rows, n_cols);
    let leverage = [1.0, 2.0, 3.0, 1.5, 1.0, 4.0];
    let mode = [
        LeverageMode::Lazy,
        LeverageMode::Eager,
        LeverageMode::Lazy,
        LeverageMode::Eager,
        LeverageMode::Lazy,
        LeverageMode::Lazy,
    ];
    for groups in [
        Groups::ungrouped(n_cols),
        Groups::from_lens(&[2, 3, 1], n_cols).unwrap(),
    ] {
        let args = FromOrders {
            n_rows,
            n_cols,
            init_cash: FlexArray::scalar(800.0),
            close: FlexArray::full(&close, n_rows, n_cols).unwrap(),
            size: FlexArray::full(&size, n_rows, n_cols).unwrap(),
            fees: FlexArray::scalar(0.001),
            direction: FlexArray::scalar(Direction::Both),
            leverage: FlexArray::per_column(&leverage, n_cols).unwrap(),
            leverage_mode: FlexArray::per_column(&mode, n_cols).unwrap(),
            log: FlexArray::scalar(true),
            groups,
            ..FromOrders::default()
        };
        let serial = simulate_from_orders(&args).unwrap();
        assert!(!serial.debt.is_empty());
        for threads in [2, 3, 6] {
            let parallel = simulate_from_orders_parallel(&args, threads).unwrap();
            assert_eq!(all_bits(&serial), all_bits(&parallel), "{threads} threads");
            let logged = simulate_from_orders_logged_parallel(&args, threads).unwrap();
            assert_eq!(
                all_bits(&serial),
                all_bits(&logged.orders),
                "{threads} threads, logged"
            );
        }
    }

    let entries: Vec<bool> = (0..n_rows * n_cols).map(|k| k % 5 == 0).collect();
    let exits: Vec<bool> = (0..n_rows * n_cols).map(|k| k % 5 == 3).collect();
    let args = FromSignals {
        n_rows,
        n_cols,
        close: FlexArray::full(&close, n_rows, n_cols).unwrap(),
        entries: FlexArray::full(&entries, n_rows, n_cols).unwrap(),
        exits: FlexArray::full(&exits, n_rows, n_cols).unwrap(),
        direction: FlexArray::scalar(Direction::Both),
        leverage: FlexArray::per_column(&leverage, n_cols).unwrap(),
        leverage_mode: FlexArray::per_column(&mode, n_cols).unwrap(),
        ..FromSignals::default()
    };
    let serial = simulate_from_signals(&args).unwrap();
    assert!(!serial.debt.is_empty());
    for threads in [2, 3, 6] {
        let parallel = simulate_from_signals_parallel(&args, threads).unwrap();
        assert_eq!(all_bits(&serial), all_bits(&parallel), "{threads} threads");
    }
}
