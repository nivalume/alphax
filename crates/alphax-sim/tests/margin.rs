//! Margin columns through both loops.
//!
//! The kernel's tests pin down a single margin fill. What they cannot see is
//! the loop's side: that a margin column settles its profit into cash every
//! bar, that it counts at its profit in a shared account's value, that a cash
//! column beside it trades exactly as it would alone, and that threads change
//! nothing. These are hand-calculated cases and properties of that.
#![allow(clippy::float_cmp, reason = "the hand-calculated fills are exact")]

use alphax_core::flex::FlexArray;
use alphax_model::{AccountType, Direction, MarginBasis, SizeType, StatusInfo};
use alphax_records::OrderRecords;
use alphax_sim::groups::Groups;
use alphax_sim::{
    FromOrders, FromSignals, simulate_from_orders, simulate_from_orders_logged,
    simulate_from_orders_logged_parallel, simulate_from_orders_parallel, simulate_from_signals,
    simulate_from_signals_parallel,
};

const N: f64 = f64::NAN;

/// One margin column of 1000, posting `initial_margin` of the notional.
fn one_column<'a>(close: &'a [f64], size: &'a [f64], initial_margin: f64) -> FromOrders<'a> {
    FromOrders {
        n_rows: close.len(),
        n_cols: 1,
        init_cash: FlexArray::scalar(1000.0),
        close: FlexArray::full(close, close.len(), 1).unwrap(),
        size: FlexArray::full(size, close.len(), 1).unwrap(),
        account_type: FlexArray::scalar(AccountType::Margin),
        initial_margin: FlexArray::scalar(initial_margin),
        log: FlexArray::scalar(true),
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
    bits.extend(records.margin.iter().map(|&m| u64::from(m)));
    bits
}

#[test]
fn a_margin_long_settles_its_profit_into_cash_every_bar() {
    let close = [100.0, 110.0, 90.0, 95.0];
    let size = [10.0, N, N, -10.0];
    let journal = simulate_from_orders_logged(&one_column(&close, &size, 0.2)).unwrap();
    let (fills, logs) = (&journal.orders, &journal.logs);
    // 1000 of contracts post 200 and spend nothing; selling at 95 returns
    // the margin and leaves the 50 lost from the entry in cash.
    assert_eq!(fills.size, [10.0, 10.0]);
    assert_eq!(fills.debt, [1000.0, 0.0]);
    assert_eq!(fills.locked_cash, [200.0, 0.0]);
    assert_eq!(fills.margin, [true, true]);
    // +100, -200 and +50 settle into cash at each bar's open, before the
    // order, and the cost basis follows the price.
    assert_eq!(logs.cash, [1000.0, 1100.0, 900.0, 950.0]);
    assert_eq!(logs.debt, [0.0, 1100.0, 900.0, 950.0]);
    assert_eq!(logs.free_cash, [1000.0, 900.0, 700.0, 750.0]);
    assert_eq!(logs.new_free_cash, [800.0, 900.0, 700.0, 950.0]);
    // Settled, the position adds nothing to the value: it is all cash.
    assert_eq!(logs.value, [1000.0, 1100.0, 900.0, 950.0]);
    assert_eq!(logs.req_account_type, [1, 1, 1, 1]);
    assert_eq!(logs.req_initial_margin, [0.2, 0.2, 0.2, 0.2]);
}

#[test]
fn a_margin_short_posts_the_margin_and_keeps_its_proceeds_out_of_cash() {
    let close = [100.0, 80.0];
    let size = [-10.0, 10.0];
    let journal = simulate_from_orders_logged(&one_column(&close, &size, 0.25)).unwrap();
    let (fills, logs) = (&journal.orders, &journal.logs);
    assert_eq!(fills.debt, [1000.0, 0.0]);
    assert_eq!(fills.locked_cash, [250.0, 0.0]);
    // The short's 200 settles at 80, and covering releases the 250.
    assert_eq!(logs.new_cash, [1000.0, 1200.0]);
    assert_eq!(logs.free_cash, [1000.0, 950.0]);
    assert_eq!(logs.new_free_cash, [750.0, 1200.0]);
}

#[test]
fn a_loss_past_the_cash_takes_it_negative_and_only_closing_is_allowed() {
    // 100 posts 10% of 1000 of contracts; a 20% fall loses 200.
    let close = [100.0, 80.0, 80.0];
    let size = [10.0, 1.0, -10.0];
    let journal = simulate_from_orders_logged(&FromOrders {
        init_cash: FlexArray::scalar(100.0),
        ..one_column(&close, &size, 0.1)
    })
    .unwrap();
    let logs = &journal.logs;
    assert_eq!(logs.new_free_cash[0], 0.0);
    assert_eq!(logs.cash[1], -100.0);
    assert_eq!(logs.free_cash[1], -200.0);
    assert_eq!(logs.res_status_info[1], StatusInfo::NoCashLong as i64);
    // Closing realizes nothing more and returns the 100 posted.
    assert_eq!(journal.orders.size, [10.0, 10.0]);
    assert_eq!(logs.new_cash[2], -100.0);
    assert_eq!(logs.new_free_cash[2], -100.0);
}

#[test]
fn a_unit_margin_is_cash_per_contract_whatever_the_price() {
    let close = [100.0, 400.0];
    let size = [5.0, 5.0];
    let fills = simulate_from_orders(&FromOrders {
        margin_basis: FlexArray::scalar(MarginBasis::Unit),
        ..one_column(&close, &size, 15.0)
    })
    .unwrap();
    assert_eq!(fills.locked_cash, [75.0, 150.0]);
}

#[test]
fn a_hedged_spot_and_perpetual_pair_holds_its_value_in_one_account() {
    // Column 0 buys 5 at 100 for cash; column 1 shorts 5 at 100 on 10%
    // margin. At 120 the spot gains what the short loses.
    let close = [100.0, 120.0, 100.0, 120.0];
    let size = [5.0, N, -5.0, N];
    let types = [AccountType::Cash, AccountType::Margin];
    let margin = [N, 0.1];
    let journal = simulate_from_orders_logged(&FromOrders {
        n_rows: 2,
        n_cols: 2,
        init_cash: FlexArray::scalar(1000.0),
        close: FlexArray::full(&close, 2, 2).unwrap(),
        size: FlexArray::full(&size, 2, 2).unwrap(),
        account_type: FlexArray::per_column(&types, 2).unwrap(),
        initial_margin: FlexArray::per_column(&margin, 2).unwrap(),
        log: FlexArray::scalar(true),
        groups: Groups::from_lens(&[2], 2).unwrap(),
        ..FromOrders::default()
    })
    .unwrap();
    let (fills, logs) = (&journal.orders, &journal.logs);
    assert_eq!(fills.margin, [false, true]);
    // The spot's is the cash it put up, as in any run that keeps balances.
    assert_eq!(fills.locked_cash, [500.0, 50.0]);
    // Bar 0: 500 spent on the spot, 50 posted for the short.
    assert_eq!(logs.new_free_cash[..2], [500.0, 450.0]);
    // Bar 1: the short's 100 loss settles out of the shared cash, and the
    // account is still worth what it started with.
    assert_eq!(logs.cash[2..], [400.0, 400.0]);
    assert_eq!(logs.free_cash[2..], [350.0, 350.0]);
    assert_eq!(logs.value[2..], [1000.0, 1000.0]);
}

#[test]
fn update_value_carries_a_margin_fill_into_the_next_columns_target() {
    // Two margin columns sharing 1000, each targeting half the account.
    // The first pays 5 in fees; with `update_value` the second sees 995.
    let close = [100.0, 100.0];
    let size = [0.5, 0.5];
    let run = |update_value| {
        simulate_from_orders(&FromOrders {
            n_rows: 1,
            n_cols: 2,
            init_cash: FlexArray::scalar(1000.0),
            close: FlexArray::full(&close, 1, 2).unwrap(),
            size: FlexArray::full(&size, 1, 2).unwrap(),
            size_type: FlexArray::scalar(SizeType::TargetPercent),
            fees: FlexArray::scalar(0.01),
            account_type: FlexArray::scalar(AccountType::Margin),
            initial_margin: FlexArray::scalar(0.1),
            groups: Groups::from_lens(&[2], 2).unwrap(),
            update_value,
            ..FromOrders::default()
        })
        .unwrap()
    };
    assert_eq!(run(false).size, [5.0, 5.0]);
    assert_eq!(run(true).size, [5.0, 4.975]);
}

#[test]
fn a_margin_signal_settles_on_the_bars_between_its_signals() {
    // Column 0 goes long on bar 0 and is quiet on bar 1, where its 200 of
    // profit must already be free cash for column 1's entry to post.
    let close = [100.0, 110.0, 100.0, 100.0];
    let entries = [true, false, false, true];
    let exits = [false; 4];
    let fills = simulate_from_signals(&FromSignals {
        n_rows: 2,
        n_cols: 2,
        init_cash: FlexArray::scalar(1000.0),
        close: FlexArray::full(&close, 2, 2).unwrap(),
        entries: FlexArray::full(&entries, 2, 2).unwrap(),
        exits: FlexArray::full(&exits, 2, 2).unwrap(),
        account_type: FlexArray::scalar(AccountType::Margin),
        initial_margin: FlexArray::scalar(0.5),
        groups: Some(Groups::from_lens(&[2], 2).unwrap()),
        ..FromSignals::default()
    })
    .unwrap();
    // As much as the margin allows: 1000 posts half of 2000, then the 200
    // settled on bar 1 posts half of 400.
    assert_eq!(fills.col, [0, 1]);
    assert_eq!(fills.size, [20.0, 4.0]);
    assert_eq!(fills.locked_cash, [1000.0, 200.0]);
}

#[test]
fn an_account_type_that_changes_down_a_column_is_an_error() {
    let close = [100.0, 100.0];
    let size = [1.0, 1.0];
    let types = [AccountType::Cash, AccountType::Margin];
    let err = simulate_from_orders(&FromOrders {
        account_type: FlexArray::full(&types, 2, 1).unwrap(),
        ..one_column(&close, &size, 0.1)
    })
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("account_type changes down column 0"),
        "{err}"
    );
}

#[test]
fn a_cash_column_is_unchanged_inside_a_margin_run() {
    let (n_rows, n_cols) = (200, 4);
    let close = prices(n_rows, n_cols);
    let size = sizes(n_rows, n_cols);
    let types = [
        AccountType::Cash,
        AccountType::Margin,
        AccountType::Cash,
        AccountType::Margin,
    ];
    let base = || FromOrders {
        n_rows,
        n_cols,
        init_cash: FlexArray::scalar(500.0),
        close: FlexArray::full(&close, n_rows, n_cols).unwrap(),
        size: FlexArray::full(&size, n_rows, n_cols).unwrap(),
        fees: FlexArray::scalar(0.002),
        direction: FlexArray::scalar(Direction::Both),
        initial_margin: FlexArray::scalar(0.2),
        groups: Groups::ungrouped(n_cols),
        ..FromOrders::default()
    };
    let plain = simulate_from_orders(&base()).unwrap();
    let mixed = simulate_from_orders(&FromOrders {
        account_type: FlexArray::per_column(&types, n_cols).unwrap(),
        ..base()
    })
    .unwrap();
    assert_eq!(mixed.margin.len(), mixed.len());
    for col in [0, 2] {
        assert_eq!(
            column_bits(&plain, col),
            column_bits(&mixed, col),
            "column {col}"
        );
    }
    // And the margin columns did trade differently, or this proves nothing.
    assert_ne!(column_bits(&plain, 1), column_bits(&mixed, 1));

    let entries: Vec<bool> = (0..n_rows * n_cols).map(|k| k % 7 == 0).collect();
    let exits: Vec<bool> = (0..n_rows * n_cols).map(|k| k % 7 == 4).collect();
    let base = || FromSignals {
        n_rows,
        n_cols,
        close: FlexArray::full(&close, n_rows, n_cols).unwrap(),
        entries: FlexArray::full(&entries, n_rows, n_cols).unwrap(),
        exits: FlexArray::full(&exits, n_rows, n_cols).unwrap(),
        direction: FlexArray::scalar(Direction::Both),
        fees: FlexArray::scalar(0.001),
        initial_margin: FlexArray::scalar(0.2),
        ..FromSignals::default()
    };
    let plain = simulate_from_signals(&base()).unwrap();
    let mixed = simulate_from_signals(&FromSignals {
        account_type: FlexArray::per_column(&types, n_cols).unwrap(),
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
fn serial_and_parallel_margin_runs_agree_bit_for_bit() {
    let (n_rows, n_cols) = (150, 6);
    let close = prices(n_rows, n_cols);
    let size = sizes(n_rows, n_cols);
    let types = [
        AccountType::Margin,
        AccountType::Cash,
        AccountType::Margin,
        AccountType::Margin,
        AccountType::Cash,
        AccountType::Margin,
    ];
    let margin = [0.1, N, 0.5, N, N, 2.0];
    let basis = [
        MarginBasis::Notional,
        MarginBasis::Notional,
        MarginBasis::Notional,
        MarginBasis::Notional,
        MarginBasis::Notional,
        MarginBasis::Unit,
    ];
    let leverage = [1.0, 2.0, 1.0, 4.0, 1.0, 1.0];
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
            account_type: FlexArray::per_column(&types, n_cols).unwrap(),
            initial_margin: FlexArray::per_column(&margin, n_cols).unwrap(),
            margin_basis: FlexArray::per_column(&basis, n_cols).unwrap(),
            log: FlexArray::scalar(true),
            groups,
            ..FromOrders::default()
        };
        let serial = simulate_from_orders(&args).unwrap();
        assert!(serial.margin.contains(&true));
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
        account_type: FlexArray::per_column(&types, n_cols).unwrap(),
        initial_margin: FlexArray::per_column(&margin, n_cols).unwrap(),
        margin_basis: FlexArray::per_column(&basis, n_cols).unwrap(),
        ..FromSignals::default()
    };
    let serial = simulate_from_signals(&args).unwrap();
    assert!(serial.margin.contains(&true));
    for threads in [2, 3, 6] {
        let parallel = simulate_from_signals_parallel(&args, threads).unwrap();
        assert_eq!(all_bits(&serial), all_bits(&parallel), "{threads} threads");
    }
}
