//! Maintenance margin and liquidation through both loops.
//!
//! The module's tests pin down where one account breaches. These check the
//! loop's side: that the test runs on the bar's path before its orders, that
//! the fill is recorded and logged as a liquidation, that a flat account
//! left below zero and an isolated column's excess loss are written off,
//! that a stop which fires first saves the position, that a cross account
//! closes together and an isolated column alone, and that threads change
//! nothing. Prices and rates are chosen so every amount is exact.
#![allow(clippy::float_cmp, reason = "the hand-calculated amounts are exact")]

use alphax_core::flex::FlexArray;
use alphax_model::{AccountType, CashFlowKind, Direction, MarginMode, StatusInfo};
use alphax_records::OrderRecords;
use alphax_sim::groups::Groups;
use alphax_sim::{
    FromOrders, FromSignals, simulate_from_orders, simulate_from_orders_logged,
    simulate_from_orders_parallel, simulate_from_signals, simulate_from_signals_logged,
    simulate_from_signals_parallel,
};

const N: f64 = f64::NAN;
/// A sixteenth of the notional: a 100 long at 10% margin breaches at 96.
const MM: f64 = 1.0 / 16.0;
const LIQUIDATION: i8 = CashFlowKind::Liquidation as i8;

/// One perpetual of 1000 on 10% margin, a sixteenth maintenance, logged.
fn perpetual<'a>(
    open: &'a [f64],
    high: &'a [f64],
    low: &'a [f64],
    close: &'a [f64],
    size: &'a [f64],
) -> FromOrders<'a> {
    let n = close.len();
    FromOrders {
        n_rows: n,
        n_cols: 1,
        init_cash: FlexArray::scalar(1000.0),
        open: FlexArray::per_row(open, n).unwrap(),
        high: FlexArray::per_row(high, n).unwrap(),
        low: FlexArray::per_row(low, n).unwrap(),
        close: FlexArray::per_row(close, n).unwrap(),
        size: FlexArray::per_row(size, n).unwrap(),
        account_type: FlexArray::scalar(AccountType::Margin),
        initial_margin: FlexArray::scalar(0.1),
        maintenance_margin: FlexArray::scalar(MM),
        log: FlexArray::scalar(true),
        groups: Groups::ungrouped(1),
        ..FromOrders::default()
    }
}

/// Prices that wander enough to liquidate a leveraged column now and then.
fn prices(n_rows: usize, n_cols: usize) -> Vec<f64> {
    let mut close = Vec::with_capacity(n_rows * n_cols);
    for col in 0..n_cols {
        let mut price = 10.0 + col as f64;
        for i in 0..n_rows {
            let wiggle = ((i * 2_654_435_761 + col * 40_503) % 1000) as f64 / 1000.0 - 0.5;
            price = (price * (1.0 + wiggle * 0.08)).max(0.5);
            close.push(price);
        }
    }
    close
}

/// Every fill, its balances and its flags, and every cash record, as bits.
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
    bits.extend(records.isolated.iter().map(|&b| u64::from(b)));
    bits.extend(records.liquidation.iter().map(|&b| u64::from(b)));
    let cash = &records.cash;
    for k in 0..cash.len() {
        bits.extend([
            cash.col[k] as u64,
            cash.idx[k] as u64,
            cash.amount[k].to_bits(),
            cash.kind[k] as u64,
        ]);
    }
    bits
}

#[test]
fn a_long_is_liquidated_at_its_price_and_the_bars_order_still_executes() {
    // 100 long at 100 posts all 1000. On bar 1 the low reaches 90: the
    // equity 100p − 9000 meets the maintenance 6.25p at 96.
    let open = [100.0, 100.0, 95.0];
    let high = [100.0, 100.0, 96.0];
    let low = [100.0, 90.0, 94.0];
    let close = [100.0, 95.0, 95.0];
    let size = [100.0, N, 1.0];
    let journal = simulate_from_orders_logged(&FromOrders {
        liquidation_fee: FlexArray::scalar(0.0125),
        ..perpetual(&open, &high, &low, &close, &size)
    })
    .unwrap();
    let (fills, logs) = (&journal.orders, &journal.logs);
    assert_eq!(fills.idx, [0, 1, 2]);
    assert_eq!(fills.price, [100.0, 96.0, 95.0]);
    assert_eq!(fills.size, [100.0, 100.0, 1.0]);
    assert_eq!(fills.fees, [0.0, 120.0, 0.0]);
    assert_eq!(fills.liquidation, [false, true, false]);
    // Settled at the close of 95 first, then closed at 96: 1000 − 400 − 120.
    assert_eq!(logs.liquidation, [false, true, false, false]);
    assert_eq!(logs.idx, [0, 1, 1, 2]);
    assert_eq!(logs.cash[1], 500.0);
    assert_eq!(logs.new_cash[1], 480.0);
    assert_eq!(logs.new_position[1], 0.0);
    assert_eq!(logs.req_fees[1], 0.0125);
    // Nothing was owed, so nothing was written off; the bar-2 buy posts
    // 9.5 of the 480.
    assert!(fills.cash.is_empty());
    assert_eq!(logs.new_free_cash[3], 470.5);
}

#[test]
fn a_gap_through_the_bankruptcy_price_is_written_off() {
    // Opening at 80 closes the long there: 2000 lost on 1000. The account
    // holds nothing more, so the insurance fund takes the 1000 it owes.
    let open = [100.0, 80.0, 80.0];
    let high = [100.0, 80.0, 80.0];
    let low = [100.0, 75.0, 80.0];
    let close = [100.0, 78.0, 80.0];
    let size = [100.0, N, 1.0];
    let journal =
        simulate_from_orders_logged(&perpetual(&open, &high, &low, &close, &size)).unwrap();
    let (fills, logs) = (&journal.orders, &journal.logs);
    assert_eq!(fills.price, [100.0, 80.0]);
    assert_eq!(fills.cash.idx, [1]);
    assert_eq!(fills.cash.amount, [1000.0]);
    assert_eq!(fills.cash.kind, [LIQUIDATION]);
    // Bar 2 opens with nothing and cannot buy. Row 1 of the log is the
    // liquidation, row 2 bar 1's own order.
    assert_eq!(logs.liquidation, [false, true, false, false]);
    assert_eq!(logs.cash[3], 0.0);
    assert_eq!(logs.res_status_info[3], StatusInfo::NoCashLong as i64);
}

#[test]
fn a_short_is_tested_at_the_high() {
    // 100 short at 100: the equity 11000 − 100p meets 6.25p at 103.53..,
    // so a high of 103 survives and one of 104 does not.
    let survive = simulate_from_orders(&perpetual(
        &[100.0, 100.0],
        &[100.0, 103.0],
        &[100.0, 99.0],
        &[100.0, 101.0],
        &[-100.0, N],
    ))
    .unwrap();
    assert_eq!(survive.liquidation, [false]);
    let fail = simulate_from_orders(&perpetual(
        &[100.0, 100.0],
        &[100.0, 104.0],
        &[100.0, 99.0],
        &[100.0, 101.0],
        &[-100.0, N],
    ))
    .unwrap();
    assert_eq!(fail.liquidation, [false, true]);
    let level = 11000.0 / 106.25;
    assert!((fail.price[1] - level).abs() < 1e-9, "{}", fail.price[1]);
}

#[test]
fn without_a_maintenance_margin_or_a_breach_nothing_changes() {
    let open = [100.0, 100.0, 95.0, 90.0];
    let high = [100.0, 100.0, 99.0, 92.0];
    let low = [100.0, 97.0, 94.0, 85.0];
    let close = [100.0, 98.0, 95.0, 91.0];
    let size = [50.0, N, -20.0, 10.0];
    let plain = simulate_from_orders(&FromOrders {
        maintenance_margin: FlexArray::scalar(0.0),
        ..perpetual(&open, &high, &low, &close, &size)
    })
    .unwrap();
    assert_eq!(plain.liquidation, [false, false, false]);
    // A maintenance margin the account never breaches changes nothing.
    let held = simulate_from_orders(&FromOrders {
        maintenance_margin: FlexArray::scalar(0.01),
        ..perpetual(&open, &high, &low, &close, &size)
    })
    .unwrap();
    assert_eq!(all_bits(&plain), all_bits(&held));
}

/// One long perpetual signal, entered on bar 0 and stopped `sl_stop` below.
fn signal_run(sl_stop: f64) -> alphax_records::Journal {
    let open = [100.0, 100.0];
    let high = [100.0, 100.0];
    let low = [100.0, 90.0];
    let close = [100.0, 99.0];
    let entries = [true, false];
    let exits = [false, false];
    simulate_from_signals_logged(&FromSignals {
        n_rows: 2,
        n_cols: 1,
        init_cash: FlexArray::scalar(1000.0),
        open: FlexArray::per_row(&open, 2).unwrap(),
        high: FlexArray::per_row(&high, 2).unwrap(),
        low: FlexArray::per_row(&low, 2).unwrap(),
        close: FlexArray::per_row(&close, 2).unwrap(),
        entries: FlexArray::per_row(&entries, 2).unwrap(),
        exits: FlexArray::per_row(&exits, 2).unwrap(),
        size: FlexArray::scalar(100.0),
        sl_stop: FlexArray::scalar(sl_stop),
        account_type: FlexArray::scalar(AccountType::Margin),
        initial_margin: FlexArray::scalar(0.1),
        maintenance_margin: FlexArray::scalar(MM),
        log: FlexArray::scalar(true),
        ..FromSignals::default()
    })
    .unwrap()
}

#[test]
fn a_stop_inside_the_liquidation_price_closes_the_position_first() {
    // Stopped at 97, before the path reaches 96.
    let journal = signal_run(0.03);
    assert_eq!(journal.orders.price, [100.0, 97.0]);
    assert_eq!(journal.orders.liquidation, [false, false]);
}

#[test]
fn a_stop_beyond_the_liquidation_price_does_not_save_it() {
    // Stopped at 92, but the path reaches 96 first: liquidated there, and
    // the stop's order, sized against a position that is gone, dropped.
    let journal = signal_run(0.08);
    assert_eq!(journal.orders.price, [100.0, 96.0]);
    assert_eq!(journal.orders.liquidation, [false, true]);
    assert_eq!(journal.logs.liquidation, [false, true]);
    assert_eq!(
        journal.logs.len(),
        2,
        "the stop order never reached the kernel"
    );
}

#[test]
fn a_quiet_signal_bar_is_tested_too() {
    let journal = signal_run(f64::NAN);
    assert_eq!(journal.orders.liquidation, [false, true]);
}

#[test]
fn a_cross_account_closes_every_position_together() {
    // Two longs of 50 at 100 share 1000, each posting 500. Column 0's bar
    // falls to 80, column 1's only to 99; the account breaches on the way
    // and both are closed at their own point of the path.
    let open = [100.0, 100.0, 100.0, 100.0];
    let high = [100.0, 100.0, 100.0, 100.0];
    let low = [100.0, 80.0, 100.0, 99.0];
    let close = [100.0, 90.0, 100.0, 99.5];
    let size = [50.0, N, 50.0, N];
    let journal = simulate_from_orders_logged(&FromOrders {
        n_rows: 2,
        n_cols: 2,
        open: FlexArray::full(&open, 2, 2).unwrap(),
        high: FlexArray::full(&high, 2, 2).unwrap(),
        low: FlexArray::full(&low, 2, 2).unwrap(),
        close: FlexArray::full(&close, 2, 2).unwrap(),
        size: FlexArray::full(&size, 2, 2).unwrap(),
        groups: Groups::from_lens(&[2], 2).unwrap(),
        ..perpetual(&open[..2], &high[..2], &low[..2], &close[..2], &size[..2])
    })
    .unwrap();
    let fills = &journal.orders;
    assert_eq!(fills.liquidation, [false, false, true, true]);
    assert_eq!(fills.col, [0, 1, 0, 1]);
    // Along the paths, 100 → 80 and 100 → 99, the equity 1000 − 1050α meets
    // the maintenance 625 − 65.625α at α = 8/21.
    let alpha = 8.0 / 21.0;
    assert!((fills.price[2] - (100.0 - 20.0 * alpha)).abs() < 1e-12);
    assert!((fills.price[3] - (100.0 - alpha)).abs() < 1e-12);
    let equity = 50.0 * (fills.price[2] - 100.0) + 50.0 * (fills.price[3] - 100.0) + 1000.0;
    let required = MM * 50.0 * (fills.price[2] + fills.price[3]);
    assert!((equity - required).abs() < 1e-9, "{equity} vs {required}");
}

#[test]
fn an_isolated_column_is_liquidated_alone_and_never_settled() {
    // Column 0 is isolated, column 1 cross; both buy 100 at 100 on 10%
    // margin out of 3000. Column 0's bar falls to 90, column 1's to 99.
    let open = [100.0, 100.0, 100.0, 100.0];
    let high = [100.0, 100.0, 100.0, 100.0];
    let low = [100.0, 90.0, 100.0, 99.0];
    let close = [100.0, 95.0, 100.0, 99.0];
    let size = [100.0, N, 100.0, N];
    let modes = [MarginMode::Isolated, MarginMode::Cross];
    let journal = simulate_from_orders_logged(&FromOrders {
        n_rows: 2,
        n_cols: 2,
        init_cash: FlexArray::scalar(3000.0),
        open: FlexArray::full(&open, 2, 2).unwrap(),
        high: FlexArray::full(&high, 2, 2).unwrap(),
        low: FlexArray::full(&low, 2, 2).unwrap(),
        close: FlexArray::full(&close, 2, 2).unwrap(),
        size: FlexArray::full(&size, 2, 2).unwrap(),
        margin_mode: FlexArray::per_column(&modes, 2).unwrap(),
        groups: Groups::from_lens(&[2], 2).unwrap(),
        ..perpetual(&open[..2], &high[..2], &low[..2], &close[..2], &size[..2])
    })
    .unwrap();
    let (fills, logs) = (&journal.orders, &journal.logs);
    assert_eq!(fills.isolated, [true, false, true]);
    assert_eq!(fills.liquidation, [false, false, true]);
    assert_eq!(fills.price[2], 96.0);
    // Bar 1: only the cross long settles (−100); the isolated one carries
    // its loss until it closes. Liquidated at 96 it realizes −400 and
    // returns its 1000, which covers it: nothing is written off.
    assert_eq!(logs.cash[2], 2900.0);
    assert_eq!(logs.free_cash[2], 900.0);
    assert_eq!(logs.new_cash[2], 2500.0);
    assert_eq!(logs.new_free_cash[2], 1500.0);
    assert!(fills.cash.is_empty());
}

#[test]
fn an_isolated_column_loses_at_most_its_margin() {
    // Opening at 85 is 1500 lost on 1000 posted: the 500 beyond it is
    // written off to the column, and the account keeps the rest.
    let open = [100.0, 85.0];
    let high = [100.0, 86.0];
    let low = [100.0, 84.0];
    let close = [100.0, 85.0];
    let size = [100.0, N];
    let journal = simulate_from_orders_logged(&FromOrders {
        init_cash: FlexArray::scalar(2000.0),
        margin_mode: FlexArray::scalar(MarginMode::Isolated),
        ..perpetual(&open, &high, &low, &close, &size)
    })
    .unwrap();
    let fills = &journal.orders;
    assert_eq!(fills.price, [100.0, 85.0]);
    assert_eq!(fills.cash.amount, [500.0]);
    assert_eq!(fills.cash.col, [0]);
    assert_eq!(journal.logs.new_cash[1], 500.0);
}

#[test]
fn a_cross_loss_does_not_eat_an_isolated_positions_margin() {
    // 1500 holds an isolated long of 50 (500 posted) and a cross long of 100
    // (1000 posted). The cross long gaps to 80: 2000 lost on 1000. Its
    // shortfall is written off against the cash net of the isolated margin,
    // which stays the isolated position's.
    let open = [100.0, 100.0, 100.0, 80.0];
    let high = [100.0, 100.0, 100.0, 80.0];
    let low = [100.0, 100.0, 100.0, 80.0];
    let close = [100.0, 100.0, 100.0, 80.0];
    let size = [50.0, N, 100.0, N];
    let modes = [MarginMode::Isolated, MarginMode::Cross];
    let journal = simulate_from_orders_logged(&FromOrders {
        n_rows: 2,
        n_cols: 2,
        init_cash: FlexArray::scalar(1500.0),
        open: FlexArray::full(&open, 2, 2).unwrap(),
        high: FlexArray::full(&high, 2, 2).unwrap(),
        low: FlexArray::full(&low, 2, 2).unwrap(),
        close: FlexArray::full(&close, 2, 2).unwrap(),
        size: FlexArray::full(&size, 2, 2).unwrap(),
        margin_mode: FlexArray::per_column(&modes, 2).unwrap(),
        groups: Groups::from_lens(&[2], 2).unwrap(),
        ..perpetual(&open[..2], &high[..2], &low[..2], &close[..2], &size[..2])
    })
    .unwrap();
    let (fills, logs) = (&journal.orders, &journal.logs);
    assert_eq!(fills.liquidation, [false, false, true]);
    assert_eq!(fills.col[2], 1);
    assert_eq!(fills.cash.amount, [1000.0]);
    assert_eq!(fills.cash.col, [0]);
    // What is left is the isolated position's margin, and nothing free.
    let last = logs.len() - 1;
    assert_eq!(logs.cash[last], 500.0);
    assert_eq!(logs.free_cash[last], 0.0);
    assert_eq!(logs.position[last - 1], 50.0);
}

#[test]
fn a_hedge_is_safe_at_one_mark_price() {
    // 5000 of spot bought out of 5600 and a perpetual short of the same 50
    // on 10% margin. The spot's low and the perpetual's high on the same
    // bar would leave 600 against a maintenance of 750; a mark price for
    // both puts them at one instant, where they cancel.
    let open = [100.0, 100.0, 100.0, 100.0];
    let high = [100.0, 150.0, 100.0, 150.0];
    let low = [100.0, 50.0, 100.0, 50.0];
    let close = [100.0, 100.0, 100.0, 100.0];
    let size = [50.0, N, -50.0, N];
    let types = [AccountType::Cash, AccountType::Margin];
    let run = |mark: FlexArray<'static, f64>| {
        simulate_from_orders(&FromOrders {
            n_rows: 2,
            n_cols: 2,
            init_cash: FlexArray::scalar(5600.0),
            open: FlexArray::full(&open, 2, 2).unwrap(),
            high: FlexArray::full(&high, 2, 2).unwrap(),
            low: FlexArray::full(&low, 2, 2).unwrap(),
            close: FlexArray::full(&close, 2, 2).unwrap(),
            size: FlexArray::full(&size, 2, 2).unwrap(),
            account_type: FlexArray::per_column(&types, 2).unwrap(),
            initial_margin: FlexArray::scalar(0.1),
            maintenance_margin: FlexArray::scalar(0.1),
            mark_price: mark,
            groups: Groups::from_lens(&[2], 2).unwrap(),
            ..FromOrders::default()
        })
        .unwrap()
    };
    assert_eq!(
        run(FlexArray::Scalar(f64::NAN)).liquidation,
        [false, false, true]
    );
    assert_eq!(run(FlexArray::Scalar(110.0)).liquidation, [false, false]);
}

#[test]
fn a_margin_mode_that_changes_down_a_column_is_an_error() {
    let modes = [MarginMode::Cross, MarginMode::Isolated];
    let err = simulate_from_orders(&FromOrders {
        margin_mode: FlexArray::per_row(&modes, 2).unwrap(),
        ..perpetual(
            &[100.0, 100.0],
            &[100.0, 100.0],
            &[100.0, 100.0],
            &[100.0, 100.0],
            &[1.0, N],
        )
    })
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("margin_mode changes down column 0"),
        "{err}"
    );
}

#[test]
fn serial_and_parallel_runs_with_liquidations_agree_bit_for_bit() {
    let (n_rows, n_cols) = (200, 6);
    let close = prices(n_rows, n_cols);
    // Each bar opens at the previous close, except for a gap down and a gap
    // up every 13 bars, past where a leveraged position can pay.
    let open: Vec<f64> = (0..n_rows * n_cols)
        .map(|k| match (k % n_rows, k % n_rows % 13) {
            (0, _) => close[k],
            (_, 6) => close[k] * 0.85,
            (_, 7) => close[k] * 1.15,
            _ => close[k - 1],
        })
        .collect();
    let high: Vec<f64> = (0..close.len())
        .map(|k| open[k].max(close[k]) * 1.01)
        .collect();
    let low: Vec<f64> = (0..close.len())
        .map(|k| open[k].min(close[k]) * 0.99)
        .collect();
    let size: Vec<f64> = (0..n_rows * n_cols)
        .map(|k| match k % 7 {
            0 => 60.0,
            3 => -90.0,
            _ => N,
        })
        .collect();
    let types = [
        AccountType::Margin,
        AccountType::Cash,
        AccountType::Margin,
        AccountType::Margin,
        AccountType::Margin,
        AccountType::Cash,
    ];
    let modes = [
        MarginMode::Cross,
        MarginMode::Cross,
        MarginMode::Isolated,
        MarginMode::Cross,
        MarginMode::Isolated,
        MarginMode::Cross,
    ];
    let args = |groups| FromOrders {
        n_rows,
        n_cols,
        init_cash: FlexArray::scalar(100.0),
        open: FlexArray::full(&open, n_rows, n_cols).unwrap(),
        high: FlexArray::full(&high, n_rows, n_cols).unwrap(),
        low: FlexArray::full(&low, n_rows, n_cols).unwrap(),
        close: FlexArray::full(&close, n_rows, n_cols).unwrap(),
        size: FlexArray::full(&size, n_rows, n_cols).unwrap(),
        fees: FlexArray::scalar(0.001),
        direction: FlexArray::scalar(Direction::Both),
        account_type: FlexArray::per_column(&types, n_cols).unwrap(),
        margin_mode: FlexArray::per_column(&modes, n_cols).unwrap(),
        initial_margin: FlexArray::scalar(0.1),
        maintenance_margin: FlexArray::scalar(0.05),
        liquidation_fee: FlexArray::scalar(0.005),
        cash_deposits: FlexArray::scalar(5.0),
        groups,
        ..FromOrders::default()
    };
    for groups in [
        Groups::ungrouped(n_cols),
        Groups::from_lens(&[2, 3, 1], n_cols).unwrap(),
    ] {
        let args = args(groups);
        let serial = simulate_from_orders(&args).unwrap();
        assert!(serial.liquidation.contains(&true), "nothing was liquidated");
        assert!(
            serial.cash.kind.contains(&LIQUIDATION),
            "nothing was written off"
        );
        for threads in [2, 3, 6] {
            let parallel = simulate_from_orders_parallel(&args, threads).unwrap();
            assert_eq!(all_bits(&serial), all_bits(&parallel), "{threads} threads");
        }
    }

    let entries: Vec<bool> = (0..n_rows * n_cols).map(|k| k % 9 == 0).collect();
    let exits: Vec<bool> = (0..n_rows * n_cols).map(|k| k % 9 == 5).collect();
    let args = FromSignals {
        n_rows,
        n_cols,
        init_cash: FlexArray::scalar(100.0),
        open: FlexArray::full(&open, n_rows, n_cols).unwrap(),
        high: FlexArray::full(&high, n_rows, n_cols).unwrap(),
        low: FlexArray::full(&low, n_rows, n_cols).unwrap(),
        close: FlexArray::full(&close, n_rows, n_cols).unwrap(),
        entries: FlexArray::full(&entries, n_rows, n_cols).unwrap(),
        exits: FlexArray::full(&exits, n_rows, n_cols).unwrap(),
        direction: FlexArray::scalar(Direction::Both),
        sl_stop: FlexArray::scalar(0.04),
        account_type: FlexArray::per_column(&types, n_cols).unwrap(),
        margin_mode: FlexArray::per_column(&modes, n_cols).unwrap(),
        initial_margin: FlexArray::scalar(0.1),
        maintenance_margin: FlexArray::scalar(0.05),
        groups: Some(Groups::from_lens(&[3, 3], n_cols).unwrap()),
        ..FromSignals::default()
    };
    let serial = simulate_from_signals(&args).unwrap();
    assert!(serial.liquidation.contains(&true), "nothing was liquidated");
    for threads in [2, 3] {
        let parallel = simulate_from_signals_parallel(&args, threads).unwrap();
        assert_eq!(all_bits(&serial), all_bits(&parallel), "{threads} threads");
    }
}
