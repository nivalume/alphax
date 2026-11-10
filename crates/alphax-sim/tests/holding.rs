//! Holding charges through both loops: funding, financing, interest on what a
//! long borrowed and the fee for the shares a short borrowed.
//!
//! The module's own tests pin down one bar's arithmetic. These check the
//! loop's side: that a charge lands at the bar's close on the position the
//! bar's orders left, on quiet bars too, in a shared account's cash, uncapped,
//! and that threads change nothing. Rates and prices are powers of two, so
//! every amount is exact.
#![allow(clippy::float_cmp, reason = "the hand-calculated amounts are exact")]

use alphax_core::flex::FlexArray;
use alphax_model::{AccountType, CashFlowKind, Direction, StatusInfo};
use alphax_records::OrderRecords;
use alphax_sim::groups::Groups;
use alphax_sim::{
    FromOrders, FromSignals, simulate_from_orders, simulate_from_orders_logged,
    simulate_from_orders_parallel, simulate_from_signals, simulate_from_signals_parallel,
};

const N: f64 = f64::NAN;
/// 2⁻¹⁰: a 1024 notional pays exactly 1.
const R: f64 = 1.0 / 1024.0;

const FUNDING: i8 = CashFlowKind::Funding as i8;
const FINANCING: i8 = CashFlowKind::Financing as i8;
const INTEREST: i8 = CashFlowKind::Interest as i8;
const BORROW_FEE: i8 = CashFlowKind::BorrowFee as i8;

/// One column of 1000, logged.
fn one_column<'a>(close: &'a [f64], size: &'a [f64]) -> FromOrders<'a> {
    FromOrders {
        n_rows: close.len(),
        n_cols: 1,
        init_cash: FlexArray::scalar(1000.0),
        close: FlexArray::full(close, close.len(), 1).unwrap(),
        size: FlexArray::full(size, close.len(), 1).unwrap(),
        log: FlexArray::scalar(true),
        groups: Groups::ungrouped(1),
        ..FromOrders::default()
    }
}

/// A margin column posting half the notional.
fn perpetual<'a>(close: &'a [f64], size: &'a [f64]) -> FromOrders<'a> {
    FromOrders {
        account_type: FlexArray::scalar(AccountType::Margin),
        initial_margin: FlexArray::scalar(0.5),
        ..one_column(close, size)
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

/// Every fill and every cash record as raw bits, so equality is bit-identity.
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

/// The fills alone, without the balances a levered run adds.
fn fill_bits(records: &OrderRecords) -> Vec<u64> {
    let mut plain = records.clone();
    plain.debt.clear();
    plain.locked_cash.clear();
    plain.margin.clear();
    plain.cash = alphax_records::CashRecords::new();
    all_bits(&plain)
}

#[test]
fn funding_is_paid_at_the_close_on_the_position_the_bars_orders_left() {
    // Long 8 at 128 on half margin; funding settles on bars 0, 2 and 3.
    let close = [128.0, 128.0, 160.0, 160.0];
    let size = [8.0, N, N, -8.0];
    let funding = [R, N, R, R];
    let journal = simulate_from_orders_logged(&FromOrders {
        funding_rate: FlexArray::per_row(&funding, 4).unwrap(),
        ..perpetual(&close, &size)
    })
    .unwrap();
    let (cash, logs) = (&journal.orders.cash, &journal.logs);
    // Bar 0 pays on the long its own order opened; bar 3's order closes it
    // first, so bar 3 pays nothing.
    assert_eq!(cash.idx, [0, 2]);
    assert_eq!(cash.amount, [-1.0, -1.25]);
    assert_eq!(cash.kind, [FUNDING, FUNDING]);
    // The bar-2 profit of 256 settles before the order; the funding is out
    // of the next bar's cash and free cash.
    assert_eq!(logs.cash, [1000.0, 999.0, 1255.0, 1253.75]);
    assert_eq!(logs.free_cash, [1000.0, 487.0, 743.0, 741.75]);
    assert_eq!(logs.new_cash[3], 1253.75);
    assert_eq!(logs.new_free_cash[3], 1253.75);
}

#[test]
fn a_short_receives_funding_on_its_value_at_the_mark_price() {
    let close = [128.0, 128.0];
    let mark = [256.0, 256.0];
    let size = [-8.0, 8.0];
    let funding = [R, N];
    let fills = simulate_from_orders(&FromOrders {
        funding_rate: FlexArray::per_row(&funding, 2).unwrap(),
        mark_price: FlexArray::per_row(&mark, 2).unwrap(),
        ..perpetual(&close, &size)
    })
    .unwrap();
    assert_eq!(fills.cash.amount, [2.0]);
    assert_eq!(fills.cash.kind, [FUNDING]);
}

#[test]
fn a_hedged_pair_collects_the_shorts_funding_in_one_account() {
    // Column 0 holds 8 spot, column 1 is short 8 perpetuals on a quarter
    // margin. Only the perpetual has a funding rate.
    let close = [128.0, 160.0, 128.0, 128.0, 160.0, 128.0];
    let size = [8.0, N, N, -8.0, N, N];
    let funding = [N, N, N, R, R, R];
    let types = [AccountType::Cash, AccountType::Margin];
    let journal = simulate_from_orders_logged(&FromOrders {
        n_rows: 3,
        n_cols: 2,
        init_cash: FlexArray::scalar(2000.0),
        close: FlexArray::full(&close, 3, 2).unwrap(),
        size: FlexArray::full(&size, 3, 2).unwrap(),
        account_type: FlexArray::per_column(&types, 2).unwrap(),
        initial_margin: FlexArray::scalar(0.25),
        funding_rate: FlexArray::full(&funding, 3, 2).unwrap(),
        log: FlexArray::scalar(true),
        groups: Groups::from_lens(&[2], 2).unwrap(),
        ..FromOrders::default()
    })
    .unwrap();
    let (cash, logs) = (&journal.orders.cash, &journal.logs);
    assert_eq!(cash.col, [1, 1, 1]);
    assert_eq!(cash.amount, [1.0, 1.25, 1.0]);
    // The legs cancel, so the account is worth its funding so far: each bar
    // opens on the previous bars' payments.
    assert_eq!(logs.value[..2], [2000.0, 2000.0]);
    assert_eq!(logs.value[2..4], [2001.0, 2001.0]);
    assert_eq!(logs.value[4..], [2002.25, 2002.25]);
}

#[test]
fn funding_between_a_long_and_a_short_of_one_size_sums_to_zero() {
    // A conservation check: funding moves money from one side to the other
    // and creates none. Equal positions at one mark price and one rate, in
    // separate accounts and in one, pay and receive exactly the same.
    let n_rows = 60;
    let close: Vec<f64> = prices(n_rows, 1);
    let mark: Vec<f64> = close.iter().map(|c| c * 1.003).collect();
    let funding: Vec<f64> = (0..n_rows)
        .map(|i| match i % 4 {
            0 => 1e-4 * ((i % 7) as f64 - 3.0),
            2 => N,
            _ => 0.0,
        })
        .collect();
    let both: Vec<f64> = [close.clone(), close.clone()].concat();
    let both_mark: Vec<f64> = [mark.clone(), mark].concat();
    let mut size = vec![N; 2 * n_rows];
    size[0] = 8.0;
    size[n_rows] = -8.0;
    for groups in [Groups::ungrouped(2), Groups::from_lens(&[2], 2).unwrap()] {
        let fills = simulate_from_orders(&FromOrders {
            n_rows,
            n_cols: 2,
            init_cash: FlexArray::scalar(1000.0),
            close: FlexArray::full(&both, n_rows, 2).unwrap(),
            size: FlexArray::full(&size, n_rows, 2).unwrap(),
            account_type: FlexArray::scalar(AccountType::Margin),
            initial_margin: FlexArray::scalar(0.1),
            funding_rate: FlexArray::per_row(&funding, n_rows).unwrap(),
            mark_price: FlexArray::full(&both_mark, n_rows, 2).unwrap(),
            groups,
            ..FromOrders::default()
        })
        .unwrap();
        let cash = &fills.cash;
        assert!(cash.len() > 20, "too few funding payments to mean anything");
        let mut net = vec![0.0; n_rows];
        for k in 0..cash.len() {
            assert_eq!(cash.kind[k], FUNDING);
            net[cash.idx[k] as usize] += cash.amount[k];
        }
        assert!(net.iter().all(|&v| v == 0.0), "{net:?}");
    }
}

#[test]
fn financing_charges_each_side_its_own_rate() {
    // Long and short 8 at 128, unshared. The long pays 2⁻¹⁰ a bar; the short's
    // rate is negative, so it is paid 2⁻¹¹ a bar.
    let close = [128.0, 128.0, 128.0, 128.0];
    let size = [8.0, N, -8.0, N];
    let short = -R / 2.0;
    let fills = simulate_from_orders(&FromOrders {
        n_rows: 2,
        n_cols: 2,
        init_cash: FlexArray::scalar(1000.0),
        close: FlexArray::full(&close, 2, 2).unwrap(),
        size: FlexArray::full(&size, 2, 2).unwrap(),
        account_type: FlexArray::scalar(AccountType::Margin),
        initial_margin: FlexArray::scalar(0.5),
        long_financing_rate: FlexArray::scalar(R),
        short_financing_rate: FlexArray::scalar(short),
        groups: Groups::ungrouped(2),
        ..FromOrders::default()
    })
    .unwrap();
    assert_eq!(fills.cash.col, [0, 0, 1, 1]);
    assert_eq!(fills.cash.amount, [-1.0, -1.0, 0.5, 0.5]);
    assert_eq!(fills.cash.kind, [FINANCING; 4]);
}

#[test]
fn a_leveraged_long_pays_interest_on_what_it_borrowed() {
    // 1000 of own cash buys 1500 at leverage 2, lazily: 500 is borrowed.
    // A margin column beside it borrows nothing and pays nothing.
    let close = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0];
    let size = [15.0, N, -15.0, 15.0, N, -15.0];
    let types = [AccountType::Cash, AccountType::Margin];
    let fills = simulate_from_orders(&FromOrders {
        n_rows: 3,
        n_cols: 2,
        init_cash: FlexArray::scalar(1000.0),
        close: FlexArray::full(&close, 3, 2).unwrap(),
        size: FlexArray::full(&size, 3, 2).unwrap(),
        leverage: FlexArray::scalar(2.0),
        account_type: FlexArray::per_column(&types, 2).unwrap(),
        debt_rate: FlexArray::scalar(1.0 / 128.0),
        groups: Groups::ungrouped(2),
        ..FromOrders::default()
    })
    .unwrap();
    assert_eq!(fills.debt[0], 500.0);
    // Bars 0 and 1; bar 2's sale repays the debt first.
    assert_eq!(fills.cash.col, [0, 0]);
    assert_eq!(fills.cash.idx, [0, 1]);
    assert_eq!(fills.cash.amount, [-3.906_25, -3.906_25]);
    assert_eq!(fills.cash.kind, [INTEREST, INTEREST]);
}

#[test]
fn a_cash_short_pays_its_borrow_fee_and_otherwise_trades_as_before() {
    let close = [128.0, 128.0, 128.0];
    let size = [-8.0, N, 8.0];
    let plain = simulate_from_orders(&one_column(&close, &size)).unwrap();
    let charged = simulate_from_orders(&FromOrders {
        borrow_rate: FlexArray::scalar(R),
        ..one_column(&close, &size)
    })
    .unwrap();
    assert_eq!(charged.cash.amount, [-1.0, -1.0]);
    assert_eq!(charged.cash.kind, [BORROW_FEE, BORROW_FEE]);
    // A charge is owed whatever the balance, so the run keeps balances as one
    // that may borrow does; the fills are the same.
    assert!(plain.debt.is_empty());
    assert_eq!(charged.debt, [1024.0, 0.0]);
    assert_eq!(fill_bits(&plain), fill_bits(&charged));
}

#[test]
fn a_charge_is_not_capped_and_can_leave_the_account_owing() {
    // 10 posts a tenth of 100; funding at half the notional takes 50.
    let close = [100.0, 100.0, 100.0];
    let size = [1.0, N, 1.0];
    let funding = [0.5, N, N];
    let journal = simulate_from_orders_logged(&FromOrders {
        init_cash: FlexArray::scalar(10.0),
        initial_margin: FlexArray::scalar(0.1),
        funding_rate: FlexArray::per_row(&funding, 3).unwrap(),
        ..perpetual(&close, &size)
    })
    .unwrap();
    let logs = &journal.logs;
    assert_eq!(journal.orders.cash.amount, [-50.0]);
    assert_eq!(logs.cash[1], -40.0);
    assert_eq!(logs.free_cash[1], -50.0);
    assert_eq!(logs.res_status_info[2], StatusInfo::NoCashLong as i64);
}

#[test]
fn a_quiet_signal_bar_is_charged_too() {
    let close = [128.0, 128.0, 128.0, 128.0];
    let entries = [true, false, false, false];
    let exits = [false; 4];
    let fills = simulate_from_signals(&FromSignals {
        n_rows: 4,
        n_cols: 1,
        init_cash: FlexArray::scalar(1024.0),
        close: FlexArray::per_row(&close, 4).unwrap(),
        entries: FlexArray::per_row(&entries, 4).unwrap(),
        exits: FlexArray::per_row(&exits, 4).unwrap(),
        size: FlexArray::scalar(8.0),
        account_type: FlexArray::scalar(AccountType::Margin),
        initial_margin: FlexArray::scalar(0.5),
        funding_rate: FlexArray::scalar(R),
        ..FromSignals::default()
    })
    .unwrap();
    assert_eq!(fills.len(), 1);
    assert_eq!(fills.cash.idx, [0, 1, 2, 3]);
    assert_eq!(fills.cash.amount, [-1.0; 4]);
}

#[test]
fn zero_rates_change_no_fill() {
    // Rates given per column are on, so the run keeps balances, but rates of
    // zero charge nothing: the fills are the run without them, bit for bit.
    let (n_rows, n_cols) = (120, 4);
    let close = prices(n_rows, n_cols);
    let size = sizes(n_rows, n_cols);
    let zeros = [0.0; 4];
    let base = || FromOrders {
        n_rows,
        n_cols,
        init_cash: FlexArray::scalar(500.0),
        close: FlexArray::full(&close, n_rows, n_cols).unwrap(),
        size: FlexArray::full(&size, n_rows, n_cols).unwrap(),
        fees: FlexArray::scalar(0.002),
        direction: FlexArray::scalar(Direction::Both),
        groups: Groups::from_lens(&[1, 3], n_cols).unwrap(),
        ..FromOrders::default()
    };
    let plain = simulate_from_orders(&base()).unwrap();
    let zero = simulate_from_orders(&FromOrders {
        funding_rate: FlexArray::per_column(&zeros, n_cols).unwrap(),
        borrow_rate: FlexArray::per_column(&zeros, n_cols).unwrap(),
        ..base()
    })
    .unwrap();
    assert!(zero.cash.is_empty());
    assert_eq!(zero.debt.len(), zero.len());
    assert_eq!(fill_bits(&plain), fill_bits(&zero));
}

#[test]
fn an_infinite_rate_is_an_error() {
    let close = [128.0, 128.0];
    let size = [8.0, N];
    let rate = [N, f64::INFINITY];
    let err = simulate_from_orders(&FromOrders {
        debt_rate: FlexArray::scalar(0.0),
        long_financing_rate: FlexArray::per_row(&rate, 2).unwrap(),
        ..perpetual(&close, &size)
    })
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("long_financing_rate is infinite at row 1"),
        "{err}"
    );
}

#[test]
fn serial_and_parallel_runs_with_charges_agree_bit_for_bit() {
    let (n_rows, n_cols) = (150, 6);
    let close = prices(n_rows, n_cols);
    let size = sizes(n_rows, n_cols);
    let types = [
        AccountType::Margin,
        AccountType::Cash,
        AccountType::Margin,
        AccountType::Cash,
        AccountType::Cash,
        AccountType::Margin,
    ];
    let leverage = [1.0, 2.0, 1.0, 3.0, 1.0, 1.0];
    let funding: Vec<f64> = (0..n_rows)
        .map(|i| {
            if i % 8 == 0 {
                1e-4 * (i % 3) as f64 - 1e-4
            } else {
                N
            }
        })
        .collect();
    let mark: Vec<f64> = close.iter().map(|c| c * 1.001).collect();
    let charged = |groups| FromOrders {
        n_rows,
        n_cols,
        init_cash: FlexArray::scalar(800.0),
        close: FlexArray::full(&close, n_rows, n_cols).unwrap(),
        size: FlexArray::full(&size, n_rows, n_cols).unwrap(),
        fees: FlexArray::scalar(0.001),
        direction: FlexArray::scalar(Direction::Both),
        leverage: FlexArray::per_column(&leverage, n_cols).unwrap(),
        account_type: FlexArray::per_column(&types, n_cols).unwrap(),
        initial_margin: FlexArray::scalar(0.2),
        funding_rate: FlexArray::per_row(&funding, n_rows).unwrap(),
        long_financing_rate: FlexArray::scalar(2e-5),
        short_financing_rate: FlexArray::scalar(1e-5),
        debt_rate: FlexArray::scalar(3e-5),
        borrow_rate: FlexArray::scalar(4e-5),
        mark_price: FlexArray::full(&mark, n_rows, n_cols).unwrap(),
        groups,
        ..FromOrders::default()
    };
    for groups in [
        Groups::ungrouped(n_cols),
        Groups::from_lens(&[2, 3, 1], n_cols).unwrap(),
    ] {
        let args = charged(groups);
        let serial = simulate_from_orders(&args).unwrap();
        let kinds: Vec<i8> = serial.cash.kind.clone();
        for kind in [FUNDING, FINANCING, INTEREST, BORROW_FEE] {
            assert!(kinds.contains(&kind), "no charge of kind {kind}");
        }
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
        close: FlexArray::full(&close, n_rows, n_cols).unwrap(),
        entries: FlexArray::full(&entries, n_rows, n_cols).unwrap(),
        exits: FlexArray::full(&exits, n_rows, n_cols).unwrap(),
        direction: FlexArray::scalar(Direction::Both),
        leverage: FlexArray::per_column(&leverage, n_cols).unwrap(),
        account_type: FlexArray::per_column(&types, n_cols).unwrap(),
        initial_margin: FlexArray::scalar(0.2),
        funding_rate: FlexArray::per_row(&funding, n_rows).unwrap(),
        long_financing_rate: FlexArray::scalar(2e-5),
        debt_rate: FlexArray::scalar(3e-5),
        borrow_rate: FlexArray::scalar(4e-5),
        groups: Some(Groups::from_lens(&[3, 3], n_cols).unwrap()),
        ..FromSignals::default()
    };
    let serial = simulate_from_signals(&args).unwrap();
    assert!(!serial.cash.is_empty());
    for threads in [2, 3] {
        let parallel = simulate_from_signals_parallel(&args, threads).unwrap();
        assert_eq!(all_bits(&serial), all_bits(&parallel), "{threads} threads");
    }
}
