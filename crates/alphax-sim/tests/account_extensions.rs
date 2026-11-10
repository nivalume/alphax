//! Initial positions and prices below zero in both loops, worked by hand.
use alphax_core::flex::FlexArray;
use alphax_model::{AccountType, CashFlowKind};
use alphax_records::OrderRecords;
use alphax_sim::groups::Groups;
use alphax_sim::{
    FromOrders, FromSignals, simulate_from_orders, simulate_from_orders_logged,
    simulate_from_orders_parallel, simulate_from_signals,
};

const N: f64 = f64::NAN;

fn fills(orders: &OrderRecords) -> Vec<(i64, i64, f64, f64, i8)> {
    (0..orders.len())
        .map(|k| {
            (
                orders.col[k],
                orders.idx[k],
                orders.size[k],
                orders.price[k],
                orders.side[k],
            )
        })
        .collect()
}

#[test]
fn an_initial_position_is_a_first_bar_fill_balanced_by_its_cost() {
    let close = [100.0, 105.0, 110.0];
    let size = [N, -4.0, -6.0];
    let args = FromOrders {
        n_rows: 3,
        n_cols: 1,
        init_cash: FlexArray::scalar(1000.0),
        init_position: FlexArray::scalar(10.0),
        init_price: FlexArray::scalar(100.0),
        close: FlexArray::per_row(&close, 3).unwrap(),
        size: FlexArray::per_row(&size, 3).unwrap(),
        log: FlexArray::scalar(true),
        groups: Groups::ungrouped(1),
        ..FromOrders::default()
    };
    let journal = simulate_from_orders_logged(&args).unwrap();
    let orders = &journal.orders;
    assert_eq!(
        fills(orders),
        [
            (0, 0, 10.0, 100.0, 0),
            (0, 1, 4.0, 105.0, 1),
            (0, 2, 6.0, 110.0, 1)
        ]
    );
    assert_eq!(orders.fees, [0.0, 0.0, 0.0]);
    assert_eq!(orders.cash.amount, [1000.0]);
    assert_eq!(orders.cash.kind, [CashFlowKind::InitialPosition as i8]);
    // Bar 0's (empty) order already sees the cash untouched and the
    // position held; the opening fill itself is not logged.
    let logs = &journal.logs;
    assert_eq!(logs.idx, [0, 1, 2]);
    assert_eq!((logs.cash[0], logs.position[0]), (1000.0, 10.0));
    assert_eq!(logs.new_cash, [1000.0, 1420.0, 2080.0]);
}

#[test]
fn an_initial_position_is_held_by_from_signals_and_a_margin_column() {
    let close = [100.0, 101.0, 102.0];
    let entries = [true, false, false];
    let exits = [false, false, true];
    let signals = FromSignals {
        n_rows: 3,
        n_cols: 1,
        init_cash: FlexArray::scalar(0.0),
        init_position: FlexArray::scalar(2.0),
        close: FlexArray::per_row(&close, 3).unwrap(),
        entries: FlexArray::per_row(&entries[..], 3).unwrap(),
        exits: FlexArray::per_row(&exits[..], 3).unwrap(),
        size: FlexArray::scalar(1.0),
        ..FromSignals::default()
    };
    // Held already: the entry adds nothing, the exit sells both, and the
    // price is the first close.
    assert_eq!(
        fills(&simulate_from_signals(&signals).unwrap()),
        [(0, 0, 2.0, 100.0, 0), (0, 2, 2.0, 102.0, 1)]
    );
    // A margin short posts its margin: 5 × 100 × 10%.
    let margin = FromOrders {
        n_rows: 3,
        n_cols: 1,
        init_cash: FlexArray::scalar(1000.0),
        init_position: FlexArray::scalar(-5.0),
        close: FlexArray::per_row(&close, 3).unwrap(),
        size: FlexArray::scalar(N),
        account_type: FlexArray::scalar(AccountType::Margin),
        initial_margin: FlexArray::scalar(0.1),
        groups: Groups::ungrouped(1),
        ..FromOrders::default()
    };
    let orders = simulate_from_orders(&margin).unwrap();
    assert_eq!(fills(&orders), [(0, 0, 5.0, 100.0, 1)]);
    assert_eq!((orders.locked_cash[0], orders.margin[0]), (50.0, true));
    assert!(orders.cash.is_empty());
    let poor = FromOrders {
        init_cash: FlexArray::scalar(10.0),
        ..margin
    };
    let err = simulate_from_orders(&poor).unwrap_err().to_string();
    assert!(err.contains("cannot be held"), "{err}");
}

#[test]
fn initial_positions_do_not_depend_on_the_thread_count() {
    let (n_rows, n_cols) = (40, 6);
    let close: Vec<f64> = (0..n_rows * n_cols)
        .map(|k| 50.0 + (k % 17) as f64)
        .collect();
    let size: Vec<f64> = (0..n_rows * n_cols)
        .map(|k| {
            if k % 5 == 0 {
                -1.0
            } else if k % 7 == 0 {
                2.0
            } else {
                N
            }
        })
        .collect();
    let init = [3.0, 0.0, 5.0, 1.0, 0.0, 2.0];
    let price = [N, N, 60.0, 40.0, N, 55.0];
    let args = FromOrders {
        n_rows,
        n_cols,
        init_cash: FlexArray::scalar(1e4),
        init_position: FlexArray::per_column(&init, n_cols).unwrap(),
        init_price: FlexArray::per_column(&price, n_cols).unwrap(),
        close: FlexArray::full(&close, n_rows, n_cols).unwrap(),
        size: FlexArray::full(&size, n_rows, n_cols).unwrap(),
        groups: Groups::from_lens(&[2, 3, 1], n_cols).unwrap(),
        ..FromOrders::default()
    };
    let serial = simulate_from_orders(&args).unwrap();
    let parallel = simulate_from_orders_parallel(&args, 3).unwrap();
    assert_eq!(fills(&serial), fills(&parallel));
    assert_eq!(serial.cash.amount, parallel.cash.amount);
    assert_eq!(serial.cash.amount.len(), 4);
}

#[test]
fn a_price_below_zero_trades_only_when_allowed() {
    let close = [18.0, -37.6, 10.0, 20.0];
    let size = [1.0, N, N, -1.0];
    let args = FromOrders {
        n_rows: 4,
        n_cols: 1,
        init_cash: FlexArray::scalar(1e5),
        close: FlexArray::per_row(&close, 4).unwrap(),
        size: FlexArray::per_row(&size, 4).unwrap(),
        account_type: FlexArray::scalar(AccountType::Margin),
        initial_margin: FlexArray::scalar(0.1),
        multiplier: FlexArray::scalar(1000.0),
        allow_negative_price: true,
        groups: Groups::ungrouped(1),
        ..FromOrders::default()
    };
    let orders = simulate_from_orders(&args).unwrap();
    assert_eq!(fills(&orders), [(0, 0, 1.0, 18.0, 0), (0, 3, 1.0, 20.0, 1)]);
    // Buying at -37.6 in a cash column, refused without the switch.
    let size = [N, 1.0, N, -1.0];
    let cash = FromOrders {
        size: FlexArray::per_row(&size, 4).unwrap(),
        account_type: FlexArray::scalar(AccountType::Cash),
        multiplier: FlexArray::scalar(1.0),
        ..args.clone()
    };
    assert_eq!(
        fills(&simulate_from_orders(&cash).unwrap()),
        [(0, 1, 1.0, -37.6, 0), (0, 3, 1.0, 20.0, 1)]
    );
    let refused = FromOrders {
        allow_negative_price: false,
        ..cash
    };
    assert!(simulate_from_orders(&refused).is_err());
}
