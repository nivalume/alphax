//! `from_ago` per bar, an explicit call sequence, and `call_seq = "auto"`
//! by what an order does to free cash in runs with margin columns.
use alphax_core::flex::FlexArray;
use alphax_model::AccountType;
use alphax_records::OrderRecords;
use alphax_sim::groups::{CallSeq, Groups};
use alphax_sim::{FromOrders, FromSignals, simulate_from_orders, simulate_from_signals};

const N: f64 = f64::NAN;

fn fills(orders: &OrderRecords) -> Vec<(i64, i64, f64, f64)> {
    (0..orders.len())
        .map(|k| {
            (
                orders.col[k],
                orders.idx[k],
                orders.size[k],
                orders.price[k],
            )
        })
        .collect()
}

#[test]
fn from_ago_is_read_on_the_executing_bar() {
    let close = [100.0, 101.0, 102.0, 103.0];
    let size = [1.0, 2.0, 3.0, 4.0];
    // Bar 1 executes bar 0's order again, bar 3 reaches before the first row.
    let ago = [0_usize, 1, 0, 5];
    let orders = simulate_from_orders(&FromOrders {
        n_rows: 4,
        n_cols: 1,
        init_cash: FlexArray::scalar(1e6),
        close: FlexArray::per_row(&close, 4).unwrap(),
        size: FlexArray::per_row(&size, 4).unwrap(),
        from_ago: FlexArray::per_row(&ago, 4).unwrap(),
        groups: Groups::ungrouped(1),
        ..FromOrders::default()
    })
    .unwrap();
    assert_eq!(
        fills(&orders),
        [(0, 0, 1.0, 100.0), (0, 1, 1.0, 101.0), (0, 2, 3.0, 102.0)]
    );
    // The same in `from_signals`: an entry on bar 0 executes on bar 2.
    let entries = [true, false, false, false];
    let ago = [0_usize, 5, 2, 0];
    let orders = simulate_from_signals(&FromSignals {
        n_rows: 4,
        n_cols: 1,
        init_cash: FlexArray::scalar(1e6),
        close: FlexArray::per_row(&close, 4).unwrap(),
        entries: FlexArray::per_row(&entries[..], 4).unwrap(),
        size: FlexArray::scalar(1.0),
        from_ago: FlexArray::per_row(&ago, 4).unwrap(),
        ..FromSignals::default()
    })
    .unwrap();
    // Bar 0 takes its own entry; bar 2 reads it again but already holds.
    assert_eq!(fills(&orders), [(0, 0, 1.0, 100.0)]);
    let ago = [3_usize, 5, 2, 0];
    let orders = simulate_from_signals(&FromSignals {
        n_rows: 4,
        n_cols: 1,
        init_cash: FlexArray::scalar(1e6),
        close: FlexArray::per_row(&close, 4).unwrap(),
        entries: FlexArray::per_row(&entries[..], 4).unwrap(),
        size: FlexArray::scalar(1.0),
        from_ago: FlexArray::per_row(&ago, 4).unwrap(),
        ..FromSignals::default()
    })
    .unwrap();
    assert_eq!(fills(&orders), [(0, 2, 1.0, 102.0)]);
}

#[test]
fn an_explicit_call_sequence_decides_who_gets_the_cash() {
    // 100 of shared cash and two buys of 100: whoever goes first fills.
    let close = [100.0, 100.0];
    let size = [1.0, 1.0];
    let run = |seq: CallSeq<'_>| {
        simulate_from_orders(&FromOrders {
            n_rows: 1,
            n_cols: 2,
            init_cash: FlexArray::scalar(100.0),
            close: FlexArray::full(&close, 1, 2).unwrap(),
            size: FlexArray::full(&size, 1, 2).unwrap(),
            groups: Groups::from_lens(&[2], 2).unwrap(),
            call_seq: seq,
            ..FromOrders::default()
        })
    };
    assert_eq!(fills(&run(CallSeq::Default).unwrap()), [(0, 0, 1.0, 100.0)]);
    // Column-major: row 0 of "column" 0 holds the first call, 1.
    let second_first = [1_usize, 0];
    assert_eq!(
        fills(&run(CallSeq::Explicit(&second_first)).unwrap()),
        [(1, 0, 1.0, 100.0)]
    );
    let twice = [1_usize, 1];
    let err = run(CallSeq::Explicit(&twice)).unwrap_err().to_string();
    assert!(err.contains("appears twice"), "{err}");
}

#[test]
fn auto_calls_the_margin_close_before_the_margin_open() {
    // Two margin columns on 50% margin sharing 50 of cash. Bar 0: column 1
    // buys 2 at 50, posting all 50. Bar 1: column 0 shorts 2 (needs 50 of
    // margin) and column 1 sells its 2 (frees 50). Both are sells of 100,
    // so the default order keeps column 0 first, which finds no margin;
    // by free cash the close goes first and both fill.
    let close = [50.0, 50.0, 50.0, 50.0];
    let size = [N, -2.0, 2.0, -2.0];
    let run = |seq: CallSeq<'static>| {
        simulate_from_orders(&FromOrders {
            n_rows: 2,
            n_cols: 2,
            init_cash: FlexArray::scalar(50.0),
            close: FlexArray::full(&close, 2, 2).unwrap(),
            size: FlexArray::full(&size, 2, 2).unwrap(),
            account_type: FlexArray::scalar(AccountType::Margin),
            initial_margin: FlexArray::scalar(0.5),
            groups: Groups::from_lens(&[2], 2).unwrap(),
            call_seq: seq,
            ..FromOrders::default()
        })
        .unwrap()
    };
    let by_layout = run(CallSeq::Default);
    assert_eq!(fills(&by_layout), [(1, 0, 2.0, 50.0), (1, 1, 2.0, 50.0)]);
    let auto = run(CallSeq::Auto);
    assert_eq!(
        fills(&auto),
        [(1, 0, 2.0, 50.0), (1, 1, 2.0, 50.0), (0, 1, 2.0, 50.0)]
    );
}
