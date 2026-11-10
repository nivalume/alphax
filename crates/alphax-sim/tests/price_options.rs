//! Price options in both loops, worked by hand on five bars.
//!
//! ```text
//! bar   open  high   low  close
//!  0     100   101    99    100
//!  1      99   100    96     97
//!  2      97    98    94     96
//!  3      93    96    92     95
//!  4      95   103    95    102
//! ```
use alphax_core::flex::FlexArray;
use alphax_exec::SizeType;
use alphax_model::{Direction, PriceAreaVioMode, PriceType};
use alphax_records::OrderRecords;
use alphax_sim::groups::Groups;
use alphax_sim::{FromOrders, FromSignals, simulate_from_orders, simulate_from_signals};

const OPEN: [f64; 5] = [100., 99., 97., 93., 95.];
const HIGH: [f64; 5] = [101., 100., 98., 96., 103.];
const LOW: [f64; 5] = [99., 96., 94., 92., 95.];
const CLOSE: [f64; 5] = [100., 97., 96., 95., 102.];
const N: f64 = f64::NAN;

fn fills(orders: &OrderRecords) -> Vec<(i64, f64)> {
    orders
        .idx
        .iter()
        .copied()
        .zip(orders.price.iter().copied())
        .collect()
}

fn orders<'a>(size: &'a [f64]) -> FromOrders<'a> {
    FromOrders {
        n_rows: 5,
        n_cols: 1,
        init_cash: FlexArray::scalar(1000.),
        open: FlexArray::per_row(&OPEN, 5).unwrap(),
        high: FlexArray::per_row(&HIGH, 5).unwrap(),
        low: FlexArray::per_row(&LOW, 5).unwrap(),
        close: FlexArray::per_row(&CLOSE, 5).unwrap(),
        size: FlexArray::per_row(size, 5).unwrap(),
        groups: Groups::ungrouped(1),
        ..FromOrders::default()
    }
}

fn signals<'a>(entries: &'a [bool], exits: &'a [bool]) -> FromSignals<'a> {
    FromSignals {
        n_rows: 5,
        n_cols: 1,
        init_cash: FlexArray::scalar(1000.),
        open: FlexArray::per_row(&OPEN, 5).unwrap(),
        high: FlexArray::per_row(&HIGH, 5).unwrap(),
        low: FlexArray::per_row(&LOW, 5).unwrap(),
        close: FlexArray::per_row(&CLOSE, 5).unwrap(),
        entries: FlexArray::per_row(entries, 5).unwrap(),
        exits: FlexArray::per_row(exits, 5).unwrap(),
        size: FlexArray::scalar(1.),
        ..FromSignals::default()
    }
}

#[test]
fn an_order_executes_at_the_open_or_the_close_without_a_price_array() {
    let size = [1., N, -1., N, N];
    let at = |price_type| {
        fills(
            &simulate_from_orders(&FromOrders {
                price_type: FlexArray::scalar(price_type),
                ..orders(&size)
            })
            .unwrap(),
        )
    };
    assert_eq!(at(PriceType::Open), [(0, 100.), (2, 97.)]);
    assert_eq!(at(PriceType::Close), [(0, 100.), (2, 96.)]);
    // `Price` keeps `price`'s own conventions: `-inf` is the previous close
    // in `from_orders`, and there is none before bar 0.
    let prev = FromOrders {
        price: FlexArray::scalar(f64::NEG_INFINITY),
        ..orders(&size)
    };
    assert_eq!(fills(&simulate_from_orders(&prev).unwrap()), [(2, 97.)]);
}

#[test]
fn the_next_open_executes_the_bar_before_s_order() {
    // Decided on bars 0 and 2, executed at the opens of bars 1 and 3.
    let size = [1., N, -1., N, N];
    let next_open = FromOrders {
        price_type: FlexArray::scalar(PriceType::Open),
        from_ago: FlexArray::Scalar(1),
        ..orders(&size)
    };
    assert_eq!(
        fills(&simulate_from_orders(&next_open).unwrap()),
        [(1, 99.), (3, 93.)]
    );
    let entries = [true, false, false, false, false];
    let exits = [false, false, true, false, false];
    let next_open = FromSignals {
        price_type: FlexArray::scalar(PriceType::Open),
        from_ago: FlexArray::Scalar(1),
        ..signals(&entries, &exits)
    };
    assert_eq!(
        fills(&simulate_from_signals(&next_open).unwrap()),
        [(1, 99.), (3, 93.)]
    );
    // An explicit price is part of the order, and moves with it.
    let price = [98., N, N, 97.5, N];
    let shifted = FromOrders {
        price: FlexArray::per_row(&price, 5).unwrap(),
        from_ago: FlexArray::Scalar(1),
        ..orders(&[1., N, N, -1., N])
    };
    assert_eq!(
        fills(&simulate_from_orders(&shifted).unwrap()),
        [(1, 98.), (4, 97.5)]
    );
}

#[test]
fn a_buy_executes_at_the_ask_and_a_sell_at_the_bid() {
    let bid = [99.5, 96.5, 95.5, 94.5, 101.5];
    let ask = [100.5, 97.5, 96.5, 95.5, 102.5];
    let quoted = |args: FromOrders<'_>| {
        fills(
            &simulate_from_orders(&FromOrders {
                bid_price: FlexArray::per_row(&bid, 5).unwrap(),
                ask_price: FlexArray::per_row(&ask, 5).unwrap(),
                ..args
            })
            .unwrap(),
        )
    };
    assert_eq!(quoted(orders(&[1., N, -1., N, N])), [(0, 100.5), (2, 95.5)]);
    // A target's side is read from the position it moves away from.
    let target = FromOrders {
        size_type: FlexArray::scalar(SizeType::TargetAmount),
        ..orders(&[2., N, 1., N, 3.])
    };
    assert_eq!(quoted(target), [(0, 100.5), (2, 95.5), (4, 102.5)]);
    // And in from_signals, a long exit sells at the bid.
    let entries = [true, false, false, false, false];
    let exits = [false, false, false, true, false];
    let args = FromSignals {
        bid_price: FlexArray::per_row(&bid, 5).unwrap(),
        ask_price: FlexArray::per_row(&ask, 5).unwrap(),
        ..signals(&entries, &exits)
    };
    assert_eq!(
        fills(&simulate_from_signals(&args).unwrap()),
        [(0, 100.5), (3, 94.5)]
    );
}

#[test]
fn a_price_outside_the_bar_is_kept_capped_or_refused() {
    let price = [120., N, N, N, N];
    let run = |mode| {
        simulate_from_orders(&FromOrders {
            price: FlexArray::per_row(&price, 5).unwrap(),
            price_area_vio_mode: FlexArray::scalar(mode),
            ..orders(&[1., N, N, N, N])
        })
    };
    assert_eq!(fills(&run(PriceAreaVioMode::Ignore).unwrap()), [(0, 120.)]);
    assert_eq!(fills(&run(PriceAreaVioMode::Cap).unwrap()), [(0, 101.)]);
    let err = run(PriceAreaVioMode::Error).unwrap_err().to_string();
    assert!(err.contains("outside the bar's range [99, 101]"), "{err}");
    // And in from_signals: a buy at 50 on a bar whose low is 99.
    let entries = [true, false, false, false, false];
    let exits = [false; 5];
    let args = FromSignals {
        direction: FlexArray::scalar(Direction::LongOnly),
        price: FlexArray::scalar(50.),
        price_area_vio_mode: FlexArray::scalar(PriceAreaVioMode::Cap),
        ..signals(&entries, &exits)
    };
    assert_eq!(fills(&simulate_from_signals(&args).unwrap()), [(0, 99.)]);
}
