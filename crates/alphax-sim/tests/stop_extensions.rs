//! The trailing, time and laddered stops and stop-limit exits in
//! `from_signals`, worked by hand.
//!
//! A rising series for the trailing stop, the time stops and the
//! take-profit ladder:
//!
//! ```text
//! bar   open  high   low  close
//!  0     100   100   100    100
//!  1     101   110   100    108
//!  2     107   108   104    105
//!  3     112   115   111    114
//! ```
//!
//! and the falling five bars of `limit_orders.rs` for the stop-loss ladder
//! and stop-limit orders. Every entry is on bar 0 at its close.
use alphax_core::flex::FlexArray;
use alphax_model::{DeltaFormat, OrderType, StopLadderMode};
use alphax_records::OrderRecords;
use alphax_sim::{FromSignals, simulate_from_signals, simulate_from_signals_parallel};

const UP_OPEN: [f64; 4] = [100., 101., 107., 112.];
const UP_HIGH: [f64; 4] = [100., 110., 108., 115.];
const UP_LOW: [f64; 4] = [100., 100., 104., 111.];
const UP_CLOSE: [f64; 4] = [100., 108., 105., 114.];

const DOWN_OPEN: [f64; 5] = [100., 99., 97., 93., 95.];
const DOWN_HIGH: [f64; 5] = [101., 100., 98., 96., 103.];
const DOWN_LOW: [f64; 5] = [99., 96., 94., 92., 95.];
const DOWN_CLOSE: [f64; 5] = [100., 97., 96., 95., 102.];

const ENTRY: [bool; 5] = [true, false, false, false, false];
const NO: [bool; 5] = [false; 5];

fn up() -> FromSignals<'static> {
    FromSignals {
        n_rows: 4,
        n_cols: 1,
        init_cash: FlexArray::scalar(1000.),
        open: FlexArray::per_row(&UP_OPEN, 4).unwrap(),
        high: FlexArray::per_row(&UP_HIGH, 4).unwrap(),
        low: FlexArray::per_row(&UP_LOW, 4).unwrap(),
        close: FlexArray::per_row(&UP_CLOSE, 4).unwrap(),
        entries: FlexArray::per_row(&ENTRY[..4], 4).unwrap(),
        exits: FlexArray::per_row(&NO[..4], 4).unwrap(),
        size: FlexArray::scalar(1.),
        ..FromSignals::default()
    }
}

fn down() -> FromSignals<'static> {
    FromSignals {
        n_rows: 5,
        n_cols: 1,
        init_cash: FlexArray::scalar(1000.),
        open: FlexArray::per_row(&DOWN_OPEN, 5).unwrap(),
        high: FlexArray::per_row(&DOWN_HIGH, 5).unwrap(),
        low: FlexArray::per_row(&DOWN_LOW, 5).unwrap(),
        close: FlexArray::per_row(&DOWN_CLOSE, 5).unwrap(),
        entries: FlexArray::per_row(&ENTRY, 5).unwrap(),
        exits: FlexArray::per_row(&NO, 5).unwrap(),
        size: FlexArray::scalar(1.),
        ..FromSignals::default()
    }
}

/// `(bar, price, size)` of each fill.
fn fills(orders: &OrderRecords) -> Vec<(i64, f64, f64)> {
    (0..orders.len())
        .map(|k| (orders.idx[k], orders.price[k], orders.size[k]))
        .collect()
}

fn run(args: &FromSignals<'_>) -> Vec<(i64, f64, f64)> {
    fills(&simulate_from_signals(args).unwrap())
}

#[test]
fn a_trailing_stop_follows_the_best_price_once_past_its_threshold() {
    // Best 110 after bar 1; 5% below it is 104.5, which bar 2 reaches.
    let tsl = FromSignals {
        tsl_stop: FlexArray::scalar(0.05),
        ..up()
    };
    assert_eq!(run(&tsl), [(0, 100., 1.), (2, 104.5, 1.)]);
    // Not live until the price has first risen 20%: it never does.
    let th = FromSignals {
        tsl_stop: FlexArray::scalar(0.05),
        tsl_th: FlexArray::scalar(0.2),
        ..up()
    };
    assert_eq!(run(&th), [(0, 100., 1.)]);
    // 5% is reached at 105 on bar 1, so from bar 2 it is live.
    let reached = FromSignals {
        tsl_th: FlexArray::scalar(0.05),
        ..tsl
    };
    assert_eq!(run(&reached), [(0, 100., 1.), (2, 104.5, 1.)]);
}

#[test]
fn a_time_stop_exits_at_the_open() {
    let td = FromSignals {
        td_stop: FlexArray::scalar(2.),
        ..up()
    };
    assert_eq!(run(&td), [(0, 100., 1.), (2, 107., 1.)]);
    let dt = FromSignals {
        dt_stop: FlexArray::scalar(3.),
        ..up()
    };
    assert_eq!(run(&dt), [(0, 100., 1.), (3, 112., 1.)]);
    // Before a price stop on the same bar: bar 2 would reach the trailing
    // stop at 104.5, but the time stop exits at its open first.
    let both = FromSignals {
        tsl_stop: FlexArray::scalar(0.05),
        ..td
    };
    assert_eq!(run(&both), [(0, 100., 1.), (2, 107., 1.)]);
}

#[test]
fn a_take_profit_ladder_closes_the_position_in_steps() {
    let steps = [0.05, 0.1];
    // Uniform: half at 105 on bar 1; the rest at 110, which bar 3 opens past.
    let uniform = FromSignals {
        size: FlexArray::scalar(2.),
        tp_stop: FlexArray::per_row(&steps, 2).unwrap(),
        stop_ladder: StopLadderMode::Uniform,
        ..up()
    };
    assert_eq!(run(&uniform), [(0, 100., 2.), (1, 105., 1.), (3, 112., 1.)]);
    // Weighted over 5% and 20%: the first step is a quarter.
    let wide = [0.05, 0.2];
    let weighted = FromSignals {
        size: FlexArray::scalar(4.),
        tp_stop: FlexArray::per_row(&wide, 2).unwrap(),
        stop_ladder: StopLadderMode::Weighted,
        ..up()
    };
    assert_eq!(run(&weighted), [(0, 100., 4.), (1, 105., 1.)]);
}

#[test]
fn a_stop_loss_ladder_steps_down() {
    // 98 on bar 1, 96 on bar 2, half each.
    let steps = [0.02, 0.04];
    let args = FromSignals {
        size: FlexArray::scalar(2.),
        sl_stop: FlexArray::per_row(&steps, 2).unwrap(),
        stop_ladder: StopLadderMode::Uniform,
        ..down()
    };
    assert_eq!(run(&args), [(0, 100., 2.), (1, 98., 1.), (2, 96., 1.)]);
    // Decreasing steps are refused.
    let falling = [0.04, 0.02];
    let bad = FromSignals {
        sl_stop: FlexArray::per_row(&falling, 2).unwrap(),
        ..args
    };
    let err = simulate_from_signals(&bad).unwrap_err().to_string();
    assert!(
        err.contains("must be finite, 0 or greater and increasing"),
        "{err}"
    );
}

#[test]
fn a_stop_limit_order_fills_at_its_level_or_rests_after_a_gap() {
    // 3.125% below 100 is 96.875, reached inside bar 1: the limit 1/128 below
    // it is reached at once.
    let inside = FromSignals {
        sl_stop: FlexArray::scalar(0.031_25),
        stop_order_type: FlexArray::scalar(OrderType::Limit),
        stop_limit_delta: FlexArray::scalar(1. / 128.),
        ..down()
    };
    assert_eq!(run(&inside), [(0, 100., 1.), (1, 96.875, 1.)]);
    // 6.25% below is 93.75, which bar 3 opens past at 93: below the limit
    // at 93.017578125, so it rests, and bar 4 opens above it.
    let gap = FromSignals {
        sl_stop: FlexArray::scalar(0.0625),
        ..inside.clone()
    };
    assert_eq!(run(&gap), [(0, 100., 1.), (4, 95., 1.)]);
    // At market the gap fills at the open.
    let market = FromSignals {
        stop_order_type: FlexArray::scalar(OrderType::Market),
        ..gap
    };
    assert_eq!(run(&market), [(0, 100., 1.), (3, 93., 1.)]);
}

#[test]
fn the_extended_stops_do_not_depend_on_the_thread_count() {
    let (n_rows, n_cols) = (300, 5);
    let mut close = Vec::new();
    let (mut open, mut high, mut low) = (Vec::new(), Vec::new(), Vec::new());
    let mut entries = Vec::new();
    for c in 0..n_cols {
        let mut p = 100.0;
        for r in 0..n_rows {
            let o = p;
            p *= 1.0 + 0.03 * ((r * 7 + c * 13) as f64 * 0.37).sin();
            open.push(o);
            close.push(p);
            high.push(o.max(p) * 1.01);
            low.push(o.min(p) * 0.99);
            entries.push((r + c) % 9 == 0);
        }
    }
    let steps = [0.02, 0.05, 0.08];
    let args = FromSignals {
        n_rows,
        n_cols,
        init_cash: FlexArray::scalar(10_000.),
        open: FlexArray::full(&open, n_rows, n_cols).unwrap(),
        high: FlexArray::full(&high, n_rows, n_cols).unwrap(),
        low: FlexArray::full(&low, n_rows, n_cols).unwrap(),
        close: FlexArray::full(&close, n_rows, n_cols).unwrap(),
        entries: FlexArray::full(&entries, n_rows, n_cols).unwrap(),
        size: FlexArray::scalar(3.),
        tp_stop: FlexArray::per_row(&steps, 3).unwrap(),
        stop_ladder: StopLadderMode::Uniform,
        tsl_stop: FlexArray::scalar(0.04),
        tsl_th: FlexArray::scalar(0.01),
        td_stop: FlexArray::scalar(12.),
        stop_order_type: FlexArray::scalar(OrderType::Limit),
        stop_limit_delta: FlexArray::scalar(0.002),
        ..FromSignals::default()
    };
    let serial = simulate_from_signals(&args).unwrap();
    assert!(serial.len() > 50, "too few fills");
    for threads in [2, 3] {
        assert_eq!(
            serial,
            simulate_from_signals_parallel(&args, threads).unwrap()
        );
    }
}

#[test]
fn delta_format_reads_the_stop_distances() {
    // A take-profit 8 above 100 is 108, which bar 1 reaches, however written.
    for (format, tp) in [
        (DeltaFormat::Absolute, 8.),
        (DeltaFormat::Percent100, 8.),
        (DeltaFormat::Target, 108.),
    ] {
        let args = FromSignals {
            tp_stop: FlexArray::scalar(tp),
            delta_format: FlexArray::scalar(format),
            ..up()
        };
        assert_eq!(run(&args), [(0, 100., 1.), (1, 108., 1.)], "{format:?}");
    }
    // 5 below 100 on the falling bars: bar 2's low of 94 reaches 95.
    let sl = FromSignals {
        sl_stop: FlexArray::scalar(5.),
        delta_format: FlexArray::scalar(DeltaFormat::Absolute),
        ..down()
    };
    assert_eq!(run(&sl), [(0, 100., 1.), (2, 95., 1.)]);
}

#[test]
fn a_trailing_stop_ladder_steps_back_from_the_best_price() {
    // Steps 2% and 5% below the best price. Bar 2 opens at 107, under 2%
    // below bar 1's 110 (107.8): half goes at the open. 5% below 110 is
    // 104.5, which nothing after reaches.
    static STEPS: [f64; 2] = [0.02, 0.05];
    let args = FromSignals {
        size: FlexArray::scalar(2.),
        tsl_stop: FlexArray::PerRow(&STEPS),
        stop_ladder: StopLadderMode::Uniform,
        ..up()
    };
    assert_eq!(run(&args), [(0, 100., 2.), (2, 107., 1.)]);
}

#[test]
fn a_time_stop_can_exit_with_a_limit_order() {
    // Two bars after the entry: at bar 2's open, 107, at market. As a limit
    // to sell at 109 or better it rests, and bar 3's open of 112 fills it.
    let market = FromSignals {
        td_stop: FlexArray::scalar(2.),
        ..up()
    };
    assert_eq!(run(&market), [(0, 100., 1.), (2, 107., 1.)]);
    let limit = FromSignals {
        stop_order_type: FlexArray::scalar(OrderType::Limit),
        stop_limit_delta: FlexArray::scalar(109.),
        delta_format: FlexArray::scalar(DeltaFormat::Target),
        ..market
    };
    assert_eq!(run(&limit), [(0, 100., 1.), (3, 112., 1.)]);
}
