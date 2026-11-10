//! Limit orders in `from_signals`, worked by hand on five bars.
//!
//! ```text
//! bar   open  high   low  close
//!  0     100   101    99    100
//!  1      99   100    96     97
//!  2      97    98    94     96
//!  3      93    96    92     95
//!  4      95   103    95    102
//! ```
//!
//! Every run pays 1% slippage on a market order, which a limit order does
//! not, so a fill price says which of the two it was.
use alphax_core::flex::FlexArray;
use alphax_model::{DeltaFormat, Direction, LimitOrderPrice, OrderType, PendingConflictMode};
use alphax_records::OrderRecords;
use alphax_sim::groups::Groups;
use alphax_sim::{
    FromSignals, simulate_from_signals, simulate_from_signals_logged,
    simulate_from_signals_parallel,
};

const OPEN: [f64; 5] = [100., 99., 97., 93., 95.];
const HIGH: [f64; 5] = [101., 100., 98., 96., 103.];
const LOW: [f64; 5] = [99., 96., 94., 92., 95.];
const CLOSE: [f64; 5] = [100., 97., 96., 95., 102.];
const NO: [bool; 5] = [false; 5];
const FIRST: [bool; 5] = [true, false, false, false, false];

fn bars<'a>(entries: &'a [bool], exits: &'a [bool]) -> FromSignals<'a> {
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
        slippage: FlexArray::scalar(0.01),
        order_type: FlexArray::scalar(OrderType::Limit),
        ..FromSignals::default()
    }
}

/// `(bar, price)` of each fill.
fn fills(orders: &OrderRecords) -> Vec<(i64, f64)> {
    orders
        .idx
        .iter()
        .copied()
        .zip(orders.price.iter().copied())
        .collect()
}

fn run(args: &FromSignals<'_>) -> Vec<(i64, f64)> {
    fills(&simulate_from_signals(args).unwrap())
}

#[test]
fn a_buy_limit_rests_below_the_close_and_fills_at_its_price_without_slippage() {
    let args = FromSignals {
        limit_delta: FlexArray::scalar(5.),
        delta_format: FlexArray::scalar(DeltaFormat::Absolute),
        ..bars(&FIRST, &NO)
    };
    // 95: not reached on bar 1 (low 96), reached on bar 2 (low 94).
    assert_eq!(run(&args), [(2, 95.)]);
    // A percentage delta: 5% below 100.
    let percent = FromSignals {
        limit_delta: FlexArray::scalar(0.05),
        ..bars(&FIRST, &NO)
    };
    assert_eq!(run(&percent), [(2, 100. * (1. - 0.05))]);
}

#[test]
fn a_gap_through_the_limit_fills_at_the_open_the_limit_or_the_close() {
    // 93.5 is first reached by bar 3, which opens below it at 93.
    let at = |order_price| {
        run(&FromSignals {
            limit_delta: FlexArray::scalar(93.5),
            delta_format: FlexArray::scalar(DeltaFormat::Target),
            limit_order_price: FlexArray::scalar(order_price),
            ..bars(&FIRST, &NO)
        })
    };
    assert_eq!(at(LimitOrderPrice::Limit), [(3, 93.)]);
    assert_eq!(at(LimitOrderPrice::HardLimit), [(3, 93.5)]);
    assert_eq!(at(LimitOrderPrice::Close), [(3, 95.)]);
}

#[test]
fn a_limit_the_price_already_reaches_fills_on_its_own_bar() {
    // At the close itself, without the market order's slippage.
    assert_eq!(run(&bars(&FIRST, &NO)), [(0, 100.)]);
    let market = FromSignals {
        order_type: FlexArray::scalar(OrderType::Market),
        ..bars(&FIRST, &NO)
    };
    assert_eq!(run(&market), [(0, 101.)]);
    // Unless delayed: then it waits for bar 1, which opens below it.
    let delayed = FromSignals {
        limit_delay: FlexArray::scalar(true),
        ..bars(&FIRST, &NO)
    };
    assert_eq!(run(&delayed), [(1, 99.)]);
}

#[test]
fn a_reversed_buy_is_a_breakout_entry() {
    // A buy stop 2 above the close: first reached at bar 4's high.
    let args = FromSignals {
        limit_delta: FlexArray::scalar(2.),
        delta_format: FlexArray::scalar(DeltaFormat::Absolute),
        limit_reverse: FlexArray::scalar(true),
        ..bars(&FIRST, &NO)
    };
    assert_eq!(run(&args), [(4, 102.)]);
}

#[test]
fn an_order_placed_at_the_open_is_tested_on_the_rest_of_its_bar() {
    let second = [false, true, false, false, false];
    let at_open = |delay| {
        run(&FromSignals {
            price: FlexArray::scalar(f64::NEG_INFINITY),
            limit_delta: FlexArray::scalar(2.),
            delta_format: FlexArray::scalar(DeltaFormat::Absolute),
            limit_delay: FlexArray::scalar(delay),
            ..bars(&second, &NO)
        })
    };
    // 97, below bar 1's open of 99 and above its low of 96.
    assert_eq!(at_open(false), [(1, 97.)]);
    // Delayed, bar 2 opens at it.
    assert_eq!(at_open(true), [(2, 97.)]);
    // At the close nothing after the order is known: bar 2 again.
    let at_close = FromSignals {
        limit_delta: FlexArray::scalar(1.),
        delta_format: FlexArray::scalar(DeltaFormat::Absolute),
        ..bars(&second, &NO)
    };
    assert_eq!(run(&at_close), [(2, 96.)]);
}

#[test]
fn an_order_rests_for_its_time_in_force_and_not_into_its_expiry() {
    let resting = |tif: f64, expiry: f64| {
        run(&FromSignals {
            limit_delta: FlexArray::scalar(5.),
            delta_format: FlexArray::scalar(DeltaFormat::Absolute),
            limit_tif: FlexArray::scalar(tif),
            limit_expiry: FlexArray::scalar(expiry),
            ..bars(&FIRST, &NO)
        })
    };
    // Reached on bar 2: two bars after the one it was placed on.
    assert_eq!(resting(1., f64::NAN), []);
    assert_eq!(resting(2., f64::NAN), [(2, 95.)]);
    assert_eq!(resting(f64::NAN, 2.), []);
    assert_eq!(resting(f64::NAN, 3.), [(2, 95.)]);
}

#[test]
fn a_signal_on_the_resting_side_is_ignored_and_one_on_the_other_cancels_it() {
    // A second entry on bar 1 would rest at 92 and fill on bar 3. It is
    // ignored, so the first fills at 95 on bar 2.
    let twice = [true, true, false, false, false];
    let args = FromSignals {
        limit_delta: FlexArray::scalar(5.),
        delta_format: FlexArray::scalar(DeltaFormat::Absolute),
        ..bars(&twice, &NO)
    };
    assert_eq!(run(&args), [(2, 95.)]);

    // In a two-sided column an exit on bar 1 is a short entry: a sell. It
    // cancels the resting buy and executes, here as a market order.
    let types = [
        OrderType::Limit,
        OrderType::Market,
        OrderType::Market,
        OrderType::Market,
        OrderType::Market,
    ];
    let exit = [false, true, false, false, false];
    let args = FromSignals {
        direction: FlexArray::scalar(Direction::Both),
        order_type: FlexArray::per_row(&types, 5).unwrap(),
        limit_delta: FlexArray::scalar(5.),
        delta_format: FlexArray::scalar(DeltaFormat::Absolute),
        ..bars(&FIRST, &exit)
    };
    let orders = simulate_from_signals(&args).unwrap();
    assert_eq!(fills(&orders), [(1, 97. * 0.99)]);
    assert_eq!(orders.side, [1]);
}

#[test]
fn a_stop_that_fills_cancels_the_resting_order() {
    // Long 1 at 100, a sell limit 5 above bar 1's close (102, reached on
    // bar 4) and a stop-loss 5.5 below the entry at 94.5, hit on bar 2
    // first. `delta_format` reads both distances.
    let types = [
        OrderType::Market,
        OrderType::Limit,
        OrderType::Limit,
        OrderType::Limit,
        OrderType::Limit,
    ];
    let exit = [false, true, false, false, false];
    let args = FromSignals {
        order_type: FlexArray::per_row(&types, 5).unwrap(),
        slippage: FlexArray::scalar(0.),
        limit_delta: FlexArray::scalar(5.),
        delta_format: FlexArray::scalar(DeltaFormat::Absolute),
        sl_stop: FlexArray::scalar(5.5),
        ..bars(&FIRST, &exit)
    };
    assert_eq!(run(&args), [(0, 100.), (2, 94.5)]);
    // Left resting, it would reach the kernel on bar 4 and be refused there:
    // the log shows it never did.
    let logged = FromSignals {
        log: FlexArray::scalar(true),
        ..args.clone()
    };
    assert_eq!(
        simulate_from_signals_logged(&logged).unwrap().logs.idx,
        [0, 2]
    );
    // Without the stop the limit takes the position out on bar 4.
    let unstopped = FromSignals {
        sl_stop: FlexArray::scalar(f64::NAN),
        ..args
    };
    assert_eq!(run(&unstopped), [(0, 100.), (4, 102.)]);
}

#[test]
fn a_liquidation_cancels_the_resting_order() {
    // 100 long at 100 on 10% margin with a sixteenth maintenance is
    // liquidated at 96: not on bar 1, whose low only meets the requirement,
    // but on bar 2, while the exit's sell at 102 rests from bar 1.
    let types = [
        OrderType::Market,
        OrderType::Limit,
        OrderType::Limit,
        OrderType::Limit,
        OrderType::Limit,
    ];
    let exit = [false, true, false, false, false];
    let args = FromSignals {
        order_type: FlexArray::per_row(&types, 5).unwrap(),
        size: FlexArray::scalar(100.),
        slippage: FlexArray::scalar(0.),
        account_type: FlexArray::scalar(alphax_model::AccountType::Margin),
        initial_margin: FlexArray::scalar(0.1),
        maintenance_margin: FlexArray::scalar(1. / 16.),
        limit_delta: FlexArray::scalar(5.),
        delta_format: FlexArray::scalar(DeltaFormat::Absolute),
        log: FlexArray::scalar(true),
        ..bars(&FIRST, &exit)
    };
    let journal = simulate_from_signals_logged(&args).unwrap();
    assert_eq!(fills(&journal.orders), [(0, 100.), (2, 96.)]);
    assert_eq!(journal.orders.liquidation, [false, true]);
    assert_eq!(journal.logs.idx, [0, 2]);
}

#[test]
fn a_limit_fill_is_logged_as_the_order_it_was() {
    let args = FromSignals {
        limit_delta: FlexArray::scalar(5.),
        delta_format: FlexArray::scalar(DeltaFormat::Absolute),
        log: FlexArray::scalar(true),
        ..bars(&FIRST, &NO)
    };
    let journal = simulate_from_signals_logged(&args).unwrap();
    // Placing it reaches no kernel, so only the fill has a row.
    assert_eq!(journal.logs.idx, [2]);
    assert_eq!(journal.logs.req_price, [95.]);
    assert_eq!(journal.logs.req_slippage, [0.]);
    assert_eq!(journal.logs.order_id, [0]);
}

#[test]
fn bad_limit_inputs_are_refused() {
    for (delta, format, tif, message) in [
        (-1., DeltaFormat::Absolute, f64::NAN, "limit_delta must be"),
        (
            100.,
            DeltaFormat::Absolute,
            f64::NAN,
            "a limit price must be finite",
        ),
        (
            5.,
            DeltaFormat::Absolute,
            -1.,
            "limit_tif and limit_expiry must be",
        ),
    ] {
        let args = FromSignals {
            limit_delta: FlexArray::scalar(delta),
            delta_format: FlexArray::scalar(format),
            limit_tif: FlexArray::scalar(tif),
            ..bars(&FIRST, &NO)
        };
        let err = simulate_from_signals(&args).unwrap_err().to_string();
        assert!(err.contains(message), "{err}");
    }
}

const SECOND: [bool; 5] = [false, true, false, false, false];

#[test]
fn the_limit_conflict_modes_decide_between_the_resting_order_and_the_signal() {
    use PendingConflictMode::{CancelExecute, CancelIgnore, KeepExecute, KeepIgnore};
    // A buy rests at 95 from bar 0; a second entry on bar 1 would rest at 92
    // and fill on bar 3.
    let twice = [true, true, false, false, false];
    let adjacent = |mode| {
        run(&FromSignals {
            limit_delta: FlexArray::scalar(5.),
            delta_format: FlexArray::scalar(DeltaFormat::Absolute),
            upon_adj_limit_conflict: FlexArray::scalar(mode),
            ..bars(&twice, &NO)
        })
    };
    assert_eq!(adjacent(KeepIgnore), [(2, 95.)]);
    // A new limit order takes the place of one kept.
    assert_eq!(adjacent(KeepExecute), [(3, 92.)]);
    assert_eq!(adjacent(CancelExecute), [(3, 92.)]);
    assert_eq!(adjacent(CancelIgnore), []);

    // As a market order, the second entry fills on bar 1 and the kept order
    // still fills on bar 2, with the size it was placed with.
    let types = [
        OrderType::Limit,
        OrderType::Market,
        OrderType::Market,
        OrderType::Market,
        OrderType::Market,
    ];
    let args = FromSignals {
        order_type: FlexArray::per_row(&types, 5).unwrap(),
        accumulate: FlexArray::scalar(alphax_model::AccumulationMode::Both),
        limit_delta: FlexArray::scalar(5.),
        delta_format: FlexArray::scalar(DeltaFormat::Absolute),
        upon_adj_limit_conflict: FlexArray::scalar(KeepExecute),
        ..bars(&twice, &NO)
    };
    let orders = simulate_from_signals(&args).unwrap();
    assert_eq!(fills(&orders), [(1, 97. * 1.01), (2, 95.)]);
    assert_eq!(orders.size, [1., 1.]);

    // An opposite signal kept out: the short entry is ignored.
    let args = FromSignals {
        direction: FlexArray::scalar(Direction::Both),
        order_type: FlexArray::per_row(&types, 5).unwrap(),
        limit_delta: FlexArray::scalar(5.),
        delta_format: FlexArray::scalar(DeltaFormat::Absolute),
        upon_opp_limit_conflict: FlexArray::scalar(KeepIgnore),
        ..bars(&FIRST, &SECOND)
    };
    assert_eq!(run(&args), [(2, 95.)]);
}

#[test]
fn the_stop_conflict_modes_decide_between_a_fired_stop_and_the_signal() {
    use PendingConflictMode::{CancelExecute, CancelIgnore, KeepExecute, KeepIgnore};
    // Long 1 at 100 with a 5.5% stop, armed at the entry only and hit on
    // bar 2 at 94.5, the bar a signal also fires. Sides decide adjacency: the
    // stop sells, so a short entry is adjacent and a long entry opposite.
    let sl = [0.055, f64::NAN, f64::NAN, f64::NAN, f64::NAN];
    let third = [false, false, true, false, false];
    let base = |entries, exits, direction| FromSignals {
        direction: FlexArray::scalar(direction),
        order_type: FlexArray::scalar(OrderType::Market),
        slippage: FlexArray::scalar(0.),
        sl_stop: FlexArray::per_row(&sl, 5).unwrap(),
        ..bars(entries, exits)
    };
    let outcome = |args: &FromSignals<'_>| {
        let orders = simulate_from_signals(args).unwrap();
        (fills(&orders), orders.size, orders.side)
    };
    // A short entry: in a two-sided column, an exit.
    let short_entry = base(&FIRST, &third, Direction::Both);
    let adjacent = |mode| {
        outcome(&FromSignals {
            upon_adj_stop_conflict: FlexArray::scalar(mode),
            ..short_entry.clone()
        })
    };
    // The stop replaces the signal.
    assert_eq!(
        adjacent(KeepIgnore),
        (vec![(0, 100.), (2, 94.5)], vec![1., 1.], vec![0, 1])
    );
    // The stop, then the short entry sized against the flat it left.
    assert_eq!(
        adjacent(KeepExecute),
        (
            vec![(0, 100.), (2, 94.5), (2, 96.)],
            vec![1., 1., 1.],
            vec![0, 1, 1]
        )
    );
    // No stop: the short entry reverses the long at the close.
    assert_eq!(
        adjacent(CancelExecute),
        (vec![(0, 100.), (2, 96.)], vec![1., 2.], vec![0, 1])
    );
    assert_eq!(adjacent(CancelIgnore), (vec![(0, 100.)], vec![1.], vec![0]));

    // A long entry on the same bar: stopped out, then back in.
    let reentry = [true, false, true, false, false];
    let long_entry = base(&reentry, &NO, Direction::LongOnly);
    let opposite = |mode| {
        outcome(&FromSignals {
            upon_opp_stop_conflict: FlexArray::scalar(mode),
            ..long_entry.clone()
        })
    };
    assert_eq!(
        opposite(KeepExecute),
        (
            vec![(0, 100.), (2, 94.5), (2, 96.)],
            vec![1., 1., 1.],
            vec![0, 1, 0]
        )
    );
    // Cancelled, the entry asks for nothing while long, so there is nothing
    // left: the position stays, without its stop.
    assert_eq!(
        opposite(CancelExecute),
        (vec![(0, 100.)], vec![1.], vec![0])
    );
    assert_eq!(opposite(KeepIgnore), adjacent(KeepIgnore));

    // A long exit is on the stop's side: cancelled, it exits at the close.
    let long_exit = FromSignals {
        upon_adj_stop_conflict: FlexArray::scalar(CancelExecute),
        ..base(&FIRST, &third, Direction::LongOnly)
    };
    assert_eq!(run(&long_exit), [(0, 100.), (2, 96.)]);
}

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

struct Random {
    n_rows: usize,
    n_cols: usize,
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    long_entries: Vec<bool>,
    long_exits: Vec<bool>,
    short_entries: Vec<bool>,
    short_exits: Vec<bool>,
}

fn random(seed: u64) -> Random {
    let (n_rows, n_cols) = (200, 6);
    let mut rng = Lcg(seed);
    let (mut open, mut high, mut low, mut close) = (vec![], vec![], vec![], vec![]);
    for _ in 0..n_cols {
        let mut p = 50.0 + 50.0 * rng.next();
        for _ in 0..n_rows {
            let o = p * (1.0 + 0.02 * (rng.next() - 0.5));
            p = o * (1.0 + 0.04 * (rng.next() - 0.5));
            open.push(o);
            close.push(p);
            high.push(o.max(p) * (1.0 + 0.01 * rng.next()));
            low.push(o.min(p) * (1.0 - 0.01 * rng.next()));
        }
    }
    let mut flags = |p: f64| (0..n_rows * n_cols).map(|_| rng.next() < p).collect();
    Random {
        n_rows,
        n_cols,
        open,
        high,
        low,
        close,
        long_entries: flags(0.08),
        long_exits: flags(0.05),
        short_entries: flags(0.08),
        short_exits: flags(0.05),
    }
}

fn random_args<'a>(r: &'a Random, groups: Groups) -> FromSignals<'a> {
    let full = |v: &'a [f64]| FlexArray::full(v, r.n_rows, r.n_cols).unwrap();
    let flags = |v: &'a [bool]| FlexArray::full(v, r.n_rows, r.n_cols).unwrap();
    FromSignals {
        n_rows: r.n_rows,
        n_cols: r.n_cols,
        init_cash: FlexArray::scalar(1000.),
        open: full(&r.open),
        high: full(&r.high),
        low: full(&r.low),
        close: full(&r.close),
        long_entries: flags(&r.long_entries),
        long_exits: flags(&r.long_exits),
        short_entries: flags(&r.short_entries),
        short_exits: flags(&r.short_exits),
        size: FlexArray::scalar(5.),
        fees: FlexArray::scalar(0.001),
        sl_stop: FlexArray::scalar(0.03),
        tp_stop: FlexArray::scalar(0.05),
        groups: Some(groups),
        ..FromSignals::default()
    }
}

#[test]
fn a_limit_at_the_order_price_is_a_market_order_without_slippage() {
    let r = random(7);
    for groups in [Groups::ungrouped(6), Groups::from_lens(&[3, 3], 6).unwrap()] {
        let market = random_args(&r, groups.clone());
        let limit = FromSignals {
            order_type: FlexArray::scalar(OrderType::Limit),
            ..random_args(&r, groups)
        };
        let (a, b) = (
            simulate_from_signals(&market).unwrap(),
            simulate_from_signals(&limit).unwrap(),
        );
        assert!(a.len() > 50, "too few fills");
        assert_eq!(a, b);
    }
}

#[test]
fn resting_orders_change_the_fills_and_not_with_the_thread_count() {
    let r = random(11);
    for groups in [Groups::ungrouped(6), Groups::from_lens(&[2, 4], 6).unwrap()] {
        let args = FromSignals {
            order_type: FlexArray::scalar(OrderType::Limit),
            limit_delta: FlexArray::scalar(0.01),
            limit_tif: FlexArray::scalar(5.),
            ..random_args(&r, groups.clone())
        };
        let serial = simulate_from_signals(&args).unwrap();
        assert!(serial.len() > 30, "too few fills");
        assert_ne!(
            serial,
            simulate_from_signals(&random_args(&r, groups)).unwrap()
        );
        for threads in [2, 3] {
            assert_eq!(
                serial,
                simulate_from_signals_parallel(&args, threads).unwrap()
            );
        }
    }
}
