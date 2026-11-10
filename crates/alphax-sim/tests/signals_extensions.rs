//! Stop exits without a simulation, and targets and order mode in
//! `from_signals`.
use alphax_core::flex::FlexArray;
use alphax_exec::SizeType;
use alphax_model::{AccumulationMode, Direction, OrderSide};
use alphax_records::OrderRecords;
use alphax_sim::groups::Groups;
use alphax_sim::stop_exits::{OhlcStopExits, generate_ohlc_stop_exits};
use alphax_sim::{FromOrders, FromSignals, simulate_from_orders, simulate_from_signals};

const N: f64 = f64::NAN;

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

fn matrix(values: &[f64], n_rows: usize, n_cols: usize) -> FlexArray<'_, f64> {
    FlexArray::full(values, n_rows, n_cols).unwrap()
}

struct Bars {
    n_rows: usize,
    n_cols: usize,
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
}

fn bars(rng: &mut Lcg, n_rows: usize, n_cols: usize) -> Bars {
    let (mut open, mut high, mut low, mut close) = (vec![], vec![], vec![], vec![]);
    for _ in 0..n_cols {
        let mut p = 50.0 + 50.0 * rng.next();
        for _ in 0..n_rows {
            let gap = if rng.next() < 0.1 { 0.06 } else { 0.01 };
            let o = p * (1.0 + gap * (rng.next() - 0.5) * 2.0);
            p = o * (1.0 + 0.05 * (rng.next() - 0.5));
            open.push(o);
            close.push(p);
            high.push(o.max(p) * (1.0 + 0.01 * rng.next()));
            low.push(o.min(p) * (1.0 - 0.01 * rng.next()));
        }
    }
    Bars {
        n_rows,
        n_cols,
        open,
        high,
        low,
        close,
    }
}

#[test]
fn the_generated_stop_exits_are_the_simulation_s() {
    let mut rng = Lcg(21);
    let b = bars(&mut rng, 400, 6);
    let (n_rows, n_cols) = (b.n_rows, b.n_cols);
    let entries: Vec<bool> = (0..n_rows * n_cols).map(|_| rng.next() < 0.08).collect();
    // Each column its own mix of stops.
    let sl = [0.03, N, 0.05, N, 0.02, 0.04];
    let tsl = [N, 0.04, N, 0.03, N, 0.06];
    let th = [N, N, N, 0.02, N, N];
    let tp = [0.06, N, 0.05, N, N, 0.08];
    let td = [N, N, 7., N, 12., N];
    let full = |v| matrix(v, n_rows, n_cols);
    for short in [false, true] {
        let generated = generate_ohlc_stop_exits(&OhlcStopExits {
            n_rows,
            n_cols,
            entries: FlexArray::full(&entries, n_rows, n_cols).unwrap(),
            open: full(&b.open),
            high: full(&b.high),
            low: full(&b.low),
            close: full(&b.close),
            sl_stop: FlexArray::per_column(&sl, n_cols).unwrap(),
            tsl_stop: FlexArray::per_column(&tsl, n_cols).unwrap(),
            tsl_th: FlexArray::per_column(&th, n_cols).unwrap(),
            tp_stop: FlexArray::per_column(&tp, n_cols).unwrap(),
            td_stop: FlexArray::per_column(&td, n_cols).unwrap(),
            short,
            until_next: false,
            skip_until_exit: true,
            ..OhlcStopExits::default()
        })
        .unwrap();
        let simulated = simulate_from_signals(&FromSignals {
            n_rows,
            n_cols,
            init_cash: FlexArray::scalar(1e9),
            open: full(&b.open),
            high: full(&b.high),
            low: full(&b.low),
            close: full(&b.close),
            entries: FlexArray::full(&entries, n_rows, n_cols).unwrap(),
            direction: FlexArray::scalar(if short {
                Direction::ShortOnly
            } else {
                Direction::LongOnly
            }),
            size: FlexArray::scalar(1.),
            sl_stop: FlexArray::per_column(&sl, n_cols).unwrap(),
            tsl_stop: FlexArray::per_column(&tsl, n_cols).unwrap(),
            tsl_th: FlexArray::per_column(&th, n_cols).unwrap(),
            tp_stop: FlexArray::per_column(&tp, n_cols).unwrap(),
            td_stop: FlexArray::per_column(&td, n_cols).unwrap(),
            ..FromSignals::default()
        })
        .unwrap();
        // The simulation's exits are its fills against the position.
        let exit_side = if short {
            OrderSide::Buy
        } else {
            OrderSide::Sell
        } as i8;
        let mut from_sim = vec![];
        for k in 0..simulated.len() {
            if simulated.side[k] == exit_side {
                let at = simulated.col[k] as usize * n_rows + simulated.idx[k] as usize;
                from_sim.push((at, simulated.price[k]));
            }
        }
        let from_gen: Vec<(usize, f64)> = (0..n_rows * n_cols)
            .filter(|&k| generated.exits[k])
            .map(|k| (k, generated.price[k]))
            .collect();
        assert!(from_gen.len() > 40, "too few exits");
        assert_eq!(from_gen, from_sim, "short: {short}");
        // Every kind of stop fired somewhere.
        for kind in 0..4 {
            assert!(generated.kind.contains(&kind), "no exit of kind {kind}");
        }
    }
}

fn fills(orders: &OrderRecords) -> Vec<(i64, f64, f64, i8)> {
    (0..orders.len())
        .map(|k| {
            (
                orders.idx[k],
                orders.price[k],
                orders.size[k],
                orders.side[k],
            )
        })
        .collect()
}

#[test]
fn a_target_entry_goes_to_its_target_and_an_exit_closes() {
    // 1000 at 100: half the account long is 5; then a short entry reverses
    // to half the account short in one order; a short exit closes it.
    let close = [100., 100., 100., 100.];
    let entries = [true, false, true, false];
    let exits = [false, true, false, false];
    let short_exits = [false, false, false, true];
    let args = FromSignals {
        n_rows: 4,
        n_cols: 1,
        init_cash: FlexArray::scalar(1000.),
        close: FlexArray::per_row(&close, 4).unwrap(),
        long_entries: FlexArray::per_row(&entries[..], 4).unwrap(),
        short_entries: FlexArray::per_row(&exits[..], 4).unwrap(),
        short_exits: FlexArray::per_row(&short_exits[..], 4).unwrap(),
        size: FlexArray::scalar(0.5),
        size_type: FlexArray::scalar(SizeType::TargetPercent),
        ..FromSignals::default()
    };
    let orders = simulate_from_signals(&args).unwrap();
    // Buy 5; sell 10 to -5; buy back... the entry on bar 2 reverses again to
    // +5 (10 bought); the short exit on bar 3 has nothing short to close.
    assert_eq!(
        fills(&orders),
        [(0, 100., 5., 0), (1, 100., 10., 1), (2, 100., 10., 0)]
    );
    // An entry on the side held adds only when accumulation allows, and then
    // moves to the target rather than by it.
    let again = [true, true, false, false];
    let add = |accumulate| {
        fills(
            &simulate_from_signals(&FromSignals {
                long_entries: FlexArray::per_row(&again[..], 4).unwrap(),
                short_entries: FlexArray::scalar(false),
                short_exits: FlexArray::scalar(false),
                size: FlexArray::scalar(3.),
                size_type: FlexArray::scalar(SizeType::TargetAmount),
                accumulate: FlexArray::scalar(accumulate),
                ..args.clone()
            })
            .unwrap(),
        )
    };
    assert_eq!(add(AccumulationMode::Disabled), [(0, 100., 3., 0)]);
    assert_eq!(add(AccumulationMode::Both), [(0, 100., 3., 0)]);
}

#[test]
fn order_mode_is_from_orders_with_the_signal_engine() {
    let mut rng = Lcg(5);
    let b = bars(&mut rng, 300, 4);
    let (n_rows, n_cols) = (b.n_rows, b.n_cols);
    let size: Vec<f64> = (0..n_rows * n_cols)
        .map(|_| {
            let u = rng.next();
            if u < 0.7 { N } else { (u - 0.85) * 40.0 }
        })
        .collect();
    let types = [
        SizeType::Amount,
        SizeType::TargetAmount,
        SizeType::Value,
        SizeType::TargetPercent,
    ];
    let full = |v| matrix(v, n_rows, n_cols);
    for groups in [Groups::ungrouped(4), Groups::from_lens(&[2, 2], 4).unwrap()] {
        let orders = simulate_from_orders(&FromOrders {
            n_rows,
            n_cols,
            init_cash: FlexArray::scalar(5000.),
            close: full(&b.close),
            size: full(&size),
            size_type: FlexArray::per_column(&types, n_cols).unwrap(),
            fees: FlexArray::scalar(0.001),
            slippage: FlexArray::scalar(0.001),
            groups: groups.clone(),
            ..FromOrders::default()
        })
        .unwrap();
        let signals = simulate_from_signals(&FromSignals {
            n_rows,
            n_cols,
            init_cash: FlexArray::scalar(5000.),
            close: full(&b.close),
            size: full(&size),
            size_type: FlexArray::per_column(&types, n_cols).unwrap(),
            direction: FlexArray::scalar(Direction::Both),
            fees: FlexArray::scalar(0.001),
            slippage: FlexArray::scalar(0.001),
            order_mode: true,
            groups: Some(groups),
            ..FromSignals::default()
        })
        .unwrap();
        assert!(orders.len() > 100, "too few fills");
        assert_eq!(orders, signals);
    }
    // With a stop: a target set in order mode is stopped out like an entry.
    let close = [100., 100., 90., 90.];
    let open = [100., 100., 96., 90.];
    let low = [100., 100., 89., 90.];
    let size = [2., N, N, N];
    let stopped = simulate_from_signals(&FromSignals {
        n_rows: 4,
        n_cols: 1,
        init_cash: FlexArray::scalar(1000.),
        close: FlexArray::per_row(&close, 4).unwrap(),
        open: FlexArray::per_row(&open, 4).unwrap(),
        low: FlexArray::per_row(&low, 4).unwrap(),
        size: FlexArray::per_row(&size, 4).unwrap(),
        size_type: FlexArray::scalar(SizeType::TargetAmount),
        sl_stop: FlexArray::scalar(0.05),
        order_mode: true,
        ..FromSignals::default()
    })
    .unwrap();
    assert_eq!(fills(&stopped), [(0, 100., 2., 0), (2, 95., 2., 1)]);
}

#[test]
fn the_percent_spellings_size_as_the_fractions_they_write() {
    let mut rng = Lcg(8);
    let b = bars(&mut rng, 200, 3);
    let (n_rows, n_cols) = (b.n_rows, b.n_cols);
    let size: Vec<f64> = (0..n_rows * n_cols)
        .map(|_| {
            let u = rng.next();
            if u < 0.7 { N } else { (u - 0.85) * 2.0 }
        })
        .collect();
    let hundredfold: Vec<f64> = size.iter().map(|s| s * 100.0).collect();
    let run = |size: &[f64], size_type| {
        simulate_from_orders(&FromOrders {
            n_rows,
            n_cols,
            init_cash: FlexArray::scalar(5000.),
            close: matrix(&b.close, n_rows, n_cols),
            size: matrix(size, n_rows, n_cols),
            size_type: FlexArray::scalar(size_type),
            groups: Groups::from_lens(&[3], 3).unwrap(),
            ..FromOrders::default()
        })
        .unwrap()
    };
    for (spelled, plain) in [
        (SizeType::Percent100, SizeType::Percent),
        (SizeType::TargetPercent100, SizeType::TargetPercent),
        (SizeType::ValuePercent100, SizeType::ValuePercent),
    ] {
        let (a, b) = (run(&hundredfold, spelled), run(&size, plain));
        assert!(a.len() > 30, "too few fills");
        assert_eq!(a.size.len(), b.size.len(), "{spelled:?}");
        for k in 0..a.len() {
            assert!((a.size[k] - b.size[k]).abs() < 1e-9 * b.size[k].abs().max(1.0));
        }
    }
    // `ValuePercent` against `Value` is the kernel's test: it needs the
    // value the order is sized against.
}

#[test]
fn a_forced_exit_closes_the_position_in_place_of_the_bar_s_orders() {
    let close = [100., 101., 102., 103., 104.];
    let force = [false, false, true, false, false];
    // from_orders: bought on bar 0, the order on bar 2 is replaced by a close.
    let size = [2., N, 5., N, 1.];
    let orders = simulate_from_orders(&FromOrders {
        n_rows: 5,
        n_cols: 1,
        init_cash: FlexArray::scalar(1000.),
        close: FlexArray::per_row(&close, 5).unwrap(),
        size: FlexArray::per_row(&size, 5).unwrap(),
        force_exit: FlexArray::per_row(&force, 5).unwrap(),
        groups: Groups::ungrouped(1),
        ..FromOrders::default()
    })
    .unwrap();
    assert_eq!(
        fills(&orders),
        [(0, 100., 2., 0), (2, 102., 2., 1), (4, 104., 1., 0)]
    );
    // from_signals: an entry on the forced bar is not taken either.
    let entries = [true, false, true, true, false];
    let signals = FromSignals {
        n_rows: 5,
        n_cols: 1,
        init_cash: FlexArray::scalar(1000.),
        close: FlexArray::per_row(&close, 5).unwrap(),
        entries: FlexArray::per_row(&entries[..], 5).unwrap(),
        size: FlexArray::scalar(1.),
        force_exit: FlexArray::per_row(&force, 5).unwrap(),
        ..FromSignals::default()
    };
    assert_eq!(
        fills(&simulate_from_signals(&signals).unwrap()),
        [(0, 100., 1., 0), (2, 102., 1., 1), (3, 103., 1., 0)]
    );
    // A stop that fires inside the forced bar fills first, at its level.
    let low = [100., 101., 90., 103., 104.];
    let stopped = FromSignals {
        low: FlexArray::per_row(&low, 5).unwrap(),
        sl_stop: FlexArray::scalar(0.05),
        ..signals
    };
    assert_eq!(
        fills(&simulate_from_signals(&stopped).unwrap()),
        [(0, 100., 1., 0), (2, 95., 1., 1), (3, 103., 1., 0)]
    );
}

#[test]
fn the_generated_ladder_and_absolute_exits_are_the_simulation_s() {
    use alphax_model::{DeltaFormat, StopLadderMode};
    let mut rng = Lcg(5);
    let b = bars(&mut rng, 400, 4);
    let (n_rows, n_cols) = (b.n_rows, b.n_cols);
    let entries: Vec<bool> = (0..n_rows * n_cols).map(|_| rng.next() < 0.06).collect();
    let full = |v| matrix(v, n_rows, n_cols);
    static SL: [f64; 2] = [0.02, 0.04];
    static TP: [f64; 3] = [0.03, 0.06, 0.09];
    let configs = [
        (
            StopLadderMode::Uniform,
            DeltaFormat::Percent,
            &SL[..],
            &TP[..],
        ),
        (
            StopLadderMode::AdaptWeighted,
            DeltaFormat::Percent,
            &SL[..],
            &TP[..],
        ),
        (
            StopLadderMode::Disabled,
            DeltaFormat::Absolute,
            &[2.0][..],
            &[3.5][..],
        ),
    ];
    for (ladder, format, sl, tp) in configs {
        let steps = |v: &'static [f64]| FlexArray::PerRow(v);
        let generated = generate_ohlc_stop_exits(&OhlcStopExits {
            n_rows,
            n_cols,
            entries: FlexArray::full(&entries, n_rows, n_cols).unwrap(),
            open: full(&b.open),
            high: full(&b.high),
            low: full(&b.low),
            close: full(&b.close),
            sl_stop: if sl.len() > 1 {
                steps(sl)
            } else {
                FlexArray::scalar(sl[0])
            },
            tp_stop: if tp.len() > 1 {
                steps(tp)
            } else {
                FlexArray::scalar(tp[0])
            },
            delta_format: FlexArray::scalar(format),
            stop_ladder: ladder,
            until_next: false,
            skip_until_exit: true,
            ..OhlcStopExits::default()
        })
        .unwrap();
        let simulated = simulate_from_signals(&FromSignals {
            n_rows,
            n_cols,
            init_cash: FlexArray::scalar(1e9),
            open: full(&b.open),
            high: full(&b.high),
            low: full(&b.low),
            close: full(&b.close),
            entries: FlexArray::full(&entries, n_rows, n_cols).unwrap(),
            size: FlexArray::scalar(4.),
            sl_stop: if sl.len() > 1 {
                steps(sl)
            } else {
                FlexArray::scalar(sl[0])
            },
            tp_stop: if tp.len() > 1 {
                steps(tp)
            } else {
                FlexArray::scalar(tp[0])
            },
            delta_format: FlexArray::scalar(format),
            stop_ladder: ladder,
            ..FromSignals::default()
        })
        .unwrap();
        let mut from_sim = vec![];
        for k in 0..simulated.len() {
            if simulated.side[k] == OrderSide::Sell as i8 {
                let at = simulated.col[k] as usize * n_rows + simulated.idx[k] as usize;
                from_sim.push((at, simulated.price[k], simulated.size[k]));
            }
        }
        let from_gen: Vec<(usize, f64, f64)> = (0..n_rows * n_cols)
            .filter(|&k| generated.exits[k])
            .map(|k| (k, generated.price[k], generated.size[k] * 4.))
            .collect();
        assert!(from_gen.len() > 30, "too few exits");
        assert_eq!(from_gen.len(), from_sim.len(), "{ladder:?} {format:?}");
        for (g, s) in from_gen.iter().zip(&from_sim) {
            assert_eq!((g.0, g.1), (s.0, s.1), "{ladder:?} {format:?}");
            assert!((g.2 - s.2).abs() < 1e-9, "{ladder:?}: {g:?} {s:?}");
        }
    }
}
