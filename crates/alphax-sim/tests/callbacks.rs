//! Callbacks in `from_orders` and `from_signals`: a callback that changes
//! nothing changes no record, one that decides the orders matches the arrays
//! that make the same decision, flex orders, stopping a group, `row_wise`,
//! and the errors a callback cannot get past.
use std::ops::Range;
use std::sync::Mutex;

use alphax_core::error::{Error, Result};
use alphax_core::flex::FlexArray;
use alphax_exec::{AccountType, Order, OrderResult, SizeType};
use alphax_model::CashFlowKind;
use alphax_records::{Journal, OrderRecords};
use alphax_sim::callbacks::{Context, Decision, Flow, OrderFunc, SignalFunc, risk_size};
use alphax_sim::groups::{CallSeq, Groups};
use alphax_sim::snapshot::Run;
use alphax_sim::state::StopState;
use alphax_sim::{
    FromOrders, FromSignals, Schedule, simulate_from_order_func, simulate_from_orders,
    simulate_from_signal_func, simulate_from_signals,
};

const N_ROWS: usize = 60;
const N_COLS: usize = 6;

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

/// Prices with some movement, per column.
fn prices() -> Vec<f64> {
    (0..N_COLS)
        .flat_map(|c| {
            (0..N_ROWS).map(move |r| {
                let x = (r as f64) * 0.37 + (c as f64) * 1.3;
                100.0 + 8.0 * x.sin() + (r as f64) * 0.05
            })
        })
        .collect()
}

/// Sizes with gaps, signs and both kinds of `NaN`-free zero.
fn sizes() -> Vec<f64> {
    (0..N_COLS * N_ROWS)
        .map(|n| match (n * 7 + n / 5) % 6 {
            0 => 1.5,
            1 => -1.0,
            2 => f64::NAN,
            3 => 0.0,
            4 => 2.0,
            _ => -2.5,
        })
        .collect()
}

/// Does nothing but is on: every hook is called and changes nothing.
struct Passive;

impl OrderFunc for Passive {
    type State = usize;

    fn pre_group(&self, _group: usize, _cols: Range<usize>) -> Result<usize> {
        Ok(0)
    }

    fn order(
        &self,
        _ctx: &Context<'_>,
        _k: usize,
        _order: &mut Order,
        calls: &mut usize,
    ) -> Result<()> {
        *calls += 1;
        Ok(())
    }
}

impl SignalFunc for Passive {
    type State = usize;

    fn pre_group(&self, _group: usize, _cols: Range<usize>) -> Result<usize> {
        Ok(0)
    }

    fn signals(
        &self,
        _ctx: &Context<'_>,
        _k: usize,
        _decision: &mut Decision,
        calls: &mut usize,
    ) -> Result<()> {
        *calls += 1;
        Ok(())
    }
}

fn order_args<'a>(close: &'a [f64], size: &'a [f64], groups: Groups) -> FromOrders<'a> {
    FromOrders {
        n_rows: N_ROWS,
        n_cols: N_COLS,
        init_cash: FlexArray::scalar(500.0),
        close: FlexArray::full(close, N_ROWS, N_COLS).unwrap(),
        size: FlexArray::full(size, N_ROWS, N_COLS).unwrap(),
        fees: FlexArray::scalar(0.001),
        groups,
        ..FromOrders::default()
    }
}

#[test]
fn a_callback_that_changes_nothing_changes_no_record() {
    let close = prices();
    let size = sizes();
    let multiplier = [1.0, 2.0, 1.0, 5.0, 1.0, 1.0];
    for groups in [
        Groups::ungrouped(N_COLS),
        Groups::from_lens(&[2, 3, 1], N_COLS).unwrap(),
    ] {
        for levered in [false, true] {
            let mut args = order_args(&close, &size, groups.clone());
            args.call_seq = CallSeq::Auto;
            if levered {
                args.leverage = FlexArray::scalar(2.0);
                args.multiplier = FlexArray::PerColumn(&multiplier);
            }
            let plain = simulate_from_orders(&args).unwrap();
            for schedule in [
                Schedule::Groups { n_threads: 1 },
                Schedule::Groups { n_threads: 3 },
                Schedule::RowWise,
            ] {
                let Run {
                    records,
                    states: calls,
                    ..
                }: Run<OrderRecords, _> =
                    simulate_from_order_func(&args, &Passive, schedule).unwrap();
                assert_eq!(records, plain, "{schedule:?}, levered {levered}");
                assert_eq!(calls.iter().sum::<usize>(), N_ROWS * N_COLS);
            }
            let logged = alphax_sim::simulate_from_orders_logged(&FromOrders {
                log: FlexArray::scalar(true),
                ..args.clone()
            })
            .unwrap();
            let Run {
                records: journal, ..
            }: Run<Journal, _> = simulate_from_order_func(
                &FromOrders {
                    log: FlexArray::scalar(true),
                    ..args.clone()
                },
                &Passive,
                Schedule::RowWise,
            )
            .unwrap();
            // Logs hold `NaN`s, which compare unequal to themselves.
            assert_eq!(format!("{journal:?}"), format!("{logged:?}"));
        }
    }
}

/// Buys one unit when the close rose, sells one when it fell.
struct Momentum<'a> {
    close: &'a [f64],
}

impl OrderFunc for Momentum<'_> {
    type State = ();

    fn pre_group(&self, _group: usize, _cols: Range<usize>) -> Result<()> {
        Ok(())
    }

    fn order(&self, ctx: &Context<'_>, k: usize, order: &mut Order, (): &mut ()) -> Result<()> {
        let col = ctx.cols.start + k;
        let i = ctx.row;
        order.size = if i == 0 {
            f64::NAN
        } else {
            let (now, before) = (
                self.close[col * N_ROWS + i],
                self.close[col * N_ROWS + i - 1],
            );
            if now > before { 1.0 } else { -1.0 }
        };
        Ok(())
    }
}

#[test]
fn a_callback_that_decides_matches_the_arrays_that_decide_the_same() {
    let close = prices();
    let decided: Vec<f64> = (0..N_COLS * N_ROWS)
        .map(|n| {
            let i = n % N_ROWS;
            if i == 0 {
                f64::NAN
            } else if close[n] > close[n - 1] {
                1.0
            } else {
                -1.0
            }
        })
        .collect();
    let blank = vec![f64::NAN; N_COLS * N_ROWS];
    let groups = Groups::from_lens(&[3, 3], N_COLS).unwrap();
    let expected = simulate_from_orders(&order_args(&close, &decided, groups.clone())).unwrap();
    let Run { records, .. }: Run<OrderRecords, _> = simulate_from_order_func(
        &order_args(&close, &blank, groups),
        &Momentum { close: &close },
        Schedule::Groups { n_threads: 2 },
    )
    .unwrap();
    assert_eq!(records, expected);
    assert!(records.len() > 100);
}

/// Two orders a bar in column 0: buy 2, then sell 1; and stops after
/// `stop_after` fills.
struct TwoLegs {
    stop_after: usize,
}

impl OrderFunc for TwoLegs {
    type State = usize;
    const FLEX: bool = true;

    fn pre_group(&self, _group: usize, _cols: Range<usize>) -> Result<usize> {
        Ok(0)
    }

    fn flex_order(
        &self,
        ctx: &Context<'_>,
        call: usize,
        _fills: &mut usize,
    ) -> Result<Option<(usize, Order)>> {
        let mut order = ctx.order(0);
        order.size = match call {
            0 => 2.0,
            1 => -1.0,
            _ => return Ok(None),
        };
        Ok(Some((0, order)))
    }

    fn post_order(
        &self,
        _ctx: &Context<'_>,
        _k: usize,
        _order: &Order,
        result: &OrderResult,
        fills: &mut usize,
    ) -> Result<Flow> {
        if result.is_filled() {
            *fills += 1;
        }
        Ok(if *fills == self.stop_after {
            Flow::StopGroup
        } else {
            Flow::Continue
        })
    }
}

#[test]
fn flex_orders_fill_in_the_order_given_and_a_stopped_group_still_earns() {
    let close = [10.0, 11.0, 12.0, 13.0];
    let earnings = [1.0, 1.0, 1.0, 1.0];
    let args = FromOrders {
        n_rows: 4,
        n_cols: 1,
        init_cash: FlexArray::scalar(1000.0),
        close: FlexArray::per_row(&close, 4).unwrap(),
        size: FlexArray::scalar(f64::NAN),
        cash_earnings: FlexArray::per_row(&earnings, 4).unwrap(),
        groups: Groups::ungrouped(1),
        ..FromOrders::default()
    };
    let Run {
        records,
        states: state,
        ..
    }: Run<OrderRecords, _> =
        simulate_from_order_func(&args, &TwoLegs { stop_after: 3 }, Schedule::RowWise).unwrap();
    assert_eq!(
        fills(&records),
        [(0, 0, 2.0, 10.0), (0, 0, 1.0, 10.0), (0, 1, 2.0, 11.0)]
    );
    assert_eq!(state, [3]);
    // Earned on every bar, the ones after the stop included.
    assert_eq!(records.cash.len(), 4);
    assert!(
        records
            .cash
            .kind
            .iter()
            .all(|&k| k == CashFlowKind::Earning as i8)
    );
}

/// One budget shared by every group: the first `per_row` buys of a bar get
/// it. Only `row_wise` makes "first" mean the same thing every run.
struct Budget {
    per_row: usize,
    taken: Mutex<Vec<(usize, usize)>>,
}

impl OrderFunc for Budget {
    type State = ();

    fn pre_group(&self, _group: usize, _cols: Range<usize>) -> Result<()> {
        Ok(())
    }

    fn order(&self, ctx: &Context<'_>, _k: usize, order: &mut Order, (): &mut ()) -> Result<()> {
        let mut taken = self.taken.lock().unwrap();
        let this_row = taken.iter().filter(|(row, _)| *row == ctx.row).count();
        if this_row < self.per_row {
            taken.push((ctx.row, ctx.group));
            order.size = 1.0;
        } else {
            order.size = f64::NAN;
        }
        Ok(())
    }
}

#[test]
fn row_wise_runs_every_group_s_bar_before_the_next_bar() {
    let close = prices();
    let blank = vec![f64::NAN; N_COLS * N_ROWS];
    let args = order_args(&close, &blank, Groups::ungrouped(N_COLS));
    let budget = Budget {
        per_row: 2,
        taken: Mutex::new(Vec::new()),
    };
    let Run { records, .. }: Run<OrderRecords, _> =
        simulate_from_order_func(&args, &budget, Schedule::RowWise).unwrap();
    // Groups 0 and 1 take every bar's budget.
    let taken = budget.taken.into_inner().unwrap();
    let expected: Vec<(usize, usize)> = (0..N_ROWS).flat_map(|i| [(i, 0), (i, 1)]).collect();
    assert_eq!(taken, expected);
    assert!(records.col.iter().all(|&c| c < 2));
    assert!(records.col.contains(&0) && records.col.contains(&1));
}

/// Asks for something the run's records were not set up for.
struct Levers;

impl OrderFunc for Levers {
    type State = ();

    fn pre_group(&self, _group: usize, _cols: Range<usize>) -> Result<()> {
        Ok(())
    }

    fn order(&self, ctx: &Context<'_>, _k: usize, order: &mut Order, (): &mut ()) -> Result<()> {
        if ctx.row == 3 {
            order.leverage = 3.0;
        }
        if ctx.row == 5 {
            return Err(Error::Invariant("the callback's own error".into()));
        }
        Ok(())
    }
}

/// Names a column outside its group.
struct Outside;

impl OrderFunc for Outside {
    type State = ();
    const FLEX: bool = true;

    fn pre_group(&self, _group: usize, _cols: Range<usize>) -> Result<()> {
        Ok(())
    }

    fn flex_order(
        &self,
        ctx: &Context<'_>,
        _call: usize,
        (): &mut (),
    ) -> Result<Option<(usize, Order)>> {
        Ok(Some((ctx.len(), ctx.order(0))))
    }
}

#[test]
fn what_a_callback_cannot_do_is_an_error() {
    let close = prices();
    let size = sizes();
    let args = order_args(&close, &size, Groups::ungrouped(N_COLS));
    let one = Schedule::Groups { n_threads: 1 };
    let err = simulate_from_order_func::<OrderRecords, _>(&args, &Levers, one).unwrap_err();
    assert!(err.to_string().contains("leverage"), "{err}");
    let mut levered = args.clone();
    levered.leverage = FlexArray::scalar(1.5);
    let err = simulate_from_order_func::<OrderRecords, _>(&levered, &Levers, one).unwrap_err();
    assert!(
        err.to_string().contains("the callback's own error"),
        "{err}"
    );
    let err = simulate_from_order_func::<OrderRecords, _>(&args, &Outside, one).unwrap_err();
    assert!(
        err.to_string().contains("flex_order named column 1"),
        "{err}"
    );
}

/// Sees prices as the strategy wrote them in a run with a multiplier.
struct Unscaled<'a> {
    close: &'a [f64],
    seen: Mutex<Vec<(f64, f64)>>,
}

impl OrderFunc for Unscaled<'_> {
    type State = ();

    fn pre_group(&self, _group: usize, _cols: Range<usize>) -> Result<()> {
        Ok(())
    }

    fn order(&self, ctx: &Context<'_>, k: usize, order: &mut Order, (): &mut ()) -> Result<()> {
        assert!((order.price - self.close[ctx.row]).abs() < 1e-12);
        assert!((ctx.multiplier(k) - 50.0).abs() < 1e-12);
        order.size = if ctx.row == 0 { 1.0 } else { f64::NAN };
        Ok(())
    }

    fn post_order(
        &self,
        ctx: &Context<'_>,
        k: usize,
        _order: &Order,
        result: &OrderResult,
        (): &mut (),
    ) -> Result<Flow> {
        if result.is_filled() {
            self.seen
                .lock()
                .unwrap()
                .push((result.price, ctx.val_price(k)));
        }
        Ok(Flow::Continue)
    }
}

#[test]
fn a_callback_sees_the_strategy_s_prices_whatever_the_multiplier() {
    let close = [20.0, 21.0, 22.0];
    let cb = Unscaled {
        close: &close,
        seen: Mutex::new(Vec::new()),
    };
    let args = FromOrders {
        n_rows: 3,
        n_cols: 1,
        init_cash: FlexArray::scalar(10_000.0),
        close: FlexArray::per_row(&close, 3).unwrap(),
        multiplier: FlexArray::scalar(50.0),
        size: FlexArray::scalar(f64::NAN),
        update_value: true,
        groups: Groups::ungrouped(1),
        ..FromOrders::default()
    };
    let Run { records, .. }: Run<OrderRecords, _> =
        simulate_from_order_func(&args, &cb, Schedule::Groups { n_threads: 1 }).unwrap();
    assert_eq!(fills(&records), [(0, 0, 1.0, 20.0)]);
    let seen = cb.seen.into_inner().unwrap();
    assert_eq!(seen.len(), 1);
    assert!((seen[0].0 - 20.0).abs() < 1e-12 && (seen[0].1 - 20.0).abs() < 1e-12);
}

fn signal_args<'a>(
    close: &'a [f64],
    entries: &'a [bool],
    exits: &'a [bool],
    groups: Groups,
) -> FromSignals<'a> {
    FromSignals {
        n_rows: N_ROWS,
        n_cols: N_COLS,
        init_cash: FlexArray::scalar(500.0),
        close: FlexArray::full(close, N_ROWS, N_COLS).unwrap(),
        entries: FlexArray::full(entries, N_ROWS, N_COLS).unwrap(),
        exits: FlexArray::full(exits, N_ROWS, N_COLS).unwrap(),
        size: FlexArray::scalar(1.0),
        fees: FlexArray::scalar(0.001),
        groups: Some(groups),
        ..FromSignals::default()
    }
}

fn crossings(close: &[f64]) -> (Vec<bool>, Vec<bool>) {
    let up = (0..close.len())
        .map(|n| n % N_ROWS > 0 && close[n] > close[n - 1] && (n / 3) % 2 == 0)
        .collect();
    let down = (0..close.len())
        .map(|n| n % N_ROWS > 0 && close[n] < close[n - 1] && (n / 4) % 2 == 1)
        .collect();
    (up, down)
}

#[test]
fn a_signal_callback_that_changes_nothing_changes_no_record() {
    let close = prices();
    let (entries, exits) = crossings(&close);
    for groups in [
        Groups::ungrouped(N_COLS),
        Groups::from_lens(&[2, 4], N_COLS).unwrap(),
    ] {
        for stops in [false, true] {
            let mut args = signal_args(&close, &entries, &exits, groups.clone());
            if stops {
                args.use_stops = true;
                args.sl_stop = FlexArray::scalar(0.03);
                args.tp_stop = FlexArray::scalar(0.05);
                args.leverage = FlexArray::scalar(2.0);
            }
            let plain = simulate_from_signals(&args).unwrap();
            for n_threads in [1, 3] {
                let Run {
                    records,
                    states: calls,
                    ..
                }: Run<OrderRecords, _> =
                    simulate_from_signal_func(&args, &Passive, n_threads).unwrap();
                assert_eq!(records, plain, "stops {stops}");
                assert_eq!(calls.iter().sum::<usize>(), N_ROWS * N_COLS);
            }
        }
    }
}

/// Enters on the bars the arrays would, read from its own copy of them.
struct Crossing<'a> {
    entries: &'a [bool],
    exits: &'a [bool],
}

impl SignalFunc for Crossing<'_> {
    type State = ();

    fn pre_group(&self, _group: usize, _cols: Range<usize>) -> Result<()> {
        Ok(())
    }

    fn signals(
        &self,
        ctx: &Context<'_>,
        k: usize,
        decision: &mut Decision,
        (): &mut (),
    ) -> Result<()> {
        let n = (ctx.cols.start + k) * N_ROWS + ctx.row;
        decision.signals.long_entry = self.entries[n];
        decision.signals.long_exit = self.exits[n];
        Ok(())
    }
}

#[test]
fn signals_from_a_callback_match_the_same_signals_from_arrays() {
    let close = prices();
    let (entries, exits) = crossings(&close);
    let none = vec![false; N_ROWS * N_COLS];
    let groups = Groups::ungrouped(N_COLS);
    let expected =
        simulate_from_signals(&signal_args(&close, &entries, &exits, groups.clone())).unwrap();
    let cb = Crossing {
        entries: &entries,
        exits: &exits,
    };
    let Run { records, .. }: Run<OrderRecords, _> =
        simulate_from_signal_func(&signal_args(&close, &none, &none, groups), &cb, 2).unwrap();
    assert_eq!(records, expected);
    assert!(records.len() > 20);
}

/// Sizes each entry so that its stop-loss loses 1% of the account, and
/// moves the stop to break-even once the trade is 2% up.
struct Risk {
    sl: f64,
}

impl SignalFunc for Risk {
    type State = ();

    fn pre_group(&self, _group: usize, _cols: Range<usize>) -> Result<()> {
        Ok(())
    }

    fn adjust(
        &self,
        ctx: &Context<'_>,
        k: usize,
        stops: &mut StopState,
        (): &mut (),
    ) -> Result<()> {
        if ctx.position(k) > 0.0 && ctx.close(k) >= stops.sl_price * 1.02 {
            // Measured from the entry: a distance of 0 is break-even.
            stops.sl_stop = 0.0;
        }
        Ok(())
    }

    fn signals(
        &self,
        ctx: &Context<'_>,
        k: usize,
        decision: &mut Decision,
        (): &mut (),
    ) -> Result<()> {
        let price = ctx.close(k);
        decision.size = risk_size(ctx.value, 0.01, price, price * (1.0 - self.sl), 1.0);
        decision.size_type = SizeType::Amount;
        Ok(())
    }
}

#[test]
fn risk_sizing_and_a_break_even_stop_through_callbacks() {
    // Entry at 100 on bar 1; up 3% on bar 2; back to the entry on bar 4.
    let close = [100.0, 100.0, 103.0, 101.0, 99.0, 98.0];
    let entries = [false, true, false, false, false, false];
    let none = [false; 6];
    let args = FromSignals {
        n_rows: 6,
        n_cols: 1,
        init_cash: FlexArray::scalar(10_000.0),
        close: FlexArray::per_row(&close, 6).unwrap(),
        entries: FlexArray::per_row(&entries[..], 6).unwrap(),
        exits: FlexArray::per_row(&none[..], 6).unwrap(),
        use_stops: true,
        sl_stop: FlexArray::scalar(0.05),
        ..FromSignals::default()
    };
    let Run { records, .. }: Run<OrderRecords, _> =
        simulate_from_signal_func(&args, &Risk { sl: 0.05 }, 1).unwrap();
    let f = fills(&records);
    // 1% of 10,000 over a 5-point stop is 20 units.
    assert_eq!(f[0].1, 1);
    assert!((f[0].2 - 20.0).abs() < 1e-9, "{f:?}");
    // Without the move to break-even, the 5% stop would not have fired by
    // bar 5; at break-even it fires on bar 4, the first close below 100.
    assert_eq!(f.len(), 2);
    assert_eq!(f[1].1, 4);
    // A close-only bar opens at its close, so it gaps through the stop.
    assert!((f[1].3 - 99.0).abs() < 1e-9, "{f:?}");
}

#[test]
fn a_signal_callback_with_from_ago_is_an_error() {
    let close = prices();
    let (entries, exits) = crossings(&close);
    let mut args = signal_args(&close, &entries, &exits, Groups::ungrouped(N_COLS));
    args.from_ago = FlexArray::scalar(1);
    let err = simulate_from_signal_func::<OrderRecords, _>(&args, &Passive, 1).unwrap_err();
    assert!(err.to_string().contains("from_ago"), "{err}");
}

#[test]
fn a_margin_callback_keeps_its_column_s_account_type() {
    struct Cash;
    impl OrderFunc for Cash {
        type State = ();
        fn pre_group(&self, _group: usize, _cols: Range<usize>) -> Result<()> {
            Ok(())
        }
        fn order(&self, _ctx: &Context<'_>, _k: usize, o: &mut Order, (): &mut ()) -> Result<()> {
            o.account_type = AccountType::Cash;
            Ok(())
        }
    }
    let close = prices();
    let size = sizes();
    let mut args = order_args(&close, &size, Groups::ungrouped(N_COLS));
    args.account_type = FlexArray::scalar(AccountType::Margin);
    args.initial_margin = FlexArray::scalar(0.1);
    let err =
        simulate_from_order_func::<OrderRecords, _>(&args, &Cash, Schedule::RowWise).unwrap_err();
    assert!(err.to_string().contains("account type"), "{err}");
}

/// A ladder the callback runs: take a third off at each 2% step above the
/// entry, and after each step move the stop-loss up to that step's level.
struct Ladder;

#[derive(Default)]
struct Steps {
    entry: f64,
    done: usize,
}

impl SignalFunc for Ladder {
    type State = Steps;

    fn pre_group(&self, _group: usize, _cols: Range<usize>) -> Result<Steps> {
        Ok(Steps::default())
    }

    fn adjust(
        &self,
        _ctx: &Context<'_>,
        _k: usize,
        stops: &mut StopState,
        steps: &mut Steps,
    ) -> Result<()> {
        if steps.done > 0 {
            stops.sl_price = steps.entry * (1.0 + 0.02 * steps.done as f64);
            stops.sl_stop = 0.0;
        }
        Ok(())
    }

    fn signals(
        &self,
        ctx: &Context<'_>,
        k: usize,
        decision: &mut Decision,
        steps: &mut Steps,
    ) -> Result<()> {
        if ctx.row == 0 {
            decision.signals.long_entry = true;
            decision.size = 9.0;
        } else if ctx.position(k) > 0.0
            && steps.done < 2
            && ctx.close(k) >= steps.entry * (1.0 + 0.02 * (steps.done + 1) as f64)
        {
            decision.signals.long_exit = true;
            decision.size = 3.0;
            steps.done += 1;
        }
        Ok(())
    }

    fn post_order(
        &self,
        ctx: &Context<'_>,
        _k: usize,
        result: &OrderResult,
        steps: &mut Steps,
    ) -> Result<Flow> {
        if ctx.row == 0 && result.is_filled() {
            steps.entry = result.price;
        }
        Ok(Flow::Continue)
    }
}

#[test]
fn a_dynamic_ladder_through_callbacks() {
    let close = [100.0, 102.0, 104.0, 106.0, 103.0, 101.0];
    let args = FromSignals {
        n_rows: 6,
        n_cols: 1,
        init_cash: FlexArray::scalar(10_000.0),
        close: FlexArray::per_row(&close, 6).unwrap(),
        accumulate: FlexArray::scalar(alphax_model::AccumulationMode::Both),
        use_stops: true,
        ..FromSignals::default()
    };
    let Run {
        records,
        states: steps,
        ..
    }: Run<OrderRecords, Steps> = simulate_from_signal_func(&args, &Ladder, 1).unwrap();
    // In at 100; a third out at 102 and at 104; the rest stopped at the
    // second step's level, which bar 4 opens below.
    assert_eq!(
        fills(&records),
        [
            (0, 0, 9.0, 100.0),
            (0, 1, 3.0, 102.0),
            (0, 2, 3.0, 104.0),
            (0, 4, 3.0, 103.0)
        ]
    );
    assert_eq!(steps[0].done, 2);
}
