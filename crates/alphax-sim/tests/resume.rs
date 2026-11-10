//! Continuing a run: a run split at any row, its second part continued from
//! the first's state, writes what the one run writes, bit for bit and in
//! the same order, and ends in the same state. Both ways of continuing are
//! tested: on the whole arrays from `next_row`, and on arrays that start
//! later (`row_offset`) with the rows the bars look back at.

use std::ops::Range;

use alphax_core::flex::FlexArray;
use alphax_core::random::uniform;
use alphax_model::{AccountType, OrderType};
use alphax_records::{Journal, OrderRecords};
use alphax_sim::groups::{CallSeq, Groups};
use alphax_sim::snapshot::{Run, SimState};
use alphax_sim::{
    FromOrders, FromSignals, simulate_from_orders_run, simulate_from_signal_func,
    simulate_from_signals_run,
};

const N: usize = 90;
const C: usize = 6;
/// The rows before a cut a later part carries: one for the previous close,
/// one for `from_ago = 1`.
const LOOKBACK: usize = 2;

/// Column-major market data and decisions for rows `rows` of a fixed run.
struct Data {
    n_rows: usize,
    close: Vec<f64>,
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    size: Vec<f64>,
    /// `size` times 8, as target percents a margin account can be
    /// liquidated on.
    big: Vec<f64>,
    entries: Vec<bool>,
    exits: Vec<bool>,
}

impl Data {
    fn new(seed: u64) -> Self {
        let mut close = vec![0.0; N * C];
        let mut open = vec![0.0; N * C];
        let mut high = vec![0.0; N * C];
        let mut low = vec![0.0; N * C];
        let mut size = vec![0.0; N * C];
        let mut entries = vec![false; N * C];
        let mut exits = vec![false; N * C];
        for c in 0..C {
            let mut p = 100.0 + 10.0 * c as f64;
            for r in 0..N {
                let u = |k: u64| uniform(seed * 31 + k, r, c);
                let o = p * (1.0 + 0.01 * (u(1) - 0.5));
                let cl = o * (1.0 + 0.06 * (u(2) - 0.5));
                let n = c * N + r;
                open[n] = o;
                close[n] = cl;
                high[n] = o.max(cl) * (1.0 + 0.02 * u(3));
                low[n] = o.min(cl) * (1.0 - 0.02 * u(4));
                p = cl;
                size[n] = match (u(5) * 6.0) as u32 {
                    0 => 1.5,
                    1 => -1.0,
                    2 => f64::NAN,
                    3 => 0.4,
                    4 => -0.7,
                    _ => 0.0,
                };
                entries[n] = u(6) < 0.15;
                exits[n] = u(7) < 0.12;
            }
        }
        Self {
            n_rows: N,
            close,
            open,
            high,
            low,
            big: size.iter().map(|s| s * 8.0).collect(),
            size,
            entries,
            exits,
        }
    }

    /// Rows `rows` of every column.
    fn rows(&self, rows: Range<usize>) -> Self {
        fn cut<T: Copy>(v: &[T], rows: &Range<usize>) -> Vec<T> {
            (0..C)
                .flat_map(|c| v[c * N + rows.start..c * N + rows.end].iter().copied())
                .collect()
        }
        Self {
            n_rows: rows.len(),
            close: cut(&self.close, &rows),
            open: cut(&self.open, &rows),
            high: cut(&self.high, &rows),
            low: cut(&self.low, &rows),
            size: cut(&self.size, &rows),
            big: cut(&self.big, &rows),
            entries: cut(&self.entries, &rows),
            exits: cut(&self.exits, &rows),
        }
    }
}

fn full<T: Copy>(v: &[T], n_rows: usize) -> FlexArray<'_, T> {
    FlexArray::full(v, n_rows, C).unwrap()
}

/// The `from_orders` runs: each a different set of features on.
fn orders_args(d: &Data, case: usize) -> FromOrders<'_> {
    let mut a = FromOrders {
        n_rows: d.n_rows,
        n_cols: C,
        init_cash: FlexArray::scalar(1_000.0),
        close: full(&d.close, d.n_rows),
        open: full(&d.open, d.n_rows),
        high: full(&d.high, d.n_rows),
        low: full(&d.low, d.n_rows),
        size: full(&d.size, d.n_rows),
        fees: FlexArray::scalar(0.001),
        groups: Groups::ungrouped(C),
        ..FromOrders::default()
    };
    match case {
        0 => {}
        1 => {
            a.groups = Groups::from_lens(&[2, 3, 1], C).unwrap();
            a.call_seq = CallSeq::Auto;
            a.update_value = true;
            a.price = FlexArray::scalar(f64::NEG_INFINITY);
            a.cash_earnings = FlexArray::scalar(0.5);
        }
        2 => {
            a.leverage = FlexArray::scalar(3.0);
            a.reject_prob = FlexArray::scalar(0.2);
            a.seed = 7;
            a.from_ago = FlexArray::scalar(1);
            a.debt_rate = FlexArray::scalar(0.001);
        }
        _ => {
            a.groups = Groups::from_lens(&[3, 3], C).unwrap();
            a.account_type = FlexArray::scalar(AccountType::Margin);
            a.leverage = FlexArray::scalar(20.0);
            a.maintenance_margin = FlexArray::scalar(0.04);
            a.funding_rate = FlexArray::scalar(0.0005);
            a.size = FlexArray::scalar(f64::NAN);
            a.multiplier = FlexArray::scalar(5.0);
        }
    }
    if case == 3 {
        // Bigger orders, so that liquidations happen.
        a.size = full(&d.big, d.n_rows);
        a.size_type = FlexArray::scalar(alphax_exec::SizeType::TargetPercent);
    }
    a
}

/// The `from_signals` runs.
fn signals_args(d: &Data, case: usize) -> FromSignals<'_> {
    let mut a = FromSignals {
        n_rows: d.n_rows,
        n_cols: C,
        init_cash: FlexArray::scalar(1_000.0),
        close: full(&d.close, d.n_rows),
        open: full(&d.open, d.n_rows),
        high: full(&d.high, d.n_rows),
        low: full(&d.low, d.n_rows),
        entries: full(&d.entries, d.n_rows),
        exits: full(&d.exits, d.n_rows),
        size: FlexArray::scalar(2.0),
        fees: FlexArray::scalar(0.001),
        ..FromSignals::default()
    };
    match case {
        0 => {}
        1 => {
            a.use_stops = true;
            a.sl_stop = FlexArray::scalar(0.03);
            a.tp_stop = FlexArray::scalar(0.05);
            a.tsl_stop = FlexArray::scalar(0.04);
            a.td_stop = FlexArray::scalar(6.0);
            a.direction = FlexArray::scalar(alphax_exec::Direction::Both);
        }
        2 => {
            a.order_type = FlexArray::scalar(OrderType::Limit);
            a.limit_delta = FlexArray::scalar(0.01);
            a.limit_tif = FlexArray::scalar(5.0);
            a.reject_prob = FlexArray::scalar(0.1);
            a.seed = 3;
        }
        _ => {
            a.groups = Some(Groups::from_lens(&[2, 4], C).unwrap());
            a.account_type = FlexArray::scalar(AccountType::Margin);
            a.leverage = FlexArray::scalar(20.0);
            a.maintenance_margin = FlexArray::scalar(0.04);
            a.size = FlexArray::scalar(80.0);
            a.direction = FlexArray::scalar(alphax_exec::Direction::Both);
            a.cash_earnings = FlexArray::scalar(1.0);
        }
    }
    a
}

fn group_of(groups: &Groups) -> Vec<usize> {
    (0..C).map(|c| groups.of_column(c)).collect()
}

/// What a run wrote and ended in, comparably: `NaN`s are equal in a
/// `Debug` string and not under `==`.
fn shown(records: &Journal, state: &SimState) -> (String, Vec<u8>) {
    (format!("{records:?}"), state.to_bytes())
}

fn chained(parts: [Run<Journal>; 2], group_of: &[usize]) -> (String, Vec<u8>) {
    let [a, b] = parts;
    let records = Journal::chain(&[a.records, b.records], group_of);
    shown(&records, &b.last_state)
}

const CUTS: [usize; 5] = [1, 2, 23, 57, N - 1];

#[test]
fn from_orders_continued_is_the_one_run() {
    let data = Data::new(11);
    for case in 0..4 {
        let whole = orders_args(&data, case);
        let one: Run<Journal> = simulate_from_orders_run(&whole, 1).unwrap();
        let expected = shown(&one.records, &one.last_state);
        let groups = group_of(&whole.groups);
        assert!(one.records.orders.len() > 20, "case {case} trades");
        if case == 3 {
            assert!(
                one.records.orders.liquidation.contains(&true),
                "liquidations"
            );
        }
        for cut in CUTS {
            let head_data = data.rows(0..cut);
            let head: Run<Journal> =
                simulate_from_orders_run(&orders_args(&head_data, case), 1).unwrap();
            assert_eq!(head.last_state.next_row, cut);

            // On the whole arrays, from the cut.
            let rest = FromOrders {
                resume: Some(&head.last_state),
                ..orders_args(&data, case)
            };
            let tail: Run<Journal> = simulate_from_orders_run(&rest, 3).unwrap();
            let head_again: Run<Journal> =
                simulate_from_orders_run(&orders_args(&head_data, case), 1).unwrap();
            assert_eq!(
                chained([head_again, tail], &groups),
                expected,
                "case {case}, cut {cut}, whole arrays"
            );

            // On arrays from a little before the cut.
            let from = cut.saturating_sub(LOOKBACK);
            let tail_data = data.rows(from..N);
            let rest = FromOrders {
                resume: Some(&head.last_state),
                row_offset: from,
                ..orders_args(&tail_data, case)
            };
            let tail: Run<Journal> = simulate_from_orders_run(&rest, 1).unwrap();
            assert_eq!(
                chained([head, tail], &groups),
                expected,
                "case {case}, cut {cut}, arrays from row {from}"
            );
        }
    }
}

#[test]
fn from_signals_continued_is_the_one_run() {
    let data = Data::new(5);
    for case in 0..4 {
        let whole = signals_args(&data, case);
        let one: Run<Journal> = simulate_from_signals_run(&whole, 1).unwrap();
        let expected = shown(&one.records, &one.last_state);
        let groups = group_of(&whole.resolved_groups().unwrap());
        assert!(one.records.orders.len() > 10, "case {case} trades");
        if case == 3 {
            assert!(
                one.records.orders.liquidation.contains(&true),
                "liquidations"
            );
        }
        for cut in CUTS {
            let head_data = data.rows(0..cut);
            let head: Run<Journal> =
                simulate_from_signals_run(&signals_args(&head_data, case), 1).unwrap();
            let rest = FromSignals {
                resume: Some(&head.last_state),
                ..signals_args(&data, case)
            };
            let tail: Run<Journal> = simulate_from_signals_run(&rest, 2).unwrap();
            let head_again: Run<Journal> =
                simulate_from_signals_run(&signals_args(&head_data, case), 1).unwrap();
            assert_eq!(
                chained([head_again, tail], &groups),
                expected,
                "case {case}, cut {cut}, whole arrays"
            );

            let from = cut.saturating_sub(LOOKBACK);
            let tail_data = data.rows(from..N);
            let rest = FromSignals {
                resume: Some(&head.last_state),
                row_offset: from,
                ..signals_args(&tail_data, case)
            };
            let tail: Run<Journal> = simulate_from_signals_run(&rest, 1).unwrap();
            assert_eq!(
                chained([head, tail], &groups),
                expected,
                "case {case}, cut {cut}, arrays from row {from}"
            );
        }
    }
}

#[test]
fn many_pieces_are_one_run() {
    // A long history in pieces of 7 rows, each continuing the last.
    let data = Data::new(2);
    let case = 2;
    let whole = signals_args(&data, case);
    let one: Run<Journal> = simulate_from_signals_run(&whole, 1).unwrap();
    let groups = group_of(&whole.resolved_groups().unwrap());
    let mut parts = Vec::new();
    let mut state: Option<SimState> = None;
    let mut at = 0;
    while at < N {
        let end = (at + 7).min(N);
        let from = at.saturating_sub(LOOKBACK);
        let piece = data.rows(from..end);
        let args = FromSignals {
            resume: state.as_ref(),
            row_offset: from,
            ..signals_args(&piece, case)
        };
        // A first piece starts the run: it has no state, and starts at 0.
        assert!(state.is_some() || from == 0);
        let run: Run<Journal> = simulate_from_signals_run(&args, 1).unwrap();
        parts.push(run.records);
        state = Some(run.last_state);
        at = end;
    }
    let records = Journal::chain(&parts, &groups);
    assert_eq!(
        shown(&records, state.as_ref().unwrap()),
        shown(&one.records, &one.last_state)
    );
    // And the state survives bytes.
    let state = state.unwrap();
    assert_eq!(
        SimState::from_bytes(&state.to_bytes()).unwrap().to_bytes(),
        state.to_bytes()
    );
}

#[test]
fn a_state_that_does_not_fit_is_refused() {
    let data = Data::new(1);
    let head: Run<OrderRecords> =
        simulate_from_orders_run(&orders_args(&data.rows(0..30), 0), 1).unwrap();
    // Arrays that end before the state's row.
    let short = data.rows(0..20);
    let err = simulate_from_orders_run::<OrderRecords>(
        &FromOrders {
            resume: Some(&head.last_state),
            ..orders_args(&short, 0)
        },
        1,
    )
    .unwrap_err();
    assert!(err.to_string().contains("continues at row 30"), "{err}");
    // Another grouping.
    let err = simulate_from_orders_run::<OrderRecords>(
        &FromOrders {
            resume: Some(&head.last_state),
            ..orders_args(&data, 1)
        },
        1,
    )
    .unwrap_err();
    assert!(err.to_string().contains("groups"), "{err}");
    // Another loop.
    let err = simulate_from_signals_run::<OrderRecords>(
        &FromSignals {
            resume: Some(&head.last_state),
            ..signals_args(&data, 0)
        },
        1,
    )
    .unwrap_err();
    assert!(err.to_string().contains("FromOrders"), "{err}");
    // Leverage the first part did not have.
    let err = simulate_from_orders_run::<OrderRecords>(
        &FromOrders {
            resume: Some(&head.last_state),
            ..orders_args(&data, 2)
        },
        1,
    )
    .unwrap_err();
    assert!(err.to_string().contains("did not"), "{err}");
}

#[test]
fn the_built_in_rules_continue_and_ignore_the_thread_count() {
    let data = Data::new(5);
    let rules = alphax_sim::Builtin {
        risk: 0.02,
        risk_stop: 0.03,
        breakeven: 0.01,
        max_positions: 2,
        ..alphax_sim::Builtin::default()
    };
    for case in [1, 3] {
        let whole = signals_args(&data, case);
        let groups = group_of(&whole.resolved_groups().unwrap());
        let one: Run<Journal, _> = simulate_from_signal_func(&whole, &rules, 1).unwrap();
        let expected = shown(&one.records, &one.last_state);
        let plain: Run<Journal> = simulate_from_signals_run(&whole, 1).unwrap();
        assert_ne!(
            shown(&plain.records, &plain.last_state),
            expected,
            "case {case}: the rules act"
        );
        let threaded: Run<Journal, _> = simulate_from_signal_func(&whole, &rules, 3).unwrap();
        assert_eq!(
            shown(&threaded.records, &threaded.last_state),
            expected,
            "case {case}: threads"
        );
        for cut in CUTS {
            let head: Run<Journal, _> =
                simulate_from_signal_func(&signals_args(&data.rows(0..cut), case), &rules, 1)
                    .unwrap();
            let rest = FromSignals {
                resume: Some(&head.last_state),
                ..signals_args(&data, case)
            };
            let tail: Run<Journal, _> = simulate_from_signal_func(&rest, &rules, 1).unwrap();
            let strip = |r: Run<Journal, _>| Run {
                records: r.records,
                states: Vec::new(),
                last_state: r.last_state,
            };
            assert_eq!(
                chained([strip(head), strip(tail)], &groups),
                expected,
                "case {case}, cut {cut}"
            );
        }
    }
}
