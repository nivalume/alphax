//! Hand-calculated shared accounts, including the signal-to-analysis path.
#![allow(
    clippy::float_cmp,
    reason = "integer-price accounts have exact balances"
)]
use alphax_analysis::portfolio::Portfolio;
use alphax_core::flex::FlexArray;
use alphax_model::SizeType;
use alphax_sim::from_orders::FromOrders;
use alphax_sim::from_signals::FromSignals;
use alphax_sim::groups::{CallSeq, Groups};

#[test]
fn signal_rotation_releases_cash_and_accounts_stay_isolated() {
    let close = [10., 10., 12., 10., 10., 10., 10., 10., 12., 10., 10., 10.];
    let entries = [
        false, true, false, true, false, false, false, true, false, true, false, false,
    ];
    let exits = [
        false, false, false, false, true, false, false, false, false, false, true, false,
    ];
    let capital = [100., 100., 200., 200.];
    let args = FromSignals {
        n_rows: 3,
        n_cols: 4,
        close: FlexArray::full(&close, 3, 4).unwrap(),
        entries: FlexArray::full(&entries, 3, 4).unwrap(),
        exits: FlexArray::full(&exits, 3, 4).unwrap(),
        init_cash: FlexArray::per_column(&capital, 4).unwrap(),
        groups: Some(Groups::from_lens(&[2, 2], 4).unwrap()),
        call_seq: CallSeq::Auto,
        ..FromSignals::default()
    };
    let pf = Portfolio::from_signals(&args, 1).unwrap();
    assert_eq!(pf.n_cols(), 4);
    assert_eq!(pf.n_groups(), 2);
    assert_eq!(pf.value().column(0), [100., 100., 120.]);
    assert_eq!(pf.value().column(1), [200., 200., 240.]);
    assert_eq!(pf.assets().column(0), [0., 10., 10.]);
    assert_eq!(pf.trade_counts().unwrap(), [2, 2]);
    for threads in [2, 4] {
        let parallel = Portfolio::from_signals(&args, threads).unwrap();
        assert_eq!(pf.orders(), parallel.orders());
        assert_eq!(pf.value(), parallel.value());
    }
    let default = FromSignals {
        call_seq: CallSeq::Default,
        ..args
    };
    let pf = Portfolio::from_signals(&default, 1).unwrap();
    assert_eq!(pf.value().column(0), [100., 100., 100.]);
    assert_eq!(pf.orders().len(), 4);
}

#[test]
fn target_weights_use_account_value_and_count_fees_once() {
    let close = [10., 12., 20., 18.];
    let weights = [0.4, f64::NAN, 0.4, f64::NAN];
    let args = FromOrders {
        n_rows: 2,
        n_cols: 2,
        close: FlexArray::full(&close, 2, 2).unwrap(),
        size: FlexArray::full(&weights, 2, 2).unwrap(),
        size_type: FlexArray::scalar(SizeType::TargetPercent),
        fees: FlexArray::scalar(0.01),
        init_cash: FlexArray::scalar(100.),
        groups: Groups::from_lens(&[2], 2).unwrap(),
        call_seq: CallSeq::Auto,
        ..FromOrders::default()
    };
    let pf = Portfolio::from_orders(&args, 2).unwrap();
    assert!((pf.cash().get(0, 0) - 19.2).abs() < 1e-12);
    assert!((pf.value().get(0, 0) - 99.2).abs() < 1e-12);
    assert!((pf.value().get(1, 0) - 103.2).abs() < 1e-12);
    assert!((pf.returns().get(0, 0) + 0.008).abs() < 1e-12);
    assert_eq!(pf.trades().unwrap().col, [0, 1]);
    assert_eq!(pf.trade_counts().unwrap(), [2]);
}

#[test]
fn flat_missing_symbol_does_not_poison_the_account() {
    let close = [10., 11., f64::NAN, f64::NAN];
    let size = [1., 0., f64::NAN, f64::NAN];
    let args = FromOrders {
        n_rows: 2,
        n_cols: 2,
        close: FlexArray::full(&close, 2, 2).unwrap(),
        size: FlexArray::full(&size, 2, 2).unwrap(),
        groups: Groups::from_lens(&[2], 2).unwrap(),
        ..FromOrders::default()
    };
    let pf = Portfolio::from_orders(&args, 1).unwrap();
    assert_eq!(pf.value().column(0), [100., 101.]);
}

#[test]
fn shared_signal_logs_identify_accounts_and_survive_parallel_merge() {
    let close = [10.; 12];
    let entries = [
        true, false, false, true, false, false, true, false, false, true, false, false,
    ];
    let exits = [
        false, true, false, false, true, false, false, true, false, false, true, false,
    ];
    let args = FromSignals {
        n_rows: 3,
        n_cols: 4,
        close: FlexArray::full(&close, 3, 4).unwrap(),
        entries: FlexArray::full(&entries, 3, 4).unwrap(),
        exits: FlexArray::full(&exits, 3, 4).unwrap(),
        size: FlexArray::scalar(8.),
        groups: Some(Groups::from_lens(&[2, 2], 4).unwrap()),
        call_seq: CallSeq::Auto,
        log: FlexArray::scalar(true),
        ..FromSignals::default()
    };
    let serial = alphax_sim::simulate_from_signals_logged(&args).unwrap();
    // Each account buys 8 then only 2: both share the same 100 cash.
    assert_eq!(serial.orders.size, [8., 2., 8., 2., 8., 2., 8., 2.]);
    for i in 0..serial.logs.len() {
        assert_eq!(serial.logs.group[i], serial.logs.col[i] / 2);
        let order = serial.logs.order_id[i] as usize;
        assert_eq!(serial.logs.col[i], serial.orders.col[order]);
        assert_eq!(serial.logs.idx[i], serial.orders.idx[order]);
    }
    for threads in [2, 4] {
        let parallel = alphax_sim::simulate_from_signals_logged_parallel(&args, threads).unwrap();
        assert_eq!(serial.orders, parallel.orders);
        assert_eq!(format!("{:?}", serial.logs), format!("{:?}", parallel.logs));
    }
}

#[test]
fn grouped_benchmark_width_and_signal_grouping_are_validated() {
    let args = FromSignals {
        n_rows: 2,
        n_cols: 2,
        close: FlexArray::scalar(10.),
        groups: Some(Groups::from_lens(&[2], 2).unwrap()),
        ..FromSignals::default()
    };
    let pf = Portfolio::from_signals(&args, 1).unwrap();
    assert!(
        pf.with_benchmark(alphax_core::matrix::Matrix::zeros(2, 2))
            .is_err()
    );
    let pf = Portfolio::from_signals(&args, 1).unwrap();
    assert!(
        pf.with_benchmark(alphax_core::matrix::Matrix::zeros(2, 1))
            .is_ok()
    );
    let invalid = FromSignals {
        groups: Some(Groups::ungrouped(3)),
        ..args
    };
    assert!(alphax_sim::simulate_from_signals(&invalid).is_err());
    assert!(alphax_sim::simulate_from_signals_parallel(&invalid, 2).is_err());
}
