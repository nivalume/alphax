//! Exposure against total account value, worked by hand.
//!
//! Three accounts over three bars:
//! - account 0 (column 0) buys 5 at 10, marks at 12, sells at 12;
//! - account 1 (column 1) shorts 5 at 10 and marks at 8;
//! - account 2 (columns 2 and 3, shared cash) is long 4 of one symbol and
//!   short 2 of another, dollar-neutral until the last bar.
use alphax_analysis::portfolio::Portfolio;
use alphax_core::flex::FlexArray;
use alphax_core::matrix::Matrix;
use alphax_sim::from_orders::FromOrders;
use alphax_sim::groups::Groups;

const N: f64 = f64::NAN;

fn close_to(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < 1e-12, "{actual:?} vs {expected:?}");
    }
}

fn run<'a>(close: &'a [f64], size: &'a [f64]) -> FromOrders<'a> {
    FromOrders {
        n_rows: 3,
        n_cols: 4,
        close: FlexArray::full(close, 3, 4).unwrap(),
        size: FlexArray::full(size, 3, 4).unwrap(),
        init_cash: FlexArray::scalar(100.),
        groups: Groups::from_lens(&[1, 1, 2], 4).unwrap(),
        ..FromOrders::default()
    }
}

const CLOSE: [f64; 12] = [
    10., 12., 12., // long-only
    10., 8., 8., // short
    10., 10., 15., // shared, long leg
    20., 20., 10., // shared, short leg
];
const SIZE: [f64; 12] = [
    5., N, -5., //
    -5., N, N, //
    4., N, N, //
    -2., N, N, //
];

#[test]
fn the_hand_built_accounts_have_the_expected_values() {
    let args = run(&CLOSE, &SIZE);
    let pf = Portfolio::from_orders(&args, 1).unwrap();
    close_to(pf.value().column(0), &[100., 110., 110.]);
    close_to(pf.value().column(1), &[100., 110., 110.]);
    close_to(pf.value().column(2), &[100., 100., 140.]);
    close_to(pf.assets().column(1), &[-5., -5., -5.]);
    close_to(pf.assets().column(3), &[-2., -2., -2.]);
}

#[test]
fn allocations_are_signed_weights_of_the_whole_account() {
    let args = run(&CLOSE, &SIZE);
    let pf = Portfolio::from_orders(&args, 1).unwrap();
    let alloc = pf.allocations();
    assert_eq!((alloc.n_rows(), alloc.n_cols()), (3, 4));
    close_to(alloc.column(0), &[0.5, 60. / 110., 0.]);
    close_to(alloc.column(1), &[-0.5, -40. / 110., -40. / 110.]);
    close_to(alloc.column(2), &[0.4, 0.4, 60. / 140.]);
    close_to(alloc.column(3), &[-0.4, -0.4, -20. / 140.]);

    // Within an account the weights add up to the invested fraction.
    let (value, cash) = (pf.value(), pf.cash());
    for group in 0..pf.n_groups() {
        for row in 0..pf.n_rows() {
            let sum: f64 = pf
                .groups()
                .range(group)
                .map(|col| alloc.get(row, col))
                .sum();
            let v = value.get(row, group);
            let invested = (v - cash.get(row, group)) / v;
            assert!((sum - invested).abs() < 1e-12, "group {group} row {row}");
        }
    }
}

#[test]
fn long_short_net_and_gross_by_hand() {
    let args = run(&CLOSE, &SIZE);
    let pf = Portfolio::from_orders(&args, 1).unwrap();
    let (long, short) = (pf.long_exposure(), pf.short_exposure());
    let (net, gross) = (pf.net_exposure(), pf.gross_long_short());
    for m in [&long, &short, &net, &gross] {
        assert_eq!((m.n_rows(), m.n_cols()), (3, 3));
    }

    close_to(long.column(0), &[0.5, 60. / 110., 0.]);
    close_to(short.column(0), &[0., 0., 0.]);
    close_to(net.column(0), &[0.5, 60. / 110., 0.]);
    close_to(gross.column(0), &[0.5, 60. / 110., 0.]);

    close_to(long.column(1), &[0., 0., 0.]);
    close_to(short.column(1), &[0.5, 40. / 110., 40. / 110.]);
    close_to(net.column(1), &[-0.5, -40. / 110., -40. / 110.]);
    close_to(gross.column(1), &[0.5, 40. / 110., 40. / 110.]);

    // Dollar-neutral until the long leg rallies and the short leg falls.
    close_to(long.column(2), &[0.4, 0.4, 60. / 140.]);
    close_to(short.column(2), &[0.4, 0.4, 20. / 140.]);
    close_to(net.column(2), &[0., 0., 40. / 140.]);
    close_to(gross.column(2), &[0.8, 0.8, 80. / 140.]);
}

#[test]
fn on_a_long_only_account_both_gross_measures_agree() {
    let args = run(&CLOSE, &SIZE);
    let pf = Portfolio::from_orders(&args, 1).unwrap();
    let free_cash_based = pf.gross_exposure().unwrap();
    close_to(free_cash_based.column(0), pf.gross_long_short().column(0));
}

#[test]
fn coverage_counts_bars_in_a_position() {
    let args = run(&CLOSE, &SIZE);
    let pf = Portfolio::from_orders(&args, 1).unwrap();
    close_to(&pf.position_coverage(), &[2. / 3., 1., 1., 1.]);
}

#[test]
fn a_worthless_account_has_no_weights() {
    let close = [10.; 3];
    let size = [N; 3];
    let args = FromOrders {
        n_rows: 3,
        n_cols: 1,
        close: FlexArray::full(&close, 3, 1).unwrap(),
        size: FlexArray::full(&size, 3, 1).unwrap(),
        init_cash: FlexArray::scalar(0.),
        groups: Groups::ungrouped(1),
        ..FromOrders::default()
    };
    let pf = Portfolio::from_orders(&args, 1).unwrap();
    let nan = |m: &Matrix| m.as_slice().iter().all(|v| v.is_nan());
    assert!(nan(&pf.allocations()));
    assert!(nan(&pf.long_exposure()));
    assert!(nan(&pf.short_exposure()));
    assert!(nan(&pf.net_exposure()));
    assert!(nan(&pf.gross_long_short()));
    close_to(&pf.position_coverage(), &[0.]);
}
