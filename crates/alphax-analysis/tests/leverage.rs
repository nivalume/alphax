//! What a levered run looks like once rebuilt from its records, worked by hand.
//!
//! Four accounts of 1000 over three bars, no fees:
//! - column 0 buys 15 at 100 on leverage 2, lazily: 500 borrowed;
//! - column 1 shorts 10 at 100 on leverage 2 and covers at 80;
//! - column 2 shorts 5 at 100 without leverage and covers at 80;
//! - column 3 buys 5 at 100 on leverage 2, eagerly: 250 borrowed.
use alphax_analysis::portfolio::Portfolio;
use alphax_core::flex::FlexArray;
use alphax_model::LeverageMode;
use alphax_sim::from_orders::FromOrders;
use alphax_sim::groups::Groups;

const N: f64 = f64::NAN;

fn close_to(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < 1e-9, "{actual:?} vs {expected:?}");
    }
}

const CLOSE: [f64; 12] = [
    100., 120., 110., //
    100., 80., 80., //
    100., 80., 80., //
    100., 100., 100., //
];
const SIZE: [f64; 12] = [
    15., N, -15., //
    -10., N, 10., //
    -5., N, 5., //
    5., N, N, //
];
const LEVERAGE: [f64; 4] = [2., 2., 1., 2.];
const MODE: [LeverageMode; 4] = [
    LeverageMode::Lazy,
    LeverageMode::Lazy,
    LeverageMode::Lazy,
    LeverageMode::Eager,
];

fn portfolio() -> Portfolio<'static> {
    let args = FromOrders {
        n_rows: 3,
        n_cols: 4,
        close: FlexArray::full(&CLOSE, 3, 4).unwrap(),
        size: FlexArray::full(&SIZE, 3, 4).unwrap(),
        init_cash: FlexArray::scalar(1000.),
        leverage: FlexArray::per_column(&LEVERAGE, 4).unwrap(),
        leverage_mode: FlexArray::per_column(&MODE, 4).unwrap(),
        groups: Groups::ungrouped(4),
        ..FromOrders::default()
    };
    Portfolio::from_orders(&args, 1).unwrap()
}

#[test]
fn value_is_equity_net_of_what_the_longs_borrowed() {
    let pf = portfolio();
    close_to(pf.value().column(0), &[1000., 1300., 1150.]);
    close_to(pf.value().column(1), &[1000., 1200., 1200.]);
    close_to(pf.value().column(2), &[1000., 1100., 1100.]);
    close_to(pf.value().column(3), &[1000., 1000., 1000.]);
}

#[test]
fn cash_is_charged_the_whole_price_and_goes_negative_on_borrowing() {
    let pf = portfolio();
    close_to(pf.cash().column(0), &[-500., -500., 1150.]);
    close_to(pf.cash().column(3), &[500., 500., 500.]);
}

#[test]
fn free_cash_reserves_what_each_short_ties_up() {
    let pf = portfolio();
    let free = pf.free_cash().unwrap();
    // Borrowing is not free: the longs' free cash is their cash.
    close_to(free.column(0), &[-500., -500., 1150.]);
    close_to(free.column(3), &[500., 500., 500.]);
    // 2000 held against 1000 of proceeds and 500 of collateral.
    close_to(free.column(1), &[500., 500., 1200.]);
    // Without leverage the reserve is twice the value, as before.
    close_to(free.column(2), &[500., 500., 1100.]);
}

#[test]
fn gross_exposure_shows_the_leverage() {
    let pf = portfolio();
    let gross = pf.gross_exposure().unwrap();
    close_to(gross.column(0), &[1.5, 1800. / 1300., 0.]);
    close_to(gross.column(1), &[1000. / 1500., 800. / 1300., 0.]);
    close_to(gross.column(2), &[0.5, 400. / 900., 0.]);
    close_to(gross.column(3), &[0.5, 0.5, 0.5]);
}
