//! What a run with margin columns looks like once rebuilt from its records,
//! worked by hand.
//!
//! Two accounts of 1000 over four bars, no fees:
//! - column 0 buys 10 at 100 on 20% margin and sells at 95;
//! - column 1 shorts 10 at 100 on 25% margin and covers at 80.
//!
//! The simulation settles each position's profit into cash every bar; the
//! records keep only the fills. These check that the rebuilt equity and free
//! cash are what the simulation saw.
use alphax_analysis::portfolio::Portfolio;
use alphax_core::flex::FlexArray;
use alphax_model::AccountType;
use alphax_sim::from_orders::FromOrders;
use alphax_sim::groups::Groups;

const N: f64 = f64::NAN;

fn close_to(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < 1e-9, "{actual:?} vs {expected:?}");
    }
}

const CLOSE: [f64; 8] = [
    100., 110., 90., 95., //
    100., 80., 90., 80., //
];
const SIZE: [f64; 8] = [
    10., N, N, -10., //
    -10., N, N, 10., //
];
const MARGIN: [f64; 2] = [0.2, 0.25];

fn portfolio() -> Portfolio<'static> {
    let args = FromOrders {
        n_rows: 4,
        n_cols: 2,
        close: FlexArray::full(&CLOSE, 4, 2).unwrap(),
        size: FlexArray::full(&SIZE, 4, 2).unwrap(),
        init_cash: FlexArray::scalar(1000.),
        account_type: FlexArray::scalar(AccountType::Margin),
        initial_margin: FlexArray::per_column(&MARGIN, 2).unwrap(),
        groups: Groups::ungrouped(2),
        ..FromOrders::default()
    };
    Portfolio::from_orders(&args, 1).unwrap()
}

#[test]
fn value_is_the_equity_the_settlements_add_up_to() {
    let pf = portfolio();
    close_to(pf.value().column(0), &[1000., 1100., 900., 950.]);
    close_to(pf.value().column(1), &[1000., 1200., 1100., 1200.]);
}

#[test]
fn cash_is_charged_the_whole_price_as_in_any_other_column() {
    let pf = portfolio();
    close_to(pf.cash().column(0), &[0., 0., 0., 950.]);
    close_to(pf.cash().column(1), &[2000., 2000., 2000., 1200.]);
}

#[test]
fn free_cash_is_the_equity_less_the_margin_posted() {
    let pf = portfolio();
    let free = pf.free_cash().unwrap();
    // The simulation's own free cash after each bar, settled.
    close_to(free.column(0), &[800., 900., 700., 950.]);
    close_to(free.column(1), &[750., 950., 850., 1200.]);
}

#[test]
fn gross_exposure_counts_the_whole_contract() {
    let pf = portfolio();
    let gross = pf.gross_exposure().unwrap();
    close_to(
        gross.column(0),
        &[1000. / 1800., 1100. / 2000., 900. / 1600., 0.],
    );
    close_to(
        gross.column(1),
        &[1000. / 1750., 800. / 1750., 900. / 1750., 0.],
    );
}

#[test]
fn a_hedged_spot_and_perpetual_pair_is_worth_what_it_started_with() {
    // Spot bought for cash and a perpetual shorted on 10% margin, sharing
    // one account: at 120 one gains what the other loses.
    let close = [100., 120., 100., 120.];
    let size = [5., N, -5., N];
    let types = [AccountType::Cash, AccountType::Margin];
    let margin = [N, 0.1];
    let args = FromOrders {
        n_rows: 2,
        n_cols: 2,
        close: FlexArray::full(&close, 2, 2).unwrap(),
        size: FlexArray::full(&size, 2, 2).unwrap(),
        init_cash: FlexArray::scalar(1000.),
        account_type: FlexArray::per_column(&types, 2).unwrap(),
        initial_margin: FlexArray::per_column(&margin, 2).unwrap(),
        groups: Groups::from_lens(&[2], 2).unwrap(),
        ..FromOrders::default()
    };
    let pf = Portfolio::from_orders(&args, 1).unwrap();
    close_to(pf.value().column(0), &[1000., 1000.]);
    close_to(pf.cash().column(0), &[1000., 1000.]);
    // 500 in the spot, 50 posted, and then the short's 100 loss.
    close_to(pf.free_cash().unwrap().column(0), &[450., 350.]);
}
