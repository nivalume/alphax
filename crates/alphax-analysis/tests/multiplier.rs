//! Contract multipliers, simulated and rebuilt from the records.
//!
//! A multiplier `k` makes one contract worth `k` units: a run at `k` must be
//! the run at 1 with every amount-sized order `k` times larger — the same
//! cash, fees, values, charges and liquidations, with sizes `k` times
//! smaller. That is checked on random runs of both loops mixing cash and
//! margin columns, leverage, shared accounts, funding, dividends and
//! liquidation; a futures position is checked by hand.
use alphax_analysis::portfolio::Portfolio;
use alphax_core::flex::FlexArray;
use alphax_model::{AccountType, CashFlowKind, Direction, MarginBasis, MarginMode, SizeType};
use alphax_records::OrderRecords;
use alphax_sim::from_orders::FromOrders;
use alphax_sim::from_signals::FromSignals;
use alphax_sim::groups::Groups;

const N: f64 = f64::NAN;

fn close_to(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
    for (a, e) in actual.iter().zip(expected) {
        assert!(
            (a - e).abs() <= 1e-9 * e.abs().max(1.0),
            "{actual:?} vs {expected:?}"
        );
    }
}

#[test]
fn a_futures_position_moves_by_the_multiplier_per_point() {
    // One E-mini at 4000, 50 a point, 12000 margin a contract.
    let close = [4000., 4010., 3990., 3990.];
    let size = [1., N, N, -1.];
    let args = FromOrders {
        n_rows: 4,
        n_cols: 1,
        close: FlexArray::per_row(&close, 4).unwrap(),
        size: FlexArray::per_row(&size, 4).unwrap(),
        init_cash: FlexArray::scalar(20_000.),
        account_type: FlexArray::scalar(AccountType::Margin),
        initial_margin: FlexArray::scalar(12_000.),
        margin_basis: FlexArray::scalar(MarginBasis::Unit),
        multiplier: FlexArray::scalar(50.),
        fees: FlexArray::scalar(1e-5),
        groups: Groups::ungrouped(1),
        ..FromOrders::default()
    };
    let pf = Portfolio::from_orders(&args, 1).unwrap();
    // A fee of 1e-5 on 200,000 of notional is 2 per side.
    let orders = pf.orders();
    assert_eq!(orders.price, [4000., 3990.]);
    assert_eq!(orders.multiplier, [50., 50.]);
    close_to(&orders.fees, &[2., 1.995]);
    close_to(
        pf.value().column(0),
        &[19_998., 20_498., 19_498., 19_496.005],
    );
    close_to(
        pf.free_cash().unwrap().column(0),
        &[7_998., 8_498., 7_498., 19_496.005],
    );
    let trades = pf.trades().unwrap();
    assert_eq!(trades.multiplier, [50.]);
    close_to(&trades.pnl, &[-500. - 2. - 1.995]);
    close_to(&trades.ret, &[(-500. - 2. - 1.995) / 200_000.]);
}

#[test]
fn a_bad_multiplier_is_refused() {
    let close = [100., 101.];
    let per_row = [1., 2.];
    for (multiplier, message) in [
        (
            FlexArray::scalar(0.),
            "multiplier must be finite and greater than 0",
        ),
        (
            FlexArray::scalar(N),
            "multiplier must be finite and greater than 0",
        ),
        (
            FlexArray::scalar(-5.),
            "multiplier must be finite and greater than 0",
        ),
        (
            FlexArray::per_row(&per_row, 2).unwrap(),
            "multiplier changes down column 0",
        ),
    ] {
        let args = FromOrders {
            n_rows: 2,
            n_cols: 1,
            close: FlexArray::per_row(&close, 2).unwrap(),
            size: FlexArray::scalar(1.),
            multiplier,
            groups: Groups::ungrouped(1),
            ..FromOrders::default()
        };
        let err = alphax_sim::simulate_from_orders(&args).unwrap_err();
        assert!(err.to_string().contains(message), "{err}");
    }
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

struct Market {
    n_rows: usize,
    n_cols: usize,
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
}

fn market(rng: &mut Lcg, n_rows: usize, n_cols: usize) -> Market {
    let (mut open, mut high, mut low, mut close) = (vec![], vec![], vec![], vec![]);
    for _ in 0..n_cols {
        let mut p = 20.0 + 10.0 * rng.next();
        for _ in 0..n_rows {
            let gap = if rng.next() < 0.1 { 0.3 } else { 0.01 };
            let o = p * (1.0 + gap * (rng.next() - 0.5) * 2.0);
            p = o * (1.0 + 0.06 * (rng.next() - 0.5));
            open.push(o);
            close.push(p);
            high.push(o.max(p) * (1.0 + 0.02 * rng.next()));
            low.push(o.min(p) * (1.0 - 0.02 * rng.next()));
        }
    }
    Market {
        n_rows,
        n_cols,
        open,
        high,
        low,
        close,
    }
}

const TYPES: [AccountType; 4] = [
    AccountType::Margin,
    AccountType::Cash,
    AccountType::Margin,
    AccountType::Margin,
];
const MODES: [MarginMode; 4] = [
    MarginMode::Cross,
    MarginMode::Cross,
    MarginMode::Isolated,
    MarginMode::Cross,
];

/// Checks that the run at `k` (`a`) is the run at 1 (`b`) in contracts of
/// `k`: its sizes are `k` times smaller and everything in cash the same.
fn same_in_contracts(a: &Portfolio<'_>, b: &Portfolio<'_>, k: f64) {
    let (oa, ob): (&OrderRecords, &OrderRecords) = (a.orders(), b.orders());
    assert!(oa.len() > 20, "too few fills to say anything");
    assert_eq!(oa.col, ob.col);
    assert_eq!(oa.idx, ob.idx);
    assert_eq!(oa.side, ob.side);
    assert_eq!(oa.liquidation, ob.liquidation);
    let scaled: Vec<f64> = oa.size.iter().map(|s| s * k).collect();
    close_to(&scaled, &ob.size);
    close_to(&oa.price, &ob.price);
    close_to(&oa.fees, &ob.fees);
    close_to(&oa.debt, &ob.debt);
    close_to(&oa.locked_cash, &ob.locked_cash);
    assert!(oa.multiplier.iter().all(|&m| m.to_bits() == k.to_bits()));
    assert_eq!(oa.cash.kind, ob.cash.kind);
    close_to(&oa.cash.amount, &ob.cash.amount);
    close_to(a.value().as_slice(), b.value().as_slice());
    close_to(a.returns().as_slice(), b.returns().as_slice());
    close_to(
        a.free_cash().unwrap().as_slice(),
        b.free_cash().unwrap().as_slice(),
    );
    let assets: Vec<f64> = a.assets().as_slice().iter().map(|s| s * k).collect();
    close_to(&assets, b.assets().as_slice());
    let (ta, tb) = (a.trades().unwrap(), b.trades().unwrap());
    close_to(&ta.pnl, &tb.pnl);
    close_to(&ta.ret, &tb.ret);
}

// Column 3 is sized as a share of the account, which a multiplier must turn
// into fewer contracts by itself.
const SIZE_TYPES: [SizeType; 4] = [
    SizeType::Amount,
    SizeType::Amount,
    SizeType::Amount,
    SizeType::TargetPercent,
];

fn orders_run<'a>(m: &'a Market, size: &'a [f64], k: f64, groups: &Groups) -> FromOrders<'a> {
    let (n_rows, n_cols) = (m.n_rows, m.n_cols);
    FromOrders {
        n_rows,
        n_cols,
        open: FlexArray::full(&m.open, n_rows, n_cols).unwrap(),
        high: FlexArray::full(&m.high, n_rows, n_cols).unwrap(),
        low: FlexArray::full(&m.low, n_rows, n_cols).unwrap(),
        close: FlexArray::full(&m.close, n_rows, n_cols).unwrap(),
        size: FlexArray::full(size, n_rows, n_cols).unwrap(),
        size_type: FlexArray::per_column(&SIZE_TYPES, n_cols).unwrap(),
        init_cash: FlexArray::scalar(300.),
        fees: FlexArray::scalar(0.001),
        fixed_fees: FlexArray::scalar(0.01),
        slippage: FlexArray::scalar(0.001),
        direction: FlexArray::scalar(Direction::Both),
        account_type: FlexArray::per_column(&TYPES, n_cols).unwrap(),
        margin_mode: FlexArray::per_column(&MODES, n_cols).unwrap(),
        leverage: FlexArray::scalar(2.),
        initial_margin: FlexArray::scalar(0.1),
        maintenance_margin: FlexArray::scalar(0.05),
        liquidation_fee: FlexArray::scalar(0.01),
        funding_rate: FlexArray::scalar(1e-4),
        borrow_rate: FlexArray::scalar(1e-4),
        debt_rate: FlexArray::scalar(1e-4),
        cash_dividends: FlexArray::scalar(0.01),
        multiplier: FlexArray::scalar(k),
        groups: groups.clone(),
        ..FromOrders::default()
    }
}

fn check_orders(groups: &Groups, seed: u64, k: f64) {
    let (n_rows, n_cols) = (120, 4);
    let mut rng = Lcg(seed);
    let m = market(&mut rng, n_rows, n_cols);
    let size: Vec<f64> = (0..n_rows * n_cols)
        .map(|cell| {
            let u = rng.next();
            if u < 0.6 {
                N
            } else if cell / n_rows == 3 {
                (u - 0.8) * 5.0
            } else {
                (u - 0.8) * 200.0
            }
        })
        .collect();
    let contracts: Vec<f64> = size
        .iter()
        .enumerate()
        .map(|(cell, s)| if cell / n_rows == 3 { *s } else { s / k })
        .collect();
    let a = orders_run(&m, &contracts, k, groups);
    let b = orders_run(&m, &size, 1.0, groups);
    let pa = Portfolio::from_orders(&a, 1).unwrap();
    let pb = Portfolio::from_orders(&b, 1).unwrap();
    assert!(
        pb.orders().liquidation.contains(&true),
        "nothing liquidated"
    );
    assert!(
        pb.orders()
            .cash
            .kind
            .contains(&(CashFlowKind::Dividend as i8)),
        "no dividend"
    );
    same_in_contracts(&pa, &pb, k);

    // And threads change nothing.
    let serial = alphax_sim::simulate_from_orders(&a).unwrap();
    let threaded = alphax_sim::simulate_from_orders_parallel(&a, 3).unwrap();
    assert_eq!(serial, threaded);
}

#[test]
fn from_orders_at_a_multiplier_is_the_run_in_larger_sizes() {
    check_orders(&Groups::ungrouped(4), 5, 50.0);
    check_orders(&Groups::ungrouped(4), 6, 0.1);
}

#[test]
fn from_orders_at_a_multiplier_is_the_run_in_larger_sizes_in_shared_accounts() {
    check_orders(&Groups::from_lens(&[2, 2], 4).unwrap(), 17, 50.0);
}

fn check_signals(seed: u64, k: f64) {
    let (n_rows, n_cols) = (150, 4);
    let mut rng = Lcg(seed);
    let m = market(&mut rng, n_rows, n_cols);
    let flags = |rng: &mut Lcg, p: f64| -> Vec<bool> {
        (0..n_rows * n_cols).map(|_| rng.next() < p).collect()
    };
    let (le, lx, se, sx) = (
        flags(&mut rng, 0.1),
        flags(&mut rng, 0.05),
        flags(&mut rng, 0.1),
        flags(&mut rng, 0.05),
    );
    let run = |size: f64, size_type: SizeType, multiplier: f64| FromSignals {
        n_rows,
        n_cols,
        open: FlexArray::full(&m.open, n_rows, n_cols).unwrap(),
        high: FlexArray::full(&m.high, n_rows, n_cols).unwrap(),
        low: FlexArray::full(&m.low, n_rows, n_cols).unwrap(),
        close: FlexArray::full(&m.close, n_rows, n_cols).unwrap(),
        long_entries: FlexArray::full(&le, n_rows, n_cols).unwrap(),
        long_exits: FlexArray::full(&lx, n_rows, n_cols).unwrap(),
        short_entries: FlexArray::full(&se, n_rows, n_cols).unwrap(),
        short_exits: FlexArray::full(&sx, n_rows, n_cols).unwrap(),
        size: FlexArray::scalar(size),
        size_type: FlexArray::scalar(size_type),
        init_cash: FlexArray::scalar(1000.),
        fees: FlexArray::scalar(0.001),
        slippage: FlexArray::scalar(0.001),
        account_type: FlexArray::per_column(&TYPES, n_cols).unwrap(),
        margin_mode: FlexArray::per_column(&MODES, n_cols).unwrap(),
        initial_margin: FlexArray::scalar(0.1),
        maintenance_margin: FlexArray::scalar(0.05),
        funding_rate: FlexArray::scalar(1e-4),
        sl_stop: FlexArray::scalar(0.04),
        tp_stop: FlexArray::scalar(0.06),
        multiplier: FlexArray::scalar(multiplier),
        ..FromSignals::default()
    };
    for (size_a, size_b, size_type) in [
        (40.0 / k, 40.0, SizeType::Amount),
        (600.0, 600.0, SizeType::Value),
    ] {
        let a = run(size_a, size_type, k);
        let b = run(size_b, size_type, 1.0);
        let pa = Portfolio::from_signals(&a, 1).unwrap();
        let pb = Portfolio::from_signals(&b, 1).unwrap();
        same_in_contracts(&pa, &pb, k);
        let serial = alphax_sim::simulate_from_signals(&a).unwrap();
        let threaded = alphax_sim::simulate_from_signals_parallel(&a, 3).unwrap();
        assert_eq!(serial, threaded);
    }
}

#[test]
fn from_signals_at_a_multiplier_is_the_run_in_larger_sizes() {
    check_signals(9, 50.0);
    check_signals(10, 0.1);
}
