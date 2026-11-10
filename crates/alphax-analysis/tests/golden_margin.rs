//! A margin account on a linear perpetual.
//!
//! `tests/fixtures/margin.tsv` holds four cases of a schedule of market
//! orders: each bar's fill, at its close, and the account's expected equity at
//! that close before the bar's order (10x leverage, netting). Replayed on a
//! alphax margin column — the same sizes, prices and taker fee — the rebuilt
//! value of every bar is that equity less the bar's fee, and every fill pays
//! the fixture's commission. The fixture rounds money to 8 decimals, hence the
//! tolerance.
use std::path::PathBuf;

use alphax_analysis::Scoring;
use alphax_analysis::portfolio::Portfolio;
use alphax_core::flex::FlexArray;
use alphax_model::{AccountType, Direction};
use alphax_sim::from_orders::FromOrders;
use alphax_sim::groups::Groups;

const TAKER_FEE: f64 = 0.000_18;

#[derive(Default)]
struct Case {
    close: Vec<f64>,
    qty: Vec<f64>,
    px: Vec<f64>,
    fee: Vec<f64>,
    equity: Vec<f64>,
}

fn cases() -> Vec<Case> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/margin.tsv");
    let text = std::fs::read_to_string(path).unwrap();
    let mut out: Vec<Case> = Vec::new();
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    let at = |name: &str| header.iter().position(|h| *h == name).unwrap();
    let (case, close, qty, px, fee, equity) = (
        at("case"),
        at("close"),
        at("fill_qty"),
        at("fill_px"),
        at("fee"),
        at("equity"),
    );
    for line in lines {
        let f: Vec<&str> = line.split('\t').collect();
        let num = |k: usize| f[k].parse::<f64>().unwrap();
        let c: usize = f[case].parse().unwrap();
        if out.len() <= c {
            out.push(Case::default());
        }
        let row = &mut out[c];
        row.close.push(num(close));
        row.qty.push(num(qty));
        row.px.push(num(px));
        row.fee.push(num(fee));
        row.equity.push(num(equity));
    }
    out
}

#[test]
fn a_margin_column_keeps_the_expected_books() {
    let all = cases();
    assert_eq!(all.len(), 4);
    for (c, case) in all.iter().enumerate() {
        let n = case.close.len();
        let size: Vec<f64> = case
            .qty
            .iter()
            .map(|&q| if q == 0.0 { f64::NAN } else { q })
            .collect();
        let args = FromOrders {
            n_rows: n,
            n_cols: 1,
            init_cash: FlexArray::scalar(100_000.0),
            close: FlexArray::per_row(&case.close, n).unwrap(),
            size: FlexArray::per_row(&size, n).unwrap(),
            price: FlexArray::per_row(&case.px, n).unwrap(),
            fees: FlexArray::scalar(TAKER_FEE),
            direction: FlexArray::scalar(Direction::Both),
            account_type: FlexArray::scalar(AccountType::Margin),
            initial_margin: FlexArray::scalar(0.1),
            groups: Groups::ungrouped(1),
            ..FromOrders::default()
        };
        let orders = alphax_sim::simulate_from_orders(&args).unwrap();
        let fills = case.qty.iter().filter(|&&q| q != 0.0).count();
        assert_eq!(orders.len(), fills, "case {c}: every scheduled order fills");
        let mut k = 0;
        for (row, &q) in case.qty.iter().enumerate() {
            if q == 0.0 {
                continue;
            }
            assert_eq!(orders.idx[k], row as i64);
            assert!(
                (orders.fees[k] - case.fee[row]).abs() < 1e-7,
                "case {c}, bar {row}: fee"
            );
            k += 1;
        }
        let pf =
            Portfolio::from_records(orders, args.close, args.init_cash, n, 1, Scoring::default())
                .unwrap();
        let value = pf.value();
        for row in 0..n {
            let ours = value.get(row, 0) + case.fee[row];
            assert!(
                (ours - case.equity[row]).abs() < 1e-6,
                "case {c}, bar {row}: alphax {ours}, expected {}",
                case.equity[row]
            );
        }
    }
}
