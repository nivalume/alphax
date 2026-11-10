//! Single-call golden test: replay the committed expected outputs.
//!
//! The expected outputs are committed in `tests/fixtures/execute_order.tsv`,
//! so this test needs no Python. They are never regenerated to make a change
//! pass: a failing line means the kernel's behaviour changed.
//!
//! One fixture is one `execute_order` call. Diffing whole simulations tells you
//! *that* something drifted; diffing single calls tells you *which* stage of
//! the size cascade did it. Chained behaviour is covered by
//! `golden_sequence.rs`.
//!
//! The comparison is field by field, never on aggregates. An equity curve that
//! matches to four decimals hides sign errors and off-by-one indices; a
//! per-field comparison does not.

mod common;

use alphax_exec::{ProcessOrderState, StatusInfo, execute_order};
use common::{Table, agree, parse_side, parse_status, parse_status_info};

#[test]
fn matches_the_golden_outputs() {
    let table = Table::load("execute_order.tsv");
    assert!(
        table.len() > 400,
        "expected the full fixture set, got {} cases",
        table.len()
    );

    let mut failures: Vec<String> = Vec::new();

    for row in table.iter() {
        let name = row.str("name");
        let state = ProcessOrderState {
            cash: row.f64("in_cash"),
            position: row.f64("in_position"),
            debt: row.f64("in_debt"),
            // The fixture has no collateral of its own: a short ties up its
            // value twice, once as debt and once as this.
            locked_cash: if row.f64("in_position") < 0.0 {
                row.f64("in_debt")
            } else {
                0.0
            },
            free_cash: row.f64("in_free_cash"),
            val_price: row.f64("in_val_price"),
            value: row.f64("in_value"),
            oidx: 0,
            lidx: 0,
        };

        let outcome = execute_order(&state, &row.order(), None);

        match (row.str("outcome"), &outcome) {
            ("err", Err(_)) => {}
            ("err", Ok((_, result))) => failures.push(format!(
                "{name}: expected an error ({}), we returned {result:?}",
                row.str("note")
            )),
            ("ok", Err(e)) => {
                failures.push(format!("{name}: expected success, we errored: {e}"));
            }
            ("ok", Ok((got_state, got_result))) => {
                let mut diffs: Vec<String> = Vec::new();
                let mut check = |field: &str, want: f64, got: f64| {
                    if !agree(want, got) {
                        diffs.push(format!("{field}: want {want:?}, got {got:?}"));
                    }
                };
                check("state.cash", row.f64("out_cash"), got_state.cash);
                check(
                    "state.position",
                    row.f64("out_position"),
                    got_state.position,
                );
                check("state.debt", row.f64("out_debt"), got_state.debt);
                check(
                    "state.free_cash",
                    row.f64("out_free_cash"),
                    got_state.free_cash,
                );
                check("result.size", row.f64("res_size"), got_result.size);
                check("result.price", row.f64("res_price"), got_result.price);
                check("result.fees", row.f64("res_fees"), got_result.fees);

                // Discrete fields must match exactly; a tolerance here would
                // defeat the purpose.
                let want_side = parse_side(row.str("res_side"));
                if want_side != got_result.side {
                    diffs.push(format!(
                        "result.side: want {want_side:?}, got {:?}",
                        got_result.side
                    ));
                }
                let want_status = parse_status(row.str("res_status"));
                if want_status != got_result.status {
                    diffs.push(format!(
                        "result.status: want {want_status:?}, got {:?}",
                        got_result.status
                    ));
                }
                let want_info = parse_status_info(row.str("res_status_info"));
                if want_info != got_result.status_info {
                    diffs.push(format!(
                        "result.status_info: want {want_info:?}, got {:?}",
                        got_result.status_info
                    ));
                }

                if !diffs.is_empty() {
                    failures.push(format!("{name}: {}", diffs.join("; ")));
                }
            }
            (other, _) => panic!("{name}: unknown outcome {other:?}"),
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} fixtures disagree with the golden outputs:\n  {}",
        failures.len(),
        table.len(),
        failures.join("\n  ")
    );
}

#[test]
fn fixtures_cover_every_reachable_status_info() {
    // A fixture set that has stopped exercising a rejection path would pass
    // silently while that path rots. RandomEvent is excluded on purpose: it
    // depends on a random draw, so it is covered by a unit test
    // against an injected draw instead.
    let table = Table::load("execute_order.tsv");
    let mut seen: Vec<StatusInfo> = Vec::new();
    for row in table.iter() {
        if row.str("outcome") == "ok"
            && let Some(info) = parse_status_info(row.str("res_status_info"))
            && !seen.contains(&info)
        {
            seen.push(info);
        }
    }

    let required = [
        StatusInfo::SizeNaN,
        StatusInfo::PriceNaN,
        StatusInfo::ValPriceNaN,
        StatusInfo::ValueNaN,
        StatusInfo::ValueZeroNeg,
        StatusInfo::SizeZero,
        StatusInfo::NoCashShort,
        StatusInfo::NoCashLong,
        StatusInfo::NoOpenPosition,
        StatusInfo::MaxSizeExceeded,
        StatusInfo::CantCoverFees,
        StatusInfo::MinSizeNotReached,
        StatusInfo::PartialFill,
    ];
    let missing: Vec<_> = required.iter().filter(|r| !seen.contains(r)).collect();
    assert!(missing.is_empty(), "fixtures no longer cover: {missing:?}");
}
