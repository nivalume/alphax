//! Sequence-level golden test: chain orders and compare at every step.
//!
//! The single-call harness (`golden.rs`) pins down one `execute_order`
//! invocation. It cannot see anything that only goes wrong when state is
//! threaded from one call to the next.
//!
//! **The replay feeds its own outputs forward.** It reads the initial cash from
//! the fixture and nothing else: every subsequent state comes from our kernel,
//! not from the expected column. Re-seeding from the fixture each step would
//! turn this back into the single-call test, because a drift introduced at
//! step 2 would be wiped before step 3 could amplify it.
//!
//! Chaining rule, as the fixture was recorded:
//!
//! ```text
//! state = flat(init_cash)
//! for each step i:
//!     val_price = fixture's val_price for the step
//!     value     = state.cash + state.position * val_price
//!     state, result = execute_order(state, order[i])
//! ```
//!
//! `val_price` and `value` are not kernel outputs — a simulation loop derives
//! them from the bar — so the rule is fixed on both sides.

mod common;

use alphax_exec::{ProcessOrderState, execute_order};
use common::{Table, agree, parse_side, parse_status, parse_status_info};

/// Replays every sequence, threading our own state forward.
#[test]
fn matches_the_golden_across_chained_orders() {
    let table = Table::load("order_sequence.tsv");
    assert!(!table.is_empty(), "sequence fixtures are empty");

    let mut failures: Vec<String> = Vec::new();
    let mut sequences_run = 0usize;

    // Rows are grouped by sequence and ordered by step, as written.
    let mut state: Option<ProcessOrderState> = None;
    let mut current_seq = String::new();
    let mut aborted = false;

    for row in table.iter() {
        let seq = row.str("seq").to_owned();
        let step = row.usize("step");

        if seq != current_seq {
            // New sequence: start flat at its initial cash.
            current_seq = seq.clone();
            sequences_run += 1;
            aborted = false;
            assert_eq!(step, 0, "sequence {seq} does not start at step 0");
            let init_cash = row.f64("init_cash");
            state = Some(ProcessOrderState::flat(init_cash, row.f64("val_price")));
        }

        if aborted {
            // A previous step errored; the recorded sequence stopped there too, so
            // there is nothing left to compare in this sequence.
            continue;
        }

        let val_price = row.f64("val_price");
        let mut st = state.expect("state is set at the start of each sequence");
        // Mark to market before executing, exactly as the fixture does.
        st.val_price = val_price;
        st.value = st.cash + st.position * val_price;

        let at = format!("{seq}[{step}]");
        let outcome = execute_order(&st, &row.order(), None);

        match (row.str("outcome"), &outcome) {
            ("err", Err(_)) => {
                aborted = true;
            }
            ("err", Ok((_, result))) => {
                failures.push(format!(
                    "{at}: expected an error ({}), we returned {result:?}",
                    row.str("note")
                ));
                aborted = true;
            }
            ("ok", Err(e)) => {
                failures.push(format!("{at}: expected success, we errored: {e}"));
                aborted = true;
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

                if diffs.is_empty() {
                    // Thread OUR state forward, not the fixture's.
                    let mut next = st;
                    next.cash = got_state.cash;
                    next.position = got_state.position;
                    next.debt = got_state.debt;
                    next.locked_cash = got_state.locked_cash;
                    next.free_cash = got_state.free_cash;
                    state = Some(next);
                } else {
                    failures.push(format!("{at}: {}", diffs.join("; ")));
                    // Everything after a divergence is noise.
                    aborted = true;
                }
            }
            (other, _) => panic!("{at}: unknown outcome {other:?}"),
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} sequences diverge from the golden outputs ({} steps total):\n  {}",
        failures.len(),
        sequences_run,
        table.len(),
        failures.join("\n  ")
    );
}

/// The sequences that exist to amplify sub-tolerance residues must be present.
///
/// Stage 2's mutation testing showed the zero-snapping machinery was
/// unverified: the residues it removes are ~1e-17 against a 1e-12 comparison
/// floor, invisible within one call. These sequences make a residue reach a
/// discrete branch one step later, where it flips a rejection into an error.
/// Losing them would silently reopen that gap.
#[test]
fn amplifier_sequences_are_present_and_exercise_an_exact_zero() {
    let table = Table::load("order_sequence.tsv");

    let amplifiers: Vec<_> = table
        .iter()
        .filter(|r| r.str("seq").starts_with("snap_amplifier"))
        .collect();
    assert!(
        amplifiers.len() >= 8,
        "expected the snap_amplifier sequences, found {} rows",
        amplifiers.len()
    );

    // Their whole point is that a balance lands on exactly zero and the next
    // step then depends on that being exact. If the recorded outputs stopped
    // containing a hard zero, the amplification is gone and the sequences no
    // longer test what they claim to.
    // Compared as text on purpose: the claim is that the fixture records a
    // hard "0.0", not a value that merely rounds to zero.
    let lands_on_zero = amplifiers
        .iter()
        .any(|r| r.str("out_cash") == "0.0" && r.str("res_status") == "Filled");
    assert!(
        lands_on_zero,
        "no amplifier sequence leaves cash at exactly zero after a fill"
    );

    let rejects_on_empty = amplifiers
        .iter()
        .any(|r| r.str("res_status_info") == "NoCashLong");
    assert!(
        rejects_on_empty,
        "no amplifier sequence rejects with NoCashLong, so nothing depends on the exact zero"
    );
}
