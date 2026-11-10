//! Generating signals: exits from entries, and alternating pairs.
//!
//! # A choice function
//!
//! The generators take a *choice function*: given a window `[from, to)` and a
//! column, return the indices to fire on. It is the right abstraction — a
//! time-based exit, a stop-based exit and a random exit are the same loop with
//! different choices.
//!
//! In Rust the callback is a generic parameter and monomorphizes away, so the
//! loop is written once and costs nothing. [`generate_exits`] takes one;
//! [`exit_after`] and [`exit_on`] are the two built on it.
//!
//! # `wait` is not a detail
//!
//! An exit generated on the same bar as its entry means a position that opens
//! and closes within one bar, which most data cannot support and most
//! backtests should not claim. `wait` defaults to 1 everywhere for
//! that reason, and 0 is allowed rather than encouraged.

use alphax_core::error::{Error, Result};

/// What an exit window may choose.
///
/// Called with a half-open window `[from, to)` and the column, and returns
/// the rows to fire on. An empty result means this entry gets no exit.
pub trait ExitChoice {
    /// Picks the rows to fire on within `[from, to)`.
    fn choose(&mut self, from: usize, to: usize, col: usize, out: &mut Vec<usize>);
}

impl<F> ExitChoice for F
where
    F: FnMut(usize, usize, usize, &mut Vec<usize>),
{
    fn choose(&mut self, from: usize, to: usize, col: usize, out: &mut Vec<usize>) {
        self(from, to, col, out);
    }
}

/// How far an exit may look for its bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitWindow {
    /// Stop at the next entry, so every exit belongs to exactly one entry.
    ///
    /// The only mode in which "which entry did this exit close?" has an
    /// answer, which is why it is the default everywhere.
    UntilNextEntry,
    /// Search to the end of the column, and skip entries that fall inside a
    /// window already resolved.
    ///
    /// Overlapping windows would otherwise let one bar exit several entries.
    ToEndSkippingResolved,
    /// Search to the end and resolve every entry independently.
    ///
    /// Exits may then coincide or cross, and no exit belongs to one entry.
    ToEnd,
}

/// Generates exits from entries, one window per entry.
///
/// `wait` bars pass after an entry before its window opens. `pick_first`
/// keeps only the earliest row the choice returns, which is what a stop or a
/// threshold wants; without it every chosen row fires.
///
/// # Errors
///
/// Returns [`Error::Invariant`] if the choice returns a row outside its
/// window, which would otherwise corrupt a neighbouring entry's range.
pub fn generate_exits<C: ExitChoice>(
    entries: &[bool],
    n_rows: usize,
    n_cols: usize,
    wait: usize,
    window: ExitWindow,
    pick_first: bool,
    choice: &mut C,
) -> Result<Vec<bool>> {
    if entries.len() != n_rows * n_cols {
        return Err(Error::Broadcast {
            left: (entries.len(), 1),
            right: (n_rows, n_cols),
        });
    }
    let mut exits = vec![false; n_rows * n_cols];
    let mut chosen: Vec<usize> = Vec::new();

    for col in 0..n_cols {
        let base = col * n_rows;
        let entry_rows: Vec<usize> = (0..n_rows).filter(|&i| entries[base + i]).collect();
        let mut last_exit: i64 = -1;

        for (k, &row) in entry_rows.iter().enumerate() {
            if window == ExitWindow::ToEndSkippingResolved && (row as i64) <= last_exit {
                continue;
            }
            let from = row + wait;
            let to = if window == ExitWindow::UntilNextEntry && k + 1 < entry_rows.len() {
                entry_rows[k + 1]
            } else {
                n_rows
            };
            if to <= from {
                continue;
            }

            chosen.clear();
            choice.choose(from, to, col, &mut chosen);
            if chosen.is_empty() {
                continue;
            }
            for &i in &chosen {
                if i < from || i >= to {
                    return Err(Error::Invariant(format!(
                        "an exit choice returned row {i}, outside its window {from}..{to}"
                    )));
                }
                exits[base + i] = true;
                if pick_first {
                    break;
                }
            }
            last_exit = if pick_first {
                chosen[0] as i64
            } else {
                *chosen.iter().max().expect("non-empty") as i64
            };
        }
    }
    Ok(exits)
}

/// Exits a fixed number of bars after each entry.
///
/// The simplest useful exit rule, and the one worth having because it is the
/// null hypothesis for every other: a strategy whose exits beat "hold for `n`
/// bars" has an exit rule; one that does not has an entry rule and a habit.
///
/// # Errors
///
/// As [`generate_exits`].
pub fn exit_after(
    entries: &[bool],
    n_rows: usize,
    n_cols: usize,
    bars: usize,
    window: ExitWindow,
) -> Result<Vec<bool>> {
    let mut choice = |from: usize, _to: usize, _col: usize, out: &mut Vec<usize>| {
        out.push(from);
    };
    // The wait *is* the holding period: the window opens `bars` after the
    // entry and the first row in it is taken.
    generate_exits(entries, n_rows, n_cols, bars, window, true, &mut choice)
}

/// Exits on the first bar in each window where `condition` holds.
///
/// `condition` is a column-major array of the same shape — a threshold
/// crossing, a stop level being breached, anything already computed.
///
/// # Errors
///
/// As [`generate_exits`], and rejects a `condition` of the wrong shape.
pub fn exit_on(
    entries: &[bool],
    condition: &[bool],
    n_rows: usize,
    n_cols: usize,
    wait: usize,
    window: ExitWindow,
) -> Result<Vec<bool>> {
    if condition.len() != n_rows * n_cols {
        return Err(Error::Broadcast {
            left: (condition.len(), 1),
            right: (n_rows, n_cols),
        });
    }
    let mut choice = |from: usize, to: usize, col: usize, out: &mut Vec<usize>| {
        let base = col * n_rows;
        if let Some(i) = (from..to).find(|&i| condition[base + i]) {
            out.push(i);
        }
    };
    generate_exits(entries, n_rows, n_cols, wait, window, true, &mut choice)
}

/// Generates an alternating entry/exit sequence from two choices.
///
/// Unlike [`crate::clean::clean_enex`], which *filters* arrays that already
/// exist, this one builds them: each side is asked where to fire next, given
/// where the other side last did. That is the difference between cleaning up
/// a signal and generating one, and it is why a stop-based exit belongs here
/// — the stop level depends on the entry that has not been chosen yet.
///
/// # Errors
///
/// As [`generate_exits`].
pub fn generate_alternating<E: ExitChoice, X: ExitChoice>(
    n_rows: usize,
    n_cols: usize,
    entry_wait: usize,
    exit_wait: usize,
    entry_choice: &mut E,
    exit_choice: &mut X,
) -> Result<(Vec<bool>, Vec<bool>)> {
    let mut entries = vec![false; n_rows * n_cols];
    let mut exits = vec![false; n_rows * n_cols];
    let mut chosen: Vec<usize> = Vec::new();

    for col in 0..n_cols {
        let base = col * n_rows;
        let mut from = 0_usize;
        // Entries go first: a column cannot exit what it has not entered.
        let mut want_entry = true;

        while from < n_rows {
            chosen.clear();
            if want_entry {
                entry_choice.choose(from, n_rows, col, &mut chosen);
            } else {
                exit_choice.choose(from, n_rows, col, &mut chosen);
            }
            let Some(&row) = chosen.first() else {
                break;
            };
            if row < from || row >= n_rows {
                return Err(Error::Invariant(format!(
                    "a choice returned row {row}, outside its window {from}..{n_rows}"
                )));
            }
            if want_entry {
                entries[base + row] = true;
                from = row + exit_wait;
            } else {
                exits[base + row] = true;
                from = row + entry_wait;
            }
            want_entry = !want_entry;
            // A wait of zero would let the same bar be chosen forever.
            if row + 1 > from {
                from = row + 1;
            }
        }
    }
    Ok((entries, exits))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(s: &str) -> Vec<bool> {
        s.chars().map(|c| c == 'X').collect()
    }

    fn render(v: &[bool]) -> String {
        v.iter().map(|b| if *b { 'X' } else { '.' }).collect()
    }

    #[test]
    fn an_exit_lands_a_fixed_number_of_bars_later() {
        let entries = parse("X....X....");
        let exits = exit_after(&entries, 10, 1, 3, ExitWindow::UntilNextEntry).unwrap();
        assert_eq!(render(&exits), "...X....X.");
    }

    #[test]
    fn a_window_that_runs_past_the_end_produces_no_exit() {
        let entries = parse("......X...");
        let exits = exit_after(&entries, 10, 1, 5, ExitWindow::UntilNextEntry).unwrap();
        assert_eq!(render(&exits), "..........", "bar 11 does not exist");
    }

    #[test]
    fn until_next_entry_gives_every_exit_one_owner() {
        // The second entry arrives before the first entry's exit would.
        let entries = parse("X.X.......");
        let exits = exit_after(&entries, 10, 1, 5, ExitWindow::UntilNextEntry).unwrap();
        // The first entry's window is cut short at bar 2 and finds nothing.
        assert_eq!(render(&exits), ".......X..");
    }

    #[test]
    fn to_end_lets_windows_overlap() {
        let entries = parse("X.X.......");
        let exits = exit_after(&entries, 10, 1, 5, ExitWindow::ToEnd).unwrap();
        // Both entries resolve: bar 5 and bar 7.
        assert_eq!(render(&exits), ".....X.X..");
    }

    #[test]
    fn skipping_resolved_drops_an_entry_inside_a_live_window() {
        let entries = parse("X.X.......");
        let exits = exit_after(&entries, 10, 1, 5, ExitWindow::ToEndSkippingResolved).unwrap();
        // The second entry sits before the first's exit at bar 5, so it is
        // skipped rather than opening a second overlapping position.
        assert_eq!(render(&exits), ".....X....");
    }

    #[test]
    fn a_condition_picks_the_first_bar_it_holds() {
        let entries = parse("X.........");
        let condition = parse("XX...X..X.");
        let exits = exit_on(&entries, &condition, 10, 1, 1, ExitWindow::UntilNextEntry).unwrap();
        // Bar 0 is inside the entry's own bar and excluded by the wait; bar 1
        // is the first in the window.
        assert_eq!(render(&exits), ".X........");
    }

    #[test]
    fn a_wait_of_zero_allows_a_same_bar_exit() {
        let entries = parse("X.........");
        let condition = parse("X.........");
        let exits = exit_on(&entries, &condition, 10, 1, 0, ExitWindow::UntilNextEntry).unwrap();
        assert_eq!(render(&exits), "X.........", "entry and exit on one bar");
    }

    #[test]
    fn a_condition_that_never_holds_produces_no_exit() {
        let entries = parse("X....X....");
        let condition = parse("..........");
        let exits = exit_on(&entries, &condition, 10, 1, 1, ExitWindow::UntilNextEntry).unwrap();
        assert_eq!(render(&exits), "..........");
    }

    #[test]
    fn a_choice_outside_its_window_is_rejected() {
        let entries = parse("X....");
        let mut rogue = |_from: usize, _to: usize, _col: usize, out: &mut Vec<usize>| {
            out.push(0); // before the window, which `wait` moved to bar 1
        };
        let err = generate_exits(
            &entries,
            5,
            1,
            1,
            ExitWindow::UntilNextEntry,
            true,
            &mut rogue,
        );
        assert!(err.is_err());
    }

    #[test]
    fn columns_are_generated_independently() {
        let entries = [true, false, false, false, true, false];
        let exits = exit_after(&entries, 3, 2, 1, ExitWindow::UntilNextEntry).unwrap();
        assert_eq!(exits, [false, true, false, false, false, true]);
    }

    #[test]
    fn alternating_never_stacks_two_of_a_kind() {
        // Enter on any even bar, exit on any odd one; the alternation has to
        // come from the loop, not from the choices.
        let mut entry = |from: usize, to: usize, _c: usize, out: &mut Vec<usize>| {
            if let Some(i) = (from..to).find(|i| i % 2 == 0) {
                out.push(i);
            }
        };
        let mut exit = |from: usize, to: usize, _c: usize, out: &mut Vec<usize>| {
            if let Some(i) = (from..to).find(|i| i % 2 == 1) {
                out.push(i);
            }
        };
        let (e, x) = generate_alternating(12, 1, 1, 1, &mut entry, &mut exit).unwrap();
        assert_eq!(render(&e), "X.X.X.X.X.X.");
        assert_eq!(render(&x), ".X.X.X.X.X.X");

        let mut expect_entry = true;
        for i in 0..12 {
            assert!(!(e[i] && x[i]));
            if e[i] {
                assert!(expect_entry);
                expect_entry = false;
            }
            if x[i] {
                assert!(!expect_entry);
                expect_entry = true;
            }
        }
    }

    #[test]
    fn alternating_stops_when_a_side_runs_out() {
        let mut entry = |from: usize, _to: usize, _c: usize, out: &mut Vec<usize>| out.push(from);
        // Never exits, so exactly one entry should be produced.
        let mut exit = |_f: usize, _t: usize, _c: usize, _o: &mut Vec<usize>| {};
        let (e, x) = generate_alternating(8, 1, 1, 1, &mut entry, &mut exit).unwrap();
        assert_eq!(render(&e), "X.......");
        assert_eq!(render(&x), "........");
    }
}
