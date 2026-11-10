//! Cleaning a raw signal array into one a simulation can use.
//!
//! A thresholded factor produces signals on *every* bar the condition holds,
//! not on the bar it starts holding. Fed to a simulation directly, that is a
//! column of redundant orders — mostly ignored, but the ones that are not
//! change the answer.
//!
//! [`clean_enex`] reduces a pair of arrays to an alternating sequence: one
//! entry, then one exit, then one entry. Nothing else here is allowed to
//! assume that shape, but everything downstream is easier when it holds.

/// Which side wins when a column has not traded yet and both arrays fire.
///
/// Only matters before the first signal. After that the alternation decides,
/// and this is ignored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FirstSignal {
    /// A column must enter before it can exit. The conservative choice: an
    /// exit with no position is not a trade, it is a rounding error waiting
    /// to happen in the accounting.
    EntryFirst,
    /// An exit may come first, which is what a short-biased column or a
    /// pre-existing position needs.
    Either,
}

/// Reduces one column's entries and exits to a strictly alternating sequence.
///
/// # A bar with both signals produces neither
///
/// Not an oversight: an array that says both "enter" and "exit" on one bar has
/// not said anything, and picking one silently would invent an opinion the
/// caller did not express. `from_signals` has five configurable conflict
/// modes for exactly this, and they belong there, where a position exists to
/// resolve against. Here there is nothing to be adjacent or opposite to.
#[must_use]
pub fn clean_enex_1d(
    entries: &[bool],
    exits: &[bool],
    first: FirstSignal,
) -> (Vec<bool>, Vec<bool>) {
    let n = entries.len().min(exits.len());
    let mut out_entries = vec![false; n];
    let mut out_exits = vec![false; n];

    // -1 before anything has fired, 1 while long, 0 while flat.
    let mut phase = -1_i8;
    for i in 0..n {
        if entries[i] && exits[i] {
            continue;
        }
        if entries[i] && (phase == -1 || phase == 0) {
            phase = 1;
            out_entries[i] = true;
        }
        if exits[i] && ((first == FirstSignal::Either && phase == -1) || phase == 1) {
            phase = 0;
            out_exits[i] = true;
        }
    }
    (out_entries, out_exits)
}

/// [`clean_enex_1d`] over a column-major pair of arrays.
///
/// # Errors
///
/// Returns [`alphax_core::Error::Broadcast`] if the two arrays disagree on shape.
pub fn clean_enex(
    entries: &[bool],
    exits: &[bool],
    n_rows: usize,
    n_cols: usize,
    first: FirstSignal,
) -> alphax_core::error::Result<(Vec<bool>, Vec<bool>)> {
    if entries.len() != n_rows * n_cols || exits.len() != n_rows * n_cols {
        return Err(alphax_core::Error::Broadcast {
            left: (entries.len(), 1),
            right: (n_rows, n_cols),
        });
    }
    let mut out_entries = vec![false; n_rows * n_cols];
    let mut out_exits = vec![false; n_rows * n_cols];
    for col in 0..n_cols {
        let range = col * n_rows..(col + 1) * n_rows;
        let (e, x) = clean_enex_1d(&entries[range.clone()], &exits[range.clone()], first);
        out_entries[range.clone()].copy_from_slice(&e);
        out_exits[range].copy_from_slice(&x);
    }
    Ok((out_entries, out_exits))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `.` is false, `X` is true.
    fn parse(s: &str) -> Vec<bool> {
        s.chars().map(|c| c == 'X').collect()
    }

    fn render(v: &[bool]) -> String {
        v.iter().map(|b| if *b { 'X' } else { '.' }).collect()
    }

    #[test]
    fn a_run_of_signals_collapses_to_its_first() {
        let (e, x) = clean_enex_1d(
            &parse("XXX...XXX."),
            &parse("...XXX...X"),
            FirstSignal::EntryFirst,
        );
        assert_eq!(render(&e), "X.....X...");
        assert_eq!(render(&x), "...X.....X");
    }

    #[test]
    fn a_bar_with_both_signals_produces_neither() {
        // Not a dropped case: the array has not expressed an opinion, and
        // inventing one here would pre-empt `from_signals`' conflict modes.
        let (e, x) = clean_enex_1d(&parse("X.X"), &parse("X.."), FirstSignal::EntryFirst);
        assert_eq!(render(&e), "..X");
        assert_eq!(render(&x), "...");
    }

    #[test]
    fn entry_first_drops_a_leading_exit() {
        let (e, x) = clean_enex_1d(&parse(".X.."), &parse("X..X"), FirstSignal::EntryFirst);
        assert_eq!(render(&e), ".X..");
        assert_eq!(render(&x), "...X");
    }

    #[test]
    fn either_lets_the_exit_come_first() {
        let (e, x) = clean_enex_1d(&parse(".X.."), &parse("X..X"), FirstSignal::Either);
        assert_eq!(render(&e), ".X..");
        assert_eq!(render(&x), "X..X");
    }

    #[test]
    fn the_output_strictly_alternates() {
        let (e, x) = clean_enex_1d(
            &parse("XX.X.XX.X."),
            &parse("..X..X..XX"),
            FirstSignal::EntryFirst,
        );
        // Walking the two together must never see two of a kind in a row.
        let mut expect_entry = true;
        for i in 0..e.len() {
            assert!(!(e[i] && x[i]), "bar {i} carries both");
            if e[i] {
                assert!(expect_entry, "a second entry at bar {i}");
                expect_entry = false;
            }
            if x[i] {
                assert!(!expect_entry, "an exit with no entry at bar {i}");
                expect_entry = true;
            }
        }
    }

    #[test]
    fn nothing_in_nothing_out() {
        let (e, x) = clean_enex_1d(&parse("...."), &parse("...."), FirstSignal::EntryFirst);
        assert_eq!(render(&e), "....");
        assert_eq!(render(&x), "....");
    }

    #[test]
    fn columns_are_cleaned_independently() {
        // Two four-bar columns, laid out column-major.
        //   col 0: entries X.X.  exits .X.X   -> already alternating
        //   col 1: entries ..X.  exits X..X   -> the leading exit is dropped
        let mut entries = parse("X.X.");
        entries.extend(parse("..X."));
        let mut exits = parse(".X.X");
        exits.extend(parse("X..X"));

        let (e, x) = clean_enex(&entries, &exits, 4, 2, FirstSignal::EntryFirst).unwrap();
        assert_eq!(render(&e[0..4]), "X.X.");
        assert_eq!(render(&x[0..4]), ".X.X");
        assert_eq!(render(&e[4..8]), "..X.");
        assert_eq!(render(&x[4..8]), "...X", "the exit at bar 0 had no entry");
    }

    #[test]
    fn a_shape_mismatch_is_rejected() {
        assert!(clean_enex(&[true], &[true, false], 2, 1, FirstSignal::EntryFirst).is_err());
    }
}
