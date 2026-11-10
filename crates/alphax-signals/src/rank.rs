//! Ranking signals within their runs.
//!
//! "Enter on the third consecutive signal" and "skip the first two signals
//! after each reset" both need to know *where in a run* a signal sits. That
//! is what these produce: an integer per `true`, and `-1` everywhere else.
//!
//! Two different questions, and they are easy to confuse:
//!
//! - [`signal_rank`] counts **signals**: the first `true` is 0, the next is
//!   1, however far apart they are.
//! - [`partition_rank`] counts **runs**: every `true` in the first run is 0,
//!   every one in the second run is 1.
//!
//! Both take an optional reset array, so a rank can restart on a regime
//! change rather than only at the start of the column.

/// How a signal rank treats a gap between runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gaps {
    /// Keep counting across gaps: the run structure is ignored.
    Allow,
    /// Restart the count at the beginning of each run.
    ResetEachRun,
}

/// What to do about a run that is already in progress at row zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeadingRun {
    /// Rank it like any other.
    Rank,
    /// Skip it, and start ranking only after the first `false`.
    ///
    /// A column that begins mid-run has no way to know how far into that run
    /// it already is, so a rank on it is a guess. This discards it.
    Skip,
}

/// The shared walk: both ranks differ only in what they do at a boundary.
fn rank_columns(
    a: &[bool],
    reset_by: Option<&[bool]>,
    n_rows: usize,
    n_cols: usize,
    leading: LeadingRun,
    mut step: impl FnMut(&mut i64, usize, usize, i64, usize),
) -> Vec<i64> {
    let mut out = vec![-1_i64; n_rows * n_cols];
    for col in 0..n_cols {
        let base = col * n_rows;
        let mut counter = 0_i64;
        let mut reset_row = 0_usize;
        let mut prev_run_end: i64 = -1;
        let mut run_start = 0_usize;
        let mut in_run = false;
        let mut seen_false = leading == LeadingRun::Rank;

        for i in 0..n_rows {
            if let Some(reset) = reset_by
                && reset[base + i]
            {
                reset_row = i;
            }
            if a[base + i] && seen_false {
                if !in_run {
                    run_start = i;
                }
                in_run = true;
                step(&mut counter, i, reset_row, prev_run_end, run_start);
                out[base + i] = counter;
            } else if !a[base + i] {
                if in_run {
                    prev_run_end = i as i64 - 1;
                }
                in_run = false;
                seen_false = true;
            }
        }
    }
    out
}

/// Rank of each signal within its run, counting signals.
///
/// The first signal is 0. With [`Gaps::ResetEachRun`] the count restarts at
/// each run; with [`Gaps::Allow`] it keeps going.
#[must_use]
pub fn signal_rank(
    a: &[bool],
    reset_by: Option<&[bool]>,
    n_rows: usize,
    n_cols: usize,
    gaps: Gaps,
    leading: LeadingRun,
) -> Vec<i64> {
    rank_columns(
        a,
        reset_by,
        n_rows,
        n_cols,
        leading,
        move |counter, i, reset_row, prev_run_end, run_start| {
            // The count restarts at whichever comes later, the reset or the
            // run's start — so a reset lands mid-run and renumbers from
            // there rather than waiting for the run to finish. The
            // `> prev_run_end` guard is what stops a *stale* reset, one from
            // before the previous run ended, firing again in this one.
            let reset_here = reset_row as i64 > prev_run_end && reset_row.max(run_start) == i;
            let run_starts_here = gaps == Gaps::ResetEachRun && run_start == i;
            if reset_here || run_starts_here {
                *counter = -1;
            }
            *counter += 1;
        },
    )
}

/// Rank of each signal's *run*, counting runs.
///
/// Every signal in the first run is 0, every signal in the second is 1.
#[must_use]
pub fn partition_rank(
    a: &[bool],
    reset_by: Option<&[bool]>,
    n_rows: usize,
    n_cols: usize,
    leading: LeadingRun,
) -> Vec<i64> {
    rank_columns(
        a,
        reset_by,
        n_rows,
        n_cols,
        leading,
        |counter, i, reset_row, prev_run_end, run_start| {
            if reset_row as i64 > prev_run_end && reset_row.max(run_start) == i {
                *counter = 0;
            } else if run_start == i {
                *counter += 1;
            }
        },
    )
}

/// The row of the `n`th signal in each column, or `-1` if there is none.
///
/// A negative `n` counts from the end: `-1` is the last signal.
#[must_use]
pub fn nth_index(a: &[bool], n_rows: usize, n_cols: usize, n: i64) -> Vec<i64> {
    (0..n_cols)
        .map(|col| {
            let base = col * n_rows;
            if n >= 0 {
                let mut found = -1_i64;
                for i in 0..n_rows {
                    if a[base + i] {
                        found += 1;
                        if found == n {
                            return i as i64;
                        }
                    }
                }
            } else {
                let mut found = 0_i64;
                for i in (0..n_rows).rev() {
                    if a[base + i] {
                        found -= 1;
                        if found == n {
                            return i as i64;
                        }
                    }
                }
            }
            -1
        })
        .collect()
}

/// Where a column's signals sit on average, rescaled to `[-1, 1]`.
///
/// `-1` means every signal is at the very start, `+1` the very end, `0`
/// evenly spread or centred. Useful for spotting a rule whose signals all
/// arrive in one part of the sample — which is usually a look-ahead bug or an
/// indicator warm-up, not a discovery.
///
/// `NaN` for a column with no signals, or one row.
#[must_use]
pub fn normalized_average_index(a: &[bool], n_rows: usize, n_cols: usize) -> Vec<f64> {
    (0..n_cols)
        .map(|col| {
            let base = col * n_rows;
            let rows: Vec<usize> = (0..n_rows).filter(|&i| a[base + i]).collect();
            // The `n_rows <= 1` half is redundant and kept as part of
            // the definition: with one row the rescale divides by zero and
            // already yields `NaN`. Mutation testing confirms removing it
            // changes nothing.
            if n_rows <= 1 || rows.is_empty() {
                return f64::NAN;
            }
            let mean = rows.iter().sum::<usize>() as f64 / rows.len() as f64;
            // Rescale [0, n-1] onto [-1, 1].
            mean / (n_rows as f64 - 1.0) * 2.0 - 1.0
        })
        .collect()
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "the normalized index has exact expected values at the endpoints"
)]
mod tests {
    use super::*;

    fn parse(s: &str) -> Vec<bool> {
        s.chars().map(|c| c == 'X').collect()
    }

    #[test]
    fn signal_rank_counts_signals_across_gaps() {
        let a = parse("XX.X.X");
        let r = signal_rank(&a, None, 6, 1, Gaps::Allow, LeadingRun::Rank);
        assert_eq!(r, [0, 1, -1, 2, -1, 3]);
    }

    #[test]
    fn resetting_each_run_restarts_the_count() {
        let a = parse("XX.XXX");
        let r = signal_rank(&a, None, 6, 1, Gaps::ResetEachRun, LeadingRun::Rank);
        assert_eq!(r, [0, 1, -1, 0, 1, 2]);
    }

    #[test]
    fn partition_rank_counts_runs_not_signals() {
        let a = parse("XX.XXX.X");
        let r = partition_rank(&a, None, 8, 1, LeadingRun::Rank);
        assert_eq!(r, [0, 0, -1, 1, 1, 1, -1, 2]);
    }

    #[test]
    fn skipping_a_leading_run_discards_it() {
        // The column starts mid-run, so there is no way to know how far in.
        let a = parse("XX.XX");
        let ranked = signal_rank(&a, None, 5, 1, Gaps::Allow, LeadingRun::Rank);
        let skipped = signal_rank(&a, None, 5, 1, Gaps::Allow, LeadingRun::Skip);
        assert_eq!(ranked, [0, 1, -1, 2, 3]);
        assert_eq!(skipped, [-1, -1, -1, 0, 1]);
    }

    #[test]
    fn a_reset_between_runs_restarts_the_count() {
        let a = parse("XX.XX.XX");
        let reset = parse("......X.");
        let r = signal_rank(&a, Some(&reset), 8, 1, Gaps::Allow, LeadingRun::Rank);
        assert_eq!(r, [0, 1, -1, 2, 3, -1, 0, 1]);
    }

    #[test]
    fn a_reset_inside_a_run_renumbers_from_that_bar() {
        // Asserted because it reads the other way at a glance: a reset does
        // not wait for the run to finish, it restarts the count where it
        // lands. Bar 1 becomes the new zero even though the run continues.
        let a = parse("XXXX");
        let reset = parse(".X..");
        let r = signal_rank(&a, Some(&reset), 4, 1, Gaps::Allow, LeadingRun::Rank);
        assert_eq!(r, [0, 0, 1, 2]);
    }

    #[test]
    fn nth_index_counts_from_either_end() {
        let a = parse("X.X..X");
        assert_eq!(nth_index(&a, 6, 1, 0), [0]);
        assert_eq!(nth_index(&a, 6, 1, 2), [5]);
        assert_eq!(nth_index(&a, 6, 1, -1), [5]);
        assert_eq!(nth_index(&a, 6, 1, -3), [0]);
        assert_eq!(nth_index(&a, 6, 1, 9), [-1], "past the end");
    }

    #[test]
    fn a_column_with_no_signals_has_no_nth_index() {
        assert_eq!(nth_index(&parse("...."), 4, 1, 0), [-1]);
    }

    #[test]
    fn the_normalized_index_marks_where_signals_cluster() {
        assert_eq!(normalized_average_index(&parse("X...."), 5, 1), [-1.0]);
        assert_eq!(normalized_average_index(&parse("....X"), 5, 1), [1.0]);
        assert_eq!(normalized_average_index(&parse("..X.."), 5, 1), [0.0]);
        assert_eq!(normalized_average_index(&parse("X...X"), 5, 1), [0.0]);
    }

    #[test]
    fn a_column_with_no_signals_has_no_average_index() {
        assert!(normalized_average_index(&parse("...."), 4, 1)[0].is_nan());
        assert!(normalized_average_index(&parse("X"), 1, 1)[0].is_nan());
    }

    #[test]
    fn columns_are_ranked_independently() {
        let a = [true, true, false, true];
        let r = signal_rank(&a, None, 2, 2, Gaps::Allow, LeadingRun::Rank);
        assert_eq!(r, [0, 1, -1, 0]);
    }
}
