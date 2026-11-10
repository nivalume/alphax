//! Signal arrays as ranges.
//!
//! A boolean column answers "was the signal on at bar `i`". The questions
//! people actually ask are about *spans*: how long did a run last, how long
//! between one signal and the next, how many bars was the rule quiet. Those
//! are ranges, and a range is a record — the same shape the order and trade
//! layers use, so the same reductions apply.
//!
//! Three readings of the same array, and they answer different questions:
//!
//! | | one range is |
//! |---|---|
//! | [`partition_ranges`] | a run of `true`, start to end |
//! | [`between_ranges`] | the gap from one `true` to the next |
//! | [`between_partition_ranges`] | the gap from the end of one run to the start of the next |

use alphax_core::error::{Error, Result};

/// Whether a range finished inside the sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i8)]
pub enum RangeStatus {
    /// Still running at the last bar, so its duration is a lower bound.
    Open = 0,
    /// Ended within the sample.
    Closed = 1,
}

/// Ranges, struct-of-arrays, matching the record layout used elsewhere.
///
/// `id` is not stored: it is the record's index, as for orders and trades.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RangeRecords {
    /// Column the range belongs to.
    pub col: Vec<i64>,
    /// First bar of the range.
    pub start_idx: Vec<i64>,
    /// Last bar of the range.
    pub end_idx: Vec<i64>,
    /// Whether it closed, as [`RangeStatus`]'s discriminant.
    pub status: Vec<i8>,
}

impl RangeRecords {
    /// An empty set.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            col: Vec::new(),
            start_idx: Vec::new(),
            end_idx: Vec::new(),
            status: Vec::new(),
        }
    }

    fn push(&mut self, col: usize, start: usize, end: usize, status: RangeStatus) {
        self.col.push(col as i64);
        self.start_idx.push(start as i64);
        self.end_idx.push(end as i64);
        self.status.push(status as i8);
    }

    /// Number of ranges.
    #[must_use]
    pub fn len(&self) -> usize {
        self.col.len()
    }

    /// Whether there are none.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.col.is_empty()
    }

    /// Duration of each range, in bars.
    ///
    /// **A closed range is measured exclusively and an open one inclusively**:
    /// `end - start` against `end - start + 1`. A closed range's end index is
    /// the bar that ended it, which belongs to what came after; an open one
    /// has no such bar, and its last index is one the range still covered.
    /// The same rule governs trade and drawdown durations, and using one
    /// convention for both shifts every average by up to a bar.
    ///
    /// An open range's duration is still a lower bound, because the range had
    /// not been observed to end. Mixing open and closed durations in one
    /// average is the standard way to understate how long a rule stays on;
    /// [`Self::mean_closed_duration`] is the one that does not.
    #[must_use]
    pub fn duration(&self) -> Vec<i64> {
        (0..self.len())
            .map(|i| {
                let span = self.end_idx[i] - self.start_idx[i];
                let open = self.status[i] == RangeStatus::Open as i8;
                span + i64::from(open)
            })
            .collect()
    }

    /// Mean duration of the closed ranges only, or `NaN` if there are none.
    #[must_use]
    pub fn mean_closed_duration(&self) -> f64 {
        let all = self.duration();
        let closed: Vec<i64> = (0..self.len())
            .filter(|&i| self.status[i] == RangeStatus::Closed as i8)
            .map(|i| all[i])
            .collect();
        if closed.is_empty() {
            return f64::NAN;
        }
        closed.iter().sum::<i64>() as f64 / closed.len() as f64
    }
}

fn check_shape(a: &[bool], n_rows: usize, n_cols: usize) -> Result<()> {
    if a.len() == n_rows * n_cols {
        Ok(())
    } else {
        Err(Error::Broadcast {
            left: (a.len(), 1),
            right: (n_rows, n_cols),
        })
    }
}

/// One range per run of `true`.
///
/// A run still going at the last bar is [`RangeStatus::Open`].
///
/// # Errors
///
/// Returns [`Error::Broadcast`] if `a` is not `n_rows * n_cols` long.
pub fn partition_ranges(a: &[bool], n_rows: usize, n_cols: usize) -> Result<RangeRecords> {
    check_shape(a, n_rows, n_cols)?;
    let mut out = RangeRecords::new();
    for col in 0..n_cols {
        let base = col * n_rows;
        let mut start = 0_usize;
        let mut in_run = false;
        for i in 0..n_rows {
            if a[base + i] {
                if !in_run {
                    start = i;
                }
                in_run = true;
            } else if in_run {
                out.push(col, start, i, RangeStatus::Closed);
                in_run = false;
            }
            if i == n_rows - 1 && in_run {
                // The end index is the last bar itself, not one past it: the
                // range has not been observed to end.
                out.push(col, start, i, RangeStatus::Open);
            }
        }
    }
    Ok(out)
}

/// One range per gap between consecutive `true` values.
///
/// Every range is closed by construction — it is bounded by two signals — so
/// a column with one signal produces none.
///
/// # Errors
///
/// As [`partition_ranges`].
pub fn between_ranges(a: &[bool], n_rows: usize, n_cols: usize) -> Result<RangeRecords> {
    check_shape(a, n_rows, n_cols)?;
    let mut out = RangeRecords::new();
    for col in 0..n_cols {
        let base = col * n_rows;
        let rows: Vec<usize> = (0..n_rows).filter(|&i| a[base + i]).collect();
        for pair in rows.windows(2) {
            out.push(col, pair[0], pair[1], RangeStatus::Closed);
        }
    }
    Ok(out)
}

/// One range per gap between the end of one run and the start of the next.
///
/// The difference from [`between_ranges`] is what a run counts as. Over
/// `XX..XX`, `between_ranges` sees three short gaps inside and between the
/// runs; this sees one gap, from bar 1 to bar 4.
///
/// # Errors
///
/// As [`partition_ranges`].
pub fn between_partition_ranges(a: &[bool], n_rows: usize, n_cols: usize) -> Result<RangeRecords> {
    check_shape(a, n_rows, n_cols)?;
    let mut out = RangeRecords::new();
    for col in 0..n_cols {
        let base = col * n_rows;
        let mut in_run = false;
        let mut last_start: i64 = -1;
        for i in 0..n_rows {
            if a[base + i] {
                if !in_run && last_start != -1 {
                    out.push(col, last_start as usize, i, RangeStatus::Closed);
                }
                in_run = true;
                last_start = i as i64;
            } else {
                in_run = false;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "mean durations of small integer sets are exact"
)]
mod tests {
    use super::*;

    fn parse(s: &str) -> Vec<bool> {
        s.chars().map(|c| c == 'X').collect()
    }

    #[test]
    fn a_run_becomes_one_range() {
        let r = partition_ranges(&parse("XX..XXX..."), 10, 1).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r.start_idx, [0, 4]);
        assert_eq!(r.end_idx, [2, 7]);
        assert!(r.status.iter().all(|s| *s == RangeStatus::Closed as i8));
    }

    #[test]
    fn a_run_reaching_the_last_bar_is_open() {
        let r = partition_ranges(&parse("..XXX"), 5, 1).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r.start_idx, [2]);
        assert_eq!(r.end_idx, [4], "the last bar, not one past it");
        assert_eq!(r.status, [RangeStatus::Open as i8]);
    }

    #[test]
    fn between_ranges_bridges_consecutive_signals() {
        let r = between_ranges(&parse("X..X.X"), 6, 1).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r.start_idx, [0, 3]);
        assert_eq!(r.end_idx, [3, 5]);
    }

    #[test]
    fn one_signal_has_nothing_to_be_between() {
        assert!(between_ranges(&parse("..X.."), 5, 1).unwrap().is_empty());
        assert!(between_ranges(&parse("....."), 5, 1).unwrap().is_empty());
    }

    #[test]
    fn the_two_between_readings_differ_on_runs() {
        let a = parse("XX..XX");
        // Signal to signal: 0->1, 1->4, 4->5.
        let signals = between_ranges(&a, 6, 1).unwrap();
        assert_eq!(signals.len(), 3);
        // Run to run: one gap, from the end of the first run to the start of
        // the second.
        let runs = between_partition_ranges(&a, 6, 1).unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs.start_idx, [1]);
        assert_eq!(runs.end_idx, [4]);
    }

    #[test]
    fn duration_measures_the_span() {
        let r = partition_ranges(&parse("XX..XXX..."), 10, 1).unwrap();
        assert_eq!(r.duration(), [2, 3]);
    }

    #[test]
    fn an_open_range_is_left_out_of_the_mean() {
        // Two closed runs of 2 bars and one open one that has only been seen
        // for 1; averaging all three would understate how long the rule stays
        // on.
        let r = partition_ranges(&parse("XX.XX.X"), 7, 1).unwrap();
        assert_eq!(r.len(), 3);
        assert_eq!(r.status[2], RangeStatus::Open as i8);
        assert_eq!(r.mean_closed_duration(), 2.0);
    }

    #[test]
    fn no_closed_ranges_means_no_mean() {
        let r = partition_ranges(&parse("XXX"), 3, 1).unwrap();
        assert!(r.mean_closed_duration().is_nan());
    }

    #[test]
    fn columns_are_kept_apart() {
        let a = [true, true, false, true];
        let r = partition_ranges(&a, 2, 2).unwrap();
        assert_eq!(r.col, [0, 1]);
    }

    #[test]
    fn a_shape_mismatch_is_rejected() {
        assert!(partition_ranges(&[true], 2, 2).is_err());
        assert!(between_ranges(&[true], 2, 2).is_err());
        assert!(between_partition_ranges(&[true], 2, 2).is_err());
    }
}
