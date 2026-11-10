//! Drawdowns as records, not as a single number.
//!
//! `alphax_analysis::metrics::max_drawdown` answers "how bad did it get". That
//! is one number out of a series that contained a dozen separate episodes, and
//! it says nothing about how long any of them lasted or whether the strategy
//! ever came back. A record per episode does: each one carries its peak, its
//! valley, where it ended and whether it recovered, so a column's drawdown
//! history reduces through the ordinary [`crate::mapped`] machinery like any
//! other record set.
//!
//! # Four indices, and only three of them are obvious
//!
//! ```text
//!        peak
//!         │ start = peak + 1
//!         ▼ ▼
//!   ──────●───────────────────●──────
//!          ╲                 ╱
//!           ╲               ╱
//!            ╲─────●───────╱
//!                  ▲       ▲
//!               valley    end
//! ```
//!
//! `peak_idx` is the last bar *before* the decline, and `start_idx` is the
//! first bar of the decline itself — always `peak_idx + 1`, stored anyway
//! because the two measure different things. Duration from the peak includes
//! the flat bar at the top; duration from the start does not, and that is the
//! one the decline phase is measured over.
//!
//! # A drawdown that never recovered is still a record
//!
//! The last episode in a column usually has not recovered, and it is marked
//! [`DrawdownStatus::Active`] with `end_idx` at the last bar reached. Its
//! recovery duration is therefore a lower bound and its recovery return is
//! whatever it had clawed back so far. Averaging active and recovered
//! episodes together is the standard way to understate how long recovery
//! takes — which is why the status is a field rather than a detail.

use alphax_core::error::{Error, Result};

use crate::mapped::ColMap;

/// Whether a drawdown came back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i8)]
pub enum DrawdownStatus {
    /// Still under water at the last bar.
    Active = 0,
    /// Regained the peak within the sample.
    Recovered = 1,
}

/// Drawdown episodes, struct-of-arrays.
///
/// `id` is not stored: it is the record's index, as for every other record
/// type here.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DrawdownRecords {
    /// Column the episode belongs to.
    pub col: Vec<i64>,
    /// Last bar at the peak, before the decline began.
    pub peak_idx: Vec<i64>,
    /// First bar of the decline: always `peak_idx + 1`.
    pub start_idx: Vec<i64>,
    /// Bar of the lowest value reached.
    pub valley_idx: Vec<i64>,
    /// Bar the episode ended on — recovery, or the last bar if still active.
    pub end_idx: Vec<i64>,
    /// Value at the peak.
    pub peak_val: Vec<f64>,
    /// Value at the valley.
    pub valley_val: Vec<f64>,
    /// Value at the end.
    pub end_val: Vec<f64>,
    /// Recovered or still active, as [`DrawdownStatus`]'s discriminant.
    pub status: Vec<i8>,
}

impl DrawdownRecords {
    /// An empty record set.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            col: Vec::new(),
            peak_idx: Vec::new(),
            start_idx: Vec::new(),
            valley_idx: Vec::new(),
            end_idx: Vec::new(),
            peak_val: Vec::new(),
            valley_val: Vec::new(),
            end_val: Vec::new(),
            status: Vec::new(),
        }
    }

    /// Number of episodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.col.len()
    }

    /// Whether there were none.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.col.is_empty()
    }

    /// The `id` of each record: its index.
    #[must_use]
    pub fn ids(&self) -> Vec<i64> {
        (0..self.len() as i64).collect()
    }

    /// Groups these episodes by column.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invariant`] if an episode names a column outside
    /// `n_cols`.
    pub fn col_map(&self, n_cols: usize) -> Result<ColMap> {
        ColMap::build(&self.col, n_cols)
    }

    /// Bars each episode spanned, from the start of the decline.
    ///
    /// **A recovered episode is measured exclusively and an active one
    /// inclusively**: `end - start` against `end - start + 1`. Same rule as
    /// [`crate::TradeRecords::duration`] and for the same reason — a recovered
    /// episode's end bar is the one it was already back above water on, while
    /// an active one's last bar is still under it.
    ///
    /// Note this is the *whole* episode, unlike [`Self::decline_duration`] and
    /// [`Self::recovery_duration`], which split it in two and use their own
    /// inclusive/exclusive conventions.
    #[must_use]
    pub fn duration(&self) -> Vec<f64> {
        (0..self.len())
            .map(|i| {
                let span = self.end_idx[i] - self.start_idx[i];
                let active = self.status[i] == DrawdownStatus::Active as i8;
                (span + i64::from(active)) as f64
            })
            .collect()
    }

    /// Depth of each episode, as a negative fraction of the peak.
    #[must_use]
    pub fn drawdown(&self) -> Vec<f64> {
        (0..self.len())
            .map(|i| (self.valley_val[i] - self.peak_val[i]) / self.peak_val[i])
            .collect()
    }

    /// Bars from the start of the decline to the valley, inclusive.
    ///
    /// Measured from `start_idx`, not `peak_idx`, so the flat bar at the top
    /// is not counted as part of the decline.
    #[must_use]
    pub fn decline_duration(&self) -> Vec<f64> {
        (0..self.len())
            .map(|i| (self.valley_idx[i] - self.start_idx[i] + 1) as f64)
            .collect()
    }

    /// Bars from the valley to the end of the episode.
    ///
    /// Not inclusive, unlike [`Self::decline_duration`] — the valley bar
    /// belongs to the decline. For an active episode this is a lower bound.
    #[must_use]
    pub fn recovery_duration(&self) -> Vec<f64> {
        (0..self.len())
            .map(|i| (self.end_idx[i] - self.valley_idx[i]) as f64)
            .collect()
    }

    /// Recovery duration over decline duration.
    ///
    /// Above 1 means the climb back took longer than the fall, which is the
    /// usual case and the reason a drawdown costs more than its depth.
    #[must_use]
    pub fn recovery_duration_ratio(&self) -> Vec<f64> {
        let decline = self.decline_duration();
        let recovery = self.recovery_duration();
        (0..self.len()).map(|i| recovery[i] / decline[i]).collect()
    }

    /// Gain from the valley to the end, as a fraction of the valley.
    ///
    /// For a recovered episode this is what it took to get back; for an
    /// active one, what has been clawed back so far.
    #[must_use]
    pub fn recovery_return(&self) -> Vec<f64> {
        (0..self.len())
            .map(|i| (self.end_val[i] - self.valley_val[i]) / self.valley_val[i])
            .collect()
    }

    /// Whether each episode recovered.
    #[must_use]
    pub fn recovered(&self) -> Vec<bool> {
        self.status
            .iter()
            .map(|s| *s == DrawdownStatus::Recovered as i8)
            .collect()
    }

    #[allow(clippy::too_many_arguments, reason = "one argument per record field")]
    fn push(
        &mut self,
        col: usize,
        peak_idx: i64,
        valley_idx: i64,
        end_idx: usize,
        peak_val: f64,
        valley_val: f64,
        end_val: f64,
        status: DrawdownStatus,
    ) {
        self.col.push(col as i64);
        self.peak_idx.push(peak_idx);
        self.start_idx.push(peak_idx + 1);
        self.valley_idx.push(valley_idx);
        self.end_idx.push(end_idx as i64);
        self.peak_val.push(peak_val);
        self.valley_val.push(valley_val);
        self.end_val.push(end_val);
        self.status.push(status as i8);
    }
}

/// Finds every drawdown episode in a column-major series.
///
/// The input is any series worth measuring drawdowns on — an equity curve, a
/// price, a cumulative return. Records come out in column-major order, which
/// is the ordering contract the whole record layer rests on.
///
/// # Missing bars are skipped, not treated as a decline
///
/// A `NaN` bar advances nothing: the peak, the valley and the running state
/// all carry through it unchanged, and it cannot end an episode. A gap in the
/// data is not evidence about the strategy.
///
/// # Errors
///
/// Returns [`Error::Broadcast`] if `ts` is not `n_rows * n_cols` long.
pub fn find_drawdowns(ts: &[f64], n_rows: usize, n_cols: usize) -> Result<DrawdownRecords> {
    if ts.len() != n_rows * n_cols {
        return Err(Error::Broadcast {
            left: (ts.len(), 1),
            right: (n_rows, n_cols),
        });
    }
    let mut out = DrawdownRecords::new();
    for col in 0..n_cols {
        let base = col * n_rows;
        let mut started = false;
        let mut peak_idx: i64 = -1;
        let mut valley_idx: i64 = -1;
        let mut peak_val = if n_rows == 0 { f64::NAN } else { ts[base] };
        let mut valley_val = peak_val;

        for i in 0..n_rows {
            let cur = ts[base + i];
            if cur.is_nan() {
                continue;
            }
            let mut store: Option<DrawdownStatus> = None;

            // A bar equal to the peak counts as a rise, so a flat stretch at
            // the top extends the peak rather than opening an episode — and,
            // once under water, regaining exactly the old peak *is* recovery.
            if peak_val.is_nan() || cur >= peak_val {
                if started {
                    // This nested test is **redundant**, and the reasoning
                    // that it is not looks convincing: the outer test also
                    // passes when `peak_val` is `NaN`, which would leave the
                    // episode open. But `started` can only have been set in
                    // the `else` branch below, which is reached only when the
                    // outer test *failed* — and that requires `peak_val` to be
                    // a number. So `peak_val` is never `NaN` here, and the
                    // outer test has already established `cur >= peak_val`.
                    // Mutation testing confirms removing it changes nothing.
                    // Kept as part of the definition.
                    if cur >= peak_val {
                        started = false;
                        store = Some(DrawdownStatus::Recovered);
                    }
                } else {
                    peak_val = cur;
                    peak_idx = i as i64;
                }
            } else if started {
                if cur < valley_val {
                    valley_val = cur;
                    valley_idx = i as i64;
                }
            } else {
                started = true;
                valley_val = cur;
                valley_idx = i as i64;
            }

            if i == n_rows - 1 && started {
                started = false;
                store = Some(DrawdownStatus::Active);
            }

            if let Some(status) = store {
                out.push(
                    col, peak_idx, valley_idx, i, peak_val, valley_val, cur, status,
                );
                // The bar that closed one episode is the peak of the next, so
                // a V-shaped recovery straight into another decline produces
                // two records rather than one long one.
                peak_idx = i as i64;
                valley_idx = i as i64;
                peak_val = cur;
                valley_val = cur;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "drawdowns of small exact inputs have exact expected values"
)]
mod tests {
    use super::*;

    #[test]
    fn a_decline_and_a_recovery_is_one_episode() {
        //          0    1    2    3    4
        let ts = [10.0, 8.0, 6.0, 9.0, 11.0];
        let d = find_drawdowns(&ts, 5, 1).unwrap();
        assert_eq!(d.len(), 1);
        assert_eq!(d.peak_idx, [0]);
        assert_eq!(d.start_idx, [1], "the decline starts after the peak");
        assert_eq!(d.valley_idx, [2]);
        assert_eq!(d.end_idx, [4]);
        assert_eq!(d.status, [DrawdownStatus::Recovered as i8]);
        assert_eq!(d.drawdown(), [-0.4]);
    }

    #[test]
    fn a_decline_that_never_came_back_is_active() {
        let ts = [10.0, 8.0, 6.0, 7.0];
        let d = find_drawdowns(&ts, 4, 1).unwrap();
        assert_eq!(d.len(), 1);
        assert_eq!(d.status, [DrawdownStatus::Active as i8]);
        assert_eq!(d.end_idx, [3], "the last bar reached, not a recovery");
        // So its recovery duration is a lower bound, not a measurement.
        assert_eq!(d.recovery_duration(), [1.0]);
    }

    #[test]
    fn a_series_that_only_rises_has_no_drawdowns() {
        assert!(find_drawdowns(&[1.0, 2.0, 3.0], 3, 1).unwrap().is_empty());
        // And a flat one does not either: an equal bar counts as a rise.
        assert!(find_drawdowns(&[5.0, 5.0, 5.0], 3, 1).unwrap().is_empty());
    }

    #[test]
    fn regaining_exactly_the_old_peak_counts_as_recovery() {
        let d = find_drawdowns(&[10.0, 8.0, 10.0], 3, 1).unwrap();
        assert_eq!(d.status, [DrawdownStatus::Recovered as i8]);
        assert_eq!(d.recovery_return(), [0.25]);
    }

    #[test]
    fn a_recovery_straight_into_another_decline_makes_two_records() {
        // The bar that closed the first episode is the peak of the second,
        // so this is two drawdowns and not one long one.
        let ts = [10.0, 8.0, 10.0, 7.0, 6.0];
        let d = find_drawdowns(&ts, 5, 1).unwrap();
        assert_eq!(d.len(), 2);
        assert_eq!(d.peak_idx, [0, 2]);
        assert_eq!(d.status[1], DrawdownStatus::Active as i8);
        assert_eq!(d.drawdown()[1], -0.4);
    }

    #[test]
    fn the_valley_is_the_lowest_bar_not_the_last_one() {
        let ts = [10.0, 6.0, 7.0, 8.0, 11.0];
        let d = find_drawdowns(&ts, 5, 1).unwrap();
        assert_eq!(d.valley_idx, [1]);
        assert_eq!(d.valley_val, [6.0]);
    }

    #[test]
    fn a_missing_bar_is_not_evidence_about_the_strategy() {
        // The gap advances nothing: the episode is the same as without it,
        // shifted only by the bars it occupies.
        let with_gap = find_drawdowns(&[10.0, f64::NAN, 8.0, 6.0, 11.0], 5, 1).unwrap();
        assert_eq!(with_gap.len(), 1);
        assert_eq!(with_gap.peak_idx, [0]);
        assert_eq!(with_gap.valley_idx, [3]);
        assert_eq!(with_gap.status, [DrawdownStatus::Recovered as i8]);
        // A gap on the last bar cannot close an episode either.
        let trailing = find_drawdowns(&[10.0, 8.0, f64::NAN], 3, 1).unwrap();
        assert!(
            trailing.is_empty(),
            "the last real bar was not the last bar, so nothing marked it active"
        );
    }

    #[test]
    fn a_recovered_episode_is_measured_exclusively_and_an_active_one_is_not() {
        // Found by the statistics table: one convention for both shifts every
        // drawdown duration by a bar, in a direction that depends on how many
        // episodes happened to still be running.
        //          0     1    2    3     4
        let closed = [10.0, 8.0, 6.0, 9.0, 11.0];
        let d = find_drawdowns(&closed, 5, 1).unwrap();
        assert_eq!(d.status, [DrawdownStatus::Recovered as i8]);
        // start 1, end 4: four bars, not five.
        assert_eq!(d.duration(), [3.0]);

        let active = [10.0, 8.0, 6.0, 7.0];
        let a = find_drawdowns(&active, 4, 1).unwrap();
        assert_eq!(a.status, [DrawdownStatus::Active as i8]);
        // start 1, end 3, still under water: the last bar counts.
        assert_eq!(a.duration(), [3.0]);
    }

    #[test]
    fn durations_measure_the_two_phases_separately() {
        //          0    1    2    3    4     5
        let ts = [10.0, 9.0, 8.0, 6.0, 9.0, 11.0];
        let d = find_drawdowns(&ts, 6, 1).unwrap();
        // Decline is bars 1..=3, inclusive of both: three bars.
        assert_eq!(d.decline_duration(), [3.0]);
        // Recovery is bars 3..5, exclusive of the valley: two bars.
        assert_eq!(d.recovery_duration(), [2.0]);
        assert_eq!(d.recovery_duration_ratio(), [2.0 / 3.0]);
    }

    #[test]
    fn columns_are_kept_apart_and_come_out_in_order() {
        // Column-major: two columns of three bars each.
        let ts = [10.0, 8.0, 11.0, 5.0, 4.0, 3.0];
        let d = find_drawdowns(&ts, 3, 2).unwrap();
        assert_eq!(d.col, [0, 1]);
        assert!(d.col_map(2).unwrap().is_contiguous());
        assert_eq!(d.status[0], DrawdownStatus::Recovered as i8);
        assert_eq!(d.status[1], DrawdownStatus::Active as i8);
    }

    #[test]
    fn a_shape_mismatch_is_rejected() {
        assert!(find_drawdowns(&[1.0], 2, 2).is_err());
    }

    #[test]
    fn a_column_of_nothing_but_gaps_has_no_drawdowns() {
        assert!(find_drawdowns(&[f64::NAN; 3], 3, 1).unwrap().is_empty());
        assert!(find_drawdowns(&[], 0, 1).unwrap().is_empty());
    }
}
