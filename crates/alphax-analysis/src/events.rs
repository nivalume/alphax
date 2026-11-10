//! Event studies: what happens to price around a signal.
//!
//! An event study answers "when this condition fires, what does the price do
//! next — and what was it doing just before?" Every event is re-anchored to
//! its own bar, so a breakout in March and one in October line up on the
//! same offset axis and can be averaged. The mean path shows the typical
//! drift, the median path shows it without the outliers, and the hit rate
//! at an offset says how often the move was positive at all — a strong mean
//! carried by one lucky event looks very different from a weak one that
//! almost always works.
//!
//! # Look-ahead: this module reads the future
//!
//! A forward path at offset `k > 0` is `close[t + k]`, a price that did not
//! exist when the event fired. That is the whole point of the analysis and
//! exactly why it must never feed back into a signal. Use these paths to
//! decide *whether* a signal is worth simulating, never as an input to the
//! simulation itself.
//!
//! # Layout
//!
//! Inputs are column-major, like the rest of the workspace: column `c` of
//! `close` is `close[c * n_rows..(c + 1) * n_rows]`, and `events` has the
//! same shape. Output paths are row-major — one row per event — because an
//! event, not a bar, is the unit of the study.
//!
//! # Edge events are kept, padded with `NaN`
//!
//! An event closer than `before` bars to the start of its column, or closer
//! than `after` bars to the end, still produces a full-width row; the
//! offsets that fall outside the column are `NaN`. Dropping such events
//! would silently bias the study toward the middle of the sample, and every
//! reduction here skips `NaN` anyway, so a padded row contributes exactly
//! the offsets it has.
//!
//! An event whose own close is `NaN` yields a row of `NaN` — including
//! offset 0 — because there is no price to anchor it to.

use alphax_core::error::{Error, Result};

/// The re-anchored price paths around every event, plus where each came from.
#[derive(Debug, Clone, PartialEq)]
pub struct EventPaths {
    /// Column of each event, in output order.
    pub col: Vec<i64>,
    /// Row (bar index within its column) of each event, in output order.
    pub idx: Vec<i64>,
    /// The offset axis, `-before..=after`.
    pub offsets: Vec<i64>,
    /// Row-major: `paths[e * offsets.len() + j]` is
    /// `close[t + offsets[j]] / close[t] - 1` for event `e` at bar `t`.
    pub paths: Vec<f64>,
}

impl EventPaths {
    /// Number of events in the study.
    #[must_use]
    pub fn n_events(&self) -> usize {
        self.col.len()
    }

    /// Width of each path, `before + after + 1`.
    #[must_use]
    pub fn width(&self) -> usize {
        self.offsets.len()
    }

    /// Values at offset position `j` across all events.
    fn column_at(&self, j: usize) -> impl Iterator<Item = f64> + '_ {
        let w = self.width();
        (0..self.n_events()).map(move |e| self.paths[e * w + j])
    }
}

/// Build the forward (and backward) return path around every event.
///
/// Events are emitted column by column, and in bar order within a column.
/// Each path is `close[t + k] / close[t] - 1` for `k` in `-before..=after`,
/// so offset 0 is `0.0` whenever the anchor price is present.
///
/// # Errors
///
/// [`Error::Invariant`] if `close` or `events` is not `n_rows * n_cols`
/// long.
pub fn forward_paths(
    close: &[f64],
    events: &[bool],
    n_rows: usize,
    n_cols: usize,
    before: usize,
    after: usize,
) -> Result<EventPaths> {
    let expected = n_rows
        .checked_mul(n_cols)
        .ok_or_else(|| Error::Invariant("n_rows * n_cols overflows".into()))?;
    if close.len() != expected {
        return Err(Error::Invariant(format!(
            "close has {} values, expected {n_rows} x {n_cols} = {expected}",
            close.len()
        )));
    }
    if events.len() != expected {
        return Err(Error::Invariant(format!(
            "events has {} values, expected {n_rows} x {n_cols} = {expected}",
            events.len()
        )));
    }

    let offsets: Vec<i64> = (-(before as i64)..=after as i64).collect();
    let mut out = EventPaths {
        col: Vec::new(),
        idx: Vec::new(),
        offsets,
        paths: Vec::new(),
    };

    for c in 0..n_cols {
        let column = &close[c * n_rows..(c + 1) * n_rows];
        let fired = &events[c * n_rows..(c + 1) * n_rows];
        for (t, _) in fired.iter().enumerate().filter(|&(_, &e)| e) {
            out.col.push(c as i64);
            out.idx.push(t as i64);
            let base = column[t];
            for &k in &out.offsets {
                let j = t as i64 + k;
                // Out of range is padding, not an error: see the module doc.
                let v = if j < 0 || j >= n_rows as i64 {
                    f64::NAN
                } else {
                    column[j as usize] / base - 1.0
                };
                out.paths.push(v);
            }
        }
    }
    Ok(out)
}

/// Mean across events at each offset, skipping `NaN`.
///
/// An offset with no finite value across all events is `NaN`.
#[must_use]
pub fn mean_path(paths: &EventPaths) -> Vec<f64> {
    (0..paths.width())
        .map(|j| {
            let (sum, count) = paths
                .column_at(j)
                .filter(|v| !v.is_nan())
                .fold((0.0, 0usize), |(s, n), v| (s + v, n + 1));
            if count == 0 {
                f64::NAN
            } else {
                sum / count as f64
            }
        })
        .collect()
}

/// Median across events at each offset, skipping `NaN`.
///
/// An even count averages the two middle values. An offset with no
/// non-`NaN` value is `NaN`.
#[must_use]
pub fn median_path(paths: &EventPaths) -> Vec<f64> {
    (0..paths.width())
        .map(|j| {
            let mut vals: Vec<f64> = paths.column_at(j).filter(|v| !v.is_nan()).collect();
            if vals.is_empty() {
                return f64::NAN;
            }
            vals.sort_by(f64::total_cmp);
            let m = vals.len() / 2;
            if vals.len() % 2 == 1 {
                vals[m]
            } else {
                0.5 * (vals[m - 1] + vals[m])
            }
        })
        .collect()
}

/// Fraction of events whose path is strictly positive at `offset`.
///
/// `NaN` values are excluded from both numerator and denominator. Returns
/// `NaN` if `offset` is not on the axis or no event has a value there.
/// A value of exactly zero is not a hit, so offset 0 always scores `0.0`.
#[must_use]
pub fn hit_rate(paths: &EventPaths, offset: i64) -> f64 {
    let Some(j) = paths.offsets.iter().position(|&o| o == offset) else {
        return f64::NAN;
    };
    let (hits, count) = paths
        .column_at(j)
        .filter(|v| !v.is_nan())
        .fold((0usize, 0usize), |(h, n), v| {
            (h + usize::from(v > 0.0), n + 1)
        });
    if count == 0 {
        f64::NAN
    } else {
        hits as f64 / count as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close_to(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    /// Column 0: `[10, 11, 12, 9, 10]`, events at bars 1 and 3.
    /// Column 1: `[20, 22, 18, 24, 20]`, event at bar 2.
    /// `before = 1`, `after = 2`, so offsets are `[-1, 0, 1, 2]`.
    fn fixture() -> EventPaths {
        let close = [10.0, 11.0, 12.0, 9.0, 10.0, 20.0, 22.0, 18.0, 24.0, 20.0];
        let events = [
            false, true, false, true, false, false, false, true, false, false,
        ];
        forward_paths(&close, &events, 5, 2, 1, 2).unwrap()
    }

    #[test]
    fn paths_match_hand_calculation() {
        let p = fixture();
        assert_eq!(p.col, vec![0, 0, 1]);
        assert_eq!(p.idx, vec![1, 3, 2]);
        assert_eq!(p.offsets, vec![-1, 0, 1, 2]);
        assert_eq!(p.paths.len(), 3 * 4);

        // Event (0, 1), anchor 11: 10, 11, 12, 9.
        let e0 = [-1.0 / 11.0, 0.0, 1.0 / 11.0, -2.0 / 11.0];
        // Event (0, 3), anchor 9: 12, 9, 10, out of range.
        let e1 = [1.0 / 3.0, 0.0, 1.0 / 9.0];
        // Event (1, 2), anchor 18: 22, 18, 24, 20.
        let e2 = [2.0 / 9.0, 0.0, 1.0 / 3.0, 1.0 / 9.0];

        for (j, &v) in e0.iter().enumerate() {
            assert!(close_to(p.paths[j], v), "e0[{j}] = {}", p.paths[j]);
        }
        for (j, &v) in e1.iter().enumerate() {
            assert!(close_to(p.paths[4 + j], v), "e1[{j}] = {}", p.paths[4 + j]);
        }
        for (j, &v) in e2.iter().enumerate() {
            assert!(close_to(p.paths[8 + j], v), "e2[{j}] = {}", p.paths[8 + j]);
        }
    }

    #[test]
    fn edge_events_are_kept_with_nan_padding() {
        let p = fixture();
        // Event (0, 3) at offset +2 would read bar 5 of a 5-bar column.
        assert!(p.paths[4 + 3].is_nan());

        // An event on the first bar pads every negative offset.
        let close = [1.0, 2.0, 4.0];
        let events = [true, false, false];
        let p = forward_paths(&close, &events, 3, 1, 2, 1).unwrap();
        assert_eq!(p.n_events(), 1);
        assert_eq!(p.offsets, vec![-2, -1, 0, 1]);
        assert!(p.paths[0].is_nan());
        assert!(p.paths[1].is_nan());
        assert!(close_to(p.paths[2], 0.0));
        assert!(close_to(p.paths[3], 1.0));
    }

    #[test]
    fn nan_anchor_gives_nan_row() {
        let close = [1.0, f64::NAN, 3.0];
        let events = [false, true, false];
        let p = forward_paths(&close, &events, 3, 1, 1, 1).unwrap();
        assert!(p.paths.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn mean_path_skips_nan() {
        let m = mean_path(&fixture());
        assert!(close_to(m[0], (-1.0 / 11.0 + 1.0 / 3.0 + 2.0 / 9.0) / 3.0));
        assert!(close_to(m[1], 0.0));
        assert!(close_to(m[2], (1.0 / 11.0 + 1.0 / 9.0 + 1.0 / 3.0) / 3.0));
        // Only two events reach offset +2.
        assert!(close_to(m[3], (-2.0 / 11.0 + 1.0 / 9.0) / 2.0));
    }

    #[test]
    fn median_path_odd_and_even_counts() {
        let m = median_path(&fixture());
        // Offset -1: {-1/11, 2/9, 1/3} -> 2/9.
        assert!(close_to(m[0], 2.0 / 9.0));
        assert!(close_to(m[1], 0.0));
        // Offset +1: {1/11, 1/9, 1/3} -> 1/9.
        assert!(close_to(m[2], 1.0 / 9.0));
        // Offset +2: {-2/11, 1/9} -> their mean.
        assert!(close_to(m[3], (-2.0 / 11.0 + 1.0 / 9.0) / 2.0));
    }

    #[test]
    fn hit_rate_counts_strictly_positive() {
        let p = fixture();
        assert!(close_to(hit_rate(&p, -1), 2.0 / 3.0));
        assert!(close_to(hit_rate(&p, 0), 0.0));
        assert!(close_to(hit_rate(&p, 1), 1.0));
        assert!(close_to(hit_rate(&p, 2), 0.5));
        assert!(hit_rate(&p, 3).is_nan());
        assert!(hit_rate(&p, -2).is_nan());
    }

    #[test]
    fn no_events_gives_nan_reductions() {
        let close = [1.0, 2.0];
        let events = [false, false];
        let p = forward_paths(&close, &events, 2, 1, 0, 1).unwrap();
        assert_eq!(p.n_events(), 0);
        assert!(p.paths.is_empty());
        assert!(mean_path(&p).iter().all(|v| v.is_nan()));
        assert!(median_path(&p).iter().all(|v| v.is_nan()));
        assert!(hit_rate(&p, 1).is_nan());
    }

    #[test]
    fn shape_errors() {
        let close = [1.0, 2.0, 3.0];
        let events = [true, false, false, false];
        assert!(matches!(
            forward_paths(&close, &events[..3], 2, 2, 0, 1),
            Err(Error::Invariant(_))
        ));
        assert!(matches!(
            forward_paths(&close, &events, 3, 1, 0, 1),
            Err(Error::Invariant(_))
        ));
    }
}
