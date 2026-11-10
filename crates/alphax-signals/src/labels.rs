//! Supervised-learning targets: forward returns, trend, triple-barrier and
//! zig-zag pivot labels.
//!
//! # Every function here reads the future
//!
//! These are *labels*, the `y` a model is trained to predict. The value at
//! bar `t` is computed from bars after `t` — that is the whole point — so
//! none of them is a signal and none may be fed to a simulation as entries,
//! exits or features. A backtest that does so is not optimistic, it is
//! wrong: it trades on prices it could not have seen.
//!
//! The rest of this crate is causal; this module is the exception, and each
//! function says how far ahead it looks so the gap a train/test split must
//! leave (the purge) can be read off the signature.
//!
//! # Conventions
//!
//! Inputs and outputs are column-major, `n_rows * n_cols`, like every other
//! array in the crate, and columns never see each other: a label near the
//! end of one column does not peek into the start of the next. Closes are
//! assumed positive; a return off a zero or negative price is not a number
//! anyone should train on, and is not guarded against.

use alphax_core::error::{Error, Result};

fn check_shape(close: &[f64], n_rows: usize, n_cols: usize) -> Result<()> {
    if close.len() != n_rows * n_cols {
        return Err(Error::Broadcast {
            left: (close.len(), 1),
            right: (n_rows, n_cols),
        });
    }
    Ok(())
}

/// Simple return from each bar to the bar `horizon` rows later.
///
/// `out[t] = close[t + horizon] / close[t] - 1`, NaN where `t + horizon`
/// runs past the end of the column or either close is NaN.
///
/// **Look-ahead:** reads `horizon` bars into the future.
///
/// # Errors
///
/// [`Error::Broadcast`] if `close` is not `n_rows * n_cols` long, and
/// [`Error::Invariant`] if `horizon` is 0 — a zero-bar return is
/// identically zero and almost always an off-by-one in the caller.
pub fn fixed_horizon_returns(
    close: &[f64],
    n_rows: usize,
    n_cols: usize,
    horizon: usize,
) -> Result<Vec<f64>> {
    check_shape(close, n_rows, n_cols)?;
    if horizon == 0 {
        return Err(Error::Invariant(
            "fixed_horizon_returns: horizon must be at least 1".into(),
        ));
    }
    let mut out = vec![f64::NAN; n_rows * n_cols];
    for col in 0..n_cols {
        let c = &close[col * n_rows..(col + 1) * n_rows];
        let o = &mut out[col * n_rows..(col + 1) * n_rows];
        // Rows within `horizon` of the end keep their NaN: there is no
        // forward bar, and borrowing one from the next column would be a
        // silent leak across assets.
        let later = c.get(horizon..).unwrap_or(&[]);
        for ((o, &now), &then) in o.iter_mut().zip(c).zip(later) {
            *o = then / now - 1.0;
        }
    }
    Ok(out)
}

/// Three-class direction of the forward return over `horizon` bars.
///
/// +1 if the return is strictly above `threshold`, -1 if strictly below
/// `-threshold`, 0 otherwise. Where the return is undefined (past the end,
/// or a NaN close) the label is 0 rather than a sentinel, because an `i8`
/// class has no NaN; callers who need to drop those rows should mask with
/// [`fixed_horizon_returns`].
///
/// **Look-ahead:** reads `horizon` bars into the future.
///
/// # Errors
///
/// As [`fixed_horizon_returns`], plus [`Error::Invariant`] if `threshold` is
/// negative or NaN — a negative band would make +1 and -1 overlap.
pub fn trend_labels(
    close: &[f64],
    n_rows: usize,
    n_cols: usize,
    horizon: usize,
    threshold: f64,
) -> Result<Vec<i8>> {
    if threshold.is_nan() || threshold < 0.0 {
        return Err(Error::Invariant(format!(
            "trend_labels: threshold must be non-negative, got {threshold}"
        )));
    }
    let returns = fixed_horizon_returns(close, n_rows, n_cols, horizon)?;
    // NaN fails both comparisons and falls through to 0, which is the
    // documented label for an undefined return.
    Ok(returns
        .iter()
        .map(|&r| {
            if r > threshold {
                1
            } else if r < -threshold {
                -1
            } else {
                0
            }
        })
        .collect())
}

/// Triple-barrier labels: which of a profit-take, a stop-loss or a time
/// limit is touched first.
///
/// For each bar `t` with entry price `p = close[t]`, bars `s = t+1 ..=
/// min(t + max_hold, n_rows - 1)` are walked in order:
///
/// - `close[s] / p - 1 >= up` first: label +1, decided at `s`;
/// - `close[s] / p - 1 <= -down` first: label -1, decided at `s`;
/// - neither: label 0, decided at the last bar walked (the vertical
///   barrier, or the column's last bar if that comes sooner).
///
/// The second vector holds that deciding row, as a row index within the
/// column (not a flat index). It is -1 where `t` is the column's last bar
/// or `close[t]` is NaN; both get label 0. A NaN close *after* `t` is
/// skipped rather than ending the walk — a missing print touches no barrier.
///
/// Barriers are on closes only. A bar whose high crossed `up` but closed
/// below it does not count, which makes these labels conservative relative
/// to an intrabar stop.
///
/// **Look-ahead:** reads up to `max_hold` bars into the future, and the
/// deciding row says exactly how far each label looked — which is what a
/// purged train/test split needs.
///
/// # Errors
///
/// [`Error::Broadcast`] on a shape mismatch; [`Error::Invariant`] if `up` or
/// `down` is not a positive finite number, or `max_hold` is 0.
pub fn triple_barrier(
    close: &[f64],
    n_rows: usize,
    n_cols: usize,
    up: f64,
    down: f64,
    max_hold: usize,
) -> Result<(Vec<i8>, Vec<i64>)> {
    check_shape(close, n_rows, n_cols)?;
    // A zero-width barrier would be touched by the first flat bar, and then
    // the +1/-1 order below would decide the label rather than the price.
    if !(up.is_finite() && up > 0.0 && down.is_finite() && down > 0.0) {
        return Err(Error::Invariant(format!(
            "triple_barrier: up and down must be positive and finite, got {up} and {down}"
        )));
    }
    if max_hold == 0 {
        return Err(Error::Invariant(
            "triple_barrier: max_hold must be at least 1".into(),
        ));
    }

    let mut labels = vec![0_i8; n_rows * n_cols];
    let mut decided = vec![-1_i64; n_rows * n_cols];
    for col in 0..n_cols {
        let base = col * n_rows;
        let c = &close[base..base + n_rows];
        for t in 0..n_rows {
            let p = c[t];
            if p.is_nan() || t + 1 >= n_rows {
                continue;
            }
            let end = (t + max_hold).min(n_rows - 1);
            let mut label = 0_i8;
            let mut at = end;
            for (s, &x) in c.iter().enumerate().take(end + 1).skip(t + 1) {
                let r = x / p - 1.0;
                // With both barriers strictly away from zero they cannot
                // hold on the same bar, so the check order is not a
                // tie-break. NaN fails both and the walk moves on.
                if r >= up {
                    label = 1;
                    at = s;
                    break;
                }
                if r <= -down {
                    label = -1;
                    at = s;
                    break;
                }
            }
            labels[base + t] = label;
            decided[base + t] = at as i64;
        }
    }
    Ok((labels, decided))
}

/// Zig-zag turning points: +1 at confirmed troughs, -1 at confirmed peaks.
///
/// The rule, per column, ignoring NaN closes entirely:
///
/// 1. Until the first swing is confirmed, track the running high and low
///    since the first finite close. The first bar that is at least
///    `threshold` above the running low (`close / low - 1 >= threshold`)
///    confirms that low as a trough; the first that is at least `threshold`
///    below the running high (`close / high - 1 <= -threshold`) confirms that
///    high as a peak.
/// 2. After a trough, the candidate peak is the highest close since. It is
///    replaced by any strictly higher close, and confirmed as a peak by the
///    first close at least `threshold` below it. Troughs mirror this.
/// 3. The candidate extreme of the last leg is never confirmed and stays 0:
///    nothing after it reversed far enough to say it was a turning point.
///
/// Strict replacement means a run of equal highs pivots on its first bar.
/// Every non-zero label therefore alternates, trough then peak, and
/// consecutive pivots differ by at least `threshold`.
///
/// **Look-ahead:** unbounded. A pivot at `t` is only known once a later
/// bar confirms it, and that bar can be arbitrarily far ahead. Of every
/// label in this module, this is the one most often leaked by accident —
/// "buy at the zig-zag trough" backtests beautifully and cannot be traded.
///
/// # Errors
///
/// [`Error::Broadcast`] on a shape mismatch; [`Error::Invariant`] if
/// `threshold` is not a positive finite number. At zero every tick is a
/// pivot and equal closes flip the trend back and forth.
pub fn pivot_labels(
    close: &[f64],
    n_rows: usize,
    n_cols: usize,
    threshold: f64,
) -> Result<Vec<i8>> {
    check_shape(close, n_rows, n_cols)?;
    if !(threshold.is_finite() && threshold > 0.0) {
        return Err(Error::Invariant(format!(
            "pivot_labels: threshold must be positive and finite, got {threshold}"
        )));
    }

    #[derive(Clone, Copy)]
    enum Leg {
        // No swing confirmed yet: both extremes are live candidates.
        Unknown { hi: (f64, usize), lo: (f64, usize) },
        // Last confirmed pivot was a trough; tracking the next peak.
        Up { peak: (f64, usize) },
        // Last confirmed pivot was a peak; tracking the next trough.
        Down { trough: (f64, usize) },
    }

    let mut out = vec![0_i8; n_rows * n_cols];
    for col in 0..n_cols {
        let base = col * n_rows;
        let c = &close[base..base + n_rows];
        let o = &mut out[base..base + n_rows];
        let mut leg: Option<Leg> = None;

        for (i, &x) in c.iter().enumerate() {
            if x.is_nan() {
                continue;
            }
            leg = Some(match leg {
                None => Leg::Unknown {
                    hi: (x, i),
                    lo: (x, i),
                },
                Some(Leg::Unknown { mut hi, mut lo }) => {
                    if x / lo.0 - 1.0 >= threshold {
                        o[lo.1] = 1;
                        Leg::Up { peak: (x, i) }
                    } else if x / hi.0 - 1.0 <= -threshold {
                        o[hi.1] = -1;
                        Leg::Down { trough: (x, i) }
                    } else {
                        if x > hi.0 {
                            hi = (x, i);
                        }
                        if x < lo.0 {
                            lo = (x, i);
                        }
                        Leg::Unknown { hi, lo }
                    }
                }
                Some(Leg::Up { peak }) => {
                    if x > peak.0 {
                        Leg::Up { peak: (x, i) }
                    } else if x / peak.0 - 1.0 <= -threshold {
                        o[peak.1] = -1;
                        Leg::Down { trough: (x, i) }
                    } else {
                        Leg::Up { peak }
                    }
                }
                Some(Leg::Down { trough }) => {
                    if x < trough.0 {
                        Leg::Down { trough: (x, i) }
                    } else if x / trough.0 - 1.0 >= threshold {
                        o[trough.1] = 1;
                        Leg::Up { peak: (x, i) }
                    } else {
                        Leg::Down { trough }
                    }
                }
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close_to(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(x, y)| (x.is_nan() && y.is_nan()) || (x - y).abs() < 1e-12)
    }

    #[test]
    fn forward_returns_by_hand() {
        let close = [100.0, 110.0, 99.0, 121.0];
        let r1 = fixed_horizon_returns(&close, 4, 1, 1).unwrap();
        assert!(close_to(&r1, &[0.1, -0.1, 121.0 / 99.0 - 1.0, f64::NAN]));
        let r2 = fixed_horizon_returns(&close, 4, 1, 2).unwrap();
        assert!(close_to(&r2, &[-0.01, 0.1, f64::NAN, f64::NAN]));
    }

    #[test]
    fn forward_returns_propagate_nan_and_survive_a_long_horizon() {
        let close = [100.0, f64::NAN, 110.0];
        let r = fixed_horizon_returns(&close, 3, 1, 1).unwrap();
        assert!(close_to(&r, &[f64::NAN, f64::NAN, f64::NAN]));
        // A horizon longer than the column is all NaN, not an underflow.
        let r = fixed_horizon_returns(&close, 3, 1, 10).unwrap();
        assert!(r.iter().all(|x| x.is_nan()));
    }

    #[test]
    fn trend_labels_by_hand() {
        let close = [100.0, 110.0, 99.0, 100.0, f64::NAN, 100.0];
        let y = trend_labels(&close, 6, 1, 1, 0.05).unwrap();
        // +10%, -10%, +1% inside the band, into a NaN, out of a NaN, past
        // the end.
        assert_eq!(y, [1, -1, 0, 0, 0, 0]);
    }

    #[test]
    fn trend_threshold_is_strict() {
        // Both moves are exactly +-25% in binary, so the boundary itself is
        // tested rather than rounding near it.
        let close = [1.0, 1.25, 0.9375, 0.5];
        let y = trend_labels(&close, 4, 1, 1, 0.25).unwrap();
        assert_eq!(y, [0, 0, -1, 0], "+-25% sits on the band, not outside it");
    }

    #[test]
    fn triple_barrier_by_hand() {
        let close = [100.0, 102.0, 106.0, 104.0, 95.0, 97.0];
        let (y, at) = triple_barrier(&close, 6, 1, 0.05, 0.05, 3).unwrap();
        // t0 hits +6% at 2. t1, t2 and t3 all hit their stop on the drop to
        // 95 at 4. t4 sees only bar 5 (+2%) before the column ends. t5 has
        // no forward bar.
        assert_eq!(y, [1, -1, -1, -1, 0, 0]);
        assert_eq!(at, [2, 4, 4, 4, 5, -1]);
    }

    #[test]
    fn triple_barrier_vertical_barrier() {
        let close = [100.0, 101.0, 102.0, 103.0, 104.0];
        let (y, at) = triple_barrier(&close, 5, 1, 0.1, 0.1, 2).unwrap();
        assert_eq!(y, [0, 0, 0, 0, 0]);
        // Time-out at t + 2, clipped to the last bar for t3.
        assert_eq!(at, [2, 3, 4, 4, -1]);
    }

    #[test]
    fn triple_barrier_skips_nan() {
        let close = [100.0, f64::NAN, 106.0, 100.0];
        let (y, at) = triple_barrier(&close, 4, 1, 0.05, 0.05, 3).unwrap();
        assert_eq!(y, [1, 0, -1, 0]);
        assert_eq!(at, [2, -1, 3, -1]);
    }

    #[test]
    fn pivots_by_hand() {
        let close = [10.0, 12.0, 9.0, 9.5, 13.0, 11.0];
        let y = pivot_labels(&close, 6, 1, 0.15).unwrap();
        // 10 -> 12 (+20%) confirms the trough at 0; 12 -> 9 (-25%) the peak
        // at 1; 9 -> 13 the trough at 2; 13 -> 11 (-15.4%) the peak at 4.
        assert_eq!(y, [1, -1, 1, 0, -1, 0]);
    }

    #[test]
    fn the_last_leg_is_unconfirmed() {
        // 13 -> 11.5 is only -11.5%, so 13 is never confirmed as a peak.
        let close = [10.0, 12.0, 9.0, 9.5, 13.0, 11.5];
        let y = pivot_labels(&close, 6, 1, 0.15).unwrap();
        assert_eq!(y, [1, -1, 1, 0, 0, 0]);
    }

    #[test]
    fn pivots_ignore_nan_and_extend_the_candidate() {
        let close = [10.0, 12.0, 9.0, f64::NAN, 8.0, 13.0, 11.0];
        let y = pivot_labels(&close, 7, 1, 0.15).unwrap();
        // The trough moves from 9 to 8 across the NaN.
        assert_eq!(y, [1, -1, 0, 0, 1, -1, 0]);
    }

    #[test]
    fn columns_are_independent() {
        // Column a ends high and column b starts low: any leak across the
        // boundary would show up in a's last rows.
        let a = [100.0, 102.0, 106.0, 104.0, 95.0];
        let b = [50.0, 60.0, 45.0, 44.0, 70.0];
        let both: Vec<f64> = a.iter().chain(&b).copied().collect();

        let joined = fixed_horizon_returns(&both, 5, 2, 2).unwrap();
        let mut split = fixed_horizon_returns(&a, 5, 1, 2).unwrap();
        split.extend(fixed_horizon_returns(&b, 5, 1, 2).unwrap());
        assert!(close_to(&joined, &split));

        let joined = trend_labels(&both, 5, 2, 1, 0.03).unwrap();
        let mut split = trend_labels(&a, 5, 1, 1, 0.03).unwrap();
        split.extend(trend_labels(&b, 5, 1, 1, 0.03).unwrap());
        assert_eq!(joined, split);

        let joined = triple_barrier(&both, 5, 2, 0.05, 0.05, 10).unwrap();
        let (mut y, mut at) = triple_barrier(&a, 5, 1, 0.05, 0.05, 10).unwrap();
        let (yb, atb) = triple_barrier(&b, 5, 1, 0.05, 0.05, 10).unwrap();
        y.extend(yb);
        at.extend(atb);
        assert_eq!(joined, (y, at));
        assert_eq!(joined.1[4], -1, "a's last bar has no forward bar");

        let joined = pivot_labels(&both, 5, 2, 0.15).unwrap();
        let mut split = pivot_labels(&a, 5, 1, 0.15).unwrap();
        split.extend(pivot_labels(&b, 5, 1, 0.15).unwrap());
        assert_eq!(joined, split);
    }

    #[test]
    fn shape_and_parameter_errors() {
        let close = [1.0, 2.0, 3.0];
        assert!(matches!(
            fixed_horizon_returns(&close, 2, 2, 1),
            Err(Error::Broadcast { .. })
        ));
        assert!(matches!(
            trend_labels(&close, 2, 2, 1, 0.0),
            Err(Error::Broadcast { .. })
        ));
        assert!(matches!(
            triple_barrier(&close, 4, 1, 0.1, 0.1, 1),
            Err(Error::Broadcast { .. })
        ));
        assert!(matches!(
            pivot_labels(&close, 1, 2, 0.1),
            Err(Error::Broadcast { .. })
        ));

        assert!(matches!(
            fixed_horizon_returns(&close, 3, 1, 0),
            Err(Error::Invariant(_))
        ));
        assert!(matches!(
            trend_labels(&close, 3, 1, 1, -0.1),
            Err(Error::Invariant(_))
        ));
        assert!(matches!(
            triple_barrier(&close, 3, 1, 0.0, 0.1, 1),
            Err(Error::Invariant(_))
        ));
        assert!(matches!(
            triple_barrier(&close, 3, 1, 0.1, 0.1, 0),
            Err(Error::Invariant(_))
        ));
        assert!(matches!(
            pivot_labels(&close, 3, 1, f64::NAN),
            Err(Error::Invariant(_))
        ));
    }

    #[test]
    fn empty_input_is_fine() {
        assert!(fixed_horizon_returns(&[], 0, 3, 1).unwrap().is_empty());
        assert!(triple_barrier(&[], 0, 0, 0.1, 0.1, 1).unwrap().0.is_empty());
        assert!(pivot_labels(&[], 0, 2, 0.1).unwrap().is_empty());
    }
}
