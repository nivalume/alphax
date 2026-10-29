//! Forward returns: what a factor stamped on a bar is scored against.

use phasor_core::error::Result;

use crate::panel::Panel;

/// `close[t + lag + horizon] / close[t + lag] - 1` for every bar and symbol,
/// column-major like `close`.
///
/// `lag` is the delay between knowing the factor and trading on it: with
/// `lag = 1`, a factor computed on bar `t`'s close is traded at bar `t + 1`'s
/// close, and the return it earns starts there. `lag = 0` is the return from
/// the factor's own bar, as Qlib's and alphalens' labels are. `NaN` where the
/// window runs past the end of the column or either close is not finite.
///
/// **Reads the future** by `lag + horizon` bars, on purpose.
///
/// # Errors
///
/// Never, today; the `Result` keeps the signature of the other kernels.
pub fn forward_returns(close: &Panel<'_>, horizon: usize, lag: usize) -> Result<Vec<f64>> {
    let n = close.n_rows();
    let mut out = vec![f64::NAN; close.data().len()];
    let ahead = lag.saturating_add(horizon);
    for c in 0..close.n_cols() {
        let col = close.col(c);
        let dst = &mut out[c * n..(c + 1) * n];
        for t in 0..n.saturating_sub(ahead) {
            let (from, to) = (col[t + lag], col[t + ahead]);
            if from.is_finite() && to.is_finite() {
                dst[t] = to / from - 1.0;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lag_shifts_the_window() {
        let close = [1.0, 2.0, 4.0, 8.0];
        let p = Panel::new(&close, 4, 1).unwrap();
        let r0 = forward_returns(&p, 1, 0).unwrap();
        let r1 = forward_returns(&p, 1, 1).unwrap();
        assert_eq!(&r0[..3], &[1.0, 1.0, 1.0]);
        assert!(r0[3].is_nan());
        assert_eq!(&r1[..2], &[1.0, 1.0]);
        assert!(r1[2].is_nan() && r1[3].is_nan());
        assert!(
            forward_returns(&p, 9, 1)
                .unwrap()
                .iter()
                .all(|v| v.is_nan())
        );
    }
}
