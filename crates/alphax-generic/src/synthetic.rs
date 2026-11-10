//! Synthetic price paths with known properties.
//!
//! # What these are for
//!
//! A backtest on real prices cannot tell a strategy that found an edge from
//! one that fit the noise: the true process is unknown. A synthetic path
//! removes that ambiguity. A [`gbm`] with `mu = 0` has no edge by
//! construction, so a strategy that reports a healthy Sharpe on it is
//! measuring its own overfitting. A [`random_walk`] has independent steps, so
//! any signal that "predicts" one is reading noise. These functions exist to
//! build those controls, not to model a market.
//!
//! # Every draw is a function of `(seed, row, col)`
//!
//! There is no generator state. Each normal or uniform draw is derived from
//! the seed, the row and the column through SplitMix64
//! ([`alphax_core::random`]), so the same arguments
//! always give the same path, bit for bit, and column `c` does not depend on
//! how many other columns were asked for. A result that came from a seed can
//! be reproduced from that seed alone.
//!
//! # Layout
//!
//! Every output is column-major: `n_rows * n_cols` values, with column `c` at
//! `[c * n_rows..(c + 1) * n_rows]`. Rows are bars, columns are independent
//! paths.

use alphax_core::error::{Error, Result};
use alphax_core::random::{bits, to_unit_nonzero};

/// Salts that keep the draws of one `(seed, row, col)` independent of each
/// other. Each draw has its own, so changing one never shifts another.
const SALT_NORMAL_U1: u64 = 0x6E6F_726D_616C_5F31;
const SALT_NORMAL_U2: u64 = 0x6E6F_726D_616C_5F32;
const SALT_HIGH: u64 = 0x6F68_6C63_5F68_6967;
const SALT_LOW: u64 = 0x6F68_6C63_5F6C_6F77;

/// A uniform draw on `(0, 1]`, keyed by `(seed, row, col, salt)`.
///
/// Never exactly `0`, which is what keeps `ln` in [`normal`] finite; see
/// [`to_unit_nonzero`] for the one draw in 2⁵³ that is exactly `1`.
fn uniform(seed: u64, row: usize, col: usize, salt: u64) -> f64 {
    to_unit_nonzero(bits(seed, row, col, salt))
}

/// A standard normal draw by Box–Muller, from two independent uniforms.
fn normal(seed: u64, row: usize, col: usize) -> f64 {
    // `uniform` cannot return 0, but the floor costs nothing and keeps the
    // logarithm finite even if that ever changes.
    let u1 = uniform(seed, row, col, SALT_NORMAL_U1).max(f64::MIN_POSITIVE);
    let u2 = uniform(seed, row, col, SALT_NORMAL_U2);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Rejects a shape whose element count does not fit in memory's index type.
fn checked_len(n_rows: usize, n_cols: usize) -> Result<usize> {
    n_rows.checked_mul(n_cols).ok_or_else(|| {
        Error::Invariant(format!(
            "n_rows ({n_rows}) * n_cols ({n_cols}) overflows usize"
        ))
    })
}

/// Fills a column-major matrix whose first row is `start` and whose later
/// rows are `step(previous, row, col)`.
fn build_paths(
    n_rows: usize,
    n_cols: usize,
    start: f64,
    mut step: impl FnMut(f64, usize, usize) -> f64,
) -> Result<Vec<f64>> {
    let mut out = vec![0.0; checked_len(n_rows, n_cols)?];
    if n_rows == 0 {
        return Ok(out);
    }
    for (col, path) in out.chunks_exact_mut(n_rows).enumerate() {
        path[0] = start;
        for row in 1..n_rows {
            path[row] = step(path[row - 1], row, col);
        }
    }
    Ok(out)
}

/// Gaussian random walks: `x_{t+1} = x_t + step_std * Z`.
///
/// The first row of every column is `start`. Steps are additive, so a path
/// can go negative; use [`gbm`] for something that stays positive.
///
/// # Errors
///
/// Returns [`Error::Invariant`] if `start` is not finite, `step_std` is
/// negative or not finite, or `n_rows * n_cols` overflows.
pub fn random_walk(
    n_rows: usize,
    n_cols: usize,
    start: f64,
    step_std: f64,
    seed: u64,
) -> Result<Vec<f64>> {
    if !start.is_finite() {
        return Err(Error::Invariant(format!(
            "start must be finite, got {start}"
        )));
    }
    if !step_std.is_finite() || step_std < 0.0 {
        return Err(Error::Invariant(format!(
            "step_std must be finite and non-negative, got {step_std}"
        )));
    }
    build_paths(n_rows, n_cols, start, |prev, row, col| {
        prev + step_std * normal(seed, row, col)
    })
}

/// Geometric Brownian motion, discretized exactly.
///
/// `S_{t+1} = S_t * exp((mu - sigma²/2) dt + sigma sqrt(dt) Z)`. This is the
/// closed-form solution over one step, not an Euler approximation, so the log
/// returns are exactly normal with mean `(mu - sigma²/2) dt` and standard
/// deviation `sigma sqrt(dt)` whatever `dt` is. With `mu = 0` the expected
/// price is flat: there is no edge to find.
///
/// The first row of every column is `start`.
///
/// # Errors
///
/// Returns [`Error::Invariant`] if `start` is not positive and finite, `mu` is
/// not finite, `sigma` is negative or not finite, `dt` is not positive and
/// finite, or `n_rows * n_cols` overflows.
pub fn gbm(
    n_rows: usize,
    n_cols: usize,
    start: f64,
    mu: f64,
    sigma: f64,
    dt: f64,
    seed: u64,
) -> Result<Vec<f64>> {
    if !start.is_finite() || start <= 0.0 {
        return Err(Error::Invariant(format!(
            "start must be positive and finite, got {start}"
        )));
    }
    if !mu.is_finite() {
        return Err(Error::Invariant(format!("mu must be finite, got {mu}")));
    }
    if !sigma.is_finite() || sigma < 0.0 {
        return Err(Error::Invariant(format!(
            "sigma must be finite and non-negative, got {sigma}"
        )));
    }
    if !dt.is_finite() || dt <= 0.0 {
        return Err(Error::Invariant(format!(
            "dt must be positive and finite, got {dt}"
        )));
    }
    let drift = (mu - 0.5 * sigma * sigma) * dt;
    let vol = sigma * dt.sqrt();
    build_paths(n_rows, n_cols, start, |prev, row, col| {
        prev * (drift + vol * normal(seed, row, col)).exp()
    })
}

/// Opens, highs and lows consistent with a close matrix.
///
/// Returns `(open, high, low)` in the same column-major layout as `close`:
///
/// - **open** is the previous bar's close in the same column; the first bar
///   opens at its own close.
/// - **high** is `max(open, close) * (1 + spread * U)`.
/// - **low** is `min(open, close) * (1 - spread * U')`.
///
/// `U` and `U'` are independent uniforms on `(0, 1]`. The extension is applied
/// to the magnitude (`hi + |hi| * spread * U`), which is the formula above for
/// positive prices and still widens the bar for negative ones, so
/// `low <= min(open, close) <= max(open, close) <= high` holds on every bar of
/// non-`NaN` input — a random walk that crosses zero included.
///
/// # Errors
///
/// Returns [`Error::Invariant`] if `close.len()` is not `n_rows * n_cols`, or
/// `spread` is not in `[0, 1)`.
pub fn ohlc_from_close(
    close: &[f64],
    n_rows: usize,
    n_cols: usize,
    spread: f64,
    seed: u64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let len = checked_len(n_rows, n_cols)?;
    if close.len() != len {
        return Err(Error::Invariant(format!(
            "close has {} values, expected n_rows ({n_rows}) * n_cols ({n_cols}) = {len}",
            close.len()
        )));
    }
    // A spread of 1 would let a positive low reach zero.
    if !(0.0..1.0).contains(&spread) {
        return Err(Error::Invariant(format!(
            "spread must be in [0, 1), got {spread}"
        )));
    }
    let mut open = vec![0.0; len];
    let mut high = vec![0.0; len];
    let mut low = vec![0.0; len];
    if n_rows == 0 {
        return Ok((open, high, low));
    }
    for col in 0..n_cols {
        let base = col * n_rows;
        for row in 0..n_rows {
            let i = base + row;
            let c = close[i];
            let o = if row == 0 { c } else { close[i - 1] };
            let hi = o.max(c);
            let lo = o.min(c);
            open[i] = o;
            high[i] = hi + hi.abs() * spread * uniform(seed, row, col, SALT_HIGH);
            low[i] = lo - lo.abs() * spread * uniform(seed, row, col, SALT_LOW);
        }
    }
    Ok((open, high, low))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits_of(a: &[f64]) -> Vec<u64> {
        a.iter().map(|x| x.to_bits()).collect()
    }

    fn mean_std(a: &[f64]) -> (f64, f64) {
        let n = a.len() as f64;
        let mean = a.iter().sum::<f64>() / n;
        let var = a.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        (mean, var.sqrt())
    }

    /// Per-bar changes within each column, `f(prev, next)`, pooled.
    fn steps(a: &[f64], n_rows: usize, f: impl Fn(f64, f64) -> f64) -> Vec<f64> {
        a.chunks_exact(n_rows)
            .flat_map(|col| col.windows(2).map(|w| f(w[0], w[1])).collect::<Vec<_>>())
            .collect()
    }

    #[test]
    fn draws_are_pinned() {
        // Recorded from the original implementation: a seed names one path,
        // so these may never change.
        let pins = [
            ((0, 0, 0, 0), 0x2382_75bc_38fc_be91, 0x3fc1_c13a_de1c_7e5e),
            (
                (0, 0, 0, SALT_NORMAL_U1),
                0xb4ad_b8f7_acd3_9cda,
                0x3fe6_95b7_1ef5_9a74,
            ),
            (
                (0, 0, 0, SALT_NORMAL_U2),
                0xb63f_23a4_b7f1_6ed4,
                0x3fe6_c7e4_7496_fe2e,
            ),
            (
                (0, 0, 0, SALT_HIGH),
                0x1efd_09d4_975b_563d,
                0x3fbe_fd09_d497_5b54,
            ),
            (
                (0, 0, 0, SALT_LOW),
                0xa360_e9d6_2c76_a900,
                0x3fe4_6c1d_3ac5_8ed6,
            ),
            ((7, 3, 4, 0), 0x7d66_a7eb_793e_ac4a, 0x3fdf_59a9_fade_4fab),
            (
                (7, 3, 4, SALT_NORMAL_U1),
                0xf8a5_afb9_9fc5_fd5a,
                0x3fef_14b5_f733_f8c0,
            ),
            (
                (7, 3, 4, SALT_NORMAL_U2),
                0x85b5_e54b_79c3_f659,
                0x3fe0_b6bc_a96f_387e,
            ),
            (
                (7, 3, 4, SALT_HIGH),
                0xd4cd_c47b_6f61_4b2a,
                0x3fea_99b8_8f6d_ec2a,
            ),
            (
                (7, 3, 4, SALT_LOW),
                0x28f7_d3df_8c35_364c,
                0x3fc4_7be9_efc6_1a9a,
            ),
            ((7, 4, 3, 0), 0xb5a9_aa4e_0d7c_cea6, 0x3fe6_b535_49c1_af9a),
            (
                (7, 4, 3, SALT_NORMAL_U1),
                0x5f91_354b_ece3_bf49,
                0x3fd7_e44d_52fb_38ef,
            ),
            (
                (7, 4, 3, SALT_NORMAL_U2),
                0x4e2b_ea0d_17b2_c141,
                0x3fd3_8afa_8345_ecb1,
            ),
            (
                (7, 4, 3, SALT_HIGH),
                0x4a8a_4105_9acc_37cb,
                0x3fd2_a290_4166_b30d,
            ),
            (
                (7, 4, 3, SALT_LOW),
                0xc9c1_15ba_6160_589f,
                0x3fe9_3822_b74c_2c0c,
            ),
            (
                (42, 1000, 17, 0),
                0x2171_2418_8479_1f91,
                0x3fc0_b892_0c42_3c8e,
            ),
            (
                (42, 1000, 17, SALT_NORMAL_U1),
                0xa306_4b2e_f967_e474,
                0x3fe4_60c9_65df_2cfc,
            ),
            (
                (42, 1000, 17, SALT_NORMAL_U2),
                0x0405_4a24_c4a8_ffeb,
                0x3f90_1528_9312_a3f0,
            ),
            (
                (42, 1000, 17, SALT_HIGH),
                0xd95c_1d11_a4af_e94f,
                0x3feb_2b83_a234_95fe,
            ),
            (
                (42, 1000, 17, SALT_LOW),
                0xe155_1c0a_a864_dac3,
                0x3fec_2aa3_8155_0c9c,
            ),
            (
                (u64::MAX, usize::MAX, 99, 0),
                0xfa44_035f_777d_fdb4,
                0x3fef_4880_6bee_efc0,
            ),
            (
                (u64::MAX, usize::MAX, 99, SALT_NORMAL_U1),
                0x9fb6_f9d0_0548_8f0a,
                0x3fe3_f6df_3a00_a912,
            ),
            (
                (u64::MAX, usize::MAX, 99, SALT_NORMAL_U2),
                0x758e_2504_2569_7f48,
                0x3fdd_6389_4109_5a5f,
            ),
            (
                (u64::MAX, usize::MAX, 99, SALT_HIGH),
                0xd3fa_1f51_0593_31ae,
                0x3fea_7f43_ea20_b266,
            ),
            (
                (u64::MAX, usize::MAX, 99, SALT_LOW),
                0xd1fb_9a47_70c8_c16a,
                0x3fea_3f73_48ee_1918,
            ),
        ];
        for ((seed, row, col, salt), want_bits, want_uniform) in pins {
            assert_eq!(
                bits(seed, row, col, salt),
                want_bits,
                "{seed} {row} {col} {salt}"
            );
            assert_eq!(uniform(seed, row, col, salt).to_bits(), want_uniform);
        }
        let normals = [
            ((0, 0, 0), 0xbfc9_5605_a1a3_c3bb),
            ((7, 3, 4), 0xbfce_995a_2d38_5388),
            ((7, 4, 3), 0xbfde_9f9a_f4f4_3498),
            ((42, 1000, 17), 0x3fee_40c3_ba23_7591),
            ((u64::MAX, usize::MAX, 99), 0xbfee_117d_5732_852c),
        ];
        for ((seed, row, col), want) in normals {
            assert_eq!(normal(seed, row, col).to_bits(), want, "{seed} {row} {col}");
        }
    }

    #[test]
    fn paths_are_pinned() {
        let walk = random_walk(3, 2, 1.0, 0.5, 42).unwrap();
        assert_eq!(
            bits_of(&walk),
            [
                0x3ff0_0000_0000_0000,
                0x3ffc_b2df_8394_5597,
                0x4000_4c8d_2c4d_a82f,
                0x3ff0_0000_0000_0000,
                0x3fd4_916a_dd03_3a46,
                0xbfdc_f9aa_e6da_b40c,
            ]
        );
        let close = gbm(3, 2, 100.0, 0.05, 0.2, 1.0 / 252.0, 42).unwrap();
        assert_eq!(
            bits_of(&close),
            [
                0x4059_0000_0000_0000,
                0x4059_820e_8cc6_7d92,
                0x4059_ab10_0d5a_4aec,
                0x4059_0000_0000_0000,
                0x4058_943d_bb8c_2ec4,
                0x4058_1b6d_fb99_eefd,
            ]
        );
        let (open, high, low) = ohlc_from_close(&close, 3, 2, 0.01, 42).unwrap();
        assert_eq!(
            bits_of(&open),
            [
                0x4059_0000_0000_0000,
                0x4059_0000_0000_0000,
                0x4059_820e_8cc6_7d92,
                0x4059_0000_0000_0000,
                0x4059_0000_0000_0000,
                0x4058_943d_bb8c_2ec4,
            ]
        );
        assert_eq!(
            bits_of(&high),
            [
                0x4059_12f6_9ddc_877f,
                0x4059_a197_1de7_14a3,
                0x4059_ceac_6b32_0b14,
                0x4059_25cd_0921_ba79,
                0x4059_0ccc_d726_36ca,
                0x4058_b0a1_844d_c763,
            ]
        );
        assert_eq!(
            bits_of(&low),
            [
                0x4058_ca08_764b_1af5,
                0x4058_cac1_b9e8_6378,
                0x4059_77f8_1b41_57cb,
                0x4058_c415_e670_cc0f,
                0x4058_8d59_a95f_3144,
                0x4058_0aba_f13a_2ab5,
            ]
        );
    }

    #[test]
    fn the_same_seed_gives_the_same_paths() {
        let a = gbm(50, 4, 100.0, 0.05, 0.2, 1.0 / 252.0, 7).unwrap();
        let b = gbm(50, 4, 100.0, 0.05, 0.2, 1.0 / 252.0, 7).unwrap();
        assert_eq!(bits_of(&a), bits_of(&b));
        let a = random_walk(50, 4, 0.0, 1.0, 7).unwrap();
        let b = random_walk(50, 4, 0.0, 1.0, 7).unwrap();
        assert_eq!(bits_of(&a), bits_of(&b));
    }

    #[test]
    fn a_different_seed_gives_different_paths() {
        let a = gbm(50, 4, 100.0, 0.05, 0.2, 1.0 / 252.0, 7).unwrap();
        let b = gbm(50, 4, 100.0, 0.05, 0.2, 1.0 / 252.0, 8).unwrap();
        assert_ne!(bits_of(&a), bits_of(&b));
    }

    #[test]
    fn a_column_does_not_depend_on_how_many_columns_were_asked_for() {
        let narrow = random_walk(30, 2, 0.0, 1.0, 3).unwrap();
        let wide = random_walk(30, 5, 0.0, 1.0, 3).unwrap();
        assert_eq!(bits_of(&narrow), bits_of(&wide[..60]));
    }

    #[test]
    fn columns_are_different_paths() {
        let n_rows = 100;
        let a = gbm(n_rows, 3, 100.0, 0.0, 0.2, 1.0, 11).unwrap();
        let cols: Vec<_> = a.chunks_exact(n_rows).map(bits_of).collect();
        assert_ne!(cols[0], cols[1]);
        assert_ne!(cols[1], cols[2]);
        assert_ne!(cols[0], cols[2]);
    }

    #[test]
    #[allow(clippy::float_cmp, reason = "the first row is copied, not computed")]
    fn the_first_row_is_start() {
        let n_rows = 10;
        for path in gbm(n_rows, 3, 42.0, 0.1, 0.3, 0.5, 1)
            .unwrap()
            .chunks_exact(n_rows)
        {
            assert_eq!(path[0], 42.0);
        }
        for path in random_walk(n_rows, 3, -5.0, 2.0, 1)
            .unwrap()
            .chunks_exact(n_rows)
        {
            assert_eq!(path[0], -5.0);
        }
    }

    #[test]
    fn gbm_log_returns_have_the_stated_mean_and_std() {
        // 200 steps x 500 paths = 100k log returns.
        let (n_rows, n_cols) = (201, 500);
        let (mu, sigma, dt) = (0.1, 0.3, 0.5);
        let s = gbm(n_rows, n_cols, 100.0, mu, sigma, dt, 2024).unwrap();
        let r = steps(&s, n_rows, |a, b| (b / a).ln());
        assert_eq!(r.len(), 100_000);
        let (mean, std) = mean_std(&r);
        let want_mean = (mu - 0.5 * sigma * sigma) * dt;
        let want_std = sigma * dt.sqrt();
        // Standard error of the mean is ~6.7e-4; allow five of them.
        assert!(
            (mean - want_mean).abs() < 5.0 * want_std / (r.len() as f64).sqrt(),
            "mean {mean} vs {want_mean}"
        );
        assert!(
            (std / want_std - 1.0).abs() < 0.015,
            "std {std} vs {want_std}"
        );
    }

    #[test]
    fn a_driftless_gbm_has_no_edge() {
        // mu = 0: the log drift is exactly -sigma²/2 dt, and the mean simple
        // return is zero.
        let (n_rows, n_cols) = (201, 500);
        let s = gbm(n_rows, n_cols, 100.0, 0.0, 0.2, 1.0, 5).unwrap();
        let r = steps(&s, n_rows, |a, b| b / a - 1.0);
        let (mean, std) = mean_std(&r);
        assert!(mean.abs() < 5.0 * std / (r.len() as f64).sqrt(), "{mean}");
    }

    #[test]
    fn random_walk_steps_have_the_stated_std() {
        let (n_rows, n_cols) = (1001, 100);
        let step_std = 2.5;
        let x = random_walk(n_rows, n_cols, 10.0, step_std, 99).unwrap();
        let d = steps(&x, n_rows, |a, b| b - a);
        assert_eq!(d.len(), 100_000);
        let (mean, std) = mean_std(&d);
        assert!(
            mean.abs() < 5.0 * step_std / (d.len() as f64).sqrt(),
            "{mean}"
        );
        assert!((std / step_std - 1.0).abs() < 0.015, "{std}");
    }

    #[test]
    fn uniforms_are_never_zero() {
        for row in 0..10_000 {
            let u = uniform(1, row, 0, SALT_NORMAL_U1);
            assert!(u > 0.0 && u <= 1.0);
            assert!(normal(1, row, 0).is_finite());
        }
    }

    #[test]
    #[allow(clippy::float_cmp, reason = "open is copied from close, not computed")]
    fn ohlc_brackets_every_bar() {
        let n_rows = 500;
        let n_cols = 4;
        // A random walk from zero crosses it, so negative closes are covered.
        for close in [
            gbm(n_rows, n_cols, 100.0, 0.0, 0.3, 1.0 / 252.0, 13).unwrap(),
            random_walk(n_rows, n_cols, 0.0, 1.0, 13).unwrap(),
        ] {
            let (open, high, low) = ohlc_from_close(&close, n_rows, n_cols, 0.01, 17).unwrap();
            for col in 0..n_cols {
                for row in 0..n_rows {
                    let i = col * n_rows + row;
                    let want_open = if row == 0 { close[i] } else { close[i - 1] };
                    assert_eq!(open[i], want_open);
                    let (o, c) = (open[i], close[i]);
                    assert!(low[i] <= o.min(c), "bar {i}");
                    assert!(o.min(c) <= o.max(c));
                    assert!(o.max(c) <= high[i], "bar {i}");
                }
            }
        }
    }

    #[test]
    #[allow(clippy::float_cmp, reason = "a zero spread adds exactly zero")]
    fn a_zero_spread_collapses_to_open_and_close() {
        let close = [1.0, 2.0, 1.5, 3.0];
        let (open, high, low) = ohlc_from_close(&close, 4, 1, 0.0, 0).unwrap();
        assert_eq!(open, vec![1.0, 1.0, 2.0, 1.5]);
        assert_eq!(high, vec![1.0, 2.0, 2.0, 3.0]);
        assert_eq!(low, vec![1.0, 1.0, 1.5, 1.5]);
    }

    #[test]
    fn ohlc_is_reproducible_and_seed_sensitive() {
        let close = gbm(100, 2, 50.0, 0.0, 0.2, 1.0, 1).unwrap();
        let a = ohlc_from_close(&close, 100, 2, 0.02, 4).unwrap();
        let b = ohlc_from_close(&close, 100, 2, 0.02, 4).unwrap();
        let c = ohlc_from_close(&close, 100, 2, 0.02, 5).unwrap();
        assert_eq!(bits_of(&a.1), bits_of(&b.1));
        assert_eq!(bits_of(&a.2), bits_of(&b.2));
        assert_ne!(bits_of(&a.1), bits_of(&c.1));
        assert_ne!(bits_of(&a.2), bits_of(&c.2));
    }

    #[test]
    fn empty_shapes_give_empty_output() {
        assert!(random_walk(0, 3, 0.0, 1.0, 0).unwrap().is_empty());
        assert!(gbm(5, 0, 1.0, 0.0, 0.2, 1.0, 0).unwrap().is_empty());
        let (o, h, l) = ohlc_from_close(&[], 0, 3, 0.1, 0).unwrap();
        assert!(o.is_empty() && h.is_empty() && l.is_empty());
    }

    #[test]
    fn bad_arguments_are_rejected() {
        let invariant = |r: Result<Vec<f64>>| matches!(r, Err(Error::Invariant(_)));
        assert!(invariant(random_walk(5, 1, f64::NAN, 1.0, 0)));
        assert!(invariant(random_walk(5, 1, 0.0, -1.0, 0)));
        assert!(invariant(random_walk(5, 1, 0.0, f64::INFINITY, 0)));
        assert!(invariant(random_walk(usize::MAX, 2, 0.0, 1.0, 0)));
        assert!(invariant(gbm(5, 1, 0.0, 0.0, 0.2, 1.0, 0)));
        assert!(invariant(gbm(5, 1, -1.0, 0.0, 0.2, 1.0, 0)));
        assert!(invariant(gbm(5, 1, 1.0, f64::NAN, 0.2, 1.0, 0)));
        assert!(invariant(gbm(5, 1, 1.0, 0.0, -0.2, 1.0, 0)));
        assert!(invariant(gbm(5, 1, 1.0, 0.0, 0.2, 0.0, 0)));
        assert!(invariant(gbm(5, 1, 1.0, 0.0, 0.2, -1.0, 0)));

        let close = [1.0, 2.0, 3.0];
        let ohlc_invariant = |n_rows, n_cols, spread| {
            matches!(
                ohlc_from_close(&close, n_rows, n_cols, spread, 0),
                Err(Error::Invariant(_))
            )
        };
        assert!(ohlc_invariant(2, 1, 0.1));
        assert!(ohlc_invariant(3, 1, -0.1));
        assert!(ohlc_invariant(3, 1, 1.0));
        assert!(ohlc_invariant(3, 1, f64::NAN));
        assert!(!ohlc_invariant(3, 1, 0.1));
    }
}
