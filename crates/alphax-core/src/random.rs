//! Counter-based random numbers: every draw is a pure function of its key.
//!
//! A stream generator makes each draw depend on every draw before it, so the
//! answer would change with the order columns are visited in, the number of
//! threads, or a column added in front. Here a draw is computed from
//! `(seed, row, col)`, plus a `salt` when one cell needs several independent
//! draws. Column `c` comes out the same whether it is generated alone or as
//! the thousandth of a sweep, and serial and threaded runs stay bit-identical.
//!
//! [`bits`] chains SplitMix64's [`mix`] over the key. Two maps turn those bits
//! into a uniform, and both are kept because existing results depend on each:
//!
//! - [`to_unit`] on `[0, 1)`. [`uniform`] is this map over the unsalted key:
//!   `alphax-sim` draws `reject_prob` with it and `alphax-signals` draws the
//!   random signals with it.
//! - [`to_unit_nonzero`] on `(0, 1]`, never zero, so a logarithm of it is
//!   finite. `alphax-generic` draws its synthetic paths with it, one salt per
//!   draw.
//!
//! The draws are not cryptographic and are not meant to match NumPy's stream.
//! A seed names one answer: changing anything here changes every seeded result
//! downstream, which the pinned tests in this module and in the three crates
//! above catch.

/// 2⁻⁵³, the spacing of the 53-bit grid both maps use.
const SCALE: f64 = 1.0 / (1_u64 << 53) as f64;

/// SplitMix64's output step: adds the golden-ratio increment, then
/// finalizes. A bijection on `u64` that avalanches.
#[inline]
pub const fn mix(z: u64) -> u64 {
    let mut z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// 64 random bits determined by `(seed, row, col, salt)` alone.
///
/// The key is chained through [`mix`] rather than XOR-ed together, so
/// `(row, col)` and `(col, row)` do not collide. The salt is XOR-ed into the
/// seed: salt `0` is the plain `(seed, row, col)` key, and a fixed nonzero
/// salt gives the same cell a second, independent draw.
#[inline]
pub const fn bits(seed: u64, row: usize, col: usize, salt: u64) -> u64 {
    mix(mix(mix(seed ^ salt) ^ row as u64) ^ col as u64)
}

/// A uniform on `[0, 1)` from the top 53 bits of `bits`.
///
/// Every value is a multiple of 2⁻⁵³ and exactly representable; `0` can be
/// returned, `1` cannot.
#[inline]
pub const fn to_unit(bits: u64) -> f64 {
    (bits >> 11) as f64 * SCALE
}

/// A uniform on `(0, 1]` from the top 53 bits of `bits`: never zero.
///
/// The 53-bit value is moved to the middle of its bucket (`+ 0.5`), which keeps
/// the result off `0`. Below 2⁵² the half is exact. Above it an `f64` has no
/// room for the half and the sum rounds to an even integer, so the upper half
/// of the range lands on a 2⁻⁵² grid and the top bucket rounds up to exactly
/// `1.0`, one draw in 2⁵³. Box–Muller is fine with `(0, 1]`, and the synthetic
/// paths are pinned to this map, so it stays as it is.
#[inline]
pub const fn to_unit_nonzero(bits: u64) -> f64 {
    ((bits >> 11) as f64 + 0.5) * SCALE
}

/// The uniform on `[0, 1)` for one `(seed, row, col)` cell: [`to_unit`] of
/// the unsalted [`bits`].
///
/// The simulation loops pass the execution kernel this draw for the order at
/// bar `row` of column `col` when `reject_prob > 0`, so a custom loop that
/// passes the same draw rejects the same orders.
#[inline]
pub const fn uniform(seed: u64, row: usize, col: usize) -> f64 {
    to_unit(bits(seed, row, col, 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mix_is_the_splitmix64_stream() {
        // SplitMix64 seeded with 0 outputs `mix(k * golden)` for k = 0, 1, …;
        // these are the stream's first four outputs.
        let golden = 0x9E37_79B9_7F4A_7C15_u64;
        let stream: Vec<u64> = (0..4).map(|k| mix(golden.wrapping_mul(k))).collect();
        assert_eq!(
            stream,
            [
                0xE220_A839_7B1D_CDAF,
                0x6E78_9E6A_A1B9_65F4,
                0x06C4_5D18_8009_454F,
                0xF88B_B8A8_724C_81EC,
            ]
        );
    }

    #[test]
    fn the_salt_is_folded_into_the_seed_and_the_key_is_ordered() {
        for (seed, row, col, salt) in [(0, 0, 0, 1), (7, 3, 4, 0x6E6F_726D), (u64::MAX, 9, 2, 5)] {
            assert_eq!(bits(seed, row, col, salt), bits(seed ^ salt, row, col, 0));
            assert_ne!(bits(seed, row, col, salt), bits(seed, row, col, 0));
        }
        assert_ne!(bits(7, 3, 4, 0), bits(7, 4, 3, 0));
    }

    #[test]
    fn uniform_is_pinned() {
        // The same values `alphax-sim` and `alphax-signals` pin: both draw
        // through this function.
        let pins = [
            ((0, 0, 0), 0x3fc1_c13a_de1c_7e5c),
            ((7, 3, 4), 0x3fdf_59a9_fade_4faa),
            ((7, 4, 3), 0x3fe6_b535_49c1_af99),
            ((42, 1000, 17), 0x3fc0_b892_0c42_3c8c),
            ((u64::MAX, usize::MAX, 99), 0x3fef_4880_6bee_efbf),
        ];
        for ((seed, row, col), want) in pins {
            assert_eq!(
                uniform(seed, row, col).to_bits(),
                want,
                "{seed} {row} {col}"
            );
        }
    }

    #[test]
    fn the_maps_reach_their_stated_ends() {
        let below_one = 1.0 - f64::EPSILON / 2.0;
        // [0, 1): 0 at the bottom, 1 - 2⁻⁵³ at the top; the low 11 bits are dropped.
        assert_eq!(to_unit(0).to_bits(), 0.0_f64.to_bits());
        assert_eq!(to_unit(0x7FF).to_bits(), 0.0_f64.to_bits());
        assert_eq!(to_unit(u64::MAX).to_bits(), below_one.to_bits());
        // (0, 1]: half a bucket above 0 at the bottom, rounded up to 1 at the top.
        assert_eq!(to_unit_nonzero(0).to_bits(), (SCALE / 2.0).to_bits());
        assert_eq!(to_unit_nonzero(u64::MAX).to_bits(), 1.0_f64.to_bits());
        // Above 2⁵² the half rounds to even, so the two buckets under the top
        // one share 1 - 2⁻⁵², and 1 - 2⁻⁵³ is never returned.
        let top = u64::MAX >> 11;
        for bucket in [top - 1, top - 2] {
            assert_eq!(
                to_unit_nonzero(bucket << 11).to_bits(),
                (1.0 - f64::EPSILON).to_bits()
            );
        }
    }

    #[test]
    fn draws_spread_over_the_unit_interval() {
        let n = 100_000;
        let draws: Vec<f64> = (0..n).map(|i| uniform(1, i, i % 13)).collect();
        assert!(draws.iter().all(|d| (0.0..1.0).contains(d)));
        let mean = draws.iter().sum::<f64>() / n as f64;
        assert!((mean - 0.5).abs() < 0.01, "mean {mean}");
        let below = draws.iter().filter(|d| **d < 0.1).count() as f64 / n as f64;
        assert!((below - 0.1).abs() < 0.01, "P(d < 0.1) = {below}");
    }
}
