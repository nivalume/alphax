//! Percentiles, spelled the way a compiled NumPy kernel spells them.
//!
//! Three metrics here are built on a percentile — the tail ratio and both
//! value-at-risk forms — and a percentile is an interpolation between two
//! order statistics. There are at least three ways to write that
//! interpolation, they are algebraically identical, and they round
//! differently:
//!
//! | | |
//! |---|---|
//! | `lower + m * (upper - lower)` | the textbook form |
//! | `upper - (upper - lower) * (1 - m)` | what NumPy switches to at `m >= 0.5` |
//! | `lower * (1 - m) + upper * m` | what Numba compiles `np.percentile` to |
//!
//! The third is the one this layer matches. Writing the first instead is invisible in a unit test
//! and shows up as a last-bit disagreement in the fixtures — which is how it
//! was found.
//!
//! The rank convention is the same one NumPy documents: position
//! `1 + (n - 1) * q / 100` in a 1-based ordering, interpolating between the
//! two neighbouring order statistics.

/// The `q`th percentile of `values`, with `q` in `[0, 100]`.
///
/// A single `NaN` anywhere in `values` makes the whole answer `NaN`, as it
/// does in NumPy: a percentile is a position in an ordering, and a
/// value that does not compare has no position. Callers that want the
/// percentile of the bars they *do* have filter first, which is what every
/// metric in [`crate::metrics`] does. An empty slice is `NaN` too.
///
/// The `q == 0` and `q == 100` cases bypass the interpolation entirely and
/// take the minimum or maximum. **The bypass is redundant** — at either
/// endpoint the interpolation weight is exactly zero, so the general path
/// already reduces to the same order statistic, and it reproduces
/// [`extreme_with_infinities`]'s strange-looking infinity rules by accident:
/// `inf * 0.0` is `NaN`, which is exactly where those rules say `NaN`. Checked
/// over every arrangement of infinities and finite values up to four elements;
/// mutation testing confirms removing it changes nothing. Kept because the
/// the rule is part of the definition and because "the general path happens to agree" is not
/// something a reader should have to rediscover.
#[must_use]
pub fn percentile(values: &[f64], q: f64) -> f64 {
    let n = values.len();
    if n == 0 || values.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    if n == 1 {
        // One value is every percentile of itself.
        return values[0];
    }
    #[allow(
        clippy::float_cmp,
        reason = "the rule dispatches on exactly these two values"
    )]
    let endpoint = if q == 100.0 {
        Some(true)
    } else if q == 0.0 {
        Some(false)
    } else {
        None
    };
    if let Some(want_max) = endpoint {
        return extreme_with_infinities(values, want_max);
    }

    let rank = 1.0 + (n as f64 - 1.0) * (q / 100.0);
    let f = rank.floor();
    let m = rank - f;
    // `f` is a 1-based rank, so `f - 1` is the lower of the two order
    // statistics and the upper is the next one along.
    let lo_k = (f as usize).saturating_sub(1);
    let hi_k = (lo_k + 1).min(n - 1);

    let sorted = sorted_nan_last(values);
    let lower = sorted[lo_k];
    let upper = sorted[hi_k];
    lower * (1.0 - m) + upper * m
}

/// `values` in ascending order, with every `NaN` at the end.
///
/// NumPy orders through its partition, which puts `NaN` last
/// whatever its sign bit. Rust's `total_cmp` does not: it reads the sign bit,
/// so a *negative* `NaN` sorts below negative infinity. Nothing constructs one
/// deliberately, but `0.0 * f64::NEG_INFINITY` and a negated `NaN` both
/// produce one, and a return series is full of arithmetic that could. Sorting
/// by sign here would move a missing bar from the top of the ordering to the
/// bottom — turning it from a value the tail metrics ignore into the worst one
/// they saw.
#[must_use]
pub fn sorted_nan_last(values: &[f64]) -> Vec<f64> {
    let mut out: Vec<f64> = values.to_vec();
    out.sort_by(|a, b| match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => a.partial_cmp(b).expect("neither is NaN"),
    });
    out
}

/// The minimum or maximum, with NumPy's handling of infinities.
///
/// NumPy's percentile does not simply return `-inf` for the 0th percentile of
/// a series containing one; it answers `NaN` in several arrangements, and
/// the compiled kernels reproduce that arrangement by arrangement. The rules read as
/// arbitrary because they are — they are what falls out of NumPy's own
/// interpolation when the operands are infinite, reverse-engineered into
/// special cases.
///
/// Reachable from this layer only through a cutoff of exactly 0 or 1, which
/// nothing here passes by default. Implemented anyway because the cutoff is a
/// caller's to choose and a silently different answer at the boundary is worse
/// than a strange-looking one.
#[must_use]
pub fn extreme_with_infinities(values: &[f64], want_max: bool) -> f64 {
    let n = values.len();
    let val = if want_max {
        values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    } else {
        values.iter().copied().fold(f64::INFINITY, f64::min)
    };
    if values.iter().all(|v| v.is_finite()) {
        return val;
    }
    if want_max {
        return if val.is_finite() { val } else { f64::NAN };
    }
    let pos_inf = values.iter().filter(|v| **v == f64::INFINITY).count();
    let neg_inf = values.iter().filter(|v| **v == f64::NEG_INFINITY).count();
    let finite = n - (pos_inf + neg_inf);
    let mut val = val;
    if finite == 0 {
        val = f64::NAN;
    }
    if pos_inf == 1 && n == 2 {
        val = f64::NAN;
    }
    if neg_inf > 1 {
        val = f64::NAN;
    }
    if finite == 1 && pos_inf > 1 && neg_inf != 1 {
        val = f64::NAN;
    }
    val
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "percentiles of small exact inputs land on exact values"
)]
mod tests {
    use super::*;

    #[test]
    fn a_percentile_interpolates_between_order_statistics() {
        let v = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(percentile(&v, 0.0), 1.0);
        assert_eq!(percentile(&v, 100.0), 4.0);
        assert_eq!(percentile(&v, 50.0), 2.5);
        // rank = 1 + 3 * 0.25 = 1.75, so three quarters of the way from 1 to 2.
        assert_eq!(percentile(&v, 25.0), 1.75);
    }

    #[test]
    fn the_input_need_not_be_sorted() {
        assert_eq!(percentile(&[4.0, 1.0, 3.0, 2.0], 50.0), 2.5);
    }

    #[test]
    fn one_value_is_every_percentile_of_itself() {
        assert_eq!(percentile(&[7.0], 5.0), 7.0);
        assert_eq!(percentile(&[7.0], 95.0), 7.0);
        assert!(percentile(&[], 50.0).is_nan());
    }

    #[test]
    fn the_interpolation_is_written_as_a_weighted_sum() {
        // The textbook `lower + m * (upper - lower)` is algebraically the
        // same and rounds differently. This input is one where it does, so a
        // "simplified" implementation would disagree in the last bit — and
        // the fixtures would catch it.
        let v = [-0.730_807_062_292_278_7, -0.023_791_033_075_584_5];
        let got = percentile(&v, 5.0);
        let (lower, upper) = (v[0], v[1]);
        let m: f64 = 1.0 + (v.len() as f64 - 1.0) * (5.0 / 100.0) - 1.0;
        // Spelled out rather than fused: `mul_add` rounds once where the
        // textbook form rounds twice, so it would be a third spelling again.
        let textbook = lower + m * (upper - lower);
        assert_ne!(
            got.to_bits(),
            textbook.to_bits(),
            "the two spellings agreed here, so this input no longer pins the form"
        );
        assert_eq!(got, lower * (1.0 - m) + upper * m);
    }

    #[test]
    fn a_missing_value_leaves_no_percentile() {
        // A percentile is a position in an ordering, and a value that does
        // not compare has no position. Callers that want the percentile of
        // the bars they do have filter first.
        assert!(percentile(&[1.0, f64::NAN, 3.0], 50.0).is_nan());
    }

    #[test]
    fn missing_values_sort_last_whatever_their_sign_bit() {
        // Rust's `total_cmp` reads the sign bit, so a negated NaN sorts below
        // negative infinity — which would move a missing bar from the top of
        // the ordering to the bottom and make the tail metrics treat it as
        // the worst value they saw. NumPy's partition does not do that, so
        // neither does this.
        let vals = [1.0, -f64::NAN, 2.0, f64::NAN, -5.0];
        let sorted = sorted_nan_last(&vals);
        assert_eq!(sorted[..3], [-5.0, 1.0, 2.0]);
        assert!(sorted[3].is_nan() && sorted[4].is_nan());
        // What `total_cmp` would have done instead.
        let mut naive = vals.to_vec();
        naive.sort_by(f64::total_cmp);
        assert!(naive[0].is_nan(), "the hazard this avoids is real");
    }

    #[test]
    fn an_infinite_extreme_is_not_simply_returned() {
        // The 0th percentile of a two-element series with one +inf is NaN,
        // not the finite value — one of several arrangements that are
        // special-cased. Strange, and reproduced rather than tidied.
        assert!(percentile(&[1.0, f64::INFINITY], 0.0).is_nan());
        assert_eq!(percentile(&[1.0, 2.0, f64::INFINITY], 0.0), 1.0);
        assert!(percentile(&[f64::NEG_INFINITY, f64::NEG_INFINITY], 0.0).is_nan());
        // A maximum that comes out infinite has no answer either.
        assert!(percentile(&[1.0, f64::INFINITY], 100.0).is_nan());
        assert_eq!(percentile(&[1.0, f64::NEG_INFINITY], 100.0), 1.0);
    }
}
