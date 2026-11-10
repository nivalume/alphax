//! Float tolerance layer.
//!
//! Every monetary comparison in the engine goes through this module. It is
//! ~50 lines and it is the single most load-bearing piece of the engine: the
//! simulator snaps `cash`, `position`, `debt`, `free_cash`, `val_price` and
//! `value` to exactly zero via [`is_close`] before it does anything else, and
//! the buy/sell paths decide affordability with [`is_close_or_less`] rather
//! than `<=`. Keep the arithmetic but not these tolerances and fills
//! diverge in ways that only show up as off-by-one-cent drift thousands of
//! bars later.
//!
//! The semantics are the [PEP 485] `math.isclose` formulation with one
//! deliberate deviation: **infinities are never close, not even to themselves**. The
//! `is_infinite` check precedes the `a == b` short circuit. That matters
//! because `f64::INFINITY` is a live sentinel in this engine (it means
//! "size: all of it" and "price: use the close"), so `is_close(inf, inf)`
//! returning `true` would silently conflate two different sentinels.
//!
//! [PEP 485]: https://peps.python.org/pep-0485/

/// Default relative tolerance: `1e-9`.
///
/// At this tolerance `1_000_000_000.0` and `1_000_000_001.0` compare equal.
pub const REL_TOL: f64 = 1e-9;

/// Default absolute tolerance: `1e-12`.
///
/// At this tolerance `0.000_000_000_001` and `0.000_000_000_002` compare
/// equal. This floor is what lets quantities decay to a clean zero instead of
/// leaving a denormal residue that keeps a position nominally open.
pub const ABS_TOL: f64 = 1e-12;

/// Whether two values are approximately equal, at the default tolerances.
///
/// `NaN` is never close to anything, including itself. Neither is any
/// infinity — see the module docs for why.
#[inline]
#[must_use]
pub fn is_close(a: f64, b: f64) -> bool {
    is_close_tol(a, b, REL_TOL, ABS_TOL)
}

/// Whether two values are approximately equal, at explicit tolerances.
#[inline]
#[must_use]
#[expect(
    clippy::float_cmp,
    reason = "the exact-equality short circuit is part of the isclose contract; \
              this function is the tolerance layer everything else defers to"
)]
pub fn is_close_tol(a: f64, b: f64, rel_tol: f64, abs_tol: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return false;
    }
    // Deliberately before the equality check: infinities are sentinels here.
    if a.is_infinite() || b.is_infinite() {
        return false;
    }
    if a == b {
        return true;
    }
    (a - b).abs() <= f64::max(rel_tol * f64::max(a.abs(), b.abs()), abs_tol)
}

/// Whether `a` is approximately less than or equal to `b`.
///
/// This is the affordability test in the buy/sell paths. Using `<=` instead
/// rejects orders that are payable to within a rounding error.
#[inline]
#[must_use]
pub fn is_close_or_less(a: f64, b: f64) -> bool {
    is_close(a, b) || a < b
}

/// Whether `a` is approximately strictly less than `b`.
///
/// Note this is *not* `!is_close_or_less(b, a)`: values within tolerance are
/// neither less nor greater, they are equal.
#[inline]
#[must_use]
pub fn is_less(a: f64, b: f64) -> bool {
    !is_close(a, b) && a < b
}

/// Whether `a + b` should be treated as exactly zero.
///
/// Opposite signs are compared by magnitude rather than by summing, because
/// `a + b` for two nearly-cancelling large values loses most of its
/// significant digits — the sum's distance from zero is then meaningless at
/// a relative tolerance.
///
/// Inlining is forced because every balance update runs this. Once `buy` and
/// `sell` were inlined into the matching kernel, the compiler stopped inlining
/// this into it, and the calls cost both simulation loops about 1.3% more
/// instructions (`benchmarks/instructions.py`).
#[inline(always)]
#[must_use]
pub fn is_addition_zero(a: f64, b: f64) -> bool {
    if sign(a) != sign(b) {
        is_close(a.abs(), b.abs())
    } else {
        is_close(a + b, 0.0)
    }
}

/// Add two floats, snapping a cancelling sum to exactly `0.0`.
///
/// Use this for every running-balance update (`cash`, `position`, `debt`,
/// `free_cash`). Plain `+` leaves residue that keeps closed positions
/// nominally open.
#[inline]
#[must_use]
pub fn add(a: f64, b: f64) -> f64 {
    if is_addition_zero(a, b) { 0.0 } else { a + b }
}

/// Three-way sign, as an integer so the comparison above is exact.
///
/// Deliberately not `f64::signum`, which maps `0.0` to `1.0` and `-0.0` to
/// `-1.0`; that would send [`is_addition_zero`] down the opposite-signs
/// branch for a zero operand. Here both zeros are `0`.
///
/// `NaN` gets its own value rather than propagating as `NaN`. The branch it
/// selects differs from the propagating version, but the outcome does not:
/// whichever branch is taken, [`is_close`] rejects anything involving `NaN`,
/// so [`is_addition_zero`] is `false` either way. See the test below.
#[inline]
#[must_use]
const fn sign(x: f64) -> i8 {
    if x.is_nan() {
        i8::MIN
    } else if x > 0.0 {
        1
    } else if x < 0.0 {
        -1
    } else {
        0
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "tests assert on exact snapped values (0.0, 2.0), which is the point"
)]
mod tests {
    use super::*;

    #[test]
    fn nan_is_close_to_nothing() {
        assert!(!is_close(f64::NAN, f64::NAN));
        assert!(!is_close(f64::NAN, 1.0));
        assert!(!is_close(1.0, f64::NAN));
    }

    #[test]
    fn infinities_are_never_close_not_even_to_themselves() {
        // The deliberate deviation from PEP 485. See module docs.
        assert!(!is_close(f64::INFINITY, f64::INFINITY));
        assert!(!is_close(f64::NEG_INFINITY, f64::NEG_INFINITY));
        assert!(!is_close(f64::INFINITY, f64::NEG_INFINITY));
        assert!(!is_close(f64::INFINITY, 1.0));
    }

    #[test]
    fn identical_values_are_close() {
        assert!(is_close(0.0, 0.0));
        assert!(is_close(1.5, 1.5));
        assert!(is_close(-1.5, -1.5));
        assert!(is_close(0.0, -0.0));
    }

    #[test]
    fn relative_tolerance_scales_with_magnitude() {
        // The documented example: 1e9 == 1e9 + 1 at rel_tol 1e-9.
        assert!(is_close(1_000_000_000.0, 1_000_000_001.0));
        // One more unit of error and it no longer holds.
        assert!(!is_close(1_000_000_000.0, 1_000_000_003.0));
    }

    #[test]
    fn absolute_tolerance_floors_near_zero() {
        // The documented example: 1e-12 == 2e-12 via abs_tol, which relative
        // tolerance alone would reject (they differ by 50%).
        assert!(is_close(0.000_000_000_001, 0.000_000_000_002));
        assert!(!is_close(0.000_000_01, 0.000_000_02));
    }

    #[test]
    fn ordering_helpers_treat_near_equal_as_equal() {
        let a = 1_000_000_000.0;
        let b = 1_000_000_001.0;
        assert!(is_close(a, b));
        // Within tolerance: "or less" holds, strict "less" does not.
        assert!(is_close_or_less(a, b));
        assert!(!is_less(a, b));
        // And it is symmetric for the equal case.
        assert!(is_close_or_less(b, a));
        assert!(!is_less(b, a));
    }

    #[test]
    fn ordering_helpers_outside_tolerance() {
        assert!(is_less(1.0, 2.0));
        assert!(is_close_or_less(1.0, 2.0));
        assert!(!is_less(2.0, 1.0));
        assert!(!is_close_or_less(2.0, 1.0));
    }

    #[test]
    fn addition_snaps_exact_cancellation_to_zero() {
        assert_eq!(add(1.5, -1.5), 0.0);
        assert_eq!(add(-1.5, 1.5), 0.0);
        assert_eq!(add(0.0, 0.0), 0.0);
    }

    #[test]
    fn addition_snaps_near_cancellation_to_zero() {
        // Opposite signs go through the magnitude comparison, so a residue
        // that is tiny *relative to the operands* still snaps.
        assert_eq!(add(1_000_000_000.0, -1_000_000_001.0), 0.0);
    }

    #[test]
    fn addition_preserves_a_real_difference() {
        assert_eq!(add(3.0, -1.0), 2.0);
        assert_eq!(add(1.0, 1.0), 2.0);
    }

    #[test]
    fn addition_zero_uses_magnitude_for_opposite_signs() {
        assert!(is_addition_zero(5.0, -5.0));
        assert!(!is_addition_zero(5.0, -4.0));
        // Same sign: only a genuine sum-to-zero counts, so two positives
        // never cancel.
        assert!(!is_addition_zero(5.0, 5.0));
        assert!(is_addition_zero(0.0, 0.0));
    }

    #[test]
    fn sign_treats_both_zeros_as_zero() {
        // f64::signum would give 1.0 and -1.0 here and flip the branch taken
        // by is_addition_zero.
        assert_eq!(sign(0.0), 0);
        assert_eq!(sign(-0.0), 0);
        assert_eq!(sign(3.0), 1);
        assert_eq!(sign(-3.0), -1);
    }

    #[test]
    fn nan_never_makes_an_addition_zero_whichever_branch_is_taken() {
        // Documents the deviation noted on `sign`: giving NaN its own sign
        // value changes which branch runs but not the answer.
        assert!(!is_addition_zero(f64::NAN, f64::NAN));
        assert!(!is_addition_zero(f64::NAN, 5.0));
        assert!(!is_addition_zero(5.0, f64::NAN));
        assert!(add(f64::NAN, 5.0).is_nan());
        assert!(add(f64::NAN, f64::NAN).is_nan());
    }
}
