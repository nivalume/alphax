//! Reproducible random draws for `reject_prob`.
//!
//! The execution kernel never draws a random number itself: the caller passes
//! the draw in. A shared generator advanced by
//! the loop would make each draw depend on how many came before it, which
//! depends on the grouping and the thread count. A draw computed from
//! `(seed, row, col)` alone depends on neither, so a run is reproducible and
//! its parallel and serial results stay identical.

/// A draw from `[0, 1)` for one order: column `col`, bar `row`.
///
/// [`alphax_core::random::uniform`], which a custom loop can call to reject
/// the same orders. Kept out of line: the loops reach it only when
/// `reject_prob > 0`, so it does not belong in their hot path.
#[inline(never)]
pub(crate) fn uniform(seed: u64, row: usize, col: usize) -> f64 {
    alphax_core::random::uniform(seed, row, col)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draws_are_pinned() {
        // Recorded from the original implementation; any change here changes
        // which orders every `reject_prob` run rejects.
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
}
