//! Segmented reduction: the one shape every per-column question has.
//!
//! Records are a flat array plus a parallel `col` array. Nearly everything
//! asked of them — mean PnL per column, the row of each column's worst trade,
//! how many trades each column made, the top five by return — is the same
//! operation: cut the flat array into per-column segments, reduce each
//! segment, get one value per column back.
//!
//! Before this module every caller hand-rolled that cut. `crate::trades` had
//! a private `column_ranges`, `alphax_analysis` built a `Vec<Vec<f64>>` of
//! buckets, and the sweep code walked `col` inline. Three spellings of one
//! idea, each with its own handling of the column that produced nothing.
//!
//! # Two cuts, and the cheap one is the one we always hit
//!
//! [`ColMap`] is the general cut: a stable counting sort of record indices by
//! column, which works whatever order the records arrive in. That is the
//! fallback for any order.
//!
//! But this engine emits records in column-major order by construction — that
//! is the ordering contract in the [crate] docs, and it is what makes a
//! parallel run reproducible. When records are already grouped, a column's
//! records are a *contiguous slice*, so there is nothing to gather: the
//! reduction reads `&values[start..end]` directly. [`ColMap::build`] notices
//! this and records it, and [`MappedArray::for_each_column`] then hands out
//! subslices with no allocation at all.
//!
//! The gathering path still exists and is still correct, because records that
//! came from somewhere else — a merge, a filter, another engine —
//! need not be sorted. `the_two_paths_agree` pins that the two produce
//! identical results on the same data.
//!
//! # An empty column is a column
//!
//! Every reduction returns one value per column, including columns that
//! produced no records at all. Those get the caller's fill value and the
//! reduce function is never called on an empty slice. Dropping them instead
//! would misalign the result against the column labels, which is the one
//! mistake that turns a sweep's output into unattributable numbers.

use std::ops::Range;

use alphax_core::error::{Error, Result};
use alphax_core::matrix::Matrix;

/// Record indices grouped by column.
///
/// Built by [`ColMap::build`]. Within a column the indices keep their original
/// relative order, so a reduction sees a column's records in the order they
/// were produced — which for trade records is chronological.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColMap {
    /// Record indices, segmented by column.
    idxs: Vec<usize>,
    /// Where each column's segment starts in `idxs`.
    starts: Vec<usize>,
    /// How long each column's segment is.
    lens: Vec<usize>,
    /// Whether `idxs` is the identity, so segments are contiguous in the
    /// original array and need no gather.
    contiguous: bool,
}

impl ColMap {
    /// Groups `col_arr`'s record indices by column.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invariant`] if any entry is negative or not below
    /// `n_cols`. A record naming a column that does not exist is a bug in
    /// whatever produced it, not something to silently drop.
    pub fn build(col_arr: &[i64], n_cols: usize) -> Result<Self> {
        let mut lens = vec![0_usize; n_cols];
        let mut contiguous = true;
        let mut last = -1_i64;
        for &c in col_arr {
            if c < 0 || c as usize >= n_cols {
                return Err(Error::Invariant(format!(
                    "record names column {c}, which is outside 0..{n_cols}"
                )));
            }
            if c < last {
                contiguous = false;
            }
            last = c;
            lens[c as usize] += 1;
        }

        let mut starts = vec![0_usize; n_cols];
        let mut acc = 0_usize;
        for col in 0..n_cols {
            starts[col] = acc;
            acc += lens[col];
        }

        let mut idxs = vec![0_usize; col_arr.len()];
        let mut filled = vec![0_usize; n_cols];
        for (r, &c) in col_arr.iter().enumerate() {
            let col = c as usize;
            idxs[starts[col] + filled[col]] = r;
            filled[col] += 1;
        }

        Ok(Self {
            idxs,
            starts,
            lens,
            contiguous,
        })
    }

    /// A map over `n_cols` columns with no records at all.
    #[must_use]
    pub fn empty(n_cols: usize) -> Self {
        Self {
            idxs: Vec::new(),
            starts: vec![0; n_cols],
            lens: vec![0; n_cols],
            contiguous: true,
        }
    }

    /// Number of columns.
    #[must_use]
    pub fn n_cols(&self) -> usize {
        self.lens.len()
    }

    /// Total number of records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.idxs.len()
    }

    /// Whether there are no records.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.idxs.is_empty()
    }

    /// How many records a column has.
    #[must_use]
    pub fn col_len(&self, col: usize) -> usize {
        self.lens[col]
    }

    /// The record count of every column, in column order.
    #[must_use]
    pub fn lens(&self) -> &[usize] {
        &self.lens
    }

    /// The record indices belonging to `col`, in production order.
    #[must_use]
    pub fn column(&self, col: usize) -> &[usize] {
        let start = self.starts[col];
        &self.idxs[start..start + self.lens[col]]
    }

    /// Whether the records were already grouped by column.
    ///
    /// True means every column's records are a contiguous run in the original
    /// array, so reductions read subslices instead of gathering. Records this
    /// engine produces always satisfy this; see the module docs.
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        self.contiguous
    }

    /// Each column's half-open range in the original array.
    ///
    /// `None` when the records are not grouped by column, because then a
    /// column does not occupy one range.
    #[must_use]
    pub fn ranges(&self) -> Option<Vec<Range<usize>>> {
        self.contiguous.then(|| {
            (0..self.n_cols())
                .map(|c| self.starts[c]..self.starts[c] + self.lens[c])
                .collect()
        })
    }

    /// The longest column, or zero when there are no records.
    #[must_use]
    pub fn max_col_len(&self) -> usize {
        self.lens.iter().copied().max().unwrap_or(0)
    }
}

/// A flat value array read through a [`ColMap`].
///
/// Borrowed rather than owning, so the same values can be read under different
/// groupings and so a reduction over one record field does not copy the other
/// six.
#[derive(Debug, Clone, Copy)]
pub struct MappedArray<'a> {
    values: &'a [f64],
    col_map: &'a ColMap,
}

impl<'a> MappedArray<'a> {
    /// Pairs values with a grouping.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invariant`] if the lengths disagree.
    pub fn new(values: &'a [f64], col_map: &'a ColMap) -> Result<Self> {
        if values.len() != col_map.len() {
            return Err(Error::Invariant(format!(
                "{} values against a map of {} records",
                values.len(),
                col_map.len()
            )));
        }
        Ok(Self { values, col_map })
    }

    /// The grouping.
    #[must_use]
    pub fn col_map(&self) -> &'a ColMap {
        self.col_map
    }

    /// Number of columns.
    #[must_use]
    pub fn n_cols(&self) -> usize {
        self.col_map.n_cols()
    }

    /// Calls `f` with each non-empty column's values, in column order.
    ///
    /// On the contiguous path `f` receives a borrowed subslice; otherwise it
    /// receives a gather buffer that is reused across columns. Either way the
    /// slice is the column's values in production order, so `f` cannot tell
    /// the paths apart — which is what lets the fast path be an optimisation
    /// rather than a second semantics.
    ///
    /// Empty columns are skipped: there is nothing to reduce, and the caller's
    /// fill value already says what they mean.
    pub fn for_each_column(&self, mut f: impl FnMut(usize, &[f64])) {
        if let Some(ranges) = self.col_map.ranges() {
            for (col, range) in ranges.into_iter().enumerate() {
                if !range.is_empty() {
                    f(col, &self.values[range]);
                }
            }
            return;
        }
        let mut scratch: Vec<f64> = Vec::with_capacity(self.col_map.max_col_len());
        for col in 0..self.n_cols() {
            let idxs = self.col_map.column(col);
            if idxs.is_empty() {
                continue;
            }
            scratch.clear();
            scratch.extend(idxs.iter().map(|&i| self.values[i]));
            f(col, &scratch);
        }
    }

    /// One value per column, from a reduction over that column's values.
    ///
    /// Columns with no records get `fill` without `reduce` being called.
    #[must_use]
    pub fn reduce(&self, fill: f64, mut reduce: impl FnMut(usize, &[f64]) -> f64) -> Vec<f64> {
        let mut out = vec![fill; self.n_cols()];
        self.for_each_column(|col, values| out[col] = reduce(col, values));
        out
    }

    /// One index per column, from a reduction that picks one of the column's
    /// records.
    ///
    /// `reduce` returns a position *within the column's values*, which is
    /// translated through `idx_arr` into whatever that array holds — usually
    /// the record's row. Returning `None` leaves the column at `fill`.
    ///
    /// A reduction that cannot choose (an all-`NaN` column under `nanargmax`,
    /// say) does not raise: `None` says the same thing without unwinding, and an unmeasurable column is common
    /// enough in a sweep that it should not be exceptional.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invariant`] if `idx_arr` is not one per record, or if
    /// `reduce` picks a position outside the column.
    pub fn reduce_to_idx(
        &self,
        idx_arr: &[i64],
        fill: i64,
        mut reduce: impl FnMut(usize, &[f64]) -> Option<usize>,
    ) -> Result<Vec<i64>> {
        if idx_arr.len() != self.col_map.len() {
            return Err(Error::Invariant(format!(
                "{} indices against {} records",
                idx_arr.len(),
                self.col_map.len()
            )));
        }
        let mut out = vec![fill; self.n_cols()];
        let mut bad: Option<(usize, usize, usize)> = None;
        self.for_each_column(|col, values| {
            if let Some(pick) = reduce(col, values) {
                if pick >= values.len() {
                    bad.get_or_insert((col, pick, values.len()));
                } else {
                    out[col] = idx_arr[self.col_map.column(col)[pick]];
                }
            }
        });
        if let Some((col, pick, len)) = bad {
            return Err(Error::Invariant(format!(
                "reduction picked record {pick} of column {col}, which has {len}"
            )));
        }
        Ok(out)
    }

    /// Each column's values stacked into a dense matrix, top-aligned.
    ///
    /// The matrix is `max_col_len` rows by `n_cols` columns; a column shorter
    /// than the longest is padded with `fill` at the bottom. Row `i` is *not*
    /// a bar index — it is the `i`th record of that column, and different
    /// columns' `i`th records generally happened at different times.
    #[must_use]
    pub fn stack_expand(&self, fill: f64) -> Matrix {
        let mut out = Matrix::filled(self.col_map.max_col_len(), self.n_cols(), fill);
        self.for_each_column(|col, values| {
            out.column_mut(col)[..values.len()].copy_from_slice(values);
        });
        out
    }

    /// A per-record mask selecting each column's `n` largest values.
    ///
    /// Ties and `NaN`s are resolved from the *end* of the column, so the later
    /// of two equal values wins a contested slot. [`Self::bottom_n_mask`]
    /// resolves from the start. That asymmetry is deliberate and it is
    /// easy to miss; `the_two_ends_break_ties_in_opposite_directions` pins it.
    ///
    /// A column with fewer than `n` records is selected entirely. A column
    /// with fewer than `n` *non-`NaN`* records takes all of them and then
    /// fills the rest with `NaN` records rather than returning short.
    #[must_use]
    pub fn top_n_mask(&self, n: usize) -> Vec<bool> {
        self.extreme_n_mask(n, true)
    }

    /// A per-record mask selecting each column's `n` smallest values.
    ///
    /// The mirror of [`Self::top_n_mask`], except that ties and `NaN`s are
    /// resolved from the start of the column.
    #[must_use]
    pub fn bottom_n_mask(&self, n: usize) -> Vec<bool> {
        self.extreme_n_mask(n, false)
    }

    fn extreme_n_mask(&self, n: usize, top: bool) -> Vec<bool> {
        let mut mask = vec![false; self.col_map.len()];
        if n == 0 {
            return mask;
        }
        // The scan direction that resolves ties and NaN padding: from the end
        // for the top, from the start for the bottom.
        let order = |len: usize, top: bool| -> Vec<usize> {
            if top {
                (0..len).rev().collect()
            } else {
                (0..len).collect()
            }
        };

        for col in 0..self.n_cols() {
            let idxs = self.col_map.column(col);
            let col_len = idxs.len();
            if col_len == 0 {
                continue;
            }
            let values: Vec<f64> = idxs.iter().map(|&i| self.values[i]).collect();
            // A shortcut, and only a shortcut: at `n == col_len` the general
            // path below reaches the same all-true answer by three different
            // routes depending on how many values are measurable. Mutation
            // testing confirms weakening this to `>` changes nothing. Kept
            // because it saves a selection
            // over a column that was going to be taken whole.
            if n >= col_len {
                for &i in idxs {
                    mask[i] = true;
                }
                continue;
            }

            let mut valid: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
            if valid.is_empty() {
                // Nothing comparable: take `n` records from the tie-break end.
                for (taken, i) in order(col_len, top).into_iter().enumerate() {
                    if taken == n {
                        break;
                    }
                    mask[idxs[i]] = true;
                }
                continue;
            }
            if n > valid.len() {
                for (i, v) in values.iter().enumerate() {
                    if !v.is_nan() {
                        mask[idxs[i]] = true;
                    }
                }
                let mut remaining = n - valid.len();
                for i in order(col_len, top) {
                    if remaining == 0 {
                        break;
                    }
                    if values[i].is_nan() {
                        mask[idxs[i]] = true;
                        remaining -= 1;
                    }
                }
                continue;
            }

            // The nth largest (or smallest) comparable value, by selection
            // rather than a full sort.
            let k = if top { valid.len() - n } else { n - 1 };
            let (_, pivot, _) = valid.select_nth_unstable_by(k, f64::total_cmp);
            let pivot = *pivot;

            let mut strictly = 0_usize;
            for (i, &v) in values.iter().enumerate() {
                let beyond = if top { v > pivot } else { v < pivot };
                if !v.is_nan() && beyond {
                    strictly += 1;
                    mask[idxs[i]] = true;
                }
            }
            let mut remaining = n.saturating_sub(strictly);
            if remaining == 0 {
                continue;
            }
            for i in order(col_len, top) {
                if remaining == 0 {
                    break;
                }
                #[allow(
                    clippy::float_cmp,
                    reason = "the pivot is one of these values, so equality is exact and is \
                              precisely the tie broken here"
                )]
                let tied = !values[i].is_nan() && values[i] == pivot;
                if tied {
                    mask[idxs[i]] = true;
                    remaining -= 1;
                }
            }
        }
        mask
    }

    /// How often each code appears in each column.
    ///
    /// The result is `n_uniques` rows by `n_cols` columns: row `k` counts the
    /// records whose code is `k`. Used to tally categorical record fields —
    /// order side, trade direction, exit status — across a sweep.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invariant`] if `codes` is not one per record, or holds
    /// a code outside `0..n_uniques`.
    pub fn value_counts(&self, codes: &[i64], n_uniques: usize) -> Result<Matrix> {
        if codes.len() != self.col_map.len() {
            return Err(Error::Invariant(format!(
                "{} codes against {} records",
                codes.len(),
                self.col_map.len()
            )));
        }
        let mut out = Matrix::zeros(n_uniques, self.n_cols());
        for col in 0..self.n_cols() {
            for &r in self.col_map.column(col) {
                let code = codes[r];
                if code < 0 || code as usize >= n_uniques {
                    return Err(Error::Invariant(format!(
                        "code {code} is outside 0..{n_uniques}"
                    )));
                }
                out.column_mut(col)[code as usize] += 1.0;
            }
        }
        Ok(out)
    }

    // --- the reductions that get asked for by name ------------------------
    //
    // Every one of these skips `NaN` rather than propagating it, which is
    // a deliberate choice and worth stating because it is not the obvious one.
    // A record field can be missing for reasons that say nothing about the
    // other records — an open trade has no exit return, a position that never
    // closed has no duration — and a single such record should not erase the
    // column's statistic. `count` is the exception: it counts records, so it
    // is the thing that tells a caller how much of a column was measurable.

    /// Number of records per column, missing values included.
    ///
    /// The only reduction here that counts `NaN`s, and deliberately: paired
    /// with any of the others it says how much of the column that other
    /// statistic was actually computed from.
    #[must_use]
    pub fn count(&self) -> Vec<f64> {
        self.col_map.lens().iter().map(|&l| l as f64).collect()
    }

    /// Sum per column, skipping `NaN`.
    ///
    /// A column of nothing but `NaN` sums to `0.0`, not `NaN` — a sum of no
    /// values is zero however the values went missing, as `nansum` agrees. Reach for [`Self::count`] to tell that apart from a
    /// column that genuinely netted out flat.
    #[must_use]
    pub fn sum(&self) -> Vec<f64> {
        self.reduce(0.0, |_, v| v.iter().filter(|x| !x.is_nan()).sum())
    }

    /// Mean per column, over the non-`NaN` values only.
    ///
    /// `NaN` for an empty column and for one with nothing measurable.
    #[must_use]
    pub fn mean(&self) -> Vec<f64> {
        self.reduce(f64::NAN, |_, v| {
            let live: f64 = v.iter().filter(|x| !x.is_nan()).count() as f64;
            v.iter().filter(|x| !x.is_nan()).sum::<f64>() / live
        })
    }

    /// Smallest value per column, ignoring `NaN`.
    #[must_use]
    pub fn min(&self) -> Vec<f64> {
        self.reduce(f64::NAN, |_, v| v.iter().copied().fold(f64::NAN, f64::min))
    }

    /// Largest value per column, ignoring `NaN`.
    #[must_use]
    pub fn max(&self) -> Vec<f64> {
        self.reduce(f64::NAN, |_, v| v.iter().copied().fold(f64::NAN, f64::max))
    }

    /// Standard deviation per column, over the non-`NaN` values, with `ddof`
    /// degrees of freedom removed.
    ///
    /// `NaN` when there are no more measurable values than `ddof`.
    #[must_use]
    pub fn std(&self, ddof: usize) -> Vec<f64> {
        self.reduce(f64::NAN, |_, v| {
            let cnt = v.iter().filter(|x| !x.is_nan()).count();
            if cnt <= ddof {
                return f64::NAN;
            }
            let mean = v.iter().filter(|x| !x.is_nan()).sum::<f64>() / cnt as f64;
            let ss: f64 = v
                .iter()
                .filter(|x| !x.is_nan())
                .map(|x| (x - mean) * (x - mean))
                .sum();
            // Population variance, then
            // rescaled — rather than the algebraically equal `ss / rcount`.
            // The two differ in the last bits, because `(ss / cnt) * cnt` does
            // not have to round back to `ss`, and this layer's whole job is to
            // reproduce its recorded values bit for bit.
            let rcount = (cnt - ddof) as f64;
            ((ss / cnt as f64) * cnt as f64 / rcount).sqrt()
        })
    }

    /// The `idx_arr` entry of each column's largest value, ignoring `NaN`.
    ///
    /// A column with nothing comparable gets `fill`.
    ///
    /// # Errors
    ///
    /// As [`Self::reduce_to_idx`].
    pub fn idxmax(&self, idx_arr: &[i64], fill: i64) -> Result<Vec<i64>> {
        self.reduce_to_idx(idx_arr, fill, |_, v| arg_extreme(v, true))
    }

    /// The `idx_arr` entry of each column's smallest value, ignoring `NaN`.
    ///
    /// # Errors
    ///
    /// As [`Self::reduce_to_idx`].
    pub fn idxmin(&self, idx_arr: &[i64], fill: i64) -> Result<Vec<i64>> {
        self.reduce_to_idx(idx_arr, fill, |_, v| arg_extreme(v, false))
    }
}

/// The position of the largest (or smallest) value; ties go to the first.
///
/// `NaN` is not skipped but *substituted* — with `-inf` when looking for a
/// maximum, `+inf` for a minimum — which is not the same thing. The two agree everywhere except on a column that
/// contains both a `NaN` and a real infinity of the losing sign: there the
/// substituted `NaN` ties with the real value, and the earlier position wins.
/// Skipping would instead always pick the real one.
///
/// `None` when every value is `NaN`, where a raise would be the alternative. Nothing else
/// here unwinds on a column it cannot measure, and a sweep has too many such
/// columns for that to be exceptional.
fn arg_extreme(values: &[f64], max: bool) -> Option<usize> {
    if values.iter().all(|v| v.is_nan()) {
        return None;
    }
    let sentinel = if max {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };
    let at = |i: usize| {
        let v = values[i];
        if v.is_nan() { sentinel } else { v }
    };
    let mut best = 0_usize;
    for i in 1..values.len() {
        if (max && at(i) > at(best)) || (!max && at(i) < at(best)) {
            best = i;
        }
    }
    Some(best)
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "reductions of small exact inputs have exact expected values"
)]
mod tests {
    use super::*;

    /// Records in column-major order, as this engine produces them.
    fn sorted() -> (Vec<i64>, Vec<f64>) {
        (
            vec![0, 0, 0, 2, 2, 3],
            vec![1.0, 2.0, 3.0, 10.0, 20.0, 100.0],
        )
    }

    /// The same records, shuffled, as a merge or filter might leave them.
    fn shuffled() -> (Vec<i64>, Vec<f64>) {
        (
            vec![2, 0, 3, 0, 2, 0],
            vec![10.0, 1.0, 100.0, 2.0, 20.0, 3.0],
        )
    }

    #[test]
    fn a_column_map_segments_by_column() {
        let (col, _) = sorted();
        let map = ColMap::build(&col, 4).unwrap();
        assert_eq!(map.lens(), [3, 0, 2, 1]);
        assert_eq!(map.column(0), [0, 1, 2]);
        assert_eq!(map.column(1), [] as [usize; 0]);
        assert_eq!(map.column(2), [3, 4]);
        assert_eq!(map.column(3), [5]);
    }

    #[test]
    fn column_major_records_take_the_contiguous_path() {
        let (col, _) = sorted();
        let map = ColMap::build(&col, 4).unwrap();
        assert!(map.is_contiguous());
        assert_eq!(map.ranges().unwrap(), [0..3, 3..3, 3..5, 5..6]);
    }

    #[test]
    fn shuffled_records_do_not() {
        let (col, _) = shuffled();
        let map = ColMap::build(&col, 4).unwrap();
        assert!(!map.is_contiguous());
        assert!(map.ranges().is_none());
    }

    #[test]
    fn the_grouping_is_stable_within_a_column() {
        // Column 0's records are 1, 3, 5 in the shuffled array, and they must
        // come back in that order: a trade column is chronological, and a
        // reduction that reorders it would silently break any statistic that
        // depends on sequence.
        let (col, _) = shuffled();
        let map = ColMap::build(&col, 4).unwrap();
        assert_eq!(map.column(0), [1, 3, 5]);
        assert_eq!(map.column(2), [0, 4]);
    }

    #[test]
    fn the_two_paths_agree() {
        // The contiguous path is an optimisation, not a second semantics: the
        // same records in a different order must reduce identically.
        let (sc, sv) = sorted();
        let (hc, hv) = shuffled();
        let smap = ColMap::build(&sc, 4).unwrap();
        let hmap = ColMap::build(&hc, 4).unwrap();
        assert!(smap.is_contiguous() && !hmap.is_contiguous());

        let a = MappedArray::new(&sv, &smap).unwrap();
        let b = MappedArray::new(&hv, &hmap).unwrap();
        assert_eq!(a.sum(), b.sum());
        assert_eq!(a.count(), b.count());
        assert_eq!(
            a.stack_expand(-1.0).as_slice(),
            b.stack_expand(-1.0).as_slice()
        );
        assert_eq!(a.top_n_mask(1).len(), b.top_n_mask(1).len());
    }

    #[test]
    fn an_empty_column_gets_the_fill_and_no_call() {
        let (col, val) = sorted();
        let map = ColMap::build(&col, 4).unwrap();
        let m = MappedArray::new(&val, &map).unwrap();
        let mut called = Vec::new();
        let out = m.reduce(-999.0, |c, v| {
            called.push(c);
            assert!(!v.is_empty(), "the reduction saw an empty column");
            v.iter().sum()
        });
        assert_eq!(out, [6.0, -999.0, 30.0, 100.0]);
        assert_eq!(called, [0, 2, 3], "column 1 must not be visited");
    }

    #[test]
    fn the_named_reductions_line_up_with_the_columns() {
        let (col, val) = sorted();
        let map = ColMap::build(&col, 4).unwrap();
        let m = MappedArray::new(&val, &map).unwrap();
        assert_eq!(m.count(), [3.0, 0.0, 2.0, 1.0]);
        assert_eq!(m.sum(), [6.0, 0.0, 30.0, 100.0]);
        assert_eq!(m.mean()[0], 2.0);
        assert!(m.mean()[1].is_nan(), "an empty column has no mean");
        assert_eq!(m.min()[2], 10.0);
        assert_eq!(m.max()[2], 20.0);
        // One value and one degree of freedom removed leaves nothing.
        assert!(m.std(1)[3].is_nan());
        assert_eq!(m.std(0)[3], 0.0);
    }

    #[test]
    fn the_reductions_skip_missing_values_rather_than_propagating_them() {
        // Not the obvious reading, and getting it wrong is silent: a single
        // open trade with no exit return would otherwise blank out the whole
        // column's mean. `count` still reports three, which is how a caller
        // sees that only two of them were measurable.
        let col = [0_i64; 3];
        let val = [1.0, f64::NAN, 3.0];
        let map = ColMap::build(&col, 1).unwrap();
        let m = MappedArray::new(&val, &map).unwrap();
        assert_eq!(m.sum(), [4.0]);
        assert_eq!(m.mean(), [2.0], "divided by two, not three");
        assert_eq!(m.min(), [1.0]);
        assert_eq!(m.max(), [3.0]);
        assert_eq!(m.count(), [3.0], "count is the one that counts records");
        assert_eq!(m.std(1), [2.0_f64.sqrt()]);
    }

    #[test]
    fn a_column_of_only_missing_values_sums_to_zero_but_has_no_mean() {
        // A sum of no values is zero however the values went missing, which is
        // `nansum`. Every other reduction has no answer.
        let col = [0_i64; 2];
        let val = [f64::NAN; 2];
        let map = ColMap::build(&col, 1).unwrap();
        let m = MappedArray::new(&val, &map).unwrap();
        assert_eq!(m.sum(), [0.0]);
        assert!(m.mean()[0].is_nan());
        assert!(m.min()[0].is_nan() && m.max()[0].is_nan());
        assert!(m.std(0)[0].is_nan());
        assert_eq!(m.count(), [2.0]);
    }

    #[test]
    fn removing_more_degrees_of_freedom_than_there_are_values_has_no_answer() {
        // Found by a mutation that the fixtures could not see, because the
        // fixtures only ever call the kernels with ddof 0 or 1. At
        // ddof == cnt the guard is what turns a division by zero into NaN
        // rather than +inf, and past it the guard is the only thing standing
        // between an unsigned subtraction and an underflow panic.
        let col = [0_i64; 2];
        let val = [1.0, 3.0];
        let map = ColMap::build(&col, 1).unwrap();
        let m = MappedArray::new(&val, &map).unwrap();
        assert_eq!(m.std(1), [2.0_f64.sqrt()]);
        assert!(
            m.std(2)[0].is_nan(),
            "two values, two removed: nothing left"
        );
        assert!(m.std(5)[0].is_nan(), "and this must not panic");
    }

    #[test]
    fn a_missing_value_ties_with_a_real_infinity() {
        // An argmax substitutes -inf for a missing value rather than skipping
        // it, so the NaN at position 0 ties with the real -inf at position 1
        // and the earlier one wins. Skipping would pick position 1. The two
        // readings differ on exactly this input and nothing else.
        let col = [0_i64; 2];
        let val = [f64::NAN, f64::NEG_INFINITY];
        let map = ColMap::build(&col, 1).unwrap();
        let m = MappedArray::new(&val, &map).unwrap();
        assert_eq!(m.idxmax(&[10, 20], -1).unwrap(), [10]);
        // Looking for a minimum substitutes +inf, so there is no tie and the
        // real -inf wins outright.
        assert_eq!(m.idxmin(&[10, 20], -1).unwrap(), [20]);
    }

    #[test]
    fn std_is_computed_the_long_way_round() {
        // A population variance is rescaled instead of dividing the
        // sum of squares once, and `(ss / n) * n` need not round back to `ss`.
        // This input is one where it does not, so a "simplified" implementation
        // would disagree in the last bits — and the golden outputs would catch it.
        let val = [0.744, 0.252, 3.751];
        let col = vec![0_i64; val.len()];
        let map = ColMap::build(&col, 1).unwrap();
        let got = MappedArray::new(&val, &map).unwrap().std(1)[0];

        let n = val.len() as f64;
        let mean = val.iter().sum::<f64>() / n;
        let ss: f64 = val.iter().map(|x| (x - mean) * (x - mean)).sum();
        let simplified = (ss / (n - 1.0)).sqrt();
        assert_ne!(
            got.to_bits(),
            simplified.to_bits(),
            "the two spellings agreed here, so this input no longer pins the association"
        );
        assert_eq!(got, ((ss / n) * n / (n - 1.0)).sqrt());
    }

    #[test]
    fn an_empty_column_sums_to_zero_but_has_no_mean() {
        // Not a fussy distinction: a sum that reported NaN would poison a
        // portfolio total, and a mean that reported 0.0 would claim a column
        // broke even when it never traded.
        let map = ColMap::empty(2);
        let m = MappedArray::new(&[], &map).unwrap();
        assert_eq!(m.sum(), [0.0, 0.0]);
        assert!(m.mean().iter().all(|v| v.is_nan()));
    }

    #[test]
    fn reducing_to_an_index_goes_through_the_index_array() {
        let (col, val) = sorted();
        let map = ColMap::build(&col, 4).unwrap();
        let m = MappedArray::new(&val, &map).unwrap();
        // Rows, as a record's `idx` field would hold them.
        let rows = [5_i64, 6, 7, 1, 2, 9];
        assert_eq!(m.idxmax(&rows, -1).unwrap(), [7, -1, 2, 9]);
        assert_eq!(m.idxmin(&rows, -1).unwrap(), [5, -1, 1, 9]);
    }

    #[test]
    fn a_column_of_only_nan_has_no_extreme() {
        let col = [0_i64, 0];
        let val = [f64::NAN, f64::NAN];
        let map = ColMap::build(&col, 1).unwrap();
        let m = MappedArray::new(&val, &map).unwrap();
        assert_eq!(m.idxmax(&[4, 5], -1).unwrap(), [-1]);
    }

    #[test]
    fn a_reduction_that_picks_past_the_end_is_rejected() {
        let (col, val) = sorted();
        let map = ColMap::build(&col, 4).unwrap();
        let m = MappedArray::new(&val, &map).unwrap();
        let rows = [0_i64; 6];
        assert!(m.reduce_to_idx(&rows, -1, |_, v| Some(v.len())).is_err());
    }

    #[test]
    fn stacking_top_aligns_and_pads() {
        let (col, val) = sorted();
        let map = ColMap::build(&col, 4).unwrap();
        let m = MappedArray::new(&val, &map).unwrap();
        let stacked = m.stack_expand(-1.0);
        assert_eq!(stacked.column(0), [1.0, 2.0, 3.0]);
        assert_eq!(stacked.column(1), [-1.0, -1.0, -1.0]);
        assert_eq!(stacked.column(2), [10.0, 20.0, -1.0]);
        assert_eq!(stacked.column(3), [100.0, -1.0, -1.0]);
    }

    #[test]
    fn value_counts_tally_codes_per_column() {
        let col = [0_i64, 0, 0, 1];
        let map = ColMap::build(&col, 2).unwrap();
        let val = [0.0; 4];
        let m = MappedArray::new(&val, &map).unwrap();
        let counts = m.value_counts(&[0, 1, 1, 0], 2).unwrap();
        assert_eq!(counts.column(0), [1.0, 2.0]);
        assert_eq!(counts.column(1), [1.0, 0.0]);
        assert!(m.value_counts(&[0, 1, 1, 9], 2).is_err());
    }

    #[test]
    fn top_n_selects_the_largest() {
        let col = [0_i64; 5];
        let val = [1.0, 5.0, 3.0, 2.0, 4.0];
        let map = ColMap::build(&col, 1).unwrap();
        let m = MappedArray::new(&val, &map).unwrap();
        assert_eq!(m.top_n_mask(2), [false, true, false, false, true]);
        assert_eq!(m.bottom_n_mask(2), [true, false, false, true, false]);
    }

    #[test]
    fn asking_for_more_than_there_are_selects_everything() {
        let col = [0_i64, 0];
        let val = [1.0, 2.0];
        let map = ColMap::build(&col, 1).unwrap();
        let m = MappedArray::new(&val, &map).unwrap();
        assert_eq!(m.top_n_mask(5), [true, true]);
        assert_eq!(m.top_n_mask(0), [false, false]);
    }

    #[test]
    fn the_two_ends_break_ties_in_opposite_directions() {
        // Three equal values, one slot. The top takes the last of them and the
        // bottom takes the first. Nothing about "largest" implies that, and
        // reading either function alone would not suggest the other differs —
        // it is a convention and it is load-bearing for anyone
        // comparing selections.
        let col = [0_i64; 3];
        let val = [7.0, 7.0, 7.0];
        let map = ColMap::build(&col, 1).unwrap();
        let m = MappedArray::new(&val, &map).unwrap();
        assert_eq!(m.top_n_mask(1), [false, false, true]);
        assert_eq!(m.bottom_n_mask(1), [true, false, false]);
    }

    #[test]
    fn nan_records_are_selected_last_and_from_the_tie_break_end() {
        // Two real values and two missing ones, asking for three: both real
        // ones go in, and the third slot goes to a NaN — the last one for the
        // top, the first for the bottom.
        let col = [0_i64; 4];
        let val = [f64::NAN, 1.0, 2.0, f64::NAN];
        let map = ColMap::build(&col, 1).unwrap();
        let m = MappedArray::new(&val, &map).unwrap();
        assert_eq!(m.top_n_mask(3), [false, true, true, true]);
        assert_eq!(m.bottom_n_mask(3), [true, true, true, false]);
    }

    #[test]
    fn a_column_with_nothing_comparable_still_fills_its_slots() {
        let col = [0_i64; 3];
        let val = [f64::NAN; 3];
        let map = ColMap::build(&col, 1).unwrap();
        let m = MappedArray::new(&val, &map).unwrap();
        assert_eq!(m.top_n_mask(1), [false, false, true]);
        assert_eq!(m.bottom_n_mask(1), [true, false, false]);
    }

    #[test]
    fn columns_are_selected_independently() {
        let col = [0_i64, 0, 1, 1];
        let val = [1.0, 2.0, 30.0, 40.0];
        let map = ColMap::build(&col, 2).unwrap();
        let m = MappedArray::new(&val, &map).unwrap();
        // Column 0's best is 2.0, which loses to every value in column 1 — and
        // is selected anyway, because the selection is per column.
        assert_eq!(m.top_n_mask(1), [false, true, false, true]);
    }

    #[test]
    fn a_record_naming_a_column_that_does_not_exist_is_rejected() {
        assert!(ColMap::build(&[0, 5], 2).is_err());
        assert!(ColMap::build(&[-1], 2).is_err());
    }

    #[test]
    fn values_and_map_must_be_the_same_length() {
        let map = ColMap::build(&[0, 0], 1).unwrap();
        assert!(MappedArray::new(&[1.0], &map).is_err());
        let m = MappedArray::new(&[1.0, 2.0], &map).unwrap();
        assert!(m.reduce_to_idx(&[0], -1, |_, _| None).is_err());
        assert!(m.value_counts(&[0], 1).is_err());
    }

    #[test]
    fn no_records_at_all_still_gives_one_value_per_column() {
        let map = ColMap::build(&[], 3).unwrap();
        let m = MappedArray::new(&[], &map).unwrap();
        assert_eq!(m.count(), [0.0, 0.0, 0.0]);
        assert_eq!(m.stack_expand(0.0).as_slice().len(), 0, "no rows to stack");
        assert!(m.top_n_mask(3).is_empty());
    }
}
