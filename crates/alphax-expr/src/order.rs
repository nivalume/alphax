//! Trailing-window order statistics.
//!
//! `Quantile`, `Med`, `Rank` and `Mad` all need the window sorted. Re-sorting
//! every window costs `O(window · log window)` per row. Two structures avoid
//! that, chosen by window length:
//!
//! * **A sorted ring** for windows up to [`RING_MAX`] rows: the window's
//!   non-`NaN` values kept in ascending order in one short `Vec`. Entering and
//!   leaving is a binary search plus a shift of at most `RING_MAX` values,
//!   which is a few cache lines, and a quantile is an index. This is the path
//!   every Alpha158 window takes.
//! * **A Fenwick tree** for longer and expanding windows: each column is
//!   sorted once, every row gets a fixed position in that order, and the tree
//!   holds which rows are currently in the window. Entering and leaving is a
//!   point update, the k-th smallest value is a descent of the tree, and "how
//!   many values are below x" is a prefix query — all `O(log n)`.
//!
//! Both are exact orderings, so `Quantile`, `Med` and `Rank` are bit-identical
//! to sorting the window whichever path runs. `Mad` sums directly over the
//! ring; the tree version sums as offsets from the column's first value.
//!
//! `IdxMax`/`IdxMin` use a monotonic queue instead: see [`rolling_arg_extreme_into`].

/// Longest window the sorted ring handles; longer windows use the tree.
///
/// A shift of 256 values is 2 KB, about the cost of the tree's two dozen
/// scattered updates and queries, and every window in Alpha158 is at most 60.
const RING_MAX: usize = 256;

/// Which order statistic [`ordered_into`] reports.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Order {
    Quantile(f64),
    Rank,
    Mad,
}

/// Runs `stat` over every trailing window of `values` into `out`, `NaN` where
/// the window has no observation.
pub(crate) fn ordered_into(values: &[f64], window: usize, stat: Order, out: &mut [f64]) {
    let window = window.max(1);
    if window <= RING_MAX {
        ring(values, window, stat, out);
        return;
    }
    let mut tree = Ordered::new(values);
    for (row, (&value, slot)) in values.iter().zip(out).enumerate() {
        if row >= window {
            tree.update(row - window, false);
        }
        tree.update(row, true);
        *slot = if tree.count == 0 {
            f64::NAN
        } else {
            match stat {
                Order::Quantile(q) => tree.quantile(q),
                Order::Rank => tree.rank(value),
                Order::Mad => tree.mad(),
            }
        };
    }
}

/// The sorted-ring path: the window's non-`NaN` values in ascending order.
#[allow(
    clippy::float_cmp,
    reason = "a window of identical values has exactly zero deviation"
)]
fn ring(values: &[f64], window: usize, stat: Order, out: &mut [f64]) {
    let mut sorted: Vec<f64> = Vec::with_capacity(window);
    let below = |sorted: &[f64], x: f64| sorted.partition_point(|&v| v < x);
    for (row, (&value, slot)) in values.iter().zip(out).enumerate() {
        if row >= window {
            let leaving = values[row - window];
            if !leaving.is_nan() {
                // Equal values are interchangeable, so any position holding
                // this value will do; total order keeps -0.0 and 0.0 apart.
                let at = sorted.partition_point(|v| v.total_cmp(&leaving).is_lt());
                sorted.remove(at);
            }
        }
        if !value.is_nan() {
            let at = sorted.partition_point(|v| v.total_cmp(&value).is_lt());
            sorted.insert(at, value);
        }
        *slot = if sorted.is_empty() {
            f64::NAN
        } else {
            match stat {
                Order::Quantile(q) => {
                    let index = (sorted.len() - 1) as f64 * q;
                    let lower = index.floor() as usize;
                    let fraction = index - lower as f64;
                    let low = sorted[lower];
                    low + fraction * (sorted[index.ceil() as usize] - low)
                }
                Order::Rank => {
                    if value.is_nan() {
                        f64::NAN
                    } else {
                        let under = below(&sorted, value) as f64;
                        let not_above = sorted.partition_point(|&v| v <= value) as f64;
                        (under + (not_above - under + 1.0) / 2.0) / sorted.len() as f64
                    }
                }
                Order::Mad => {
                    if sorted[0] == sorted[sorted.len() - 1] {
                        0.0
                    } else {
                        let n = sorted.len() as f64;
                        let mean = sorted.iter().sum::<f64>() / n;
                        sorted.iter().map(|v| (v - mean).abs()).sum::<f64>() / n
                    }
                }
            }
        };
    }
}

struct Ordered {
    /// The column's non-`NaN` values, ascending.
    sorted: Vec<f64>,
    /// Row → position in `sorted`, `None` for `NaN`.
    position: Vec<Option<usize>>,
    /// Fenwick trees (1-based) of window membership and of shifted values.
    counts: Vec<u32>,
    sums: Vec<f64>,
    shift: f64,
    count: usize,
    total: f64,
    /// Highest power of two not above `sorted.len()`, for the descent.
    top: usize,
}

impl Ordered {
    fn new(values: &[f64]) -> Self {
        let mut rows: Vec<usize> = (0..values.len())
            .filter(|&row| !values[row].is_nan())
            .collect();
        // Ties are broken by row, so the order is total and stability is moot.
        rows.sort_unstable_by(|&a, &b| values[a].total_cmp(&values[b]).then(a.cmp(&b)));
        let mut position = vec![None; values.len()];
        for (rank, &row) in rows.iter().enumerate() {
            position[row] = Some(rank);
        }
        let sorted: Vec<f64> = rows.iter().map(|&row| values[row]).collect();
        let len = sorted.len();
        Self {
            shift: sorted.first().copied().unwrap_or(0.0),
            sorted,
            position,
            counts: vec![0; len + 1],
            sums: vec![0.0; len + 1],
            count: 0,
            total: 0.0,
            top: if len == 0 { 0 } else { 1 << len.ilog2() },
        }
    }

    fn update(&mut self, row: usize, insert: bool) {
        let Some(rank) = self.position[row] else {
            return;
        };
        let d = self.sorted[rank] - self.shift;
        let (step, d) = if insert { (1_i64, d) } else { (-1, -d) };
        self.count = self.count.wrapping_add_signed(step as isize);
        self.total += d;
        let mut i = rank + 1;
        while i < self.counts.len() {
            self.counts[i] = self.counts[i].wrapping_add_signed(step as i32);
            self.sums[i] += d;
            i += i & i.wrapping_neg();
        }
    }

    /// The `k`-th smallest value in the window, 0-based.
    fn kth(&self, k: usize) -> f64 {
        let (mut at, mut remaining) = (0, k as u32 + 1);
        let mut step = self.top;
        while step > 0 {
            let next = at + step;
            if next < self.counts.len() && self.counts[next] < remaining {
                at = next;
                remaining -= self.counts[next];
            }
            step >>= 1;
        }
        self.sorted[at]
    }

    /// Count and shifted sum of the window's values at positions `< end`.
    fn prefix(&self, end: usize) -> (usize, f64) {
        let (mut count, mut sum, mut i) = (0, 0.0, end);
        while i > 0 {
            count += self.counts[i] as usize;
            sum += self.sums[i];
            i &= i - 1;
        }
        (count, sum)
    }

    /// Linear interpolation between order statistics, as NumPy's default.
    fn quantile(&self, q: f64) -> f64 {
        let index = (self.count - 1) as f64 * q;
        let lower = index.floor() as usize;
        let fraction = index - lower as f64;
        let low = self.kth(lower);
        low + fraction * (self.kth(index.ceil() as usize) - low)
    }

    /// Percentile of `last` in the window, ties sharing their average rank.
    fn rank(&self, last: f64) -> f64 {
        if last.is_nan() {
            return f64::NAN;
        }
        let (below, _) = self.prefix(self.sorted.partition_point(|&v| v < last));
        let (not_above, _) = self.prefix(self.sorted.partition_point(|&v| v <= last));
        let equal = (not_above - below) as f64;
        (below as f64 + (equal + 1.0) / 2.0) / self.count as f64
    }

    /// Mean absolute deviation around the mean.
    #[allow(
        clippy::float_cmp,
        reason = "a window of identical values has exactly zero deviation"
    )]
    fn mad(&self) -> f64 {
        // The partial sums carry rounding residue scaled by the distance from
        // the shift; a constant window must still report an exact zero.
        if self.kth(0) == self.kth(self.count - 1) {
            return 0.0;
        }
        let n = self.count as f64;
        let mean = self.total / n;
        let (below, below_sum) =
            self.prefix(self.sorted.partition_point(|&v| v - self.shift < mean));
        let below = below as f64;
        let above_sum = self.total - below_sum;
        (mean * below - below_sum + above_sum - mean * (n - below)).max(0.0) / n
    }
}

/// Qlib's rolling `IdxMax`/`IdxMin`: the 1-based window position of the first
/// extreme value, where a `NaN` counts as more extreme than any number (so the
/// first `NaN` in the window wins), and `NaN` for an all-`NaN` window.
pub(crate) fn rolling_arg_extreme_into(
    values: &[f64],
    window: usize,
    minimum: bool,
    out: &mut [f64],
) {
    use std::collections::VecDeque;

    let window = window.max(1);
    let mut nans = VecDeque::new();
    // Rows whose value no later row beats; the front is the earliest extreme.
    let mut candidates: VecDeque<usize> = VecDeque::new();
    let mut valid = 0_usize;
    for (row, (&value, slot)) in values.iter().zip(out).enumerate() {
        if row >= window {
            let leaving = row - window;
            if values[leaving].is_nan() {
                nans.pop_front();
            } else {
                valid -= 1;
                if candidates.front() == Some(&leaving) {
                    candidates.pop_front();
                }
            }
        }
        if value.is_nan() {
            nans.push_back(row);
        } else {
            valid += 1;
            while candidates.back().is_some_and(|&back| {
                if minimum {
                    values[back] > value
                } else {
                    values[back] < value
                }
            }) {
                candidates.pop_back();
            }
            candidates.push_back(row);
        }
        let start = (row + 1).saturating_sub(window);
        *slot = if valid == 0 {
            f64::NAN
        } else {
            let best = nans
                .front()
                .or(candidates.front())
                .copied()
                .unwrap_or(start);
            (best - start + 1) as f64
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ring and the tree are two implementations of one definition, so
    /// forcing both over the same series must give the same bits.
    #[test]
    fn ring_and_tree_agree_bit_for_bit() {
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let series: Vec<f64> = (0..700)
            .map(|row| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let u = (state >> 11) as f64 / (1_u64 << 53) as f64;
                match row % 7 {
                    0 => f64::NAN,
                    1 => 0.0,
                    2 => -0.0,
                    3 => (u * 4.0).floor(),
                    _ => 100.0 + u,
                }
            })
            .collect();
        for window in [1, 2, 3, 30, 60, 255, 256] {
            for stat in [Order::Quantile(0.8), Order::Quantile(0.0), Order::Rank] {
                let mut ring_out = vec![0.0; series.len()];
                ring(&series, window, stat, &mut ring_out);
                let mut tree_out = vec![0.0; series.len()];
                let mut tree = Ordered::new(&series);
                for (row, &value) in series.iter().enumerate() {
                    if row >= window {
                        tree.update(row - window, false);
                    }
                    tree.update(row, true);
                    tree_out[row] = if tree.count == 0 {
                        f64::NAN
                    } else {
                        match stat {
                            Order::Quantile(q) => tree.quantile(q),
                            Order::Rank => tree.rank(value),
                            Order::Mad => tree.mad(),
                        }
                    };
                }
                for (row, (a, b)) in ring_out.iter().zip(&tree_out).enumerate() {
                    assert!(
                        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                        "{stat:?} window {window} row {row}: ring {a:?} tree {b:?}"
                    );
                }
            }
        }
    }
}
