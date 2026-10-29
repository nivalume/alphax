//! Average ranks, ties shared.

/// Reusable scratch for [`Ranker::rank`].
#[derive(Debug, Default)]
pub(crate) struct Ranker {
    order: Vec<usize>,
}

impl Ranker {
    /// 1-based ranks of `x` into `out`, tied values sharing the average of
    /// the ranks they span (pandas' `rank()`, scipy's `rankdata`). Every value
    /// of `x` must be finite: callers drop missing values first.
    #[allow(
        clippy::float_cmp,
        reason = "ties are defined by exact equality, as pandas' rank"
    )]
    pub(crate) fn rank(&mut self, x: &[f64], out: &mut Vec<f64>) {
        let order = &mut self.order;
        order.clear();
        order.extend(0..x.len());
        order.sort_unstable_by(|&a, &b| x[a].total_cmp(&x[b]).then(a.cmp(&b)));
        out.clear();
        out.resize(x.len(), f64::NAN);
        let mut start = 0;
        while start < order.len() {
            let mut end = start + 1;
            // `==` rather than `total_cmp`: -0.0 and 0.0 tie, as in pandas.
            while end < order.len() && x[order[end]] == x[order[start]] {
                end += 1;
            }
            let rank = (start + 1 + end) as f64 / 2.0;
            for &i in &order[start..end] {
                out[i] = rank;
            }
            start = end;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ties_share_the_average_rank() {
        let mut r = Ranker::default();
        let mut out = Vec::new();
        r.rank(&[3.0, 1.0, 3.0, 2.0, -0.0, 0.0], &mut out);
        // sorted: -0.0, 0.0 (ranks 1, 2 -> 1.5), 1 (3), 2 (4), 3, 3 (5, 6 -> 5.5)
        assert_eq!(
            out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            [5.5, 3.0, 5.5, 4.0, 1.5, 1.5].map(f64::to_bits)
        );
    }
}
