//! Information coefficients: across symbols on each row, and along time for
//! each symbol.

use phasor_core::error::{Error, Result};

use crate::corr::{Corr, pearson};
use crate::panel::{Panel, map_chunks};
use crate::rank::Ranker;

/// Which correlation an IC is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// Pearson correlation of the values: the IC.
    Pearson,
    /// Pearson correlation of their average ranks: the rank IC.
    Spearman,
}

/// Cross-sectional ICs of several factors against several forward returns.
///
/// Both series are `n_factors * n_horizons * n_rows` long, laid out
/// `[factor][horizon][row]`: [`CsIc::series`] slices one out.
#[derive(Debug, Clone)]
pub struct CsIc {
    /// Number of factors.
    pub n_factors: usize,
    /// Number of forward-return panels (horizons).
    pub n_horizons: usize,
    /// Number of rows (bars).
    pub n_rows: usize,
    /// Pearson IC per row.
    pub pearson: Vec<f64>,
    /// Spearman (rank) IC per row.
    pub spearman: Vec<f64>,
}

impl CsIc {
    /// The `n_rows` ICs of factor `f` against horizon `h`.
    #[must_use]
    pub fn series(&self, method: Method, f: usize, h: usize) -> &[f64] {
        let all = match method {
            Method::Pearson => &self.pearson,
            Method::Spearman => &self.spearman,
        };
        let start = (f * self.n_horizons + h) * self.n_rows;
        &all[start..start + self.n_rows]
    }
}

/// Rows a thread transposes at a time: `BLOCK x n_cols` values per panel.
const BLOCK: usize = 64;

/// Average ranks of each row of a row-major tile over its finite values,
/// `NaN` where the value is missing, and how many values each row has.
#[derive(Debug, Default)]
struct TileRanks {
    ranks: Vec<f64>,
    counts: Vec<usize>,
    values: Vec<f64>,
    index: Vec<usize>,
    compact: Vec<f64>,
}

impl TileRanks {
    fn fill(&mut self, tile: &[f64], rows: usize, n_cols: usize, ranker: &mut Ranker) {
        self.ranks.clear();
        self.ranks.resize(rows * n_cols, f64::NAN);
        self.counts.clear();
        for k in 0..rows {
            let row = &tile[k * n_cols..(k + 1) * n_cols];
            self.values.clear();
            self.index.clear();
            for (i, &v) in row.iter().enumerate() {
                if v.is_finite() {
                    self.values.push(v);
                    self.index.push(i);
                }
            }
            ranker.rank(&self.values, &mut self.compact);
            let dst = &mut self.ranks[k * n_cols..(k + 1) * n_cols];
            for (&i, &r) in self.index.iter().zip(&self.compact) {
                dst[i] = r;
            }
            self.counts.push(self.values.len());
        }
    }
}

/// Everything one thread reuses from block to block.
#[derive(Debug, Default)]
struct CsScratch {
    ranker: Ranker,
    corr: Corr,
    y_tiles: Vec<Vec<f64>>,
    y_ranks: Vec<TileRanks>,
    x_tile: Vec<f64>,
    x_ranks: TileRanks,
    a: Vec<f64>,
    b: Vec<f64>,
}

/// Cross-sectional Pearson and Spearman IC of every factor against every
/// forward return, on every row.
///
/// On each row the symbols where both the factor and the forward return are
/// finite are kept; fewer than `min_names` of them (or a constant side) give
/// `NaN`. The rank IC ranks the kept symbols only, so it is exactly scipy's
/// `spearmanr` on them. A factor's ranks on a row are computed once and
/// shared by every horizon, and a horizon's once and shared by every factor;
/// they are recomputed only on rows where the two have different symbols
/// missing.
///
/// Rows are split across `threads` threads (0 = every core); the result does
/// not depend on how many.
///
/// # Errors
///
/// [`Error::Invariant`] if there are no factors or no forward returns, or
/// the panels' shapes differ.
pub fn cs_ic(
    factors: &[Panel<'_>],
    forwards: &[Panel<'_>],
    min_names: usize,
    threads: usize,
) -> Result<CsIc> {
    let (Some(first), false) = (factors.first(), forwards.is_empty()) else {
        return Err(Error::Invariant(
            "cs_ic needs at least one factor and one forward return".into(),
        ));
    };
    for p in factors.iter().skip(1) {
        first.same_shape(p, "factor")?;
    }
    for p in forwards {
        first.same_shape(p, "forward return")?;
    }
    let (n_rows, n_cols) = (first.n_rows(), first.n_cols());
    let (nf, nh) = (factors.len(), forwards.len());
    let min = min_names.max(2);

    let parts = map_chunks(n_rows, threads, |rows| {
        let len = rows.len();
        let mut pear = vec![f64::NAN; nf * nh * len];
        let mut spear = vec![f64::NAN; nf * nh * len];
        let mut s = CsScratch {
            y_tiles: vec![Vec::new(); nh],
            y_ranks: (0..nh).map(|_| TileRanks::default()).collect(),
            ..CsScratch::default()
        };
        let mut start = rows.start;
        while start < rows.end {
            let block = start..(start + BLOCK).min(rows.end);
            let b = block.len();
            for (h, fwd) in forwards.iter().enumerate() {
                fwd.gather_rows(block.clone(), &mut s.y_tiles[h]);
                s.y_ranks[h].fill(&s.y_tiles[h], b, n_cols, &mut s.ranker);
            }
            for (f, factor) in factors.iter().enumerate() {
                factor.gather_rows(block.clone(), &mut s.x_tile);
                s.x_ranks.fill(&s.x_tile, b, n_cols, &mut s.ranker);
                for k in 0..b {
                    let span = k * n_cols..(k + 1) * n_cols;
                    let x = &s.x_tile[span.clone()];
                    let rx = &s.x_ranks.ranks[span.clone()];
                    let nx = s.x_ranks.counts[k];
                    for h in 0..nh {
                        let y = &s.y_tiles[h][span.clone()];
                        let ry = &s.y_ranks[h].ranks[span.clone()];
                        let ny = s.y_ranks[h].counts[k];
                        s.corr.clear();
                        for (&xv, &yv) in x.iter().zip(y) {
                            s.corr.push(xv, yv);
                        }
                        let nj = s.corr.xs.len();
                        if nj < min {
                            continue;
                        }
                        let at = (f * nh + h) * len + (block.start - rows.start) + k;
                        pear[at] = s.corr.pearson();
                        spear[at] = if nj == nx && nj == ny {
                            // Same symbols on both sides: the full-row ranks
                            // are the ranks of the kept set.
                            s.a.clear();
                            s.b.clear();
                            for (&p, &q) in rx.iter().zip(ry) {
                                if !p.is_nan() && !q.is_nan() {
                                    s.a.push(p);
                                    s.b.push(q);
                                }
                            }
                            pearson(&s.a, &s.b)
                        } else {
                            s.corr.spearman()
                        };
                    }
                }
            }
            start = block.end;
        }
        (rows, pear, spear)
    });

    let mut pearson_out = vec![f64::NAN; nf * nh * n_rows];
    let mut spearman_out = vec![f64::NAN; nf * nh * n_rows];
    for (rows, pear, spear) in parts {
        let len = rows.len();
        for series in 0..nf * nh {
            let dst = series * n_rows + rows.start;
            pearson_out[dst..dst + len].copy_from_slice(&pear[series * len..(series + 1) * len]);
            spearman_out[dst..dst + len].copy_from_slice(&spear[series * len..(series + 1) * len]);
        }
    }
    Ok(CsIc {
        n_factors: nf,
        n_horizons: nh,
        n_rows,
        pearson: pearson_out,
        spearman: spearman_out,
    })
}

/// Per column of a column-major output, `f(column, scratch, dst)`, columns
/// split across threads. Each chunk of columns is a contiguous run of the
/// output, so the chunks are concatenated in order.
fn by_column<F>(n_cols: usize, per_col: usize, threads: usize, f: F) -> Vec<f64>
where
    F: Fn(usize, &mut Corr, &mut [f64]) + Sync,
{
    let parts = map_chunks(n_cols, threads, |cols| {
        let mut out = vec![f64::NAN; cols.len() * per_col];
        let mut corr = Corr::default();
        for (k, c) in cols.enumerate() {
            f(c, &mut corr, &mut out[k * per_col..(k + 1) * per_col]);
        }
        out
    });
    parts.concat()
}

/// The time-series IC of each symbol over the whole sample: the correlation
/// of its factor and forward return over the rows where both are finite,
/// `NaN` below `min_periods` rows or for a constant side. `n_cols` values.
///
/// # Errors
///
/// [`Error::Invariant`] if the shapes differ.
pub fn ts_ic(
    factor: &Panel<'_>,
    forward: &Panel<'_>,
    method: Method,
    min_periods: usize,
    threads: usize,
) -> Result<Vec<f64>> {
    factor.same_shape(forward, "forward return")?;
    Ok(by_column(factor.n_cols(), 1, threads, |c, corr, dst| {
        corr.clear();
        for (&x, &y) in factor.col(c).iter().zip(forward.col(c)) {
            corr.push(x, y);
        }
        dst[0] = corr.compute(method, min_periods);
    }))
}

/// The time-series IC of each symbol over a trailing window: on row `t`,
/// the correlation over rows `t + 1 - window ..= t` (fewer at the start)
/// where both are finite, `NaN` below `min_periods` of them or for a
/// constant side. Column-major, the shape of `factor`.
///
/// The window ends at `t`, so the value on row `t` uses forward returns that
/// are only known `lag + horizon` bars later: it describes the factor, it is
/// not a signal.
///
/// # Errors
///
/// [`Error::Invariant`] if the shapes differ or `window` is 0.
pub fn ts_rolling_ic(
    factor: &Panel<'_>,
    forward: &Panel<'_>,
    window: usize,
    min_periods: usize,
    method: Method,
    threads: usize,
) -> Result<Vec<f64>> {
    factor.same_shape(forward, "forward return")?;
    if window == 0 {
        return Err(Error::Invariant(
            "rolling IC window must be at least 1".into(),
        ));
    }
    let n = factor.n_rows();
    Ok(by_column(factor.n_cols(), n, threads, |c, corr, dst| {
        let (x, y) = (factor.col(c), forward.col(c));
        for (t, slot) in dst.iter_mut().enumerate() {
            let lo = (t + 1).saturating_sub(window);
            corr.clear();
            for (&a, &b) in x[lo..=t].iter().zip(&y[lo..=t]) {
                corr.push(a, b);
            }
            *slot = corr.compute(method, min_periods);
        }
    }))
}

/// The time-series IC of each symbol within each block of rows, such as a
/// calendar month: `blocks[t]` names row `t`'s block, and rows of a block
/// need not be adjacent.
///
/// Returns the distinct block ids in ascending order and a column-major
/// `(n_blocks, n_cols)` matrix of ICs, `NaN` below `min_periods` rows or for
/// a constant side.
///
/// # Errors
///
/// [`Error::Invariant`] if the shapes differ or `blocks` is not one id per
/// row.
pub fn ts_block_ic(
    factor: &Panel<'_>,
    forward: &Panel<'_>,
    blocks: &[i64],
    min_periods: usize,
    method: Method,
    threads: usize,
) -> Result<(Vec<i64>, Vec<f64>)> {
    factor.same_shape(forward, "forward return")?;
    if blocks.len() != factor.n_rows() {
        return Err(Error::Invariant(format!(
            "blocks has {} ids, expected one per row ({})",
            blocks.len(),
            factor.n_rows()
        )));
    }
    let mut order: Vec<usize> = (0..blocks.len()).collect();
    order.sort_by_key(|&t| (blocks[t], t));
    let mut ids = Vec::new();
    let mut bounds = Vec::new();
    let mut start = 0;
    while start < order.len() {
        let id = blocks[order[start]];
        let mut end = start + 1;
        while end < order.len() && blocks[order[end]] == id {
            end += 1;
        }
        ids.push(id);
        bounds.push(start..end);
        start = end;
    }
    let n_blocks = ids.len();
    let out = by_column(factor.n_cols(), n_blocks, threads, |c, corr, dst| {
        let (x, y) = (factor.col(c), forward.col(c));
        for (slot, range) in dst.iter_mut().zip(&bounds) {
            corr.clear();
            for &t in &order[range.clone()] {
                corr.push(x[t], y[t]);
            }
            *slot = corr.compute(method, min_periods);
        }
    });
    Ok((ids, out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use phasor_core::math::is_close;

    #[test]
    fn cs_ic_of_a_factor_against_itself_is_one() {
        // 2 rows x 4 cols.
        let f = [1.0, 4.0, 2.0, 3.0, 3.0, 2.0, 4.0, 1.0];
        let p = Panel::new(&f, 2, 4).unwrap();
        let neg: Vec<f64> = f.iter().map(|v| -v).collect();
        let q = Panel::new(&neg, 2, 4).unwrap();
        let ic = cs_ic(&[p], &[p, q], 2, 1).unwrap();
        for m in [Method::Pearson, Method::Spearman] {
            assert!(ic.series(m, 0, 0).iter().all(|&v| is_close(v, 1.0)));
            assert!(ic.series(m, 0, 1).iter().all(|&v| is_close(v, -1.0)));
        }
    }

    #[test]
    fn too_few_names_is_nan() {
        let f = [1.0, 2.0, f64::NAN];
        let p = Panel::new(&f, 1, 3).unwrap();
        let ic = cs_ic(&[p], &[p], 3, 1).unwrap();
        assert!(ic.pearson[0].is_nan() && ic.spearman[0].is_nan());
    }

    #[test]
    fn block_ids_are_sorted_and_rows_need_not_be_adjacent() {
        let x = [1.0, 5.0, 2.0, 4.0, 3.0, 9.0];
        let y = [1.0, 0.0, 2.0, 1.0, 3.0, -4.0];
        let (px, py) = (Panel::new(&x, 6, 1).unwrap(), Panel::new(&y, 6, 1).unwrap());
        let (ids, ic) = ts_block_ic(&px, &py, &[7, 3, 7, 3, 7, 3], 2, Method::Pearson, 1).unwrap();
        assert_eq!(ids, vec![3, 7]);
        assert!(is_close(ic[1], 1.0)); // block 7: x = 1, 2, 3; y = 1, 2, 3
        assert!(is_close(ic[0], -1.0)); // block 3: x = 5, 4, 9; y = 0, 1, -4, slope -1
    }
}
