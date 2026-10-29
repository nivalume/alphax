//! A borrowed column-major panel, and the one way work is split across
//! threads.

use std::ops::Range;

use phasor_core::error::{Error, Result};

/// An `(n_rows, n_cols)` matrix of `f64`, column-major, borrowed.
///
/// Column `c` is `data[c * n_rows..(c + 1) * n_rows]`: one symbol's history,
/// contiguous. A row (one bar across symbols) is strided by `n_rows`.
#[derive(Debug, Clone, Copy)]
pub struct Panel<'a> {
    data: &'a [f64],
    n_rows: usize,
    n_cols: usize,
}

impl<'a> Panel<'a> {
    /// A view of `data` as `(n_rows, n_cols)`.
    ///
    /// # Errors
    ///
    /// [`Error::Invariant`] if `data` is not `n_rows * n_cols` long.
    pub fn new(data: &'a [f64], n_rows: usize, n_cols: usize) -> Result<Self> {
        if n_rows.checked_mul(n_cols) != Some(data.len()) {
            return Err(Error::Invariant(format!(
                "panel has {} values, expected {n_rows} x {n_cols}",
                data.len()
            )));
        }
        Ok(Self {
            data,
            n_rows,
            n_cols,
        })
    }

    /// Number of rows (bars).
    #[must_use]
    pub const fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of columns (symbols).
    #[must_use]
    pub const fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// The whole buffer, column-major.
    #[must_use]
    pub const fn data(&self) -> &'a [f64] {
        self.data
    }

    /// Column `c`, contiguous.
    #[must_use]
    pub fn col(&self, c: usize) -> &'a [f64] {
        &self.data[c * self.n_rows..(c + 1) * self.n_rows]
    }

    /// Fails unless `other` has this panel's shape.
    pub(crate) fn same_shape(&self, other: &Panel<'_>, what: &str) -> Result<()> {
        if (self.n_rows, self.n_cols) == (other.n_rows, other.n_cols) {
            Ok(())
        } else {
            Err(Error::Invariant(format!(
                "{what} is {}x{}, expected {}x{}",
                other.n_rows, other.n_cols, self.n_rows, self.n_cols
            )))
        }
    }

    /// Rows `rows` of every column, copied row-major into `tile`:
    /// `tile[(r - rows.start) * n_cols + c]`. Each column is read
    /// sequentially, so a cross-sectional pass over the tile reads contiguous
    /// memory instead of striding by `n_rows`.
    pub(crate) fn gather_rows(&self, rows: Range<usize>, tile: &mut Vec<f64>) {
        let (len, n_cols) = (rows.len(), self.n_cols);
        tile.clear();
        tile.resize(len * n_cols, f64::NAN);
        for c in 0..n_cols {
            let col = &self.col(c)[rows.clone()];
            for (k, &v) in col.iter().enumerate() {
                tile[k * n_cols + c] = v;
            }
        }
    }
}

/// `threads`, or every available core when it is 0.
#[must_use]
pub fn resolve_threads(threads: usize) -> usize {
    if threads == 0 {
        std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
    } else {
        threads
    }
}

/// `0..n` cut into at most `threads` contiguous ranges whose lengths differ by
/// at most one. Depends on `(n, threads)` only.
pub(crate) fn chunks(n: usize, threads: usize) -> Vec<Range<usize>> {
    let parts = resolve_threads(threads).clamp(1, n.max(1));
    let (base, extra) = (n / parts, n % parts);
    let mut out = Vec::with_capacity(parts);
    let mut start = 0;
    for k in 0..parts {
        let len = base + usize::from(k < extra);
        out.push(start..start + len);
        start += len;
    }
    out
}

/// `f` over each of [`chunks`]`(n, threads)`, on scoped threads, results in
/// chunk order. One chunk runs on the calling thread.
pub(crate) fn map_chunks<T, F>(n: usize, threads: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(Range<usize>) -> T + Sync,
{
    let ranges = chunks(n, threads);
    if ranges.len() == 1 {
        return ranges.into_iter().map(&f).collect();
    }
    std::thread::scope(|scope| {
        let handles: Vec<_> = ranges
            .into_iter()
            .map(|range| {
                let f = &f;
                scope.spawn(move || f(range))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("a factor-evaluation thread panicked"))
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_cover_the_range_evenly() {
        assert_eq!(chunks(10, 3), vec![0..4, 4..7, 7..10]);
        assert_eq!(chunks(2, 8), vec![0..1, 1..2]);
        assert_eq!(chunks(0, 4), vec![0..0]);
    }

    #[test]
    fn gather_rows_transposes() {
        // 3 rows x 2 cols: col 0 = [0, 1, 2], col 1 = [10, 11, 12].
        let data = [0.0, 1.0, 2.0, 10.0, 11.0, 12.0];
        let p = Panel::new(&data, 3, 2).unwrap();
        let mut tile = Vec::new();
        p.gather_rows(1..3, &mut tile);
        assert_eq!(tile.len(), 4);
        assert!(
            tile.iter()
                .zip([1.0_f64, 11.0, 2.0, 12.0])
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }

    #[test]
    fn shape_is_checked() {
        assert!(Panel::new(&[1.0; 5], 2, 3).is_err());
    }
}
