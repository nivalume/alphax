//! A dense column-major `f64` matrix.
//!
//! Every derived series in the engine — cash, value, returns, factor values —
//! has the same shape: one column per simulation, scanned start to finish.
//! Column-major is not a preference here but a consequence: a column *is* the
//! unit of work, so a column has to be one contiguous slice. That is what
//! makes [`Matrix::column`] free, what lets a thread take a range of columns
//! without touching another's cache lines, and what matches
//! [`FlexArray::Full`](crate::flex::FlexArray::Full).
//!
//! NumPy is row-major, so anything crossing that boundary transposes. The
//! fixture generators do it explicitly with `order="F"` for the same reason.

use crate::error::{Error, Result};

/// A dense `n_rows` × `n_cols` matrix of `f64`, stored column by column.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    data: Vec<f64>,
    n_rows: usize,
    n_cols: usize,
}

impl Matrix {
    /// A matrix filled with `value`.
    #[must_use]
    pub fn filled(n_rows: usize, n_cols: usize, value: f64) -> Self {
        Self {
            data: vec![value; n_rows * n_cols],
            n_rows,
            n_cols,
        }
    }

    /// A matrix of zeros.
    #[must_use]
    pub fn zeros(n_rows: usize, n_cols: usize) -> Self {
        Self::filled(n_rows, n_cols, 0.0)
    }

    /// Wraps column-major `data`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::FlexLen`] if `data` is not `n_rows * n_cols` long.
    pub fn from_column_major(data: Vec<f64>, n_rows: usize, n_cols: usize) -> Result<Self> {
        if data.len() != n_rows * n_cols {
            return Err(Error::FlexLen {
                kind: "matrix",
                actual: data.len(),
                expected: n_rows * n_cols,
            });
        }
        Ok(Self {
            data,
            n_rows,
            n_cols,
        })
    }

    /// Number of rows.
    #[must_use]
    pub const fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of columns.
    #[must_use]
    pub const fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// Column `col` as a contiguous slice.
    ///
    /// # Panics
    ///
    /// Panics if `col >= n_cols`.
    #[must_use]
    pub fn column(&self, col: usize) -> &[f64] {
        assert!(
            col < self.n_cols,
            "column {col} is outside 0..{}",
            self.n_cols
        );
        &self.data[col * self.n_rows..(col + 1) * self.n_rows]
    }

    /// Column `col`, mutably.
    ///
    /// # Panics
    ///
    /// Panics if `col >= n_cols`.
    pub fn column_mut(&mut self, col: usize) -> &mut [f64] {
        assert!(
            col < self.n_cols,
            "column {col} is outside 0..{}",
            self.n_cols
        );
        &mut self.data[col * self.n_rows..(col + 1) * self.n_rows]
    }

    /// The element at `(row, col)`.
    ///
    /// # Panics
    ///
    /// Panics if either index is out of bounds.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        assert!(
            row < self.n_rows && col < self.n_cols,
            "index out of bounds"
        );
        self.data[col * self.n_rows + row]
    }

    /// The whole buffer, column by column.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// The column-major buffer, without copying it.
    #[must_use]
    pub fn into_vec(self) -> Vec<f64> {
        self.data
    }

    /// Applies `f` to each column, collecting one value per column.
    #[must_use]
    pub fn map_columns(&self, f: impl Fn(&[f64]) -> f64) -> Vec<f64> {
        (0..self.n_cols).map(|c| f(self.column(c))).collect()
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "these assert on exact stored values, not on computed ones"
)]
mod tests {
    use super::*;

    #[test]
    fn a_column_is_contiguous() {
        // 3 rows x 2 cols: col0 = [1,2,3], col1 = [4,5,6].
        let m = Matrix::from_column_major(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2).unwrap();
        assert_eq!(m.column(0), [1.0, 2.0, 3.0]);
        assert_eq!(m.column(1), [4.0, 5.0, 6.0]);
        assert_eq!(m.get(2, 1), 6.0);
    }

    #[test]
    fn a_wrong_length_is_rejected() {
        assert!(Matrix::from_column_major(vec![1.0, 2.0, 3.0], 2, 2).is_err());
    }

    #[test]
    fn map_columns_reduces_per_column() {
        let m = Matrix::from_column_major(vec![1.0, 2.0, 10.0, 20.0], 2, 2).unwrap();
        assert_eq!(m.map_columns(|c| c.iter().sum()), [3.0, 30.0]);
    }
}
