//! Zero-allocation broadcasting.
//!
//! Almost every simulation parameter — price, fees, slippage, stop distance,
//! size — can be given as a single scalar, a value per symbol, a value per
//! bar, or a full matrix. Materializing each of those to `(n_rows, n_cols)`
//! would dominate memory once a parameter sweep pushes the column count into
//! the thousands: a 100k-bar × 2000-column `f64` matrix is 1.6 GB, and there
//! are a dozen such parameters.
//!
//! [`FlexArray`] keeps the original shape and does the indexing instead. All
//! four cases read through one branch-predictable [`FlexArray::get`], and the
//! scalar case — by far the most common — touches no memory at all.
//!
//! # Layout
//!
//! [`FlexArray::Full`] is **column-major**: `data[col * n_rows + row]`. Every
//! column is an independent simulation that is scanned start-to-finish, so
//! column-major is the layout that keeps that scan sequential.

use crate::error::{Error, Result};

/// A value that broadcasts to `(n_rows, n_cols)` without being materialized.
///
/// Construct through the checked constructors rather than the variants
/// directly, so length mismatches surface at setup instead of as a silent
/// out-of-bounds read deep in a simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlexArray<'a, T> {
    /// One value for every `(row, col)`.
    Scalar(T),
    /// One value per column, constant down the rows. Length is `n_cols`.
    PerColumn(&'a [T]),
    /// One value per row, constant across the columns. Length is `n_rows`.
    PerRow(&'a [T]),
    /// Logical columns point into a shared source matrix.
    MappedColumns {
        data: &'a [T],
        n_rows: usize,
        columns: &'a [usize],
    },
    /// A full column-major matrix: `data[col * n_rows + row]`.
    Full {
        /// Column-major elements, length `n_rows * n_cols`.
        data: &'a [T],
        /// Number of rows.
        n_rows: usize,
    },
}

impl<T: Copy> FlexArray<'_, T> {
    /// Reads the value at `(row, col)`.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of range for the underlying storage. The
    /// checked constructors make this unreachable for indices inside the
    /// shape the array was built against.
    #[inline(always)]
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> T {
        // A scalar is most parameters of most runs: answered where it is
        // read, without a call. The simulation loops read five or more per
        // bar and column, and the call cost more than the read.
        if let Self::Scalar(v) = *self {
            return v;
        }
        self.get_array(row, col)
    }

    /// [`get`](Self::get) for every layout but a scalar.
    #[inline]
    fn get_array(&self, row: usize, col: usize) -> T {
        match *self {
            Self::Scalar(v) => v,
            Self::PerColumn(v) => v[col],
            Self::PerRow(v) => v[row],
            Self::MappedColumns {
                data,
                n_rows,
                columns,
            } => data[columns[col] * n_rows + row],
            Self::Full { data, n_rows } => data[col * n_rows + row],
        }
    }

    /// Whether this array varies along the row axis.
    ///
    /// A caller that only needs per-column constants can hoist the read out
    /// of its inner loop when this is `false`.
    #[must_use]
    pub const fn varies_by_row(&self) -> bool {
        matches!(
            self,
            Self::PerRow(_) | Self::Full { .. } | Self::MappedColumns { .. }
        )
    }
}

impl<'a, T> FlexArray<'a, T> {
    /// Map logical columns to source columns without expanding the matrix.
    pub fn mapped(
        data: &'a [T],
        n_rows: usize,
        source_cols: usize,
        columns: &'a [usize],
    ) -> Result<Self> {
        if n_rows.checked_mul(source_cols) != Some(data.len())
            || columns.is_empty()
            || columns.iter().any(|&c| c >= source_cols)
        {
            return Err(Error::Invariant(
                "column_map must be nonempty and index a correctly shaped source matrix".into(),
            ));
        }
        Ok(Self::MappedColumns {
            data,
            n_rows,
            columns,
        })
    }

    /// A value that is constant everywhere.
    #[must_use]
    pub const fn scalar(value: T) -> Self {
        Self::Scalar(value)
    }

    /// One value per column, checked against `n_cols`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::FlexLen`] if `values.len() != n_cols`.
    pub fn per_column(values: &'a [T], n_cols: usize) -> Result<Self> {
        if values.len() == n_cols {
            Ok(Self::PerColumn(values))
        } else {
            Err(Error::FlexLen {
                kind: "per-column",
                actual: values.len(),
                expected: n_cols,
            })
        }
    }

    /// One value per row, checked against `n_rows`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::FlexLen`] if `values.len() != n_rows`.
    pub fn per_row(values: &'a [T], n_rows: usize) -> Result<Self> {
        if values.len() == n_rows {
            Ok(Self::PerRow(values))
        } else {
            Err(Error::FlexLen {
                kind: "per-row",
                actual: values.len(),
                expected: n_rows,
            })
        }
    }

    /// A full column-major matrix, checked against `n_rows * n_cols`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::FlexLen`] if the element count does not match.
    pub fn full(data: &'a [T], n_rows: usize, n_cols: usize) -> Result<Self> {
        let expected = n_rows * n_cols;
        if data.len() == expected {
            Ok(Self::Full { data, n_rows })
        } else {
            Err(Error::FlexLen {
                kind: "full",
                actual: data.len(),
                expected,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapped_columns_share_source_and_validate_bounds() {
        let data = [1, 2, 10, 20];
        let map = [1, 0, 1];
        let a = FlexArray::mapped(&data, 2, 2, &map).unwrap();
        assert_eq!((a.get(0, 0), a.get(1, 1), a.get(1, 2)), (10, 2, 20));
        assert!(a.varies_by_row());
        assert!(FlexArray::mapped(&data, 2, 2, &[2]).is_err());
        assert!(FlexArray::mapped(&data, 2, 2, &[]).is_err());
    }

    #[test]
    fn scalar_is_constant_everywhere() {
        let a = FlexArray::scalar(7_i32);
        assert_eq!(a.get(0, 0), 7);
        assert_eq!(a.get(99, 99), 7);
        assert!(!a.varies_by_row());
    }

    #[test]
    fn per_column_is_constant_down_rows() {
        let v = [10, 20, 30];
        let a = FlexArray::per_column(&v, 3).unwrap();
        assert_eq!(a.get(0, 0), 10);
        assert_eq!(a.get(5, 0), 10);
        assert_eq!(a.get(5, 2), 30);
        assert!(!a.varies_by_row());
    }

    #[test]
    fn per_row_is_constant_across_columns() {
        let v = [10, 20, 30];
        let a = FlexArray::per_row(&v, 3).unwrap();
        assert_eq!(a.get(0, 0), 10);
        assert_eq!(a.get(0, 5), 10);
        assert_eq!(a.get(2, 5), 30);
        assert!(a.varies_by_row());
    }

    #[test]
    fn full_is_column_major() {
        // 2 rows x 3 cols, laid out column by column.
        //   col0 = [1, 2], col1 = [3, 4], col2 = [5, 6]
        let data = [1, 2, 3, 4, 5, 6];
        let a = FlexArray::full(&data, 2, 3).unwrap();
        assert_eq!(a.get(0, 0), 1);
        assert_eq!(a.get(1, 0), 2);
        assert_eq!(a.get(0, 1), 3);
        assert_eq!(a.get(1, 2), 6);
        assert!(a.varies_by_row());
    }

    #[test]
    fn constructors_reject_length_mismatch() {
        let v = [1, 2, 3];
        assert!(FlexArray::per_column(&v, 4).is_err());
        assert!(FlexArray::per_row(&v, 2).is_err());
        assert!(FlexArray::full(&v, 2, 2).is_err());
        // And accept the exact fit.
        assert!(FlexArray::full(&v, 3, 1).is_ok());
        assert!(FlexArray::full(&v, 1, 3).is_ok());
    }
}
