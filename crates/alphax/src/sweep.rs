//! Turning a factor matrix into parameterized signal columns.
//!
//! [`SignalSweep`] owns only the derived signals and their labels. Execution
//! settings stay in [`alphax_sim::FromSignals`], so fees, stops, sizing and every
//! other simulator option still have one source of truth.

use std::collections::HashSet;

use alphax_core::columns::{Columns, Label, Level};
use alphax_core::error::{Error, Result};
use alphax_core::flex::FlexArray;
use alphax_core::matrix::Matrix;
use alphax_sim::FromSignals;

/// A factor expanded into one signal column per case and input symbol.
///
/// Columns use `case_idx * n_symbols + symbol_idx`, matching indicator sweeps
/// and the simulator's column-major unit of work.
#[derive(Debug, Clone)]
pub struct SignalSweep {
    n_rows: usize,
    n_symbols: usize,
    n_cases: usize,
    columns: Columns,
    entries: Vec<bool>,
    exits: Vec<bool>,
}

impl SignalSweep {
    /// Applies `rule` to a delayed factor value for every case and symbol.
    ///
    /// The factor must be shaped `(time, symbol)`. Non-finite values do not
    /// call `rule` and emit no signal. The first `delay` rows likewise emit no
    /// signal. Signals are deliberately not cleaned: the simulator must
    /// resolve them against the position that actually exists after fills,
    /// rejections and stops.
    ///
    /// # Errors
    ///
    /// Rejects an empty factor name, factor or case list, mismatched symbol
    /// labels, duplicate case labels and dimensions that overflow `usize`.
    pub fn from_factor<P>(
        factor: (&str, &Matrix),
        symbols: &[Label],
        cases: &[(Label, P)],
        delay: usize,
        rule: impl Fn(f64, &P) -> (bool, bool),
    ) -> Result<Self> {
        let (factor_name, factor) = factor;
        if factor_name.trim().is_empty() {
            return Err(Error::Invariant("factor name cannot be empty".into()));
        }
        if factor.n_rows() == 0 || factor.n_cols() == 0 {
            return Err(Error::Invariant("factor matrix cannot be empty".into()));
        }
        if symbols.len() != factor.n_cols() {
            return Err(Error::Invariant(format!(
                "factor has {} symbol columns, but {} symbol labels were supplied",
                factor.n_cols(),
                symbols.len()
            )));
        }
        if cases.is_empty() {
            return Err(Error::Invariant(
                "signal sweep needs at least one case".into(),
            ));
        }

        let mut seen = HashSet::with_capacity(cases.len());
        for (label, _) in cases {
            if !seen.insert(label) {
                return Err(Error::Invariant(format!(
                    "duplicate signal sweep case label {label:?}"
                )));
            }
        }

        let n_rows = factor.n_rows();
        let n_symbols = factor.n_cols();
        let n_cases = cases.len();
        let n_cols = n_cases
            .checked_mul(n_symbols)
            .ok_or_else(|| Error::Invariant("signal sweep column count overflowed".into()))?;
        let len = n_rows
            .checked_mul(n_cols)
            .ok_or_else(|| Error::Invariant("signal sweep element count overflowed".into()))?;
        let mut entries = vec![false; len];
        let mut exits = vec![false; len];

        if delay < n_rows {
            for (case_idx, (_, params)) in cases.iter().enumerate() {
                for symbol in 0..n_symbols {
                    let out_col = case_idx * n_symbols + symbol;
                    let offset = out_col * n_rows;
                    for row in delay..n_rows {
                        let score = factor.get(row - delay, symbol);
                        if score.is_finite() {
                            let (entry, exit) = rule(score, params);
                            entries[offset + row] = entry;
                            exits[offset + row] = exit;
                        }
                    }
                }
            }
        }

        let factor_label = [Label::from(factor_name)];
        let case_labels: Vec<Label> = cases.iter().map(|(label, _)| label.clone()).collect();
        let columns = Columns::new(vec![
            Level::repeat("factor", &factor_label, n_cols),
            Level::repeat("case", &case_labels, n_symbols),
            Level::tile("symbol", symbols, n_cases),
        ])?;

        Ok(Self {
            n_rows,
            n_symbols,
            n_cases,
            columns,
            entries,
            exits,
        })
    }

    /// Labels that attribute each result to its factor, case and symbol.
    #[must_use]
    pub const fn columns(&self) -> &Columns {
        &self.columns
    }

    /// Repeats a `(time, symbol)` matrix across this sweep's cases.
    ///
    /// This materializes the repeated columns. For very large sweeps a
    /// repeated-column view or chunked execution can replace this method
    /// without changing the signal layout.
    ///
    /// # Errors
    ///
    /// Returns an error when `source` does not match the factor's original
    /// `(time, symbol)` shape.
    pub fn expand(&self, source: &Matrix) -> Result<Matrix> {
        if source.n_rows() != self.n_rows || source.n_cols() != self.n_symbols {
            return Err(Error::Broadcast {
                left: (source.n_rows(), source.n_cols()),
                right: (self.n_rows, self.n_symbols),
            });
        }
        let len = self
            .n_rows
            .checked_mul(self.columns.n_cols())
            .ok_or_else(|| Error::Invariant("expanded matrix size overflowed".into()))?;
        let mut data = Vec::with_capacity(len);
        for _ in 0..self.n_cases {
            for symbol in 0..self.n_symbols {
                data.extend_from_slice(source.column(symbol));
            }
        }
        Matrix::from_column_major(data, self.n_rows, self.columns.n_cols())
    }

    /// Builds the simulator arguments backed by this sweep's signals.
    ///
    /// Set fees, sizing, stops and other execution options with struct update
    /// syntax around this value. Defaults are long-only, independent cash and
    /// no accumulation, as in [`FromSignals::default`].
    ///
    /// # Errors
    ///
    /// Returns an error unless `close` has this sweep's expanded shape.
    pub fn args<'a>(&'a self, close: &'a Matrix) -> Result<FromSignals<'a>> {
        let expected = (self.n_rows, self.columns.n_cols());
        if (close.n_rows(), close.n_cols()) != expected {
            return Err(Error::Broadcast {
                left: (close.n_rows(), close.n_cols()),
                right: expected,
            });
        }
        Ok(FromSignals {
            n_rows: self.n_rows,
            n_cols: self.columns.n_cols(),
            close: FlexArray::full(close.as_slice(), self.n_rows, self.columns.n_cols())?,
            entries: FlexArray::full(&self.entries, self.n_rows, self.columns.n_cols())?,
            exits: FlexArray::full(&self.exits, self.n_rows, self.columns.n_cols())?,
            ..FromSignals::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(columns: &[&[f64]]) -> Matrix {
        let n_rows = columns[0].len();
        Matrix::from_column_major(
            columns
                .iter()
                .flat_map(|column| column.iter().copied())
                .collect(),
            n_rows,
            columns.len(),
        )
        .unwrap()
    }

    #[test]
    fn expands_cases_symbols_and_delay_in_column_order() {
        let factor = matrix(&[&[1.0, 2.0, f64::NAN], &[10.0, 20.0, 30.0]]);
        let symbols = [Label::from("BTC"), Label::from("ETH")];
        let cases = [
            (Label::from("low"), (1.0, 0.5)),
            (Label::from("high"), (15.0, 5.0)),
        ];
        let sweep = SignalSweep::from_factor(
            ("momentum", &factor),
            &symbols,
            &cases,
            1,
            |score, &(entry, exit)| (score > entry, score < exit),
        )
        .unwrap();

        let close = matrix(&[&[100.0, 101.0, 102.0], &[200.0, 201.0, 202.0]]);
        let expanded = sweep.expand(&close).unwrap();
        assert_eq!(expanded.n_cols(), 4);
        assert_eq!(expanded.column(0), close.column(0));
        assert_eq!(expanded.column(1), close.column(1));
        assert_eq!(expanded.column(2), close.column(0));
        assert_eq!(expanded.column(3), close.column(1));

        let args = sweep.args(&expanded).unwrap();
        let entry_columns: Vec<Vec<bool>> = (0..4)
            .map(|col| (0..3).map(|row| args.entries.get(row, col)).collect())
            .collect();
        assert_eq!(entry_columns[0], [false, false, true]);
        assert_eq!(entry_columns[1], [false, true, true]);
        assert_eq!(entry_columns[2], [false, false, false]);
        assert_eq!(entry_columns[3], [false, false, true]);
        assert_eq!(
            sweep.columns().label(2, "case").unwrap(),
            &Label::from("high")
        );
        assert_eq!(
            sweep.columns().label(3, "symbol").unwrap(),
            &Label::from("ETH")
        );
    }

    #[test]
    fn rejects_invalid_setup_and_shapes() {
        let factor = matrix(&[&[1.0, 2.0]]);
        let symbols = [Label::from("BTC")];
        let cases = [(Label::from("one"), ())];
        let make = |name, labels: &[Label], cases: &[(Label, ())]| {
            SignalSweep::from_factor((name, &factor), labels, cases, 0, |_, _| (false, false))
        };
        assert!(make("", &symbols, &cases).is_err());
        assert!(make("x", &[], &cases).is_err());
        assert!(make("x", &symbols, &[]).is_err());
        let duplicate = [(Label::from("same"), ()), (Label::from("same"), ())];
        assert!(make("x", &symbols, &duplicate).is_err());

        let sweep = make("x", &symbols, &cases).unwrap();
        let wrong = matrix(&[&[1.0], &[2.0]]);
        assert!(sweep.expand(&wrong).is_err());
        assert!(sweep.args(&factor).is_ok());
        assert!(sweep.args(&wrong).is_err());
    }

    #[test]
    fn non_finite_scores_do_not_call_the_rule() {
        use std::cell::Cell;

        let factor = matrix(&[&[f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.0]]);
        let symbols = [Label::from("BTC")];
        let cases = [(Label::from("one"), ())];
        let calls = Cell::new(0);
        let sweep = SignalSweep::from_factor(("x", &factor), &symbols, &cases, 0, |score, _| {
            calls.set(calls.get() + 1);
            (score >= 1.0, false)
        })
        .unwrap();
        assert_eq!(calls.get(), 1);

        let args = sweep.args(&factor).unwrap();
        assert_eq!(
            (0..4)
                .map(|row| args.entries.get(row, 0))
                .collect::<Vec<_>>(),
            [false, false, false, true]
        );
    }
}
