//! Turning parameter values into a column axis.
//!
//! This is the piece that makes a parameter sweep one call instead of a loop.
//! An indicator run over `windows = [10, 20, 30]` against four symbols does
//! not produce three results of four columns; it produces **one result of
//! twelve columns**, which the simulator then treats exactly like twelve
//! independent symbols. Parameters and symbols are the same axis — that is
//! the whole reason a sweep parallelizes for free.
//!
//! # The layout is a contract, not a convention
//!
//! ```text
//! col = combination_index * n_input_cols + input_col
//! ```
//!
//! Parameter levels **repeat** and the input level **tiles**, so one
//! combination's columns are contiguous. Writing it down here rather than
//! leaving it implicit is what lets [`crate::Indicator`] hand its output
//! straight to the simulator with no reshuffling.
//!
//! # Zipped or crossed is the caller's decision
//!
//! [`combinations`] produces the cartesian product; passing equal-length
//! vectors directly zips them instead. Making this a flag with a
//! default would be a common source of surprise — three windows and three
//! alphas are either three runs or nine, and no default is right for both.

use alphax_core::columns::{Columns, Label, Level};
use alphax_core::error::{Error, Result};

/// Every combination of indices across levels of the given sizes.
///
/// The **last** level varies fastest, as `itertools.product` does. With sizes `[3, 2]`:
///
/// ```text
/// [0,0] [0,1] [1,0] [1,1] [2,0] [2,1]
/// ```
///
/// An empty `sizes` yields one empty combination — a sweep of no parameters
/// is one run, not none. Any zero size yields nothing.
#[must_use]
pub fn combinations(sizes: &[usize]) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new()];
    for &size in sizes {
        let mut next = Vec::with_capacity(out.len() * size);
        for prefix in &out {
            for i in 0..size {
                let mut c = prefix.clone();
                c.push(i);
                next.push(c);
            }
        }
        out = next;
    }
    out
}

/// One parameter level: its name and the value for each combination.
#[derive(Debug, Clone)]
pub struct ParamLevel {
    name: String,
    /// One label per combination, not per distinct value.
    values: Vec<Label>,
}

impl ParamLevel {
    /// A level whose values are already expanded to one per combination.
    pub fn new<T>(name: impl Into<String>, values: &[T]) -> Self
    where
        T: Clone + Into<Label>,
    {
        Self {
            name: name.into(),
            values: values.iter().cloned().map(Into::into).collect(),
        }
    }

    /// How many combinations this level describes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the level is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Builds the swept column labels.
///
/// `params` carries one label per combination per level; `input` names the
/// axis being swept over and labels its columns. The result has
/// `n_combinations * input.1.len()` columns in the layout above.
///
/// # Errors
///
/// Returns [`Error::Invariant`] if the levels disagree on how many
/// combinations there are, or if a level name collides with the input's.
pub fn sweep_columns(params: &[ParamLevel], input: (&str, &[Label])) -> Result<Columns> {
    let (input_name, input_labels) = input;
    let n_input = input_labels.len();

    let n_combs = params.first().map_or(1, ParamLevel::len);
    for level in params {
        if level.len() != n_combs {
            return Err(Error::Invariant(format!(
                "parameter level {:?} describes {} combinations, but {:?} describes {n_combs}",
                level.name,
                level.len(),
                params[0].name
            )));
        }
        if level.name == input_name {
            return Err(Error::Invariant(format!(
                "parameter level {:?} collides with the input level name",
                level.name
            )));
        }
    }

    let mut levels: Vec<Level> = params
        .iter()
        .map(|p| Level::repeat(p.name.clone(), &p.values, n_input))
        .collect();
    levels.push(Level::tile(input_name, input_labels, n_combs));
    Columns::new(levels)
}

/// The output column for a combination and an input column.
#[must_use]
pub const fn column_of(combination: usize, input_col: usize, n_input_cols: usize) -> usize {
    combination * n_input_cols + input_col
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_last_level_varies_fastest() {
        assert_eq!(
            combinations(&[3, 2]),
            vec![
                vec![0, 0],
                vec![0, 1],
                vec![1, 0],
                vec![1, 1],
                vec![2, 0],
                vec![2, 1]
            ]
        );
    }

    #[test]
    fn no_parameters_is_one_run_not_none() {
        assert_eq!(combinations(&[]), vec![Vec::<usize>::new()]);
    }

    #[test]
    fn a_zero_sized_level_yields_nothing() {
        assert!(combinations(&[3, 0]).is_empty());
    }

    #[test]
    fn a_sweep_crosses_parameters_with_the_input_axis() {
        // Three windows over two symbols is six columns, laid out so that
        // each window's columns are contiguous.
        let windows = ParamLevel::new("window", &[10_usize, 20, 30]);
        let symbols = [Label::from("BTC"), Label::from("ETH")];
        let c = sweep_columns(&[windows], ("symbol", &symbols)).unwrap();

        assert_eq!(c.n_cols(), 6);
        assert_eq!(c.label(0, "window").unwrap(), &Label::from(10_usize));
        assert_eq!(c.label(1, "window").unwrap(), &Label::from(10_usize));
        assert_eq!(c.label(2, "window").unwrap(), &Label::from(20_usize));
        assert_eq!(c.label(0, "symbol").unwrap(), &Label::from("BTC"));
        assert_eq!(c.label(1, "symbol").unwrap(), &Label::from("ETH"));
    }

    #[test]
    fn the_column_formula_matches_the_labels() {
        let windows = ParamLevel::new("window", &[10_usize, 20, 30]);
        let symbols = [Label::from("BTC"), Label::from("ETH")];
        let c = sweep_columns(&[windows], ("symbol", &symbols)).unwrap();
        for comb in 0..3 {
            for (s, want) in symbols.iter().enumerate() {
                let col = column_of(comb, s, 2);
                assert_eq!(
                    c.label(col, "symbol").unwrap(),
                    want,
                    "column {col} from combination {comb}, input {s}"
                );
            }
        }
    }

    #[test]
    fn two_parameter_levels_cross() {
        let combs = combinations(&[2, 2]);
        let windows: Vec<usize> = combs.iter().map(|c| [10, 20][c[0]]).collect();
        let alphas: Vec<f64> = combs.iter().map(|c| [1.5, 2.0][c[1]]).collect();
        let symbols = [Label::from("BTC")];
        let c = sweep_columns(
            &[
                ParamLevel::new("window", &windows),
                ParamLevel::new("alpha", &alphas),
            ],
            ("symbol", &symbols),
        )
        .unwrap();
        assert_eq!(c.n_cols(), 4);
        assert!(c.describe(3).contains("window=20"));
        assert!(c.describe(3).contains("alpha=2"));
    }

    #[test]
    fn levels_that_disagree_on_the_combination_count_are_rejected() {
        let a = ParamLevel::new("window", &[10_usize, 20]);
        let b = ParamLevel::new("alpha", &[1.5_f64]);
        let symbols = [Label::from("BTC")];
        assert!(sweep_columns(&[a, b], ("symbol", &symbols)).is_err());
    }

    #[test]
    fn a_level_name_colliding_with_the_input_is_rejected() {
        let a = ParamLevel::new("symbol", &[10_usize]);
        let symbols = [Label::from("BTC")];
        assert!(sweep_columns(&[a], ("symbol", &symbols)).is_err());
    }

    #[test]
    fn an_unswept_indicator_still_gets_its_input_labels() {
        let symbols = [Label::from("BTC"), Label::from("ETH")];
        let c = sweep_columns(&[], ("symbol", &symbols)).unwrap();
        assert_eq!(c.n_cols(), 2);
        assert_eq!(c.label(1, "symbol").unwrap(), &Label::from("ETH"));
    }
}
