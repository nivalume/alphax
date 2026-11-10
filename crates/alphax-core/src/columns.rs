//! Multi-level column labels.
//!
//! Parameters, symbols and factors all collapse into a single flat column
//! axis, so the simulator never learns what a column *means* — it just runs
//! `n_cols` independent simulations. Labels are therefore the only mechanism
//! that can attribute a result back to what produced it, and the only basis
//! for aggregating a sweep ("mean Sharpe per factor", "best quantile per
//! symbol").
//!
//! # The layout invariant
//!
//! Parameter levels are built with [`Level::repeat`] and symbol levels with
//! [`Level::tile`]. With `n_params` parameter combinations and `n_symbols`
//! symbols that yields
//!
//! ```text
//! col = param_idx * n_symbols + symbol_idx
//! ```
//!
//! so every column belonging to one parameter combination is **contiguous**.
//! Nothing in this module enforces that — it is a convention the constructors
//! establish and the rest of the engine relies on.

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{Error, Result};

/// A single column label value.
///
/// `Float` compares and hashes by bit pattern, so `NaN` is equal to itself
/// and `0.0` is distinct from `-0.0`. That is the behaviour you want for
/// labels — they are identifiers, not measurements.
#[derive(Debug, Clone)]
pub enum Label {
    /// An integer parameter, e.g. a lookback window.
    Int(i64),
    /// A real-valued parameter, e.g. a quantile threshold.
    Float(f64),
    /// A name: a symbol, a factor, a mode.
    Str(Arc<str>),
    /// A flag parameter.
    Bool(bool),
}

impl Label {
    /// A string label, from anything string-like.
    pub fn str(s: impl AsRef<str>) -> Self {
        Self::Str(Arc::from(s.as_ref()))
    }

    /// Key used for equality and hashing; `Float` collapses to its bits.
    const fn key(&self) -> (u8, u64) {
        match *self {
            Self::Int(v) => (0, v as u64),
            Self::Float(v) => (1, v.to_bits()),
            Self::Bool(v) => (3, v as u64),
            // Strings are compared by content, not through this key.
            Self::Str(_) => (2, 0),
        }
    }
}

impl PartialEq for Label {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Str(a), Self::Str(b)) => a == b,
            // Distinct variants always differ in the key's tag byte, so a
            // Str compared against a non-Str falls out as unequal here.
            _ => self.key() == other.key(),
        }
    }
}

impl Eq for Label {}

impl std::hash::Hash for Label {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::Str(s) => {
                2_u8.hash(state);
                s.hash(state);
            }
            other => other.key().hash(state),
        }
    }
}

impl std::fmt::Display for Label {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Int(v) => write!(f, "{v}"),
            Self::Float(v) => write!(f, "{v}"),
            Self::Str(v) => write!(f, "{v}"),
            Self::Bool(v) => write!(f, "{v}"),
        }
    }
}

impl From<i64> for Label {
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}

impl From<usize> for Label {
    fn from(v: usize) -> Self {
        Self::Int(v as i64)
    }
}

impl From<f64> for Label {
    fn from(v: f64) -> Self {
        Self::Float(v)
    }
}

impl From<bool> for Label {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}

impl From<&str> for Label {
    fn from(v: &str) -> Self {
        Self::str(v)
    }
}

/// One named level of the column index: a label for every column.
#[derive(Debug, Clone)]
pub struct Level {
    name: String,
    labels: Vec<Label>,
}

impl Level {
    /// A level from an explicit label per column.
    pub fn new(name: impl Into<String>, labels: Vec<Label>) -> Self {
        Self {
            name: name.into(),
            labels,
        }
    }

    /// Repeats each value `times` consecutively: `[a, b]`, 3 → `[a,a,a,b,b,b]`.
    ///
    /// This is how **parameter** levels are built. Repeating (rather than
    /// tiling) is what makes one parameter combination's columns contiguous.
    pub fn repeat<T>(name: impl Into<String>, values: &[T], times: usize) -> Self
    where
        T: Clone + Into<Label>,
    {
        let mut labels = Vec::with_capacity(values.len() * times);
        for v in values {
            for _ in 0..times {
                labels.push(v.clone().into());
            }
        }
        Self::new(name, labels)
    }

    /// Tiles the whole sequence `times`: `[a, b]`, 3 → `[a,b,a,b,a,b]`.
    ///
    /// This is how **symbol** levels are built, so that each parameter
    /// block contains every symbol exactly once.
    pub fn tile<T>(name: impl Into<String>, values: &[T], times: usize) -> Self
    where
        T: Clone + Into<Label>,
    {
        let mut labels = Vec::with_capacity(values.len() * times);
        for _ in 0..times {
            for v in values {
                labels.push(v.clone().into());
            }
        }
        Self::new(name, labels)
    }

    /// The level's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The label for each column.
    #[must_use]
    pub fn labels(&self) -> &[Label] {
        &self.labels
    }

    /// How many columns this level describes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Whether this level describes no columns.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }
}

/// The full column index: zero or more named levels over `n_cols` columns.
#[derive(Debug, Clone)]
pub struct Columns {
    levels: Vec<Level>,
    n_cols: usize,
}

impl Columns {
    /// An unlabelled index of `n_cols` columns.
    #[must_use]
    pub const fn unlabelled(n_cols: usize) -> Self {
        Self {
            levels: Vec::new(),
            n_cols,
        }
    }

    /// Builds an index from levels, checking they agree on the column count.
    ///
    /// # Errors
    ///
    /// Returns [`Error::LevelWidth`] if the levels disagree.
    pub fn new(levels: Vec<Level>) -> Result<Self> {
        let Some(first) = levels.first() else {
            return Ok(Self::unlabelled(0));
        };
        let n_cols = first.len();
        for level in &levels {
            if level.len() != n_cols {
                return Err(Error::LevelWidth {
                    name: level.name.clone(),
                    actual: level.len(),
                    expected: n_cols,
                });
            }
        }
        Ok(Self { levels, n_cols })
    }

    /// Number of columns.
    #[must_use]
    pub const fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// Whether there are no columns.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.n_cols == 0
    }

    /// The levels, in order.
    #[must_use]
    pub fn levels(&self) -> &[Level] {
        &self.levels
    }

    /// Looks up a level by name.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnknownLevel`] if no such level exists.
    pub fn level(&self, name: &str) -> Result<&Level> {
        self.levels
            .iter()
            .find(|l| l.name == name)
            .ok_or_else(|| Error::UnknownLevel {
                name: name.to_owned(),
                available: self
                    .levels
                    .iter()
                    .map(|l| l.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
            })
    }

    /// The label of one column within one level.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnknownLevel`] if the level does not exist.
    ///
    /// # Panics
    ///
    /// Panics if `col >= n_cols`.
    pub fn label(&self, col: usize, level: &str) -> Result<&Label> {
        Ok(&self.level(level)?.labels[col])
    }

    /// Groups column indices by their label in `level`, in first-appearance
    /// order.
    ///
    /// This is the basis of every by-level aggregation: `group_by("factor")`
    /// over a per-column metric vector, `idxmax` within a symbol, and so on.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnknownLevel`] if the level does not exist.
    pub fn group_indices(&self, level: &str) -> Result<Vec<(Label, Vec<usize>)>> {
        let labels = self.level(level)?.labels();
        let mut order: Vec<Label> = Vec::new();
        let mut slots: HashMap<Label, Vec<usize>> = HashMap::new();
        for (col, label) in labels.iter().enumerate() {
            slots
                .entry(label.clone())
                .or_insert_with(|| {
                    order.push(label.clone());
                    Vec::new()
                })
                .push(col);
        }
        Ok(order
            .into_iter()
            .map(|label| {
                let cols = slots.remove(&label).unwrap_or_default();
                (label, cols)
            })
            .collect())
    }

    /// Renders one column as `level=label` pairs, for reporting.
    ///
    /// # Panics
    ///
    /// Panics if `col >= n_cols`.
    #[must_use]
    pub fn describe(&self, col: usize) -> String {
        if self.levels.is_empty() {
            return format!("col{col}");
        }
        self.levels
            .iter()
            .map(|l| format!("{}={}", l.name, l.labels[col]))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The canonical sweep layout: 3 windows x 2 symbols.
    fn sweep() -> Columns {
        let windows = [10_i64, 20, 50];
        let symbols = ["BTC", "ETH"];
        Columns::new(vec![
            Level::repeat("window", &windows, symbols.len()),
            Level::tile("symbol", &symbols, windows.len()),
        ])
        .unwrap()
    }

    #[test]
    fn params_repeat_and_symbols_tile() {
        let c = sweep();
        assert_eq!(c.n_cols(), 6);

        let w: Vec<String> = c
            .level("window")
            .unwrap()
            .labels()
            .iter()
            .map(ToString::to_string)
            .collect();
        assert_eq!(w, ["10", "10", "20", "20", "50", "50"]);

        let s: Vec<String> = c
            .level("symbol")
            .unwrap()
            .labels()
            .iter()
            .map(ToString::to_string)
            .collect();
        assert_eq!(s, ["BTC", "ETH", "BTC", "ETH", "BTC", "ETH"]);
    }

    #[test]
    fn layout_invariant_holds() {
        // col == param_idx * n_symbols + symbol_idx
        let c = sweep();
        let n_symbols = 2;
        for (param_idx, window) in [10_i64, 20, 50].iter().enumerate() {
            for (symbol_idx, symbol) in ["BTC", "ETH"].iter().enumerate() {
                let col = param_idx * n_symbols + symbol_idx;
                assert_eq!(c.label(col, "window").unwrap(), &Label::Int(*window));
                assert_eq!(c.label(col, "symbol").unwrap(), &Label::str(symbol));
            }
        }
    }

    #[test]
    fn one_parameter_combination_occupies_contiguous_columns() {
        let c = sweep();
        for (_, cols) in c.group_indices("window").unwrap() {
            let contiguous = cols.windows(2).all(|w| w[1] == w[0] + 1);
            assert!(
                contiguous,
                "parameter block must be contiguous, got {cols:?}"
            );
        }
    }

    #[test]
    fn group_indices_preserves_first_appearance_order() {
        let c = sweep();

        let by_window = c.group_indices("window").unwrap();
        assert_eq!(by_window.len(), 3);
        assert_eq!(by_window[0].0, Label::Int(10));
        assert_eq!(by_window[0].1, vec![0, 1]);
        assert_eq!(by_window[2].1, vec![4, 5]);

        // Symbols are strided, not contiguous — that is the tile half.
        let by_symbol = c.group_indices("symbol").unwrap();
        assert_eq!(by_symbol.len(), 2);
        assert_eq!(by_symbol[0].0, Label::str("BTC"));
        assert_eq!(by_symbol[0].1, vec![0, 2, 4]);
        assert_eq!(by_symbol[1].1, vec![1, 3, 5]);
    }

    #[test]
    fn mismatched_level_widths_are_rejected() {
        let err = Columns::new(vec![
            Level::new("a", vec![Label::Int(1), Label::Int(2)]),
            Level::new("b", vec![Label::Int(1)]),
        ]);
        assert!(err.is_err());
    }

    #[test]
    fn unknown_level_names_what_is_available() {
        let c = sweep();
        let msg = c.level("nope").unwrap_err().to_string();
        assert!(msg.contains("window"), "{msg}");
        assert!(msg.contains("symbol"), "{msg}");
    }

    #[test]
    fn float_labels_are_usable_as_keys() {
        // Quantile thresholds are the motivating case.
        let qs = [0.8_f64, 0.9, 0.95];
        let c = Columns::new(vec![Level::repeat("quantile", &qs, 1)]).unwrap();
        let groups = c.group_indices("quantile").unwrap();
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[1].0, Label::Float(0.9));
    }

    #[test]
    fn describe_renders_every_level() {
        let c = sweep();
        assert_eq!(c.describe(3), "window=20, symbol=ETH");
        assert_eq!(Columns::unlabelled(4).describe(2), "col2");
    }
}
