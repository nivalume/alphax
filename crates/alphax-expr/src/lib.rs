//! Qlib-style factor expressions over the engine's column-major panels.
//!
//! Expressions are compiled once and can then be evaluated against many
//! [`FactorInputs`]. The result is already a [`alphax_core::matrix::Matrix`], so
//! it feeds the existing signal and portfolio layers directly.

mod alpha158;
mod alpha360;
mod backend;
mod cross;
mod eval;
mod graph;
mod grid;
mod lift;
mod order;
mod parse;
mod pit;
#[cfg(feature = "polars")]
mod polars_backend;
mod stage;
mod warmup;
mod window;
mod wq101;

pub use alpha158::{
    ALPHA158_ROLLING, Alpha158Config, Alpha158Price, Alpha158Rolling, alpha158, alpha158_features,
    alpha158_with, qlib_label,
};
pub use alpha360::alpha360;
pub use backend::{ComputeBackend, NativeBackend};
pub use eval::{FactorInputs, FactorValues};
pub use grid::{FactorGrid, GridCase};
pub use parse::Expression;
pub use pit::PitRecord;
#[cfg(feature = "polars")]
pub use polars_backend::PolarsBackend;
pub use wq101::{wq101, wq101_all, wq101_formula};

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::OnceLock;

use alphax_core::error::{Error, Result};

use crate::parse::parse;

/// A named, compiled collection of factor expressions.
#[derive(Debug, Clone)]
pub struct FactorSet {
    factors: Vec<(String, Expression)>,
    /// The factors as the graph reads them: lifted subexpressions replaced
    /// by their hidden fields.
    rewritten: Vec<Expression>,
    /// What fills those fields.
    lifted: lift::Lifted,
    graph: graph::Graph,
    /// Each node's lookback and the fields it reads.
    warm: warmup::Warmup,
    /// Whether evaluation blanks the rows before each window has filled.
    warmup: bool,
    /// The stages, when a factor reads across symbols.
    plan: Option<stage::Staged>,
    /// The factors split into separately compiled groups, built on first use
    /// by a panel narrower than the thread pool.
    parts: OnceLock<Vec<eval::Part>>,
}

impl FactorSet {
    /// Parses and validates `(name, expression)` pairs.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty or repeated name, invalid syntax, an
    /// unsupported operator, or an invalid literal parameter (a fractional or
    /// negative window, a quantile outside `0..=1`, a negative EMA span).
    pub fn compile<N, S, I>(factors: I) -> Result<Self>
    where
        N: Into<String>,
        S: AsRef<str>,
        I: IntoIterator<Item = (N, S)>,
    {
        let mut seen = BTreeSet::new();
        let mut compiled = Vec::new();
        for (name, source) in factors {
            let name = name.into();
            if name.is_empty() {
                return Err(Error::Invariant("factor name cannot be empty".into()));
            }
            if !seen.insert(name.clone()) {
                return Err(Error::Invariant(format!("duplicate factor name {name:?}")));
            }
            let expr = parse(source.as_ref())
                .map_err(|message| Error::Invariant(format!("factor {name:?}: {message}")))?;
            compiled.push((name, expr));
        }
        if compiled.is_empty() {
            return Err(Error::Invariant("a factor set cannot be empty".into()));
        }
        Self::build(compiled)
    }

    /// Compiles parsed, arity-checked expressions.
    pub(crate) fn build(factors: Vec<(String, Expression)>) -> Result<Self> {
        let (rewritten, lifted) = lift::Lifted::lift(&factors)?;
        let graph = graph::Graph::compile(
            factors
                .iter()
                .zip(&rewritten)
                .map(|((name, _), expr)| (name.as_str(), expr)),
        )?;
        let plan = stage::Staged::plan(&graph);
        let warm = warmup::Warmup::plan(&graph, &lifted);
        Ok(Self {
            factors,
            rewritten,
            lifted,
            graph,
            warm,
            warmup: false,
            plan,
            parts: OnceLock::new(),
        })
    }

    /// Per factor, in output order: how many rows before the current one its
    /// windows and lags reach (`Mean($close, 20)` is 19, `Ref(Mean($close,
    /// 5), 2)` is 6). Its value is complete from that many rows after its
    /// inputs begin; Qlib's operators give a value earlier, from a partial
    /// window.
    #[must_use]
    pub fn lookbacks(&self) -> Vec<usize> {
        self.graph
            .outputs
            .iter()
            .map(|&node| self.warm.lookback[node])
            .collect()
    }

    /// The same set, evaluating with `NaN` on each symbol's rows before every
    /// window of a factor has filled: before the first row where the
    /// factor's input fields all have a value, plus its lookback. A
    /// cross-sectional operator then skips a symbol that is still warming up.
    /// Off by default, which is Qlib's behaviour.
    #[must_use]
    pub fn with_warmup(mut self, on: bool) -> Self {
        self.warmup = on;
        self
    }

    /// Whether [`FactorSet::with_warmup`] is on.
    #[must_use]
    pub fn warmup(&self) -> bool {
        self.warmup
    }

    /// The instruments other than its own symbols that the set reads with
    /// `ChangeInstrument` (or Qlib's `Mask(e, "X")`).
    #[must_use]
    pub fn instrument_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        self.lifted.instrument_names(&mut names);
        names
    }

    /// Names in output order.
    pub fn names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.factors.iter().map(|(name, _)| name.as_str())
    }

    /// Compiled expressions in output order.
    pub fn expressions(&self) -> impl ExactSizeIterator<Item = (&str, &Expression)> {
        self.factors
            .iter()
            .map(|(name, expression)| (name.as_str(), expression))
    }

    /// Whether the set reads the bar timestamps
    /// ([`FactorInputs::with_timestamps`]): `DayLast`, `DayCumsum`, `Date`,
    /// `P` and `PRef`, here or in an instrument's expression.
    #[must_use]
    pub fn needs_timestamps(&self) -> bool {
        self.graph
            .fields
            .iter()
            .any(|field| field == graph::DAY || field == graph::MINUTE)
            || !self.lifted.pits.is_empty()
            || self
                .lifted
                .instruments
                .iter()
                .any(|instrument| instrument.set.needs_timestamps())
    }

    /// Source fields required to evaluate this set. A point-in-time field is
    /// listed as `$name` (it is written `$$name`), and an instrument's fields
    /// are listed with the panel's.
    #[must_use]
    pub fn required_fields(&self) -> BTreeSet<&str> {
        let mut fields = BTreeSet::new();
        for (_, expr) in &self.factors {
            expr.fields(&mut fields);
        }
        fields
    }

    /// Whether any expression reads a future row.
    #[must_use]
    pub fn uses_future(&self) -> bool {
        self.factors.iter().any(|(_, expr)| expr.uses_future())
    }

    /// Evaluates every factor over every input symbol.
    ///
    /// # Errors
    ///
    /// Returns an error when an input field is missing or has the wrong shape,
    /// or an operator receives invalid arguments.
    pub fn evaluate(&self, inputs: &FactorInputs<'_>) -> Result<FactorValues> {
        self.evaluate_with(inputs, &NativeBackend)
    }

    /// Evaluates with an explicitly selected compute backend.
    ///
    /// # Errors
    ///
    /// Propagates input, operator, or backend execution errors.
    pub fn evaluate_with(
        &self,
        inputs: &FactorInputs<'_>,
        backend: &dyn ComputeBackend,
    ) -> Result<FactorValues> {
        let values = backend.evaluate(self, inputs)?;
        if !self.warmup {
            return Ok(values);
        }
        let starts = warmup::Warmup::starts(&self.graph, inputs)?;
        let n_rows = inputs.shape().0;
        let (names, matrices): (Vec<String>, Vec<_>) = values
            .into_matrices()
            .enumerate()
            .map(|(k, (name, matrix))| {
                let (rows, cols) = (matrix.n_rows(), matrix.n_cols());
                let mut cells = matrix.into_vec();
                self.warm
                    .mask(self.graph.outputs[k], &starts, &mut cells, n_rows);
                (
                    name,
                    alphax_core::matrix::Matrix::from_column_major(cells, rows, cols),
                )
            })
            .unzip();
        FactorValues::from_matrices(names, matrices.into_iter().collect::<Result<_>>()?)
    }

    /// Evaluates `batch` symbols at a time, handing each batch's values to
    /// `sink` with the range of symbols they cover, in order. Without a
    /// cross-sectional operator only one batch is held at a time, so peak
    /// memory is bounded by `batch`, not by the panel; with one, every symbol
    /// is needed for every row, so the whole panel is evaluated first and
    /// then handed over in batches.
    ///
    /// # Errors
    ///
    /// As [`FactorSet::evaluate`], a zero `batch`, and whatever `sink`
    /// returns, which stops the run.
    pub fn evaluate_batches(
        &self,
        inputs: &FactorInputs<'_>,
        batch: usize,
        mut sink: impl FnMut(Range<usize>, FactorValues) -> Result<()>,
    ) -> Result<()> {
        if batch == 0 {
            return Err(Error::Invariant(
                "a batch must hold at least one symbol".into(),
            ));
        }
        let (_, n_cols) = inputs.shape();
        if self.plan.is_some() {
            let all = self.evaluate(inputs)?;
            for start in (0..n_cols).step_by(batch) {
                let cols = start..(start + batch).min(n_cols);
                sink(cols.clone(), all.columns(cols)?)?;
            }
            return Ok(());
        }
        let needed = self.instrument_names();
        for start in (0..n_cols).step_by(batch) {
            let cols = start..(start + batch).min(n_cols);
            let part = inputs.columns(cols.clone(), &needed)?;
            sink(cols, self.evaluate(&part)?)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alphax_model::OhlcvPanel;

    #[test]
    fn literals_and_fields_broadcast_and_shared_nodes_are_reused() {
        let panel = OhlcvPanel {
            close: vec![1.0, 2.0, 4.0, 10.0, 20.0, 40.0],
            n_rows: 3,
            n_cols: 2,
            ..OhlcvPanel::default()
        };
        let inputs = FactorInputs::from_ohlcv(&panel).unwrap();
        let factors = FactorSet::compile([
            ("constant", "1 + 2"),
            ("field", "+$close"),
            ("left", "2 * $close"),
            ("right", "$close - 1"),
            ("choice", "If($close > 3, 1, -1)"),
            ("rolling_literal", "Mean(4, 2)"),
            ("pair_literal", "Cov($close, 5, 2)"),
            ("shared", "Ref($close, 1) / Ref($close, 1)"),
            ("repeat", "2 * $close"),
        ])
        .unwrap();
        let values = factors.evaluate(&inputs).unwrap();
        let column = |name: &str, col: usize| values.get(name).unwrap().column(col).to_vec();

        assert_eq!(column("constant", 1), [3.0; 3]);
        assert_eq!(column("field", 1), [10.0, 20.0, 40.0]);
        assert_eq!(column("left", 0), [2.0, 4.0, 8.0]);
        assert_eq!(column("right", 1), [9.0, 19.0, 39.0]);
        assert_eq!(column("choice", 0), [-1.0, -1.0, 1.0]);
        assert_eq!(column("rolling_literal", 1), [4.0; 3]);
        assert!(column("pair_literal", 0)[0].is_nan());
        assert_eq!(column("pair_literal", 0)[1..], [0.0, 0.0]);
        assert!(column("shared", 0)[0].is_nan());
        assert_eq!(column("shared", 1)[1..], [1.0, 1.0]);
        assert_eq!(column("repeat", 1), column("left", 1));
    }

    #[test]
    fn missing_fields_fail_before_any_work() {
        let panel = OhlcvPanel {
            close: vec![1.0, 2.0],
            n_rows: 2,
            n_cols: 1,
            ..OhlcvPanel::default()
        };
        let inputs = FactorInputs::from_ohlcv(&panel).unwrap();
        let factors = FactorSet::compile([("f", "$vwap / $close")]).unwrap();
        let error = factors.evaluate(&inputs).unwrap_err().to_string();
        assert!(error.contains("$vwap"), "{error}");
    }

    #[test]
    fn each_symbol_evaluates_exactly_as_it_would_alone() {
        let (n_rows, n_cols) = (90, 3);
        let series = |base: f64| {
            (0..n_cols)
                .flat_map(|col| {
                    (0..n_rows).map(move |row| {
                        if (row + col) % 17 == 0 {
                            f64::NAN
                        } else {
                            base + col as f64 * 3.0 + ((row * 7 + col) % 11) as f64
                        }
                    })
                })
                .collect::<Vec<_>>()
        };
        let panel = OhlcvPanel {
            open: series(10.0),
            high: series(12.0),
            low: series(8.0),
            close: series(10.5),
            volume: series(1000.0),
            n_rows,
            n_cols,
        };
        let vwap = series(10.2);
        let factors = alpha158().unwrap();
        let all = factors
            .evaluate(
                &FactorInputs::from_ohlcv(&panel)
                    .unwrap()
                    .with_field("vwap", &vwap)
                    .unwrap(),
            )
            .unwrap();
        for col in 0..n_cols {
            let one = |values: &[f64]| values[col * n_rows..(col + 1) * n_rows].to_vec();
            let single = OhlcvPanel {
                open: one(&panel.open),
                high: one(&panel.high),
                low: one(&panel.low),
                close: one(&panel.close),
                volume: one(&panel.volume),
                n_rows,
                n_cols: 1,
            };
            let single_vwap = one(&vwap);
            let alone = factors
                .evaluate(
                    &FactorInputs::from_ohlcv(&single)
                        .unwrap()
                        .with_field("vwap", &single_vwap)
                        .unwrap(),
                )
                .unwrap();
            for ((name, matrix), (_, expected)) in all.iter().zip(alone.iter()) {
                let same = matrix
                    .column(col)
                    .iter()
                    .zip(expected.column(0))
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                assert!(same, "{name} column {col}");
            }
        }
    }

    #[test]
    fn expressions_feed_column_major_factor_matrices() {
        let panel = OhlcvPanel {
            close: vec![1.0, 2.0, 4.0, 10.0, 20.0, 40.0],
            n_rows: 3,
            n_cols: 2,
            ..OhlcvPanel::default()
        };
        let inputs = FactorInputs::from_ohlcv(&panel).unwrap();
        let factors = FactorSet::compile([
            ("roc", "Ref($close, 1) / $close"),
            ("ma", "Mean($close, 2) / $close"),
        ])
        .unwrap();
        let values = factors.evaluate(&inputs).unwrap();

        assert!(values.get("roc").unwrap().get(0, 0).is_nan());
        assert!((values.get("roc").unwrap().get(2, 0) - 0.5).abs() < 1e-12);
        assert!((values.get("ma").unwrap().get(1, 1) - 0.75).abs() < 1e-12);
    }

    #[test]
    fn alpha158_has_the_expected_shape_and_fields() {
        let factors = alpha158().unwrap();
        assert_eq!(factors.names().len(), 158);
        assert_eq!(
            factors.names().skip(13).take(7).collect::<Vec<_>>(),
            ["ROC5", "ROC10", "ROC20", "ROC30", "ROC60", "MA5", "MA10"]
        );
        assert_eq!(
            factors.required_fields(),
            BTreeSet::from(["close", "high", "low", "open", "volume", "vwap"])
        );
        assert!(!factors.uses_future());
        assert!(
            FactorSet::compile([("label", "Ref($close, -2)")])
                .unwrap()
                .uses_future()
        );
    }

    #[test]
    fn every_alpha158_factor_evaluates_for_every_symbol() {
        let n_rows = 70;
        let n_cols = 2;
        let series = |base: f64, step: f64| {
            (0..n_cols)
                .flat_map(|col| {
                    (0..n_rows).map(move |row| base + col as f64 * 10.0 + row as f64 * step)
                })
                .collect::<Vec<_>>()
        };
        let panel = OhlcvPanel {
            open: series(100.0, 0.10),
            high: series(101.0, 0.12),
            low: series(99.0, 0.08),
            close: series(100.5, 0.11),
            volume: series(1_000.0, 3.0),
            n_rows,
            n_cols,
        };
        let vwap = series(100.4, 0.105);
        let inputs = FactorInputs::from_ohlcv(&panel)
            .unwrap()
            .with_field("vwap", &vwap)
            .unwrap();
        let values = alpha158().unwrap().evaluate(&inputs).unwrap();

        assert_eq!(values.iter().len(), 158);
        assert!(
            values
                .iter()
                .all(|(_, value)| { value.n_rows() == n_rows && value.n_cols() == n_cols })
        );
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "these values are exactly representable results of simple arithmetic"
    )]
    fn common_qlib_conditionals_and_expanding_windows_work() {
        let panel = OhlcvPanel {
            close: vec![1.0, 2.0, 3.0, 4.0],
            n_rows: 4,
            n_cols: 1,
            ..OhlcvPanel::default()
        };
        let values = FactorSet::compile([
            (
                "conditional",
                "If(($close>=2)&($close<4),Power($close,2),-1)",
            ),
            ("expanding", "Mean($close,0)"),
        ])
        .unwrap()
        .evaluate(&FactorInputs::from_ohlcv(&panel).unwrap())
        .unwrap();

        assert_eq!(
            values.get("conditional").unwrap().column(0),
            [-1.0, 4.0, 9.0, -1.0]
        );
        assert_eq!(
            values.get("expanding").unwrap().column(0),
            [1.0, 1.5, 2.0, 2.5]
        );
    }

    #[test]
    fn zero_offset_ref_and_delta_read_the_first_row_as_qlib() {
        let panel = OhlcvPanel {
            close: vec![1.0, 2.0, 3.0, f64::NAN, 5.0, 6.0],
            n_rows: 3,
            n_cols: 2,
            ..OhlcvPanel::default()
        };
        let values = FactorSet::compile([("first", "Ref($close,0)"), ("since", "Delta($close,0)")])
            .unwrap()
            .evaluate(&FactorInputs::from_ohlcv(&panel).unwrap())
            .unwrap();

        // Each symbol's first row, even when it is missing.
        assert_eq!(values.get("first").unwrap().column(0), [1.0, 1.0, 1.0]);
        assert_eq!(values.get("since").unwrap().column(0), [0.0, 1.0, 2.0]);
        assert!(
            values
                .get("first")
                .unwrap()
                .column(1)
                .iter()
                .all(|v| v.is_nan())
        );
        assert!(
            values
                .get("since")
                .unwrap()
                .column(1)
                .iter()
                .all(|v| v.is_nan())
        );
    }
}
