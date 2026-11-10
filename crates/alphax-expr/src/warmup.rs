//! Lookbacks and the warm-up of windows.
//!
//! Qlib's operators have `min_periods = 1`: a 20-row mean has a value on the
//! first row, from one observation. Qlib hides this by loading history
//! before the requested start; a backtest over a panel would trade on it. A
//! node's *lookback* is how many rows before the current one its windows and
//! lags reach, so its value is complete from `lookback` rows after its
//! inputs begin:
//!
//! * `Ref(x, k)` with `k ≥ 0` adds `k`, `Delta(x, n)` adds `n`;
//! * a rolling window of `n` (and `Corr` / `Cov`) adds `n - 1`, an expanding
//!   one nothing;
//! * `EMA` counts as a window of its span, `2 / alpha - 1`;
//! * everything else reads its inputs' rows: the largest of theirs;
//! * an instrument's series (`ChangeInstrument`) brings its expression's
//!   lookback; `P` is complete on every row.
//!
//! With warm-up on ([`crate::FactorSet::with_warmup`]), a factor is `NaN` on
//! each symbol's rows before `start + lookback`, where `start` is the first
//! row where every input field the factor reads has a value for that symbol.
//! A cross-sectional operator likewise never sees a symbol's value before it
//! is complete, so a symbol still warming up does not move the others' ranks.

use alphax_core::error::Result;

use crate::FactorInputs;
use crate::graph::{Graph, NodeId, Op, Window};
use crate::lift::Lifted;

/// Per node of a graph: its lookback and the input fields it reads.
#[derive(Debug, Clone)]
pub(crate) struct Warmup {
    pub(crate) lookback: Vec<usize>,
    /// Indices into [`Graph::fields`] of the panel fields (not hidden ones)
    /// a node reads, sorted.
    fields: Vec<Vec<usize>>,
}

impl Warmup {
    pub(crate) fn plan(graph: &Graph, lifted: &Lifted) -> Self {
        let instrument_lookback = |name: &str| {
            lifted.instruments.iter().find_map(|instrument| {
                let at = instrument.set.names().position(|n| n == name)?;
                Some(instrument.set.lookbacks()[at])
            })
        };
        let mut lookback = vec![0; graph.nodes.len()];
        let mut fields: Vec<Vec<usize>> = vec![Vec::new(); graph.nodes.len()];
        for (at, node) in graph.nodes.iter().enumerate() {
            let inputs = node.inputs.iter().map(|&i| lookback[i]).max().unwrap_or(0);
            lookback[at] = match node.op {
                Op::Field(index) => {
                    let name = &graph.fields[index];
                    if !name.starts_with('@') {
                        fields[at].push(index);
                    }
                    instrument_lookback(name).unwrap_or(0)
                }
                Op::Ref(offset) => inputs + usize::try_from(offset).unwrap_or(0),
                Op::Delta(offset) => inputs + offset,
                Op::Ema(Some(bits)) => {
                    let span = (2.0 / f64::from_bits(bits) - 1.0).round().max(1.0);
                    inputs + span as usize - 1
                }
                Op::Rolling(_, Window::Fixed(window))
                | Op::Pair {
                    window: Window::Fixed(window),
                    ..
                } => inputs + window.saturating_sub(1),
                _ => inputs,
            };
            if !node.inputs.is_empty() {
                let mut read: Vec<usize> = node
                    .inputs
                    .iter()
                    .flat_map(|&i| fields[i].iter().copied())
                    .collect();
                read.sort_unstable();
                read.dedup();
                fields[at] = read;
            }
        }
        Self { lookback, fields }
    }

    /// Per field of `graph`, per symbol: the first row with a value (the row
    /// count when there is none). Hidden fields start at row 0.
    pub(crate) fn starts(graph: &Graph, inputs: &FactorInputs<'_>) -> Result<Vec<Vec<usize>>> {
        let (n_rows, n_cols) = inputs.shape();
        graph
            .fields
            .iter()
            .map(|name| {
                if name.starts_with('@') {
                    return Ok(Vec::new());
                }
                let values = inputs.field(name)?;
                Ok((0..n_cols)
                    .map(|col| {
                        let column = &values[col * n_rows..(col + 1) * n_rows];
                        column.iter().position(|v| !v.is_nan()).unwrap_or(n_rows)
                    })
                    .collect())
            })
            .collect()
    }

    /// `NaN` over each symbol's rows of `node` before it is complete.
    pub(crate) fn mask(
        &self,
        node: NodeId,
        starts: &[Vec<usize>],
        values: &mut [f64],
        n_rows: usize,
    ) {
        for (col, column) in values.chunks_mut(n_rows).enumerate() {
            let start = self.fields[node]
                .iter()
                .map(|&field| starts[field][col])
                .max()
                .unwrap_or(0);
            let end = start.saturating_add(self.lookback[node]).min(n_rows);
            column[..end].fill(f64::NAN);
        }
    }
}
