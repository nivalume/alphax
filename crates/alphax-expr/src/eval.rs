use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use alphax_core::error::{Error, Result};
use alphax_core::matrix::Matrix;
use alphax_generic::rolling::{rolling_max_1d, rolling_mean_1d, rolling_min_1d};
use alphax_model::OhlcvPanel;
use rayon::prelude::*;

use crate::FactorSet;
use crate::cross;
use crate::graph::{self, BinaryOp, Cross, DAY, Graph, MINUTE, NodeId, Op, Rolling, UnaryOp};
use crate::order::{Order, ordered_into, rolling_arg_extreme_into};
use crate::pit::{self, PitRecord};
use crate::stage::{Dest, Source, Staged, Step};
use crate::warmup::Warmup;
use crate::window::{Count, Deviation, Fit, Pair, Regression, Shape, Sum, Wma, slide_into};

/// Named column-major fields supplied to a factor set, and what the
/// operators beyond a symbol's own fields read: the bar timestamps
/// (`DayLast`, `DayCumsum`, `Date`, `P`), the symbols and other instruments
/// (`ChangeInstrument`), and point-in-time records (`$$field`).
#[derive(Debug, Clone)]
pub struct FactorInputs<'a> {
    n_rows: usize,
    n_cols: usize,
    fields: HashMap<String, &'a [f64]>,
    /// Per row: the calendar day and the minute of the day.
    calendar: Option<Arc<(Vec<f64>, Vec<f64>)>>,
    symbols: Option<Vec<String>>,
    instruments: HashMap<String, FactorInputs<'a>>,
    pit: HashMap<String, Vec<PitRecord>>,
}

const NANOS_PER_DAY: i64 = 86_400_000_000_000;
const NANOS_PER_MINUTE: i64 = 60_000_000_000;

impl<'a> FactorInputs<'a> {
    /// Creates an empty input set with a fixed `(rows, symbols)` shape.
    ///
    /// # Errors
    ///
    /// Returns an error when either dimension is zero.
    pub fn new(n_rows: usize, n_cols: usize) -> Result<Self> {
        if n_rows == 0 || n_cols == 0 {
            return Err(Error::Invariant(format!(
                "factor input shape must be non-zero, got {n_rows}x{n_cols}"
            )));
        }
        Ok(Self {
            n_rows,
            n_cols,
            fields: HashMap::new(),
            calendar: None,
            symbols: None,
            instruments: HashMap::new(),
            pit: HashMap::new(),
        })
    }

    /// Borrows the populated fields of an OHLCV panel.
    ///
    /// # Errors
    ///
    /// Returns an error when a populated field does not match the panel shape.
    pub fn from_ohlcv(panel: &'a OhlcvPanel) -> Result<Self> {
        let mut inputs = Self::new(panel.n_rows, panel.n_cols)?;
        for (name, values) in [
            ("open", panel.open.as_slice()),
            ("high", panel.high.as_slice()),
            ("low", panel.low.as_slice()),
            ("close", panel.close.as_slice()),
            ("volume", panel.volume.as_slice()),
        ] {
            if !values.is_empty() {
                inputs.add_field(name, values)?;
            }
        }
        Ok(inputs)
    }

    /// Adds a custom field such as `vwap`.
    ///
    /// # Errors
    ///
    /// Returns an error when its length differs from `n_rows * n_cols`.
    pub fn with_field(mut self, name: &str, values: &'a [f64]) -> Result<Self> {
        self.add_field(name, values)?;
        Ok(self)
    }

    fn add_field(&mut self, name: &str, values: &'a [f64]) -> Result<()> {
        let expected = self.n_rows * self.n_cols;
        if values.len() != expected {
            return Err(Error::FlexLen {
                kind: "factor field",
                actual: values.len(),
                expected,
            });
        }
        let name = name.to_lowercase();
        if name.is_empty() {
            return Err(Error::Invariant("factor field name cannot be empty".into()));
        }
        if self.fields.insert(name.clone(), values).is_some() {
            return Err(Error::Invariant(format!("duplicate factor field {name:?}")));
        }
        Ok(())
    }

    /// The time of each row as nanoseconds since 1970-01-01 on the wall
    /// clock of the market (local time, not UTC): what `DayLast`,
    /// `DayCumsum`, `Date` and `P` group and compare by.
    ///
    /// # Errors
    ///
    /// A length other than the row count, or times that go backwards.
    pub fn with_timestamps(mut self, nanos: &[i64]) -> Result<Self> {
        if nanos.len() != self.n_rows {
            return Err(Error::FlexLen {
                kind: "factor timestamps",
                actual: nanos.len(),
                expected: self.n_rows,
            });
        }
        if nanos.windows(2).any(|pair| pair[1] < pair[0]) {
            return Err(Error::Invariant(
                "factor timestamps must not decrease".into(),
            ));
        }
        let day = nanos
            .iter()
            .map(|&t| t.div_euclid(NANOS_PER_DAY) as f64)
            .collect();
        let minute = nanos
            .iter()
            .map(|&t| (t.rem_euclid(NANOS_PER_DAY) / NANOS_PER_MINUTE) as f64)
            .collect();
        self.calendar = Some(Arc::new((day, minute)));
        Ok(self)
    }

    /// The name of each symbol (column), which `ChangeInstrument("X", ...)`
    /// looks `X` up among.
    ///
    /// # Errors
    ///
    /// A count other than the column count, or a repeated name.
    pub fn with_symbols<S: AsRef<str>>(mut self, names: &[S]) -> Result<Self> {
        if names.len() != self.n_cols {
            return Err(Error::FlexLen {
                kind: "factor symbols",
                actual: names.len(),
                expected: self.n_cols,
            });
        }
        let names: Vec<String> = names.iter().map(|name| name.as_ref().to_owned()).collect();
        for (at, name) in names.iter().enumerate() {
            if names[..at].contains(name) {
                return Err(Error::Invariant(format!(
                    "duplicate factor symbol {name:?}"
                )));
            }
        }
        self.symbols = Some(names);
        Ok(self)
    }

    /// An instrument outside the panel, such as a benchmark index, that
    /// `ChangeInstrument(name, ...)` reads: one column over the same rows.
    /// It reads the panel's timestamps unless it has its own.
    ///
    /// # Errors
    ///
    /// A shape other than `(rows, 1)`, or a name already supplied.
    pub fn with_instrument(mut self, name: &str, inputs: FactorInputs<'a>) -> Result<Self> {
        if inputs.shape() != (self.n_rows, 1) {
            return Err(Error::Invariant(format!(
                "instrument {name:?} has shape {}x{}, expected {}x1",
                inputs.n_rows, inputs.n_cols, self.n_rows
            )));
        }
        if self.instruments.insert(name.to_owned(), inputs).is_some() {
            return Err(Error::Invariant(format!("duplicate instrument {name:?}")));
        }
        Ok(self)
    }

    /// The records of a point-in-time field (`$$field` in an expression),
    /// read by `P` and `PRef`. The name ends with `_q` (quarterly periods
    /// `YYYY * 100 + Q`) or `_a` (years), as in Qlib; leading `$` are ignored.
    ///
    /// # Errors
    ///
    /// A name without either suffix, a record whose symbol is out of range or
    /// whose period is not a quarter (or year) of 1900..=2099, or a field
    /// already supplied.
    pub fn with_pit(mut self, field: &str, records: Vec<PitRecord>) -> Result<Self> {
        let field = field.trim_start_matches('$').to_lowercase();
        pit::quarterly(&field)?;
        for record in &records {
            if record.col >= self.n_cols {
                return Err(Error::Invariant(format!(
                    "$${field} has a record for symbol {} of {}",
                    record.col, self.n_cols
                )));
            }
            pit::check_period(&field, record.period)?;
        }
        if self.pit.insert(field.clone(), records).is_some() {
            return Err(Error::Invariant(format!(
                "duplicate point-in-time field $${field}"
            )));
        }
        Ok(self)
    }

    /// `(rows, symbols)` of every supplied field.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.n_rows, self.n_cols)
    }

    pub(crate) fn days(&self) -> Option<&[f64]> {
        self.calendar.as_deref().map(|(day, _)| day.as_slice())
    }

    pub(crate) fn pit_records(&self, field: &str) -> Option<&[PitRecord]> {
        self.pit.get(field).map(Vec::as_slice)
    }

    /// An input field by name.
    pub(crate) fn field(&self, name: &str) -> Result<&'a [f64]> {
        self.fields
            .get(name)
            .copied()
            .ok_or_else(|| Error::Invariant(format!("factor input is missing field ${name}")))
    }

    /// A field by name: an input, or a hidden one, the calendar's or one a
    /// lifted subexpression computed into `extra`. A hidden field may hold
    /// one value per row, shared by every symbol.
    fn lookup<'s>(&'s self, name: &str, extra: &'s HashMap<String, Vec<f64>>) -> Result<&'s [f64]> {
        if let Some(values) = self.fields.get(name) {
            return Ok(values);
        }
        if let Some(values) = extra.get(name) {
            return Ok(values);
        }
        if name == DAY || name == MINUTE {
            let (day, minute) = self.calendar.as_deref().ok_or_else(|| {
                Error::Invariant(
                    "DayLast, DayCumsum and Date need the bar timestamps (FactorInputs::with_timestamps)"
                        .into(),
                )
            })?;
            return Ok(if name == DAY { day } else { minute });
        }
        Err(Error::Invariant(format!(
            "factor input is missing field ${name}"
        )))
    }

    /// Symbols `cols` only, with everything they read: the timestamps,
    /// the supplied instruments and their own point-in-time records.
    fn slice(&self, cols: Range<usize>) -> Result<Self> {
        let n_rows = self.n_rows;
        let mut part = Self::new(n_rows, cols.len())?;
        for (name, values) in &self.fields {
            part.add_field(name, &values[cols.start * n_rows..cols.end * n_rows])?;
        }
        part.calendar.clone_from(&self.calendar);
        part.symbols = self
            .symbols
            .as_ref()
            .map(|names| names[cols.clone()].to_vec());
        part.instruments.clone_from(&self.instruments);
        part.pit = self
            .pit
            .iter()
            .map(|(field, records)| {
                let kept = records
                    .iter()
                    .filter(|record| cols.contains(&record.col))
                    .map(|record| PitRecord {
                        col: record.col - cols.start,
                        ..*record
                    })
                    .collect();
                (field.clone(), kept)
            })
            .collect();
        Ok(part)
    }

    /// As [`Self::slice`], plus each instrument of `needed` that is a symbol
    /// of this panel, so a part of the panel can still read it.
    pub(crate) fn columns(&self, cols: Range<usize>, needed: &[String]) -> Result<Self> {
        let mut part = self.slice(cols)?;
        for name in needed {
            if part.instruments.contains_key(name) {
                continue;
            }
            if let Some(col) = self.symbol(name) {
                part.instruments
                    .insert(name.clone(), self.slice(col..col + 1)?);
            }
        }
        Ok(part)
    }

    fn symbol(&self, name: &str) -> Option<usize> {
        self.symbols
            .as_ref()?
            .iter()
            .position(|symbol| symbol == name)
    }

    /// The one-column inputs of instrument `name`: a supplied instrument
    /// (reading this panel's timestamps and instruments unless it has its
    /// own), else the panel's symbol of that name.
    pub(crate) fn instrument(&self, name: &str, needed: &[String]) -> Result<Self> {
        if let Some(inputs) = self.instruments.get(name) {
            let mut inputs = inputs.clone();
            if inputs.calendar.is_none() {
                inputs.calendar.clone_from(&self.calendar);
            }
            for (other, supplied) in &self.instruments {
                inputs
                    .instruments
                    .entry(other.clone())
                    .or_insert_with(|| supplied.clone());
            }
            return Ok(inputs);
        }
        let col = self.symbol(name).ok_or_else(|| {
            Error::Invariant(format!(
                "instrument {name:?} is neither a symbol of the panel nor supplied with with_instrument"
            ))
        })?;
        self.columns(col..col + 1, needed)
    }

    /// Iterates over the named column-major input fields.
    pub fn fields(&self) -> impl Iterator<Item = (&str, &'a [f64])> + '_ {
        self.fields
            .iter()
            .map(|(name, values)| (name.as_str(), *values))
    }
}

/// Named factor matrices returned by [`FactorSet::evaluate`].
#[derive(Debug, Clone)]
pub struct FactorValues {
    names: Vec<String>,
    values: Vec<Arc<Matrix>>,
}

impl FactorValues {
    /// Builds backend output from names and matrices in the same order.
    ///
    /// # Errors
    ///
    /// Returns an error when the name and value counts differ.
    pub fn from_matrices(names: Vec<String>, values: Vec<Matrix>) -> Result<Self> {
        if names.len() != values.len() {
            return Err(Error::Invariant(format!(
                "factor backend returned {} names but {} matrices",
                names.len(),
                values.len()
            )));
        }
        Ok(Self {
            names,
            values: values.into_iter().map(Arc::new).collect(),
        })
    }

    /// Looks up one factor by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Matrix> {
        self.names
            .iter()
            .position(|candidate| candidate == name)
            .map(|index| self.values[index].as_ref())
    }

    /// Iterates in factor-set order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&str, &Matrix)> {
        self.names
            .iter()
            .zip(&self.values)
            .map(|(name, value)| (name.as_str(), value.as_ref()))
    }

    /// The values of symbols `cols` only.
    ///
    /// # Errors
    ///
    /// A range past the last symbol.
    pub fn columns(&self, cols: Range<usize>) -> Result<Self> {
        let matrices = self
            .values
            .iter()
            .map(|m| {
                if cols.end > m.n_cols() || cols.start > cols.end {
                    return Err(Error::Invariant(format!(
                        "symbols {cols:?} of {}",
                        m.n_cols()
                    )));
                }
                let n_rows = m.n_rows();
                Matrix::from_column_major(
                    m.as_slice()[cols.start * n_rows..cols.end * n_rows].to_vec(),
                    n_rows,
                    cols.len(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Self::from_matrices(self.names.clone(), matrices)
    }

    /// Takes the matrices in factor-set order. A matrix is moved out rather
    /// than copied unless two factors share the same expression.
    pub fn into_matrices(self) -> impl ExactSizeIterator<Item = (String, Matrix)> {
        self.names
            .into_iter()
            .zip(self.values)
            .map(|(name, value)| {
                (
                    name,
                    Arc::try_unwrap(value).unwrap_or_else(|shared| (*shared).clone()),
                )
            })
    }
}

/// The hidden fields of the subexpressions lifted out of `set`: each
/// instrument's series, one value per row, and each `P`'s panel.
fn lifted_fields(set: &FactorSet, inputs: &FactorInputs<'_>) -> Result<HashMap<String, Vec<f64>>> {
    let mut extra = HashMap::new();
    if set.lifted.is_empty() {
        return Ok(extra);
    }
    let needed = set.instrument_names();
    for instrument in &set.lifted.instruments {
        let own = inputs.instrument(&instrument.name, &needed)?;
        for (name, values) in instrument.set.evaluate(&own)?.into_matrices() {
            extra.insert(name, values.into_vec());
        }
    }
    for job in &set.lifted.pits {
        extra.insert(job.field.clone(), pit::evaluate(job, inputs)?);
    }
    Ok(extra)
}

/// Symbol `col`'s series of a field; a field of one value per row is every
/// symbol's.
fn column(values: &[f64], col: usize, n_rows: usize) -> &[f64] {
    if values.len() == n_rows {
        values
    } else {
        &values[col * n_rows..(col + 1) * n_rows]
    }
}

pub(crate) fn evaluate(set: &FactorSet, inputs: &FactorInputs<'_>) -> Result<FactorValues> {
    let graph = &set.graph;
    let (n_rows, n_cols) = (inputs.n_rows, inputs.n_cols);
    let extra = lifted_fields(set, inputs)?;
    let fields = graph
        .fields
        .iter()
        .map(|name| inputs.lookup(name, &extra))
        .collect::<Result<Vec<&[f64]>>>()?;

    let outputs = match &set.plan {
        // Fewer symbols than threads: split the factors as well, so the pool
        // has work. Every node computes the same values in any graph, so the
        // result is the same bit for bit.
        None if n_cols < rayon::current_num_threads() && set.factors.len() > 1 => {
            let parts = set.parts.get_or_init(|| split(set));
            run_parts(parts, graph.outputs.len(), inputs, &extra, n_rows, n_cols)?
        }
        None => run_graph(graph, &fields, n_rows, n_cols),
        Some(staged) => {
            let starts = if set.warmup() {
                Some(Warmup::starts(graph, inputs)?)
            } else {
                None
            };
            let warm = starts.as_deref().map(|starts| (&set.warm, starts));
            run_staged(staged, graph.outputs.len(), &fields, n_rows, n_cols, warm)
        }
    };
    let names = set.names().map(str::to_owned).collect();
    let matrices = outputs
        .into_iter()
        .map(|cells| Matrix::from_column_major(cells, n_rows, n_cols))
        .collect::<Result<Vec<_>>>()?;
    FactorValues::from_matrices(names, matrices)
}

/// A group of factors compiled on its own.
#[derive(Debug, Clone)]
pub(crate) struct Part {
    graph: Graph,
    /// Each output's position in the whole set.
    positions: Vec<usize>,
}

/// The factors in contiguous groups of about equal size. Neighbouring
/// factors tend to share subexpressions (Alpha158's `MA5`, `MA10`, ... read
/// the same close), so contiguous groups recompute little.
fn split(set: &FactorSet) -> Vec<Part> {
    const PARTS: usize = 16;
    let n = set.factors.len();
    let per = n.div_ceil(PARTS.min(n));
    (0..n)
        .step_by(per)
        .map(|start| {
            let positions: Vec<usize> = (start..(start + per).min(n)).collect();
            let graph = Graph::compile(
                positions
                    .iter()
                    .map(|&k| (set.factors[k].0.as_str(), &set.rewritten[k])),
            )
            .expect("a subset of a compiled set compiles");
            Part { graph, positions }
        })
        .collect()
}

/// Every (group, symbol) pair as one task.
fn run_parts(
    parts: &[Part],
    n_outputs: usize,
    inputs: &FactorInputs<'_>,
    extra: &HashMap<String, Vec<f64>>,
    n_rows: usize,
    n_cols: usize,
) -> Result<Vec<Vec<f64>>> {
    let mut outputs: Vec<Vec<f64>> = (0..n_outputs).map(|_| vec![0.0; n_rows * n_cols]).collect();
    if n_rows == 0 {
        return Ok(outputs);
    }
    let mut cells: Vec<Vec<Option<&mut [f64]>>> = outputs
        .iter_mut()
        .map(|output| output.chunks_mut(n_rows).map(Some).collect())
        .collect();
    let mut tasks = Vec::with_capacity(parts.len() * n_cols);
    for part in parts {
        let fields = part
            .graph
            .fields
            .iter()
            .map(|name| inputs.lookup(name, extra))
            .collect::<Result<Vec<&[f64]>>>()?;
        #[allow(
            clippy::needless_range_loop,
            reason = "col picks one cell of every output and one slice of every field"
        )]
        for col in 0..n_cols {
            let out: Vec<&mut [f64]> = part
                .positions
                .iter()
                .map(|&p| cells[p][col].take().expect("each cell has one writer"))
                .collect();
            let mut slices = Vec::with_capacity(fields.len());
            for values in &fields {
                slices.push(column(values, col, n_rows));
            }
            tasks.push((&part.graph, slices, out));
        }
    }
    tasks.into_par_iter().for_each(|(graph, column, mut out)| {
        let mut scratch = Scratch::new(graph.slots, n_rows);
        run_column(graph, &column, &mut scratch, &mut out);
    });
    Ok(outputs)
}

/// Runs the stages in order: each per-symbol stage over every column, each
/// cross-sectional node over every row.
/// With `warm`, a cross-sectional node reads each panel blanked where its
/// windows have not yet filled, so a symbol still warming up is skipped.
fn run_staged(
    staged: &Staged,
    n_outputs: usize,
    fields: &[&[f64]],
    n_rows: usize,
    n_cols: usize,
    warm: Option<(&Warmup, &[Vec<usize>])>,
) -> Vec<Vec<f64>> {
    let mut outputs: Vec<Option<Vec<f64>>> = vec![None; n_outputs];
    let mut panels: Vec<Option<Vec<f64>>> = vec![None; staged.panels];
    for (at, step) in staged.steps.iter().enumerate() {
        let read = |source: &Source| -> Arg<'_> {
            match *source {
                Source::Field(index) => Arg::Series(fields[index]),
                Source::Const(value) => Arg::Scalar(value),
                Source::Panel(k) => Arg::Series(
                    panels[k]
                        .as_deref()
                        .expect("a panel is written before it is read"),
                ),
            }
        };
        match step {
            Step::Columns {
                graph,
                sources,
                dests,
            } => {
                let inputs: Vec<&[f64]> = sources
                    .iter()
                    .map(|source| match read(source) {
                        Arg::Series(values) => values,
                        _ => unreachable!("a stage's fields are series"),
                    })
                    .collect();
                let results = run_graph(graph, &inputs, n_rows, n_cols);
                for (values, dest) in results.into_iter().zip(dests) {
                    match *dest {
                        Dest::Output(position) => outputs[position] = Some(values),
                        Dest::Panel(k) => panels[k] = Some(values),
                    }
                }
            }
            Step::Cross {
                op,
                inputs,
                panel,
                outputs: positions,
            } => {
                let blanked: Vec<Option<Vec<f64>>> = inputs
                    .iter()
                    .map(|source| match (source, warm) {
                        (Source::Panel(k), Some((warm, starts))) => {
                            let mut copy = panels[*k]
                                .clone()
                                .expect("a panel is written before it is read");
                            warm.mask(staged.nodes[*k], starts, &mut copy, n_rows);
                            Some(copy)
                        }
                        _ => None,
                    })
                    .collect();
                let input = |at: usize| match &blanked[at] {
                    Some(values) => Arg::Series(values),
                    None => read(&inputs[at]),
                };
                let x = input(0);
                let y = if inputs.len() > 1 {
                    input(1)
                } else {
                    Arg::Scalar(f64::NAN)
                };
                let values = cross_panel(*op, x, y, n_rows, n_cols);
                for &position in positions {
                    outputs[position] = Some(values.clone());
                }
                panels[*panel] = Some(values);
            }
        }
        for &k in &staged.release[at] {
            panels[k] = None;
        }
    }
    outputs
        .into_iter()
        .map(|values| values.expect("every factor is written by some step"))
        .collect()
}

/// A cross-sectional node over every row, rows in parallel. Rows are built
/// row-major, so each is one contiguous slice, then transposed.
fn cross_panel(op: Cross, x: Arg<'_>, y: Arg<'_>, n_rows: usize, n_cols: usize) -> Vec<f64> {
    let at = |arg: Arg<'_>, row: usize, col: usize| match arg {
        Arg::Scalar(value) => value,
        Arg::Series(values) => column(values, col, n_rows)[row],
        Arg::Shifted { .. } => unreachable!("cross-sectional inputs are whole series"),
    };
    let mut rows = vec![0.0; n_rows * n_cols];
    if n_cols > 0 {
        rows.par_chunks_mut(n_cols).enumerate().for_each_init(
            || {
                (
                    vec![0.0; n_cols],
                    vec![0.0; n_cols],
                    cross::Scratch::default(),
                )
            },
            |(xs, ys, scratch), (row, out)| {
                for col in 0..n_cols {
                    xs[col] = at(x, row, col);
                    ys[col] = at(y, row, col);
                }
                cross::row(op, xs, ys, out, scratch);
            },
        );
    }
    let mut columns = vec![0.0; n_rows * n_cols];
    if n_rows > 0 {
        columns
            .par_chunks_mut(n_rows)
            .enumerate()
            .for_each(|(col, column)| {
                for (row, slot) in column.iter_mut().enumerate() {
                    *slot = rows[row * n_cols + col];
                }
            });
    }
    columns
}

/// Every output of a graph without cross-sectional nodes, one column-major
/// series each, columns in parallel.
fn run_graph(graph: &Graph, fields: &[&[f64]], n_rows: usize, n_cols: usize) -> Vec<Vec<f64>> {
    // One allocation per output rather than cloning a zeroed buffer, so the
    // pages are zeroed lazily by the columns that write them, in parallel.
    let mut outputs: Vec<Vec<f64>> = (0..graph.outputs.len())
        .map(|_| vec![0.0; n_rows * n_cols])
        .collect();
    if n_rows > 0 {
        // Each column's slice of every output, so columns can be filled in
        // parallel without sharing a mutable matrix.
        let mut columns: Vec<Vec<&mut [f64]>> = (0..n_cols)
            .map(|_| Vec::with_capacity(outputs.len()))
            .collect();
        for output in &mut outputs {
            for (column, cells) in columns.iter_mut().zip(output.chunks_mut(n_rows)) {
                column.push(cells);
            }
        }
        columns.into_par_iter().enumerate().for_each_init(
            || Scratch::new(graph.slots, n_rows),
            |scratch, (col, mut out)| {
                let fields: Vec<&[f64]> = fields
                    .iter()
                    .map(|values| column(values, col, n_rows))
                    .collect();
                run_column(graph, &fields, scratch, &mut out);
            },
        );
    }
    outputs
}

/// One worker's scratch: the graph's slots, each a series of `n_rows`, plus a
/// spare for materializing a literal or a view where a kernel needs a series.
/// Allocated once per worker and reused across every column it evaluates.
struct Scratch {
    slots: Vec<Vec<f64>>,
    spare: Vec<Vec<f64>>,
}

impl Scratch {
    fn new(slots: usize, n_rows: usize) -> Self {
        Self {
            slots: (0..slots).map(|_| vec![0.0; n_rows]).collect(),
            spare: (0..2).map(|_| vec![0.0; n_rows]).collect(),
        }
    }
}

/// An operand of one symbol's step: a literal, a series, or a series read
/// `offset` rows late (`Ref`), which is `NaN` for the first `offset` rows.
#[derive(Clone, Copy)]
enum Arg<'s> {
    Scalar(f64),
    Series(&'s [f64]),
    Shifted { values: &'s [f64], offset: usize },
}

/// An operand restricted to a row range over which it is one plain thing.
#[derive(Clone, Copy)]
enum Seg<'s> {
    Scalar(f64),
    Slice(&'s [f64]),
}

impl<'s> Arg<'s> {
    /// Row boundaries inside `0..n_rows` where this operand changes form.
    fn cut(self, n_rows: usize) -> Option<usize> {
        match self {
            Self::Shifted { offset, .. } if offset < n_rows => Some(offset),
            _ => None,
        }
    }

    /// The operand over `start..end`, which must not straddle its cut.
    fn segment(self, start: usize, end: usize) -> Seg<'s> {
        match self {
            Self::Scalar(value) => Seg::Scalar(value),
            Self::Series(values) => Seg::Slice(&values[start..end]),
            Self::Shifted { values, offset } => {
                if offset >= end {
                    Seg::Scalar(f64::NAN)
                } else {
                    Seg::Slice(&values[start - offset..end - offset])
                }
            }
        }
    }

    /// The operand as a series, materializing a literal or a view into `buffer`.
    fn series<'b>(self, buffer: &'b mut [f64]) -> &'b [f64]
    where
        's: 'b,
    {
        match self {
            Self::Series(values) => values,
            Self::Scalar(value) => {
                buffer.fill(value);
                buffer
            }
            Self::Shifted { values, offset } => {
                let n = buffer.len();
                let offset = offset.min(n);
                buffer[..offset].fill(f64::NAN);
                buffer[offset..].copy_from_slice(&values[..n - offset]);
                buffer
            }
        }
    }
}

/// Splits `0..n_rows` at every operand's cut, so each piece can run as a
/// straight loop over plain slices and scalars.
fn segments(args: &[Arg<'_>], n_rows: usize) -> impl Iterator<Item = (usize, usize)> {
    let mut cuts = [n_rows; 4];
    let mut count = 0;
    for arg in args {
        if let Some(cut) = arg.cut(n_rows) {
            cuts[count] = cut;
            count += 1;
        }
    }
    cuts.sort_unstable();
    let mut start = 0;
    cuts.into_iter().filter_map(move |end| {
        let piece = (end > start).then_some((start, end));
        start = start.max(end);
        piece
    })
}

/// Evaluates the whole graph for one symbol, writing each factor's column.
fn run_column(graph: &Graph, fields: &[&[f64]], scratch: &mut Scratch, out: &mut [&mut [f64]]) {
    let n_rows = out
        .first()
        .map_or_else(|| fields[0].len(), |column| column.len());
    for node in &graph.nodes {
        fn arg<'s>(
            graph: &Graph,
            fields: &[&'s [f64]],
            slots: &'s [Vec<f64>],
            id: NodeId,
        ) -> Arg<'s> {
            let input = &graph.nodes[id];
            let series = |id: NodeId| -> &'s [f64] {
                match (graph.nodes[id].op, graph.nodes[id].slot) {
                    (Op::Field(index), _) => fields[index],
                    (_, Some(slot)) => slots[slot].as_slice(),
                    (_, None) => unreachable!("computed inputs have slots"),
                }
            };
            match (input.op, input.view) {
                (Op::Const(bits), _) => Arg::Scalar(f64::from_bits(bits)),
                (_, Some((base, offset))) => Arg::Shifted {
                    values: series(base),
                    offset,
                },
                _ => Arg::Series(series(id)),
            }
        }
        let is_source = matches!(node.op, Op::Field(_) | Op::Const(_)) || node.view.is_some();
        // Take the spare and the target out of the scratch so the inputs can
        // be borrowed; the schedule never gives a node an input's slot.
        let mut spare = std::mem::take(&mut scratch.spare);
        let mut target = node
            .slot
            .map(|slot| std::mem::take(&mut scratch.slots[slot]));
        if is_source {
            // Read in place by later nodes; written only where a factor is
            // the bare field, literal or shifted series.
            if let Some((&first, rest)) = node.outputs.split_first() {
                let value = match node.op {
                    Op::Field(index) => Arg::Series(fields[index]),
                    Op::Const(bits) => Arg::Scalar(f64::from_bits(bits)),
                    _ => {
                        let (base, offset) =
                            node.view.expect("a source is a field, literal or view");
                        arg(graph, fields, &scratch.slots, base).shifted(offset)
                    }
                };
                out[first].copy_from_slice(value.series(&mut spare[0]));
                for &position in rest {
                    let (done, later) = out.split_at_mut(position);
                    later[0].copy_from_slice(done[first]);
                }
            }
        } else {
            let mut args = [Arg::Scalar(0.0); 3];
            for (slot, &id) in args.iter_mut().zip(&node.inputs) {
                *slot = arg(graph, fields, &scratch.slots, id);
            }
            let args = &args[..node.inputs.len()];
            match &mut target {
                Some(target) => {
                    step(node.op, args, target, &mut spare);
                    for &position in &node.outputs {
                        out[position].copy_from_slice(target);
                    }
                }
                None => {
                    // Nothing reads this node: compute it straight into its output.
                    let (&first, rest) = node
                        .outputs
                        .split_first()
                        .expect("a computed node is read or is an output");
                    step(node.op, args, out[first], &mut spare);
                    for &position in rest {
                        let (done, later) = out.split_at_mut(position);
                        later[0].copy_from_slice(done[first]);
                    }
                }
            }
        }
        scratch.spare = spare;
        if let (Some(slot), Some(target)) = (node.slot, target) {
            scratch.slots[slot] = target;
        }
        debug_assert!(n_rows == 0 || node.outputs.iter().all(|&p| out[p].len() == n_rows));
    }
}

impl<'s> Arg<'s> {
    /// This operand read `offset` rows late.
    fn shifted(self, offset: usize) -> Self {
        match self {
            Self::Scalar(value) => Self::Scalar(value),
            Self::Series(values) => Self::Shifted { values, offset },
            Self::Shifted { values, offset: o } => Self::Shifted {
                values,
                offset: o + offset,
            },
        }
    }
}

/// Computes one node's series for one symbol into `target`.
fn step(op: Op, args: &[Arg<'_>], target: &mut [f64], spare: &mut [Vec<f64>]) {
    let n_rows = target.len();
    let (first, second) = spare.split_at_mut(1);
    let (first, second) = (first[0].as_mut_slice(), second[0].as_mut_slice());
    match op {
        Op::Unary(op) => unary_series(op, args[0], target),
        Op::Binary(op) => binary_series(op, args[0], args[1], target),
        Op::If => {
            for (start, end) in segments(args, n_rows) {
                let (c, a, b) = (
                    args[0].segment(start, end),
                    args[1].segment(start, end),
                    args[2].segment(start, end),
                );
                for (row, slot) in target[start..end].iter_mut().enumerate() {
                    *slot = graph::if_else(c.at(row), a.at(row), b.at(row));
                }
            }
        }
        Op::Ref(offset) => {
            // Only a future-reading Ref is computed; the past is a view.
            let column = args[0].series(first);
            let ahead = offset.unsigned_abs() as usize;
            let kept = n_rows.saturating_sub(ahead);
            target[..kept].copy_from_slice(&column[ahead..]);
            target[kept..].fill(f64::NAN);
        }
        Op::Delta(offset) => binary_series(BinaryOp::Sub, args[0], args[0].shifted(offset), target),
        Op::First => {
            if let Some(&value) = args[0].series(first).first() {
                target.fill(value);
            }
        }
        Op::Ema(alpha) => {
            let column = args[0].series(first);
            match alpha {
                None => ema_expanding_into(column, target),
                Some(bits) => ema_adjusted_into(column, f64::from_bits(bits), target),
            }
        }
        Op::Rolling(kind, window) => {
            let column = args[0].series(first);
            rolling_into(kind, column, window.resolve(n_rows), target);
        }
        Op::Pair { corr, window } => {
            let left = args[0].series(first);
            let right = args[1].series(second);
            let pairs: Vec<(f64, f64)> = left.iter().copied().zip(right.iter().copied()).collect();
            let window = window.resolve(n_rows);
            if corr {
                slide_into(&pairs, window, &mut Pair::corr(), target);
            } else {
                slide_into(&pairs, window, &mut Pair::cov(), target);
            }
        }
        Op::Fill { forward } => fill_into(args[0].series(first), forward, target),
        Op::DayLast => day_last_into(args[0].series(first), calendar(args[1]), target),
        Op::DayCumsum { start, end, step } => day_cumsum_into(
            args[0].series(first),
            calendar(args[1]),
            calendar(args[2]),
            (start, end, step),
            target,
        ),
        Op::Cut { left, right } => {
            target.copy_from_slice(args[0].series(first));
            target[..left.min(n_rows)].fill(f64::NAN);
            target[n_rows.saturating_sub(right)..].fill(f64::NAN);
        }
        Op::Field(_) | Op::Const(_) => unreachable!("fields and literals are not computed"),
        Op::Cross(_) => unreachable!("cross-sectional nodes run between stages"),
    }
}

/// A calendar input, which is always a field read in place.
fn calendar(arg: Arg<'_>) -> &[f64] {
    match arg {
        Arg::Series(values) => values,
        _ => unreachable!("calendar inputs are fields"),
    }
}

/// `NaN` replaced by the last value present before it (`forward`) or the
/// next one after it; `NaN` where there is none.
fn fill_into(input: &[f64], forward: bool, out: &mut [f64]) {
    let mut last = f64::NAN;
    let mut fill = |(slot, &value): (&mut f64, &f64)| {
        if !value.is_nan() {
            last = value;
        }
        *slot = last;
    };
    if forward {
        out.iter_mut().zip(input).for_each(&mut fill);
    } else {
        out.iter_mut()
            .rev()
            .zip(input.iter().rev())
            .for_each(&mut fill);
    }
}

/// The rows of each day: runs of equal `day`.
fn days(day: &[f64]) -> impl Iterator<Item = Range<usize>> + '_ {
    let mut start = 0;
    std::iter::from_fn(move || {
        if start >= day.len() {
            return None;
        }
        let first = day[start];
        let len = day[start..]
            .iter()
            .take_while(|&&d| d.to_bits() == first.to_bits())
            .count();
        let rows = start..start + len;
        start += len;
        Some(rows)
    })
}

/// Each day's last value present (pandas `groupby(...).transform("last")`),
/// `NaN` for a day without one.
fn day_last_into(input: &[f64], day: &[f64], out: &mut [f64]) {
    for rows in days(day) {
        let last = input[rows.clone()]
            .iter()
            .rev()
            .copied()
            .find(|value| !value.is_nan())
            .unwrap_or(f64::NAN);
        out[rows].fill(last);
    }
}

/// Qlib's `DayCumsum`: within each day, zero before the bar that contains
/// `start`, the running sum of the values present from that bar through the
/// last bar that starts by `end` (`NaN` where the value is missing, the sum
/// carrying on), and zero after it.
fn day_cumsum_into(
    input: &[f64],
    minute: &[f64],
    day: &[f64],
    (start, end, step): (u32, u32, u32),
    out: &mut [f64],
) {
    let (start, end, step) = (f64::from(start), f64::from(end), f64::from(step));
    for rows in days(day) {
        let mut sum = 0.0;
        for row in rows {
            let bar = minute[row];
            out[row] = if bar + step <= start || bar > end {
                0.0
            } else if input[row].is_nan() {
                f64::NAN
            } else {
                sum += input[row];
                sum
            };
        }
    }
}

impl Seg<'_> {
    fn at(self, row: usize) -> f64 {
        match self {
            Self::Scalar(value) => value,
            Self::Slice(values) => values[row],
        }
    }
}

/// An elementwise unary operator with the operator and operand kind resolved
/// outside the loop.
fn unary_series(op: UnaryOp, input: Arg<'_>, target: &mut [f64]) {
    fn apply(input: Arg<'_>, target: &mut [f64], f: impl Fn(f64) -> f64) {
        for (start, end) in segments(&[input], target.len()) {
            match input.segment(start, end) {
                Seg::Slice(values) => {
                    for (slot, &value) in target[start..end].iter_mut().zip(values) {
                        *slot = f(value);
                    }
                }
                Seg::Scalar(value) => target[start..end].fill(f(value)),
            }
        }
    }
    macro_rules! dispatch {
        ($($variant:ident),*) => {
            match op {
                $(UnaryOp::$variant => apply(input, target, |a| graph::unary(UnaryOp::$variant, a)),)*
            }
        };
    }
    dispatch!(Neg, Abs, Sign, Log, Not, Exp, Sqrt, IsNan, IsInf);
}

/// An elementwise binary operator with the operator and operand kinds
/// resolved outside the loop, so each case compiles to a straight loop.
fn binary_series(op: BinaryOp, left: Arg<'_>, right: Arg<'_>, target: &mut [f64]) {
    fn apply(left: Arg<'_>, right: Arg<'_>, target: &mut [f64], f: impl Fn(f64, f64) -> f64) {
        for (start, end) in segments(&[left, right], target.len()) {
            let out = &mut target[start..end];
            match (left.segment(start, end), right.segment(start, end)) {
                (Seg::Slice(a), Seg::Slice(b)) => {
                    for (slot, (&a, &b)) in out.iter_mut().zip(a.iter().zip(b)) {
                        *slot = f(a, b);
                    }
                }
                (Seg::Slice(a), Seg::Scalar(b)) => {
                    for (slot, &a) in out.iter_mut().zip(a) {
                        *slot = f(a, b);
                    }
                }
                (Seg::Scalar(a), Seg::Slice(b)) => {
                    for (slot, &b) in out.iter_mut().zip(b) {
                        *slot = f(a, b);
                    }
                }
                (Seg::Scalar(a), Seg::Scalar(b)) => out.fill(f(a, b)),
            }
        }
    }
    macro_rules! dispatch {
        ($($variant:ident),*) => {
            match op {
                $(BinaryOp::$variant => apply(left, right, target, |a, b| {
                    graph::binary(BinaryOp::$variant, a, b)
                }),)*
            }
        };
    }
    dispatch!(
        Add,
        Sub,
        Mul,
        Div,
        Pow,
        Gt,
        Lt,
        Ge,
        Le,
        Eq,
        Ne,
        And,
        Or,
        Greater,
        Less,
        FillNa,
        Mask,
        SignedPower
    );
}

/// [`rolling_into`] as a fresh series, for tests.
#[cfg(test)]
fn rolling(kind: Rolling, input: &[f64], window: usize) -> Vec<f64> {
    let mut out = vec![0.0; input.len()];
    rolling_into(kind, input, window, &mut out);
    out
}

fn rolling_into(kind: Rolling, input: &[f64], window: usize, out: &mut [f64]) {
    let window = window.max(1);
    let checked = "window and min_periods are valid";
    match kind {
        Rolling::Mean => {
            out.copy_from_slice(&rolling_mean_1d(input, window, Some(1)).expect(checked));
        }
        Rolling::Max => {
            out.copy_from_slice(&rolling_max_1d(input, window, Some(1)).expect(checked));
        }
        Rolling::Min => {
            out.copy_from_slice(&rolling_min_1d(input, window, Some(1)).expect(checked));
        }
        Rolling::Std => slide_into(input, window, &mut Deviation::std(), out),
        Rolling::Var => slide_into(input, window, &mut Deviation::var(), out),
        Rolling::Sum => slide_into(input, window, &mut Sum::default(), out),
        Rolling::Count => slide_into(input, window, &mut Count::default(), out),
        Rolling::Quantile(q) => {
            ordered_into(input, window, Order::Quantile(f64::from_bits(q)), out);
        }
        Rolling::Med => ordered_into(input, window, Order::Quantile(0.5), out),
        Rolling::Mad => ordered_into(input, window, Order::Mad, out),
        Rolling::Rank => ordered_into(input, window, Order::Rank, out),
        Rolling::Skew => slide_into(input, window, &mut Shape::skew(), out),
        Rolling::Kurt => slide_into(input, window, &mut Shape::kurt(), out),
        Rolling::Wma => slide_into(input, window, &mut Wma::default(), out),
        Rolling::IdxMax => rolling_arg_extreme_into(input, window, false, out),
        Rolling::IdxMin => rolling_arg_extreme_into(input, window, true, out),
        Rolling::Slope => slide_into(input, window, &mut Regression::new(Fit::Slope), out),
        Rolling::Rsquare => slide_into(input, window, &mut Regression::new(Fit::Rsquare), out),
        Rolling::Resi => slide_into(input, window, &mut Regression::new(Fit::Resi), out),
        Rolling::Decay => decay_into(input, window, out),
        Rolling::Prod => prod_into(input, window, out),
    }
}

/// Linearly decaying weights `1..=len` over each trailing window, newest
/// heaviest, normalized over the values present. A direct `O(window)` sum per
/// row: the windows these are used with are short, and a running sum would
/// carry rounding from rows long gone.
fn decay_into(input: &[f64], window: usize, out: &mut [f64]) {
    for (row, slot) in out.iter_mut().enumerate() {
        let start = (row + 1).saturating_sub(window);
        let (mut num, mut den) = (0.0, 0.0);
        for (k, &value) in input[start..=row].iter().enumerate() {
            if !value.is_nan() {
                let weight = (k + 1) as f64;
                num += weight * value;
                den += weight;
            }
        }
        *slot = if den > 0.0 { num / den } else { f64::NAN };
    }
}

/// The product of the values present in each trailing window; `NaN` when
/// there are none. Direct, since a zero makes a running quotient useless.
fn prod_into(input: &[f64], window: usize, out: &mut [f64]) {
    for (row, slot) in out.iter_mut().enumerate() {
        let start = (row + 1).saturating_sub(window);
        let mut product = 1.0;
        let mut any = false;
        for &value in &input[start..=row] {
            if !value.is_nan() {
                product *= value;
                any = true;
            }
        }
        *slot = if any { product } else { f64::NAN };
    }
}

#[cfg(any(test, feature = "polars"))]
pub(crate) fn quantile(values: &[f64], q: f64) -> f64 {
    let mut values: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    values.sort_by(f64::total_cmp);
    if values.is_empty() {
        return f64::NAN;
    }
    let index = (values.len() - 1) as f64 * q;
    let lower = index.floor() as usize;
    let fraction = index - lower as f64;
    values[lower] + fraction * (values[index.ceil() as usize] - values[lower])
}

#[cfg(any(test, feature = "polars"))]
pub(crate) fn mad(values: &[f64]) -> f64 {
    let values: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| !value.is_nan())
        .collect();
    if values.is_empty() {
        return f64::NAN;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|value| (value - mean).abs()).sum::<f64>() / values.len() as f64
}

#[cfg(any(test, feature = "polars"))]
pub(crate) fn skew(values: &[f64]) -> f64 {
    let values: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| !value.is_nan())
        .collect();
    let n = values.len() as f64;
    if values.len() < 3 {
        return f64::NAN;
    }
    let mean = values.iter().sum::<f64>() / n;
    let m2 = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>();
    let m3 = values
        .iter()
        .map(|value| (value - mean).powi(3))
        .sum::<f64>();
    n * (n - 1.0).sqrt() / (n - 2.0) * m3 / m2.powf(1.5)
}

#[cfg(any(test, feature = "polars"))]
pub(crate) fn kurt(values: &[f64]) -> f64 {
    let values: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| !value.is_nan())
        .collect();
    let n = values.len() as f64;
    if values.len() < 4 {
        return f64::NAN;
    }
    let mean = values.iter().sum::<f64>() / n;
    let m2 = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>();
    let m4 = values
        .iter()
        .map(|value| (value - mean).powi(4))
        .sum::<f64>();
    let biased_fisher = n * m4 / (m2 * m2) - 3.0;
    (n - 1.0) / ((n - 2.0) * (n - 3.0)) * ((n + 1.0) * biased_fisher + 6.0)
}

#[cfg(any(test, feature = "polars"))]
pub(crate) fn wma(values: &[f64]) -> f64 {
    let weight_sum = (values.len() * (values.len() + 1) / 2) as f64;
    let mut sum = 0.0;
    let mut count = 0;
    for (index, &value) in values.iter().enumerate() {
        if !value.is_nan() {
            sum += (index + 1) as f64 / weight_sum * value;
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

#[allow(
    clippy::float_cmp,
    reason = "zero denominator exactly means no EMA observation has appeared"
)]
fn ema_adjusted_into(values: &[f64], alpha: f64, out: &mut [f64]) {
    let decay = 1.0 - alpha;
    let (mut numerator, mut denominator) = (0.0, 0.0);
    for (&value, slot) in values.iter().zip(out) {
        numerator *= decay;
        denominator *= decay;
        if !value.is_nan() {
            numerator += value;
            denominator += 1.0;
        }
        *slot = if denominator == 0.0 {
            f64::NAN
        } else {
            numerator / denominator
        };
    }
}

/// Qlib's `EMA(x, 0)`: the decay depends on how many rows have been seen, so
/// every row is a fresh weighted average of its whole prefix.
#[cfg(feature = "polars")]
pub(crate) fn ema_expanding(values: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; values.len()];
    ema_expanding_into(values, &mut out);
    out
}

fn ema_expanding_into(values: &[f64], out: &mut [f64]) {
    for (length, slot) in (1..=values.len()).zip(out) {
        *slot = if values[..length].iter().all(|value| value.is_nan()) {
            f64::NAN
        } else {
            let decay = 1.0 - 2.0 / (1.0 + length as f64);
            let denominator: f64 = (0..length)
                .map(|index| decay.powi((length - index - 1) as i32))
                .sum();
            values[..length]
                .iter()
                .enumerate()
                .filter(|(_, value)| !value.is_nan())
                .map(|(index, value)| decay.powi((length - index - 1) as i32) / denominator * value)
                .sum()
        };
    }
}

#[cfg(any(test, feature = "polars"))]
#[allow(
    clippy::float_cmp,
    reason = "Qlib's average-rank tie groups are defined by exact equality"
)]
pub(crate) fn rank_last(values: &[f64]) -> f64 {
    let Some(&last) = values.last() else {
        return f64::NAN;
    };
    if last.is_nan() {
        return f64::NAN;
    }
    let clean: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    let below = clean.iter().filter(|&&value| value < last).count() as f64;
    let equal = clean.iter().filter(|&&value| value == last).count() as f64;
    (below + (equal + 1.0) / 2.0) / clean.len() as f64
}

#[cfg(any(test, feature = "polars"))]
pub(crate) fn arg_extreme(values: &[f64], minimum: bool) -> f64 {
    let mut best = 0;
    for index in 1..values.len() {
        if (!values[best].is_nan() && values[index].is_nan())
            || (!values[best].is_nan()
                && !values[index].is_nan()
                && if minimum {
                    values[index] < values[best]
                } else {
                    values[index] > values[best]
                })
        {
            best = index;
        }
    }
    (best + 1) as f64
}

#[cfg(any(test, feature = "polars"))]
/// `(slope, r², residual-at-last-row)` against row positions.
pub(crate) fn regression(values: &[f64]) -> (f64, f64, f64) {
    let points: Vec<(f64, f64)> = values
        .iter()
        .enumerate()
        .filter_map(|(index, &value)| (!value.is_nan()).then_some(((index + 1) as f64, value)))
        .collect();
    if points.len() < 2 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let n = points.len() as f64;
    let sx: f64 = points.iter().map(|p| p.0).sum();
    let sy: f64 = points.iter().map(|p| p.1).sum();
    let sxx: f64 = points.iter().map(|p| p.0 * p.0).sum();
    let syy: f64 = points.iter().map(|p| p.1 * p.1).sum();
    let sxy: f64 = points.iter().map(|p| p.0 * p.1).sum();
    let xvar = n * sxx - sx * sx;
    let yvar = n * syy - sy * sy;
    let covariance = n * sxy - sx * sy;
    let slope = covariance / xvar;
    let intercept = sy / n - slope * sx / n;
    let sample_std = (yvar / (n * (n - 1.0))).sqrt();
    let rsquare = if xvar <= 0.0 || yvar <= 0.0 || sample_std <= 2e-5 {
        f64::NAN
    } else {
        covariance * covariance / (xvar * yvar)
    };
    let residual =
        values.last().copied().unwrap_or(f64::NAN) - (slope * values.len() as f64 + intercept);
    (slope, rsquare, residual)
}

#[cfg(any(test, feature = "polars"))]
pub(crate) fn corr(left: &[f64], right: &[f64]) -> f64 {
    let standalone_std = |values: &[f64]| {
        let (count, sum) = values
            .iter()
            .filter(|value| !value.is_nan())
            .fold((0, 0.0), |(count, sum), value| (count + 1, sum + value));
        if count < 2 {
            return f64::NAN;
        }
        let mean = sum / count as f64;
        (values
            .iter()
            .filter(|value| !value.is_nan())
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (count - 1) as f64)
            .sqrt()
    };
    if standalone_std(left) <= 2e-5 || standalone_std(right) <= 2e-5 {
        return f64::NAN;
    }
    let (count, sum_a, sum_b) = left
        .iter()
        .zip(right)
        .filter(|(a, b)| !a.is_nan() && !b.is_nan())
        .fold((0, 0.0, 0.0), |(count, sum_a, sum_b), (&a, &b)| {
            (count + 1, sum_a + a, sum_b + b)
        });
    if count < 2 {
        return f64::NAN;
    }
    let ma = sum_a / count as f64;
    let mb = sum_b / count as f64;
    let (va, vb, covariance) = left
        .iter()
        .zip(right)
        .filter(|(a, b)| !a.is_nan() && !b.is_nan())
        .fold((0.0, 0.0, 0.0), |(va, vb, covariance), (&a, &b)| {
            let da = a - ma;
            let db = b - mb;
            (va + da * da, vb + db * db, covariance + da * db)
        });
    if va == 0.0 || vb == 0.0 {
        return f64::NAN;
    }
    covariance / (va * vb).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::window::slide;
    use alphax_generic::rolling::rolling_apply_1d;

    #[test]
    fn qlib_rank_is_the_last_values_percentile() {
        assert!((rank_last(&[3.0, 1.0, 2.0]) - 2.0 / 3.0).abs() < 1e-12);
        assert!((rank_last(&[1.0, 2.0, 2.0]) - 2.5 / 3.0).abs() < 1e-12);
    }

    /// Series that stress the running sums: gaps, all-`NaN` stretches,
    /// constant runs, a large level with small moves, and plain noise.
    fn stress_series() -> Vec<Vec<f64>> {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1_u64 << 53) as f64
        };
        let mut out = Vec::new();
        for case in 0..6 {
            let series = (0..300)
                .map(|row| {
                    let u = next();
                    match case {
                        0 => u,
                        1 if u < 0.2 => f64::NAN,
                        1 => 100.0 + u,
                        2 if (40..80).contains(&row) => f64::NAN,
                        2 if (120..170).contains(&row) => 7.25,
                        2 => 1e6 * u,
                        3 if u < 0.6 => f64::NAN,
                        3 => (u * 4.0).floor(),
                        4 => 1e4 + row as f64 * 0.5 + u * 1e-3,
                        _ => (row / 25) as f64,
                    }
                })
                .collect();
            out.push(series);
        }
        out
    }

    /// [`regression`] with two-pass centered sums instead of raw ones. The
    /// raw-sum form carries a cancellation error of its own (a
    /// two-point fit of prices near 100 gives r² = 0.999999998), so the
    /// incremental kernels are checked against the exact form here and against
    /// Qlib itself in the `parity_qlib` test.
    fn centered_regression(values: &[f64]) -> (f64, f64, f64) {
        let points: Vec<(f64, f64)> = values
            .iter()
            .enumerate()
            .filter_map(|(i, &y)| (!y.is_nan()).then_some(((i + 1) as f64, y)))
            .collect();
        if points.len() < 2 {
            return (f64::NAN, f64::NAN, f64::NAN);
        }
        let n = points.len() as f64;
        let mx = points.iter().map(|p| p.0).sum::<f64>() / n;
        let my = points.iter().map(|p| p.1).sum::<f64>() / n;
        let sxx: f64 = points.iter().map(|p| (p.0 - mx).powi(2)).sum();
        let syy: f64 = points.iter().map(|p| (p.1 - my).powi(2)).sum();
        let sxy: f64 = points.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
        let slope = sxy / sxx;
        let sample_std = (syy / (n - 1.0)).sqrt();
        let rsquare = if sxx <= 0.0 || syy <= 0.0 || sample_std <= 2e-5 {
            f64::NAN
        } else {
            sxy * sxy / (sxx * syy)
        };
        let residual = values.last().copied().unwrap_or(f64::NAN)
            - (slope * values.len() as f64 + my - slope * mx);
        (slope, rsquare, residual)
    }

    type Reducer = fn(&[f64]) -> f64;

    fn rolling_single(name: &str, input: &[f64], window: usize, q: f64) -> Result<Vec<f64>> {
        let kind = match name {
            "Sum" => Rolling::Sum,
            "Std" => Rolling::Std,
            "Var" => Rolling::Var,
            "WMA" => Rolling::Wma,
            "Slope" => Rolling::Slope,
            "Rsquare" => Rolling::Rsquare,
            "Resi" => Rolling::Resi,
            "Med" => Rolling::Med,
            "Mad" => Rolling::Mad,
            "Skew" => Rolling::Skew,
            "Kurt" => Rolling::Kurt,
            "Rank" => Rolling::Rank,
            "IdxMax" => Rolling::IdxMax,
            "IdxMin" => Rolling::IdxMin,
            "Quantile" => Rolling::Quantile(q.to_bits()),
            "Max" => Rolling::Max,
            "Min" => Rolling::Min,
            other => panic!("no rolling operator {other}"),
        };
        Ok(rolling(kind, input, window))
    }

    fn assert_close(label: &str, fast: &[f64], naive: &[f64]) {
        for (row, (&a, &b)) in fast.iter().zip(naive).enumerate() {
            let ok = (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-9 * b.abs().max(1.0);
            assert!(ok, "{label}[{row}]: incremental={a:?}, rescan={b:?}");
        }
    }

    #[test]
    fn incremental_windows_match_a_full_rescan() {
        let reference: [(&str, Reducer); 14] = [
            ("Sum", |v| v.iter().filter(|x| !x.is_nan()).sum()),
            ("Std", |v| {
                rolling_std_qlib(v, v.len(), 1)
                    .unwrap()
                    .last()
                    .copied()
                    .unwrap()
            }),
            ("Var", |v| {
                rolling_std_qlib(v, v.len(), 1)
                    .unwrap()
                    .last()
                    .map(|x| x * x)
                    .unwrap()
            }),
            ("WMA", wma),
            ("Slope", |v| centered_regression(v).0),
            ("Rsquare", |v| centered_regression(v).1),
            ("Resi", |v| centered_regression(v).2),
            ("Med", |v| quantile(v, 0.5)),
            ("Mad", mad),
            ("Skew", skew),
            ("Kurt", kurt),
            ("Rank", rank_last),
            ("IdxMax", |v| arg_extreme(v, false)),
            ("IdxMin", |v| arg_extreme(v, true)),
        ];
        for (case, series) in stress_series().iter().enumerate() {
            for window in [1, 2, 3, 5, 20, 61, 299, 300, 1000] {
                for (name, reduce) in reference {
                    let fast = rolling_single(name, series, window, 0.0).unwrap();
                    let naive =
                        rolling_apply_1d(series, window, Some(1), |_, v| reduce(v)).unwrap();
                    assert_close(
                        &format!("{name} case {case} window {window}"),
                        &fast,
                        &naive,
                    );
                }
                for q in [0.0, 0.2, 0.5, 0.8, 1.0] {
                    let fast = rolling_single("Quantile", series, window, q).unwrap();
                    let naive =
                        rolling_apply_1d(series, window, Some(1), |_, v| quantile(v, q)).unwrap();
                    assert_close(
                        &format!("Quantile {q} case {case} window {window}"),
                        &fast,
                        &naive,
                    );
                }
                for (name, fast, reducer) in [
                    (
                        "Max",
                        rolling_single("Max", series, window, 0.0).unwrap(),
                        (|v: &[f64]| {
                            v.iter()
                                .copied()
                                .filter(|x| !x.is_nan())
                                .fold(f64::NAN, f64::max)
                        }) as Reducer,
                    ),
                    (
                        "Min",
                        rolling_single("Min", series, window, 0.0).unwrap(),
                        |v: &[f64]| {
                            v.iter()
                                .copied()
                                .filter(|x| !x.is_nan())
                                .fold(f64::NAN, f64::min)
                        },
                    ),
                ] {
                    let naive =
                        rolling_apply_1d(series, window, Some(1), |_, v| reducer(v)).unwrap();
                    assert_close(
                        &format!("{name} case {case} window {window}"),
                        &fast,
                        &naive,
                    );
                }
            }
        }
    }

    #[test]
    fn incremental_pairs_match_a_full_rescan() {
        let series = stress_series();
        for (i, left) in series.iter().enumerate() {
            for (j, right) in series.iter().enumerate() {
                let pairs: Vec<(f64, f64)> =
                    left.iter().copied().zip(right.iter().copied()).collect();
                let left = Matrix::from_column_major(left.clone(), left.len(), 1).unwrap();
                let right = Matrix::from_column_major(right.clone(), right.len(), 1).unwrap();
                for window in [1, 2, 5, 20, 61, 300] {
                    let label = format!("case {i}x{j} window {window}");
                    let fast = slide(&pairs, window, &mut Pair::corr());
                    let naive = naive_rolling_pair(&left, &right, window, corr).unwrap();
                    assert_close(&format!("Corr {label}"), &fast, naive.as_slice());
                    let fast = slide(&pairs, window, &mut Pair::cov());
                    let naive = naive_rolling_pair(&left, &right, window, cov).unwrap();
                    assert_close(&format!("Cov {label}"), &fast, naive.as_slice());
                }
            }
        }
    }

    #[test]
    fn regression_finds_a_line_and_its_last_residual() {
        let (slope, r2, residual) = regression(&[2.0, 4.0, 6.0]);
        assert!((slope - 2.0).abs() < 1e-12);
        assert!((r2 - 1.0).abs() < 1e-12);
        assert!(residual.abs() < 1e-12);
    }

    fn rolling_std_qlib(input: &[f64], window: usize, ddof: usize) -> Result<Vec<f64>> {
        rolling_apply_1d(input, window, Some(1), |_, values| {
            let count = values.iter().filter(|value| !value.is_nan()).count();
            if count <= ddof {
                return f64::NAN;
            }
            let mean = values.iter().filter(|value| !value.is_nan()).sum::<f64>() / count as f64;
            let m2 = values
                .iter()
                .filter(|value| !value.is_nan())
                .map(|value| (value - mean).powi(2))
                .sum::<f64>();
            (m2 / (count - ddof) as f64).sqrt()
        })
    }

    fn naive_rolling_pair(
        left: &Matrix,
        right: &Matrix,
        window: usize,
        reducer: impl Fn(&[f64], &[f64]) -> f64,
    ) -> Result<Matrix> {
        if (left.n_rows(), left.n_cols()) != (right.n_rows(), right.n_cols()) {
            return Err(Error::Broadcast {
                left: (left.n_rows(), left.n_cols()),
                right: (right.n_rows(), right.n_cols()),
            });
        }
        let mut out = Matrix::filled(left.n_rows(), left.n_cols(), f64::NAN);
        for col in 0..left.n_cols() {
            for row in 0..left.n_rows() {
                let start = (row + 1).saturating_sub(window);
                out.column_mut(col)[row] = reducer(
                    &left.column(col)[start..=row],
                    &right.column(col)[start..=row],
                );
            }
        }
        Ok(out)
    }

    fn cov(left: &[f64], right: &[f64]) -> f64 {
        let pairs: Vec<(f64, f64)> = left
            .iter()
            .zip(right)
            .filter_map(|(&a, &b)| (!a.is_nan() && !b.is_nan()).then_some((a, b)))
            .collect();
        if pairs.len() < 2 {
            return f64::NAN;
        }
        let n = pairs.len() as f64;
        let ma = pairs.iter().map(|p| p.0).sum::<f64>() / n;
        let mb = pairs.iter().map(|p| p.1).sum::<f64>() / n;
        pairs.iter().map(|p| (p.0 - ma) * (p.1 - mb)).sum::<f64>() / (n - 1.0)
    }
}
