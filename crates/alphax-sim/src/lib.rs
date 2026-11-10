//! Vectorized simulation loops, and where parallelism happens.
//!
//! # Account groups are independent
//!
//! Order and signal loops share capital inside each group. Groups own their
//! state and execute in parallel; symbols within a group execute sequentially.
//! With no sharing, each column is its own account.
//!
//! Keeping every engine instance `Send`, with no global state, is a hard
//! constraint on this codebase: it is what lets a sweep run as threads in one
//! process rather than as many processes.
//!
//! [`from_orders::simulate_from_orders_parallel`] returns records equal to the
//! serial version for the same input, whatever the thread count. That is not
//! incidental: each column fills its own buffer and record ids are assigned
//! after merging in column order, so nothing depends on how threads
//! interleave.
//!
//! # Why this loop is not SIMD
//!
//! Along the time axis there is a hard loop-carried dependency:
//! `state[i] = f(state[i-1], bar[i])`. Across columns, SIMD is possible in
//! principle and a bad trade in practice — the execution kernel is a deep
//! branch tree with fourteen data-dependent early exits, and lanes would
//! diverge on nearly every bar. SIMD belongs in the factor and analysis
//! layers, where the operations have no such dependency. The two techniques
//! stack; they are not alternatives.
//!
//! # Contents
//!
//! `from_orders`, `from_signals` with its seven-step
//! signal pipeline and stop engine, cash sharing with the automatic call
//! order, and a logged variant of each loop.
//!
//! Cash sharing arrived after the rest, and moved the unit of parallel work
//! from the column to the group. With the default grouping the two are the
//! same thing, so nothing changed for independent symbols; with one shared
//! portfolio there is a single unit of work, which is inherent — a shared cash
//! balance is a sequential dependency across the whole group.

use alphax_model::{AccountType, MarginMode};

pub mod builtin;
pub mod callbacks;
pub mod cash;
pub mod from_orders;
pub mod from_signals;
pub mod groups;
mod holding;
mod initial;
pub mod limit;
mod liquidation;
mod margin;
mod prices;
mod random;
pub mod signals;
pub mod snapshot;
pub mod state;
pub mod stop_exits;
mod stops;

/// How many cells of `array` over `cols` satisfy `pred`, without expanding it.
///
/// The loops preallocate their record buffers to a bound on the fills, and a
/// bound of one fill per cell reserves `n_rows × n_cols` records — eight
/// gigabytes of address space for 100k bars × 2,000 columns — when the
/// inputs say most cells cannot fill at all. Counting the cells that can is
/// one pass over the inputs, which the simulation reads anyway.
pub(crate) fn count_where<T: Copy>(
    array: &alphax_core::flex::FlexArray<'_, T>,
    n_rows: usize,
    cols: std::ops::Range<usize>,
    pred: impl Fn(T) -> bool,
) -> usize {
    use alphax_core::flex::FlexArray;
    let count = |cells: &[T]| cells.iter().filter(|&&v| pred(v)).count();
    match *array {
        FlexArray::Scalar(v) => {
            if pred(v) {
                n_rows * cols.len()
            } else {
                0
            }
        }
        FlexArray::PerColumn(v) => n_rows * count(&v[cols]),
        FlexArray::PerRow(v) => count(v) * cols.len(),
        FlexArray::Full { data, n_rows } => count(&data[cols.start * n_rows..cols.end * n_rows]),
        FlexArray::MappedColumns {
            data,
            n_rows,
            columns,
        } => {
            // Source columns repeat across a sweep, so count each once.
            let mut counts = vec![usize::MAX; data.len() / n_rows.max(1)];
            cols.map(|col| {
                let source = columns[col];
                if counts[source] == usize::MAX {
                    counts[source] = count(&data[source * n_rows..(source + 1) * n_rows]);
                }
                counts[source]
            })
            .sum()
        }
    }
}

/// Whether any column of a run may borrow or trade on margin: whether
/// `leverage` exceeds 1 anywhere in it, or any column is a margin column —
/// or has a contract multiplier other than 1, which the loops apply in the
/// same copy.
///
/// Decided once per run, because it decides whether the records keep
/// balances, and they keep them for every fill or for none. The defaults, a
/// scalar 1, a scalar cash account and a scalar 1, are answered without a
/// scan.
pub(crate) fn any_levered(
    leverage: &alphax_core::flex::FlexArray<'_, f64>,
    account_type: &alphax_core::flex::FlexArray<'_, AccountType>,
    multiplier: &alphax_core::flex::FlexArray<'_, f64>,
    n_rows: usize,
    n_cols: usize,
) -> bool {
    count_where(leverage, n_rows, 0..n_cols, |l| l > 1.0) > 0
        || count_where(account_type, n_rows, 0..n_cols, |a| {
            a == AccountType::Margin
        }) > 0
        // Bits rather than `!=`: a `NaN` is not 1 either, and is refused
        // by `multipliers` rather than read as one.
        || count_where(multiplier, n_rows, 0..n_cols, |m| {
            m.to_bits() != 1.0_f64.to_bits()
        }) > 0
}

/// The contract multiplier of each of the columns `cols`: what one unit of
/// position is worth per unit of price.
///
/// The loops run a multiplied column at its prices times its multiplier —
/// every value, cash flow, margin and charge follows from them — and write
/// the prices back unscaled into the records and logs. Sizes stay in
/// contracts, so `min_size`, `size_granularity` and a per-unit margin are per
/// contract. A property of the column for the whole run: one that changes
/// down a column is an error, because the records keep one per fill and the
/// analysis values a position at it.
pub(crate) fn multipliers(
    multiplier: &alphax_core::flex::FlexArray<'_, f64>,
    n_rows: usize,
    cols: std::ops::Range<usize>,
) -> alphax_core::error::Result<Vec<f64>> {
    cols.map(|col| {
        let first = multiplier.get(0, col);
        if !first.is_finite() || first <= 0.0 {
            return Err(alphax_core::error::Error::Invariant(format!(
                "multiplier must be finite and greater than 0; got {first} for column {col}"
            )));
        }
        if multiplier.varies_by_row()
            && (1..n_rows).any(|i| multiplier.get(i, col).to_bits() != first.to_bits())
        {
            return Err(alphax_core::error::Error::Invariant(format!(
                "multiplier changes down column {col}; a column has one contract multiplier \
                 for the whole run"
            )));
        }
        Ok(first)
    })
    .collect()
}

/// Writes a log entry's prices back as the strategy wrote them, from the
/// multiplied prices the loop ran a column at.
///
/// Dividing by the multiplier undoes multiplying by it to within a rounding
/// error, and exactly at 1.
pub(crate) fn unscale_log(entry: &mut alphax_records::LogEntry, multiplier: f64) {
    entry.val_price /= multiplier;
    entry.new_val_price /= multiplier;
    entry.request.price /= multiplier;
    entry.result.price /= multiplier;
    entry.multiplier = multiplier;
}

/// Which of the columns `cols` are margin columns.
///
/// An account type decides how a column's balances read, so it is a property
/// of the column for the whole run: one that changes down a column is an
/// error rather than being read at its first row.
pub(crate) fn margin_columns(
    account_type: &alphax_core::flex::FlexArray<'_, AccountType>,
    n_rows: usize,
    cols: std::ops::Range<usize>,
) -> alphax_core::error::Result<Vec<bool>> {
    cols.map(|col| {
        let first = account_type.get(0, col);
        if account_type.varies_by_row() && (1..n_rows).any(|i| account_type.get(i, col) != first) {
            return Err(alphax_core::error::Error::Invariant(format!(
                "account_type changes down column {col}; a column is a cash or a margin column \
                 for the whole run"
            )));
        }
        Ok(first == AccountType::Margin)
    })
    .collect()
}

/// Which of the columns `cols` are isolated margin columns, given which are
/// margin columns at all: a cash column's `margin_mode` means nothing.
///
/// A property of the column for the whole run, like its account type.
pub(crate) fn isolated_columns(
    margin_mode: &alphax_core::flex::FlexArray<'_, MarginMode>,
    n_rows: usize,
    cols: std::ops::Range<usize>,
    margin: &[bool],
) -> alphax_core::error::Result<Vec<bool>> {
    cols.enumerate()
        .map(|(k, col)| {
            let first = margin_mode.get(0, col);
            if margin_mode.varies_by_row() && (1..n_rows).any(|i| margin_mode.get(i, col) != first)
            {
                return Err(alphax_core::error::Error::Invariant(format!(
                    "margin_mode changes down column {col}; a margin column is cross or \
                     isolated for the whole run"
                )));
            }
            Ok(margin[k] && first == MarginMode::Isolated)
        })
        .collect()
}

/// The smallest fill accepted by default.
///
/// Not zero, which is what one would write. The default is `1e-8`, and the
/// difference is observable: a `5e-9` order is **rejected** by it and would
/// fill with a zero minimum, so a caller who omits `min_size` gets a dust
/// guard.
///
/// It is a dust guard rather than a tolerance — the float tolerance layer is
/// separate and finer. What it buys is that a target-size cascade landing a
/// hair away from its target does not emit a fill nobody can act on, and that
/// the rejection says `SizeTooSmall` rather than leaving a record of a
/// nanogram trade.
///
/// Visible in the log records: `req_min_size` is one of the 37 fields. No
/// fixture of random sizes exercises it in either loop, because a randomly
/// generated size is never dust.
pub const DEFAULT_MIN_SIZE: f64 = 1e-8;

pub use builtin::{Builtin, BuiltinState};
pub use callbacks::{Context, Decision, Flow, NoCallbacks, OrderFunc, SignalFunc, risk_size};
pub use from_orders::{
    FromOrders, Schedule, simulate_from_order_func, simulate_from_orders,
    simulate_from_orders_logged, simulate_from_orders_logged_parallel,
    simulate_from_orders_parallel, simulate_from_orders_run,
};
pub use from_signals::{
    FromSignals, simulate_from_signal_func, simulate_from_signals, simulate_from_signals_logged,
    simulate_from_signals_logged_parallel, simulate_from_signals_parallel,
    simulate_from_signals_run,
};
pub use snapshot::{GroupState, Loop, Run, SimState};
