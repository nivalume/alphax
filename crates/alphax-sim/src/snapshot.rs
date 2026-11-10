//! The state a run ends in, to continue it later.
//!
//! A run's accounts carry four things from one bar to the next: each group's
//! cash and free cash, and each column's [`ColumnState`] (position, debt,
//! locked cash, valuation price, stops and resting limit order). A run hands
//! them back as a [`SimState`]; a run given one through `resume` starts from
//! it on its `next_row` instead of from `init_cash` and `init_position`.
//! Split anywhere, two runs chained this way write the records the one run
//! would have, bit for bit, in the same order once
//! [`OrderRecords::chain`](alphax_records::OrderRecords::chain)ed.
//!
//! # Rows
//!
//! The state's rows are the run's: a time stop's deadline and a limit
//! order's last row are absolute. A run's arrays may start later than row 0
//! (`row_offset`), so a long history can run in pieces without holding it all;
//! the loops then read their arrays from 0 and convert where the state and the
//! records meet them. A piece has to carry the rows before `next_row` its
//! bars look back at: the previous close for a `-inf` price, and `from_ago`
//! rows for an order decided earlier. Without them the first bars of the
//! piece read `NaN` where the one run read a price.
//!
//! # What is not in it
//!
//! A callback's own state: a callback run resumed calls `pre_group` again,
//! and a callback that carries state rebuilds it there. Nor are the records:
//! the state is what the next bar needs, the records what the past did.
//!
//! # Bytes
//!
//! [`SimState::to_bytes`] writes a little-endian binary with a magic number
//! and a version; [`SimState::from_bytes`] refuses any other. Library crates
//! do not write files, so where the bytes go is the caller's.

use alphax_core::error::{Error, Result};
use alphax_exec::{Direction, SizeType};
use alphax_model::{DeltaFormat, LimitOrderPrice};

use crate::state::{ColumnState, LimitOrder, StopState};

/// One account group's state between two bars.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupState {
    /// The group's cash.
    pub cash: f64,
    /// Its cash not tied up by positions.
    pub free_cash: f64,
    /// Each column's state, in the group's column order; rows absolute.
    pub columns: Vec<ColumnState>,
    /// Whether a callback stopped the group's trading.
    pub stopped: bool,
}

/// Which loop a state is from: each continues only its own, because
/// `from_orders` keeps a valuation price times the contract multiplier and
/// `from_signals` keeps the price, and converting one into the other is not
/// exact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Loop {
    /// `from_orders` and `from_order_func`.
    FromOrders = 0,
    /// `from_signals`.
    FromSignals = 1,
}

/// A run's state after its last bar: what a run continuing it starts from.
#[derive(Debug, Clone, PartialEq)]
pub struct SimState {
    /// The loop that wrote it.
    pub source: Loop,
    /// The first row the continuing run simulates.
    pub next_row: usize,
    /// Whether the run kept balances with its fills; a continuing run keeps
    /// them too, so the records stay one shape.
    pub levered: bool,
    /// Each account group's state, in group order.
    pub groups: Vec<GroupState>,
}

const MAGIC: &[u8; 4] = b"PHSS";
const VERSION: u32 = 1;

impl SimState {
    /// The state of group `group` for a run whose arrays start at
    /// `row_offset`, with rows read from there, after checking it fits a
    /// group of `len` columns.
    pub(crate) fn group_for(
        &self,
        group: usize,
        len: usize,
        row_offset: usize,
    ) -> Result<GroupState> {
        let Some(state) = self.groups.get(group) else {
            return Err(Error::Invariant(format!(
                "the state has {} groups; the run has group {group}",
                self.groups.len()
            )));
        };
        if state.columns.len() != len {
            return Err(Error::Invariant(format!(
                "group {group} has {} columns in the state and {len} in the run",
                state.columns.len()
            )));
        }
        let mut state = state.clone();
        for column in &mut state.columns {
            rebase(column, |row| row - row_offset.min(row));
        }
        Ok(state)
    }

    /// Checks the state can continue a run of `source` with `n_groups`
    /// groups whose arrays cover rows `row_offset..row_offset + n_rows`, and
    /// returns the first of them to simulate, counted from `row_offset`.
    pub(crate) fn start_in(
        &self,
        source: Loop,
        n_groups: usize,
        row_offset: usize,
        n_rows: usize,
    ) -> Result<usize> {
        if self.source != source {
            return Err(Error::Invariant(format!(
                "a state from {:?} cannot continue a {source:?} run",
                self.source
            )));
        }
        if self.groups.len() != n_groups {
            return Err(Error::Invariant(format!(
                "the state has {} groups; the run has {n_groups}",
                self.groups.len()
            )));
        }
        if self.next_row < row_offset || self.next_row > row_offset + n_rows {
            return Err(Error::Invariant(format!(
                "the state continues at row {}, outside the rows {row_offset}..{} the arrays \
                 cover",
                self.next_row,
                row_offset + n_rows
            )));
        }
        Ok(self.next_row - row_offset)
    }

    /// The state as bytes; see the [module](self).
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(32 + self.groups.len() * 32);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.push(self.source as u8);
        put_usize(&mut out, self.next_row);
        out.push(u8::from(self.levered));
        put_usize(&mut out, self.groups.len());
        for group in &self.groups {
            put_f64(&mut out, group.cash);
            put_f64(&mut out, group.free_cash);
            out.push(u8::from(group.stopped));
            put_usize(&mut out, group.columns.len());
            for column in &group.columns {
                put_column(&mut out, column);
            }
        }
        out
    }

    /// A state from [`to_bytes`](Self::to_bytes)' output.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invariant`] for bytes that are not a state of this
    /// version, are cut short, or run on past one.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut r = Reader { bytes, at: 0 };
        if r.take(4)? != MAGIC {
            return Err(Error::Invariant("not a alphax simulation state".into()));
        }
        let version = u32::from_le_bytes(r.array()?);
        if version != VERSION {
            return Err(Error::Invariant(format!(
                "simulation state version {version}; this build reads version {VERSION}"
            )));
        }
        let source = match r.u8()? {
            0 => Loop::FromOrders,
            1 => Loop::FromSignals,
            b => return Err(Error::Invariant(format!("{b} is not a simulation loop"))),
        };
        let next_row = r.usize()?;
        let levered = r.bool()?;
        let n_groups = r.usize()?;
        let mut groups = Vec::with_capacity(n_groups.min(1 << 20));
        for _ in 0..n_groups {
            let cash = r.f64()?;
            let free_cash = r.f64()?;
            let stopped = r.bool()?;
            let n_cols = r.usize()?;
            let mut columns = Vec::with_capacity(n_cols.min(1 << 20));
            for _ in 0..n_cols {
                columns.push(r.column()?);
            }
            groups.push(GroupState {
                cash,
                free_cash,
                columns,
                stopped,
            });
        }
        if r.at != bytes.len() {
            return Err(Error::Invariant(format!(
                "{} bytes after the simulation state",
                bytes.len() - r.at
            )));
        }
        Ok(Self {
            source,
            next_row,
            levered,
            groups,
        })
    }
}

/// What a run returns: its records, each group's callback state (`()`
/// without callbacks) in group order, and the state to continue it from.
#[derive(Debug, Clone)]
pub struct Run<S, T = ()> {
    /// The fills and cash flows, with logs when `S` keeps them.
    pub records: S,
    /// Each group's callback state, in group order.
    pub states: Vec<T>,
    /// The accounts after the last bar.
    pub last_state: SimState,
}

/// A group's state at the end of a run whose arrays start at `row_offset`,
/// with its rows made absolute.
pub(crate) fn finish(
    cash: f64,
    free_cash: f64,
    mut columns: Vec<ColumnState>,
    stopped: bool,
    row_offset: usize,
) -> GroupState {
    if row_offset != 0 {
        for column in &mut columns {
            rebase(column, |row| row + row_offset);
        }
    }
    GroupState {
        cash,
        free_cash,
        columns,
        stopped,
    }
}

/// Moves a column's rows with `f`, leaving the "never" rows as they are.
fn rebase(column: &mut ColumnState, f: impl Fn(usize) -> usize) {
    let moved = |row: &mut usize| {
        if *row != usize::MAX {
            *row = f(*row);
        }
    };
    moved(&mut column.stops.td_row);
    moved(&mut column.stops.dt_row);
    // An empty slot's rows are placeholders, kept as they are. A resting
    // order's `created` is read only when it is placed, to set `last_row`;
    // it is history here, and a row before the arrays becomes 0.
    if column.limit.is_pending() {
        moved(&mut column.limit.created);
        moved(&mut column.limit.last_row);
    }
}

fn put_f64(out: &mut Vec<u8>, v: f64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_usize(out: &mut Vec<u8>, v: usize) {
    out.extend_from_slice(&(v as u64).to_le_bytes());
}

fn put_column(out: &mut Vec<u8>, c: &ColumnState) {
    for v in [c.position, c.debt, c.locked_cash, c.last_val_price] {
        put_f64(out, v);
    }
    let s = &c.stops;
    put_f64(out, s.sl_price);
    put_f64(out, s.sl_stop);
    out.push(u8::from(s.sl_trail));
    for v in [
        s.tp_price,
        s.tp_stop,
        s.tsl_price,
        s.tsl_stop,
        s.tsl_th,
        s.tsl_anchor,
    ] {
        put_f64(out, v);
    }
    out.push(u8::from(s.tsl_active));
    for v in [s.td_row, s.dt_row, s.sl_step, s.tp_step, s.tsl_step] {
        put_usize(out, v);
    }
    out.push(s.delta_format as u8);
    put_f64(out, s.ladder_base);
    out.push(u8::from(s.extended));
    let l = &c.limit;
    put_f64(out, l.price);
    out.push(u8::from(l.buy));
    out.push(u8::from(l.reverse));
    out.push(l.order_price as u8);
    put_f64(out, l.size);
    out.push(l.size_type as u8);
    out.push(l.direction as u8);
    put_usize(out, l.created);
    put_usize(out, l.last_row);
}

struct Reader<'b> {
    bytes: &'b [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        let end = self.at + n;
        if end > self.bytes.len() {
            return Err(Error::Invariant("the simulation state is cut short".into()));
        }
        let out = &self.bytes[self.at..end];
        self.at = end;
        Ok(out)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut out = [0; N];
        out.copy_from_slice(self.take(N)?);
        Ok(out)
    }

    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.array()?))
    }

    fn usize(&mut self) -> Result<usize> {
        let v = u64::from_le_bytes(self.array()?);
        usize::try_from(v).map_err(|_| Error::Invariant(format!("{v} does not fit a usize")))
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn bool(&mut self) -> Result<bool> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            b => Err(Error::Invariant(format!("{b} is not a flag"))),
        }
    }

    fn column(&mut self) -> Result<ColumnState> {
        let (position, debt, locked_cash, last_val_price) =
            (self.f64()?, self.f64()?, self.f64()?, self.f64()?);
        let sl_price = self.f64()?;
        let sl_stop = self.f64()?;
        let sl_trail = self.bool()?;
        let (tp_price, tp_stop) = (self.f64()?, self.f64()?);
        let (tsl_price, tsl_stop, tsl_th, tsl_anchor) =
            (self.f64()?, self.f64()?, self.f64()?, self.f64()?);
        let tsl_active = self.bool()?;
        let (td_row, dt_row) = (self.usize()?, self.usize()?);
        let (sl_step, tp_step, tsl_step) = (self.usize()?, self.usize()?, self.usize()?);
        let delta_format = match self.u8()? {
            0 => DeltaFormat::Percent,
            1 => DeltaFormat::Absolute,
            2 => DeltaFormat::Target,
            3 => DeltaFormat::Percent100,
            b => return Err(Error::Invariant(format!("{b} is not a delta format"))),
        };
        let ladder_base = self.f64()?;
        let extended = self.bool()?;
        let price = self.f64()?;
        let buy = self.bool()?;
        let reverse = self.bool()?;
        let order_price = match self.u8()? {
            0 => LimitOrderPrice::Limit,
            1 => LimitOrderPrice::HardLimit,
            2 => LimitOrderPrice::Close,
            b => return Err(Error::Invariant(format!("{b} is not a limit order price"))),
        };
        let size = self.f64()?;
        let size_type = match self.u8()? {
            0 => SizeType::Amount,
            1 => SizeType::Value,
            2 => SizeType::Percent,
            3 => SizeType::TargetAmount,
            4 => SizeType::TargetValue,
            5 => SizeType::TargetPercent,
            6 => SizeType::ValuePercent,
            7 => SizeType::Percent100,
            8 => SizeType::ValuePercent100,
            9 => SizeType::TargetPercent100,
            b => return Err(Error::Invariant(format!("{b} is not a size type"))),
        };
        let direction = match self.u8()? {
            0 => Direction::LongOnly,
            1 => Direction::ShortOnly,
            2 => Direction::Both,
            b => return Err(Error::Invariant(format!("{b} is not a direction"))),
        };
        let (created, last_row) = (self.usize()?, self.usize()?);
        Ok(ColumnState {
            position,
            debt,
            locked_cash,
            last_val_price,
            stops: StopState {
                sl_price,
                sl_stop,
                sl_trail,
                tp_price,
                tp_stop,
                tsl_price,
                tsl_stop,
                tsl_th,
                tsl_anchor,
                tsl_active,
                td_row,
                dt_row,
                sl_step,
                tp_step,
                tsl_step,
                delta_format,
                ladder_base,
                extended,
            },
            limit: LimitOrder {
                price,
                buy,
                reverse,
                order_price,
                size,
                size_type,
                direction,
                created,
                last_row,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> SimState {
        let mut a = ColumnState::new();
        a.position = 2.5;
        a.debt = -1.0;
        a.last_val_price = 101.25;
        a.stops.sl_stop = 0.05;
        a.stops.td_row = 40;
        a.stops.delta_format = DeltaFormat::Percent100;
        a.limit.price = 99.0;
        a.limit.size = 1.0;
        a.limit.size_type = SizeType::TargetPercent;
        a.limit.direction = Direction::ShortOnly;
        a.limit.order_price = LimitOrderPrice::HardLimit;
        a.limit.created = 30;
        a.limit.last_row = usize::MAX;
        SimState {
            source: Loop::FromSignals,
            next_row: 35,
            levered: true,
            groups: vec![
                GroupState {
                    cash: 1234.5,
                    free_cash: f64::NAN,
                    columns: vec![a, ColumnState::new()],
                    stopped: false,
                },
                GroupState {
                    cash: -3.0,
                    free_cash: 0.0,
                    columns: vec![ColumnState::new()],
                    stopped: true,
                },
            ],
        }
    }

    #[test]
    fn bytes_round_trip_bit_for_bit() {
        let s = state();
        let back = SimState::from_bytes(&s.to_bytes()).unwrap();
        // NaN is not equal to itself, so compare the bytes again.
        assert_eq!(back.to_bytes(), s.to_bytes());
        assert_eq!(back.groups[0].columns[0].stops.td_row, 40);
        assert!(back.groups[0].free_cash.is_nan());
    }

    #[test]
    fn bad_bytes_are_refused() {
        let bytes = state().to_bytes();
        assert!(SimState::from_bytes(&bytes[..bytes.len() - 1]).is_err());
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(SimState::from_bytes(&longer).is_err());
        let mut other = bytes;
        other[4] = 9;
        other[5] = 0;
        let err = SimState::from_bytes(&other).unwrap_err().to_string();
        assert!(err.contains("version 9"), "{err}");
        assert!(SimState::from_bytes(b"nope").is_err());
    }

    #[test]
    fn rows_are_read_from_the_arrays_first_row() {
        let s = state();
        let g = s.group_for(0, 2, 30).unwrap();
        assert_eq!(g.columns[0].stops.td_row, 10);
        assert_eq!(g.columns[0].limit.created, 0);
        assert_eq!(g.columns[0].limit.last_row, usize::MAX);
        assert_eq!(g.columns[1].stops.dt_row, usize::MAX);
        let back = finish(g.cash, g.free_cash, g.columns, false, 30);
        assert_eq!(back.columns[0].stops.td_row, 40);
        assert!(s.group_for(0, 3, 0).is_err());
        let signals = Loop::FromSignals;
        assert_eq!(s.start_in(signals, 2, 30, 10).unwrap(), 5);
        assert!(s.start_in(signals, 2, 36, 10).is_err());
        assert!(s.start_in(signals, 2, 0, 34).is_err());
        assert!(s.start_in(signals, 3, 0, 40).is_err());
        assert!(s.start_in(Loop::FromOrders, 2, 0, 40).is_err());
    }
}
