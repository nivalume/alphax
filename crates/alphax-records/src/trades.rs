//! Trade derivation: three views over one schema.
//!
//! A fill is not a trade. Turning a stream of fills into trades needs a rule
//! for attributing exits back to entries, and there are three defensible
//! rules — so all three are provided, sharing one record layout and differing
//! only in what one record *means*. Mixing them up produces plausible
//! statistics that answer a different question than the one asked, which is
//! why this is the part of the design most easily confused.
//!
//! | View | One record is | Exit attribution |
//! |---|---|---|
//! | [`entry_trades`] | one entry fill | the position's size-weighted average exit |
//! | [`exit_trades`] | one exit fill | its own price; entries averaged into it |
//! | [`positions`] | one position | aggregated from either of the above |
//!
//! All three agree on total PnL. They disagree on how many trades there
//! were, what each one returned, and therefore on win rate, expectancy and
//! every other per-trade statistic.
//!
//! # Contract
//!
//! Input records must be grouped by column and ascending in row within a
//! column — the ordering [`OrderRecords`] already guarantees. That is what
//! lets this work on contiguous slices instead of an index map.

use std::ops::Range;

use alphax_core::error::{Error, Result};
use alphax_core::flex::FlexArray;
use alphax_core::math::{add, is_close, is_close_or_less, is_less};
use alphax_model::{OrderSide, TradeDirection, TradeStatus};

use crate::OrderRecords;
use crate::mapped::{ColMap, MappedArray};

/// Trades, positions and entry/exit views, struct-of-arrays.
///
/// One schema for all three views: what changes between them is the
/// attribution, not the layout. `id` is not stored — it is the record's
/// index, as for [`OrderRecords`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TradeRecords {
    /// Column the trade belongs to.
    pub col: Vec<i64>,
    /// Size of the trade, always positive.
    pub size: Vec<f64>,
    /// Row the entry happened on.
    pub entry_idx: Vec<i64>,
    /// Entry price, size-weighted where several fills are merged.
    pub entry_price: Vec<f64>,
    /// Entry fees attributable to this trade.
    pub entry_fees: Vec<f64>,
    /// Row the exit happened on; the last bar for an open trade.
    pub exit_idx: Vec<i64>,
    /// Exit price, size-weighted where several fills are merged.
    pub exit_price: Vec<f64>,
    /// Exit fees attributable to this trade.
    pub exit_fees: Vec<f64>,
    /// Profit or loss, net of both sides' fees.
    pub pnl: Vec<f64>,
    /// `pnl` over the entry value.
    pub ret: Vec<f64>,
    /// Long or short, as [`TradeDirection`]'s discriminant.
    pub direction: Vec<i8>,
    /// Open or closed, as [`TradeStatus`]'s discriminant.
    pub status: Vec<i8>,
    /// Which position this trade belongs to.
    pub parent_id: Vec<i64>,
    /// The column's contract multiplier. `pnl` is in cash, so it is
    /// multiplied by it; `ret` is a fraction of the entry value, which is
    /// multiplied too, so it is not.
    pub multiplier: Vec<f64>,
}

/// PnL and return for one trade.
///
/// A short's value difference is negated rather than its size, which is why
/// `size` can stay positive in every record. `add` rather than `-` so that a
/// round trip at the same price gives exactly zero instead of a residue that
/// would show up as a nonzero return.
fn trade_stats(
    size: f64,
    entry_price: f64,
    entry_fees: f64,
    exit_price: f64,
    exit_fees: f64,
    direction: TradeDirection,
    multiplier: f64,
) -> (f64, f64) {
    let entry_val = size * entry_price * multiplier;
    let exit_val = size * exit_price * multiplier;
    let mut val_diff = add(exit_val, -entry_val);
    if val_diff != 0.0 && direction == TradeDirection::Short {
        val_diff = -val_diff;
    }
    let pnl = val_diff - entry_fees - exit_fees;
    // The entry's size: a price below 0 still makes a return of a cost.
    (pnl, pnl / entry_val.abs())
}

impl TradeRecords {
    /// An empty record set.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            col: Vec::new(),
            size: Vec::new(),
            entry_idx: Vec::new(),
            entry_price: Vec::new(),
            entry_fees: Vec::new(),
            exit_idx: Vec::new(),
            exit_price: Vec::new(),
            exit_fees: Vec::new(),
            pnl: Vec::new(),
            ret: Vec::new(),
            direction: Vec::new(),
            status: Vec::new(),
            parent_id: Vec::new(),
            multiplier: Vec::new(),
        }
    }

    /// Number of trades.
    #[must_use]
    pub fn len(&self) -> usize {
        self.col.len()
    }

    /// Whether there are no trades.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.col.is_empty()
    }

    /// The `id` of each record: its index.
    #[must_use]
    pub fn ids(&self) -> Vec<i64> {
        (0..self.len() as i64).collect()
    }

    /// Bars each trade spanned.
    ///
    /// **A closed trade is measured exclusively and an open one inclusively**:
    /// `exit - entry` against `exit - entry + 1`. That is the
    /// rule, and it is not arbitrary — a closed trade's exit bar belongs to
    /// whatever came next, while an open trade has no exit yet and its last
    /// bar is one it was still held through. Using one rule for both shifts
    /// every average holding period by up to a bar, in a direction that
    /// depends on how many trades happened to be open.
    #[must_use]
    pub fn duration(&self) -> Vec<f64> {
        (0..self.len())
            .map(|i| {
                let span = self.exit_idx[i] - self.entry_idx[i];
                let open = self.status[i] == TradeStatus::Open as i8;
                (span + i64::from(open)) as f64
            })
            .collect()
    }

    /// Groups these trades by column.
    ///
    /// Build this once and read several fields through it — the grouping
    /// depends only on `col`, so recomputing it per field is pure waste.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invariant`] if a trade names a column outside
    /// `n_cols`.
    pub fn col_map(&self, n_cols: usize) -> Result<ColMap> {
        ColMap::build(&self.col, n_cols)
    }

    /// Reads one field of these trades through a grouping.
    ///
    /// ```ignore
    /// let map = trades.col_map(n_cols)?;
    /// let mean_pnl = trades.mapped(&trades.pnl, &map)?.mean();
    /// let worst_bar = trades.mapped(&trades.pnl, &map)?.idxmin(&trades.exit_idx, -1)?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invariant`] if `field` is not one value per trade.
    pub fn mapped<'a>(&self, field: &'a [f64], map: &'a ColMap) -> Result<MappedArray<'a>> {
        MappedArray::new(field, map)
    }

    /// Appends one trade, deriving its PnL and return.
    #[allow(clippy::too_many_arguments)]
    fn push(
        &mut self,
        col: i64,
        size: f64,
        entry_idx: i64,
        entry_price: f64,
        entry_fees: f64,
        exit_idx: i64,
        exit_price: f64,
        exit_fees: f64,
        direction: TradeDirection,
        status: TradeStatus,
        parent_id: i64,
        multiplier: f64,
    ) {
        let (pnl, ret) = trade_stats(
            size,
            entry_price,
            entry_fees,
            exit_price,
            exit_fees,
            direction,
            multiplier,
        );
        self.col.push(col);
        self.size.push(size);
        self.entry_idx.push(entry_idx);
        self.entry_price.push(entry_price);
        self.entry_fees.push(entry_fees);
        self.exit_idx.push(exit_idx);
        self.exit_price.push(exit_price);
        self.exit_fees.push(exit_fees);
        self.pnl.push(pnl);
        self.ret.push(ret);
        self.direction.push(direction as i8);
        self.status.push(status as i8);
        self.parent_id.push(parent_id);
        self.multiplier.push(multiplier);
    }

    /// Copies record `i` of `src` verbatim, renumbering its parent.
    ///
    /// Used where a position contains exactly one trade. Not merely a
    /// shortcut: re-deriving the aggregate would compute `size * price / size`
    /// for the price and recompute the PnL from it, which is not bit-identical
    /// to the values already stored. Copying keeps a one-trade position equal
    /// to its trade.
    fn push_copy(&mut self, src: &Self, i: usize, parent_id: i64) {
        self.col.push(src.col[i]);
        self.size.push(src.size[i]);
        self.entry_idx.push(src.entry_idx[i]);
        self.entry_price.push(src.entry_price[i]);
        self.entry_fees.push(src.entry_fees[i]);
        self.exit_idx.push(src.exit_idx[i]);
        self.exit_price.push(src.exit_price[i]);
        self.exit_fees.push(src.exit_fees[i]);
        self.pnl.push(src.pnl[i]);
        self.ret.push(src.ret[i]);
        self.direction.push(src.direction[i]);
        self.status.push(src.status[i]);
        self.parent_id.push(parent_id);
        self.multiplier
            .push(src.multiplier.get(i).copied().unwrap_or(1.0));
    }
}

/// The half-open range of records belonging to each column.
///
/// Relies on the column-major ordering contract; a column that produced no
/// records gets an empty range.
///
/// # Errors
///
/// Returns [`Error::Invariant`] if the records are not grouped by column, or
/// name a column outside `n_cols`.
fn column_ranges(col: &[i64], n_cols: usize) -> Result<Vec<Range<usize>>> {
    // Trade derivation reads each column as one contiguous run — it walks a
    // column's fills in order to build positions — so unlike a reduction it
    // cannot fall back to the gathering path. An unsorted input is a bug in
    // whatever produced the records, and saying so beats deriving trades from
    // a sequence that is not the one they happened in.
    ColMap::build(col, n_cols)?.ranges().ok_or_else(|| {
        Error::Invariant("records must be grouped by column, and these are not".to_owned())
    })
}

/// Whether a fill adds to a position held in `direction`.
const fn is_entry(direction: TradeDirection, side: i8) -> bool {
    matches!(
        (direction, side),
        (TradeDirection::Long, x) if x == OrderSide::Buy as i8
    ) || matches!(
        (direction, side),
        (TradeDirection::Short, x) if x == OrderSide::Sell as i8
    )
}

/// The direction a position opened by this fill runs in.
fn opened_direction(side: i8) -> TradeDirection {
    if side == OrderSide::Buy as i8 {
        TradeDirection::Long
    } else {
        TradeDirection::Short
    }
}

/// Rejects the two things that would make the arithmetic below meaningless.
fn check_order(orders: &OrderRecords, i: usize) -> Result<()> {
    if orders.size[i] <= 0.0 {
        return Err(Error::Invariant(
            "found an order record with size 0 or less".into(),
        ));
    }
    let price = orders.price[i];
    if !price.is_finite() || (price <= 0.0 && !orders.allow_negative_price) {
        return Err(Error::Invariant(
            "found an order record with a price that is not finite, or 0 or less in a run \
             without allow_negative_price"
                .into(),
        ));
    }
    Ok(())
}

/// State accumulated while walking one position, for [`entry_trades`].
struct Position {
    first: usize,
    direction: TradeDirection,
    entry_size_sum: f64,
    first_entry_size: f64,
    first_entry_fees: f64,
    exit_size_sum: f64,
    exit_gross_sum: f64,
    exit_fees_sum: f64,
}

/// Emits one record per entry fill in a closed or closing position.
///
/// Every entry in the position is charged the same size-weighted average exit
/// price, and a share of the exit fees proportional to its size. That is what
/// makes this view "FIFO allocated back to entries": the exit is spread over
/// the entries rather than standing on its own.
#[allow(clippy::too_many_arguments)]
fn flush_entry_trades(
    out: &mut TradeRecords,
    orders: &OrderRecords,
    pos: &Position,
    last: usize,
    exit_idx: i64,
    status: TradeStatus,
    parent_id: i64,
) {
    let exit_price = pos.exit_gross_sum / pos.exit_size_sum;
    for c in pos.first..=last {
        if !is_entry(pos.direction, orders.side[c]) {
            continue;
        }
        let (entry_size, entry_fees) = if c == pos.first {
            (pos.first_entry_size, pos.first_entry_fees)
        } else {
            (orders.size[c], orders.fees[c])
        };
        let exit_fees = entry_size / pos.exit_size_sum * pos.exit_fees_sum;
        out.push(
            orders.col[c],
            entry_size,
            orders.idx[c],
            orders.price[c],
            entry_fees,
            exit_idx,
            exit_price,
            exit_fees,
            pos.direction,
            status,
            parent_id,
            orders.multiplier_of(c),
        );
    }
}

/// One record per entry fill: buys in a long position, sells in a short one.
///
/// Answers "how did each entry do?". A position built from three buys and
/// closed by one sell becomes three records, all sharing that sell's price.
///
/// `close` supplies the marking price for a position still open at the end;
/// only its last row is read.
///
/// # Errors
///
/// Returns [`Error::Invariant`] if the records violate the ordering contract
/// or contain a size that is not above 0, or a price that is not finite (or
/// is 0 or less without `allow_negative_price`).
pub fn entry_trades(
    orders: &OrderRecords,
    close: &FlexArray<'_, f64>,
    n_rows: usize,
    n_cols: usize,
) -> Result<TradeRecords> {
    let ranges = column_ranges(&orders.col, n_cols)?;
    let mut out = TradeRecords::new();
    // Positions are numbered across the whole record set, not per column, so
    // a parent id identifies a position on its own.
    let mut parent_id = -1_i64;

    for (col, range) in ranges.iter().enumerate() {
        let mut pos: Option<Position> = None;

        for c in range.clone() {
            check_order(orders, c)?;
            let (size, price, fees, side) = (
                orders.size[c],
                orders.price[c],
                orders.fees[c],
                orders.side[c],
            );

            if pos.is_none() {
                parent_id += 1;
                pos = Some(Position {
                    first: c,
                    direction: opened_direction(side),
                    entry_size_sum: 0.0,
                    first_entry_size: size,
                    first_entry_fees: fees,
                    exit_size_sum: 0.0,
                    exit_gross_sum: 0.0,
                    exit_fees_sum: 0.0,
                });
            }
            let p = pos.as_mut().expect("just opened if absent");

            if is_entry(p.direction, side) {
                p.entry_size_sum += size;
                continue;
            }

            if is_close(p.exit_size_sum + size, p.entry_size_sum) {
                // Exactly closed.
                p.exit_size_sum = p.entry_size_sum;
                p.exit_gross_sum += size * price;
                p.exit_fees_sum += fees;
                flush_entry_trades(
                    &mut out,
                    orders,
                    p,
                    c,
                    orders.idx[c],
                    TradeStatus::Closed,
                    parent_id,
                );
                pos = None;
            } else if is_less(p.exit_size_sum + size, p.entry_size_sum) {
                // Partially closed; the position stays open.
                p.exit_size_sum += size;
                p.exit_gross_sum += size * price;
                p.exit_fees_sum += fees;
            } else {
                // Closed and reversed by one fill. The fill is split: the part
                // that closes belongs to the old position, the rest opens a
                // new one, and both take a proportional share of its fees.
                let remaining = add(p.entry_size_sum, -p.exit_size_sum);
                p.exit_size_sum = p.entry_size_sum;
                p.exit_gross_sum += remaining * price;
                p.exit_fees_sum += remaining / size * fees;
                flush_entry_trades(
                    &mut out,
                    orders,
                    p,
                    c,
                    orders.idx[c],
                    TradeStatus::Closed,
                    parent_id,
                );

                parent_id += 1;
                let opened = add(size, -remaining);
                pos = Some(Position {
                    first: c,
                    direction: opened_direction(side),
                    entry_size_sum: opened,
                    first_entry_size: opened,
                    first_entry_fees: opened / size * fees,
                    exit_size_sum: 0.0,
                    exit_gross_sum: 0.0,
                    exit_fees_sum: 0.0,
                });
            }
        }

        // A position still open at the last bar is marked to the close and
        // reported as open rather than dropped, so its unrealized PnL counts.
        if let Some(p) = pos.as_mut()
            && is_less(p.exit_size_sum, p.entry_size_sum)
        {
            let remaining = add(p.entry_size_sum, -p.exit_size_sum);
            p.exit_size_sum = p.entry_size_sum;
            p.exit_gross_sum += remaining * close.get(n_rows - 1, col);
            flush_entry_trades(
                &mut out,
                orders,
                p,
                range.end - 1,
                n_rows as i64 - 1,
                TradeStatus::Open,
                parent_id,
            );
        }
    }

    Ok(out)
}

/// One record per exit fill: sells in a long position, buys in a short one.
///
/// Answers "how did each exit do?". A position built from three buys and
/// closed by one sell becomes one record, entered at the three buys'
/// size-weighted average.
///
/// A partial exit scales the remaining entry sums down rather than consuming
/// them front to back: what is left of the position keeps the same average
/// entry price, and the fees already attributed to the exited part are gone
/// from it.
///
/// # Errors
///
/// As [`entry_trades`].
pub fn exit_trades(
    orders: &OrderRecords,
    close: &FlexArray<'_, f64>,
    n_rows: usize,
    n_cols: usize,
) -> Result<TradeRecords> {
    let ranges = column_ranges(&orders.col, n_cols)?;
    let mut out = TradeRecords::new();
    let mut parent_id = -1_i64;

    for (col, range) in ranges.iter().enumerate() {
        let mut in_position = false;
        let mut direction = TradeDirection::Long;
        let mut entry_idx = -1_i64;
        let mut entry_size_sum = 0.0;
        let mut entry_gross_sum = 0.0;
        let mut entry_fees_sum = 0.0;

        for c in range.clone() {
            check_order(orders, c)?;
            let (size, price, fees, side) = (
                orders.size[c],
                orders.price[c],
                orders.fees[c],
                orders.side[c],
            );

            if !in_position {
                in_position = true;
                entry_idx = orders.idx[c];
                direction = opened_direction(side);
                parent_id += 1;
                entry_size_sum = 0.0;
                entry_gross_sum = 0.0;
                entry_fees_sum = 0.0;
            }

            if is_entry(direction, side) {
                entry_size_sum += size;
                entry_gross_sum += size * price;
                entry_fees_sum += fees;
                continue;
            }

            let entry_price = entry_gross_sum / entry_size_sum;

            if is_close_or_less(size, entry_size_sum) {
                // The exit fits inside the position. Snapping an
                // almost-exact exit to the position's size is what keeps a
                // full close from leaving a dust position behind.
                let exit_size = if is_close(size, entry_size_sum) {
                    entry_size_sum
                } else {
                    size
                };
                out.push(
                    orders.col[c],
                    exit_size,
                    entry_idx,
                    entry_price,
                    exit_size / entry_size_sum * entry_fees_sum,
                    orders.idx[c],
                    price,
                    fees,
                    direction,
                    TradeStatus::Closed,
                    parent_id,
                    orders.multiplier_of(c),
                );

                if is_close(size, entry_size_sum) {
                    in_position = false;
                    entry_idx = -1;
                } else {
                    let kept = (entry_size_sum - size) / entry_size_sum;
                    entry_size_sum *= kept;
                    entry_gross_sum *= kept;
                    entry_fees_sum *= kept;
                }
            } else {
                // The fill closes the position and opens the opposite one.
                let closed_size = entry_size_sum;
                let closed_fees = closed_size / size * fees;
                out.push(
                    orders.col[c],
                    closed_size,
                    entry_idx,
                    entry_price,
                    entry_fees_sum,
                    orders.idx[c],
                    price,
                    closed_fees,
                    direction,
                    TradeStatus::Closed,
                    parent_id,
                    orders.multiplier_of(c),
                );

                entry_size_sum = size - closed_size;
                entry_gross_sum = entry_size_sum * price;
                entry_fees_sum = fees - closed_fees;
                entry_idx = orders.idx[c];
                direction = match direction {
                    TradeDirection::Long => TradeDirection::Short,
                    TradeDirection::Short => TradeDirection::Long,
                };
                parent_id += 1;
            }
        }

        if in_position && is_less(-entry_size_sum, 0.0) {
            out.push(
                orders.col[range.end - 1],
                entry_size_sum,
                entry_idx,
                entry_gross_sum / entry_size_sum,
                entry_fees_sum,
                n_rows as i64 - 1,
                close.get(n_rows - 1, col),
                0.0,
                direction,
                TradeStatus::Open,
                parent_id,
                orders.multiplier_of(range.end - 1),
            );
        }
    }

    Ok(out)
}

/// Aggregates trades sharing a parent into one record per position.
///
/// Accepts either view — and its own output — because all three carry the
/// same `parent_id`, and aggregating an already-aggregated position is the
/// identity.
///
/// # Errors
///
/// Returns [`Error::Invariant`] if the records violate the ordering contract.
pub fn positions(trades: &TradeRecords, n_cols: usize) -> Result<TradeRecords> {
    let ranges = column_ranges(&trades.col, n_cols)?;
    let mut out = TradeRecords::new();

    for range in ranges {
        let mut start = range.start;
        while start < range.end {
            let parent = trades.parent_id[start];
            let mut end = start + 1;
            while end < range.end && trades.parent_id[end] == parent {
                end += 1;
            }
            let id = out.len() as i64;

            if end - start == 1 {
                // See `push_copy`: re-deriving a one-trade position from
                // itself is not bit-identical to copying it.
                out.push_copy(trades, start, id);
            } else {
                let size: f64 = trades.size[start..end].iter().sum();
                let weighted = |v: &[f64]| -> f64 {
                    trades.size[start..end]
                        .iter()
                        .zip(v)
                        .map(|(s, p)| s * p)
                        .sum::<f64>()
                        / size
                };
                out.push(
                    trades.col[start],
                    size,
                    trades.entry_idx[start],
                    weighted(&trades.entry_price[start..end]),
                    trades.entry_fees[start..end].iter().sum(),
                    trades.exit_idx[end - 1],
                    weighted(&trades.exit_price[start..end]),
                    trades.exit_fees[start..end].iter().sum(),
                    // The last trade decides both: a position is open if its
                    // final trade is, and reversals already split positions,
                    // so the direction cannot disagree within one.
                    if trades.direction[end - 1] == TradeDirection::Short as i8 {
                        TradeDirection::Short
                    } else {
                        TradeDirection::Long
                    },
                    if trades.status[end - 1] == TradeStatus::Open as i8 {
                        TradeStatus::Open
                    } else {
                        TradeStatus::Closed
                    },
                    id,
                    trades.multiplier.get(start).copied().unwrap_or(1.0),
                );
            }
            start = end;
        }
    }

    Ok(out)
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "these assert on exact expected prices and PnLs; an approximate \
              assertion would not catch the attribution errors the tests exist \
              to catch"
)]
mod tests {
    use super::*;
    use crate::RecordSink;

    /// Buys at 1, 2, 3 then sells all three at 10.
    fn three_buys_one_sell() -> OrderRecords {
        let mut o = OrderRecords::new();
        o.push_order(0, 0, 1.0, 1.0, 0.0, OrderSide::Buy);
        o.push_order(1, 0, 1.0, 2.0, 0.0, OrderSide::Buy);
        o.push_order(2, 0, 1.0, 3.0, 0.0, OrderSide::Buy);
        o.push_order(3, 0, 3.0, 10.0, 0.0, OrderSide::Sell);
        o
    }

    fn close() -> FlexArray<'static, f64> {
        FlexArray::scalar(10.0)
    }

    #[test]
    fn entry_view_gives_one_record_per_entry() {
        let t = entry_trades(&three_buys_one_sell(), &close(), 4, 1).unwrap();
        assert_eq!(t.len(), 3);
        // Each entry keeps its own price and shares the single exit.
        assert_eq!(t.entry_price, [1.0, 2.0, 3.0]);
        assert_eq!(t.exit_price, [10.0, 10.0, 10.0]);
        assert_eq!(t.pnl, [9.0, 8.0, 7.0]);
        // One position, so one parent.
        assert_eq!(t.parent_id, [0, 0, 0]);
    }

    #[test]
    fn exit_view_gives_one_record_per_exit() {
        let t = exit_trades(&three_buys_one_sell(), &close(), 4, 1).unwrap();
        assert_eq!(t.len(), 1);
        assert_eq!(t.size, [3.0]);
        // The three entries collapse into their size-weighted average.
        assert_eq!(t.entry_price, [2.0]);
        assert_eq!(t.pnl, [24.0]);
    }

    #[test]
    fn the_views_disagree_on_count_but_not_on_pnl() {
        let o = three_buys_one_sell();
        let entry = entry_trades(&o, &close(), 4, 1).unwrap();
        let exit = exit_trades(&o, &close(), 4, 1).unwrap();
        assert_ne!(entry.len(), exit.len());
        let sum = |v: &[f64]| v.iter().sum::<f64>();
        assert_eq!(sum(&entry.pnl), sum(&exit.pnl));
    }

    #[test]
    fn positions_aggregate_either_view_alike() {
        let o = three_buys_one_sell();
        let from_entry = positions(&entry_trades(&o, &close(), 4, 1).unwrap(), 1).unwrap();
        let from_exit = positions(&exit_trades(&o, &close(), 4, 1).unwrap(), 1).unwrap();
        assert_eq!(from_entry.len(), 1);
        assert_eq!(from_exit.len(), 1);
        assert_eq!(from_entry.size, from_exit.size);
        assert_eq!(from_entry.pnl, from_exit.pnl);
        assert_eq!(from_entry.entry_price, [2.0]);
    }

    #[test]
    fn positions_of_positions_is_the_identity() {
        let o = three_buys_one_sell();
        let once = positions(&exit_trades(&o, &close(), 4, 1).unwrap(), 1).unwrap();
        let twice = positions(&once, 1).unwrap();
        assert_eq!(once, twice);
    }

    #[test]
    fn an_open_position_is_marked_to_the_last_close() {
        let mut o = OrderRecords::new();
        o.push_order(0, 0, 2.0, 5.0, 0.0, OrderSide::Buy);
        let t = exit_trades(&o, &FlexArray::scalar(8.0), 4, 1).unwrap();
        assert_eq!(t.status, [TradeStatus::Open as i8]);
        assert_eq!(t.exit_price, [8.0]);
        assert_eq!(t.exit_idx, [3]);
        assert_eq!(t.pnl, [6.0]);
    }

    #[test]
    fn a_reversal_splits_into_two_positions() {
        // Long 1, then sell 3: closes the long and opens a short of 2.
        let mut o = OrderRecords::new();
        o.push_order(0, 0, 1.0, 10.0, 0.0, OrderSide::Buy);
        o.push_order(1, 0, 3.0, 12.0, 0.0, OrderSide::Sell);
        let t = exit_trades(&o, &FlexArray::scalar(11.0), 3, 1).unwrap();
        assert_eq!(t.len(), 2);
        assert_eq!(t.parent_id, [0, 1]);
        assert_eq!(
            t.direction,
            [TradeDirection::Long as i8, TradeDirection::Short as i8]
        );
        assert_eq!(t.size, [1.0, 2.0]);
        assert_eq!(
            t.status,
            [TradeStatus::Closed as i8, TradeStatus::Open as i8]
        );
        // The short entered at 12 and is marked at 11, so it is up 2.
        assert_eq!(t.pnl, [2.0, 2.0]);
    }

    #[test]
    fn a_short_profits_when_the_price_falls() {
        let mut o = OrderRecords::new();
        o.push_order(0, 0, 2.0, 10.0, 0.0, OrderSide::Sell);
        o.push_order(1, 0, 2.0, 6.0, 0.0, OrderSide::Buy);
        let t = exit_trades(&o, &close(), 2, 1).unwrap();
        assert_eq!(t.direction, [TradeDirection::Short as i8]);
        assert_eq!(t.pnl, [8.0]);
        assert_eq!(t.ret, [0.4]);
    }

    #[test]
    fn fees_come_off_both_sides() {
        let mut o = OrderRecords::new();
        o.push_order(0, 0, 1.0, 10.0, 0.5, OrderSide::Buy);
        o.push_order(1, 0, 1.0, 20.0, 1.0, OrderSide::Sell);
        let t = exit_trades(&o, &close(), 2, 1).unwrap();
        assert_eq!(t.pnl, [8.5]);
    }

    #[test]
    fn a_partial_exit_leaves_the_average_entry_alone() {
        let mut o = OrderRecords::new();
        o.push_order(0, 0, 1.0, 1.0, 0.0, OrderSide::Buy);
        o.push_order(1, 0, 1.0, 3.0, 0.0, OrderSide::Buy);
        o.push_order(2, 0, 1.0, 10.0, 0.0, OrderSide::Sell);
        let t = exit_trades(&o, &FlexArray::scalar(10.0), 4, 1).unwrap();
        assert_eq!(t.len(), 2);
        // Both the realized and the still-open half entered at the average.
        assert_eq!(t.entry_price, [2.0, 2.0]);
        assert_eq!(
            t.status,
            [TradeStatus::Closed as i8, TradeStatus::Open as i8]
        );
        // One position: the partial exit did not close it.
        assert_eq!(t.parent_id, [0, 0]);
    }

    // The three tests below assert on *exactness*, not on values. Each pins a
    // property the golden harness structurally cannot check, because the
    // difference it guards against is smaller than the tolerance that
    // comparison uses. Mutation testing found all three by surviving it.

    #[test]
    fn a_break_even_round_trip_returns_exactly_zero() {
        // Entering in three lots and leaving at the same price is
        // mathematically break-even, but the averaged entry price does not
        // divide out exactly: the difference here is 5.7e-14. `add` snaps it.
        // Plain subtraction would leave it, and a downstream `pnl > 0` would
        // then count this trade as a winner.
        let mut o = OrderRecords::new();
        o.push_order(0, 0, 4.745, 41.082, 0.0, OrderSide::Buy);
        o.push_order(1, 0, 2.035, 41.082, 0.0, OrderSide::Buy);
        o.push_order(2, 0, 0.337, 41.082, 0.0, OrderSide::Buy);
        o.push_order(3, 0, 7.117, 41.082, 0.0, OrderSide::Sell);
        let t = exit_trades(&o, &FlexArray::scalar(41.082), 4, 1).unwrap();
        assert_eq!(t.pnl, [0.0]);
        assert!(t.pnl[0].is_sign_positive(), "and not a negative zero");
    }

    #[test]
    fn an_almost_full_exit_records_the_whole_position() {
        // An exit one ULP short of the position closes it either way. What
        // the snap decides is the recorded size: the position's, not the
        // order's. Without it the record carries a size that never existed.
        let mut o = OrderRecords::new();
        o.push_order(0, 0, 1.0, 10.0, 0.0, OrderSide::Buy);
        o.push_order(1, 0, 1.0, 10.0, 0.0, OrderSide::Buy);
        o.push_order(
            2,
            0,
            f64::from_bits(2.0_f64.to_bits() - 1),
            12.0,
            0.0,
            OrderSide::Sell,
        );
        let t = exit_trades(&o, &FlexArray::scalar(12.0), 3, 1).unwrap();
        assert_eq!(t.len(), 1);
        assert_eq!(
            t.size,
            [2.0],
            "the exit size must snap to the position size"
        );
        assert_eq!(t.status, [TradeStatus::Closed as i8]);
    }

    #[test]
    fn a_one_trade_position_is_its_trade_bit_for_bit() {
        // Re-deriving would compute `sum(size * price) / size` for the price
        // and recompute the PnL from that. The result differs in the last
        // bits, which no tolerant comparison can see and which would make a
        // position disagree with the only trade it contains.
        // Two things had to line up for this to bite, and both took
        // searching. The entry view is the one that stores a raw order price;
        // the exit view divides a gross sum by a size, so re-deriving it is
        // idempotent and the mutation survives. And 5.128 * 58.78 / 5.128 is
        // 58.779999999999994, not 58.78 — most pairs do round-trip.
        let mut o = OrderRecords::new();
        o.push_order(0, 0, 5.128, 58.78, 0.017, OrderSide::Buy);
        o.push_order(1, 0, 5.128, 27.888, 0.013, OrderSide::Sell);
        let trades = entry_trades(&o, &FlexArray::scalar(27.888), 2, 1).unwrap();
        let pos = positions(&trades, 1).unwrap();
        assert_eq!(pos.len(), 1);
        assert_eq!(
            pos.entry_price[0].to_bits(),
            trades.entry_price[0].to_bits()
        );
        assert_eq!(pos.exit_price[0].to_bits(), trades.exit_price[0].to_bits());
        assert_eq!(pos.pnl[0].to_bits(), trades.pnl[0].to_bits());
        assert_eq!(pos.ret[0].to_bits(), trades.ret[0].to_bits());
    }

    #[test]
    fn columns_are_kept_apart() {
        let mut o = OrderRecords::new();
        o.push_order(0, 0, 1.0, 10.0, 0.0, OrderSide::Buy);
        o.push_order(1, 0, 1.0, 12.0, 0.0, OrderSide::Sell);
        o.push_order(0, 1, 1.0, 10.0, 0.0, OrderSide::Buy);
        o.push_order(1, 1, 1.0, 8.0, 0.0, OrderSide::Sell);
        let t = exit_trades(&o, &close(), 2, 2).unwrap();
        assert_eq!(t.col, [0, 1]);
        assert_eq!(t.pnl, [2.0, -2.0]);
        // Positions are numbered across the whole set, not per column.
        assert_eq!(t.parent_id, [0, 1]);
    }

    #[test]
    fn records_out_of_column_order_are_rejected() {
        let mut o = OrderRecords::new();
        o.push_order(0, 1, 1.0, 10.0, 0.0, OrderSide::Buy);
        o.push_order(1, 0, 1.0, 10.0, 0.0, OrderSide::Buy);
        assert!(exit_trades(&o, &close(), 2, 2).is_err());
    }

    #[test]
    fn a_price_that_is_not_finite_is_rejected_and_one_below_zero_is_a_price() {
        let mut o = OrderRecords::new();
        o.push_order(0, 0, 1.0, f64::NAN, 0.0, OrderSide::Buy);
        assert!(exit_trades(&o, &close(), 2, 1).is_err());
        assert!(entry_trades(&o, &close(), 2, 1).is_err());
        // Bought at -20, sold at 10: 30 made on an entry worth 20.
        let mut o = OrderRecords::new();
        o.allow_negative_price = true;
        o.push_order(0, 0, 1.0, -20.0, 0.0, OrderSide::Buy);
        o.push_order(1, 0, 1.0, 10.0, 0.0, OrderSide::Sell);
        let t = exit_trades(&o, &close(), 2, 1).unwrap();
        assert_eq!((t.pnl[0], t.ret[0]), (30.0, 1.5));
    }

    #[test]
    fn no_orders_means_no_trades() {
        let o = OrderRecords::new();
        assert!(entry_trades(&o, &close(), 4, 3).unwrap().is_empty());
        assert!(exit_trades(&o, &close(), 4, 3).unwrap().is_empty());
    }
}
