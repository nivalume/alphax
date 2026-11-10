//! Stop exits without a simulation: where the stops of each entry would
//! fire, read off the bars.
//!
//! [`generate_ohlc_stop_exits`] arms the stops of `crate::stops` at each
//! entry's close and walks the bars after it with the same test and the same
//! trailing the simulation uses, so the two agree. With
//! `skip_until_exit = true` and `until_next = false` the exits are the stop
//! exits `from_signals` fills for the same entries, stops and bars when it
//! ignores an entry while a position is held (`accumulate` off).
//!
//! - `wait` is how many bars after the entry the first test is, at least 1:
//!   the entry is at its bar's close.
//! - `until_next` ends an entry's search at the next entry, which starts its
//!   own; a stop that fires on that bar fires first, inside the bar.
//! - `skip_until_exit` ignores the entries before an entry's exit, as a
//!   simulation holding the position would. Without it each entry searches
//!   on its own, and a bar two of them exit on keeps the first's exit.
//! - A ladder (`stop_ladder`) exits once per step, each closing the share
//!   of the entry's position `size` says; the last step closes the rest.
//! - `stop_order_type = Limit` places a stop-limit order where a price stop
//!   fires, `stop_limit_delta` worse than its level; the exit is where that
//!   order fills, which may be a later bar or none. It cannot be combined
//!   with a ladder here.

use alphax_core::error::{Error, Result};
use alphax_core::flex::FlexArray;
use alphax_model::{DeltaFormat, LimitOrderPrice, OrderType, StopLadderMode};

use crate::stops::{self, Kind};

/// Inputs to [`generate_ohlc_stop_exits`]. Stops are fractions of the
/// entry's close, `NaN` disabling one, read on the entry's row; the time
/// stops count rows. See `crate::stops`.
#[derive(Debug, Clone)]
pub struct OhlcStopExits<'a> {
    /// Number of bars.
    pub n_rows: usize,
    /// Number of columns.
    pub n_cols: usize,
    /// Where positions are entered, at the bar's close.
    pub entries: FlexArray<'a, bool>,
    /// Open; `NaN` is the close.
    pub open: FlexArray<'a, f64>,
    /// High; `NaN` is the higher of the open and the close.
    pub high: FlexArray<'a, f64>,
    /// Low; `NaN` is the lower of the open and the close.
    pub low: FlexArray<'a, f64>,
    /// Close, which the stops are anchored at.
    pub close: FlexArray<'a, f64>,
    /// Stop-loss distance.
    pub sl_stop: FlexArray<'a, f64>,
    /// Whether the stop-loss trails.
    pub sl_trail: FlexArray<'a, bool>,
    /// Trailing stop distance.
    pub tsl_stop: FlexArray<'a, f64>,
    /// The trailing stop's activation threshold.
    pub tsl_th: FlexArray<'a, f64>,
    /// Take-profit distance.
    pub tp_stop: FlexArray<'a, f64>,
    /// Exit this many bars after the entry.
    pub td_stop: FlexArray<'a, f64>,
    /// Exit on this row.
    pub dt_stop: FlexArray<'a, f64>,
    /// How the stop distances read. See `crate::stops`.
    pub delta_format: FlexArray<'a, DeltaFormat>,
    /// Whether the rows of `sl_stop`, `tsl_stop` and `tp_stop` are a
    /// ladder's steps.
    pub stop_ladder: StopLadderMode,
    /// Whether a price stop that fires exits at market or places a limit.
    pub stop_order_type: FlexArray<'a, OrderType>,
    /// How much worse than the stop's level its limit is.
    pub stop_limit_delta: FlexArray<'a, f64>,
    /// What a stop-limit order fills at. See [`crate::limit`].
    pub limit_order_price: FlexArray<'a, LimitOrderPrice>,
    /// Whether the positions are short, which mirrors every stop.
    pub short: bool,
    /// Bars after the entry before the first test; at least 1.
    pub wait: usize,
    /// Whether an entry's search ends at the next entry.
    pub until_next: bool,
    /// Whether entries before an exit are ignored.
    pub skip_until_exit: bool,
}

impl Default for OhlcStopExits<'_> {
    fn default() -> Self {
        Self {
            n_rows: 0,
            n_cols: 0,
            entries: FlexArray::Scalar(false),
            open: FlexArray::Scalar(f64::NAN),
            high: FlexArray::Scalar(f64::NAN),
            low: FlexArray::Scalar(f64::NAN),
            close: FlexArray::Scalar(f64::NAN),
            sl_stop: FlexArray::Scalar(f64::NAN),
            sl_trail: FlexArray::Scalar(false),
            tsl_stop: FlexArray::Scalar(f64::NAN),
            tsl_th: FlexArray::Scalar(f64::NAN),
            tp_stop: FlexArray::Scalar(f64::NAN),
            td_stop: FlexArray::Scalar(f64::NAN),
            dt_stop: FlexArray::Scalar(f64::NAN),
            delta_format: FlexArray::Scalar(DeltaFormat::Percent),
            stop_ladder: StopLadderMode::Disabled,
            stop_order_type: FlexArray::Scalar(OrderType::Market),
            stop_limit_delta: FlexArray::Scalar(f64::NAN),
            limit_order_price: FlexArray::Scalar(LimitOrderPrice::Limit),
            short: false,
            wait: 1,
            until_next: true,
            skip_until_exit: false,
        }
    }
}

/// Where the stops fired, column-major like the inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct StopExits {
    /// Whether a stop fired on the bar.
    pub exits: Vec<bool>,
    /// The price it fired at, the level or the open past it; `NaN` elsewhere.
    pub price: Vec<f64>,
    /// Which stop: 0 stop-loss, 1 trailing, 2 take-profit, 3 time; `-1`
    /// elsewhere.
    pub kind: Vec<i8>,
    /// The share of the entry's position the exit closes: 1 but for a
    /// ladder's steps; `NaN` elsewhere.
    pub size: Vec<f64>,
}

/// One exit of an entry: its bar, where it filled, which stop, and the
/// share of the position it closed.
#[derive(Debug, Clone, Copy)]
struct Exit {
    row: usize,
    price: f64,
    kind: Kind,
    size: f64,
}

const fn kind_code(kind: Kind) -> i8 {
    match kind {
        Kind::StopLoss => 0,
        Kind::Trailing => 1,
        Kind::TakeProfit => 2,
        Kind::Time => 3,
    }
}

/// The exits the stops of `args`' entries would take. See the module docs.
///
/// # Errors
///
/// Returns [`Error::Invariant`] for `wait` 0, a negative stop distance or a
/// time stop that cannot be read as one.
pub fn generate_ohlc_stop_exits(args: &OhlcStopExits<'_>) -> Result<StopExits> {
    if args.wait == 0 {
        return Err(Error::Invariant(
            "wait must be 1 or more: an entry is at its bar's close".into(),
        ));
    }
    let (n_rows, n_cols) = (args.n_rows, args.n_cols);
    let mut out = StopExits {
        exits: vec![false; n_rows * n_cols],
        price: vec![f64::NAN; n_rows * n_cols],
        kind: vec![-1; n_rows * n_cols],
        size: vec![f64::NAN; n_rows * n_cols],
    };
    let inputs = stops::Inputs {
        sl_stop: &args.sl_stop,
        sl_trail: &args.sl_trail,
        tp_stop: &args.tp_stop,
        tsl_stop: &args.tsl_stop,
        tsl_th: &args.tsl_th,
        td_stop: &args.td_stop,
        dt_stop: &args.dt_stop,
        delta_format: &args.delta_format,
        ladder: args.stop_ladder,
    };
    let position = if args.short { -1.0 } else { 1.0 };
    let mut exits = Vec::new();
    for col in 0..n_cols {
        let mut row = 0;
        while row < n_rows {
            if !args.entries.get(row, col) {
                row += 1;
                continue;
            }
            exits.clear();
            let (closed, stopped) = search(args, &inputs, row, col, position, &mut exits)?;
            for e in &exits {
                let k = col * n_rows + e.row;
                if !out.exits[k] {
                    out.exits[k] = true;
                    out.price[k] = e.price;
                    out.kind[k] = kind_code(e.kind);
                    out.size[k] = e.size;
                }
            }
            row = if !args.skip_until_exit {
                row + 1
            } else if closed {
                exits.last().map_or(stopped, |e| e.row + 1)
            } else {
                stopped
            };
        }
    }
    Ok(out)
}

/// One entry's search, pushing its exits: whether the position was closed,
/// and the row the search stopped at — the next entry under `until_next`,
/// or the last bar.
fn search(
    args: &OhlcStopExits<'_>,
    inputs: &stops::Inputs<'_, '_>,
    entry: usize,
    col: usize,
    position: f64,
    exits: &mut Vec<Exit>,
) -> Result<(bool, usize)> {
    let anchor = args.close.get(entry, col);
    let mut state = stops::arm(inputs, entry, col, anchor, position)?;
    let mut held = position.abs();
    // A stop-limit order waiting to fill: its level, and the stop it is for.
    let mut resting: Option<(f64, Kind, LimitOrderPrice)> = None;
    // An exit sells a long and buys back a short.
    let buy = position < 0.0;
    for row in entry + 1..args.n_rows {
        let close = args.close.get(row, col);
        let mut open = args.open.get(row, col);
        if open.is_nan() {
            open = close;
        }
        let mut high = args.high.get(row, col);
        if high.is_nan() {
            high = open.max(close);
        }
        let mut low = args.low.get(row, col);
        if low.is_nan() {
            low = open.min(close);
        }
        if row >= entry + args.wait {
            if let Some((level, kind, order_price)) = resting {
                let price = crate::limit::fill_price(
                    level,
                    buy,
                    false,
                    order_price,
                    open,
                    high,
                    low,
                    close,
                );
                if !price.is_nan() {
                    exits.push(Exit {
                        row,
                        price,
                        kind,
                        size: held,
                    });
                    return Ok((true, row));
                }
            } else if let Some(hit) = stops::test(&state, position, row, open, high, low)? {
                if args.stop_order_type.get(row, col) == OrderType::Limit {
                    if inputs.ladder != StopLadderMode::Disabled {
                        return Err(Error::Invariant(
                            "stop exits cannot combine a ladder with stop-limit orders".into(),
                        ));
                    }
                    let level = crate::limit::limit_price(
                        hit.level,
                        buy,
                        true,
                        args.stop_limit_delta.get(row, col),
                        args.delta_format.get(row, col),
                    )?;
                    let order_price = args.limit_order_price.get(row, col);
                    let price = crate::limit::fill_price(
                        level,
                        buy,
                        false,
                        order_price,
                        hit.price,
                        f64::NAN,
                        f64::NAN,
                        close,
                    );
                    if !price.is_nan() {
                        exits.push(Exit {
                            row,
                            price,
                            kind: hit.kind,
                            size: held,
                        });
                        return Ok((true, row));
                    }
                    resting = Some((level, hit.kind, order_price));
                } else {
                    match stops::ladder_exit(inputs, col, hit.kind, held * position, &mut state) {
                        Some(part) => {
                            exits.push(Exit {
                                row,
                                price: hit.price,
                                kind: hit.kind,
                                size: part,
                            });
                            held -= part;
                        }
                        None => {
                            exits.push(Exit {
                                row,
                                price: hit.price,
                                kind: hit.kind,
                                size: held,
                            });
                            return Ok((true, row));
                        }
                    }
                }
            }
        }
        if args.until_next && args.entries.get(row, col) {
            return Ok((false, row));
        }
        stops::trail(&mut state, position, high, low);
    }
    Ok((false, args.n_rows))
}

#[cfg(test)]
#[allow(clippy::float_cmp, reason = "the prices are exact in binary")]
mod tests {
    use super::*;

    const OPEN: [f64; 5] = [100., 99., 97., 93., 95.];
    const HIGH: [f64; 5] = [101., 100., 98., 96., 103.];
    const LOW: [f64; 5] = [99., 96., 94., 92., 95.];
    const CLOSE: [f64; 5] = [100., 97., 96., 95., 102.];

    fn bars<'a>(entries: &'a [bool]) -> OhlcStopExits<'a> {
        OhlcStopExits {
            n_rows: 5,
            n_cols: 1,
            entries: FlexArray::per_row(entries, 5).unwrap(),
            open: FlexArray::per_row(&OPEN, 5).unwrap(),
            high: FlexArray::per_row(&HIGH, 5).unwrap(),
            low: FlexArray::per_row(&LOW, 5).unwrap(),
            close: FlexArray::per_row(&CLOSE, 5).unwrap(),
            ..OhlcStopExits::default()
        }
    }

    #[test]
    fn an_entry_exits_where_its_stop_first_fires() {
        let entries = [true, false, false, false, false];
        let out = generate_ohlc_stop_exits(&OhlcStopExits {
            sl_stop: FlexArray::scalar(0.05),
            ..bars(&entries)
        })
        .unwrap();
        // 95 is first reached on bar 2.
        assert_eq!(out.exits, [false, false, true, false, false]);
        assert_eq!(out.price[2], 100. * (1. - 0.05));
        assert_eq!(out.kind, [-1, -1, 0, -1, -1]);
    }

    #[test]
    fn the_next_entry_ends_a_search_and_skipping_waits_for_the_exit() {
        // Entries on bars 0 and 1; a 6.25% stop: 93.75 from 100 (reached by
        // bar 3's open), 90.9375 from 97 (never).
        let entries = [true, true, false, false, false];
        let base = OhlcStopExits {
            sl_stop: FlexArray::scalar(0.0625),
            ..bars(&entries)
        };
        let run = |until_next, skip| {
            generate_ohlc_stop_exits(&OhlcStopExits {
                until_next,
                skip_until_exit: skip,
                ..base.clone()
            })
            .unwrap()
            .exits
        };
        // The second entry ends the first's search before bar 3.
        assert_eq!(run(true, false), [false; 5]);
        assert_eq!(run(false, false), [false, false, false, true, false]);
        assert_eq!(run(false, true), [false, false, false, true, false]);
        // Waiting two bars changes nothing here; zero is refused.
        let wait = OhlcStopExits {
            wait: 0,
            ..base.clone()
        };
        assert!(generate_ohlc_stop_exits(&wait).is_err());
    }

    #[test]
    fn a_short_mirrors_the_stops_and_a_time_stop_fires_at_the_open() {
        let entries = [true, false, false, false, false];
        let out = generate_ohlc_stop_exits(&OhlcStopExits {
            tp_stop: FlexArray::scalar(0.0625),
            short: true,
            ..bars(&entries)
        })
        .unwrap();
        // A short's take-profit at 93.75: bar 3 opens at 93.
        assert_eq!((out.exits[3], out.price[3], out.kind[3]), (true, 93., 2));
        let out = generate_ohlc_stop_exits(&OhlcStopExits {
            td_stop: FlexArray::scalar(2.),
            ..bars(&entries)
        })
        .unwrap();
        assert_eq!((out.exits[2], out.price[2], out.kind[2]), (true, 97., 3));
    }

    #[test]
    fn a_ladder_exits_once_per_step() {
        static STEPS: [f64; 2] = [0.02, 0.05];
        let entries = [true, false, false, false, false];
        let out = generate_ohlc_stop_exits(&OhlcStopExits {
            sl_stop: FlexArray::PerRow(&STEPS),
            stop_ladder: StopLadderMode::Uniform,
            ..bars(&entries)
        })
        .unwrap();
        // 98 on bar 1, 95 on bar 2: half each.
        assert_eq!(out.exits, [false, true, true, false, false]);
        assert_eq!((out.price[1], out.price[2]), (98., 95.));
        assert_eq!((out.size[1], out.size[2]), (0.5, 0.5));
        assert!(out.size[0].is_nan());
    }

    #[test]
    fn a_stop_limit_exit_is_where_its_limit_fills() {
        let entries = [true, false, false, false, false];
        let out = generate_ohlc_stop_exits(&OhlcStopExits {
            // 93.5; bar 3 gaps to 93, below the limit 0.1% under the level,
            // which a sell then waits on: bar 4 opens at 95 and fills it.
            sl_stop: FlexArray::scalar(0.065),
            stop_order_type: FlexArray::scalar(OrderType::Limit),
            stop_limit_delta: FlexArray::scalar(0.001),
            ..bars(&entries)
        })
        .unwrap();
        assert_eq!(out.exits, [false, false, false, false, true]);
        assert_eq!((out.price[4], out.kind[4], out.size[4]), (95., 0, 1.));
        // Without the delta the gap price is the limit's own, and fills.
        let out = generate_ohlc_stop_exits(&OhlcStopExits {
            sl_stop: FlexArray::scalar(0.07),
            stop_order_type: FlexArray::scalar(OrderType::Limit),
            ..bars(&entries)
        })
        .unwrap();
        assert_eq!(out.exits, [false, false, false, true, false]);
        assert_eq!(out.price[3], 93.);
    }
}
