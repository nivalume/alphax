//! The stops beyond the plain stop-loss and take-profit, in
//! `from_signals`: a trailing stop with an activation threshold, time stops
//! and stop ladders. A stop that fires can also place a limit order instead
//! of exiting at market (`stop_order_type`, in `from_signals`).
//!
//! # The stops
//!
//! | Input | Fires | At |
//! |---|---|---|
//! | `sl_stop` | the price falls `sl_stop` below its anchor (a long; a short mirrors it) | the level, or the open past it |
//! | `tsl_stop`, `tsl_th` | the price falls `tsl_stop` below the best price since entry, once it has first risen `tsl_th` above the anchor (`NaN` or 0: at once) | the level, or the open past it |
//! | `tp_stop` | the price rises `tp_stop` above its anchor | the level, or the open past it |
//! | `td_stop` | `td_stop` bars after the entry bar | the open |
//! | `dt_stop` | on the row `dt_stop` | the open |
//!
//! The anchor is the `stop_entry_price` of the fill that opened the
//! position. Each bar tests the time stops first, then the stop-loss, the
//! trailing stop and the take-profit, and the first that fires exits. The
//! best price and the activation are updated after the test, so a bar is
//! judged against the best price as of the bar before. The engine has no
//! timestamps: both time stops count rows.
//!
//! # Ladders
//!
//! With `stop_ladder` on, the rows of `sl_stop`, `tsl_stop` and `tp_stop`
//! are the steps of a ladder rather than bars: increasing distances, `NaN`
//! after the last. A step that fires closes part of the position and arms
//! the next step; the last step closes what is left, as a stop without a
//! ladder does. The part:
//!
//! | Mode | Each step closes |
//! |---|---|
//! | `Uniform` | an equal share of the position the ladder was armed on |
//! | `Weighted` | a share of it proportional to how far the step is beyond the one before |
//! | `AdaptUniform` | an equal share of what is left: the position over the steps still to fire |
//! | `AdaptWeighted` | a share of what is left proportional to the step's distance from the one before over the distance still to go |
//!
//! The first two size against the armed position, so a position cut by
//! something else closes sooner; the adaptive two size against what is held
//! when the step fires. Adding to the position does not re-arm a ladder.
//!
//! # Distances
//!
//! `delta_format` says how a distance reads, from the anchor (or, trailing,
//! the best price): `Percent` a fraction (the default), `Percent100` a
//! percentage, `Absolute` an amount of price, `Target` the level itself. A
//! ladder takes distances, not levels.

use alphax_core::error::{Error, Result};
use alphax_core::flex::FlexArray;
use alphax_model::{DeltaFormat, StopLadderMode, StopUpdateMode};

use crate::from_signals::stop_price;
use crate::state::StopState;

/// The stop inputs, borrowed from `from_signals`' arguments.
#[derive(Clone, Copy)]
pub(crate) struct Inputs<'s, 'a> {
    pub(crate) sl_stop: &'s FlexArray<'a, f64>,
    pub(crate) sl_trail: &'s FlexArray<'a, bool>,
    pub(crate) tp_stop: &'s FlexArray<'a, f64>,
    pub(crate) tsl_stop: &'s FlexArray<'a, f64>,
    pub(crate) tsl_th: &'s FlexArray<'a, f64>,
    pub(crate) td_stop: &'s FlexArray<'a, f64>,
    pub(crate) dt_stop: &'s FlexArray<'a, f64>,
    pub(crate) delta_format: &'s FlexArray<'a, DeltaFormat>,
    pub(crate) ladder: StopLadderMode,
}

impl Inputs<'_, '_> {
    /// Whether any column may arm a stop beyond the plain stop-loss and
    /// take-profit, or a ladder: decided once, so a run without them keeps the
    /// plain stop engine.
    pub(crate) fn on(&self) -> bool {
        let nan = |a: &FlexArray<'_, f64>| matches!(a, FlexArray::Scalar(v) if v.is_nan());
        self.ladder != StopLadderMode::Disabled
            || !matches!(self.delta_format, FlexArray::Scalar(DeltaFormat::Percent))
            || !nan(self.tsl_stop)
            || !nan(self.td_stop)
            || !nan(self.dt_stop)
    }
}

/// Which stop fired.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    StopLoss,
    Trailing,
    TakeProfit,
    Time,
}

/// A stop that fired on a bar.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Hit {
    pub(crate) kind: Kind,
    /// Where it fired: its level, or the open the bar gapped to.
    pub(crate) price: f64,
    /// Its level; the open for a time stop.
    pub(crate) level: f64,
}

/// A stop's level for a position: below the anchor for a long's stop-loss
/// and a short's take-profit, above it otherwise, `stop` read by `format`.
pub(crate) fn level(position: f64, anchor: f64, stop: f64, loss: bool, format: DeltaFormat) -> f64 {
    let below = (position > 0.0) == loss;
    match format {
        DeltaFormat::Percent if below => anchor * (1.0 - stop),
        DeltaFormat::Percent => anchor * (1.0 + stop),
        DeltaFormat::Percent100 if below => anchor * (1.0 - stop / 100.0),
        DeltaFormat::Percent100 => anchor * (1.0 + stop / 100.0),
        DeltaFormat::Absolute if below => anchor - stop,
        DeltaFormat::Absolute => anchor + stop,
        DeltaFormat::Target => stop,
    }
}

/// Where a stop at `level` fills on a bar: the open when the bar gaps past
/// it, the level when the bar reaches it, `NaN` otherwise. `below` is a
/// level the price falls to.
pub(crate) fn fill(level: f64, below: bool, open: f64, low: f64, high: f64) -> f64 {
    let gapped = if below { open <= level } else { level <= open };
    if gapped {
        open
    } else if low <= level && level <= high {
        level
    } else {
        f64::NAN
    }
}

/// Tests a position's stops on bar `row`, in the order of the module table.
///
/// # Errors
///
/// Returns [`Error::Invariant`] for a negative stop distance.
pub(crate) fn test(
    stops: &StopState,
    position: f64,
    row: usize,
    open: f64,
    high: f64,
    low: f64,
) -> Result<Option<Hit>> {
    if position == 0.0 {
        return Ok(None);
    }
    if row >= stops.td_row.min(stops.dt_row) {
        return Ok(Some(Hit {
            kind: Kind::Time,
            price: open,
            level: open,
        }));
    }
    let price_stops = [
        (Kind::StopLoss, stops.sl_price, stops.sl_stop, true),
        (
            Kind::Trailing,
            stops.tsl_price,
            if stops.tsl_active {
                stops.tsl_stop
            } else {
                f64::NAN
            },
            true,
        ),
        (Kind::TakeProfit, stops.tp_price, stops.tp_stop, false),
    ];
    for (kind, anchor, stop, loss) in price_stops {
        if stop.is_nan() {
            continue;
        }
        let price = if stops.delta_format == DeltaFormat::Percent {
            stop_price(position, anchor, stop, open, low, high, loss)?
        } else {
            if stop < 0.0 {
                return Err(Error::Invariant(
                    "a stop distance must be 0 or greater".into(),
                ));
            }
            let at = level(position, anchor, stop, loss, stops.delta_format);
            fill(at, (position > 0.0) == loss, open, low, high)
        };
        if !price.is_nan() {
            return Ok(Some(Hit {
                kind,
                price,
                level: level(position, anchor, stop, loss, stops.delta_format),
            }));
        }
    }
    Ok(None)
}

/// Moves the trailing stops after a bar's test: the stop-loss with
/// `sl_trail`, and the trailing stop's best price and activation.
pub(crate) fn trail(stops: &mut StopState, position: f64, high: f64, low: f64) {
    let long = position > 0.0;
    if !stops.sl_stop.is_nan() && stops.sl_trail {
        if long && high > stops.sl_price {
            stops.sl_price = high;
        } else if position < 0.0 && low < stops.sl_price {
            stops.sl_price = low;
        }
    }
    if stops.tsl_stop.is_nan() || position == 0.0 {
        return;
    }
    // The threshold is a take-profit-like level, read as the distances are.
    let activation = level(
        position,
        stops.tsl_anchor,
        stops.tsl_th,
        false,
        stops.delta_format,
    );
    if long {
        if high > stops.tsl_price {
            stops.tsl_price = high;
        }
        if !stops.tsl_active && high >= activation {
            stops.tsl_active = true;
        }
    } else {
        if low < stops.tsl_price {
            stops.tsl_price = low;
        }
        if !stops.tsl_active && low <= activation {
            stops.tsl_active = true;
        }
    }
}

/// How many steps a ladder has in column `col`: its rows up to the first
/// `NaN`. A scalar or one value per column is one step.
fn ladder_len(steps: &FlexArray<'_, f64>, col: usize) -> usize {
    let rows = match *steps {
        FlexArray::Scalar(_) | FlexArray::PerColumn(_) => 1,
        FlexArray::PerRow(v) => v.len(),
        FlexArray::Full { n_rows, .. } | FlexArray::MappedColumns { n_rows, .. } => n_rows,
    };
    (0..rows)
        .take_while(|&k| !steps.get(k, col).is_nan())
        .count()
}

/// Checks a ladder's steps: 0 or greater and increasing.
fn check_ladder(steps: &FlexArray<'_, f64>, col: usize, name: &str) -> Result<()> {
    let mut last = 0.0;
    for k in 0..ladder_len(steps, col) {
        let step = steps.get(k, col);
        if !step.is_finite() || step < last {
            return Err(Error::Invariant(format!(
                "the steps of a {name} ladder must be finite, 0 or greater and increasing; \
                 step {k} of column {col} is {step}"
            )));
        }
        last = step;
    }
    Ok(())
}

/// The last row a position may be held through before a time stop: `None`
/// in `NaN` is no time stop.
fn time_row(value: f64, from: usize, name: &str) -> Result<usize> {
    if value.is_nan() {
        return Ok(usize::MAX);
    }
    if !value.is_finite() || value < 0.0 {
        return Err(Error::Invariant(format!(
            "{name} must be NaN, or finite and 0 or greater; got {value}"
        )));
    }
    Ok(from.saturating_add(value.ceil() as usize))
}

/// The stops a position opened (or reversed) on bar `row` at `anchor`
/// starts with.
///
/// # Errors
///
/// Returns [`Error::Invariant`] for a time stop or a ladder that cannot be
/// read as one.
pub(crate) fn arm(
    inputs: &Inputs<'_, '_>,
    row: usize,
    col: usize,
    anchor: f64,
    position: f64,
) -> Result<StopState> {
    let ladder = inputs.ladder != StopLadderMode::Disabled;
    let delta_format = inputs.delta_format.get(row, col);
    let (sl, tp, tsl) = if ladder {
        if delta_format == DeltaFormat::Target {
            return Err(Error::Invariant(
                "a stop ladder takes distances; delta_format='target' gives levels".into(),
            ));
        }
        check_ladder(inputs.sl_stop, col, "sl_stop")?;
        check_ladder(inputs.tsl_stop, col, "tsl_stop")?;
        check_ladder(inputs.tp_stop, col, "tp_stop")?;
        (
            inputs.sl_stop.get(0, col),
            inputs.tp_stop.get(0, col),
            inputs.tsl_stop.get(0, col),
        )
    } else {
        (
            inputs.sl_stop.get(row, col),
            inputs.tp_stop.get(row, col),
            inputs.tsl_stop.get(row, col),
        )
    };
    let th = inputs.tsl_th.get(row, col);
    let td_row = time_row(inputs.td_stop.get(row, col), row, "td_stop")?;
    let dt_row = time_row(inputs.dt_stop.get(row, col), 0, "dt_stop")?;
    Ok(StopState {
        sl_price: anchor,
        sl_stop: sl,
        sl_trail: inputs.sl_trail.get(row, col),
        tp_price: anchor,
        tp_stop: tp,
        tsl_price: anchor,
        tsl_stop: tsl,
        tsl_th: th,
        tsl_anchor: anchor,
        tsl_active: th.is_nan() || th <= 0.0,
        td_row,
        dt_row,
        sl_step: 0,
        tp_step: 0,
        tsl_step: 0,
        delta_format,
        ladder_base: position.abs(),
        extended: true,
    })
}

/// Re-arms the stops of a position added to on bar `row`, as
/// `upon_stop_update` says. A ladder keeps its steps.
///
/// # Errors
///
/// As [`arm`].
pub(crate) fn update(
    inputs: &Inputs<'_, '_>,
    row: usize,
    col: usize,
    anchor: f64,
    mode: StopUpdateMode,
    stops: &mut StopState,
) -> Result<()> {
    let replace = |stop: f64| match mode {
        StopUpdateMode::Keep => false,
        StopUpdateMode::Override => !stop.is_nan(),
        StopUpdateMode::OverrideNaN => true,
    };
    if inputs.ladder == StopLadderMode::Disabled {
        let sl = inputs.sl_stop.get(row, col);
        if replace(sl) {
            stops.sl_price = anchor;
            stops.sl_stop = sl;
            stops.sl_trail = inputs.sl_trail.get(row, col);
        }
        let tp = inputs.tp_stop.get(row, col);
        if replace(tp) {
            stops.tp_price = anchor;
            stops.tp_stop = tp;
        }
    }
    let tsl = inputs.tsl_stop.get(row, col);
    if inputs.ladder == StopLadderMode::Disabled && replace(tsl) {
        let th = inputs.tsl_th.get(row, col);
        stops.tsl_price = anchor;
        stops.tsl_stop = tsl;
        stops.tsl_th = th;
        stops.tsl_anchor = anchor;
        stops.tsl_active = th.is_nan() || th <= 0.0;
    }
    let td = inputs.td_stop.get(row, col);
    if replace(td) {
        stops.td_row = time_row(td, row, "td_stop")?;
    }
    let dt = inputs.dt_stop.get(row, col);
    if replace(dt) {
        stops.dt_row = time_row(dt, 0, "dt_stop")?;
    }
    Ok(())
}

/// For a ladder step that fired, the size it closes and the next step it
/// arms; `None` when the step is the last, or there is no ladder, and the
/// stop closes what is left.
pub(crate) fn ladder_exit(
    inputs: &Inputs<'_, '_>,
    col: usize,
    kind: Kind,
    position: f64,
    stops: &mut StopState,
) -> Option<f64> {
    if inputs.ladder == StopLadderMode::Disabled {
        return None;
    }
    let (steps, step, distance) = match kind {
        Kind::StopLoss => (inputs.sl_stop, &mut stops.sl_step, &mut stops.sl_stop),
        Kind::Trailing => (inputs.tsl_stop, &mut stops.tsl_step, &mut stops.tsl_stop),
        Kind::TakeProfit => (inputs.tp_stop, &mut stops.tp_step, &mut stops.tp_stop),
        Kind::Time => return None,
    };
    let n = ladder_len(steps, col);
    let k = *step;
    if k + 1 >= n {
        *distance = f64::NAN;
        return None;
    }
    let before = if k == 0 { 0.0 } else { steps.get(k - 1, col) };
    let gap = steps.get(k, col) - before;
    let last = steps.get(n - 1, col);
    let size = match inputs.ladder {
        StopLadderMode::Weighted => stops.ladder_base * gap / last,
        StopLadderMode::AdaptUniform => position.abs() / (n - k) as f64,
        StopLadderMode::AdaptWeighted => position.abs() * gap / (last - before),
        _ => stops.ladder_base / n as f64,
    };
    *step = k + 1;
    *distance = steps.get(k + 1, col);
    (size < position.abs()).then_some(size)
}

#[cfg(test)]
#[allow(clippy::float_cmp, reason = "the prices are exact in binary")]
mod tests {
    use super::*;

    const N: f64 = f64::NAN;

    fn inputs<'s>(
        sl: &'s FlexArray<'static, f64>,
        tp: &'s FlexArray<'static, f64>,
        ladder: StopLadderMode,
    ) -> Inputs<'s, 'static> {
        const NAN: FlexArray<'static, f64> = FlexArray::Scalar(f64::NAN);
        const NO: FlexArray<'static, bool> = FlexArray::Scalar(false);
        const PCT: FlexArray<'static, DeltaFormat> = FlexArray::Scalar(DeltaFormat::Percent);
        Inputs {
            sl_stop: sl,
            sl_trail: &NO,
            tp_stop: tp,
            tsl_stop: &NAN,
            tsl_th: &NAN,
            td_stop: &NAN,
            dt_stop: &NAN,
            delta_format: &PCT,
            ladder,
        }
    }

    #[test]
    fn a_trailing_stop_waits_for_its_threshold_then_trails_the_best_price() {
        let mut s = StopState {
            tsl_price: 100.0,
            tsl_stop: 0.25,
            tsl_th: 0.5,
            tsl_anchor: 100.0,
            tsl_active: false,
            extended: true,
            ..StopState::cleared()
        };
        // A fall of 25% from 100 before the threshold does not fire it.
        assert!(test(&s, 1.0, 1, 100.0, 100.0, 70.0).unwrap().is_none());
        trail(&mut s, 1.0, 140.0, 70.0);
        assert!(!s.tsl_active);
        trail(&mut s, 1.0, 160.0, 120.0);
        assert!(s.tsl_active);
        assert_eq!(s.tsl_price, 160.0);
        // 25% below 160.
        let hit = test(&s, 1.0, 3, 150.0, 150.0, 100.0).unwrap().unwrap();
        assert_eq!(
            (hit.kind, hit.price, hit.level),
            (Kind::Trailing, 120.0, 120.0)
        );
        // Short: the best price is the lowest.
        let mut short = StopState {
            tsl_price: 64.0,
            tsl_stop: 0.5,
            tsl_anchor: 64.0,
            tsl_active: true,
            extended: true,
            ..StopState::cleared()
        };
        trail(&mut short, -1.0, 70.0, 32.0);
        let hit = test(&short, -1.0, 2, 40.0, 50.0, 40.0).unwrap().unwrap();
        assert_eq!((hit.price, hit.level), (48.0, 48.0));
    }

    #[test]
    fn a_time_stop_fires_at_the_open_before_any_price_stop() {
        let s = StopState {
            sl_price: 100.0,
            sl_stop: 0.1,
            td_row: 5,
            dt_row: usize::MAX,
            extended: true,
            ..StopState::cleared()
        };
        assert!(test(&s, 1.0, 4, 100.0, 100.0, 95.0).unwrap().is_none());
        let hit = test(&s, 1.0, 5, 80.0, 90.0, 70.0).unwrap().unwrap();
        assert_eq!((hit.kind, hit.price), (Kind::Time, 80.0));
        assert!(test(&s, 0.0, 6, 80.0, 90.0, 70.0).unwrap().is_none());
    }

    #[test]
    fn stops_are_tested_loss_first_and_report_their_level() {
        let s = StopState {
            sl_price: 100.0,
            sl_stop: 0.25,
            tp_price: 100.0,
            tp_stop: 0.5,
            ..StopState::cleared()
        };
        // Both reached: the stop-loss.
        let hit = test(&s, 1.0, 1, 100.0, 160.0, 70.0).unwrap().unwrap();
        assert_eq!((hit.kind, hit.price), (Kind::StopLoss, 75.0));
        // A gap past the take-profit fills at the open, with the level kept.
        let hit = test(&s, 1.0, 1, 170.0, 180.0, 160.0).unwrap().unwrap();
        assert_eq!(
            (hit.kind, hit.price, hit.level),
            (Kind::TakeProfit, 170.0, 150.0)
        );
    }

    #[test]
    fn arming_reads_the_time_stops_as_rows() {
        let none = FlexArray::Scalar(N);
        let mut i = inputs(&none, &none, StopLadderMode::Disabled);
        let (td, dt) = (FlexArray::Scalar(3.0), FlexArray::Scalar(7.5));
        i.td_stop = &td;
        i.dt_stop = &dt;
        let s = arm(&i, 10, 0, 100.0, 2.0).unwrap();
        assert_eq!((s.td_row, s.dt_row), (13, 8));
        let bad = FlexArray::Scalar(-1.0);
        i.td_stop = &bad;
        assert!(arm(&i, 10, 0, 100.0, 2.0).is_err());
    }

    #[test]
    fn a_ladder_closes_shares_of_the_position_it_was_armed_on() {
        static STEPS: [f64; 3] = [0.1, 0.2, 0.4];
        let tp = FlexArray::PerRow(&STEPS);
        let none = FlexArray::Scalar(N);
        for (mode, shares) in [
            (StopLadderMode::Uniform, [10.0 / 3.0, 10.0 / 3.0]),
            (StopLadderMode::Weighted, [2.5, 2.5]),
        ] {
            let i = inputs(&none, &tp, mode);
            let mut s = arm(&i, 0, 0, 100.0, 10.0).unwrap();
            assert_eq!(s.tp_stop, 0.1);
            let first = ladder_exit(&i, 0, Kind::TakeProfit, 10.0, &mut s).unwrap();
            assert!((first - shares[0]).abs() < 1e-12, "{mode:?}");
            assert_eq!((s.tp_step, s.tp_stop), (1, 0.2));
            let second = ladder_exit(&i, 0, Kind::TakeProfit, 10.0 - first, &mut s).unwrap();
            assert!((second - shares[1]).abs() < 1e-12, "{mode:?}");
            // The last step closes the rest.
            assert_eq!(ladder_exit(&i, 0, Kind::TakeProfit, 5.0, &mut s), None);
            assert!(s.tp_stop.is_nan());
        }
        static FALLING: [f64; 2] = [0.2, 0.1];
        let bad = FlexArray::PerRow(&FALLING);
        assert!(
            arm(
                &inputs(&bad, &none, StopLadderMode::Uniform),
                0,
                0,
                100.0,
                1.0
            )
            .is_err()
        );
    }

    #[test]
    fn the_adaptive_ladders_size_against_what_is_held() {
        static STEPS: [f64; 3] = [0.1, 0.2, 0.4];
        let tp = FlexArray::PerRow(&STEPS);
        let none = FlexArray::Scalar(N);
        // Armed on 10; the second step fires holding 3 (something else sold
        // the rest): the fixed modes still close their share of 10, at least
        // what is left, so the step closes it all.
        for (mode, first, second) in [
            (StopLadderMode::AdaptUniform, 10.0 / 3.0, Some(1.5)),
            (StopLadderMode::AdaptWeighted, 2.5, Some(1.0)),
            (StopLadderMode::Uniform, 10.0 / 3.0, None),
        ] {
            let i = inputs(&none, &tp, mode);
            let mut s = arm(&i, 0, 0, 100.0, 10.0).unwrap();
            let got = ladder_exit(&i, 0, Kind::TakeProfit, 10.0, &mut s).unwrap();
            assert!((got - first).abs() < 1e-12, "{mode:?}");
            let got = ladder_exit(&i, 0, Kind::TakeProfit, 3.0, &mut s);
            match (got, second) {
                (Some(g), Some(w)) => assert!((g - w).abs() < 1e-12, "{mode:?}"),
                (None, None) => {}
                other => panic!("{mode:?}: {other:?}"),
            }
        }
    }

    #[test]
    fn a_trailing_stop_can_be_a_ladder() {
        static STEPS: [f64; 2] = [0.1, 0.2];
        let tsl = FlexArray::PerRow(&STEPS);
        let none = FlexArray::Scalar(N);
        let mut i = inputs(&none, &none, StopLadderMode::Uniform);
        i.tsl_stop = &tsl;
        let mut s = arm(&i, 0, 0, 100.0, 10.0).unwrap();
        assert_eq!(s.tsl_stop, 0.1);
        assert_eq!(ladder_exit(&i, 0, Kind::Trailing, 10.0, &mut s), Some(5.0));
        assert_eq!((s.tsl_step, s.tsl_stop), (1, 0.2));
        assert_eq!(ladder_exit(&i, 0, Kind::Trailing, 5.0, &mut s), None);
    }

    #[test]
    fn delta_format_reads_a_distance_as_a_fraction_a_percentage_a_price_or_a_level() {
        for (format, stop, below, above) in [
            (DeltaFormat::Percent, 0.25, 75.0, 125.0),
            (DeltaFormat::Percent100, 25.0, 75.0, 125.0),
            (DeltaFormat::Absolute, 25.0, 75.0, 125.0),
            (DeltaFormat::Target, 75.0, 75.0, 75.0),
        ] {
            // A long's stop-loss is below, its take-profit above.
            assert_eq!(level(1.0, 100.0, stop, true, format), below, "{format:?}");
            assert_eq!(level(1.0, 100.0, stop, false, format), above, "{format:?}");
        }
        // An absolute stop-loss, tested: 5 below 100 is reached by a low of 94.
        let s = StopState {
            sl_price: 100.0,
            sl_stop: 5.0,
            delta_format: DeltaFormat::Absolute,
            ..StopState::cleared()
        };
        let hit = test(&s, 1.0, 1, 99.0, 100.0, 94.0).unwrap().unwrap();
        assert_eq!((hit.price, hit.level), (95.0, 95.0));
        // A ladder takes distances, not levels.
        static STEPS: [f64; 2] = [90.0, 80.0];
        let sl = FlexArray::PerRow(&STEPS);
        let none = FlexArray::Scalar(N);
        let target = FlexArray::Scalar(DeltaFormat::Target);
        let mut i = inputs(&sl, &none, StopLadderMode::Uniform);
        i.delta_format = &target;
        assert!(arm(&i, 0, 0, 100.0, 1.0).is_err());
    }
}
