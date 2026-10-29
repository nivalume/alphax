//! Common `from_signals` callbacks, with parameters, run in Rust.
//!
//! [`Builtin`] bundles three rules that strategies otherwise write as
//! callbacks; each is off until its parameter is set, and they combine:
//!
//! * **Risk sizing** (`risk`, `risk_stop`): an order that opens or reverses a
//!   position buys or sells [`risk_size`] units, so that a move of
//!   `risk_stop` (a fraction of the bar's close) against it loses `risk` of
//!   the group's value.
//! * **Break-even stop** (`breakeven`): once the last close is `breakeven`
//!   (a fraction) past the entry in the position's favour, the stop-loss
//!   moves to the entry price, never loosening a stop already tighter. The
//!   trigger reads the last bar's close, so it cannot see the bar whose
//!   range then tests the stop. The run must test stops (`use_stops`).
//! * **Budget** (`max_positions`, `max_exposure`): a signal that would open
//!   a new position is dropped while the group already holds
//!   `max_positions` open positions (counting those opened earlier on the
//!   same bar), or while its gross exposure at the last valuation is at
//!   least `max_exposure` times its value. Exits and reversals of a held
//!   position are never dropped.
//!
//! The rules read only the group's own state, so they keep the loop's
//! guarantees: thread count and continuing a run change nothing.

use std::ops::Range;

use phasor_core::error::{Error, Result};
use phasor_exec::SizeType;
use phasor_model::DeltaFormat;

use crate::callbacks::{Context, Decision, SignalFunc, risk_size};
use crate::state::StopState;

/// The parameters; `NaN` and `0` mean off.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Builtin {
    /// The fraction of the group's value a stop-out may lose.
    pub risk: f64,
    /// The stop distance risk sizing assumes, as a fraction of the close.
    pub risk_stop: f64,
    /// The favourable move, as a fraction of the entry, that moves the
    /// stop-loss to the entry.
    pub breakeven: f64,
    /// The most positions open at once in a group; `0` is no limit.
    pub max_positions: usize,
    /// The most gross exposure, as a multiple of the group's value, at which
    /// a new position may still open.
    pub max_exposure: f64,
}

impl Default for Builtin {
    fn default() -> Self {
        Self {
            risk: f64::NAN,
            risk_stop: f64::NAN,
            breakeven: f64::NAN,
            max_positions: 0,
            max_exposure: f64::NAN,
        }
    }
}

impl Builtin {
    /// Checks the parameters.
    ///
    /// # Errors
    ///
    /// `risk` without `risk_stop` or the reverse, a negative or non-finite
    /// parameter that is set, or `breakeven` in a run that does not test
    /// stops (`use_stops = false`), where the moved stop would never fire.
    pub fn validate(&self, use_stops: bool) -> Result<()> {
        if !self.breakeven.is_nan() && !use_stops {
            return Err(Error::Invariant(
                "breakeven moves a stop, and this run does not test stops (use_stops)".into(),
            ));
        }
        if self.risk.is_nan() != self.risk_stop.is_nan() {
            return Err(Error::Invariant(
                "risk sizing needs both risk and risk_stop".into(),
            ));
        }
        for (name, v) in [
            ("risk", self.risk),
            ("risk_stop", self.risk_stop),
            ("breakeven", self.breakeven),
            ("max_exposure", self.max_exposure),
        ] {
            if !(v.is_nan() || v.is_finite() && v >= 0.0) {
                return Err(Error::Invariant(format!(
                    "{name} must be a nonnegative number; got {v}"
                )));
            }
        }
        Ok(())
    }
}

/// A group's state: the bar the entry count is for, and how many new
/// positions were let open on it.
#[derive(Debug, Clone, Copy, Default)]
pub struct BuiltinState {
    row: usize,
    opened: usize,
}

/// The price the stop-loss stands at, or `NaN` without one.
fn sl_level(position: f64, s: &StopState) -> f64 {
    if s.sl_stop.is_nan() {
        return f64::NAN;
    }
    crate::stops::level(position, s.sl_price, s.sl_stop, true, s.delta_format)
}

impl SignalFunc for Builtin {
    type State = BuiltinState;

    fn pre_group(&self, _group: usize, _cols: Range<usize>) -> Result<BuiltinState> {
        Ok(BuiltinState {
            row: usize::MAX,
            opened: 0,
        })
    }

    fn adjust(
        &self,
        ctx: &Context<'_>,
        k: usize,
        stops: &mut StopState,
        _state: &mut BuiltinState,
    ) -> Result<()> {
        let position = ctx.position(k);
        // The take-profit's anchor is the entry price, set when the stops
        // were armed whether or not a take-profit is.
        let entry = stops.tp_price;
        if self.breakeven.is_nan() || position == 0.0 || !entry.is_finite() {
            return Ok(());
        }
        let last = ctx.val_price(k);
        let gain = if position > 0.0 {
            last / entry - 1.0
        } else {
            1.0 - last / entry
        };
        if gain.is_nan() || gain < self.breakeven {
            return Ok(());
        }
        let current = sl_level(position, stops);
        let tighter = if position > 0.0 {
            current >= entry
        } else {
            current <= entry
        };
        if tighter {
            return Ok(());
        }
        stops.sl_price = entry;
        stops.sl_trail = false;
        stops.sl_stop = match stops.delta_format {
            DeltaFormat::Target => entry,
            DeltaFormat::Percent | DeltaFormat::Percent100 | DeltaFormat::Absolute => 0.0,
        };
        Ok(())
    }

    fn signals(
        &self,
        ctx: &Context<'_>,
        k: usize,
        decision: &mut Decision,
        state: &mut BuiltinState,
    ) -> Result<()> {
        if state.row != ctx.row {
            *state = BuiltinState {
                row: ctx.row,
                opened: 0,
            };
        }
        let position = ctx.position(k);
        let s = &mut decision.signals;
        let opens_long = s.long_entry && position <= 0.0;
        let opens_short = s.short_entry && position >= 0.0;
        if !(opens_long || opens_short) {
            return Ok(());
        }
        if position == 0.0 && !self.room(ctx, state) {
            s.long_entry = false;
            s.short_entry = false;
            return Ok(());
        }
        if position == 0.0 {
            state.opened += 1;
        }
        if !self.risk.is_nan() {
            let price = ctx.close(k);
            let stop = if opens_long {
                price * (1.0 - self.risk_stop)
            } else {
                price * (1.0 + self.risk_stop)
            };
            decision.size = risk_size(ctx.value, self.risk, price, stop, ctx.multiplier(k));
            decision.size_type = SizeType::Amount;
        }
        Ok(())
    }
}

impl Builtin {
    /// Whether the group may open one more position on this bar.
    fn room(&self, ctx: &Context<'_>, state: &BuiltinState) -> bool {
        if self.max_positions > 0 {
            let open = (0..ctx.len()).filter(|&k| ctx.position(k) != 0.0).count();
            if open + state.opened >= self.max_positions {
                return false;
            }
        }
        if !self.max_exposure.is_nan() {
            let gross: f64 = (0..ctx.len())
                .map(|k| (ctx.position(k) * ctx.val_price(k) * ctx.multiplier(k)).abs())
                .filter(|v| v.is_finite())
                .sum();
            if gross >= self.max_exposure * ctx.value {
                return false;
            }
        }
        true
    }
}
