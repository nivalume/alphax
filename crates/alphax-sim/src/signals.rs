//! Turning signals into an order request.
//!
//! `from_orders` is told what to trade. `from_signals` is told *when*, and has
//! to work out the rest — which means resolving several kinds of ambiguity
//! that a signal array can express and an order cannot:
//!
//! 1. A stop fired: it overrides the input signals entirely.
//! 2. Both an entry and an exit fired on the same bar, in one direction.
//! 3. A long entry and a short entry fired on the same bar.
//! 4. An entry fired against the position already held.
//! 5. What size that all adds up to, given the accumulation rules.
//!
//! Each stage is a pure function here, and the order between them is fixed:
//! a later stage assumes the earlier ones have already narrowed the
//! possibilities. Resolving the direction conflict before the entry/exit
//! conflict, for instance, changes which signal survives.
//!
//! Keeping them out of the loop is what lets them be tested one at a time
//! against golden outputs, the same reason `execute_order` is a separate
//! function from the loop that drives it.

use alphax_core::error::{Error, Result};
use alphax_core::math::is_less;
use alphax_model::{
    AccumulationMode, ConflictMode, Direction, DirectionConflictMode, OppositeEntryMode,
    PendingConflictMode, SizeType, StopExitMode,
};

/// The four direction-aware signals for one bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Signals {
    /// Open or add to a long.
    pub long_entry: bool,
    /// Close or reduce a long.
    pub long_exit: bool,
    /// Open or add to a short.
    pub short_entry: bool,
    /// Close or reduce a short.
    pub short_exit: bool,
}

impl Signals {
    /// Whether any signal fired.
    #[must_use]
    pub const fn any(self) -> bool {
        self.long_entry || self.long_exit || self.short_entry || self.short_exit
    }
}

/// Maps a direction-agnostic entry/exit pair onto direction-aware signals.
///
/// This is the older calling convention: one `entries` array, one `exits`
/// array, and a `direction` saying how to read them. In a short-only column
/// an "entry" means *short*, which is the whole reason the two conventions
/// cannot simply be concatenated.
#[must_use]
pub fn from_direction_mode(is_entry: bool, is_exit: bool, direction: Direction) -> Signals {
    match direction {
        Direction::LongOnly => Signals {
            long_entry: is_entry,
            long_exit: is_exit,
            ..Signals::default()
        },
        Direction::ShortOnly => Signals {
            short_entry: is_entry,
            short_exit: is_exit,
            ..Signals::default()
        },
        Direction::Both => Signals {
            long_entry: is_entry,
            short_entry: is_exit,
            ..Signals::default()
        },
    }
}

/// Turns a fired stop into signals, and possibly disables accumulation.
///
/// A stop does not mean "sell"; it means "get out of whatever you are in",
/// so the direction is read from the position rather than from the signal.
/// The modes that close rather than reduce also switch accumulation off: a
/// stop that only trimmed the position would leave it exposed to the very
/// move that triggered the stop.
#[must_use]
pub fn from_stop(
    position: f64,
    upon_stop_exit: StopExitMode,
    accumulate: AccumulationMode,
) -> (Signals, AccumulationMode) {
    let mut signals = Signals::default();
    let mut accumulate = accumulate;
    if position > 0.0 {
        match upon_stop_exit {
            StopExitMode::Close => {
                signals.long_exit = true;
                accumulate = AccumulationMode::Disabled;
            }
            StopExitMode::CloseReduce => signals.long_exit = true,
            StopExitMode::Reverse => {
                signals.short_entry = true;
                accumulate = AccumulationMode::Disabled;
            }
            StopExitMode::ReverseReduce => signals.short_entry = true,
        }
    } else if position < 0.0 {
        match upon_stop_exit {
            StopExitMode::Close => {
                signals.short_exit = true;
                accumulate = AccumulationMode::Disabled;
            }
            StopExitMode::CloseReduce => signals.short_exit = true,
            StopExitMode::Reverse => {
                signals.long_entry = true;
                accumulate = AccumulationMode::Disabled;
            }
            StopExitMode::ReverseReduce => signals.long_entry = true,
        }
    }
    (signals, accumulate)
}

/// Resolves an entry and an exit firing together in one direction.
///
/// `Adjacent` and `Opposite` are relative to the position held, so with no
/// position there is nothing to be adjacent or opposite *to* and both signals
/// are dropped. Silently picking one would invent a trade the signals did not
/// ask for.
#[must_use]
pub fn resolve_signal_conflict(
    position: f64,
    is_entry: bool,
    is_exit: bool,
    direction: Direction,
    mode: ConflictMode,
) -> (bool, bool) {
    if !(is_entry && is_exit) {
        return (is_entry, is_exit);
    }
    match mode {
        ConflictMode::Entry => (true, false),
        ConflictMode::Exit => (false, true),
        ConflictMode::Adjacent => {
            if position == 0.0 {
                (false, false)
            } else if direction == Direction::Both {
                if position > 0.0 {
                    (true, false)
                } else {
                    (false, true)
                }
            } else {
                // In a single-direction column the entry is always the
                // adjacent one: there is no opposite side to be in.
                (true, false)
            }
        }
        ConflictMode::Opposite => {
            if position == 0.0 {
                (false, false)
            } else if direction == Direction::Both {
                if position > 0.0 {
                    (false, true)
                } else {
                    (true, false)
                }
            } else {
                (false, true)
            }
        }
        ConflictMode::Ignore => (false, false),
    }
}

/// Resolves a long entry and a short entry firing on the same bar.
#[must_use]
pub fn resolve_dir_conflict(
    position: f64,
    long_entry: bool,
    short_entry: bool,
    mode: DirectionConflictMode,
) -> (bool, bool) {
    if !(long_entry && short_entry) {
        return (long_entry, short_entry);
    }
    match mode {
        DirectionConflictMode::Long => (true, false),
        DirectionConflictMode::Short => (false, true),
        DirectionConflictMode::Adjacent => {
            if position > 0.0 {
                (true, false)
            } else if position < 0.0 {
                (false, true)
            } else {
                (false, false)
            }
        }
        DirectionConflictMode::Opposite => {
            if position > 0.0 {
                (false, true)
            } else if position < 0.0 {
                (true, false)
            } else {
                (false, false)
            }
        }
        DirectionConflictMode::Ignore => (false, false),
    }
}

/// What becomes of a pending order and a bar's signal that meet: whether
/// the pending order is kept, and whether the signal executes.
///
/// The signal is *adjacent* when it trades on the same side as the pending
/// order — both buy or both sell — and *opposite* otherwise; each case has
/// its own mode.
#[must_use]
pub const fn resolve_pending_conflict(
    pending_buys: bool,
    signal_buys: bool,
    adjacent: PendingConflictMode,
    opposite: PendingConflictMode,
) -> (bool, bool) {
    let mode = if pending_buys == signal_buys {
        adjacent
    } else {
        opposite
    };
    match mode {
        PendingConflictMode::KeepIgnore => (true, false),
        PendingConflictMode::KeepExecute => (true, true),
        PendingConflictMode::CancelIgnore => (false, false),
        PendingConflictMode::CancelExecute => (false, true),
    }
}

/// Resolves an entry that opposes the position already held.
///
/// The distinction the five modes draw is between *closing* and *reversing*:
/// a short entry while long can mean "get flat" or "go short", and the
/// `Reduce` variants differ from their plain counterparts only in whether
/// accumulation is allowed to shrink the result afterwards.
#[must_use]
pub fn resolve_opposite_entry(
    position: f64,
    signals: Signals,
    mode: OppositeEntryMode,
    accumulate: AccumulationMode,
) -> (Signals, AccumulationMode) {
    let mut s = signals;
    let mut acc = accumulate;
    if position > 0.0 && s.short_entry {
        match mode {
            OppositeEntryMode::Ignore => s.short_entry = false,
            OppositeEntryMode::Close => {
                s.short_entry = false;
                s.long_exit = true;
                acc = AccumulationMode::Disabled;
            }
            OppositeEntryMode::CloseReduce => {
                s.short_entry = false;
                s.long_exit = true;
            }
            OppositeEntryMode::Reverse => acc = AccumulationMode::Disabled,
            OppositeEntryMode::ReverseReduce => {}
        }
    }
    if position < 0.0 && s.long_entry {
        match mode {
            OppositeEntryMode::Ignore => s.long_entry = false,
            OppositeEntryMode::Close => {
                s.long_entry = false;
                s.short_exit = true;
                acc = AccumulationMode::Disabled;
            }
            OppositeEntryMode::CloseReduce => {
                s.long_entry = false;
                s.short_exit = true;
            }
            OppositeEntryMode::Reverse => acc = AccumulationMode::Disabled,
            OppositeEntryMode::ReverseReduce => {}
        }
    }
    (s, acc)
}

/// What [`signals_to_size`] resolved a bar's signals into.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SizedOrder {
    /// Signed size; zero means no order.
    pub size: f64,
    /// How to interpret it.
    pub size_type: SizeType,
    /// Which sides the resulting order may take.
    pub direction: Direction,
}

/// Turns resolved signals into a size, a size type and a direction.
///
/// # Why `size` must be positive
///
/// Direction is carried by the *signals*, never by the sign of `size`. A
/// negative size would mean two things could disagree, so it is rejected
/// rather than reconciled.
///
/// # Reversal is one order, not two
///
/// Reversing sends a single order sized to close the position *and* open the
/// opposite one. That is why a percentage size cannot reverse: a percentage
/// of what — the position being closed, or the one being opened? So a
/// percentage size is rejected.
///
/// # Errors
///
/// Returns [`Error::Invariant`] for a negative size, a size type other than
/// `Amount`, `Value` or `Percent`, or a percentage reversal.
pub fn signals_to_size(
    position: f64,
    signals: Signals,
    size: f64,
    size_type: SizeType,
    accumulate: AccumulationMode,
    val_price: f64,
) -> Result<SizedOrder> {
    if !matches!(
        size_type,
        SizeType::Amount | SizeType::Value | SizeType::Percent
    ) {
        return Err(Error::Invariant(
            "from_signals supports only Amount, Value and Percent sizes".into(),
        ));
    }
    if is_less(size, 0.0) {
        return Err(Error::Invariant(
            "a negative size is not allowed: express direction with signals instead".into(),
        ));
    }

    let adds = matches!(
        accumulate,
        AccumulationMode::Both | AccumulationMode::AddOnly
    );
    let removes = matches!(
        accumulate,
        AccumulationMode::Both | AccumulationMode::RemoveOnly
    );

    let mut order_size = 0.0;
    let mut out_type = size_type;
    let mut direction = Direction::Both;

    // A reversal's extra leg, in the direction of travel. `sign` is +1 when
    // reversing from short to long.
    let reverse = |order_size: &mut f64, out_type: &mut SizeType, sign: f64| -> Result<()> {
        *order_size = sign * position.abs();
        if !size.is_nan() {
            if size_type == SizeType::Percent {
                return Err(Error::Invariant(
                    "a percentage size cannot reverse a position: the percentage \
                     is ambiguous between the position closed and the one opened"
                        .into(),
                ));
            }
            *order_size += sign
                * if size_type == SizeType::Value {
                    size / val_price
                } else {
                    size
                };
        }
        *out_type = SizeType::Amount;
        Ok(())
    };

    if position > 0.0 {
        if signals.short_entry {
            if removes {
                order_size = -size;
            } else {
                reverse(&mut order_size, &mut out_type, -1.0)?;
            }
        } else if signals.long_exit {
            direction = Direction::LongOnly;
            if removes {
                order_size = -size;
            } else {
                order_size = -position.abs();
                out_type = SizeType::Amount;
            }
        } else if signals.long_entry {
            direction = Direction::LongOnly;
            if adds {
                order_size = size;
            }
        }
    } else if position < 0.0 {
        if signals.long_entry {
            if removes {
                order_size = size;
            } else {
                reverse(&mut order_size, &mut out_type, 1.0)?;
            }
        } else if signals.short_exit {
            direction = Direction::ShortOnly;
            if removes {
                order_size = size;
            } else {
                order_size = position.abs();
                out_type = SizeType::Amount;
            }
        } else if signals.short_entry {
            direction = Direction::ShortOnly;
            if adds {
                order_size = -size;
            }
        }
    } else if signals.long_entry {
        order_size = size;
    } else if signals.short_entry {
        order_size = -size;
    }

    Ok(SizedOrder {
        size: order_size,
        size_type: out_type,
        direction,
    })
}

/// Whether a size type sets a target rather than an amount to trade.
#[must_use]
pub const fn is_target(size_type: SizeType) -> bool {
    matches!(
        size_type,
        SizeType::TargetAmount
            | SizeType::TargetValue
            | SizeType::TargetPercent
            | SizeType::TargetPercent100
    )
}

/// [`signals_to_size`] for the target size types, which
/// [`signals_to_size`] rejects.
///
/// An entry moves the position to its target — `size` for a long, `-size`
/// for a short — from flat, from the other side (a reversal is one order to
/// the opposite target), or, where `accumulate` lets an entry add, from the
/// same side. An exit closes the position: a target says where to be, not
/// how much of it to leave. A target of zero asks for nothing. The orders
/// are on [`Direction::Both`], so the kernel's target arithmetic is signed.
///
/// # Errors
///
/// Returns [`Error::Invariant`] for a negative size.
#[inline(never)]
pub fn signals_to_target(
    position: f64,
    signals: Signals,
    size: f64,
    size_type: SizeType,
    accumulate: AccumulationMode,
) -> Result<SizedOrder> {
    if is_less(size, 0.0) {
        return Err(Error::Invariant(
            "a negative size is not allowed: express direction with signals instead".into(),
        ));
    }
    let adds = matches!(
        accumulate,
        AccumulationMode::Both | AccumulationMode::AddOnly
    );
    let target = |size: f64| SizedOrder {
        size,
        size_type,
        direction: Direction::Both,
    };
    let none = SizedOrder {
        size: 0.0,
        size_type,
        direction: Direction::Both,
    };
    Ok(if position > 0.0 {
        if signals.short_entry {
            target(-size)
        } else if signals.long_exit {
            SizedOrder {
                size: -position,
                size_type: SizeType::Amount,
                direction: Direction::LongOnly,
            }
        } else if signals.long_entry && adds {
            target(size)
        } else {
            none
        }
    } else if position < 0.0 {
        if signals.long_entry {
            target(size)
        } else if signals.short_exit {
            SizedOrder {
                size: -position,
                size_type: SizeType::Amount,
                direction: Direction::ShortOnly,
            }
        } else if signals.short_entry && adds {
            target(-size)
        } else {
            none
        }
    } else if signals.long_entry {
        target(size)
    } else if signals.short_entry {
        target(-size)
    } else {
        none
    })
}

/// Sizes a bar's resolved signals: [`signals_to_target`] for a target size
/// type, [`signals_to_size`] otherwise.
///
/// The percent spellings (`*100`) are sized as the fractions they write and
/// keep their own type, which the kernel divides. `ValuePercent` is sized as
/// `Percent` is — a share of something the signal path does not know, so it
/// cannot reverse a position either.
///
/// # Errors
///
/// As the one it calls.
#[inline(always)]
pub fn size_signals(
    position: f64,
    signals: Signals,
    size: f64,
    size_type: SizeType,
    accumulate: AccumulationMode,
    val_price: f64,
) -> Result<SizedOrder> {
    if is_target(size_type) {
        signals_to_target(position, signals, size, size_type, accumulate)
    } else if matches!(
        size_type,
        SizeType::Amount | SizeType::Value | SizeType::Percent
    ) {
        signals_to_size(position, signals, size, size_type, accumulate, val_price)
    } else {
        size_percent_spelling(position, signals, size, size_type, accumulate)
    }
}

/// [`size_signals`] for `ValuePercent` and the `*100` spellings of the
/// non-target types: sized as `Percent`, then given back their own type.
#[inline(never)]
fn size_percent_spelling(
    position: f64,
    signals: Signals,
    size: f64,
    size_type: SizeType,
    accumulate: AccumulationMode,
) -> Result<SizedOrder> {
    let mut sized = signals_to_size(
        position,
        signals,
        size,
        SizeType::Percent,
        accumulate,
        f64::NAN,
    )?;
    if sized.size_type == SizeType::Percent {
        sized.size_type = size_type;
    }
    Ok(sized)
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "these assert on exact requested sizes, which is the contract"
)]
mod tests {
    use super::*;

    fn entry_only() -> Signals {
        Signals {
            long_entry: true,
            ..Signals::default()
        }
    }

    #[test]
    fn a_pending_conflict_follows_the_mode_of_its_side() {
        use PendingConflictMode::{CancelExecute, CancelIgnore, KeepExecute, KeepIgnore};
        let modes = [KeepIgnore, KeepExecute, CancelIgnore, CancelExecute];
        let outcome = |mode| match mode {
            KeepIgnore => (true, false),
            KeepExecute => (true, true),
            CancelIgnore => (false, false),
            CancelExecute => (false, true),
        };
        for pending_buys in [false, true] {
            for signal_buys in [false, true] {
                for adjacent in modes {
                    for opposite in modes {
                        let expected = if pending_buys == signal_buys {
                            outcome(adjacent)
                        } else {
                            outcome(opposite)
                        };
                        assert_eq!(
                            resolve_pending_conflict(pending_buys, signal_buys, adjacent, opposite),
                            expected
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_target_entry_goes_to_its_target_and_an_exit_closes() {
        let t = SizeType::TargetPercent;
        let sig = |le, lx, se, sx| Signals {
            long_entry: le,
            long_exit: lx,
            short_entry: se,
            short_exit: sx,
        };
        let off = AccumulationMode::Disabled;
        let at = |position, s, acc| signals_to_target(position, s, 0.5, t, acc).unwrap();
        // From flat, to either side.
        let o = at(0.0, sig(true, false, false, false), off);
        assert_eq!(
            (o.size, o.size_type, o.direction),
            (0.5, t, Direction::Both)
        );
        assert_eq!(at(0.0, sig(false, false, true, false), off).size, -0.5);
        // A reversal is one order to the other target.
        assert_eq!(at(3.0, sig(false, false, true, false), off).size, -0.5);
        assert_eq!(at(-3.0, sig(true, false, false, false), off).size, 0.5);
        // An exit closes, whatever the target.
        let o = at(3.0, sig(false, true, false, false), off);
        assert_eq!(
            (o.size, o.size_type, o.direction),
            (-3.0, SizeType::Amount, Direction::LongOnly)
        );
        let o = at(-3.0, sig(false, false, false, true), off);
        assert_eq!((o.size, o.direction), (3.0, Direction::ShortOnly));
        // An entry on the side held moves to the target only if it may add.
        assert_eq!(at(3.0, sig(true, false, false, false), off).size, 0.0);
        assert_eq!(
            at(3.0, sig(true, false, false, false), AccumulationMode::Both).size,
            0.5
        );
        assert!(signals_to_target(0.0, sig(true, false, false, false), -1.0, t, off).is_err());
        // The percent spelling of a target is a target.
        assert_eq!(
            size_signals(
                0.0,
                sig(true, false, false, false),
                50.0,
                SizeType::TargetPercent100,
                off,
                1.0
            )
            .unwrap()
            .size_type,
            SizeType::TargetPercent100
        );
        let v = size_signals(
            0.0,
            sig(true, false, false, false),
            0.5,
            SizeType::ValuePercent,
            off,
            1.0,
        )
        .unwrap();
        assert_eq!((v.size, v.size_type), (0.5, SizeType::ValuePercent));
        // An exit is an amount, whatever the spelling.
        let x = size_signals(
            4.0,
            sig(false, true, false, false),
            50.0,
            SizeType::Percent100,
            off,
            1.0,
        )
        .unwrap();
        assert_eq!((x.size, x.size_type), (-4.0, SizeType::Amount));
        // The plain function still refuses targets.
        assert!(signals_to_size(0.0, sig(true, false, false, false), 1.0, t, off, 1.0).is_err());
        assert_eq!(
            size_signals(0.0, sig(true, false, false, false), 1.0, t, off, 1.0)
                .unwrap()
                .size,
            1.0
        );
    }

    #[test]
    fn direction_mode_reads_entry_as_short_in_a_short_column() {
        let s = from_direction_mode(true, false, Direction::ShortOnly);
        assert!(s.short_entry && !s.long_entry);
    }

    #[test]
    fn direction_mode_both_maps_exit_to_a_short_entry() {
        // In a two-sided column an "exit" is not a close but a reversal
        // request: the signal pair means long or short, not in or out.
        let s = from_direction_mode(true, true, Direction::Both);
        assert!(s.long_entry && s.short_entry);
        assert!(!s.long_exit && !s.short_exit);
    }

    #[test]
    fn a_conflict_with_no_position_cannot_be_resolved_by_adjacency() {
        assert_eq!(
            resolve_signal_conflict(0.0, true, true, Direction::Both, ConflictMode::Adjacent),
            (false, false)
        );
        assert_eq!(
            resolve_signal_conflict(0.0, true, true, Direction::Both, ConflictMode::Opposite),
            (false, false)
        );
    }

    #[test]
    fn adjacency_follows_the_side_held() {
        // Long: the entry is adjacent. Short: the exit is.
        assert_eq!(
            resolve_signal_conflict(5.0, true, true, Direction::Both, ConflictMode::Adjacent),
            (true, false)
        );
        assert_eq!(
            resolve_signal_conflict(-5.0, true, true, Direction::Both, ConflictMode::Adjacent),
            (false, true)
        );
    }

    #[test]
    fn a_single_direction_column_always_picks_the_entry_as_adjacent() {
        assert_eq!(
            resolve_signal_conflict(5.0, true, true, Direction::LongOnly, ConflictMode::Adjacent),
            (true, false)
        );
    }

    #[test]
    fn a_direction_conflict_with_no_position_drops_both() {
        assert_eq!(
            resolve_dir_conflict(0.0, true, true, DirectionConflictMode::Adjacent),
            (false, false)
        );
    }

    #[test]
    fn a_non_conflicting_pair_passes_through_untouched() {
        assert_eq!(
            resolve_signal_conflict(0.0, true, false, Direction::Both, ConflictMode::Ignore),
            (true, false)
        );
        assert_eq!(
            resolve_dir_conflict(0.0, false, true, DirectionConflictMode::Ignore),
            (false, true)
        );
    }

    #[test]
    fn an_opposite_entry_can_close_instead_of_reversing() {
        let s = Signals {
            short_entry: true,
            ..Signals::default()
        };
        let (out, acc) =
            resolve_opposite_entry(5.0, s, OppositeEntryMode::Close, AccumulationMode::Both);
        assert!(!out.short_entry && out.long_exit);
        assert_eq!(acc, AccumulationMode::Disabled);
    }

    #[test]
    fn close_reduce_keeps_accumulation_on() {
        let s = Signals {
            short_entry: true,
            ..Signals::default()
        };
        let (out, acc) = resolve_opposite_entry(
            5.0,
            s,
            OppositeEntryMode::CloseReduce,
            AccumulationMode::Both,
        );
        assert!(out.long_exit);
        assert_eq!(
            acc,
            AccumulationMode::Both,
            "the only difference from Close"
        );
    }

    #[test]
    fn a_stop_reads_its_direction_from_the_position() {
        let (long, _) = from_stop(5.0, StopExitMode::Close, AccumulationMode::Both);
        assert!(long.long_exit);
        let (short, _) = from_stop(-5.0, StopExitMode::Close, AccumulationMode::Both);
        assert!(short.short_exit);
        // Flat: a stop cannot fire, so nothing is signalled.
        let (flat, _) = from_stop(0.0, StopExitMode::Close, AccumulationMode::Both);
        assert!(!flat.any());
    }

    #[test]
    fn an_entry_from_flat_opens_the_requested_size() {
        let o = signals_to_size(
            0.0,
            entry_only(),
            3.0,
            SizeType::Amount,
            AccumulationMode::Disabled,
            10.0,
        )
        .unwrap();
        assert_eq!(o.size, 3.0);
        assert_eq!(o.direction, Direction::Both);
    }

    #[test]
    fn an_exit_without_accumulation_closes_the_whole_position() {
        let signals = Signals {
            long_exit: true,
            ..Signals::default()
        };
        let o = signals_to_size(
            7.0,
            signals,
            3.0,
            SizeType::Amount,
            AccumulationMode::Disabled,
            10.0,
        )
        .unwrap();
        // The requested 3.0 is ignored: without accumulation an exit is a close.
        assert_eq!(o.size, -7.0);
        assert_eq!(o.direction, Direction::LongOnly);
    }

    #[test]
    fn an_exit_with_accumulation_reduces_by_the_requested_size() {
        let signals = Signals {
            long_exit: true,
            ..Signals::default()
        };
        let o = signals_to_size(
            7.0,
            signals,
            3.0,
            SizeType::Amount,
            AccumulationMode::Both,
            10.0,
        )
        .unwrap();
        assert_eq!(o.size, -3.0);
    }

    #[test]
    fn add_only_ignores_an_exit_and_remove_only_ignores_an_entry() {
        let exit = Signals {
            long_exit: true,
            ..Signals::default()
        };
        // AddOnly does not remove, so the exit closes outright.
        let o = signals_to_size(
            7.0,
            exit,
            3.0,
            SizeType::Amount,
            AccumulationMode::AddOnly,
            10.0,
        )
        .unwrap();
        assert_eq!(o.size, -7.0);
        // RemoveOnly does not add, so an entry while long does nothing.
        let o = signals_to_size(
            7.0,
            entry_only(),
            3.0,
            SizeType::Amount,
            AccumulationMode::RemoveOnly,
            10.0,
        )
        .unwrap();
        assert_eq!(o.size, 0.0);
    }

    #[test]
    fn a_reversal_is_one_order_covering_both_legs() {
        let signals = Signals {
            short_entry: true,
            ..Signals::default()
        };
        let o = signals_to_size(
            7.0,
            signals,
            3.0,
            SizeType::Amount,
            AccumulationMode::Disabled,
            10.0,
        )
        .unwrap();
        // Close 7 long and open 3 short in one request.
        assert_eq!(o.size, -10.0);
        assert_eq!(o.direction, Direction::Both);
    }

    #[test]
    fn a_value_sized_reversal_converts_through_the_valuation_price() {
        let signals = Signals {
            short_entry: true,
            ..Signals::default()
        };
        let o = signals_to_size(
            7.0,
            signals,
            50.0,
            SizeType::Value,
            AccumulationMode::Disabled,
            10.0,
        )
        .unwrap();
        // 50 of value at 10 is 5 units, on top of closing 7.
        assert_eq!(o.size, -12.0);
        assert_eq!(o.size_type, SizeType::Amount);
    }

    #[test]
    fn a_nan_size_reverses_by_exactly_the_position() {
        let signals = Signals {
            short_entry: true,
            ..Signals::default()
        };
        let o = signals_to_size(
            7.0,
            signals,
            f64::NAN,
            SizeType::Amount,
            AccumulationMode::Disabled,
            10.0,
        )
        .unwrap();
        assert_eq!(o.size, -7.0);
    }

    #[test]
    fn a_percentage_cannot_reverse() {
        let signals = Signals {
            short_entry: true,
            ..Signals::default()
        };
        assert!(
            signals_to_size(
                7.0,
                signals,
                0.5,
                SizeType::Percent,
                AccumulationMode::Disabled,
                10.0,
            )
            .is_err()
        );
    }

    #[test]
    fn a_negative_size_is_rejected() {
        assert!(
            signals_to_size(
                0.0,
                entry_only(),
                -1.0,
                SizeType::Amount,
                AccumulationMode::Disabled,
                10.0,
            )
            .is_err()
        );
    }

    #[test]
    fn an_unsupported_size_type_is_rejected() {
        assert!(
            signals_to_size(
                0.0,
                entry_only(),
                1.0,
                SizeType::TargetPercent,
                AccumulationMode::Disabled,
                10.0,
            )
            .is_err()
        );
    }

    #[test]
    fn no_signal_means_no_order() {
        let o = signals_to_size(
            5.0,
            Signals::default(),
            3.0,
            SizeType::Amount,
            AccumulationMode::Both,
            10.0,
        )
        .unwrap();
        // Exactly zero, which the kernel then ignores. This is why a sparse
        // signal column costs so little: most bars stop here.
        assert_eq!(o.size, 0.0);
    }
}
