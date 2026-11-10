//! Golden test for the signal resolution pipeline.
//!
//! The fixture set is **exhaustive**, not sampled: the pipeline's inputs are a
//! position sign, a few booleans and some small enums, so the whole space fits
//! in five thousand rows. Sampling a space that small would be choosing to
//! test less for no reason — and the cases that matter are the ones a sample
//! misses, like `Adjacent` with a flat position (which drops both signals) or
//! `CloseReduce`, which differs from `Close` in exactly one field.
//!
//! The expected outputs are committed in `tests/fixtures/signals.tsv`, so this
//! needs no Python.

use std::collections::HashMap;
use std::path::PathBuf;

use alphax_core::math::is_close;
use alphax_model::{
    AccumulationMode, ConflictMode, Direction, DirectionConflictMode, OppositeEntryMode, SizeType,
    StopExitMode,
};
use alphax_sim::signals::{
    Signals, from_stop, resolve_dir_conflict, resolve_opposite_entry, resolve_signal_conflict,
    signals_to_size,
};

struct Fixture {
    header: HashMap<String, usize>,
    rows: Vec<Vec<String>>,
}

impl Fixture {
    fn load() -> Self {
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/signals.tsv");
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let mut lines = text
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty());
        let names: Vec<String> = lines
            .next()
            .expect("fixture has no header")
            .split('\t')
            .map(str::to_owned)
            .collect();
        let rows = lines
            .map(|line| {
                let fields: Vec<String> = line.split('\t').map(str::to_owned).collect();
                assert_eq!(fields.len(), names.len(), "row width mismatch");
                fields
            })
            .collect();
        let header = names.into_iter().enumerate().map(|(i, n)| (n, i)).collect();
        Self { header, rows }
    }

    fn get<'a>(&self, row: &'a [String], name: &str) -> &'a str {
        let idx = *self
            .header
            .get(name)
            .unwrap_or_else(|| panic!("fixture has no column {name:?}"));
        &row[idx]
    }
}

fn boolean(s: &str) -> bool {
    match s {
        "true" => true,
        "false" => false,
        other => panic!("cannot parse {other:?} as bool"),
    }
}

fn number(s: &str) -> f64 {
    match s {
        "inf" => f64::INFINITY,
        "-inf" => f64::NEG_INFINITY,
        "nan" => f64::NAN,
        other => other
            .parse()
            .unwrap_or_else(|_| panic!("cannot parse {other:?} as f64")),
    }
}

fn accumulation(s: &str) -> AccumulationMode {
    match s {
        "Disabled" => AccumulationMode::Disabled,
        "Both" => AccumulationMode::Both,
        "AddOnly" => AccumulationMode::AddOnly,
        "RemoveOnly" => AccumulationMode::RemoveOnly,
        other => panic!("unknown accumulation mode {other:?}"),
    }
}

fn direction(s: &str) -> Direction {
    match s {
        "LongOnly" => Direction::LongOnly,
        "ShortOnly" => Direction::ShortOnly,
        "Both" => Direction::Both,
        other => panic!("unknown direction {other:?}"),
    }
}

fn size_type(s: &str) -> SizeType {
    match s {
        "Amount" => SizeType::Amount,
        "Value" => SizeType::Value,
        "Percent" => SizeType::Percent,
        "TargetAmount" => SizeType::TargetAmount,
        "TargetValue" => SizeType::TargetValue,
        "TargetPercent" => SizeType::TargetPercent,
        other => panic!("unknown size type {other:?}"),
    }
}

/// Sentinel-aware agreement; see the note in `alphax-exec`'s harness.
fn agree(a: f64, b: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return a.is_nan() && b.is_nan();
    }
    if a.is_infinite() || b.is_infinite() {
        return a.is_infinite() && b.is_infinite() && a.is_sign_positive() == b.is_sign_positive();
    }
    is_close(a, b)
}

#[test]
fn matches_the_golden_pipeline() {
    let fixture = Fixture::load();
    assert!(
        fixture.rows.len() > 5000,
        "expected the exhaustive fixture set, got {}",
        fixture.rows.len()
    );

    let mut failures: Vec<String> = Vec::new();
    let mut seen: HashMap<&str, usize> = HashMap::new();

    for (n, row) in fixture.rows.iter().enumerate() {
        let g = |name: &str| fixture.get(row, name);
        let kind = g("fn");
        *seen.entry(kind).or_default() += 1;
        let position = number(g("position"));
        let mut fail = |msg: String| failures.push(format!("row {n} ({kind}): {msg}"));

        match kind {
            "signal_conflict" => {
                let (e, x) = resolve_signal_conflict(
                    position,
                    boolean(g("is_entry")),
                    boolean(g("is_exit")),
                    direction(g("direction")),
                    match g("mode") {
                        "Ignore" => ConflictMode::Ignore,
                        "Entry" => ConflictMode::Entry,
                        "Exit" => ConflictMode::Exit,
                        "Adjacent" => ConflictMode::Adjacent,
                        "Opposite" => ConflictMode::Opposite,
                        other => panic!("unknown conflict mode {other:?}"),
                    },
                );
                if e != boolean(g("out_is_entry")) || x != boolean(g("out_is_exit")) {
                    fail(format!(
                        "got ({e}, {x}), want ({}, {})",
                        g("out_is_entry"),
                        g("out_is_exit")
                    ));
                }
            }
            "dir_conflict" => {
                let (l, s) = resolve_dir_conflict(
                    position,
                    boolean(g("long_entry")),
                    boolean(g("short_entry")),
                    match g("mode") {
                        "Ignore" => DirectionConflictMode::Ignore,
                        "Long" => DirectionConflictMode::Long,
                        "Short" => DirectionConflictMode::Short,
                        "Adjacent" => DirectionConflictMode::Adjacent,
                        "Opposite" => DirectionConflictMode::Opposite,
                        other => panic!("unknown direction conflict mode {other:?}"),
                    },
                );
                if l != boolean(g("out_long_entry")) || s != boolean(g("out_short_entry")) {
                    fail(format!(
                        "got ({l}, {s}), want ({}, {})",
                        g("out_long_entry"),
                        g("out_short_entry")
                    ));
                }
            }
            "opposite_entry" | "stop" => {
                let (out, acc) = if kind == "stop" {
                    from_stop(
                        position,
                        match g("mode") {
                            "Close" => StopExitMode::Close,
                            "CloseReduce" => StopExitMode::CloseReduce,
                            "Reverse" => StopExitMode::Reverse,
                            "ReverseReduce" => StopExitMode::ReverseReduce,
                            other => panic!("unknown stop exit mode {other:?}"),
                        },
                        accumulation(g("accumulate")),
                    )
                } else {
                    resolve_opposite_entry(
                        position,
                        Signals {
                            long_entry: boolean(g("long_entry")),
                            long_exit: boolean(g("long_exit")),
                            short_entry: boolean(g("short_entry")),
                            short_exit: boolean(g("short_exit")),
                        },
                        match g("mode") {
                            "Ignore" => OppositeEntryMode::Ignore,
                            "Close" => OppositeEntryMode::Close,
                            "CloseReduce" => OppositeEntryMode::CloseReduce,
                            "Reverse" => OppositeEntryMode::Reverse,
                            "ReverseReduce" => OppositeEntryMode::ReverseReduce,
                            other => panic!("unknown opposite entry mode {other:?}"),
                        },
                        accumulation(g("accumulate")),
                    )
                };
                let want = Signals {
                    long_entry: boolean(g("out_long_entry")),
                    long_exit: boolean(g("out_long_exit")),
                    short_entry: boolean(g("out_short_entry")),
                    short_exit: boolean(g("out_short_exit")),
                };
                if out != want {
                    fail(format!("got {out:?}, want {want:?}"));
                }
                if acc != accumulation(g("out_accumulate")) {
                    fail(format!(
                        "accumulation: got {acc:?}, want {}",
                        g("out_accumulate")
                    ));
                }
            }
            "to_size" => {
                let result = signals_to_size(
                    position,
                    Signals {
                        long_entry: boolean(g("long_entry")),
                        long_exit: boolean(g("long_exit")),
                        short_entry: boolean(g("short_entry")),
                        short_exit: boolean(g("short_exit")),
                    },
                    number(g("size")),
                    size_type(g("size_type")),
                    accumulation(g("accumulate")),
                    number(g("val_price")),
                );
                match (g("outcome"), result) {
                    ("error", Ok(o)) => fail(format!("expected a rejection, got {o:?}")),
                    ("error", Err(_)) => {}
                    ("ok", Err(e)) => fail(format!("unexpected rejection: {e}")),
                    ("ok", Ok(o)) => {
                        if !agree(o.size, number(g("out_size"))) {
                            fail(format!("size: got {}, want {}", o.size, g("out_size")));
                        }
                        if o.size_type != size_type(g("out_size_type")) {
                            fail(format!(
                                "size type: got {:?}, want {}",
                                o.size_type,
                                g("out_size_type")
                            ));
                        }
                        if o.direction != direction(g("out_direction")) {
                            fail(format!(
                                "direction: got {:?}, want {}",
                                o.direction,
                                g("out_direction")
                            ));
                        }
                    }
                    (other, _) => panic!("unknown outcome {other:?}"),
                }
            }
            other => panic!("unknown fixture function {other:?}"),
        }
    }

    for name in [
        "signal_conflict",
        "dir_conflict",
        "opposite_entry",
        "stop",
        "to_size",
    ] {
        assert!(seen.contains_key(name), "no fixtures for {name}");
    }
    assert!(
        failures.is_empty(),
        "{} of {} rows disagree with the golden outputs:\n{}",
        failures.len(),
        fixture.rows.len(),
        failures
            .iter()
            .take(40)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// The fixture set must keep covering the rejections.
///
/// Two of them — a negative size and a percentage reversal — exist because
/// the pipeline can express something an order cannot. A fixture set that
/// drifted to only valid inputs would still pass the test above while leaving
/// both unverified.
#[test]
fn the_fixtures_still_cover_the_rejections() {
    let fixture = Fixture::load();
    let rejections = fixture
        .rows
        .iter()
        .filter(|r| fixture.get(r, "outcome") == "error")
        .count();
    assert!(
        rejections > 100,
        "only {rejections} rejection cases in the set"
    );

    // And specifically the percentage reversal, which is the subtle one: it
    // is rejected only when a position is actually being reversed.
    let percent_reversal = fixture.rows.iter().any(|r| {
        fixture.get(r, "outcome") == "error"
            && fixture.get(r, "size_type") == "Percent"
            && fixture.get(r, "position") != "0.0"
    });
    assert!(percent_reversal, "no percentage-reversal rejection covered");
}
