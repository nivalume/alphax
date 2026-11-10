//! Qlib parity beyond Alpha158 and Alpha360: configurable Alpha158, the
//! high-frequency operators of `qlib.contrib.ops.high_freq`, `ChangeInstrument`
//! (and Qlib's `Mask(e, "X")`), and point-in-time `P` / `PRef`.
//!
//! The fixtures come from `tests/generators/generate_qlib_extensions.py`,
//! which runs Qlib's own operators (and, for `P`, Qlib's own point-in-time
//! provider over real binary files). Each expression column's header is the
//! Qlib source, compiled here unchanged.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use alphax_expr::{
    Alpha158Config, Alpha158Price, Alpha158Rolling, FactorInputs, FactorSet, FactorValues,
    PitRecord, alpha158_features,
};

fn read(file_name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(file_name);
    fs::read_to_string(path).unwrap()
}

/// A TSV fixture: its header and rows, comments dropped.
fn table(file_name: &str) -> (Vec<String>, Vec<Vec<String>>) {
    let text = read(file_name);
    let mut lines = text.lines().filter(|line| !line.starts_with('#'));
    let header = lines
        .next()
        .unwrap()
        .split('\t')
        .map(str::to_owned)
        .collect();
    let rows = lines
        .map(|line| line.split('\t').map(str::to_owned).collect())
        .collect();
    (header, rows)
}

/// Days since 1970-01-01 of a proleptic Gregorian date (Howard Hinnant's
/// `days_from_civil`).
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let yoe = year - era * 400;
    let doy = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// `YYYY-MM-DD` or `YYYY-MM-DDTHH:MM:SS` as wall-clock nanoseconds.
fn nanos(text: &str) -> i64 {
    let number = |range: std::ops::Range<usize>| text[range].parse::<i64>().unwrap();
    let day = days_from_civil(number(0..4), number(5..7), number(8..10));
    let seconds = if text.len() > 10 {
        number(11..13) * 3600 + number(14..16) * 60 + number(17..19)
    } else {
        0
    };
    (day * 86_400 + seconds) * 1_000_000_000
}

fn compare(actual: &FactorValues, expected: &HashMap<String, Vec<f64>>, epsilon: f64) {
    let mut mismatches = Vec::new();
    for (name, matrix) in actual.iter() {
        for (index, (&got, &want)) in matrix.as_slice().iter().zip(&expected[name]).enumerate() {
            let same = if want.is_nan() || want.is_infinite() {
                got.to_bits() == want.to_bits() || (want.is_nan() && got.is_nan())
            } else {
                (got - want).abs() <= epsilon * want.abs().max(1.0)
            };
            if !same {
                mismatches.push(format!("{name}[{index}]: qlib={want:?}, alphax={got:?}"));
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} mismatches:\n{}",
        mismatches.len(),
        mismatches[..mismatches.len().min(40)].join("\n")
    );
}

fn bits(values: &FactorValues) -> Vec<(String, Vec<u64>)> {
    values
        .iter()
        .map(|(name, matrix)| {
            (
                name.to_owned(),
                matrix.as_slice().iter().map(|v| v.to_bits()).collect(),
            )
        })
        .collect()
}

/// Evaluates on 1 and 4 threads and whole versus one symbol at a time, and
/// insists all four agree bit for bit.
fn evaluate_everywhere(set: &FactorSet, inputs: &FactorInputs<'_>) -> FactorValues {
    let pool = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
    };
    let one = pool(1).install(|| set.evaluate(inputs)).unwrap();
    let four = pool(4).install(|| set.evaluate(inputs)).unwrap();
    assert_eq!(bits(&one), bits(&four), "thread count changed a value");
    let (n_rows, _) = inputs.shape();
    let mut batched: HashMap<String, Vec<f64>> = HashMap::new();
    set.evaluate_batches(inputs, 1, |_, values| {
        for (name, matrix) in values.iter() {
            batched
                .entry(name.to_owned())
                .or_default()
                .extend_from_slice(matrix.as_slice());
        }
        Ok(())
    })
    .unwrap();
    for (name, matrix) in one.iter() {
        let whole: Vec<u64> = matrix.as_slice().iter().map(|v| v.to_bits()).collect();
        let parts: Vec<u64> = batched[name].iter().map(|v| v.to_bits()).collect();
        assert_eq!(
            whole, parts,
            "{name}: a batch of one symbol differs from the panel"
        );
        assert_eq!(parts.len() % n_rows, 0);
    }
    one
}

#[test]
fn alpha158_configs_match_qlib() {
    let (header, rows) = table("qlib_alpha158_configs.tsv");
    assert_eq!(header, ["config", "position", "name", "expression"]);
    let strings = |items: &[&str]| items.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
    let configs: [(&str, Alpha158Config); 6] = [
        ("default", Alpha158Config::default()),
        (
            "full",
            Alpha158Config {
                kbar: true,
                price: Some(Alpha158Price::default()),
                volume: Some(vec![0, 1, 2, 3, 4]),
                rolling: Some(Alpha158Rolling::default()),
            },
        ),
        (
            "implicit",
            Alpha158Config {
                kbar: false,
                price: Some(Alpha158Price::default()),
                volume: Some((0..5).collect()),
                rolling: None,
            },
        ),
        (
            "subset",
            Alpha158Config {
                kbar: false,
                price: None,
                volume: None,
                rolling: Some(Alpha158Rolling {
                    windows: vec![3, 7],
                    include: Some(strings(&["ROC", "MA", "LOW", "CORD", "VSUMD"])),
                    exclude: strings(&["MA"]),
                }),
            },
        ),
        (
            "exclude",
            Alpha158Config {
                kbar: true,
                price: None,
                volume: None,
                rolling: Some(Alpha158Rolling {
                    exclude: strings(&["RANK", "CORR", "CORD"]),
                    ..Alpha158Rolling::default()
                }),
            },
        ),
        (
            "price_only",
            Alpha158Config {
                kbar: false,
                price: Some(Alpha158Price {
                    windows: vec![2, 0],
                    features: strings(&["close", "Vwap"]),
                }),
                volume: None,
                rolling: None,
            },
        ),
    ];
    for (id, config) in configs {
        let expected: Vec<(String, String)> = rows
            .iter()
            .filter(|row| row[0] == id)
            .map(|row| (row[2].clone(), row[3].clone()))
            .collect();
        assert!(!expected.is_empty(), "the fixture has config {id}");
        let actual = alpha158_features(&config).unwrap();
        assert_eq!(actual, expected, "config {id}");
        FactorSet::compile(actual).unwrap();
    }
}

#[test]
fn high_frequency_and_instrument_operators_match_qlib() {
    let (header, rows) = table("qlib_highfreq.tsv");
    let inputs_at = |name: &str| header.iter().position(|h| h == name).unwrap();
    let symbols: Vec<String> = {
        let mut symbols: Vec<String> = rows.iter().map(|row| row[0].clone()).collect();
        symbols.dedup();
        symbols
    };
    let n_rows = rows.len() / symbols.len();
    let column = |name: &str| -> Vec<f64> {
        let at = inputs_at(name);
        rows.iter().map(|row| row[at].parse().unwrap()).collect()
    };
    let fields: Vec<(&str, Vec<f64>)> = ["close", "volume", "paused_num", "x"]
        .into_iter()
        .map(|name| (name, column(name)))
        .collect();
    let timestamps: Vec<i64> = rows[..n_rows].iter().map(|row| nanos(&row[2])).collect();
    let expressions = &header[7..];
    assert_eq!(expressions.len(), 24);
    let expected: HashMap<String, Vec<f64>> = expressions
        .iter()
        .map(|expression| (expression.clone(), column(expression)))
        .collect();

    let mut inputs = FactorInputs::new(n_rows, symbols.len())
        .unwrap()
        .with_timestamps(&timestamps)
        .unwrap()
        .with_symbols(&symbols)
        .unwrap();
    for (name, values) in &fields {
        inputs = inputs.with_field(name, values).unwrap();
    }
    let set = FactorSet::compile(expressions.iter().map(|e| (e.clone(), e.clone()))).unwrap();
    assert_eq!(set.instrument_names(), ["BBB"]);
    // `Corr` over rolling incremental sums in pandas, recomputed here.
    let (corr, rest): (Vec<&String>, Vec<&String>) =
        expressions.iter().partition(|e| e.starts_with("Corr"));
    let values = evaluate_everywhere(&set, &inputs);
    let pick = |names: &[&String]| {
        names
            .iter()
            .map(|name| ((*name).clone(), expected[*name].clone()))
            .collect::<HashMap<_, _>>()
    };
    let only = |names: &[&String]| {
        FactorSet::compile(names.iter().map(|e| ((*e).clone(), (*e).clone()))).unwrap()
    };
    compare(
        &evaluate_everywhere(&only(&rest), &inputs),
        &pick(&rest),
        1e-9,
    );
    compare(
        &evaluate_everywhere(&only(&corr), &inputs),
        &pick(&corr),
        1e-6,
    );
    // Lifted into the whole set, every factor computes the same values.
    for (name, matrix) in values.iter() {
        let alone = FactorSet::compile([(name, name)])
            .unwrap()
            .evaluate(&inputs)
            .unwrap();
        let alone = alone.iter().next().unwrap().1;
        assert_eq!(
            matrix
                .as_slice()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            alone
                .as_slice()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            "{name}"
        );
    }
}

#[test]
fn an_instrument_outside_the_panel_reads_like_a_symbol() {
    let (header, rows) = table("qlib_highfreq.tsv");
    let n_rows = rows.len() / 2;
    let close: Vec<f64> = {
        let at = header.iter().position(|h| h == "close").unwrap();
        rows.iter().map(|row| row[at].parse().unwrap()).collect()
    };
    let timestamps: Vec<i64> = rows[..n_rows].iter().map(|row| nanos(&row[2])).collect();
    let set = FactorSet::compile([
        ("change", "$close/ChangeInstrument('BBB', $close)"),
        ("last", "ChangeInstrument('BBB', DayLast($close))"),
    ])
    .unwrap();
    let in_panel = FactorInputs::new(n_rows, 2)
        .unwrap()
        .with_timestamps(&timestamps)
        .unwrap()
        .with_symbols(&["AAA", "BBB"])
        .unwrap()
        .with_field("close", &close)
        .unwrap();
    let bbb = FactorInputs::new(n_rows, 1)
        .unwrap()
        .with_field("close", &close[n_rows..])
        .unwrap();
    let outside = FactorInputs::new(n_rows, 1)
        .unwrap()
        .with_timestamps(&timestamps)
        .unwrap()
        .with_field("close", &close[..n_rows])
        .unwrap()
        .with_instrument("BBB", bbb)
        .unwrap();
    let whole = set.evaluate(&in_panel).unwrap();
    let alone = set.evaluate(&outside).unwrap();
    for name in ["change", "last"] {
        let a = &whole.get(name).unwrap().as_slice()[..n_rows];
        let b = alone.get(name).unwrap().as_slice();
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{name}"
        );
    }
    let error = set
        .evaluate(
            &FactorInputs::new(n_rows, 1)
                .unwrap()
                .with_field("close", &close[..n_rows])
                .unwrap(),
        )
        .unwrap_err()
        .to_string();
    assert!(error.contains("BBB"), "{error}");
}

#[test]
fn point_in_time_operators_match_qlib() {
    let (_, records) = table("qlib_pit_records.tsv");
    let (header, rows) = table("qlib_pit.tsv");
    let mut symbols: Vec<String> = rows.iter().map(|row| row[0].clone()).collect();
    symbols.dedup();
    let n_rows = rows.len() / symbols.len();
    let timestamps: Vec<i64> = rows[..n_rows].iter().map(|row| nanos(&row[2])).collect();
    let mut by_field: Vec<(String, Vec<PitRecord>)> = Vec::new();
    for record in &records {
        let entry = match by_field.iter_mut().find(|(field, _)| *field == record[1]) {
            Some(entry) => entry,
            None => {
                by_field.push((record[1].clone(), Vec::new()));
                by_field.last_mut().unwrap()
            }
        };
        entry.1.push(PitRecord {
            col: symbols.iter().position(|s| *s == record[0]).unwrap(),
            day: nanos(&record[2]) / 86_400_000_000_000,
            period: record[3].parse().unwrap(),
            value: record[4].parse().unwrap(),
        });
    }
    let mut inputs = FactorInputs::new(n_rows, symbols.len())
        .unwrap()
        .with_timestamps(&timestamps)
        .unwrap();
    for (field, records) in by_field {
        inputs = inputs.with_pit(&field, records).unwrap();
    }
    let expressions = &header[3..];
    assert_eq!(expressions.len(), 17);
    let expected: HashMap<String, Vec<f64>> = expressions
        .iter()
        .enumerate()
        .map(|(k, expression)| {
            let values = rows.iter().map(|row| row[3 + k].parse().unwrap()).collect();
            (expression.clone(), values)
        })
        .collect();
    let set = FactorSet::compile(expressions.iter().map(|e| (e.clone(), e.clone()))).unwrap();
    assert!(!set.uses_future());
    assert!(set.required_fields().contains("$roe_q"));
    // Qlib's P is float32.
    compare(&evaluate_everywhere(&set, &inputs), &expected, 1e-6);
    let delta = &expected["P(Delta($$roe_q, 1))"];
    assert!(
        delta.iter().all(|v| v.is_nan()),
        "the fixture pins Qlib's Delta quirk"
    );
}

#[test]
fn point_in_time_needs_timestamps_and_its_records() {
    let set = FactorSet::compile([("roe", "P($$roe_q)")]).unwrap();
    let bare = FactorInputs::new(3, 1).unwrap();
    assert!(
        set.evaluate(&bare)
            .unwrap_err()
            .to_string()
            .contains("timestamps")
    );
    let dated = bare
        .with_timestamps(&[0, 86_400_000_000_000, 2 * 86_400_000_000_000])
        .unwrap();
    assert!(
        set.evaluate(&dated)
            .unwrap_err()
            .to_string()
            .contains("$$roe_q")
    );
    let record = |day, value| PitRecord {
        col: 0,
        day,
        period: 201_904,
        value,
    };
    let known = dated
        .with_pit("roe_q", vec![record(1, 0.5), record(1, 0.7)])
        .unwrap();
    let values = set.evaluate(&known).unwrap();
    let column = values.get("roe").unwrap().as_slice();
    assert!(column[0].is_nan());
    assert_eq!(column[1..], [0.7, 0.7], "the later record of a day wins");
    let bad = FactorInputs::new(3, 1)
        .unwrap()
        .with_pit("roe_q", vec![record(0, 1.0)]);
    assert!(bad.is_ok());
    let bad_period = FactorInputs::new(3, 1).unwrap().with_pit(
        "roe_q",
        vec![PitRecord {
            period: 2019,
            ..record(0, 1.0)
        }],
    );
    assert!(bad_period.unwrap_err().to_string().contains("2019"));
    let bad_name = FactorInputs::new(3, 1).unwrap().with_pit("roe", Vec::new());
    assert!(bad_name.unwrap_err().to_string().contains("_q"));
}

#[test]
fn high_frequency_operators_need_timestamps_and_flag_the_future() {
    let close = [1.0, f64::NAN, 3.0];
    let inputs = FactorInputs::new(3, 1)
        .unwrap()
        .with_field("close", &close)
        .unwrap();
    let set = FactorSet::compile([("last", "DayLast($close)")]).unwrap();
    assert!(
        set.evaluate(&inputs)
            .unwrap_err()
            .to_string()
            .contains("timestamps")
    );
    for (source, future) in [
        ("DayLast($close)", true),
        ("BFillNan($close)", true),
        ("Cut($close, 1, -1)", true),
        ("Cut($close, 1, None)", false),
        ("FFillNan($close)", false),
        ("DayCumsum($close, '9:30', '14:59')", false),
        ("ChangeInstrument('X', Ref($close, -1))", true),
    ] {
        let set = FactorSet::compile([("f", source)]).unwrap();
        assert_eq!(set.uses_future(), future, "{source}");
    }
    for (source, message) in [
        ("Cut($close, -1, None)", "positive"),
        ("DayCumsum($close, '25:00', '14:59')", "time of day"),
        ("DayCumsum($close, 930)", "string"),
        ("$close + 'x'", "string"),
        ("ChangeInstrument($close, $close)", "quoted"),
    ] {
        let error = FactorSet::compile([("f", source)]).unwrap_err().to_string();
        assert!(error.contains(message), "{source}: {error}");
    }
    let forward = FactorSet::compile([("f", "FFillNan($close)"), ("b", "BFillNan($close)")])
        .unwrap()
        .evaluate(&inputs)
        .unwrap();
    assert_eq!(forward.get("f").unwrap().as_slice(), [1.0, 1.0, 3.0]);
    assert_eq!(forward.get("b").unwrap().as_slice(), [1.0, 3.0, 3.0]);
}

#[test]
fn python_dependent_semantics_match_qlib() {
    let (header, rows) = table("qlib_semantics.tsv");
    let mut symbols: Vec<String> = rows.iter().map(|row| row[0].clone()).collect();
    symbols.dedup();
    let n_rows = rows.len() / symbols.len();
    let column = |name: &str| -> Vec<f64> {
        let at = header.iter().position(|h| h == name).unwrap();
        rows.iter().map(|row| row[at].parse().unwrap()).collect()
    };
    let inputs_named = ["close", "volume", "收盘价", "成交量（手）"];
    let fields: Vec<(&str, Vec<f64>)> = inputs_named.iter().map(|&n| (n, column(n))).collect();
    let mut inputs = FactorInputs::new(n_rows, symbols.len()).unwrap();
    for (name, values) in &fields {
        inputs = inputs.with_field(name, values).unwrap();
    }
    let expressions = &header[7..];
    assert_eq!(expressions.len(), 19);
    let expected: HashMap<String, Vec<f64>> =
        expressions.iter().map(|e| (e.clone(), column(e))).collect();
    let set = FactorSet::compile(expressions.iter().map(|e| (e.clone(), e.clone()))).unwrap();
    let (corr, rest): (Vec<&String>, Vec<&String>) =
        expressions.iter().partition(|e| e.starts_with("Corr"));
    let pick = |names: &[&String]| {
        names
            .iter()
            .map(|name| ((*name).clone(), expected[*name].clone()))
            .collect::<HashMap<_, _>>()
    };
    let only = |names: &[&String]| {
        FactorSet::compile(names.iter().map(|e| ((*e).clone(), (*e).clone()))).unwrap()
    };
    compare(
        &evaluate_everywhere(&only(&rest), &inputs),
        &pick(&rest),
        1e-9,
    );
    compare(
        &evaluate_everywhere(&only(&corr), &inputs),
        &pick(&corr),
        1e-6,
    );
    #[cfg(feature = "polars")]
    {
        let polars = set
            .evaluate_with(&inputs, &alphax_expr::PolarsBackend)
            .unwrap();
        compare(&polars, &expected, 1e-6);
    }
    let _ = set;
}
