//! Benchmark runner for `from_orders` and `from_signals`.
//!
//! Reads `benchmarks/workloads.tsv`, runs each workload, and writes results in
//! the format `benchmarks/compare.py` expects. Every row reports a checksum of
//! the arrays it generated and the fill count it produced, so that serial and
//! threaded runs doing the same work can be checked rather than assumed.
//!
//! `benchmarks/workloads_signals.tsv` drives `from_signals` the same way. Those
//! rows exist for the regression gate, because `from_signals` is the loop the
//! Python layer calls most.
//!
//! ```text
//! cargo run --release --example bench -p alphax-sim -- benchmarks/results/alphax.tsv
//! cargo run --release --example bench -p alphax-sim -- --once signals_dense
//! ```
//!
//! `--once NAME` runs one workload once, single-threaded and untimed, and
//! prints its fill count and checksum. It is what `benchmarks/instructions.py`
//! runs under callgrind.
//!
//! Release only. A debug build of the execution kernel is roughly fifty times
//! slower and the numbers mean nothing.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use alphax_core::flex::FlexArray;
use alphax_model::{AccountType, Direction, OrderType, SizeType};
use alphax_sim::groups::Groups;
use alphax_sim::{
    FromOrders, FromSignals, simulate_from_orders, simulate_from_orders_parallel,
    simulate_from_signals, simulate_from_signals_parallel,
};

/// One row of `workloads.tsv`.
struct Workload {
    name: String,
    n_rows: usize,
    n_cols: usize,
    fill_every: usize,
    size_type: SizeType,
    direction: Direction,
    price_mode: String,
    fees: f64,
    fixed_fees: f64,
    slippage: f64,
}

/// Generates the input arrays.
///
/// All the arithmetic is IEEE double and the integer term stays well inside
/// 64 bits, so the inputs are identical on every platform; the checksum in the
/// output is what proves it on any given run.
fn generate(w: &Workload) -> (Vec<f64>, Vec<f64>) {
    let n = w.n_rows * w.n_cols;
    let mut close = Vec::with_capacity(n);
    let mut size = Vec::with_capacity(n);
    for col in 0..w.n_cols {
        let mut price = 10.0 + (col % 37) as f64;
        for i in 0..w.n_rows {
            let wiggle = ((i * 2_654_435_761 + col * 40_503) % 1000) as f64 / 1000.0 - 0.5;
            price = (price * (1.0 + wiggle * 0.02)).max(0.5);
            close.push(price);

            size.push(if (i + col) % w.fill_every != 0 {
                // Size zero is ignored by the kernel and writes no record.
                0.0
            } else if w.size_type == SizeType::TargetPercent {
                if (i / w.fill_every) % 2 == 0 {
                    0.5
                } else {
                    0.2
                }
            } else if (i / w.fill_every) % 2 == 0 {
                1.0
            } else {
                // Safe for a long-only column too: the kernel clamps a sell to
                // the position rather than letting it go short.
                -0.5
            });
        }
    }
    (close, size)
}

/// Order-independent-free checksum over the raw bit patterns.
///
/// Wrapping addition of `f64::to_bits`, which Python reproduces exactly with
/// `struct.unpack("<Q", struct.pack("<d", v))`. Cheap, and sensitive to any
/// difference that matters.
fn checksum(values: &[f64]) -> u64 {
    values
        .iter()
        .fold(0u64, |acc, v| acc.wrapping_add(v.to_bits()))
}

fn parse_workloads(path: &Path) -> Vec<Workload> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let mut lines = text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty());
    let header: Vec<&str> = lines.next().expect("no header").split('\t').collect();
    let col = |name: &str| {
        header
            .iter()
            .position(|h| *h == name)
            .unwrap_or_else(|| panic!("workloads.tsv has no column {name:?}"))
    };
    let (c_name, c_rows, c_cols, c_every) =
        (col("name"), col("n_rows"), col("n_cols"), col("fill_every"));
    let (c_st, c_dir, c_pm) = (col("size_type"), col("direction"), col("price_mode"));
    let (c_fees, c_fixed, c_slip) = (col("fees"), col("fixed_fees"), col("slippage"));

    lines
        .map(|line| {
            let f: Vec<&str> = line.split('\t').collect();
            Workload {
                name: f[c_name].to_owned(),
                n_rows: f[c_rows].parse().unwrap(),
                n_cols: f[c_cols].parse().unwrap(),
                fill_every: f[c_every].parse().unwrap(),
                size_type: match f[c_st] {
                    "Amount" => SizeType::Amount,
                    "TargetPercent" => SizeType::TargetPercent,
                    other => panic!("unsupported size_type {other:?}"),
                },
                direction: match f[c_dir] {
                    "Both" => Direction::Both,
                    "LongOnly" => Direction::LongOnly,
                    "ShortOnly" => Direction::ShortOnly,
                    other => panic!("unsupported direction {other:?}"),
                },
                price_mode: f[c_pm].to_owned(),
                fees: f[c_fees].parse().unwrap(),
                fixed_fees: f[c_fixed].parse().unwrap(),
                slippage: f[c_slip].parse().unwrap(),
            }
        })
        .collect()
}

/// One row of `workloads_signals.tsv`.
struct SignalWorkload {
    name: String,
    n_rows: usize,
    n_cols: usize,
    entry_every: usize,
    hold: usize,
    direction: Direction,
    sl_stop: f64,
    tp_stop: f64,
    fees: f64,
    /// Features switched on, from the optional `features` column: `limit`,
    /// `leverage`, `margin`, `timestop`. Each feature row has a row without
    /// it to compare against.
    features: Vec<String>,
}

/// Generates close/high/low and an entry/exit pattern for `from_signals`.
///
/// Prices follow the `from_orders` formula so the two loops see the same
/// series. An entry fires every `entry_every` bars (offset by column) and
/// its exit `hold` bars later; with stops on, the stop engine usually gets
/// there first, which is the point of that workload.
#[allow(clippy::type_complexity)]
fn generate_signals(w: &SignalWorkload) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<bool>, Vec<bool>) {
    let n = w.n_rows * w.n_cols;
    let mut close = Vec::with_capacity(n);
    let mut high = Vec::with_capacity(n);
    let mut low = Vec::with_capacity(n);
    let mut entries = vec![false; n];
    let mut exits = vec![false; n];
    for col in 0..w.n_cols {
        let mut price = 10.0 + (col % 37) as f64;
        for i in 0..w.n_rows {
            let wiggle = ((i * 2_654_435_761 + col * 40_503) % 1000) as f64 / 1000.0 - 0.5;
            price = (price * (1.0 + wiggle * 0.02)).max(0.5);
            close.push(price);
            high.push(price * 1.01);
            low.push(price * 0.99);
            let k = col * w.n_rows + i;
            if (i + col) % w.entry_every == 0 {
                entries[k] = true;
            }
            if (i + col) % w.entry_every == w.hold % w.entry_every && i >= w.hold {
                exits[k] = true;
            }
        }
    }
    (close, high, low, entries, exits)
}

fn parse_signal_workloads(path: &Path) -> Vec<SignalWorkload> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let mut lines = text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty());
    let header: Vec<&str> = lines.next().expect("no header").split('\t').collect();
    let col = |name: &str| {
        header
            .iter()
            .position(|h| *h == name)
            .unwrap_or_else(|| panic!("workloads_signals.tsv has no column {name:?}"))
    };
    let (c_name, c_rows, c_cols) = (col("name"), col("n_rows"), col("n_cols"));
    let (c_every, c_hold, c_dir) = (col("entry_every"), col("hold"), col("direction"));
    let (c_sl, c_tp, c_fees) = (col("sl_stop"), col("tp_stop"), col("fees"));
    let c_features = header.iter().position(|h| *h == "features");
    lines
        .map(|line| {
            let f: Vec<&str> = line.split('\t').collect();
            SignalWorkload {
                name: f[c_name].to_owned(),
                n_rows: f[c_rows].parse().unwrap(),
                n_cols: f[c_cols].parse().unwrap(),
                entry_every: f[c_every].parse().unwrap(),
                hold: f[c_hold].parse().unwrap(),
                direction: match f[c_dir] {
                    "Both" => Direction::Both,
                    "LongOnly" => Direction::LongOnly,
                    "ShortOnly" => Direction::ShortOnly,
                    other => panic!("unsupported direction {other:?}"),
                },
                sl_stop: f[c_sl].parse().unwrap(),
                tp_stop: f[c_tp].parse().unwrap(),
                fees: f[c_fees].parse().unwrap(),
                features: c_features
                    .and_then(|c| f.get(c))
                    .filter(|v| **v != "-")
                    .map(|v| v.split(',').map(str::to_owned).collect())
                    .unwrap_or_default(),
            }
        })
        .collect()
}

/// Best of seven at each thread count, appended to `out` in the results format.
///
/// The shortest run is the least contaminated estimate on a shared machine,
/// and more samples tighten it: at three, repeated runs of identical code
/// moved by up to 10%, which is wider than several of the optimizations worth
/// measuring. Costs about a minute for the whole suite.
fn time_workload(
    out: &mut String,
    name: &str,
    n_rows: usize,
    n_cols: usize,
    sum: u64,
    cores: usize,
    run: impl Fn(usize) -> usize,
) {
    let elems = (n_rows * n_cols) as f64;
    let fills = run(1);
    // Serial first, so the numbers are the single-threaded baseline, then the
    // thread counts this machine can actually use.
    let mut thread_counts = vec![1usize];
    thread_counts.extend((2..=cores).filter(|t| *t == 2 || *t == cores));
    for threads in thread_counts {
        let mut best = f64::INFINITY;
        for _ in 0..7 {
            let start = Instant::now();
            let got = run(threads);
            best = best.min(start.elapsed().as_secs_f64());
            assert_eq!(got, fills, "thread count changed the result");
        }
        writeln!(
            out,
            "{name}\talphax\t{threads}\t{n_rows}\t{n_cols}\t{fills}\t{sum}\t{best:.6}\t{:.0}",
            elems / best
        )
        .unwrap();
        println!(
            "{name:<22} {threads:>2}t  {best:>8.4}s  {:>12.0} elem/s  ({fills} fills)",
            elems / best
        );
    }
}

/// Runs one workload once, serially, and returns its fill count.
///
/// `benchmarks/instructions.py` counts instructions inside this function only
/// (callgrind's `--toggle-collect`), which leaves input generation out of the
/// count. It must stay a function of its own for that to work.
#[inline(never)]
fn run_once(run: impl FnOnce() -> usize) -> usize {
    run()
}

/// Times a workload, or under `--once` runs it a single time.
fn measure(
    once: bool,
    out: &mut String,
    name: &str,
    n_rows: usize,
    n_cols: usize,
    sum: u64,
    cores: usize,
    run: impl Fn(usize) -> usize,
) {
    if once {
        let fills = run_once(|| run(1));
        println!("{name}\t{fills}\t{sum}");
    } else {
        time_workload(out, name, n_rows, n_cols, sum, cores, run);
    }
}

fn main() {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let cli: Vec<String> = std::env::args().skip(1).collect();
    let only = match cli.first().map(String::as_str) {
        Some("--once") => Some(cli.get(1).expect("--once needs a workload name").clone()),
        _ => None,
    };
    let skip = |name: &str| only.as_ref().is_some_and(|o| o != name);
    let out_path = cli.first().filter(|_| only.is_none()).map_or_else(
        || repo_root.join("benchmarks/results/alphax.tsv"),
        PathBuf::from,
    );
    let workloads = parse_workloads(&repo_root.join("benchmarks/workloads.tsv"));

    let cores = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    let mut out = String::new();
    writeln!(out, "# alphax (this engine), release build").unwrap();
    writeln!(out, "# cores: {cores}").unwrap();
    writeln!(
        out,
        "workload\timpl\tthreads\tn_rows\tn_cols\tfills\tchecksum\tbest_secs\telems_per_sec"
    )
    .unwrap();

    for w in workloads.iter().filter(|w| !skip(&w.name)) {
        let (close, size) = generate(w);
        let sum = checksum(&close).wrapping_add(checksum(&size));
        let (price, val_price) = match w.price_mode.as_str() {
            "close" => (f64::INFINITY, f64::INFINITY),
            "prev" => (f64::NEG_INFINITY, f64::INFINITY),
            other => panic!("unsupported price_mode {other:?}"),
        };

        let args = FromOrders {
            n_rows: w.n_rows,
            n_cols: w.n_cols,
            groups: Groups::ungrouped(w.n_cols),
            init_cash: FlexArray::Scalar(100_000.0),
            close: FlexArray::full(&close, w.n_rows, w.n_cols).unwrap(),
            size: FlexArray::full(&size, w.n_rows, w.n_cols).unwrap(),
            price: FlexArray::Scalar(price),
            val_price: FlexArray::Scalar(val_price),
            size_type: FlexArray::Scalar(w.size_type),
            direction: FlexArray::Scalar(w.direction),
            fees: FlexArray::Scalar(w.fees),
            fixed_fees: FlexArray::Scalar(w.fixed_fees),
            slippage: FlexArray::Scalar(w.slippage),
            ..Default::default()
        };
        measure(
            only.is_some(),
            &mut out,
            &w.name,
            w.n_rows,
            w.n_cols,
            sum,
            cores,
            |threads| {
                if threads == 1 {
                    simulate_from_orders(&args)
                } else {
                    simulate_from_orders_parallel(&args, threads)
                }
                .expect("simulation failed")
                .len()
            },
        );
    }

    let signal_workloads =
        parse_signal_workloads(&repo_root.join("benchmarks/workloads_signals.tsv"));
    for w in signal_workloads.iter().filter(|w| !skip(&w.name)) {
        let (close, high, low, entries, exits) = generate_signals(w);
        let sum = checksum(&close)
            .wrapping_add(entries.iter().filter(|e| **e).count() as u64)
            .wrapping_add(exits.iter().filter(|e| **e).count() as u64);
        let on = |name: &str| w.features.iter().any(|f| f == name);
        for f in &w.features {
            assert!(
                ["limit", "leverage", "margin", "timestop"].contains(&f.as_str()),
                "unknown feature {f:?} in workload {}",
                w.name
            );
        }
        let args = FromSignals {
            n_rows: w.n_rows,
            n_cols: w.n_cols,
            init_cash: FlexArray::Scalar(100_000.0),
            close: FlexArray::full(&close, w.n_rows, w.n_cols).unwrap(),
            high: FlexArray::full(&high, w.n_rows, w.n_cols).unwrap(),
            low: FlexArray::full(&low, w.n_rows, w.n_cols).unwrap(),
            entries: FlexArray::full(&entries, w.n_rows, w.n_cols).unwrap(),
            exits: FlexArray::full(&exits, w.n_rows, w.n_cols).unwrap(),
            direction: FlexArray::Scalar(w.direction),
            sl_stop: FlexArray::Scalar(w.sl_stop),
            tp_stop: FlexArray::Scalar(w.tp_stop),
            fees: FlexArray::Scalar(w.fees),
            order_type: FlexArray::Scalar(if on("limit") {
                OrderType::Limit
            } else {
                OrderType::Market
            }),
            limit_delta: FlexArray::Scalar(if on("limit") { 0.002 } else { f64::NAN }),
            leverage: FlexArray::Scalar(if on("leverage") { 2.0 } else { 1.0 }),
            account_type: FlexArray::Scalar(if on("margin") {
                AccountType::Margin
            } else {
                AccountType::Cash
            }),
            initial_margin: FlexArray::Scalar(if on("margin") { 0.1 } else { f64::NAN }),
            maintenance_margin: FlexArray::Scalar(if on("margin") { 0.05 } else { 0.0 }),
            funding_rate: FlexArray::Scalar(if on("margin") { 0.0001 } else { 0.0 }),
            td_stop: FlexArray::Scalar(if on("timestop") { 7.0 } else { f64::NAN }),
            use_stops: !(w.sl_stop.is_nan() && w.tp_stop.is_nan()) || on("timestop"),
            ..Default::default()
        };
        measure(
            only.is_some(),
            &mut out,
            &w.name,
            w.n_rows,
            w.n_cols,
            sum,
            cores,
            |threads| {
                if threads == 1 {
                    simulate_from_signals(&args)
                } else {
                    simulate_from_signals_parallel(&args, threads)
                }
                .expect("simulation failed")
                .len()
            },
        );
    }

    if let Some(name) = &only {
        let known = workloads.iter().map(|w| &w.name);
        assert!(
            known
                .chain(signal_workloads.iter().map(|w| &w.name))
                .any(|w| w == name),
            "no workload named {name:?}"
        );
        return;
    }
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).expect("cannot create results directory");
    }
    std::fs::write(&out_path, out).expect("cannot write results");
    println!("\nwrote {}", out_path.display());
}
