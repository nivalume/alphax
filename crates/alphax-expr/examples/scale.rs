//! Wall-clock timing of the native backend at a realistic panel size.
//!
//! ```bash
//! cargo run --release --example scale -p alphax-expr -- 2520 500          # everything
//! cargo run --release --example scale -p alphax-expr -- 2520 500 sets     # only the sets
//! cargo run --release --example scale -p alphax-expr -- 20000 16 ops      # only operators
//! cargo run --release --example scale -p alphax-expr -- 2520 500 grid     # a parameter grid
//! cargo run --release --example scale -p alphax-expr -- 2520 500 warmup   # warm-up's cost
//! ```
//!
//! Criterion (`benches/backends.rs`) covers small panels with statistics; this
//! covers the size where an `O(window)` kernel or an unbounded cache shows up.
//! Peak RSS (`VmHWM`) is printed after each set on Linux; it is a high-water
//! mark for the process, so the first set's number includes the inputs.

use std::time::{Duration, Instant};

use alphax_expr::{FactorGrid, FactorInputs, FactorSet, GridCase, alpha158, alpha360};
use alphax_model::OhlcvPanel;

fn main() {
    let mut args = std::env::args().skip(1);
    let n_rows = args.next().map_or(2_520, |value| value.parse().unwrap());
    let n_cols = args.next().map_or(500, |value| value.parse().unwrap());
    let which = args.next().unwrap_or_else(|| "all".to_owned());
    let (ops, sets) = match which.as_str() {
        "all" => (true, true),
        "ops" => (true, false),
        "sets" | "grid" | "warmup" => (false, which == "sets"),
        other => panic!("unknown selection {other:?}; use all, ops, sets, grid or warmup"),
    };
    let series = |base: f64, trend: f64, wave: f64| {
        (0..n_cols)
            .flat_map(|column| {
                (0..n_rows).map(move |row| {
                    base + column as f64 * 7.0
                        + row as f64 * trend
                        + ((row * 17 + column * 11) % 23) as f64 * wave
                })
            })
            .collect::<Vec<_>>()
    };
    let panel = OhlcvPanel {
        open: series(100.0, 0.010, 0.03),
        high: series(101.0, 0.012, 0.04),
        low: series(99.0, 0.008, 0.02),
        close: series(100.5, 0.011, 0.05),
        volume: series(1_000.0, 0.30, 0.7),
        n_rows,
        n_cols,
    };
    let vwap = series(100.4, 0.0105, 0.025);
    let inputs = FactorInputs::from_ohlcv(&panel)
        .unwrap()
        .with_field("vwap", &vwap)
        .unwrap();

    println!("panel {n_rows}x{n_cols}");
    let single = [
        "Mean($close,60)",
        "Sum($volume,60)",
        "Std($close,60)",
        "Var($close,60)",
        "Max($high,60)",
        "Min($low,60)",
        "WMA($close,60)",
        "Slope($close,60)",
        "Rsquare($close,60)",
        "Resi($close,60)",
        "Corr($close,Log($volume+1),60)",
        "Cov($close,$volume,60)",
        "Quantile($close,60,0.8)",
        "Med($close,60)",
        "Rank($close,60)",
        "Mad($close,60)",
        "Skew($close,60)",
        "Kurt($close,60)",
        "IdxMax($high,60)",
        "IdxMin($low,60)",
        "Count($close,60)",
        "($close-$open)/($high-$low+1e-12)",
    ];
    if ops {
        for source in single {
            let factors = FactorSet::compile([("f", source)]).unwrap();
            println!("{:>40}  {:>9.1} ms", source, time(&factors, &inputs));
        }
    }
    match which.as_str() {
        "grid" => grid(&inputs),
        "warmup" => warmup(&inputs),
        _ => {}
    }
    if sets {
        for (name, set) in [
            ("alpha158", alpha158().unwrap()),
            ("alpha360", alpha360().unwrap()),
        ] {
            let ms = time(&set, &inputs);
            match peak_rss_mb() {
                Some(mb) => println!("{name:>40}  {ms:>9.1} ms   peak RSS {mb:.0} MB"),
                None => println!("{name:>40}  {ms:>9.1} ms"),
            }
        }
    }
}

/// A 20-case grid (5 windows × 4 thresholds) as one [`FactorGrid`] against
/// each case compiled and evaluated on its own.
fn grid(inputs: &FactorInputs<'_>) {
    let factors = [
        (
            "z",
            "($close - Mean($close, {n})) / (Std($close, {n}) + 1e-12) > {k}",
        ),
        ("trend", "Slope($close, {n}) / $close > {k} / 100"),
        (
            "range",
            "($close - Min($low, {n})) / (Max($high, {n}) - Min($low, {n}) + 1e-12)",
        ),
    ];
    let mut cases = Vec::new();
    for n in ["10", "20", "40", "60", "120"] {
        for k in ["0", "0.5", "1", "2"] {
            let case: GridCase = [("n", n), ("k", k)]
                .into_iter()
                .map(|(a, b)| (a.to_owned(), b.to_owned()))
                .collect();
            cases.push(case);
        }
    }
    let start = Instant::now();
    let grid = FactorGrid::compile(factors, cases.clone()).unwrap();
    let compile_ms = start.elapsed().as_secs_f64() * 1e3;
    let grid_ms = best(|| {
        std::hint::black_box(grid.evaluate(inputs).unwrap());
    });
    let mut each_ms = 0.0;
    for case in &cases {
        let set = FactorSet::compile(factors.iter().map(|(name, source)| {
            let mut text = (*source).to_owned();
            for (param, value) in case {
                text = text.replace(&format!("{{{param}}}"), value);
            }
            (*name, text)
        }))
        .unwrap();
        each_ms += time(&set, inputs);
    }
    println!(
        "{:>40}  {grid_ms:>9.1} ms   (compile {compile_ms:.1} ms)",
        format!("grid: {} factors x {} cases", factors.len(), cases.len())
    );
    println!("{:>40}  {each_ms:>9.1} ms", "each case on its own, summed");
}

/// Alpha158 with and without warm-up.
fn warmup(inputs: &FactorInputs<'_>) {
    let plain = alpha158().unwrap();
    let warm = plain.clone().with_warmup(true);
    println!("{:>40}  {:>9.1} ms", "alpha158", time(&plain, inputs));
    println!(
        "{:>40}  {:>9.1} ms",
        "alpha158 with warm-up",
        time(&warm, inputs)
    );
}

fn best(mut run: impl FnMut()) -> f64 {
    let mut best = Duration::MAX;
    for _ in 0..3 {
        let start = Instant::now();
        run();
        best = best.min(start.elapsed());
    }
    best.as_secs_f64() * 1e3
}

/// The process high-water mark from `/proc/self/status`, Linux only.
fn peak_rss_mb() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kb: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024.0)
}

fn time(factors: &FactorSet, inputs: &FactorInputs<'_>) -> f64 {
    let mut best = Duration::MAX;
    for _ in 0..3 {
        let start = Instant::now();
        std::hint::black_box(factors.evaluate(inputs).unwrap());
        best = best.min(start.elapsed());
    }
    best.as_secs_f64() * 1e3
}
