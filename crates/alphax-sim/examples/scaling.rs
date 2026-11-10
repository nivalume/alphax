//! Measures how `from_orders` scales across threads.
//!
//! Columns are fully independent, so this workload should scale close to
//! linearly. **Sub-linear scaling is a finding, not a characteristic to
//! accept** — it means false sharing, allocator contention, memory bandwidth,
//! or state that is not as column-local as the design claims.
//!
//! Two workloads run, because they stress different things:
//!
//! - **dense** — every element fills. Record writing dominates; hundreds of
//!   megabytes move per run and the limit is memory, not cores.
//! - **sparse** — roughly one element in twenty fills, which is closer to a
//!   real sweep. The simulation loop dominates and scaling shows what the
//!   parallel decomposition is actually worth.
//!
//! Comparing the two separates "our parallelism is broken" from "this
//! particular workload is memory-bound".
//!
//! Run in release; a debug build of the execution kernel is roughly fifty
//! times slower and the numbers mean nothing:
//!
//! ```text
//! cargo run --release --example scaling -p alphax-sim
//! ```

use std::time::Instant;

use alphax_core::flex::FlexArray;
use alphax_sim::groups::Groups;
use alphax_sim::{FromOrders, simulate_from_orders_parallel};

/// A sweep-shaped workload: a price path per column, orders every `every`
/// bars, alternating sign. `every == 1` fills on every element.
fn build(n_rows: usize, n_cols: usize, every: usize) -> (Vec<f64>, Vec<f64>) {
    let mut close = Vec::with_capacity(n_rows * n_cols);
    let mut size = Vec::with_capacity(n_rows * n_cols);
    for col in 0..n_cols {
        // A cheap deterministic walk; no RNG dependency.
        let mut price = 10.0 + (col % 37) as f64;
        for i in 0..n_rows {
            let wiggle = ((i * 2_654_435_761 + col * 40_503) % 1000) as f64 / 1000.0 - 0.5;
            price = (price * (1.0 + wiggle * 0.02)).max(0.5);
            close.push(price);
            // Size zero is ignored by the kernel, so it writes no record.
            size.push(if (i + col) % every == 0 {
                if (i / every) % 2 == 0 { 1.0 } else { -0.5 }
            } else {
                0.0
            });
        }
    }
    (close, size)
}

fn run(label: &str, n_rows: usize, n_cols: usize, every: usize, cores: usize) {
    let (close, size) = build(n_rows, n_cols, every);
    let args = FromOrders {
        n_rows,
        n_cols,
        groups: Groups::ungrouped(n_cols),
        init_cash: FlexArray::Scalar(100_000.0),
        close: FlexArray::full(&close, n_rows, n_cols).unwrap(),
        size: FlexArray::full(&size, n_rows, n_cols).unwrap(),
        fees: FlexArray::Scalar(0.001),
        slippage: FlexArray::Scalar(0.0005),
        ..Default::default()
    };

    let elems = (n_rows * n_cols) as f64;
    let warm = simulate_from_orders_parallel(&args, 1).expect("simulation failed");
    println!(
        "\n{label}: {n_rows} bars x {n_cols} cols = {} elements, {} fills ({:.0}%)",
        n_rows * n_cols,
        warm.len(),
        100.0 * warm.len() as f64 / elems,
    );
    println!(
        "{:>7}  {:>9}  {:>13}  {:>8}  {:>10}",
        "threads", "time", "elem/s", "speedup", "efficiency"
    );

    let mut baseline = 0.0_f64;
    for threads in [1, 2, 3, 4] {
        if threads > cores {
            break;
        }
        // Best of three: the shortest run is the least polluted by scheduling.
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let start = Instant::now();
            let records = simulate_from_orders_parallel(&args, threads).expect("simulation failed");
            let secs = start.elapsed().as_secs_f64();
            assert_eq!(records.len(), warm.len(), "thread count changed the result");
            best = best.min(secs);
        }
        if threads == 1 {
            baseline = best;
        }
        let speedup = baseline / best;
        println!(
            "{threads:>7}  {:>8.3}s  {:>13.0}  {speedup:>7.2}x  {:>9.0}%",
            best,
            elems / best,
            100.0 * speedup / threads as f64,
        );
    }
}

fn main() {
    let cores = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    println!("{cores} cores available");

    run("dense  (every element fills)", 20_000, 256, 1, cores);
    run("sparse (1 in 20 fills)", 20_000, 256, 20, cores);
}
