use std::hint::black_box;
use std::time::Duration;

use alphax_expr::{
    ComputeBackend, FactorInputs, FactorSet, NativeBackend, PolarsBackend, alpha158,
};
use alphax_model::OhlcvPanel;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

fn factor_backends(criterion: &mut Criterion) {
    let n_rows = 1_024;
    let n_cols = 16;
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
    let workloads = [
        (
            "elementwise",
            FactorSet::compile([
                ("return", "$close/Ref($close,1)-1"),
                ("body", "($close-$open)/($high-$low+1e-12)"),
                ("upper", "($high-Greater($open,$close))/$open"),
                ("range", "($high-$low)/$close"),
            ])
            .unwrap(),
        ),
        (
            "standard_rolling",
            FactorSet::compile([
                ("mean", "Mean($close,20)"),
                ("std", "Std($close,20)"),
                ("sum", "Sum($volume,20)"),
                ("max", "Max($high,20)"),
                ("min", "Min($low,20)"),
                ("quantile", "Quantile($close,20,0.8)"),
            ])
            .unwrap(),
        ),
        ("alpha158", alpha158().unwrap()),
    ];
    let native = NativeBackend;
    let polars = PolarsBackend;
    let backends: [&dyn ComputeBackend; 2] = [&native, &polars];

    for (workload, factors) in workloads {
        let mut group = criterion.benchmark_group(format!("{workload}/{n_rows}x{n_cols}"));
        group.sample_size(10);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(4));
        group.throughput(Throughput::Elements((n_rows * n_cols) as u64));
        for backend in backends {
            group.bench_with_input(
                BenchmarkId::from_parameter(backend.name()),
                &(),
                |bench, ()| {
                    bench.iter(|| {
                        black_box(factors.evaluate_with(black_box(&inputs), backend).unwrap())
                    });
                },
            );
        }
        group.finish();
    }
}

criterion_group!(benches, factor_backends);
criterion_main!(benches);
