//! Streams reproduce the batch indicators bit for bit.

use alphax_core::matrix::Matrix;
use alphax_indicators::{
    Bar, BbParams, Indicator, MaParams, MacdParams, StochParams, Stream, atr, bbands, ma, macd,
    mstd, obv, rsi, stoch,
};

fn series(n: usize, seed: u64, level: f64) -> Vec<f64> {
    let mut state = seed;
    let mut x = level;
    (0..n)
        .map(|i| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            x *= 1.0 + 0.02 * u;
            // Gaps, a flat stretch and a leading NaN.
            if i == 0 || i % 37 == 5 || (60..64).contains(&i) {
                if (60..64).contains(&i) {
                    100.0
                } else {
                    f64::NAN
                }
            } else {
                x
            }
        })
        .collect()
}

fn col(v: &[f64]) -> Matrix {
    Matrix::from_column_major(v.to_vec(), v.len(), 1).unwrap()
}

fn same(name: &str, batch: &[f64], streamed: &[f64]) {
    for (i, (a, b)) in batch.iter().zip(streamed).enumerate() {
        assert!(
            a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
            "{name} bar {i}: batch {a} stream {b}"
        );
    }
}

fn check(name: &str, batch: &Indicator, mut stream: Stream, bars: &[Bar]) {
    let names = stream.names();
    let mut outs = vec![Vec::new(); names.len()];
    for &bar in bars {
        for (k, &v) in stream.update(bar).iter().enumerate() {
            outs[k].push(v);
        }
    }
    for (k, output) in names.iter().enumerate() {
        same(
            &format!("{name}.{output}"),
            batch.get(output).unwrap().column(0),
            &outs[k],
        );
    }
}

#[test]
fn every_stream_matches_its_batch_indicator() {
    let n = 300;
    let close = series(n, 1, 100.0);
    let high: Vec<f64> = close
        .iter()
        .zip(series(n, 2, 1.0))
        .map(|(c, u)| c + u.abs())
        .collect();
    let low: Vec<f64> = close
        .iter()
        .zip(series(n, 3, 1.0))
        .map(|(c, u)| c - u.abs())
        .collect();
    let volume = series(n, 4, 1000.0);
    let bars: Vec<Bar> = (0..n)
        .map(|i| Bar {
            high: high[i],
            low: low[i],
            close: close[i],
            volume: volume[i],
        })
        .collect();
    let (c, h, l, v) = (col(&close), col(&high), col(&low), col(&volume));
    for adjust in [false, true] {
        for ewm in [false, true] {
            for window in [1, 2, 14] {
                let p = MaParams { window, ewm };
                let tag = format!("w{window} ewm{ewm} adjust{adjust}");
                check(
                    &format!("ma {tag}"),
                    &ma(&c, &[p], adjust, None).unwrap(),
                    Stream::ma(p, adjust).unwrap(),
                    &bars,
                );
                for ddof in [0, 1] {
                    check(
                        &format!("mstd {tag} ddof{ddof}"),
                        &mstd(&c, &[p], adjust, ddof, None).unwrap(),
                        Stream::mstd(p, adjust, ddof).unwrap(),
                        &bars,
                    );
                }
                let b = BbParams {
                    window,
                    ewm,
                    alpha: 2.0,
                };
                check(
                    &format!("bbands {tag}"),
                    &bbands(&c, &[b], adjust, 1, None).unwrap(),
                    Stream::bbands(b, adjust, 1).unwrap(),
                    &bars,
                );
                check(
                    &format!("rsi {tag}"),
                    &rsi(&c, &[p], adjust, None).unwrap(),
                    Stream::rsi(p, adjust).unwrap(),
                    &bars,
                );
                check(
                    &format!("atr {tag}"),
                    &atr(&h, &l, &c, &[p], adjust, None).unwrap(),
                    Stream::atr(p, adjust).unwrap(),
                    &bars,
                );
                let s = StochParams {
                    k_window: window + 2,
                    d_window: 3,
                    d_ewm: ewm,
                };
                check(
                    &format!("stoch {tag}"),
                    &stoch(&h, &l, &c, &[s], adjust, None).unwrap(),
                    Stream::stoch(s, adjust).unwrap(),
                    &bars,
                );
                let m = MacdParams {
                    fast_window: window,
                    slow_window: window + 12,
                    signal_window: 9,
                    macd_ewm: ewm,
                    signal_ewm: !ewm,
                };
                check(
                    &format!("macd {tag}"),
                    &macd(&c, &[m], adjust, None).unwrap(),
                    Stream::macd(m, adjust).unwrap(),
                    &bars,
                );
            }
        }
    }
    check("obv", &obv(&c, &v, None).unwrap(), Stream::obv(), &bars);
}
