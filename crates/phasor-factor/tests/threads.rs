//! Thread invariance tests: 1, 2, 4, 8 threads must produce bit-identical results.

use phasor_factor::{
    Method, Panel, cs_ic, cs_permutation, ts_block_ic, ts_circular_shift, ts_ic, ts_rolling_ic,
};

#[test]
fn threads_produce_bit_identical_results() {
    let n_rows = 120;
    let n_cols = 16;
    let mut f_data = vec![0.0; n_rows * n_cols];
    let mut ret_data = vec![0.0; n_rows * n_cols];
    let mut blocks = vec![0i64; n_rows];

    // Seeded deterministic values with some NaNs and ties
    for c in 0..n_cols {
        for r in 0..n_rows {
            let idx = c * n_rows + r;
            if (r + c * 3) % 17 == 0 {
                f_data[idx] = f64::NAN;
            } else {
                f_data[idx] = ((r * 7 + c * 13) % 29) as f64 * 0.1;
            }

            if (r * 2 + c) % 19 == 0 {
                ret_data[idx] = f64::NAN;
            } else {
                ret_data[idx] = ((r * 11 + c * 5) % 31) as f64 * 0.05 - 0.7;
            }
        }
    }
    for (r, b) in blocks.iter_mut().enumerate() {
        *b = (r / 20) as i64;
    }

    let p_f = Panel::new(&f_data, n_rows, n_cols).unwrap();
    let p_r = Panel::new(&ret_data, n_rows, n_cols).unwrap();

    let baseline_cs = cs_ic(&[p_f], &[p_r], 4, 1).unwrap();
    let baseline_ts = ts_ic(&p_f, &p_r, Method::Spearman, 10, 1).unwrap();
    let baseline_roll = ts_rolling_ic(&p_f, &p_r, 20, 10, Method::Spearman, 1).unwrap();
    let baseline_block = ts_block_ic(&p_f, &p_r, &blocks, 5, Method::Spearman, 1).unwrap();

    for threads in [2, 3, 4, 8] {
        let cs = cs_ic(&[p_f], &[p_r], 4, threads).unwrap();
        assert_eq!(
            baseline_cs
                .pearson
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            cs.pearson.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "cs pearson differed on {threads} threads"
        );
        assert_eq!(
            baseline_cs
                .spearman
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            cs.spearman.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "cs spearman differed on {threads} threads"
        );

        let ts = ts_ic(&p_f, &p_r, Method::Spearman, 10, threads).unwrap();
        assert_eq!(
            baseline_ts.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            ts.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "ts spearman differed on {threads} threads"
        );

        let roll = ts_rolling_ic(&p_f, &p_r, 20, 10, Method::Spearman, threads).unwrap();
        assert_eq!(
            baseline_roll
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            roll.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "ts rolling spearman differed on {threads} threads"
        );

        let block = ts_block_ic(&p_f, &p_r, &blocks, 5, Method::Spearman, threads).unwrap();
        assert_eq!(baseline_block.0, block.0);
        assert_eq!(
            baseline_block
                .1
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            block.1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "ts block spearman differed on {threads} threads"
        );
    }
}

#[test]
fn placebo_threads_are_bit_identical() {
    let n_rows = 36;
    let n_cols = 8;
    let mut factor = vec![0.0; n_rows * n_cols];
    let mut forward = vec![0.0; n_rows * n_cols];
    for c in 0..n_cols {
        for r in 0..n_rows {
            let idx = c * n_rows + r;
            factor[idx] = if (r + c) % 7 == 0 {
                f64::NAN
            } else {
                (r * 3 + c) as f64
            };
            forward[idx] = if (r * 2 + c) % 5 == 0 {
                f64::NAN
            } else {
                (c * 4) as f64 - r as f64
            };
        }
    }
    let px = Panel::new(&factor, n_rows, n_cols).unwrap();
    let py = Panel::new(&forward, n_rows, n_cols).unwrap();
    let cs = cs_permutation(&px, &py, 10, 7, 3, 1).unwrap();
    let ts = ts_circular_shift(&px, &py, 8, 7, 6, 1).unwrap();
    for threads in [2, 4] {
        let other = cs_permutation(&px, &py, 10, 7, 3, threads).unwrap();
        assert_eq!(cs.observed.to_bits(), other.observed.to_bits());
        assert_eq!(cs.p_value.to_bits(), other.p_value.to_bits());
        assert_eq!(
            cs.null_distribution
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            other
                .null_distribution
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        let shifted = ts_circular_shift(&px, &py, 8, 7, 6, threads).unwrap();
        assert_eq!(ts.observed.to_bits(), shifted.observed.to_bits());
        assert_eq!(
            ts.null_distribution
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            shifted
                .null_distribution
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    }
}
