use alphax_core::error::Result;

use crate::FactorSet;

/// Compiles Qlib's default Alpha360 feature library.
///
/// The formulas and output names follow `Alpha360DL.get_feature_config()`:
/// 60 normalized observations for close, open, high, low, VWAP, and volume.
///
/// # Errors
///
/// Propagates an internal expression parse error.
pub fn alpha360() -> Result<FactorSet> {
    let mut factors = Vec::with_capacity(360);
    for field in ["close", "open", "high", "low", "vwap"] {
        let prefix = field.to_ascii_uppercase();
        for offset in (0..60).rev() {
            let expression = if offset == 0 {
                format!("${field}/$close")
            } else {
                format!("Ref(${field},{offset})/$close")
            };
            factors.push((format!("{prefix}{offset}"), expression));
        }
    }
    for offset in (0..60).rev() {
        let expression = if offset == 0 {
            "$volume/($volume+1e-12)".to_owned()
        } else {
            format!("Ref($volume,{offset})/($volume+1e-12)")
        };
        factors.push((format!("VOLUME{offset}"), expression));
    }
    FactorSet::compile(factors)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::FactorInputs;
    use alphax_model::OhlcvPanel;

    #[test]
    fn alpha360_has_the_expected_names_and_metadata() {
        let factors = alpha360().unwrap();
        let expected = ["CLOSE", "OPEN", "HIGH", "LOW", "VWAP", "VOLUME"]
            .into_iter()
            .flat_map(|field| (0..60).rev().map(move |offset| format!("{field}{offset}")))
            .collect::<Vec<_>>();

        assert_eq!(factors.names().collect::<Vec<_>>(), expected);
        assert_eq!(
            factors.required_fields(),
            BTreeSet::from(["close", "high", "low", "open", "volume", "vwap"])
        );
        assert!(!factors.uses_future());
    }

    #[test]
    fn alpha360_evaluates_history_and_keeps_symbols_separate() {
        let n_rows = 61;
        let n_cols = 2;
        let series = |base: f64| {
            (0..n_cols)
                .flat_map(|column| {
                    (0..n_rows).map(move |row| base + column as f64 * 1_000.0 + row as f64)
                })
                .collect::<Vec<_>>()
        };
        let panel = OhlcvPanel {
            open: series(100.0),
            high: series(101.0),
            low: series(99.0),
            close: series(100.5),
            volume: series(1_000.0),
            n_rows,
            n_cols,
        };
        let vwap = series(100.25);
        let inputs = FactorInputs::from_ohlcv(&panel)
            .unwrap()
            .with_field("vwap", &vwap)
            .unwrap();
        let values = alpha360().unwrap().evaluate(&inputs).unwrap();
        let assert_close = |actual: f64, expected: f64| {
            assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
        };

        assert_eq!(values.iter().len(), 360);
        assert!(
            values
                .iter()
                .all(|(_, value)| value.n_rows() == n_rows && value.n_cols() == n_cols)
        );
        let close59 = values.get("CLOSE59").unwrap();
        assert!(close59.get(58, 0).is_nan());
        assert_close(close59.get(59, 0), panel.close[0] / panel.close[59]);
        assert_close(
            close59.get(59, 1),
            panel.close[n_rows] / panel.close[n_rows + 59],
        );
        assert_close(values.get("CLOSE0").unwrap().get(60, 1), 1.0);
        assert_close(
            values.get("OPEN59").unwrap().get(59, 0),
            panel.open[0] / panel.close[59],
        );
        assert_close(
            values.get("VWAP59").unwrap().get(59, 0),
            vwap[0] / panel.close[59],
        );
        assert_close(
            values.get("VOLUME59").unwrap().get(59, 0),
            panel.volume[0] / (panel.volume[59] + 1e-12),
        );
    }
}
