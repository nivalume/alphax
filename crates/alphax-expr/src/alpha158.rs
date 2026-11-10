use alphax_core::error::{Error, Result};

use crate::FactorSet;

/// The rolling families of Alpha158 in Qlib's order, by the key Qlib's
/// `include` / `exclude` use for them. `LOW` is the key of the `MIN` family.
pub const ALPHA158_ROLLING: [&str; 29] = [
    "ROC", "MA", "STD", "BETA", "RSQR", "RESI", "MAX", "LOW", "QTLU", "QTLD", "RANK", "RSV",
    "IMAX", "IMIN", "IMXD", "CORR", "CORD", "CNTP", "CNTN", "CNTD", "SUMP", "SUMN", "SUMD", "VMA",
    "VSTD", "WVMA", "VSUMP", "VSUMN", "VSUMD",
];

/// Which features `Alpha158DL.get_feature_config(config)` builds. A family
/// is on when its field is present, as a key is in Qlib's config dict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Alpha158Config {
    /// The nine candlestick features (`KMID` ... `KSFT2`).
    pub kbar: bool,
    /// `FIELDd = Ref($field, d)/$close`.
    pub price: Option<Alpha158Price>,
    /// `VOLUMEd = Ref($volume, d)/($volume+1e-12)` for each window; Qlib's
    /// default when the key has no windows is `0..5`.
    pub volume: Option<Vec<usize>>,
    /// The rolling families over their windows.
    pub rolling: Option<Alpha158Rolling>,
}

/// Qlib's `price` section. [`Default`] is what Qlib uses for keys the
/// section leaves out: windows `0..5` over open, high, low, close and vwap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Alpha158Price {
    pub windows: Vec<usize>,
    /// Field names; case does not matter (`OPEN` reads `$open`).
    pub features: Vec<String>,
}

impl Default for Alpha158Price {
    fn default() -> Self {
        Self {
            windows: (0..5).collect(),
            features: ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"]
                .map(String::from)
                .to_vec(),
        }
    }
}

/// Qlib's `rolling` section. [`Default`] is its defaults: windows 5, 10, 20,
/// 30 and 60, every family.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Alpha158Rolling {
    pub windows: Vec<usize>,
    /// Only these families ([`ALPHA158_ROLLING`] keys); `None` is all.
    pub include: Option<Vec<String>>,
    /// Never these families.
    pub exclude: Vec<String>,
}

impl Default for Alpha158Rolling {
    fn default() -> Self {
        Self {
            windows: vec![5, 10, 20, 30, 60],
            include: None,
            exclude: Vec::new(),
        }
    }
}

impl Default for Alpha158Config {
    /// Qlib's Alpha158: candlesticks, today's open, high, low and vwap over
    /// close, and every rolling family over the default windows.
    fn default() -> Self {
        Self {
            kbar: true,
            price: Some(Alpha158Price {
                windows: vec![0],
                features: ["OPEN", "HIGH", "LOW", "VWAP"].map(String::from).to_vec(),
            }),
            volume: None,
            rolling: Some(Alpha158Rolling::default()),
        }
    }
}

/// The `(name, expression)` pairs `Alpha158DL.get_feature_config(config)`
/// returns, with Qlib's names, order and expression text.
///
/// # Errors
///
/// A rolling family in `include` or `exclude` that is not one of
/// [`ALPHA158_ROLLING`]. (Qlib ignores an unknown key; here a misspelt
/// family is reported rather than silently built or left out.)
pub fn alpha158_features(config: &Alpha158Config) -> Result<Vec<(String, String)>> {
    let mut factors: Vec<(String, String)> = Vec::new();
    if config.kbar {
        for (name, expression) in [
            ("KMID", "($close-$open)/$open"),
            ("KLEN", "($high-$low)/$open"),
            ("KMID2", "($close-$open)/($high-$low+1e-12)"),
            ("KUP", "($high-Greater($open, $close))/$open"),
            ("KUP2", "($high-Greater($open, $close))/($high-$low+1e-12)"),
            ("KLOW", "(Less($open, $close)-$low)/$open"),
            ("KLOW2", "(Less($open, $close)-$low)/($high-$low+1e-12)"),
            ("KSFT", "(2*$close-$high-$low)/$open"),
            ("KSFT2", "(2*$close-$high-$low)/($high-$low+1e-12)"),
        ] {
            factors.push((name.into(), expression.into()));
        }
    }
    if let Some(price) = &config.price {
        for feature in &price.features {
            let field = feature.to_ascii_lowercase();
            for &d in &price.windows {
                let expression = if d == 0 {
                    format!("${field}/$close")
                } else {
                    format!("Ref(${field}, {d})/$close")
                };
                factors.push((format!("{}{d}", field.to_ascii_uppercase()), expression));
            }
        }
    }
    if let Some(windows) = &config.volume {
        for &d in windows {
            let expression = if d == 0 {
                "$volume/($volume+1e-12)".to_owned()
            } else {
                format!("Ref($volume, {d})/($volume+1e-12)")
            };
            factors.push((format!("VOLUME{d}"), expression));
        }
    }
    if let Some(rolling) = &config.rolling {
        let known = |key: &String| {
            if ALPHA158_ROLLING.contains(&key.as_str()) {
                Ok(())
            } else {
                let hint = if key == "MIN" {
                    " (the MIN family's key is LOW)"
                } else {
                    ""
                };
                Err(Error::Invariant(format!(
                    "unknown Alpha158 rolling family {key:?}{hint}"
                )))
            }
        };
        for key in rolling.include.iter().flatten().chain(&rolling.exclude) {
            known(key)?;
        }
        let used = |key: &str| {
            !rolling.exclude.iter().any(|k| k == key)
                && rolling
                    .include
                    .as_ref()
                    .is_none_or(|include| include.iter().any(|k| k == key))
        };
        for key in ALPHA158_ROLLING.into_iter().filter(|&key| used(key)) {
            let name = if key == "LOW" { "MIN" } else { key };
            for &d in &rolling.windows {
                factors.push((format!("{name}{d}"), rolling_expression(key, d)));
            }
        }
    }
    Ok(factors)
}

/// Qlib's expression for rolling family `key` over `d` rows.
fn rolling_expression(key: &str, d: usize) -> String {
    match key {
        "ROC" => format!("Ref($close, {d})/$close"),
        "MA" => format!("Mean($close, {d})/$close"),
        "STD" => format!("Std($close, {d})/$close"),
        "BETA" => format!("Slope($close, {d})/$close"),
        "RSQR" => format!("Rsquare($close, {d})"),
        "RESI" => format!("Resi($close, {d})/$close"),
        "MAX" => format!("Max($high, {d})/$close"),
        "LOW" => format!("Min($low, {d})/$close"),
        "QTLU" => format!("Quantile($close, {d}, 0.8)/$close"),
        "QTLD" => format!("Quantile($close, {d}, 0.2)/$close"),
        "RANK" => format!("Rank($close, {d})"),
        "RSV" => format!("($close-Min($low, {d}))/(Max($high, {d})-Min($low, {d})+1e-12)"),
        "IMAX" => format!("IdxMax($high, {d})/{d}"),
        "IMIN" => format!("IdxMin($low, {d})/{d}"),
        "IMXD" => format!("(IdxMax($high, {d})-IdxMin($low, {d}))/{d}"),
        "CORR" => format!("Corr($close, Log($volume+1), {d})"),
        "CORD" => format!("Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), {d})"),
        "CNTP" => format!("Mean($close>Ref($close, 1), {d})"),
        "CNTN" => format!("Mean($close<Ref($close, 1), {d})"),
        "CNTD" => format!("Mean($close>Ref($close, 1), {d})-Mean($close<Ref($close, 1), {d})"),
        "SUMP" => format!(
            "Sum(Greater($close-Ref($close, 1), 0), {d})/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)"
        ),
        "SUMN" => format!(
            "Sum(Greater(Ref($close, 1)-$close, 0), {d})/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)"
        ),
        "SUMD" => format!(
            "(Sum(Greater($close-Ref($close, 1), 0), {d})-Sum(Greater(Ref($close, 1)-$close, 0), {d}))/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)"
        ),
        "VMA" => format!("Mean($volume, {d})/($volume+1e-12)"),
        "VSTD" => format!("Std($volume, {d})/($volume+1e-12)"),
        "WVMA" => format!(
            "Std(Abs($close/Ref($close, 1)-1)*$volume, {d})/(Mean(Abs($close/Ref($close, 1)-1)*$volume, {d})+1e-12)"
        ),
        "VSUMP" => format!(
            "Sum(Greater($volume-Ref($volume, 1), 0), {d})/(Sum(Abs($volume-Ref($volume, 1)), {d})+1e-12)"
        ),
        "VSUMN" => format!(
            "Sum(Greater(Ref($volume, 1)-$volume, 0), {d})/(Sum(Abs($volume-Ref($volume, 1)), {d})+1e-12)"
        ),
        "VSUMD" => format!(
            "(Sum(Greater($volume-Ref($volume, 1), 0), {d})-Sum(Greater(Ref($volume, 1)-$volume, 0), {d}))/(Sum(Abs($volume-Ref($volume, 1)), {d})+1e-12)"
        ),
        _ => unreachable!("keys come from ALPHA158_ROLLING"),
    }
}

/// Compiles Qlib's default Alpha158 feature library.
///
/// The formulas and output names follow `Alpha158DL.get_feature_config()`:
/// nine candlestick features, four current-price features, and 29 rolling
/// families over five windows.
///
/// # Errors
///
/// Propagates an internal expression parse error.
pub fn alpha158() -> Result<FactorSet> {
    alpha158_with(&Alpha158Config::default())
}

/// Compiles the Alpha158 features `config` selects.
///
/// # Errors
///
/// As [`alpha158_features`], or a configuration that selects nothing.
pub fn alpha158_with(config: &Alpha158Config) -> Result<FactorSet> {
    FactorSet::compile(alpha158_features(config)?)
}

/// Qlib's Alpha158 / Alpha360 label `LABEL0`: the return from the next bar
/// to the one after, `Ref($price, -2)/Ref($price, -1) - 1`, on `close`
/// (Qlib's `Alpha158`) or `vwap` (`Alpha158vwap`). It reads two bars ahead,
/// so it is a training target, never a signal: the set's
/// [`FactorSet::uses_future`] is true.
///
/// # Errors
///
/// A price that is not a plain field name.
pub fn qlib_label(price: &str) -> Result<FactorSet> {
    if price.is_empty()
        || !price
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_')
    {
        return Err(Error::Invariant(format!("invalid label price {price:?}")));
    }
    let price = price.to_ascii_lowercase();
    FactorSet::compile([("LABEL0", format!("Ref(${price}, -2)/Ref(${price}, -1) - 1"))])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_is_qlibs_158_in_family_order() {
        let factors = alpha158_features(&Alpha158Config::default()).unwrap();
        assert_eq!(factors.len(), 158);
        let names: Vec<&str> = factors.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(
            names[..14],
            [
                "KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2", "OPEN0",
                "HIGH0", "LOW0", "VWAP0", "ROC5"
            ]
        );
        assert_eq!(names[48..53], ["MIN5", "MIN10", "MIN20", "MIN30", "MIN60"]);
        assert_eq!(names[157], "VSUMD60");
    }

    #[test]
    fn unknown_families_are_reported() {
        let config = Alpha158Config {
            rolling: Some(Alpha158Rolling {
                include: Some(vec!["MIN".into()]),
                ..Alpha158Rolling::default()
            }),
            ..Alpha158Config::default()
        };
        let error = alpha158_features(&config).unwrap_err().to_string();
        assert!(error.contains("LOW"), "{error}");
    }

    #[test]
    fn the_label_reads_two_bars_ahead() {
        let label = qlib_label("vwap").unwrap();
        assert!(label.uses_future());
        assert_eq!(label.names().collect::<Vec<_>>(), ["LABEL0"]);
        assert!(qlib_label("$close").is_err());
    }
}
