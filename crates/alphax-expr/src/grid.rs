//! Parameterized factor expressions: `{name}` placeholders over a grid.
//!
//! `($close - Mean($close, {n})) / Std($close, {n})` over `n = 10, 20, 50`
//! is three factors, and they share everything that does not depend on
//! `n`. [`FactorGrid`] substitutes each case into each expression and
//! compiles them all into **one** [`FactorSet`]: one graph, so `$close` (and
//! any parameter-free subexpression) is read once, and one pass over the
//! symbols evaluates every case. A parameter can stand anywhere in the
//! source text: a window, a threshold, a field (`${field}`), an instrument.
//!
//! Each factor's result lays the cases side by side, the layout of every
//! parameter sweep in alphax: column `case * n_symbols + symbol`, so a
//! `(rows, symbols)` price panel is shared through
//! `column_map = [0, 1, ..., n_symbols - 1]` repeated once per case.

use std::collections::BTreeMap;

use alphax_core::error::{Error, Result};
use alphax_core::matrix::Matrix;

use crate::{FactorInputs, FactorSet, FactorValues};

/// One case of a grid: each parameter's value, as source text (`"20"`,
/// `"0.5"`, `"close"`).
pub type GridCase = BTreeMap<String, String>;

/// Factor expressions compiled for every case of a parameter grid.
#[derive(Debug, Clone)]
pub struct FactorGrid {
    set: FactorSet,
    names: Vec<String>,
    cases: Vec<GridCase>,
}

impl FactorGrid {
    /// Substitutes every case into every `(name, expression)` and compiles
    /// them together.
    ///
    /// # Errors
    ///
    /// No case, an empty or repeated name, a placeholder some case gives no
    /// value, or any error compiling an instantiated expression (reported
    /// with the factor's name and the case).
    pub fn compile<N, S, I>(factors: I, cases: Vec<GridCase>) -> Result<Self>
    where
        N: Into<String>,
        S: AsRef<str>,
        I: IntoIterator<Item = (N, S)>,
    {
        if cases.is_empty() {
            return Err(Error::Invariant(
                "a factor grid needs at least one case".into(),
            ));
        }
        let mut names = Vec::new();
        let mut flat = Vec::new();
        for (name, source) in factors {
            let name = name.into();
            if name.is_empty() || names.contains(&name) {
                return Err(Error::Invariant(format!(
                    "invalid or repeated factor name {name:?}"
                )));
            }
            for (k, case) in cases.iter().enumerate() {
                let expression = substitute(source.as_ref(), case)
                    .map_err(|message| Error::Invariant(format!("factor {name:?}: {message}")))?;
                flat.push((format!("{name}#{k}"), expression));
            }
            names.push(name);
        }
        let set = FactorSet::compile(flat).map_err(|error| match error {
            Error::Invariant(message) => Error::Invariant(locate(&message, &cases)),
            other => other,
        })?;
        Ok(Self { set, names, cases })
    }

    /// Factor names in output order.
    pub fn names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.names.iter().map(String::as_str)
    }

    /// The cases in column order.
    #[must_use]
    pub fn cases(&self) -> &[GridCase] {
        &self.cases
    }

    /// Every instantiated factor as one set, factor-major: factor `f` of case
    /// `c` is its `f * n_cases + c`-th, named `name#c`.
    #[must_use]
    pub fn set(&self) -> &FactorSet {
        &self.set
    }

    /// As [`FactorSet::with_warmup`].
    #[must_use]
    pub fn with_warmup(mut self, on: bool) -> Self {
        self.set = self.set.with_warmup(on);
        self
    }

    /// Per factor, per case: [`FactorSet::lookbacks`].
    #[must_use]
    pub fn lookbacks(&self) -> Vec<Vec<usize>> {
        self.set
            .lookbacks()
            .chunks(self.cases.len())
            .map(<[usize]>::to_vec)
            .collect()
    }

    /// Every factor over every case, each `(rows, n_cases * symbols)` with
    /// column `case * symbols + symbol`.
    ///
    /// # Errors
    ///
    /// As [`FactorSet::evaluate`].
    pub fn evaluate(&self, inputs: &FactorInputs<'_>) -> Result<FactorValues> {
        let (n_rows, n_cols) = inputs.shape();
        let n_cases = self.cases.len();
        let mut flat = self.set.evaluate(inputs)?.into_matrices();
        let mut matrices = Vec::with_capacity(self.names.len());
        for _ in &self.names {
            let mut cells = Vec::with_capacity(n_rows * n_cols * n_cases);
            for _ in 0..n_cases {
                let (_, matrix) = flat.next().expect("one matrix per factor and case");
                cells.extend_from_slice(matrix.as_slice());
            }
            matrices.push(Matrix::from_column_major(cells, n_rows, n_cols * n_cases)?);
        }
        FactorValues::from_matrices(self.names.clone(), matrices)
    }
}

/// `source` with each `{name}` replaced by the case's value.
fn substitute(source: &str, case: &GridCase) -> std::result::Result<String, String> {
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        let close = after
            .find('}')
            .ok_or_else(|| format!("unclosed placeholder in {source:?}"))?;
        let name = after[..close].trim();
        let value = case
            .get(name)
            .ok_or_else(|| format!("no value for {{{name}}}"))?;
        out.push_str(value);
        rest = &after[close + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// A compile error of `name#k` as the factor's name and case `k`'s values.
fn locate(message: &str, cases: &[GridCase]) -> String {
    let Some(rest) = message.strip_prefix("factor \"") else {
        return message.to_owned();
    };
    let Some((flat, detail)) = rest.split_once("\": ") else {
        return message.to_owned();
    };
    match flat
        .rsplit_once('#')
        .and_then(|(name, k)| Some((name, k.parse::<usize>().ok()?)))
    {
        Some((name, k)) if k < cases.len() => {
            format!("factor {name:?} with {:?}: {detail}", cases[k])
        }
        _ => message.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(pairs: &[(&str, &str)]) -> GridCase {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect()
    }

    #[test]
    fn placeholders_are_substituted_anywhere() {
        let c = case(&[("n", "20"), ("field", "close")]);
        assert_eq!(
            substitute("Mean(${field}, {n})", &c).unwrap(),
            "Mean($close, 20)"
        );
        assert!(
            substitute("Mean($close, {m})", &c)
                .unwrap_err()
                .contains("{m}")
        );
        assert!(
            substitute("Mean($close, {n", &c)
                .unwrap_err()
                .contains("unclosed")
        );
    }

    #[test]
    fn errors_name_the_factor_and_case() {
        let cases = vec![case(&[("n", "5")]), case(&[("n", "2.5")])];
        let error = FactorGrid::compile([("ma", "Mean($close, {n})")], cases)
            .unwrap_err()
            .to_string();
        assert!(error.contains("\"ma\"") && error.contains("2.5"), "{error}");
    }
}
