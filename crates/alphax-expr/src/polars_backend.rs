use std::sync::Arc;

use alphax_core::error::{Error, Result};
use alphax_core::matrix::Matrix;
use polars::lazy::dsl::{Expr as PolarsExpr, GetOutput, map_multiple, rolling_cov};
use polars::prelude::*;

use crate::eval::{
    arg_extreme, corr, ema_expanding, kurt, mad, quantile, rank_last, regression, skew, wma,
};
use crate::{ComputeBackend, Expression, FactorInputs, FactorSet, FactorValues};

const SYMBOL_COLUMN: &str = "__alphax_expr_symbol";
const FACTOR_BATCH_SIZE: usize = 64;

/// Polars Lazy compute backend for offline batch factor evaluation.
#[derive(Debug, Clone, Copy, Default)]
pub struct PolarsBackend;

impl ComputeBackend for PolarsBackend {
    fn name(&self) -> &'static str {
        "polars"
    }

    fn evaluate(&self, set: &FactorSet, inputs: &FactorInputs<'_>) -> Result<FactorValues> {
        let (n_rows, n_cols) = inputs.shape();
        let plan = set
            .expressions()
            .map(|(name, expression)| Ok(by_symbol(compile(expression, n_rows)?).alias(name)))
            .collect::<Result<Vec<_>>>()?;
        let names = set.names().map(str::to_owned).collect::<Vec<_>>();
        if inputs.fields().any(|(name, _)| name == SYMBOL_COLUMN) {
            return Err(Error::Invariant(format!(
                "factor input field {SYMBOL_COLUMN:?} is reserved by the Polars backend"
            )));
        }
        let mut columns = inputs
            .fields()
            .map(|(name, values)| Series::new(name.into(), values).into_column())
            .collect::<Vec<_>>();
        let symbols = (0..n_cols)
            .flat_map(|symbol| std::iter::repeat_n(symbol as u32, n_rows))
            .collect::<Vec<_>>();
        columns.push(Series::new(SYMBOL_COLUMN.into(), symbols).into_column());
        let input = DataFrame::new(columns)
            .map_err(|error| backend_error("building input frame", error))?;
        let mut matrices = Vec::with_capacity(names.len());
        for (expressions, batch_names) in plan
            .chunks(FACTOR_BATCH_SIZE)
            .zip(names.chunks(FACTOR_BATCH_SIZE))
        {
            let frame = input
                .clone()
                .lazy()
                .select(expressions)
                .collect()
                .map_err(|error| backend_error("collecting factor plan", error))?;
            for name in batch_names {
                let values = frame
                    .column(name)
                    .and_then(|column| column.f64())
                    .map_err(|error| backend_error("reading factor output", error))?
                    .into_iter()
                    .map(|value| value.unwrap_or(f64::NAN))
                    .collect();
                matrices.push(Matrix::from_column_major(values, n_rows, n_cols)?);
            }
        }
        FactorValues::from_matrices(names, matrices)
    }
}

fn backend_error(action: &str, error: PolarsError) -> Error {
    Error::Invariant(format!("polars backend failed while {action}: {error}"))
}

fn compile(expression: &Expression, n_rows: usize) -> Result<PolarsExpr> {
    Ok(match expression {
        Expression::Number(bits) => lit(f64::from_bits(*bits)),
        Expression::Field(name) => col(name),
        Expression::Unary('+', child) => compile(child, n_rows)?,
        Expression::Unary('-', child) => -compile(child, n_rows)?,
        Expression::Unary(operator, _) => {
            return Err(Error::Invariant(format!(
                "unsupported unary operator {operator}"
            )));
        }
        Expression::Binary(operator, left, right) => {
            binary(operator, compile(left, n_rows)?, compile(right, n_rows)?)?
        }
        Expression::Call(name, arguments) => call(name, arguments, n_rows)?,
        Expression::Str(_) | Expression::None => {
            return Err(Error::Invariant(
                "unsupported operator: string and None arguments".into(),
            ));
        }
    })
}

fn binary(operator: &str, left: PolarsExpr, right: PolarsExpr) -> Result<PolarsExpr> {
    Ok(match operator {
        "+" => left + right,
        "-" => left - right,
        "*" => left * right,
        "/" => left / right,
        "**" => left.pow(right),
        ">" | "<" | ">=" | "<=" | "==" | "!=" => compare(operator, left, right),
        "&" => numeric_bool(truthy(left).and(truthy(right))),
        "|" => numeric_bool(truthy(left).or(truthy(right))),
        _ => {
            return Err(Error::Invariant(format!(
                "unsupported binary operator {operator}"
            )));
        }
    })
}

fn call(name: &str, arguments: &[Expression], n_rows: usize) -> Result<PolarsExpr> {
    Ok(match name {
        "Abs" => compile(&arguments[0], n_rows)?.abs(),
        "Sign" => compile(&arguments[0], n_rows)?.sign(),
        "Log" => compile(&arguments[0], n_rows)?.log(std::f64::consts::E),
        "Not" => numeric_bool(compile(&arguments[0], n_rows)?.eq(lit(0.0))),
        "Power" | "Add" | "Sub" | "Mul" | "Div" | "Gt" | "Ge" | "Lt" | "Le" | "Eq" | "Ne"
        | "And" | "Or" => {
            let operator = match name {
                "Power" => "**",
                "Add" => "+",
                "Sub" => "-",
                "Mul" => "*",
                "Div" => "/",
                "Gt" => ">",
                "Ge" => ">=",
                "Lt" => "<",
                "Le" => "<=",
                "Eq" => "==",
                "Ne" => "!=",
                "And" => "&",
                "Or" => "|",
                _ => unreachable!(),
            };
            binary(
                operator,
                compile(&arguments[0], n_rows)?,
                compile(&arguments[1], n_rows)?,
            )?
        }
        "If" => when(truthy(compile(&arguments[0], n_rows)?))
            .then(compile(&arguments[1], n_rows)?)
            .otherwise(compile(&arguments[2], n_rows)?),
        "Greater" | "Less" => extreme(
            compile(&arguments[0], n_rows)?,
            compile(&arguments[1], n_rows)?,
            name == "Greater",
        ),
        "Ref" => {
            let source = compile(&arguments[0], n_rows)?;
            let offset = integer(&arguments[1], "Ref offset")?;
            if offset == 0 {
                source.first()
            } else {
                source.shift(lit(offset)).fill_null(lit(f64::NAN))
            }
        }
        "Delta" => {
            let source = compile(&arguments[0], n_rows)?;
            let offset = integer(&arguments[1], "Delta offset")?;
            if offset < 0 {
                return Err(Error::Invariant("Delta offset cannot be negative".into()));
            }
            if offset == 0 {
                source.clone() - source.first()
            } else {
                source.clone() - source.shift(lit(offset)).fill_null(lit(f64::NAN))
            }
        }
        "Mean" | "Std" | "Sum" | "Max" | "Min" | "Var" | "Med" | "Count"
            if (0.0..1.0).contains(&number(&arguments[1], "window")?)
                && number(&arguments[1], "window")? > 0.0 =>
        {
            ewm(
                compile(&arguments[0], n_rows)?,
                number(&arguments[1], "window")?,
            )
        }
        "Mean" | "Std" | "Sum" | "Max" | "Min" | "Quantile" | "Rank" | "IdxMax" | "IdxMin"
        | "Slope" | "Rsquare" | "Resi" | "Var" | "Skew" | "Kurt" | "Med" | "Mad" | "Count"
        | "WMA" => {
            let source = compile(&arguments[0], n_rows)?;
            let window = rolling_window(&arguments[1], n_rows)?;
            let parameter = if name == "Quantile" {
                number(&arguments[2], "quantile")?
            } else {
                0.0
            };
            rolling(name, source, window, parameter)?
        }
        "EMA" => {
            let source = compile(&arguments[0], n_rows)?;
            let parameter = number(&arguments[1], "EMA parameter")?;
            if parameter < 0.0 {
                return Err(Error::Invariant("EMA parameter cannot be negative".into()));
            }
            if parameter == 0.0 {
                expanding_ema(source)
            } else {
                let alpha = if parameter < 1.0 {
                    parameter
                } else {
                    2.0 / (parameter + 1.0)
                };
                ewm(source, alpha)
            }
        }
        "Corr" | "Cov" => {
            let left = compile(&arguments[0], n_rows)?.fill_nan(lit(NULL));
            let right = compile(&arguments[1], n_rows)?.fill_nan(lit(NULL));
            let window = rolling_window(&arguments[2], n_rows)?;
            let options = RollingCovOptions {
                window_size: window as IdxSize,
                min_periods: 1,
                ddof: 1,
            };
            (if name == "Corr" {
                rolling_corr_stable(left, right, window)
            } else {
                rolling_cov(left, right, options)
            })
            .fill_null(lit(f64::NAN))
        }
        _ => return Err(Error::Invariant(format!("unsupported operator {name}"))),
    })
}

/// pandas' `ewm(alpha, adjust=True, min_periods=1).mean()`. Polars leaves a
/// missing row empty; pandas carries the mean so far, which the forward
/// fill restores (the weights already decayed across the gap).
fn ewm(source: PolarsExpr, alpha: f64) -> PolarsExpr {
    source
        .fill_nan(lit(NULL))
        .ewm_mean(EWMOptions {
            alpha,
            adjust: true,
            bias: false,
            min_periods: 1,
            ignore_nulls: false,
        })
        .forward_fill(None)
        .fill_null(lit(f64::NAN))
}

fn rolling_corr_stable(left: PolarsExpr, right: PolarsExpr, window: usize) -> PolarsExpr {
    map_multiple(
        move |columns| {
            let values = columns
                .iter()
                .map(|column| {
                    Ok(column
                        .f64()?
                        .into_iter()
                        .map(|value| value.unwrap_or(f64::NAN))
                        .collect::<Vec<_>>())
                })
                .take(2)
                .collect::<PolarsResult<Vec<_>>>()?;
            let symbols = columns[2].u32()?;
            let mut symbol_start = 0;
            let output = (0..values[0].len())
                .map(|row| {
                    if row > 0 && symbols.get(row) != symbols.get(row - 1) {
                        symbol_start = row;
                    }
                    let start = (row + 1).saturating_sub(window).max(symbol_start);
                    corr(&values[0][start..=row], &values[1][start..=row])
                })
                .collect::<Vec<_>>();
            Ok(Some(
                Series::new(columns[0].name().clone(), output).into_column(),
            ))
        },
        [left, right, col(SYMBOL_COLUMN)],
        GetOutput::from_type(DataType::Float64),
    )
}

fn numeric_bool(expression: PolarsExpr) -> PolarsExpr {
    expression.cast(DataType::Float64)
}

fn compare(operator: &str, left: PolarsExpr, right: PolarsExpr) -> PolarsExpr {
    let invalid = left.clone().is_nan().or(right.clone().is_nan());
    let comparison = match operator {
        ">" => left.gt(right),
        "<" => left.lt(right),
        ">=" => left.gt_eq(right),
        "<=" => left.lt_eq(right),
        "==" => left.eq(right),
        "!=" => left.neq(right),
        _ => unreachable!(),
    };
    when(invalid)
        .then(lit(f64::from(operator == "!=")))
        .otherwise(numeric_bool(comparison))
}

fn truthy(expression: PolarsExpr) -> PolarsExpr {
    expression.neq(lit(0.0))
}

fn by_symbol(expression: PolarsExpr) -> PolarsExpr {
    expression.over([col(SYMBOL_COLUMN)])
}

fn extreme(left: PolarsExpr, right: PolarsExpr, maximum: bool) -> PolarsExpr {
    let invalid = left.clone().is_nan().or(right.clone().is_nan());
    let comparison = if maximum {
        left.clone().gt(right.clone())
    } else {
        left.clone().lt(right.clone())
    };
    when(invalid)
        .then(lit(f64::NAN))
        .when(comparison)
        .then(left)
        .otherwise(right)
}

fn rolling(name: &str, source: PolarsExpr, window: usize, parameter: f64) -> Result<PolarsExpr> {
    let options = RollingOptionsFixedWindow {
        window_size: window,
        min_periods: 1,
        ..Default::default()
    };
    let nullable = source.clone().fill_nan(lit(NULL));
    let expression = match name {
        "Mean" => nullable.rolling_mean(options),
        "Std" => nullable.rolling_std(options),
        "Sum" => nullable.rolling_sum(options),
        "Max" => nullable.rolling_max(options),
        "Min" => nullable.rolling_min(options),
        "Var" => nullable.rolling_var(options),
        "Quantile" => {
            if !(0.0..=1.0).contains(&parameter) {
                return Err(Error::Invariant("quantile must be in 0..=1".into()));
            }
            nullable.rolling_quantile(QuantileMethod::Linear, parameter, options)
        }
        "Med" => nullable.rolling_median(options),
        _ => custom_rolling(source.fill_null(lit(f64::NAN)), name, parameter, options),
    };
    Ok(expression.fill_null(lit(f64::NAN)))
}

fn custom_rolling(
    source: PolarsExpr,
    name: &str,
    parameter: f64,
    options: RollingOptionsFixedWindow,
) -> PolarsExpr {
    let name = name.to_owned();
    source.rolling_map(
        Arc::new(move |series| {
            let values = series
                .f64()
                .expect("factor rolling input is always Float64")
                .into_iter()
                .map(|value| value.unwrap_or(f64::NAN))
                .collect::<Vec<_>>();
            let value = match name.as_str() {
                "Rank" => rank_last(&values),
                "IdxMax" => arg_extreme(&values, false),
                "IdxMin" => arg_extreme(&values, true),
                "Slope" => regression(&values).0,
                "Rsquare" => regression(&values).1,
                "Resi" => regression(&values).2,
                "Skew" => skew(&values),
                "Kurt" => kurt(&values),
                "Mad" => mad(&values),
                "Count" => values.iter().filter(|value| !value.is_nan()).count() as f64,
                "WMA" => wma(&values),
                "Quantile" => quantile(&values, parameter),
                _ => f64::NAN,
            };
            Series::new(series.name().clone(), [value])
        }),
        GetOutput::from_type(DataType::Float64),
        options,
    )
}

fn expanding_ema(source: PolarsExpr) -> PolarsExpr {
    source.map(
        |column| {
            let series = column.as_materialized_series();
            let values = series
                .f64()?
                .into_iter()
                .map(|value| value.unwrap_or(f64::NAN))
                .collect::<Vec<_>>();
            Ok(Some(
                Series::new(series.name().clone(), ema_expanding(&values)).into_column(),
            ))
        },
        GetOutput::from_type(DataType::Float64),
    )
}

fn number(expression: &Expression, label: &str) -> Result<f64> {
    expression
        .constant()
        .ok_or_else(|| Error::Invariant(format!("{label} must be a numeric literal")))
}

fn integer(expression: &Expression, label: &str) -> Result<i64> {
    let value = number(expression, label)?;
    if value.fract() != 0.0 || !value.is_finite() {
        return Err(Error::Invariant(format!("{label} must be an integer")));
    }
    Ok(value as i64)
}

fn rolling_window(expression: &Expression, n_rows: usize) -> Result<usize> {
    let value = integer(expression, "window")?;
    if value < 0 {
        return Err(Error::Invariant("window cannot be negative".into()));
    }
    Ok(if value == 0 {
        n_rows.max(1)
    } else {
        value as usize
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NativeBackend, alpha158};
    use alphax_model::OhlcvPanel;

    #[test]
    fn native_and_polars_match_alpha158() {
        let n_rows = 70;
        let n_cols = 3;
        let series = |base: f64, trend: f64, wave: f64| {
            (0..n_cols)
                .flat_map(|column| {
                    (0..n_rows).map(move |row| {
                        base + column as f64 * 7.0
                            + row as f64 * trend
                            + ((row * 17 + column * 11) % 13) as f64 * wave
                    })
                })
                .collect::<Vec<_>>()
        };
        let panel = OhlcvPanel {
            open: series(100.0, 0.10, 0.03),
            high: series(101.0, 0.12, 0.04),
            low: series(99.0, 0.08, 0.02),
            close: series(100.5, 0.11, 0.05),
            volume: series(1_000.0, 3.0, 0.7),
            n_rows,
            n_cols,
        };
        let vwap = series(100.4, 0.105, 0.025);
        let inputs = FactorInputs::from_ohlcv(&panel)
            .unwrap()
            .with_field("vwap", &vwap)
            .unwrap();
        let factors = alpha158().unwrap();
        let native = factors.evaluate_with(&inputs, &NativeBackend).unwrap();
        let polars = factors.evaluate_with(&inputs, &PolarsBackend).unwrap();

        for ((name, expected), (actual_name, actual)) in native.iter().zip(polars.iter()) {
            assert_eq!(name, actual_name);
            for (index, (&expected, &actual)) in expected
                .as_slice()
                .iter()
                .zip(actual.as_slice())
                .enumerate()
            {
                if expected.is_nan() && actual.is_nan() {
                    continue;
                }
                let tolerance = 1e-8 * expected.abs().max(1.0);
                assert!(
                    (expected - actual).abs() <= tolerance,
                    "{name}[{index}]: native={expected:?}, polars={actual:?}"
                );
            }
        }
    }
}
