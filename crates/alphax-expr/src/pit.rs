//! Point-in-time financial data: `P(e)` and `PRef(e, period)`.
//!
//! A point-in-time field (`$$roe_q`) is a list of records per symbol: on day
//! `d`, period `p` (a quarter `YYYYQ` as `YYYY * 100 + Q`, or a year) was
//! published or revised to `v`. On each row, `P(e)` evaluates `e` over the
//! periods as they were known on that row's day and keeps the latest period's
//! value, as Qlib's `P` does:
//!
//! 1. The known records are those dated on or before the row's day.
//! 2. The periods run from the earliest to the latest known period, every
//!    quarter (or year) in between, so a missing quarter is `NaN`.
//! 3. Only the latest `left + 1` of them are kept, where `left` is how far
//!    back `e` reads ([`crate::lift::extent`]). Qlib loads exactly this
//!    window, and operators with `min_periods = 1` see nothing older.
//! 4. Each period's value is its latest revision among the known records.
//!
//! `PRef(e, p)` evaluates `e` over period `p` alone.
//!
//! The known records change only on publication days, so `e` is evaluated
//! once per distinct state a row can see, not once per row. States of every
//! symbol with the same number of periods are evaluated together, as the
//! columns of one panel.
//!
//! Fields of one `P` share one period axis. Qlib gives each field its own
//! axis and aligns them by period after each operator; the two agree unless
//! the fields' earliest or latest known periods differ.

use std::collections::BTreeMap;

use alphax_core::error::{Error, Result};

use crate::{FactorInputs, FactorSet};

/// One record of a point-in-time field: on `day` (days since 1970-01-01) the
/// value of `period` for symbol `col` was published, or revised, as `value`.
/// Records of one day apply in the order given.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PitRecord {
    pub col: usize,
    pub day: i64,
    pub period: u32,
    pub value: f64,
}

/// One lifted `P` or `PRef`.
#[derive(Debug, Clone)]
pub(crate) struct Job {
    /// The hidden field it fills.
    pub(crate) field: String,
    /// The point-in-time fields it reads, without `$`.
    pub(crate) fields: Vec<String>,
    /// The expression over those fields, as one factor.
    pub(crate) set: FactorSet,
    /// The latest periods it reads.
    pub(crate) window: usize,
    /// `PRef`'s period.
    pub(crate) period: Option<u32>,
    pub(crate) quarterly: bool,
}

/// Whether a point-in-time field is quarterly (`_q`) or annual (`_a`); Qlib
/// names them so, and the name is what says how periods follow each other.
pub(crate) fn quarterly(field: &str) -> Result<bool> {
    if field.ends_with("_q") {
        Ok(true)
    } else if field.ends_with("_a") {
        Ok(false)
    } else {
        Err(Error::Invariant(format!(
            "point-in-time field $${field} must end with _q (quarterly) or _a (annual)"
        )))
    }
}

/// Rejects a period that is not a quarter `YYYY * 100 + Q` (or a year) of
/// 1900..=2099.
pub(crate) fn check_period(field: &str, period: u32) -> Result<()> {
    let valid = if quarterly(field)? {
        (190_001..=209_904).contains(&period) && (1..=4).contains(&(period % 100))
    } else {
        (1900..=2099).contains(&period)
    };
    if valid {
        Ok(())
    } else {
        Err(Error::Invariant(format!(
            "invalid period {period} for $${field}"
        )))
    }
}

/// Every quarter (or year) from `first` to `last`.
fn periods(first: u32, last: u32, quarterly: bool) -> Vec<u32> {
    if !quarterly {
        return (first..=last).collect();
    }
    (first / 100..=last / 100)
        .flat_map(|year| (1..=4).map(move |quarter| year * 100 + quarter))
        .filter(|period| (first..=last).contains(period))
        .collect()
}

/// A state to evaluate: symbol `col` as of its `state`-th publication day,
/// with each field's values over the period axis.
struct Task {
    col: usize,
    state: usize,
    values: Vec<Vec<f64>>,
}

/// The job's value on every row of every symbol, column-major.
pub(crate) fn evaluate(job: &Job, inputs: &FactorInputs<'_>) -> Result<Vec<f64>> {
    let (n_rows, n_cols) = inputs.shape();
    let days = inputs.days().ok_or_else(|| {
        Error::Invariant(
            "P() and PRef() need the bar timestamps (FactorInputs::with_timestamps)".into(),
        )
    })?;
    // Per field, per symbol: its records in publication order.
    let mut records: Vec<Vec<Vec<PitRecord>>> = Vec::with_capacity(job.fields.len());
    for field in &job.fields {
        let all = inputs.pit_records(field).ok_or_else(|| {
            Error::Invariant(format!(
                "factor input is missing point-in-time field $${field}"
            ))
        })?;
        let mut by_col = vec![Vec::new(); n_cols];
        for record in all {
            by_col[record.col].push(*record);
        }
        for column in &mut by_col {
            column.sort_by_key(|record| record.day);
        }
        records.push(by_col);
    }

    // Each row's state: how many publication days of its symbol it has seen.
    let mut row_states = vec![0usize; n_rows * n_cols];
    let mut by_len: BTreeMap<usize, Vec<Task>> = BTreeMap::new();
    let mut n_states = vec![0usize; n_cols];
    for col in 0..n_cols {
        let mut dates: Vec<i64> = records
            .iter()
            .flat_map(|field| field[col].iter().map(|record| record.day))
            .collect();
        dates.sort_unstable();
        dates.dedup();
        n_states[col] = dates.len();
        let mut seen = 0;
        let mut used = Vec::new();
        for (row, &day) in days.iter().enumerate() {
            while seen < dates.len() && dates[seen] as f64 <= day {
                seen += 1;
            }
            row_states[col * n_rows + row] = seen;
            if seen > 0 && used.last() != Some(&seen) {
                used.push(seen);
            }
        }
        for state in used {
            let day = dates[state - 1];
            let known: Vec<&[PitRecord]> = records
                .iter()
                .map(|field| {
                    let column = &field[col];
                    &column[..column.partition_point(|record| record.day <= day)]
                })
                .collect();
            let axis = match job.period {
                Some(period) => vec![period],
                None => {
                    let span = known.iter().flat_map(|k| k.iter().map(|r| r.period));
                    let (Some(first), Some(last)) = (span.clone().min(), span.max()) else {
                        continue;
                    };
                    let mut axis = periods(first, last, job.quarterly);
                    axis.drain(..axis.len().saturating_sub(job.window));
                    axis
                }
            };
            let values = known
                .iter()
                .map(|known| {
                    let mut values = vec![f64::NAN; axis.len()];
                    for record in *known {
                        if let Ok(at) = axis.binary_search(&record.period) {
                            values[at] = record.value;
                        }
                    }
                    values
                })
                .collect();
            by_len
                .entry(axis.len())
                .or_default()
                .push(Task { col, state, values });
        }
    }

    // `value[col][state]`; state 0 (nothing published yet) stays NaN.
    let mut value: Vec<Vec<f64>> = n_states.iter().map(|&n| vec![f64::NAN; n + 1]).collect();
    for (len, tasks) in by_len {
        let columns: Vec<Vec<f64>> = (0..job.fields.len())
            .map(|field| {
                tasks
                    .iter()
                    .flat_map(|task| task.values[field].iter().copied())
                    .collect()
            })
            .collect();
        let mut panel = FactorInputs::new(len, tasks.len())?;
        for (name, values) in job.fields.iter().zip(&columns) {
            panel = panel.with_field(name, values)?;
        }
        let result = job.set.evaluate(&panel)?;
        let (_, matrix) = result.iter().next().expect("a job is one factor");
        for (k, task) in tasks.iter().enumerate() {
            value[task.col][task.state] = matrix.as_slice()[k * len + len - 1];
        }
    }
    Ok(row_states
        .iter()
        .enumerate()
        .map(|(cell, &state)| value[cell / n_rows][state])
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn periods_fill_missing_quarters_and_years() {
        assert_eq!(
            periods(201_903, 202_002, true),
            [201_903, 201_904, 202_001, 202_002]
        );
        assert_eq!(periods(2017, 2019, false), [2017, 2018, 2019]);
    }

    #[test]
    fn periods_are_checked_by_the_field_suffix() {
        assert!(check_period("roe_q", 201_904).is_ok());
        assert!(check_period("roe_q", 201_905).is_err());
        assert!(check_period("eps_a", 2019).is_ok());
        assert!(check_period("eps_a", 201_901).is_err());
        assert!(quarterly("roe").is_err());
    }
}
