//! Subexpressions that are not computed from a symbol's own rows, lifted out
//! of the factors into hidden fields before the graph is built.
//!
//! * `ChangeInstrument("X", e)` and Qlib's `Mask(e, "X")` evaluate `e` over
//!   instrument `X` and hand every symbol that one series: the field `@iK`,
//!   one value per row, broadcast across symbols. `X` is a symbol of the
//!   panel or an instrument supplied with [`crate::FactorInputs::with_instrument`].
//! * `P(e)` and `PRef(e, period)` evaluate `e` over point-in-time financial
//!   data (`$$field`) as it was known on each row's day: the field `@pK`, one
//!   value per row and symbol. See [`crate::pit`].
//!
//! The rest of the factor then reads `@iK` or `@pK` like any field, so the
//! graph, its stages and its schedule are unchanged. `@` cannot start a `$`
//! field, so a hidden field never collides with an input.

use std::collections::HashMap;

use alphax_core::error::{Error, Result};

use crate::FactorSet;
use crate::graph::{integer, number};
use crate::parse::Expression;
use crate::pit;

/// The expressions one instrument is asked for, compiled as one set whose
/// factors are named after the hidden fields they fill.
#[derive(Debug, Clone)]
pub(crate) struct Instrument {
    pub(crate) name: String,
    pub(crate) set: FactorSet,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct Lifted {
    pub(crate) instruments: Vec<Instrument>,
    pub(crate) pits: Vec<pit::Job>,
}

impl Lifted {
    pub(crate) fn is_empty(&self) -> bool {
        self.instruments.is_empty() && self.pits.is_empty()
    }

    /// Every instrument this set, or a set it lifts, reads.
    pub(crate) fn instrument_names(&self, out: &mut Vec<String>) {
        for instrument in &self.instruments {
            if !out.contains(&instrument.name) {
                out.push(instrument.name.clone());
            }
            instrument.set.lifted.instrument_names(out);
        }
    }

    /// The factors with every lifted subexpression replaced by its hidden
    /// field, and what fills those fields.
    pub(crate) fn lift(factors: &[(String, Expression)]) -> Result<(Vec<Expression>, Self)> {
        let mut lifter = Lifter::default();
        let rewritten = factors
            .iter()
            .map(|(name, expr)| lifter.rewrite(expr).map_err(|error| named(name, error)))
            .collect::<Result<Vec<_>>>()?;
        let mut instruments = Vec::new();
        for (name, members) in lifter.instruments {
            instruments.push(Instrument {
                name,
                set: FactorSet::build(members)?,
            });
        }
        Ok((
            rewritten,
            Self {
                instruments,
                pits: lifter.pits,
            },
        ))
    }
}

/// `error` attributed to factor `name`, in place of the hidden factor a
/// lifted set reported it under.
fn named(name: &str, error: Error) -> Error {
    match error {
        Error::Invariant(message) => {
            let message = match message.strip_prefix("factor \"@") {
                Some(rest) => rest
                    .split_once("\": ")
                    .map_or(message.clone(), |(_, m)| m.to_owned()),
                None => message,
            };
            if message.starts_with("factor \"") {
                Error::Invariant(message)
            } else {
                Error::Invariant(format!("factor {name:?}: {message}"))
            }
        }
        other => other,
    }
}

#[derive(Default)]
struct Lifter {
    /// Per instrument, in first-use order: `(hidden field, expression)`.
    instruments: Vec<(String, Vec<(String, Expression)>)>,
    pits: Vec<pit::Job>,
    seen: HashMap<Expression, String>,
    count: usize,
}

impl Lifter {
    fn rewrite(&mut self, expr: &Expression) -> Result<Expression> {
        Ok(match expr {
            Expression::Call(name, args) => match (name.as_str(), args.as_slice()) {
                ("ChangeInstrument", [instrument, inner])
                | ("Mask", [inner, instrument @ Expression::Str(_)]) => {
                    let Expression::Str(instrument) = instrument else {
                        return Err(Error::Invariant(
                            "ChangeInstrument's instrument must be a quoted string".into(),
                        ));
                    };
                    self.instrument(instrument, inner)?
                }
                ("P", [inner]) => self.pit(expr, inner, None)?,
                ("PRef", [inner, period]) => {
                    let period = integer(period, "PRef's period")?;
                    let period = u32::try_from(period)
                        .map_err(|_| Error::Invariant("PRef's period must be positive".into()))?;
                    self.pit(expr, inner, Some(period))?
                }
                _ => Expression::Call(
                    name.clone(),
                    args.iter()
                        .map(|arg| self.rewrite(arg))
                        .collect::<Result<_>>()?,
                ),
            },
            Expression::Unary(op, child) => Expression::Unary(*op, Box::new(self.rewrite(child)?)),
            Expression::Binary(op, left, right) => Expression::Binary(
                op.clone(),
                Box::new(self.rewrite(left)?),
                Box::new(self.rewrite(right)?),
            ),
            Expression::Field(name) if name.starts_with('$') => {
                return Err(Error::Invariant(format!(
                    "$${} is point-in-time data: read it inside P() or PRef()",
                    &name[1..]
                )));
            }
            other => other.clone(),
        })
    }

    fn hidden(&mut self, prefix: char) -> String {
        self.count += 1;
        format!("@{prefix}{}", self.count - 1)
    }

    fn instrument(&mut self, instrument: &str, inner: &Expression) -> Result<Expression> {
        // `Mask(e, "X")` and `ChangeInstrument("X", e)` are one series.
        let call = &Expression::Call(
            "ChangeInstrument".into(),
            vec![Expression::Str(instrument.to_owned()), inner.clone()],
        );
        if let Some(field) = self.seen.get(call) {
            return Ok(Expression::Field(field.clone()));
        }
        let field = self.hidden('i');
        let member = (field.clone(), inner.clone());
        match self
            .instruments
            .iter_mut()
            .find(|(name, _)| name == instrument)
        {
            Some((_, members)) => members.push(member),
            None => self.instruments.push((instrument.to_owned(), vec![member])),
        }
        self.seen.insert(call.clone(), field.clone());
        Ok(Expression::Field(field))
    }

    fn pit(
        &mut self,
        call: &Expression,
        inner: &Expression,
        period: Option<u32>,
    ) -> Result<Expression> {
        if let Some(field) = self.seen.get(call) {
            return Ok(Expression::Field(field.clone()));
        }
        let mut fields = Vec::new();
        let renamed = point_in_time(inner, &mut fields)?;
        let quarterly = match fields.first() {
            None => return Err(Error::Invariant("P() must read a $$ field".into())),
            Some(first) => pit::quarterly(first)?,
        };
        for field in &fields {
            if pit::quarterly(field)? != quarterly {
                return Err(Error::Invariant(
                    "P() cannot mix quarterly ($$..._q) and annual ($$..._a) fields".into(),
                ));
            }
        }
        let (left, right) = extent(inner)?;
        if right > 0 {
            return Err(Error::Invariant(
                "P() cannot read a future period (a negative Ref)".into(),
            ));
        }
        let set = FactorSet::build(vec![("@p".to_owned(), renamed)])?;
        let field = self.hidden('p');
        self.pits.push(pit::Job {
            field: field.clone(),
            fields,
            set,
            window: left + 1,
            period,
            quarterly,
        });
        self.seen.insert(call.clone(), field.clone());
        Ok(Expression::Field(field))
    }
}

/// `expr` with each `$$field` renamed to a plain field, collecting the names;
/// everything else must be something a period series can be computed with.
fn point_in_time(expr: &Expression, fields: &mut Vec<String>) -> Result<Expression> {
    Ok(match expr {
        Expression::Field(name) => {
            let Some(name) = name.strip_prefix('$') else {
                return Err(Error::Invariant(format!(
                    "P() reads only $$ fields, not ${name}"
                )));
            };
            if !fields.iter().any(|field| field == name) {
                fields.push(name.to_owned());
            }
            Expression::Field(name.to_owned())
        }
        Expression::Number(_) => expr.clone(),
        Expression::Unary(op, child) => {
            Expression::Unary(*op, Box::new(point_in_time(child, fields)?))
        }
        Expression::Binary(op, left, right) => Expression::Binary(
            op.clone(),
            Box::new(point_in_time(left, fields)?),
            Box::new(point_in_time(right, fields)?),
        ),
        Expression::Call(name, args) => {
            // `extent` rejects what has no meaning over periods.
            extent(expr)?;
            Expression::Call(
                name.clone(),
                args.iter()
                    .map(|arg| point_in_time(arg, fields))
                    .collect::<Result<_>>()?,
            )
        }
        Expression::Str(_) | Expression::None => {
            return Err(Error::Invariant(
                "P() takes no string or None argument".into(),
            ));
        }
    })
}

/// Qlib's `get_extended_window_size`: how many periods before (and after)
/// the latest one an expression reads. `P` loads exactly the latest
/// `left + 1` periods, so this is part of its result: `Delta(x, 1)` extends
/// by `N - 1 = 0` as every rolling operator does and is therefore `NaN`
/// under `P`, as it is in Qlib.
pub(crate) fn extent(expr: &Expression) -> Result<(usize, usize)> {
    let max = |a: (usize, usize), b: (usize, usize)| (a.0.max(b.0), a.1.max(b.1));
    Ok(match expr {
        Expression::Number(_) | Expression::Field(_) => (0, 0),
        Expression::Unary(_, child) => extent(child)?,
        Expression::Binary(_, left, right) => max(extent(left)?, extent(right)?),
        Expression::Str(_) | Expression::None => (0, 0),
        Expression::Call(name, args) => match name.as_str() {
            "Abs" | "Sign" | "Log" | "Not" | "Exp" | "Sqrt" | "IsNan" | "IsNull" | "IsInf"
            | "Add" | "Sub" | "Mul" | "Div" | "Power" | "Gt" | "Ge" | "Lt" | "Le" | "Eq" | "Ne"
            | "And" | "Or" | "Greater" | "Less" | "FillNa" | "SignedPower" | "If" | "Clip" => args
                .iter()
                .try_fold((0, 0), |acc, arg| Ok::<_, Error>(max(acc, extent(arg)?)))?,
            "Mask" if !matches!(args.get(1), Some(Expression::Str(_))) => {
                max(extent(&args[0])?, extent(&args[1])?)
            }
            "Ref" => {
                let (left, right) = extent(&args[0])?;
                let offset = integer(&args[1], "Ref offset")?;
                let shift = offset.unsigned_abs() as usize;
                match offset.signum() {
                    1 => (left + shift, right),
                    -1 => (left, right + shift),
                    _ => (left, right),
                }
            }
            "Mean" | "Sum" | "Std" | "Var" | "Skew" | "Kurt" | "Max" | "Min" | "IdxMax"
            | "IdxMin" | "Med" | "Mad" | "Rank" | "Count" | "Delta" | "Slope" | "Rsquare"
            | "Resi" | "WMA" | "Quantile" | "DecayLinear" | "Prod" => {
                let (left, right) = extent(&args[0])?;
                let window = integer(&args[1], "window")?.unsigned_abs() as usize;
                (left + window.saturating_sub(1), right)
            }
            "EMA" => {
                let (left, right) = extent(&args[0])?;
                let parameter = number(&args[1], "EMA parameter")?;
                let span = if parameter > 0.0 && parameter < 1.0 {
                    // (1 - alpha)^size == 1e-6
                    ((1e-6f64).ln() / (1.0 - parameter).ln()) as usize
                } else if parameter.fract() == 0.0 && parameter >= 0.0 {
                    parameter as usize
                } else {
                    return Err(Error::Invariant(
                        "inside P(), an EMA span of 1 or more must be an integer".into(),
                    ));
                };
                (left + span.saturating_sub(1), right)
            }
            "Corr" | "Cov" => {
                let window = integer(&args[2], "window")?.unsigned_abs() as usize;
                if window == 0 {
                    return Err(Error::Invariant(format!(
                        "an expanding {name} cannot be read inside P()"
                    )));
                }
                let (left, right) = max(extent(&args[0])?, extent(&args[1])?);
                (left + window - 1, right)
            }
            _ => {
                return Err(Error::Invariant(format!(
                    "{name} cannot be read inside P()"
                )));
            }
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse;

    fn extent_of(source: &str) -> (usize, usize) {
        extent(&parse(source).unwrap()).unwrap()
    }

    #[test]
    fn extents_follow_qlib() {
        assert_eq!(extent_of("$$roe_q"), (0, 0));
        assert_eq!(extent_of("Mean($$roe_q, 4)"), (3, 0));
        assert_eq!(extent_of("Ref($$roe_q, 1)"), (1, 0));
        assert_eq!(extent_of("Delta($$roe_q, 1)"), (0, 0));
        assert_eq!(extent_of("Mean(Ref($$a_q, 2), 3) / $$b_q"), (4, 0));
        assert_eq!(extent_of("Mean($$roe_q, 0)"), (0, 0));
        assert_eq!(extent_of("Ref($$roe_q, -1)"), (0, 1));
        assert_eq!(extent_of("EMA($$roe_q, 0.5)"), (18, 0));
    }

    #[test]
    fn lifted_calls_are_shared_and_misplaced_pit_fields_fail() {
        let factors = [
            ("a", "P($$roe_q) + ChangeInstrument('IDX', $close)"),
            ("b", "P($$roe_q) * Mask($close, 'IDX')"),
        ]
        .map(|(name, source)| (name.to_owned(), parse(source).unwrap()));
        let (rewritten, lifted) = Lifted::lift(&factors).unwrap();
        assert_eq!(lifted.pits.len(), 1);
        assert_eq!(lifted.instruments.len(), 1);
        assert_eq!(lifted.instruments[0].set.names().count(), 1);
        let mut fields = std::collections::BTreeSet::new();
        rewritten[1].fields(&mut fields);
        assert_eq!(fields.into_iter().collect::<Vec<_>>(), ["@i1", "@p0"]);

        let bare = [("f".to_owned(), parse("$$roe_q + 1").unwrap())];
        let error = Lifted::lift(&bare).unwrap_err().to_string();
        assert!(
            error.contains("inside P()") && error.contains("\"f\""),
            "{error}"
        );
        let mixed = [("f".to_owned(), parse("P($$roe_q / $$eps_a)").unwrap())];
        assert!(
            Lifted::lift(&mixed)
                .unwrap_err()
                .to_string()
                .contains("mix")
        );
        let daily = [("f".to_owned(), parse("P($$roe_q / $close)").unwrap())];
        assert!(
            Lifted::lift(&daily)
                .unwrap_err()
                .to_string()
                .contains("only $$")
        );
        let ahead = [("f".to_owned(), parse("P(Ref($$roe_q, -1))").unwrap())];
        assert!(
            Lifted::lift(&ahead)
                .unwrap_err()
                .to_string()
                .contains("future")
        );
    }
}
