use std::collections::BTreeSet;

/// Backend-neutral expression tree produced by [`crate::FactorSet::compile`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Expression {
    Number(u64),
    Field(String),
    Unary(char, Box<Self>),
    Binary(String, Box<Self>, Box<Self>),
    Call(String, Vec<Self>),
    /// A quoted literal: a time of day (`DayCumsum`) or an instrument
    /// (`ChangeInstrument`, Qlib's `Mask(x, "SH000300")`).
    Str(String),
    /// Python's `None`, as in Qlib's `Cut(x, 240, None)`.
    None,
}

impl Expression {
    pub(crate) fn number(value: f64) -> Self {
        Self::Number(value.to_bits())
    }

    /// The value of an expression made only of literals and arithmetic, as
    /// a literal argument (`Mean($close, 2*5)`, `Ref($close, -1)`) is.
    pub(crate) fn constant(&self) -> Option<f64> {
        match self {
            Self::Number(bits) => Some(f64::from_bits(*bits)),
            Self::Unary('-', child) => child.constant().map(|value| -value),
            Self::Unary(_, child) => child.constant(),
            Self::Binary(op, left, right) => {
                let op = crate::graph::binary_op(op)?;
                Some(crate::graph::binary(
                    op,
                    left.constant()?,
                    right.constant()?,
                ))
            }
            _ => None,
        }
    }

    pub(crate) fn fields<'a>(&'a self, out: &mut BTreeSet<&'a str>) {
        match self {
            Self::Field(name) => {
                out.insert(name);
            }
            Self::Unary(_, child) => child.fields(out),
            Self::Binary(_, left, right) => {
                left.fields(out);
                right.fields(out);
            }
            Self::Call(_, args) => {
                for arg in args {
                    arg.fields(out);
                }
            }
            Self::Number(_) | Self::Str(_) | Self::None => {}
        }
    }

    /// Whether a row's value can depend on later rows: a negative `Ref`,
    /// `BFillNan`, `DayLast` (the rest of the day; causal only when its input
    /// is lagged by at least a day, which the bar count decides), and `Cut`
    /// from the right (which rows are cut depends on how many come after).
    pub(crate) fn uses_future(&self) -> bool {
        let negative = |arg: Option<&Self>| {
            arg.and_then(Self::constant)
                .is_some_and(|value| value < 0.0)
        };
        match self {
            Self::Call(name, args) => {
                let reads_ahead = match name.as_str() {
                    "Ref" => negative(args.get(1)),
                    "Cut" => negative(args.get(2)),
                    "BFillNan" | "DayLast" => true,
                    _ => false,
                };
                reads_ahead || args.iter().any(Self::uses_future)
            }
            Self::Unary(_, child) => child.uses_future(),
            Self::Binary(_, left, right) => left.uses_future() || right.uses_future(),
            Self::Number(_) | Self::Field(_) | Self::Str(_) | Self::None => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f64),
    Field(String),
    Str(String),
    Name(String),
    Op(String),
    LParen,
    RParen,
    Comma,
    End,
}

pub(crate) fn parse(source: &str) -> std::result::Result<Expression, String> {
    if source.len() > 16_384 {
        return Err("expression exceeds 16384 bytes".into());
    }
    let tokens = lex(source)?;
    let mut parser = Parser { tokens, at: 0 };
    let expr = parser.expression(0, 0)?;
    if parser.peek() != &Token::End {
        return Err(format!("unexpected token {:?}", parser.peek()));
    }
    validate(&expr)?;
    Ok(expr)
}

/// A character of a `$field` name: Qlib's `\\w` (any letter, digit or `_`,
/// Unicode included) and the full-width punctuation `、：（）` it also allows.
fn field_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || matches!(c, '、' | '：' | '（' | '）')
}

fn lex(source: &str) -> std::result::Result<Vec<Token>, String> {
    let bytes = source.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b' ' | b'\t' | b'\r' | b'\n' => i += 1,
            b'(' => {
                out.push(Token::LParen);
                i += 1;
            }
            b')' => {
                out.push(Token::RParen);
                i += 1;
            }
            b',' => {
                out.push(Token::Comma);
                i += 1;
            }
            // Mathematics writes a power as `^` (Python as `**`).
            b'^' => {
                out.push(Token::Op("**".into()));
                i += 1;
            }
            b'+' | b'-' | b'*' | b'/' | b'>' | b'<' | b'=' | b'!' | b'&' | b'|' => {
                let start = i;
                i += 1;
                if i < bytes.len()
                    && matches!(
                        (bytes[start], bytes[i]),
                        (b'>', b'=') | (b'<', b'=') | (b'=', b'=') | (b'!', b'=') | (b'*', b'*')
                    )
                {
                    i += 1;
                }
                if i == start + 1 && matches!(bytes[start], b'=' | b'!') {
                    return Err(format!(
                        "operator {:?} must be followed by '='",
                        bytes[start] as char
                    ));
                }
                out.push(Token::Op(source[start..i].to_owned()));
            }
            b'$' => {
                // `$$name` is a point-in-time field, kept as `$name`.
                let pit = bytes.get(i + 1) == Some(&b'$');
                let start = i + 1 + usize::from(pit);
                let rest = &source[start..];
                i = start
                    + rest
                        .char_indices()
                        .find(|&(_, c)| !field_char(c))
                        .map_or(rest.len(), |(at, _)| at);
                if i == start {
                    return Err("expected a field name after '$'".into());
                }
                let name = source[start..i].to_lowercase();
                out.push(Token::Field(if pit { format!("${name}") } else { name }));
            }
            quote @ (b'\'' | b'"') => {
                let start = i + 1;
                let Some(length) = bytes[start..].iter().position(|&b| b == quote) else {
                    return Err(format!("unterminated string starting at byte {i}"));
                };
                out.push(Token::Str(source[start..start + length].to_owned()));
                i = start + length + 1;
            }
            c if c.is_ascii_digit() || c == b'.' => {
                let start = i;
                i += 1;
                while i < bytes.len()
                    && (bytes[i].is_ascii_digit()
                        || matches!(bytes[i], b'.' | b'e' | b'E' | b'+' | b'-'))
                {
                    if matches!(bytes[i], b'+' | b'-') && !matches!(bytes[i - 1], b'e' | b'E') {
                        break;
                    }
                    i += 1;
                }
                let value = source[start..i]
                    .parse::<f64>()
                    .map_err(|_| format!("invalid number {:?}", &source[start..i]))?;
                out.push(Token::Number(value));
            }
            c if c.is_ascii_alphabetic() || c == b'_' => {
                let start = i;
                i += 1;
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                out.push(Token::Name(source[start..i].to_owned()));
            }
            other => {
                return Err(format!(
                    "unexpected character {:?} at byte {i}",
                    other as char
                ));
            }
        }
    }
    out.push(Token::End);
    Ok(out)
}

struct Parser {
    tokens: Vec<Token>,
    at: usize,
}

impl Parser {
    fn peek(&self) -> &Token {
        &self.tokens[self.at]
    }
    fn take(&mut self) -> Token {
        let token = self.tokens[self.at].clone();
        self.at += 1;
        token
    }

    fn expression(&mut self, min_bp: u8, depth: usize) -> std::result::Result<Expression, String> {
        if depth > 128 {
            return Err("expression nesting exceeds 128".into());
        }
        let mut left = match self.take() {
            Token::Number(value) => Expression::number(value),
            Token::Field(name) => Expression::Field(name),
            Token::Str(text) => Expression::Str(text),
            Token::Name(name) if name == "None" && self.peek() != &Token::LParen => {
                Expression::None
            }
            Token::Name(name)
                if (name == "True" || name == "False") && self.peek() != &Token::LParen =>
            {
                Expression::number(f64::from(name == "True"))
            }
            Token::Op(op) if op == "+" || op == "-" => Expression::Unary(
                op.as_bytes()[0] as char,
                Box::new(self.expression(11, depth + 1)?),
            ),
            Token::LParen => {
                let inner = self.expression(0, depth + 1)?;
                if self.take() != Token::RParen {
                    return Err("expected ')'".into());
                }
                inner
            }
            Token::Name(name) => {
                if self.take() != Token::LParen {
                    return Err(format!("expected '(' after {name}"));
                }
                let mut args = Vec::new();
                if self.peek() != &Token::RParen {
                    loop {
                        args.push(self.expression(0, depth + 1)?);
                        if self.peek() != &Token::Comma {
                            break;
                        }
                        self.take();
                    }
                }
                if self.take() != Token::RParen {
                    return Err(format!("expected ')' after {name} arguments"));
                }
                Expression::Call(name, args)
            }
            token => return Err(format!("expected expression, found {token:?}")),
        };

        // The right operand of the last comparison of an unbroken chain:
        // `a < b < c` is `(a < b) & (b < c)`, as in mathematics.
        let mut chain: Option<Expression> = None;
        while let Token::Op(op) = self.peek() {
            let op = op.clone();
            let (left_bp, right_bp) = match op.as_str() {
                "|" => (1, 2),
                "&" => (3, 4),
                ">" | "<" | ">=" | "<=" | "==" | "!=" => (5, 6),
                "+" | "-" => (7, 8),
                "*" | "/" => (9, 10),
                "**" => (12, 11),
                _ => break,
            };
            if left_bp < min_bp {
                break;
            }
            self.take();
            let right = self.expression(right_bp, depth + 1)?;
            let comparison = left_bp == 5;
            left = match chain.take() {
                Some(middle) if comparison => Expression::Binary(
                    "&".into(),
                    Box::new(left),
                    Box::new(Expression::Binary(
                        op,
                        Box::new(middle),
                        Box::new(right.clone()),
                    )),
                ),
                _ => Expression::Binary(op, Box::new(left), Box::new(right.clone())),
            };
            if comparison {
                chain = Some(right);
            }
        }
        Ok(left)
    }
}

fn validate(expr: &Expression) -> std::result::Result<(), String> {
    if let Expression::Call(name, args) = expr {
        let arity = match name.as_str() {
            "Abs" | "Sign" | "Log" | "Not" | "Exp" | "Sqrt" | "IsNan" | "CSRank" | "CSZScore"
            | "CSDemean" | "IsNull" | "IsInf" | "FFillNan" | "BFillNan" | "DayLast" | "Date"
            | "P" => 1..=1,
            "Greater" | "Less" | "Power" | "Add" | "Sub" | "Mul" | "Div" | "Gt" | "Ge" | "Lt"
            | "Le" | "Eq" | "Ne" | "And" | "Or" | "FillNa" | "Mask" | "SignedPower"
            | "IndNeutralize" | "CSResi" | "Select" | "ChangeInstrument" | "PRef" => 2..=2,
            "Scale" => 1..=2,
            "Cut" => 1..=3,
            "DayCumsum" => 1..=4,
            "If" | "Clip" => 3..=3,
            "Ref" | "Mean" | "Std" | "Sum" | "Max" | "Min" | "Rank" | "IdxMax" | "IdxMin"
            | "Slope" | "Rsquare" | "Resi" | "Var" | "Skew" | "Kurt" | "Med" | "Mad" | "Count"
            | "Delta" | "WMA" | "EMA" | "DecayLinear" | "Prod" => 2..=2,
            "Quantile" | "Corr" | "Cov" => 3..=3,
            _ => return Err(format!("unsupported operator {name:?}")),
        };
        if !arity.contains(&args.len()) {
            return Err(format!("{name} received {} arguments", args.len()));
        }
    }
    match expr {
        Expression::Unary(_, child) => validate(child),
        Expression::Binary(_, left, right) => {
            validate(left)?;
            validate(right)
        }
        Expression::Call(_, args) => {
            for arg in args {
                validate(arg)?;
            }
            Ok(())
        }
        Expression::Number(_) | Expression::Field(_) | Expression::Str(_) | Expression::None => {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precedence_and_scientific_numbers_parse() {
        let expr = parse("($close-Ref($close, 1))/(2*$close+1e-12)").unwrap();
        assert!(matches!(expr, Expression::Binary(ref op, _, _) if op == "/"));
    }

    #[test]
    fn qlib_strings_none_and_pit_fields_parse() {
        let expr = parse("DayCumsum($volume, '9:45', \"14:44\")").unwrap();
        let Expression::Call(_, args) = expr else {
            panic!("a call")
        };
        assert_eq!(args[1], Expression::Str("9:45".into()));
        assert_eq!(args[2], Expression::Str("14:44".into()));
        let Expression::Call(_, args) = parse("Cut($close, 240, None)").unwrap() else {
            panic!("a call")
        };
        assert_eq!(args[2], Expression::None);
        assert_eq!(
            parse("P($$ROEWA_q)").unwrap(),
            Expression::Call("P".into(), vec![Expression::Field("$roewa_q".into())])
        );
        assert!(
            parse("Mask($close, 'SH000300)")
                .unwrap_err()
                .contains("unterminated")
        );
        assert!(parse("$$").unwrap_err().contains("field name"));
    }

    #[test]
    fn rejects_unknown_calls() {
        assert!(parse("Magic($close)").unwrap_err().contains("unsupported"));
    }
}
