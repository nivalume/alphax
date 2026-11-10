//! A factor set compiled once into a deduplicated dependency graph.
//!
//! [`crate::FactorSet::compile`] turns every expression tree into nodes of one
//! shared graph:
//!
//! * **Hash-consed.** A node is identified by its operator and its operand
//!   nodes, so `Ref($close, 1)` appearing in forty Alpha158 factors, or `a+b`
//!   next to `Add(a, b)`, is one node computed once.
//! * **Validated.** Window lengths, offsets and other literal parameters are
//!   checked here, so a bad factor fails at compile time rather than after the
//!   rest of the set has been evaluated.
//! * **Constant-folded.** Elementwise operators whose operands are all
//!   literals become literals.
//! * **Scheduled.** Nodes are stored in dependency order, and every computed
//!   node that a later node reads gets a scratch slot that is reused once the
//!   node's last reader has run. A symbol's whole factor set therefore needs
//!   only [`Graph::slots`] series of scratch, however many factors and
//!   intermediates it has. A node nothing reads is written straight into its
//!   factor's output and takes no slot.
//! * **Shifted views.** `Ref(x, k)` with `k >= 0` is not computed at all: it
//!   is a view of `x` displaced by `k` rows, and elementwise operators read it
//!   through the offset. Alpha360 is 354 such nodes over six fields, so this
//!   is the difference between one pass per factor and three.
//!
//! Every operator but the cross-sectional ones ([`Op::Cross`]) reads only its
//! own symbol, so a graph without them is evaluated one symbol (column) at a
//! time, symbols in parallel. A graph with them is split into stages at those
//! nodes by [`crate::stage`]: columns in parallel within a stage, one
//! cross-sectional pass between stages.

use std::collections::HashMap;

use alphax_core::error::{Error, Result};

use crate::parse::Expression;

/// Index of a node in [`Graph::nodes`].
pub(crate) type NodeId = usize;

/// The calendar day of each row (days since 1970-01-01, wall clock), a
/// hidden field supplied by [`crate::FactorInputs::with_timestamps`]. `@`
/// cannot start a `$` field, so it never collides with an input.
pub(crate) const DAY: &str = "@day";
/// The minute of the day of each row, as [`DAY`].
pub(crate) const MINUTE: &str = "@minute";

/// A window argument; `0` in the source means the whole series.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Window {
    Fixed(usize),
    All,
}

impl Window {
    pub(crate) fn resolve(self, n_rows: usize) -> usize {
        match self {
            Self::Fixed(window) => window,
            Self::All => n_rows.max(1),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum UnaryOp {
    Neg,
    Abs,
    Sign,
    Log,
    Not,
    Exp,
    Sqrt,
    IsNan,
    IsInf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Gt,
    Lt,
    Ge,
    Le,
    Eq,
    Ne,
    And,
    Or,
    Greater,
    Less,
    /// The left operand where it is not `NaN`, else the right.
    FillNa,
    /// The left operand where the right is true (non-zero, not `NaN`), else `NaN`.
    Mask,
    /// `sign(a) · |a|^b`.
    SignedPower,
}

/// Per-symbol rolling reductions over a trailing window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Rolling {
    Mean,
    Std,
    Var,
    Sum,
    Max,
    Min,
    /// The quantile's bits, so the operator stays hashable.
    Quantile(u64),
    Med,
    Mad,
    Count,
    Skew,
    Kurt,
    Wma,
    Rank,
    IdxMax,
    IdxMin,
    Slope,
    Rsquare,
    Resi,
    /// WorldQuant's `decay_linear`: weights `1..=window`, newest heaviest,
    /// normalized over the values present.
    Decay,
    Prod,
}

/// Operators whose row reads every symbol's value on that row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Cross {
    /// Percentile rank, ties averaged, `NaN` excluded (pandas
    /// `rank(pct=True)`).
    Rank,
    /// `(x - mean) / std`, sample standard deviation.
    ZScore,
    /// `x - mean`.
    Demean,
    /// `x · a / Σ|x|`; the scale's bits.
    Scale(u64),
    /// `x` minus the mean of `x` over the symbols in the same group.
    Neutralize,
    /// The residual of a least-squares fit of `y` on `x` across symbols.
    Resi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Op {
    /// Index into [`Graph::fields`].
    Field(usize),
    /// A literal's bits.
    Const(u64),
    Unary(UnaryOp),
    Binary(BinaryOp),
    If,
    Ref(i64),
    /// The first row's value on every row (Qlib's `Ref(x, 0)`).
    First,
    Delta(usize),
    /// `None` is the expanding EMA (`EMA(x, 0)`), otherwise the alpha's bits.
    Ema(Option<u64>),
    Rolling(Rolling, Window),
    Pair {
        corr: bool,
        window: Window,
    },
    Cross(Cross),
    /// `FFillNan` (forward) or `BFillNan`.
    Fill {
        forward: bool,
    },
    /// The last value present in each day; inputs `[x, @day]`.
    DayLast,
    /// The running sum over the bars of each day between two times of day
    /// (minutes), zero outside them; inputs `[x, @minute, @day]`. A bar
    /// counts from the one that contains `start`: `step` is the bar length.
    DayCumsum {
        start: u32,
        end: u32,
        step: u32,
    },
    /// `NaN` over the first `left` and the last `right` rows.
    Cut {
        left: usize,
        right: usize,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct Node {
    pub(crate) op: Op,
    pub(crate) inputs: Vec<NodeId>,
    /// Scratch slot for a computed node that a later node reads; `None` for
    /// fields, literals, views and nodes only an output consumes.
    pub(crate) slot: Option<usize>,
    /// Output positions that take this node's value.
    pub(crate) outputs: Vec<usize>,
    /// Whether any later node reads this one.
    pub(crate) read: bool,
    /// For a shifted view: the node it displaces and by how many rows.
    pub(crate) view: Option<(NodeId, usize)>,
}

#[derive(Debug, Clone)]
pub(crate) struct Graph {
    /// In dependency order: every node's inputs come before it.
    pub(crate) nodes: Vec<Node>,
    /// The node each factor reads, in factor order.
    pub(crate) outputs: Vec<NodeId>,
    /// Input field names, indexed by [`Op::Field`].
    pub(crate) fields: Vec<String>,
    /// Scratch series needed per symbol.
    pub(crate) slots: usize,
}

impl Graph {
    /// Compiles named expressions (already parsed and arity-checked) in
    /// output order.
    pub(crate) fn compile<'e>(
        factors: impl IntoIterator<Item = (&'e str, &'e Expression)>,
    ) -> Result<Self> {
        let mut builder = Builder::default();
        let mut outputs = Vec::new();
        for (name, expr) in factors {
            let node = builder.node(expr).map_err(|error| match error {
                Error::Invariant(message) => {
                    Error::Invariant(format!("factor {name:?}: {message}"))
                }
                other => other,
            })?;
            outputs.push(node);
        }
        Ok(builder.finish(outputs))
    }

    /// Resolves views, then assigns each computed node a scratch slot,
    /// reusing a slot once the node that held it has had its last reader.
    fn schedule(&mut self) {
        for at in 0..self.nodes.len() {
            let Op::Ref(offset) = self.nodes[at].op else {
                continue;
            };
            let Ok(offset) = usize::try_from(offset) else {
                continue; // reads the future: computed
            };
            let input = self.nodes[at].inputs[0];
            self.nodes[at].view = match (self.nodes[input].op, self.nodes[input].view) {
                (Op::Const(_), _) => None,
                (_, Some((base, shift))) => Some((base, shift + offset)),
                _ => Some((input, offset)),
            };
        }
        // A read of a view is a read of the series it displaces.
        let base = |nodes: &[Node], id: NodeId| nodes[id].view.map_or(id, |(base, _)| base);
        let mut last_read = vec![None; self.nodes.len()];
        for at in 0..self.nodes.len() {
            for i in 0..self.nodes[at].inputs.len() {
                let input = self.nodes[at].inputs[i];
                self.nodes[input].read = true;
                last_read[base(&self.nodes, input)] = Some(at);
            }
        }
        for (position, &node) in self.outputs.iter().enumerate() {
            self.nodes[node].outputs.push(position);
        }
        let mut free = Vec::new();
        let mut slots = 0;
        for at in 0..self.nodes.len() {
            let computed = !matches!(self.nodes[at].op, Op::Field(_) | Op::Const(_))
                && self.nodes[at].view.is_none();
            // Allocate before releasing this node's inputs, so a node never
            // writes into a slot it is reading. A node nothing reads is only
            // an output, written as it is computed, and needs no slot.
            if computed && self.nodes[at].read {
                let slot = free.pop().unwrap_or_else(|| {
                    slots += 1;
                    slots - 1
                });
                self.nodes[at].slot = Some(slot);
            }
            let mut released = Vec::new();
            for i in 0..self.nodes[at].inputs.len() {
                let input = base(&self.nodes, self.nodes[at].inputs[i]);
                if last_read[input] == Some(at) && !released.contains(&input) {
                    released.push(input);
                    if let Some(slot) = self.nodes[input].slot {
                        free.push(slot);
                    }
                }
            }
        }
        self.slots = slots;
    }
}

#[derive(Default)]
pub(crate) struct Builder {
    nodes: Vec<Node>,
    interned: HashMap<(Op, Vec<NodeId>), NodeId>,
    fields: Vec<String>,
}

impl Builder {
    /// The graph of the nodes interned so far, with `outputs` in factor
    /// order, scheduled.
    pub(crate) fn finish(self, outputs: Vec<NodeId>) -> Graph {
        let mut graph = Graph {
            nodes: self.nodes,
            outputs,
            fields: self.fields,
            slots: 0,
        };
        graph.schedule();
        graph
    }

    /// The node reading field `name`.
    pub(crate) fn field(&mut self, name: &str) -> NodeId {
        let index = match self.fields.iter().position(|field| field == name) {
            Some(index) => index,
            None => {
                self.fields.push(name.to_owned());
                self.fields.len() - 1
            }
        };
        self.intern(Op::Field(index), Vec::new())
    }

    pub(crate) fn intern(&mut self, op: Op, inputs: Vec<NodeId>) -> NodeId {
        if let Some(folded) = self.fold(op, &inputs) {
            return self.intern(Op::Const(folded.to_bits()), Vec::new());
        }
        let key = (op, inputs);
        if let Some(&id) = self.interned.get(&key) {
            return id;
        }
        let id = self.nodes.len();
        self.nodes.push(Node {
            op,
            inputs: key.1.clone(),
            slot: None,
            outputs: Vec::new(),
            read: false,
            view: None,
        });
        self.interned.insert(key, id);
        id
    }

    /// The value of an elementwise operator over literal operands.
    fn fold(&self, op: Op, inputs: &[NodeId]) -> Option<f64> {
        let literal = |id: NodeId| match self.nodes[id].op {
            Op::Const(bits) => Some(f64::from_bits(bits)),
            _ => None,
        };
        let values: Option<Vec<f64>> = inputs.iter().map(|&id| literal(id)).collect();
        let values = values?;
        match op {
            Op::Unary(op) => Some(unary(op, values[0])),
            Op::Binary(op) => Some(binary(op, values[0], values[1])),
            Op::If => Some(if_else(values[0], values[1], values[2])),
            _ => None,
        }
    }

    fn node(&mut self, expr: &Expression) -> Result<NodeId> {
        Ok(match expr {
            Expression::Number(bits) => self.intern(Op::Const(*bits), Vec::new()),
            Expression::Field(name) => self.field(name),
            Expression::Unary('+', child) => self.node(child)?,
            Expression::Unary('-', child) => {
                let child = self.node(child)?;
                self.intern(Op::Unary(UnaryOp::Neg), vec![child])
            }
            Expression::Unary(op, _) => {
                return Err(Error::Invariant(format!("unsupported unary {op}")));
            }
            Expression::Binary(op, left, right) => {
                let op = binary_op(op)
                    .ok_or_else(|| Error::Invariant(format!("unsupported operator {op}")))?;
                let inputs = vec![self.node(left)?, self.node(right)?];
                self.intern(Op::Binary(op), inputs)
            }
            Expression::Call(name, args) => self.call(name, args)?,
            Expression::Str(text) => {
                return Err(Error::Invariant(format!(
                    "the string {text:?} is only an argument of DayCumsum, ChangeInstrument or Mask"
                )));
            }
            Expression::None => {
                return Err(Error::Invariant("None is only an argument of Cut".into()));
            }
        })
    }

    fn call(&mut self, name: &str, args: &[Expression]) -> Result<NodeId> {
        let unary = match name {
            "Abs" => Some(UnaryOp::Abs),
            "Sign" => Some(UnaryOp::Sign),
            "Log" => Some(UnaryOp::Log),
            "Not" => Some(UnaryOp::Not),
            "Exp" => Some(UnaryOp::Exp),
            "Sqrt" => Some(UnaryOp::Sqrt),
            "IsNan" | "IsNull" => Some(UnaryOp::IsNan),
            "IsInf" => Some(UnaryOp::IsInf),
            _ => None,
        };
        if let Some(op) = unary {
            let input = self.node(&args[0])?;
            return Ok(self.intern(Op::Unary(op), vec![input]));
        }
        let binary = match name {
            "Power" => Some(BinaryOp::Pow),
            "Add" => Some(BinaryOp::Add),
            "Sub" => Some(BinaryOp::Sub),
            "Mul" => Some(BinaryOp::Mul),
            "Div" => Some(BinaryOp::Div),
            "Gt" => Some(BinaryOp::Gt),
            "Ge" => Some(BinaryOp::Ge),
            "Lt" => Some(BinaryOp::Lt),
            "Le" => Some(BinaryOp::Le),
            "Eq" => Some(BinaryOp::Eq),
            "Ne" => Some(BinaryOp::Ne),
            "And" => Some(BinaryOp::And),
            "Or" => Some(BinaryOp::Or),
            "Greater" => Some(BinaryOp::Greater),
            "Less" => Some(BinaryOp::Less),
            "FillNa" => Some(BinaryOp::FillNa),
            "Mask" => Some(BinaryOp::Mask),
            "SignedPower" => Some(BinaryOp::SignedPower),
            _ => None,
        };
        if let Some(op) = binary {
            let inputs = vec![self.node(&args[0])?, self.node(&args[1])?];
            return Ok(self.intern(Op::Binary(op), inputs));
        }
        let rolling = match name {
            "Mean" => Some(Rolling::Mean),
            "Std" => Some(Rolling::Std),
            "Var" => Some(Rolling::Var),
            "Sum" => Some(Rolling::Sum),
            "Max" => Some(Rolling::Max),
            "Min" => Some(Rolling::Min),
            "Quantile" => {
                let q = number(&args[2], "quantile")?;
                if !(0.0..=1.0).contains(&q) {
                    return Err(Error::Invariant("quantile must be in 0..=1".into()));
                }
                Some(Rolling::Quantile(q.to_bits()))
            }
            "Med" => Some(Rolling::Med),
            "Mad" => Some(Rolling::Mad),
            "Count" => Some(Rolling::Count),
            "Skew" => Some(Rolling::Skew),
            "Kurt" => Some(Rolling::Kurt),
            "WMA" => Some(Rolling::Wma),
            "Rank" => Some(Rolling::Rank),
            "IdxMax" => Some(Rolling::IdxMax),
            "IdxMin" => Some(Rolling::IdxMin),
            "Slope" => Some(Rolling::Slope),
            "Rsquare" => Some(Rolling::Rsquare),
            "Resi" => Some(Rolling::Resi),
            "DecayLinear" => Some(Rolling::Decay),
            "Prod" => Some(Rolling::Prod),
            _ => None,
        };
        if let Some(kind) = rolling {
            // Qlib rejects these windows when it builds the operator.
            let least = match name {
                "Skew" => 3.0,
                "Kurt" => 4.0,
                _ => 0.0,
            };
            let value = number(&args[1], "window")?;
            if least > 0.0 && value != 0.0 && value < least {
                return Err(Error::Invariant(format!(
                    "{name}'s window must be 0 or at least {least}"
                )));
            }
            // Qlib's base rolling operators read a window strictly between
            // 0 and 1 as an exponential mean with that alpha, whatever the
            // operator (`Std($close, 0.5)` too).
            if matches!(
                name,
                "Mean" | "Sum" | "Std" | "Var" | "Max" | "Min" | "Med" | "Count"
            ) {
                let value = number(&args[1], "window")?;
                if value > 0.0 && value < 1.0 {
                    let input = self.node(&args[0])?;
                    return Ok(self.intern(Op::Ema(Some(value.to_bits())), vec![input]));
                }
            }
            let window = window(&args[1])?;
            let input = self.node(&args[0])?;
            return Ok(self.intern(Op::Rolling(kind, window), vec![input]));
        }
        let cross = match name {
            "CSRank" => Some(Cross::Rank),
            "CSZScore" => Some(Cross::ZScore),
            "CSDemean" => Some(Cross::Demean),
            "Scale" => {
                let scale = args
                    .get(1)
                    .map_or(Ok(1.0), |a| number(a, "Scale's scale"))?;
                Some(Cross::Scale(scale.to_bits()))
            }
            "IndNeutralize" => Some(Cross::Neutralize),
            "CSResi" => Some(Cross::Resi),
            _ => None,
        };
        if let Some(op) = cross {
            let n = if matches!(op, Cross::Neutralize | Cross::Resi) {
                2
            } else {
                1
            };
            let inputs = args[..n]
                .iter()
                .map(|arg| self.node(arg))
                .collect::<Result<Vec<_>>>()?;
            return Ok(self.intern(Op::Cross(op), inputs));
        }
        Ok(match name {
            // Qlib drops the rows where the condition is false; on a panel
            // they stay, as NaN.
            "Select" => {
                let inputs = vec![self.node(&args[1])?, self.node(&args[0])?];
                self.intern(Op::Binary(BinaryOp::Mask), inputs)
            }
            "FFillNan" | "BFillNan" => {
                let input = self.node(&args[0])?;
                self.intern(
                    Op::Fill {
                        forward: name == "FFillNan",
                    },
                    vec![input],
                )
            }
            // Qlib's Date returns the bar's date whatever its input.
            "Date" => self.field(DAY),
            "DayLast" => {
                let inputs = vec![self.node(&args[0])?, self.field(DAY)];
                self.intern(Op::DayLast, inputs)
            }
            "DayCumsum" => {
                let start = args.get(1).map_or(Ok(9 * 60 + 30), time_of_day)?;
                let end = args.get(2).map_or(Ok(14 * 60 + 59), time_of_day)?;
                let step = args
                    .get(3)
                    .map_or(Ok(1), |arg| integer(arg, "DayCumsum granularity"))?;
                let step = u32::try_from(step)
                    .ok()
                    .filter(|&step| step > 0)
                    .ok_or_else(|| {
                        Error::Invariant("DayCumsum granularity must be positive".into())
                    })?;
                let inputs = vec![self.node(&args[0])?, self.field(MINUTE), self.field(DAY)];
                self.intern(Op::DayCumsum { start, end, step }, inputs)
            }
            "Cut" => {
                let bound = |at: usize, label: &str| -> Result<i64> {
                    match args.get(at) {
                        None | Some(Expression::None) => Ok(0),
                        Some(arg) => integer(arg, label),
                    }
                };
                let (left, right) = (bound(1, "Cut's left")?, bound(2, "Cut's right")?);
                // Qlib's rule: a left cut is positive, a right cut negative.
                if left < 0 || right > 0 {
                    return Err(Error::Invariant(
                        "Cut's left must be positive and its right negative".into(),
                    ));
                }
                let input = self.node(&args[0])?;
                if left == 0 && right == 0 {
                    return Ok(input);
                }
                self.intern(
                    Op::Cut {
                        left: left.unsigned_abs() as usize,
                        right: right.unsigned_abs() as usize,
                    },
                    vec![input],
                )
            }
            "ChangeInstrument" | "P" | "PRef" => {
                return Err(Error::Invariant(format!(
                    "{name} must be resolved before the graph is built"
                )));
            }
            "Clip" => {
                // Less(Greater(x, lo), hi): NaN stays NaN, and a floor above
                // the cap gives the cap, as NumPy's clip does.
                let inputs = vec![self.node(&args[0])?, self.node(&args[1])?];
                let floored = self.intern(Op::Binary(BinaryOp::Greater), inputs);
                let cap = self.node(&args[2])?;
                self.intern(Op::Binary(BinaryOp::Less), vec![floored, cap])
            }
            "If" => {
                let inputs = vec![
                    self.node(&args[0])?,
                    self.node(&args[1])?,
                    self.node(&args[2])?,
                ];
                self.intern(Op::If, inputs)
            }
            // Qlib: `Ref(x, 0)` is the first row's value, `Delta(x, 0)` the
            // change since it.
            "Ref" => {
                let offset = integer(&args[1], "Ref offset")?;
                let input = self.node(&args[0])?;
                let op = if offset == 0 {
                    Op::First
                } else {
                    Op::Ref(offset)
                };
                self.intern(op, vec![input])
            }
            "Delta" => {
                let offset = integer(&args[1], "Delta offset")?;
                let offset = usize::try_from(offset)
                    .map_err(|_| Error::Invariant("Delta offset cannot be negative".into()))?;
                let input = self.node(&args[0])?;
                if offset == 0 {
                    let first = self.intern(Op::First, vec![input]);
                    return Ok(self.intern(Op::Binary(BinaryOp::Sub), vec![input, first]));
                }
                self.intern(Op::Delta(offset), vec![input])
            }
            "EMA" => {
                let parameter = number(&args[1], "EMA parameter")?;
                if parameter < 0.0 {
                    return Err(Error::Invariant("EMA parameter cannot be negative".into()));
                }
                let alpha = (parameter > 0.0).then(|| {
                    if parameter < 1.0 {
                        parameter
                    } else {
                        2.0 / (parameter + 1.0)
                    }
                });
                let input = self.node(&args[0])?;
                self.intern(Op::Ema(alpha.map(f64::to_bits)), vec![input])
            }
            "Corr" | "Cov" => {
                let window = window(&args[2])?;
                let inputs = vec![self.node(&args[0])?, self.node(&args[1])?];
                self.intern(
                    Op::Pair {
                        corr: name == "Corr",
                        window,
                    },
                    inputs,
                )
            }
            _ => return Err(Error::Invariant(format!("unsupported operator {name}"))),
        })
    }
}

/// The operator an infix symbol denotes.
pub(crate) fn binary_op(symbol: &str) -> Option<BinaryOp> {
    Some(match symbol {
        "+" => BinaryOp::Add,
        "-" => BinaryOp::Sub,
        "*" => BinaryOp::Mul,
        "/" => BinaryOp::Div,
        "**" => BinaryOp::Pow,
        ">" => BinaryOp::Gt,
        "<" => BinaryOp::Lt,
        ">=" => BinaryOp::Ge,
        "<=" => BinaryOp::Le,
        "==" => BinaryOp::Eq,
        "!=" => BinaryOp::Ne,
        "&" => BinaryOp::And,
        "|" => BinaryOp::Or,
        _ => return None,
    })
}

/// A literal parameter: a number, or arithmetic over numbers (`2*5`), as
/// Python would have computed it before Qlib saw it.
pub(crate) fn number(expr: &Expression, label: &str) -> Result<f64> {
    expr.constant()
        .ok_or_else(|| Error::Invariant(format!("{label} must be a numeric literal")))
}

pub(crate) fn integer(expr: &Expression, label: &str) -> Result<i64> {
    let value = number(expr, label)?;
    if value.fract() != 0.0 || !value.is_finite() {
        return Err(Error::Invariant(format!("{label} must be an integer")));
    }
    Ok(value as i64)
}

/// `"H:MM"` as minutes after midnight.
fn time_of_day(expr: &Expression) -> Result<u32> {
    let Expression::Str(text) = expr else {
        return Err(Error::Invariant(
            "a DayCumsum time must be a string like '9:30'".into(),
        ));
    };
    let parsed = text.split_once(':').and_then(|(hours, minutes)| {
        let (hours, minutes) = (
            hours.trim().parse::<u32>().ok()?,
            minutes.trim().parse::<u32>().ok()?,
        );
        (hours < 24 && minutes < 60).then_some(hours * 60 + minutes)
    });
    parsed.ok_or_else(|| Error::Invariant(format!("invalid time of day {text:?}")))
}

fn window(expr: &Expression) -> Result<Window> {
    let value = integer(expr, "window")?;
    match usize::try_from(value) {
        Ok(0) => Ok(Window::All),
        Ok(window) => Ok(Window::Fixed(window)),
        Err(_) => Err(Error::Invariant("window cannot be negative".into())),
    }
}

#[allow(
    clippy::float_cmp,
    reason = "Qlib's Not is an exact comparison with zero"
)]
pub(crate) fn unary(op: UnaryOp, value: f64) -> f64 {
    match op {
        UnaryOp::Neg => -value,
        UnaryOp::Abs => value.abs(),
        UnaryOp::Sign => {
            if value == 0.0 {
                0.0
            } else {
                value.signum()
            }
        }
        UnaryOp::Log => value.ln(),
        UnaryOp::Not => f64::from(value == 0.0),
        UnaryOp::Exp => value.exp(),
        UnaryOp::Sqrt => value.sqrt(),
        UnaryOp::IsNan => f64::from(value.is_nan()),
        UnaryOp::IsInf => f64::from(value.is_infinite()),
    }
}

#[allow(
    clippy::float_cmp,
    reason = "Qlib's Eq/Ne and numeric boolean operators use exact NumPy comparisons"
)]
pub(crate) fn binary(op: BinaryOp, a: f64, b: f64) -> f64 {
    match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => a - b,
        BinaryOp::Mul => a * b,
        BinaryOp::Div => a / b,
        BinaryOp::Pow => a.powf(b),
        BinaryOp::Gt => f64::from(a > b),
        BinaryOp::Lt => f64::from(a < b),
        BinaryOp::Ge => f64::from(a >= b),
        BinaryOp::Le => f64::from(a <= b),
        BinaryOp::Eq => f64::from(a == b),
        BinaryOp::Ne => f64::from(a != b),
        BinaryOp::And => f64::from(a != 0.0 && b != 0.0),
        BinaryOp::Or => f64::from(a != 0.0 || b != 0.0),
        BinaryOp::Greater => {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        }
        BinaryOp::Less => {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.min(b)
            }
        }
        BinaryOp::FillNa => {
            if a.is_nan() {
                b
            } else {
                a
            }
        }
        BinaryOp::Mask => {
            if b.is_nan() || b == 0.0 {
                f64::NAN
            } else {
                a
            }
        }
        BinaryOp::SignedPower => {
            if a == 0.0 {
                0.0
            } else {
                a.signum() * a.abs().powf(b)
            }
        }
    }
}

pub(crate) fn if_else(condition: f64, yes: f64, no: f64) -> f64 {
    if condition != 0.0 { yes } else { no }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse;

    fn graph(sources: &[&str]) -> Result<Graph> {
        let parsed: Vec<Expression> = sources.iter().map(|s| parse(s).unwrap()).collect();
        Graph::compile(parsed.iter().map(|expr| ("f", expr)))
    }

    #[test]
    fn shared_subexpressions_and_spellings_are_one_node() {
        let g = graph(&[
            "Ref($close,1)/$close",
            "Div(Ref($close,1), $close)",
            "+Ref($close, 1)",
        ])
        .unwrap();
        // $close, Ref, Div.
        assert_eq!(g.nodes.len(), 3);
        assert_eq!(g.outputs[0], g.outputs[1]);
        assert_eq!(g.nodes[g.outputs[1]].outputs, [0, 1]);
    }

    #[test]
    fn literals_fold_and_take_no_slot() {
        let g = graph(&["-(1 + 2) * 3", "If(1 > 2, 4, 5)"]).unwrap();
        let value = |id: NodeId| match g.nodes[id].op {
            Op::Const(bits) => f64::from_bits(bits),
            ref other => panic!("not folded: {other:?}"),
        };
        assert_eq!(value(g.outputs[0]).to_bits(), (-9.0_f64).to_bits());
        assert_eq!(value(g.outputs[1]).to_bits(), 5.0_f64.to_bits());
        assert_eq!(g.slots, 0);
    }

    #[test]
    fn slots_are_reused_after_the_last_read() {
        // A chain of ten operators never needs more than two live series, and
        // the last one, which nothing reads, goes straight to the output.
        let g = graph(&["Abs(Abs(Abs(Abs(Abs(Abs(Abs(Abs(Abs(Abs($close))))))))))"]).unwrap();
        assert_eq!(g.slots, 2);
        assert_eq!(g.nodes[g.outputs[0]].slot, None);
        assert!(!g.nodes[g.outputs[0]].read);
        for node in &g.nodes {
            if let (Some(slot), [input]) = (node.slot, node.inputs.as_slice()) {
                assert_ne!(Some(slot), g.nodes[*input].slot);
            }
        }
    }

    #[test]
    fn past_refs_are_views_that_keep_their_base_alive() {
        // Ref(Ref(x,1),2) is x displaced by 3; the views take no slot, and
        // the series they displace stays live until the view's last reader.
        let g = graph(&["Ref(Ref(Abs($close),1),2) / $close", "Ref($close,-1)"]).unwrap();
        let abs = g
            .nodes
            .iter()
            .position(|n| n.op == Op::Unary(UnaryOp::Abs))
            .unwrap();
        let view = g.nodes.iter().position(|n| n.op == Op::Ref(2)).unwrap();
        assert_eq!(g.nodes[view].view, Some((abs, 3)));
        assert_eq!(g.nodes[view].slot, None);
        assert!(g.nodes[abs].read);
        assert!(g.nodes[abs].slot.is_some());
        // The future-reading Ref stays a computed node.
        let future = g.nodes.iter().find(|n| n.op == Op::Ref(-1)).unwrap();
        assert_eq!(future.view, None);
        // Abs, the division; the views and the future Ref (an unread output)
        // take none.
        assert_eq!(g.slots, 1);
    }

    #[test]
    fn bad_parameters_fail_at_compile_time() {
        for (source, message) in [
            ("Mean($close, 2.5)", "integer"),
            ("Mean($close, -1)", "negative"),
            ("Quantile($close, 5, 1.5)", "0..=1"),
            ("EMA($close, -1)", "negative"),
            ("Delta($close, -1)", "negative"),
            ("Ref($close, $open)", "literal"),
            ("Skew($close, 2)", "at least 3"),
            ("Kurt($close, 0.5)", "at least 4"),
        ] {
            let error = graph(&[source]).unwrap_err().to_string();
            assert!(error.contains(message), "{source}: {error}");
        }
    }
}
