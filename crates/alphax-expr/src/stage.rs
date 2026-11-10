//! Graphs with cross-sectional nodes, split into stages.
//!
//! A cross-sectional node's row reads every symbol's value on that row, so
//! everything it reads must be finished for all symbols first. [`Staged::plan`]
//! gives each node a level: inputs' highest level, plus one for a
//! cross-sectional node. Level `L`'s per-symbol nodes form one ordinary
//! [`Graph`], run a symbol at a time like a graph without cross-sectional
//! nodes; between level `L - 1` and level `L` the cross-sectional nodes of
//! level `L` run a row at a time.
//!
//! A series that crosses a stage boundary (read by a cross-sectional node, or
//! by a node of a later level) is kept whole as a panel; inside a stage's
//! graph it is a field named `#k` for panel `k`. A panel is dropped after the
//! last step that reads it. Fields and literals are read where they are and
//! never become panels.

use crate::graph::{Builder, Cross, Graph, NodeId, Op};

/// What a step reads.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Source {
    /// Index into the whole graph's fields.
    Field(usize),
    Const(f64),
    Panel(usize),
}

/// Where a stage's output goes.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Dest {
    /// A factor position.
    Output(usize),
    Panel(usize),
}

#[derive(Debug, Clone)]
pub(crate) enum Step {
    /// Per-symbol nodes: `sources` by the stage graph's field index, `dests`
    /// by its output position.
    Columns {
        graph: Graph,
        sources: Vec<Source>,
        dests: Vec<Dest>,
    },
    /// One cross-sectional node, written to `panel` and to factor `outputs`.
    Cross {
        op: Cross,
        inputs: Vec<Source>,
        panel: usize,
        outputs: Vec<usize>,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct Staged {
    pub(crate) steps: Vec<Step>,
    pub(crate) panels: usize,
    /// The node of the whole graph each panel holds.
    pub(crate) nodes: Vec<NodeId>,
    /// Per step, the panels no later step reads.
    pub(crate) release: Vec<Vec<usize>>,
}

const PANEL: char = '#';

impl Staged {
    /// The stages of `graph`, or `None` when it has no cross-sectional node.
    pub(crate) fn plan(graph: &Graph) -> Option<Self> {
        let nodes = &graph.nodes;
        let cross = |id: NodeId| matches!(nodes[id].op, Op::Cross(_));
        if !(0..nodes.len()).any(cross) {
            return None;
        }
        let source = |id: NodeId| matches!(nodes[id].op, Op::Field(_) | Op::Const(_));
        let mut level = vec![0; nodes.len()];
        let mut crossing = vec![false; nodes.len()];
        for at in 0..nodes.len() {
            let top = nodes[at]
                .inputs
                .iter()
                .map(|&i| level[i])
                .max()
                .unwrap_or(0);
            level[at] = top + usize::from(cross(at));
            for &input in &nodes[at].inputs {
                crossing[input] |= cross(at) || level[at] > level[input];
            }
        }
        let mut panel = vec![None; nodes.len()];
        let mut panels = 0;
        let mut held = Vec::new();
        for at in 0..nodes.len() {
            if cross(at) || (crossing[at] && !source(at)) {
                panel[at] = Some(panels);
                held.push(at);
                panels += 1;
            }
        }
        let positions = |id: NodeId| {
            graph
                .outputs
                .iter()
                .enumerate()
                .filter(move |&(_, &out)| out == id)
                .map(|(position, _)| position)
        };
        let read = |id: NodeId| match nodes[id].op {
            Op::Const(bits) => Source::Const(f64::from_bits(bits)),
            Op::Field(index) => Source::Field(index),
            _ => Source::Panel(panel[id].expect("a series read across stages is a panel")),
        };

        let top = level.iter().copied().max().unwrap_or(0);
        let mut steps = Vec::new();
        for stage in 0..=top {
            for at in (0..nodes.len()).filter(|&at| cross(at) && level[at] == stage) {
                let Op::Cross(op) = nodes[at].op else {
                    unreachable!()
                };
                steps.push(Step::Cross {
                    op,
                    inputs: nodes[at].inputs.iter().map(|&i| read(i)).collect(),
                    panel: panel[at].expect("cross-sectional nodes are panels"),
                    outputs: positions(at).collect(),
                });
            }
            let mut b = Builder::default();
            let mut local: Vec<Option<NodeId>> = vec![None; nodes.len()];
            let mut outputs = Vec::new();
            let mut dests = Vec::new();
            let input = |b: &mut Builder, local: &[Option<NodeId>], id: NodeId| match nodes[id].op {
                Op::Const(_) => b.intern(nodes[id].op, Vec::new()),
                Op::Field(index) => b.field(&graph.fields[index]),
                _ => local[id].unwrap_or_else(|| {
                    let k = panel[id].expect("a series read across stages is a panel");
                    b.field(&format!("{PANEL}{k}"))
                }),
            };
            for at in 0..nodes.len() {
                let here = if stage == 0 && source(at) {
                    // A bare field or literal that is itself a factor.
                    positions(at).next().is_some()
                } else {
                    level[at] == stage && !cross(at) && !source(at)
                };
                if !here {
                    continue;
                }
                let id = if source(at) {
                    input(&mut b, &local, at)
                } else {
                    let inputs = nodes[at]
                        .inputs
                        .iter()
                        .map(|&i| input(&mut b, &local, i))
                        .collect();
                    b.intern(nodes[at].op, inputs)
                };
                local[at] = Some(id);
                if let (Some(k), false) = (panel[at], source(at)) {
                    outputs.push(id);
                    dests.push(Dest::Panel(k));
                }
                for position in positions(at) {
                    outputs.push(id);
                    dests.push(Dest::Output(position));
                }
            }
            if outputs.is_empty() {
                continue;
            }
            let graph_here = b.finish(outputs);
            let sources = graph_here
                .fields
                .iter()
                .map(|name| match name.strip_prefix(PANEL) {
                    Some(k) => Source::Panel(k.parse().expect("panel names are numbers")),
                    None => Source::Field(
                        graph
                            .fields
                            .iter()
                            .position(|field| field == name)
                            .expect("a stage reads the graph's fields"),
                    ),
                })
                .collect();
            steps.push(Step::Columns {
                graph: graph_here,
                sources,
                dests,
            });
        }

        let mut last = vec![0; panels];
        for (at, step) in steps.iter().enumerate() {
            let (reads, own): (&[Source], Option<usize>) = match step {
                Step::Columns { sources, .. } => (sources, None),
                Step::Cross { inputs, panel, .. } => (inputs, Some(*panel)),
            };
            for k in reads
                .iter()
                .filter_map(|s| match s {
                    Source::Panel(k) => Some(*k),
                    _ => None,
                })
                .chain(own)
            {
                last[k] = last[k].max(at);
            }
        }
        let mut release = vec![Vec::new(); steps.len()];
        for (k, &at) in last.iter().enumerate() {
            release[at].push(k);
        }
        Some(Self {
            steps,
            panels,
            nodes: held,
            release,
        })
    }
}
