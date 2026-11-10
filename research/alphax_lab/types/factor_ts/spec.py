"""Specs of the single-asset time-series factor type.

A spec is a alphax expression template with {param} placeholders, parameter
values (an ensemble, not a search range), a signal rule, and a pre-registered
way to use the grid (`selection`). Pure Python, no third-party imports.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass

FIELDS = ("open", "high", "low", "close", "volume", "vwap")

# Mirrors alphax/crates/alphax-expr/src/parse.rs:validate.
ARITY = {
    **dict.fromkeys(("Abs", "Sign", "Log", "Not"), 1),
    **dict.fromkeys(
        ("Greater", "Less", "Power", "Add", "Sub", "Mul", "Div", "Gt", "Ge", "Lt", "Le", "Eq", "Ne", "And", "Or"),
        2,
    ),
    "If": 3,
    **dict.fromkeys(
        (
            "Ref", "Mean", "Std", "Sum", "Max", "Min", "Rank", "IdxMax", "IdxMin", "Slope", "Rsquare", "Resi",
            "Var", "Skew", "Kurt", "Med", "Mad", "Count", "Delta", "WMA", "EMA",
        ),
        2,
    ),
    **dict.fromkeys(("Quantile", "Corr", "Cov"), 3),
}
COMMUTATIVE = {"+", "*", "==", "!=", "&", "|", "Add", "Mul", "Eq", "Ne", "And", "Or", "Corr", "Cov"}

SIGNAL_RULES = {
    # rule -> {param: (low, high)}; values must lie in [low, high].
    "sign": {},
    "zscore": {"window": (24, 8760), "scale": (0.1, 5.0)},
    "band": {"window": (24, 8760), "entry": (0.1, 5.0), "exit": (-2.0, 4.9)},
}
SIDES = ("both", "long", "short")
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
CONSTRAINT_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*|-?\d+(?:\.\d+)?)\s*(<=|>=|<|>|==|!=)\s*([A-Za-z_][A-Za-z0-9_]*|-?\d+(?:\.\d+)?)\s*$")
MAX_PARAM_VALUES = 8


class SpecError(ValueError):
    """The specification cannot be expanded."""


# ---------------------------------------------------------------- expression AST


@dataclass(frozen=True)
class Node:
    kind: str  # num | field | param | unary | bin | call
    label: str
    children: tuple[Node, ...] = ()

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()

    def size(self) -> int:
        return sum(1 for _ in self.walk())


def _lex(source: str) -> list[tuple[str, str]]:
    tokens, i = [], 0
    while i < len(source):
        c = source[i]
        if c.isspace():
            i += 1
        elif c in "(),":
            tokens.append((c, c))
            i += 1
        elif c == "{":
            end = source.find("}", i)
            if end < 0 or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", source[i + 1 : end]):
                raise SpecError(f"malformed parameter placeholder at byte {i}")
            tokens.append(("param", source[i + 1 : end]))
            i = end + 1
        elif c in "+-*/<>=!&|":
            two = source[i : i + 2]
            if two in (">=", "<=", "==", "!=", "**"):
                tokens.append(("op", two))
                i += 2
            elif c in "=!":
                raise SpecError(f"operator {c!r} must be followed by '='")
            else:
                tokens.append(("op", c))
                i += 1
        elif c == "$":
            match = re.match(r"\$([A-Za-z0-9_]+)", source[i:])
            if not match:
                raise SpecError("expected a field name after '$'")
            tokens.append(("field", match.group(1).lower()))
            i += len(match.group(0))
        elif c.isdigit() or c == ".":
            match = re.match(r"(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", source[i:])
            if match is None:
                raise SpecError(f"invalid number at byte {i}")
            tokens.append(("num", match.group(0)))
            i += len(match.group(0))
        elif c.isalpha() or c == "_":
            match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", source[i:])
            if match is None:
                raise SpecError(f"invalid name at byte {i}")
            tokens.append(("name", match.group(0)))
            i += len(match.group(0))
        else:
            raise SpecError(f"unexpected character {c!r} at byte {i}")
    tokens.append(("end", ""))
    return tokens


_BINDING = {"|": (1, 2), "&": (3, 4), "+": (7, 8), "-": (7, 8), "*": (9, 10), "/": (9, 10), "**": (12, 11)}
_BINDING.update(dict.fromkeys((">", "<", ">=", "<=", "==", "!="), (5, 6)))


class _Parser:
    def __init__(self, source: str) -> None:
        self.tokens = _lex(source)
        self.at = 0

    def take(self):
        token = self.tokens[self.at]
        self.at += 1
        return token

    def expect(self, kind: str, message: str) -> None:
        if self.take()[0] != kind:
            raise SpecError(message)

    def expression(self, min_bp: int = 0, depth: int = 0) -> Node:
        if depth > 128:
            raise SpecError("expression nesting exceeds 128")
        kind, value = self.take()
        if kind == "num":
            left = Node("num", value)
        elif kind == "field":
            left = Node("field", value)
        elif kind == "param":
            left = Node("param", value)
        elif kind == "op" and value in "+-":
            left = Node("unary", value, (self.expression(11, depth + 1),))
        elif kind == "(":
            left = self.expression(0, depth + 1)
            self.expect(")", "expected ')'")
        elif kind == "name":
            self.expect("(", f"expected '(' after {value}")
            args = []
            if self.tokens[self.at][0] != ")":
                while True:
                    args.append(self.expression(0, depth + 1))
                    if self.tokens[self.at][0] != ",":
                        break
                    self.take()
            self.expect(")", f"expected ')' after {value} arguments")
            left = Node("call", value, tuple(args))
        else:
            raise SpecError(f"expected expression, found {value or kind!r}")
        while self.tokens[self.at][0] == "op":
            op = self.tokens[self.at][1]
            left_bp, right_bp = _BINDING[op]
            if left_bp < min_bp:
                break
            self.take()
            left = Node("bin", op, (left, self.expression(right_bp, depth + 1)))
        return left


def parse(source: str) -> Node:
    """Parses a alphax expression; `{name}` placeholders become param nodes."""
    if len(source) > 4096:
        raise SpecError("expression exceeds 4096 characters")
    parser = _Parser(source)
    tree = parser.expression()
    if parser.tokens[parser.at][0] != "end":
        raise SpecError(f"unexpected token {parser.tokens[parser.at][1]!r}")
    for node in tree.walk():
        if node.kind == "call":
            if node.label not in ARITY:
                raise SpecError(f"unsupported operator {node.label!r}; allowed: {', '.join(sorted(ARITY))}")
            if len(node.children) != ARITY[node.label]:
                raise SpecError(f"{node.label} takes {ARITY[node.label]} arguments, got {len(node.children)}")
        if node.kind == "field" and node.label not in FIELDS:
            raise SpecError(f"unknown field ${node.label}; available: {', '.join('$' + f for f in FIELDS)}")
        if node.kind == "call" and node.label == "Ref":
            offset = node.children[1]
            if offset.kind == "unary" and offset.label == "-":
                raise SpecError("Ref with a negative offset reads future data")
    return tree


def complexity(tree: Node) -> dict:
    nodes = list(tree.walk())
    free = sum(node.kind in ("num", "param") for node in nodes)
    return {
        "nodes": len(nodes),
        "free_ratio": free / len(nodes),
        "fields": sorted({node.label for node in nodes if node.kind == "field"}),
    }


PRICE_FIELDS = {"open", "high", "low", "close", "vwap"}


def _label(node: Node, unify_prices: bool) -> tuple[str, str]:
    # Numbers and sweep parameters are interchangeable free arguments.
    if node.kind in ("num", "param"):
        return ("free", "")
    if unify_prices and node.kind == "field" and node.label in PRICE_FIELDS:
        return ("field", "price")
    return (node.kind, node.label)


def _match(a: Node, b: Node, memo: dict, unify_prices: bool) -> int:
    key = (id(a), id(b))
    if key in memo:
        return memo[key]
    if _label(a, unify_prices) != _label(b, unify_prices) or len(a.children) != len(b.children):
        memo[key] = 0
        return 0
    straight = sum(_match(x, y, memo, unify_prices) for x, y in zip(a.children, b.children))
    if a.label in COMMUTATIVE and len(a.children) == 2:
        swapped = _match(a.children[0], b.children[1], memo, unify_prices) + _match(
            a.children[1], b.children[0], memo, unify_prices
        )
        straight = max(straight, swapped)
    memo[key] = 1 + straight
    return memo[key]


def common_subtree(a: Node, b: Node, unify_prices: bool = False) -> int:
    """Size of the largest top-down common subtree (AlphaAgent-style originality check).

    With `unify_prices`, $open/$high/$low/$close/$vwap count as the same field,
    so swapping $close for $vwap does not make a structure new.
    """
    memo: dict = {}
    return max((_match(x, y, memo, unify_prices) for x in a.walk() for y in b.walk()), default=0)


def same_structure(a: Node, b: Node) -> bool:
    return a.size() == b.size() == common_subtree(a, b, unify_prices=True)


# ---------------------------------------------------------------- spec expansion


def _number(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SpecError(f"parameter values must be finite numbers, got {value!r}")
    return float(value)


def format_value(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def _values(name: str, raw) -> list[float]:
    values = raw if isinstance(raw, list) else [raw]
    if not values or len(values) > MAX_PARAM_VALUES:
        raise SpecError(f"parameter {name!r} needs 1-{MAX_PARAM_VALUES} values")
    out = [_number(value) for value in values]
    if len(set(out)) != len(out):
        raise SpecError(f"parameter {name!r} has duplicate values")
    return sorted(out)


def _constraint(text: str):
    match = CONSTRAINT_RE.match(text)
    if not match:
        raise SpecError(f"constraint {text!r} must look like 'fast < slow'")
    left, op, right = match.groups()
    compare = {
        "<": lambda x, y: x < y,
        "<=": lambda x, y: x <= y,
        ">": lambda x, y: x > y,
        ">=": lambda x, y: x >= y,
        "==": lambda x, y: x == y,
        "!=": lambda x, y: x != y,
    }[op]

    def resolve(token: str, values: dict) -> float:
        if re.fullmatch(r"-?\d+(?:\.\d+)?", token):
            return float(token)
        if token not in values:
            raise SpecError(f"constraint {text!r} names unknown parameter {token!r}")
        return values[token]

    return lambda values: compare(resolve(left, values), resolve(right, values))


def normalize(spec: dict, horizons: list[int]) -> dict:
    """Validates the shape of an LLM-proposed spec and fills defaults."""
    if not isinstance(spec, dict):
        raise SpecError("each factor must be a JSON object")
    name = str(spec.get("name", "")).strip().lower()
    if not NAME_RE.match(name):
        raise SpecError(f"factor name {name!r} must match {NAME_RE.pattern}")
    expression = str(spec.get("expression", "")).strip()
    tree = parse(expression)
    raw_params = spec.get("params") or {}
    if not isinstance(raw_params, dict):
        raise SpecError("params must be an object of name -> list of values")
    params = {str(key): _values(str(key), value) for key, value in raw_params.items()}
    used = {node.label for node in tree.walk() if node.kind == "param"}
    if used - set(params):
        raise SpecError(f"expression uses undefined parameters: {sorted(used - set(params))}")
    if set(params) - used:
        raise SpecError(f"parameters not used in the expression: {sorted(set(params) - used)}")
    signal = spec.get("signal") or {}
    if not isinstance(signal, dict):
        raise SpecError("signal must be an object")
    rule = str(signal.get("rule", "zscore"))
    if rule not in SIGNAL_RULES:
        raise SpecError(f"signal.rule must be one of {sorted(SIGNAL_RULES)}")
    side = str(signal.get("side", "both"))
    if side not in SIDES:
        raise SpecError(f"signal.side must be one of {SIDES}")
    raw_signal_params = signal.get("params") or {}
    if not isinstance(raw_signal_params, dict):
        raise SpecError("signal.params must be an object")
    bounds = SIGNAL_RULES[rule]
    extra = set(raw_signal_params) - set(bounds)
    missing = set(bounds) - set(raw_signal_params)
    if extra or missing:
        raise SpecError(f"signal rule {rule!r} takes exactly params {sorted(bounds)}")
    signal_params = {}
    for key, (low, high) in bounds.items():
        values = _values(key, raw_signal_params[key])
        if any(value < low or value > high for value in values):
            raise SpecError(f"signal param {key!r} must lie in [{low}, {high}]")
        signal_params[key] = values
    horizon = spec.get("horizon", horizons[-1])
    if horizon not in horizons:
        raise SpecError(f"horizon must be one of {horizons}")
    constraints = [str(item) for item in (spec.get("constraints") or [])]
    for item in constraints:
        _constraint(item)
    selection = _selection(spec.get("selection") or {"method": "ensemble"})
    return {
        "name": name,
        "family": str(spec.get("family", "other")).strip().lower() or "other",
        "expression": expression,
        "params": params,
        "constraints": constraints,
        "horizon": int(horizon),
        "signal": {"rule": rule, "side": side, "params": signal_params},
        "selection": selection,
        "rationale": str(spec.get("rationale", "")),
    }


SELECTION_METRICS = ("sharpe", "mean_return")


def _selection(raw) -> dict:
    """`ensemble` averages every grid member's position. `walk_forward` re-picks one member
    every `refit_bars` from the trailing `lookback_bars` of net returns; both are fixed
    before the run, so the whole procedure is one test."""
    if not isinstance(raw, dict):
        raise SpecError("selection must be an object")
    method = str(raw.get("method", "ensemble"))
    if method == "ensemble":
        return {"method": "ensemble"}
    if method != "walk_forward":
        raise SpecError("selection.method must be 'ensemble' or 'walk_forward'")
    metric = str(raw.get("metric", "sharpe"))
    if metric not in SELECTION_METRICS:
        raise SpecError(f"selection.metric must be one of {SELECTION_METRICS}")
    lookback, refit = raw.get("lookback_bars"), raw.get("refit_bars")
    for name, value, low in (("lookback_bars", lookback, 720), ("refit_bars", refit, 24)):
        if isinstance(value, bool) or not isinstance(value, int) or value < low:
            raise SpecError(f"selection.{name} must be an integer >= {low}")
    return {"method": method, "metric": metric, "lookback_bars": lookback, "refit_bars": refit}


def check_ranges(spec: dict, ranges: dict) -> list[str]:
    """Parameter values must lie inside the ranges the hypothesis card justified."""
    problems = []
    for name, values in {**spec["params"], **spec["signal"]["params"]}.items():
        if name not in ranges:
            problems.append(f"parameter {name!r} has no range in the hypothesis card")
            continue
        low, high = ranges[name]
        outside = [v for v in values if v < low or v > high]
        if outside:
            problems.append(f"parameter {name!r} values {outside} fall outside the card range [{low}, {high}]")
    return problems


def expand(spec: dict) -> list[dict]:
    """Expands a normalized spec into its concrete grid members."""
    axes = [("factor", key, values) for key, values in spec["params"].items()]
    axes += [("signal", key, values) for key, values in spec["signal"]["params"].items()]
    checks = [_constraint(item) for item in spec["constraints"]]
    if spec["signal"]["rule"] == "band":
        checks.append(lambda values: values["exit"] < values["entry"])
    combos = []
    for index in itertools.product(*[range(len(values)) for _, _, values in axes]):
        values = {key: axis_values[i] for (_, key, axis_values), i in zip(axes, index)}
        if not all(check(values) for check in checks):
            continue
        factor_params = {key: values[key] for kind, key, _ in axes if kind == "factor"}
        signal_params = {key: values[key] for kind, key, _ in axes if kind == "signal"}
        expression = PLACEHOLDER_RE.sub(lambda m: format_value(factor_params[m.group(1)]), spec["expression"])
        combos.append(
            {"index": tuple(index), "factor_params": factor_params, "signal_params": signal_params, "expression": expression}
        )
    if not combos:
        raise SpecError("constraints remove every parameter combination")
    return combos


def regulate(
    spec: dict,
    *,
    horizons: list[int],
    max_grid: int,
    max_nodes: int,
    max_free_ratio: float,
    max_common_subtree: int,
    others: dict[str, str],
    tried: dict[str, str] | None = None,
    ranges: dict | None = None,
) -> tuple[dict | None, list[str]]:
    """Returns the normalized spec, or None with the reasons it was rejected.

    `others` maps names of library factors and batch mates to their template
    expressions, for the shared-subtree originality check. `tried` maps
    previously evaluated factors to theirs: re-sweeping the same structure
    with new parameters only adds trials, so an identical structure is rejected.
    """
    try:
        spec = normalize(spec, horizons)
        grid = len(expand(spec))
    except SpecError as error:
        return None, [str(error)]
    problems = []
    tree = parse(spec["expression"])
    stats = complexity(tree)
    if grid > max_grid:
        problems.append(f"grid has {grid} combinations; at most {max_grid} are allowed. Use fewer values.")
    if stats["nodes"] > max_nodes:
        problems.append(f"expression has {stats['nodes']} nodes; at most {max_nodes} are allowed.")
    if stats["free_ratio"] >= max_free_ratio:
        problems.append(
            f"{stats['free_ratio']:.0%} of the expression nodes are constants or parameters; "
            f"keep it below {max_free_ratio:.0%}."
        )
    if ranges is not None:
        problems += check_ranges(spec, ranges)
    if spec["name"] in others:
        problems.append(f"name {spec['name']!r} is already used.")
    for other_name, other_expression in others.items():
        try:
            shared = common_subtree(tree, parse(other_expression))
        except SpecError:
            continue
        if shared > max_common_subtree:
            problems.append(
                f"shares a {shared}-node subtree with existing factor {other_name!r} ({other_expression}); "
                f"at most {max_common_subtree} nodes may be shared."
            )
    for other_name, other_expression in (tried or {}).items():
        try:
            other = parse(other_expression)
        except SpecError:
            continue
        if same_structure(tree, other):
            problems.append(
                f"has the same structure as already evaluated factor {other_name!r} ({other_expression}); "
                "change the construction, not only the parameters."
            )
    return (None, problems) if problems else (spec, [])
