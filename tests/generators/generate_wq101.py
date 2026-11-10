#!/usr/bin/env python3
"""WorldQuant's 101 Formulaic Alphas: alphax's translation and a reference.

`FORMULAS` is Appendix A of Kakushadze, "101 Formulaic Alphas"
(arXiv:1601.00991v3), copied unchanged from the paper's text. This script
parses each formula once and uses the tree twice:

* ``rust`` prints the table in ``crates/alphax-expr/src/wq101.rs``: each
  formula in alphax's factor syntax.
* ``fixture`` writes ``tests/fixtures/wq101.tsv``: each formula evaluated by a
  direct NumPy interpreter written from the paper's definitions (A.1), on a
  seeded synthetic panel. The interpreter shares no code with alphax; it
  re-implements each operator the slow way.

Reading the paper's notation (A.1 and A.2):

* ``rank`` is cross-sectional (``CSRank``, pandas ``rank(pct=True)``);
  ``ts_rank`` is the time-series one (``Rank``: the last value's percentile in
  its window).
* Non-integer day counts are floored, as A.1 says.
* ``min(x, y)`` and ``max(x, y)`` with a day count are ``ts_min``/``ts_max``;
  with two series (Alpha#71, #73, ...) they are elementwise.
* ``ts_argmax``/``ts_argmin`` count days from the window's oldest, 1-based
  (Qlib's ``IdxMax``/``IdxMin``).
* ``adv{d}`` is ``Mean($volume, d)``. A.2 calls it dollar volume, yet several
  alphas compare it with ``volume`` directly (#7, #17, #21, #43, #47, #100);
  pass dollar volume as ``$volume`` for A.2's reading.
* ``returns`` is ``$close / Ref($close, 1) - 1``; ``cap`` is ``$cap``;
  ``IndClass.sector`` and the other levels are the fields ``$sector``,
  ``$industry``, ``$subindustry`` holding a group label per symbol and row.

Time-series windows follow Qlib, as the rest of alphax's factor operators do:
partial windows at the start (``min_periods=1``), ``NaN`` skipped, sample
standard deviation, and ``Corr`` is ``NaN`` when either side's standard
deviation is at most 2e-5.

Usage::

    python3 tests/generators/generate_wq101.py rust      # the Rust table, to stdout
    python3 tests/generators/generate_wq101.py fixture   # needs NumPy
"""

from __future__ import annotations

import math
import re
import sys

FORMULAS = {
    1: '(rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)',
    2: '(-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))',
    3: '(-1 * correlation(rank(open), rank(volume), 10))',
    4: '(-1 * Ts_Rank(rank(low), 9))',
    5: '(rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))',
    6: '(-1 * correlation(open, volume, 10))',
    7: '((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))',
    8: '(-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))',
    9: '((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))',
    10: 'rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))',
    11: '((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))',
    12: '(sign(delta(volume, 1)) * (-1 * delta(close, 1)))',
    13: '(-1 * rank(covariance(rank(close), rank(volume), 5)))',
    14: '((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))',
    15: '(-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))',
    16: '(-1 * rank(covariance(rank(high), rank(volume), 5)))',
    17: '(((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))',
    18: '(-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10))))',
    19: '((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))',
    20: '(((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))',
    21: '((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : (((sum(close, 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) || ((volume / adv20) == 1)) ? 1 : (-1 * 1))))',
    22: '(-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))',
    23: '(((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)',
    24: '((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) || ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ? (-1 * (close - ts_min(close, 100))) : (-1 * delta(close, 3)))',
    25: 'rank(((((-1 * returns) * adv20) * vwap) * (high - close)))',
    26: '(-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))',
    27: '((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1)',
    28: 'scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))',
    29: '(min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1), 5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))',
    30: '(((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) + sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))',
    31: '((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 * delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))',
    32: '(scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230))))',
    33: 'rank((-1 * ((1 - (open / close))^1)))',
    34: 'rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))',
    35: '((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))',
    36: '(((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open - close)))) + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap, adv20, 6)))) + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))',
    37: '(rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))',
    38: '((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))',
    39: '((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(sum(returns, 250))))',
    40: '((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))',
    41: '(((high * low)^0.5) - vwap)',
    42: '(rank((vwap - close)) / rank((vwap + close)))',
    43: '(ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))',
    44: '(-1 * correlation(high, rank(volume), 5))',
    45: '(-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) * rank(correlation(sum(close, 5), sum(close, 20), 2))))',
    46: '((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ? (-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 : ((-1 * 1) * (close - delay(close, 1)))))',
    47: '((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) / 5))) - rank((vwap - delay(vwap, 5))))',
    48: '(indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) * delta(close, 1)) / close), IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250))',
    49: '(((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))',
    50: '(-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))',
    51: '(((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))',
    52: '((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))',
    53: '(-1 * delta((((close - low) - (high - close)) / (close - low)), 9))',
    54: '((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))',
    55: '(-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6))',
    56: '(0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))',
    57: '(0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))',
    58: '(-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume, 3.92795), 7.89291), 5.50322))',
    59: '(-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * 0.728317) + (vwap * (1 - 0.728317))), IndClass.industry), volume, 4.25197), 16.2289), 8.19648))',
    60: '(0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) - scale(rank(ts_argmax(close, 10))))))',
    61: '(rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))',
    62: '((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)',
    63: '((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237)) - rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180, 37.2467), 13.557), 12.2883))) * -1)',
    64: '((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054), sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404))), 3.69741))) * -1)',
    65: '((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60, 8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)',
    66: '((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)',
    67: '((rank((high - ts_min(high, 2.14593)))^rank(correlation(IndNeutralize(vwap, IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1)',
    68: '((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)',
    69: '((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412), 4.79344))^Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416), 9.0615)) * -1)',
    70: '((rank(delta(vwap, 1.29456))^Ts_Rank(correlation(IndNeutralize(close, IndClass.industry), adv50, 17.8256), 17.9171)) * -1)',
    71: 'max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180, 12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear((rank(((low + open) - (vwap + vwap)))^2), 16.4662), 4.4388))',
    72: '(rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) / rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671), 2.95011)))',
    73: '(max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)), Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open * 0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)',
    74: '((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) < rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791))) * -1)',
    75: '(rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50), 12.4413)))',
    76: '(max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)), Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low, IndClass.sector), adv81, 8.14941), 19.569), 17.1543), 19.383)) * -1)',
    77: 'min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)), rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))',
    78: '(rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428), sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492)))',
    79: '(rank(delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))), IndClass.sector), 1.23438)) < rank(correlation(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150, 9.18637), 14.6644)))',
    80: '((rank(Sign(delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))), IndClass.industry), 4.04545)))^Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1)',
    81: '((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054), 8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)',
    82: '(min(rank(decay_linear(delta(open, 1.46063), 14.8717)), Ts_Rank(decay_linear(correlation(IndNeutralize(volume, IndClass.sector), ((open * 0.634196) + (open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1)',
    83: '((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high - low) / (sum(close, 5) / 5)) / (vwap - close)))',
    84: 'SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close, 4.96796))',
    85: '(rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30, 9.61331))^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595), 7.11408)))',
    86: '((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) < rank(((open + close) - (vwap + open)))) * -1)',
    87: '(max(rank(decay_linear(delta(((close * 0.369701) + (vwap * (1 - 0.369701))), 1.91233), 2.65461)), Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81, IndClass.industry), close, 13.4132)), 4.89768), 14.4535)) * -1)',
    88: 'min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))), 8.06882)), Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60, 20.6966), 8.01266), 6.65053), 2.61957))',
    89: '(Ts_Rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10, 6.94279), 5.51607), 3.79744) - Ts_Rank(decay_linear(delta(IndNeutralize(vwap, IndClass.industry), 3.48158), 10.1466), 15.3012))',
    90: '((rank((close - ts_max(close, 4.66719)))^Ts_Rank(correlation(IndNeutralize(adv40, IndClass.subindustry), low, 5.38375), 3.21856)) * -1)',
    91: '((Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close, IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) - rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1)',
    92: 'min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221), 18.8683), Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024), 6.80584))',
    93: '(Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81, 17.4193), 19.848), 7.54455) / rank(decay_linear(delta(((close * 0.524434) + (vwap * (1 - 0.524434))), 2.77377), 16.2664)))',
    94: '((rank((vwap - ts_min(vwap, 11.5783)))^Ts_Rank(correlation(Ts_Rank(vwap, 19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)',
    95: '(rank((open - ts_min(open, 12.4105))) < Ts_Rank((rank(correlation(sum(((high + low) / 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))',
    96: '(max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878), 4.16783), 8.38151), Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404), Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)',
    97: '((rank(decay_linear(delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))), IndClass.industry), 3.3705), 20.4523)) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low, 7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1)',
    98: '(rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) - rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571), 6.95668), 8.07206)))',
    99: '((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136)) < rank(correlation(low, volume, 6.28259))) * -1)',
    100: '(0 - (1 * (((1.5 * scale(indneutralize(indneutralize(rank(((((close - low) - (high - close)) / (high - low)) * volume)), IndClass.subindustry), IndClass.subindustry))) - scale(indneutralize((correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))), IndClass.subindustry))) * (volume / adv20))))',
    101: '((close - open) / ((high - low) + .001))',
}

# ------------------------------------------------------------------- parsing

TOKEN = re.compile(r"\s*(?:(\d+\.\d*|\.\d+|\d+)|([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_]+)?)|(\|\||==|[-+*/^<>?:(),]))")


def tokenize(source: str) -> list[tuple[str, str]]:
    out, at = [], 0
    source = source.strip()
    while at < len(source):
        m = TOKEN.match(source, at)
        if not m or m.end() == at:
            raise SyntaxError(f"bad character at {at}: {source[at:at + 20]!r}")
        number, name, op = m.groups()
        out.append(("num", number) if number else ("name", name) if name else ("op", op))
        at = m.end()
    out.append(("end", ""))
    return out


class Parser:
    """Precedence, lowest first: ``?:``, ``||``, comparisons, ``+ -``,
    ``* /``, ``^``, unary minus."""

    def __init__(self, source: str):
        self.tokens = tokenize(source)
        self.at = 0

    def peek(self) -> tuple[str, str]:
        return self.tokens[self.at]

    def take(self, value: str | None = None) -> tuple[str, str]:
        token = self.tokens[self.at]
        if value is not None and token[1] != value:
            raise SyntaxError(f"expected {value!r}, found {token[1]!r}")
        self.at += 1
        return token

    def parse(self):
        tree = self.ternary()
        self.take("")
        return tree

    def ternary(self):
        cond = self.binary(0)
        if self.peek() == ("op", "?"):
            self.take()
            yes = self.ternary()
            self.take(":")
            no = self.ternary()
            return ("if", cond, yes, no)
        return cond

    LEVELS = [("||",), ("<", ">", "=="), ("+", "-"), ("*", "/"), ("^",)]

    def binary(self, level: int):
        if level == len(self.LEVELS):
            return self.unary()
        left = self.binary(level + 1)
        while self.peek()[0] == "op" and self.peek()[1] in self.LEVELS[level]:
            op = self.take()[1]
            left = ("bin", op, left, self.binary(level + 1))
        return left

    def unary(self):
        if self.peek() == ("op", "-"):
            self.take()
            return ("neg", self.unary())
        return self.primary()

    def primary(self):
        kind, value = self.take()
        if kind == "num":
            return ("num", float(value))
        if kind == "op" and value == "(":
            inner = self.ternary()
            self.take(")")
            return inner
        if kind == "name":
            if self.peek() == ("op", "("):
                self.take()
                args = [self.ternary()]
                while self.peek() == ("op", ","):
                    self.take()
                    args.append(self.ternary())
                self.take(")")
                return ("call", value.lower(), args)
            return ("var", value)
        raise SyntaxError(f"unexpected {value!r}")


def days(node) -> int:
    if node[0] != "num":
        raise SyntaxError(f"a day count must be a number, found {node}")
    d = math.floor(node[1])
    if d < 1:
        raise SyntaxError(f"day count {node[1]} floors below 1")
    return d


def groups(node) -> str:
    if node[0] != "var" or not node[1].startswith("IndClass."):
        raise SyntaxError(f"expected IndClass.<level>, found {node}")
    return node[1].split(".", 1)[1]


def fields(node, out: set[str]) -> set[str]:
    kind = node[0]
    if kind == "var":
        name = node[1]
        if name.startswith("adv") or name == "returns":
            out.add("close" if name == "returns" else "volume")
        elif name.startswith("IndClass."):
            out.add(groups(node))
        else:
            out.add(name)
    elif kind == "neg":
        fields(node[1], out)
    elif kind in ("bin",):
        fields(node[2], out)
        fields(node[3], out)
    elif kind == "if":
        for child in node[1:]:
            fields(child, out)
    elif kind == "call":
        for arg in node[2]:
            fields(arg, out)
    return out


# --------------------------------------------------------------- translation

ROLLING = {
    "ts_rank": "Rank", "ts_min": "Min", "ts_max": "Max", "ts_argmax": "IdxMax", "ts_argmin": "IdxMin",
    "sum": "Sum", "product": "Prod", "stddev": "Std", "decay_linear": "DecayLinear",
    "delay": "Ref", "delta": "Delta",
}
ELEMENTWISE = {"abs": "Abs", "log": "Log", "sign": "Sign", "rank": "CSRank"}


def number(value: float) -> str:
    return str(int(value)) if value == int(value) else repr(value)


def alphax(node) -> str:
    kind = node[0]
    if kind == "num":
        return number(node[1])
    if kind == "var":
        name = node[1]
        if name == "returns":
            return "($close / Ref($close, 1) - 1)"
        if m := re.fullmatch(r"adv(\d+)", name):
            return f"Mean($volume, {m.group(1)})"
        if name in ("open", "high", "low", "close", "volume", "vwap", "cap"):
            return f"${name}"
        raise SyntaxError(f"unknown variable {name}")
    if kind == "neg":
        return f"(-{alphax(node[1])})"
    if kind == "bin":
        op, a, b = node[1], alphax(node[2]), alphax(node[3])
        if op == "^":
            return f"Power({a}, {b})"
        return f"({a} {'|' if op == '||' else op} {b})"
    if kind == "if":
        return f"If({alphax(node[1])}, {alphax(node[2])}, {alphax(node[3])})"
    name, args = node[1], node[2]
    if name in ELEMENTWISE:
        return f"{ELEMENTWISE[name]}({alphax(args[0])})"
    if name in ROLLING:
        return f"{ROLLING[name]}({alphax(args[0])}, {days(args[1])})"
    if name in ("min", "max"):
        if args[1][0] == "num":
            return f"{'Min' if name == 'min' else 'Max'}({alphax(args[0])}, {days(args[1])})"
        return f"{'Less' if name == 'min' else 'Greater'}({alphax(args[0])}, {alphax(args[1])})"
    if name in ("correlation", "covariance"):
        op = "Corr" if name == "correlation" else "Cov"
        return f"{op}({alphax(args[0])}, {alphax(args[1])}, {days(args[2])})"
    if name == "scale":
        scale = f", {number(args[1][1])}" if len(args) > 1 else ""
        return f"Scale({alphax(args[0])}{scale})"
    if name == "signedpower":
        return f"SignedPower({alphax(args[0])}, {alphax(args[1])})"
    if name == "indneutralize":
        return f"IndNeutralize({alphax(args[0])}, ${groups(args[1])})"
    raise SyntaxError(f"unknown function {name}")


def rust() -> None:
    print("/// `(name, paper formula, alphax expression)`; generated by")
    print("/// `tests/generators/generate_wq101.py rust`.")
    print("pub(crate) const ALPHAS: [(&str, &str, &str); 101] = [")
    for k, source in FORMULAS.items():
        expr = alphax(Parser(source).parse())
        print(f'    (\n        "alpha{k:03}",\n        {rust_str(source)},\n        {rust_str(expr)},\n    ),')
    print("];")


def rust_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


# ----------------------------------------------------------------- reference


EPS = 1e-9


def reference(np):
    """The interpreter: every operator written directly from its definition.

    Each value carries a taint: whether it may depend on how a tie in exact
    arithmetic rounds. A two-point correlation is exactly 1 and a covariance
    of rank series a multiple of 1/36 in exact arithmetic, but each comes out
    within a few ulps of that, differently in any two implementations, and a
    rank, comparison or arg-extreme that reads such near-equal values can go
    either way. A cell is tainted when one of those operators saw values
    within ``EPS`` of each other (or of zero, for ``sign``), or read a tainted
    value; cross-sectional operators taint a whole row. The fixture marks
    tainted cells and the Rust test skips them.

    Values also carry exactness: whether any implementation computes them by
    the same IEEE operations in the same order, so they come out bit-identical
    (fields, elementwise arithmetic on them, ranks, comparisons, shifts,
    window extremes). A near-tie among exact values is decided the same way
    everywhere and taints nothing; one involving a windowed sum, mean,
    deviation, correlation or the like (summed in a different order by
    alphax's running windows) does.
    """

    def near(a, b):
        return np.abs(a - b) <= EPS * np.maximum(1, np.maximum(np.abs(a), np.abs(b)))

    def full(x):
        return np.broadcast_to(np.asarray(x, dtype=float), SHAPE).copy()

    def none():
        return np.zeros(SHAPE, bool)

    def roll(x, t, e, d, f, tie=None, exact=False):
        """``f`` over each trailing window; tainted if the window is, or if
        ``tie(window, exactness)`` says an inexact near-tie decides it."""
        out, taint = np.full(SHAPE, np.nan), np.zeros(SHAPE, bool)
        for c in range(SHAPE[1]):
            for r in range(SHAPE[0]):
                lo = max(0, r - d + 1)
                w = x[lo:r + 1, c]
                out[r, c] = f(w)
                taint[r, c] = t[lo:r + 1, c].any() or (tie is not None and tie(w, e[lo:r + 1, c]))
        exactness = np.array([[e[max(0, r - d + 1):r + 1, c].all() for c in range(SHAPE[1])]
                              for r in range(SHAPE[0])]) if exact else np.zeros(SHAPE, bool)
        return out, taint, exactness

    def roll2(x, tx, y, ty, d, f):
        out, taint = np.full(SHAPE, np.nan), np.zeros(SHAPE, bool)
        for c in range(SHAPE[1]):
            for r in range(SHAPE[0]):
                lo = max(0, r - d + 1)
                out[r, c] = f(x[lo:r + 1, c], y[lo:r + 1, c])
                taint[r, c] = tx[lo:r + 1, c].any() or ty[lo:r + 1, c].any()
        return out, taint, np.zeros(SHAPE, bool)

    def present(w):
        return w[~np.isnan(w)]

    def r_sum(w):
        v = present(w)
        return v.sum() if len(v) else np.nan

    def r_std(w):
        v = present(w)
        return v.std(ddof=1) if len(v) >= 2 else np.nan

    def r_min(w):
        v = present(w)
        return v.min() if len(v) else np.nan

    def r_max(w):
        v = present(w)
        return v.max() if len(v) else np.nan

    def r_prod(w):
        v = present(w)
        return v.prod() if len(v) else np.nan

    def r_rank(w):
        last = w[-1]
        if np.isnan(last):
            return np.nan
        v = present(w)
        return ((v < last).sum() + ((v == last).sum() + 1) / 2) / len(v)

    def last_tied(w, e):
        if np.isnan(w[-1]):
            return False
        others = ~np.isnan(w[:-1]) & near(w[:-1], w[-1]) & ~(e[:-1] & e[-1])
        return bool(others.any())

    def r_decay(w):
        weights = np.arange(1, len(w) + 1, dtype=float)
        keep = ~np.isnan(w)
        return (weights[keep] * w[keep]).sum() / weights[keep].sum() if keep.any() else np.nan

    def r_arg(minimum):
        def f(w):
            best = 0
            for i in range(1, len(w)):
                if (not np.isnan(w[best]) and np.isnan(w[i])) or (
                    not np.isnan(w[best]) and not np.isnan(w[i])
                    and (w[i] < w[best] if minimum else w[i] > w[best])
                ):
                    best = i
            return best + 1.0
        return f

    def extreme_tied(minimum):
        def tie(w, e):
            keep = ~np.isnan(w)
            if keep.sum() < 2:
                return False
            best = w[keep].min() if minimum else w[keep].max()
            close = keep & near(w, best)
            return int(close.sum()) > 1 and not e[close].all()
        return tie

    def r_corr(a, b):
        for side in (a, b):
            v = present(side)
            if len(v) < 2 or v.std(ddof=1) <= 2e-5:
                return np.nan
        keep = ~np.isnan(a) & ~np.isnan(b)
        a, b = a[keep], b[keep]
        if len(a) < 2:
            return np.nan
        da, db = a - a.mean(), b - b.mean()
        va, vb = (da * da).sum(), (db * db).sum()
        if va == 0 or vb == 0:
            return np.nan
        return (da * db).sum() / math.sqrt(va * vb)

    def r_cov(a, b):
        keep = ~np.isnan(a) & ~np.isnan(b)
        a, b = a[keep], b[keep]
        if len(a) < 2:
            return np.nan
        return ((a - a.mean()) * (b - b.mean())).sum() / (len(a) - 1)

    def shift(x, d):
        out = np.full(SHAPE, np.nan)
        out[d:] = x[:len(x) - d]
        return out

    def shift_taint(t, d):
        out = np.zeros(SHAPE, bool)
        out[d:] = t[:len(t) - d]
        return out

    def rows_taint(t):
        return np.broadcast_to(t.any(axis=1, keepdims=True), SHAPE).copy()

    def cs_rank(x, t, e):
        out, taint = np.full(SHAPE, np.nan), rows_taint(t)
        for r in range(SHAPE[0]):
            v = present(x[r])
            keep = ~np.isnan(x[r])
            for c in range(SHAPE[1]):
                if keep[c]:
                    out[r, c] = ((v < x[r, c]).sum() + ((v == x[r, c]).sum() + 1) / 2) / len(v)
                    close = keep & near(x[r], x[r, c])
                    taint[r, c] |= int(close.sum()) > 1 and not e[r, close].all()
        return out, taint, np.ones(SHAPE, bool)

    def scale(x, t, a):
        total = np.array([np.abs(present(row)).sum() for row in x])
        return x * a / total[:, None], rows_taint(t), np.zeros(SHAPE, bool)

    def neutralize(x, t, g):
        out = np.full(SHAPE, np.nan)
        for r in range(SHAPE[0]):
            for label in set(present(g[r]).tolist()):
                members = g[r] == label
                v = present(x[r, members])
                out[r, members] = x[r, members] - (v.mean() if len(v) else np.nan)
        return out, rows_taint(t), np.zeros(SHAPE, bool)

    def run(node, env):
        kind = node[0]
        exact = np.ones(SHAPE, bool)
        if kind == "num":
            return full(node[1]), none(), exact
        if kind == "var":
            name = node[1]
            if name == "returns":
                return env["close"] / shift(env["close"], 1) - 1, none(), exact
            if m := re.fullmatch(r"adv(\d+)", name):
                return roll(env["volume"], none(), exact, int(m.group(1)), lambda w: r_sum(w) / len(present(w)))
            return env[name], none(), exact
        if kind == "neg":
            x, t, e = run(node[1], env)
            return -x, t, e
        if kind == "bin":
            (a, ta, ea), (b, tb, eb) = run(node[2], env), run(node[3], env)
            op, taint = node[1], ta | tb
            if op in ("<", ">", "=="):
                taint = taint | (near(a, b) & ~(ea & eb))
            value = {
                "+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b, "/": lambda: a / b,
                "^": lambda: np.power(a, b),
                "<": lambda: np.less(a, b) * 1.0, ">": lambda: np.greater(a, b) * 1.0,
                "==": lambda: np.equal(a, b) * 1.0,
                "||": lambda: ((np.not_equal(a, 0)) | (np.not_equal(b, 0))) * 1.0,
            }[op]()
            return value, taint, ea & eb
        if kind == "if":
            (c, tc, ec), (a, ta, ea), (b, tb, eb) = (run(child, env) for child in node[1:])
            yes = np.not_equal(c, 0)
            return np.where(yes, a, b), tc | np.where(yes, ta, tb), ec & ea & eb
        name, args = node[1], node[2]
        x, t, e = run(args[0], env)
        if name == "abs":
            return np.abs(x), t, e
        if name == "log":
            return np.log(x), t, e
        if name == "sign":
            return np.sign(x), t | ((np.abs(x) <= EPS) & ~e), e
        if name == "rank":
            return cs_rank(x, t, e)
        if name in ("delay", "delta"):
            d = days(args[1])
            moved = shift(x, d)
            e_moved = shift_taint(e, d) | np.isnan(moved)
            value = moved if name == "delay" else x - moved
            return value, (t if name == "delta" else none()) | shift_taint(t, d), e_moved & (e if name == "delta" else True)
        single = {
            "ts_rank": (r_rank, last_tied, True), "ts_min": (r_min, None, True), "ts_max": (r_max, None, True),
            "ts_argmax": (r_arg(False), extreme_tied(False), True),
            "ts_argmin": (r_arg(True), extreme_tied(True), True),
            "sum": (r_sum, None, False), "product": (r_prod, None, False), "stddev": (r_std, None, False),
            "decay_linear": (r_decay, None, False),
        }
        if name in single:
            f, tie, keeps = single[name]
            out, taint, exactness = roll(x, t, e, days(args[1]), f, tie, exact=keeps)
            if name in ("ts_rank", "ts_argmax", "ts_argmin"):
                exactness = np.ones(SHAPE, bool)
            return out, taint, exactness
        if name in ("min", "max"):
            if args[1][0] == "num":
                return roll(x, t, e, days(args[1]), r_min if name == "min" else r_max, exact=True)
            y, ty, ey = run(args[1], env)
            f = np.minimum if name == "min" else np.maximum
            return np.where(np.isnan(x) | np.isnan(y), np.nan, f(x, y)), t | ty, e & ey
        if name in ("correlation", "covariance"):
            y, ty, _ = run(args[1], env)
            return roll2(x, t, y, ty, days(args[2]), r_corr if name == "correlation" else r_cov)
        if name == "scale":
            return scale(x, t, args[1][1] if len(args) > 1 else 1.0)
        if name == "signedpower":
            p, tp, ep = run(args[1], env)
            return np.where(x == 0, 0.0, np.sign(x) * np.power(np.abs(x), p)), t | tp, e & ep
        if name == "indneutralize":
            return neutralize(x, t, env[groups(args[1])])
        raise SyntaxError(name)

    return run


ROWS, COLS, SEED = 300, 6, 101
SHAPE = (ROWS, COLS)
KEPT = range(255, ROWS, 3)
FIELDS = ("open", "high", "low", "close", "volume", "vwap", "cap", "sector", "industry", "subindustry")


def panel(np):
    """Prices rounded to 4 decimals and integer volumes, so the inputs print
    exactly. The symbols share a common walk and differ by a mean-reverting
    term, so their order changes from day to day and cross-sectional ranks
    move (with independent walks the ranks freeze and every correlation of
    ranks is `NaN`)."""
    rng = np.random.default_rng(SEED)
    common = np.cumsum(rng.normal(0, 0.015, ROWS))
    idio = np.zeros((ROWS, COLS))
    for t in range(1, ROWS):
        idio[t] = 0.6 * idio[t - 1] + rng.normal(0, 0.02, COLS)
    close = 50 * np.exp(common[:, None] + idio)
    open_ = close * np.exp(rng.normal(0, 0.008, (ROWS, COLS)))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.006, (ROWS, COLS))))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.006, (ROWS, COLS))))
    w = rng.uniform(0.2, 0.8, (ROWS, COLS))
    vwap = low + w * (high - low)
    env = {
        "open": open_.round(4), "high": high.round(4), "low": low.round(4), "close": close.round(4),
        "vwap": vwap.round(4),
        "volume": np.round(np.exp(rng.normal(12, 0.5, (ROWS, COLS)))),
        "cap": (close * rng.integers(1_000_000, 9_000_000, COLS)).round(-3),
    }
    env["high"] = np.maximum.reduce([env["high"], env["open"], env["close"], env["vwap"]])
    env["low"] = np.minimum.reduce([env["low"], env["open"], env["close"], env["vwap"]])
    cols = np.arange(COLS)
    # Sub-industries nest in sectors; one of them has a single member.
    for name, labels in (("sector", cols % 2), ("industry", cols % 3), ("subindustry", cols % 2 * 2 + cols // 3 % 2)):
        env[name] = np.broadcast_to(labels.astype(float), (ROWS, COLS)).copy()
    return env


def fixture() -> None:
    import numpy as np

    env = panel(np)
    run = reference(np)
    names = [f"alpha{k:03}" for k in FORMULAS]
    values, ties = {}, {}
    with np.errstate(all="ignore"):
        for k, source in FORMULAS.items():
            values[f"alpha{k:03}"], ties[f"alpha{k:03}"], _ = run(Parser(source).parse(), env)
    path = "tests/fixtures/wq101.tsv"
    with open(path, "w") as f:
        f.write(f"# generated by tests/generators/generate_wq101.py fixture; numpy={np.__version__}\n")
        f.write(f"# layout=symbol-major rows={ROWS} symbols={COLS} seed={SEED}; alphas only on kept rows\n")
        f.write("# a value prefixed with ~ may depend on how a tie in exact arithmetic rounds\n")
        f.write("\t".join(["symbol", "row", "kept", *FIELDS, *names]) + "\n")
        for c in range(COLS):
            for t in range(ROWS):
                kept = t in KEPT
                cells = [str(c), str(t), str(int(kept))]
                cells += [repr(float(env[name][t, c])) for name in FIELDS]
                cells += [("~" if ties[n][t, c] else "") + repr(float(values[n][t, c])) if kept else ""
                          for n in names]
                f.write("\t".join(cells) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    {"rust": rust, "fixture": fixture}[sys.argv[1]]()
