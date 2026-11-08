"""Deterministic rule that decides whether an experiment replaces the current best."""

from __future__ import annotations

import math

from pydantic import BaseModel, Field

from alphax.core.experiment import ExperimentResult


class MetricCriterion(BaseModel):
    metric: str
    higher_is_better: bool = True
    min_improvement: float = Field(default=0.0, description="Required improvement over the reference")


class MetricGuard(BaseModel):
    metric: str
    higher_is_better: bool = True
    tolerance: float = Field(default=0.0, ge=0.0, description="Largest acceptable deterioration")


class PromotionVerdict(BaseModel):
    promote: bool
    reason: str


class PromotionRule(BaseModel):
    """Promote when the primary metric improves enough and no guard metric deteriorates too much.

    Keeping this decision out of the language model makes the evolution of the best solution
    reproducible and auditable; the model explains results, the rule decides.
    """

    primary: MetricCriterion
    guards: list[MetricGuard] = Field(default_factory=list)

    def decide(self, candidate: ExperimentResult, reference: ExperimentResult | None) -> PromotionVerdict:
        value = candidate.get(self.primary.metric)
        if value is None or not math.isfinite(value):
            return PromotionVerdict(promote=False, reason=f"{self.primary.metric} is missing from the result")
        if reference is None:
            return PromotionVerdict(promote=True, reason=f"First result: {self.primary.metric}={value:.6f}")
        reasons: list[str] = []
        promote = True
        base = reference.get(self.primary.metric)
        if base is not None and math.isfinite(base):
            gain = self._signed(value - base, self.primary.higher_is_better)
            ok = gain > self.primary.min_improvement
            promote &= ok
            reasons.append(
                f"{self.primary.metric} {base:.6f} -> {value:.6f} (gain {gain:+.6f}, "
                f"required > {self.primary.min_improvement:g})"
            )
        for guard in self.guards:
            new, old = candidate.get(guard.metric), reference.get(guard.metric)
            if new is None or old is None or not (math.isfinite(new) and math.isfinite(old)):
                continue
            loss = -self._signed(new - old, guard.higher_is_better)
            within = loss <= guard.tolerance
            promote &= within
            reasons.append(
                f"guard {guard.metric} {old:.6f} -> {new:.6f} "
                f"({'ok' if within else 'violated'}, tolerance {guard.tolerance:g})"
            )
        return PromotionVerdict(promote=promote, reason="; ".join(reasons) or "no comparable metric")

    def describe(self) -> str:
        direction = "rises" if self.primary.higher_is_better else "falls"
        parts = [f"{self.primary.metric} {direction} by more than {self.primary.min_improvement:g}"]
        for guard in self.guards:
            worse = "falls" if guard.higher_is_better else "rises"
            parts.append(f"{guard.metric} {worse} by at most {guard.tolerance:g}")
        return " and ".join(parts)

    @staticmethod
    def _signed(delta: float, higher_is_better: bool) -> float:
        return delta if higher_is_better else -delta
