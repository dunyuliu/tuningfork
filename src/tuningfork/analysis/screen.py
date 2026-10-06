"""Several edits screened at one checkpoint against one baseline: which shift the target.

Each proposal's arm is compared with a baseline arm drawn beside it — the original wording,
sampled fresh, never the resample that suggested the target — by a two-sided Fisher exact
test on how often each did what the target describes. Seeds do not reproduce on llama.cpp,
so arms are independent samples: unpaired tests are the honest ones.

The family is every proposal planned before sampling, and Holm's step-down corrects over
it: valid however the tests depend on each other, as they do through their shared baseline.
An arm that was never drawn still counts, as p = 1, so stopping early cannot shrink the
family. And a screen that did not draw everything it planned gives rates without a verdict:
watching samples land and stopping when one looks good is how false discoveries are made.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from tuningfork.analysis.distribution import Target, cluster_letter
from tuningfork.analysis.normalize import Outcome
from tuningfork.analysis.stats import Rate, fisher_exact, holm, rate


@dataclass(slots=True)
class ArmResult:
    """One arm of a screen: how often it hit the target, and its test against the baseline."""

    name: str
    target: Rate
    planned: int
    drawn: int
    errors: int
    p_value: float | None = None
    adjusted_p: float | None = None
    #: up, down or same: the arm's rate against the baseline's.
    direction: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target": self.target.to_json(),
            "planned": self.planned,
            "drawn": self.drawn,
            "errors": self.errors,
            "p_value": self.p_value,
            "adjusted_p": self.adjusted_p,
            "direction": self.direction,
        }


@dataclass(slots=True)
class ScreenReport:
    target: str
    alpha: float
    baseline: ArmResult
    arms: list[ArmResult]
    #: Every arm drew every sample it planned. Only then is there a verdict.
    complete: bool
    #: Behaviours across every arm, lettered by how often they happened overall, with how
    #: many times each arm did each.
    behaviours: list[dict[str, Any]] = field(default_factory=list)

    @property
    def family(self) -> int:
        return len(self.arms)

    def survivors(self) -> list[ArmResult]:
        """Arms that shifted the target up, past the correction. Empty without a verdict."""
        return [arm for arm in self.arms if self._passed(arm) and arm.direction == "up"] if self.complete else []

    def worse(self) -> list[ArmResult]:
        """Arms that shifted it down, past the correction: a wording that made things worse."""
        return [arm for arm in self.arms if self._passed(arm) and arm.direction == "down"] if self.complete else []

    def _passed(self, arm: ArmResult) -> bool:
        return arm.adjusted_p is not None and arm.adjusted_p < self.alpha

    def to_json(self) -> dict[str, Any]:
        survivors, worse = {arm.name for arm in self.survivors()}, {arm.name for arm in self.worse()}
        return {
            "target": self.target,
            "alpha": self.alpha,
            "family": self.family,
            "complete": self.complete,
            "baseline": self.baseline.to_json(),
            "arms": [
                {**arm.to_json(), "survives": arm.name in survivors if self.complete else None,
                 "worse": arm.name in worse if self.complete else None}
                for arm in self.arms
            ],
            "survivors": sorted(survivors),
            "behaviours": self.behaviours,
        }


def _arm(name: str, outcomes: Sequence[Outcome], target: Target, planned: int) -> tuple[ArmResult, int, int]:
    kept = [outcome for outcome in outcomes if not outcome.is_error]
    hits = sum(1 for outcome in kept if target.matches(outcome))
    result = ArmResult(
        name=name, target=rate(hits, len(kept)), planned=planned, drawn=len(outcomes),
        errors=len(outcomes) - len(kept),
    )
    return result, hits, len(kept)


def compare_to_baseline(
    baseline: Sequence[Outcome],
    arms: Mapping[str, Sequence[Outcome]],
    *,
    target: Target,
    alpha: float = 0.05,
    planned: int,
) -> ScreenReport:
    """Each arm in `arms` (in the order planned) tested against `baseline` on `target`.

    `planned` is the samples each arm, the baseline included, was meant to draw: the
    family is every arm named, drawn or not."""
    base, base_hits, base_total = _arm("original", baseline, target, planned)
    results = []
    for name, outcomes in arms.items():
        result, hits, total = _arm(name, outcomes, target, planned)
        if total and base_total:
            result.p_value = fisher_exact(hits, total - hits, base_hits, base_total - base_hits)
            point, base_point = result.target.point, base.target.point
            result.direction = "up" if point > base_point else "down" if point < base_point else "same"
        results.append(result)
    # An arm not drawn is in the family as p = 1: stopping early never shrinks the correction.
    adjusted = holm([1.0 if result.p_value is None else result.p_value for result in results])
    for result, value in zip(results, adjusted):
        if result.p_value is not None:
            result.adjusted_p = value
    complete = base.drawn >= planned and all(result.drawn >= planned for result in results)
    return ScreenReport(
        target=target.description, alpha=alpha, baseline=base, arms=results, complete=complete,
        behaviours=_behaviours({"original": baseline, **arms}),
    )


def _behaviours(by_arm: Mapping[str, Sequence[Outcome]]) -> list[dict[str, Any]]:
    """Behaviours lettered over every arm together, each with how many times each arm did it."""
    counts: Counter[str] = Counter()
    first: dict[str, Outcome] = {}
    per_arm: dict[str, Counter[str]] = {}
    for name, outcomes in by_arm.items():
        mine = per_arm.setdefault(name, Counter())
        for outcome in outcomes:
            if outcome.is_error:
                continue
            counts[outcome.key] += 1
            mine[outcome.key] += 1
            first.setdefault(outcome.key, outcome)
    ordered = sorted(counts, key=lambda key: (-counts[key], first[key].label))
    return [
        {
            "letter": cluster_letter(position),
            "outcome": first[key].to_json(),
            "count": counts[key],
            "by_arm": {name: per_arm[name][key] for name in by_arm},
        }
        for position, key in enumerate(ordered)
    ]


def needed_to_survive(baseline_hits: int, n: int, family: int, alpha: float = 0.05) -> int | None:
    """How many of an arm's n samples must hit the target, against a baseline's
    `baseline_hits` of n, for the arm to survive the correction at all — as the strongest
    of `family`, it is tested at alpha / family. None when even n of n would not."""
    for hits in range(baseline_hits + 1, n + 1):
        if fisher_exact(hits, n - hits, baseline_hits, n - baseline_hits) * max(1, family) < alpha:
            return hits
    return None
