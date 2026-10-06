"""What a target would count, shown before any sample is drawn.

A target is declared before sampling. This sets it against the behaviours already seen at the
step — the run's own, other runs of the task at the same step, and every sample drawn there —
each marked as counting or not: a check of the declaration, so a person sees what it means
before trusting it. Never a result: these were seen before, and are not the samples the test
will be run on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from tuningfork.ops.common import OperationError
from tuningfork.store.db import Store

#: Where a behaviour was seen, in the order a page lists them.
SOURCES = ("this run", "other runs", "samples")


@dataclass(slots=True)
class Seen:
    outcome: Any
    #: How often it was seen, by where: `SOURCES`.
    counts: dict[str, int] = field(default_factory=dict)
    #: Whether the target counts it; None when no target was given.
    matches: bool | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            **self.outcome.to_json(),
            "seen": self.counts,
            "count": sum(self.counts.values()),
            "matches": self.matches,
        }


@dataclass(slots=True)
class SeenBehaviours:
    run_id: str
    step_idx: int
    #: The tools the step offered the model, for a target naming one.
    tools: list[str]
    normalizers: list[str]
    target: str | None
    behaviours: list[Seen]

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "step": self.step_idx,
            "tools": self.tools,
            "normalizers": self.normalizers,
            "target": self.target,
            "behaviours": [seen.to_json() for seen in self.behaviours],
        }


def offered_tools(request: dict[str, Any]) -> list[str]:
    """The tools a request offered, by name, in either wire format."""
    names = []
    for tool in request.get("tools") or []:
        if isinstance(tool, dict):
            name = (tool.get("function") or {}).get("name") or tool.get("name")
            if isinstance(name, str) and name not in names:
                names.append(name)
    return names


def seen_behaviours(
    store: Store,
    run: str,
    step: str,
    *,
    normalizers: Sequence[str] = (),
    target_match: str | None = None,
    target_rules: Sequence[dict[str, str]] | None = None,
    base: bool = False,
    runs: int = 200,
) -> SeenBehaviours:
    """The behaviours seen at a step, each marked as the target would count it: one described
    (`target_rules`), a pattern (`target_match`), or the run's own behaviour (`base`)."""
    from tuningfork.analysis.distribution import target_by_key, target_by_pattern, target_by_rules
    from tuningfork.analysis.normalize import outcome_for, recorded_outcome, validate_normalizers
    from tuningfork.experiment.forking import forking_outcome
    from tuningfork.ops.experiments import load

    try:
        lens = validate_normalizers(normalizers)
    except ValueError as error:
        raise OperationError(str(error), field="normalizers") from None
    checkpoint = load(store, run, step)
    seen: dict[str, Seen] = {}

    def note(outcome: Any, source: str) -> None:
        if outcome.is_error:
            return
        entry = seen.setdefault(outcome.key, Seen(outcome))
        entry.counts[source] = entry.counts.get(source, 0) + 1

    own = recorded_outcome(store.response(checkpoint.step), checkpoint.step.error, lens)
    note(own, "this run")
    if checkpoint.run.task_id:
        for other in store.runs(limit=runs, task_id=checkpoint.run.task_id):
            if other.id == checkpoint.run.id or other.is_thread or other.is_fork:
                continue
            found = store.step(other.id, checkpoint.step.idx)
            if found is not None:
                note(recorded_outcome(store.response(found), found.error, lens), "other runs")
    for experiment in store.experiments(run_id=checkpoint.run.id, step_idx=checkpoint.step.idx, limit=50):
        read = forking_outcome if experiment.kind == "forking-tokens" else outcome_for
        for sample in store.samples(experiment.id):
            if not sample.error:
                note(read(store.sample_response(sample), lens), "samples")

    try:
        target = (
            target_by_rules(target_rules) if target_rules
            else target_by_pattern(target_match) if target_match
            else target_by_key(own.key, f"the run's own behaviour: {own.label}") if base
            else None
        )
    except (ValueError, re.error) as error:
        raise OperationError(str(error), field="target_rules" if target_rules else "target_match") from None
    if target is not None:
        for entry in seen.values():
            entry.matches = target.matches(entry.outcome)
    return SeenBehaviours(
        run_id=checkpoint.run.id,
        step_idx=checkpoint.step.idx,
        tools=offered_tools(checkpoint.request),
        normalizers=lens,
        target=target.description if target else None,
        behaviours=sorted(seen.values(), key=lambda entry: -sum(entry.counts.values())),
    )
