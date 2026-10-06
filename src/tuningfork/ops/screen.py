"""Several wordings of a skill sampled at one checkpoint, each against the original.

A screen draws a fresh baseline arm — the checkpoint as recorded — then one arm per
proposal, the skill's text swapped for the proposal's wherever the model saw it
(`experiment.skill.substitute`). Everything a verdict depends on is fixed before the first
sample: the arms and their order, the target, alpha and the samples per arm are stored with
every arm, so the report is always recomputed from the store against exactly the test
planned (`analysis.screen` holds the statistics).

The baseline arm is drawn first and the arms one after another, in blocks, as an A/B's
are: on a server with prefix caching, interleaving them would reprocess the context from
the skill onward at every switch.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from tuningfork.ops.common import Cancel, Cancelled, OperationError, Progress
from tuningfork.ops.experiments import _cancellable, _stopped, _tracking, default_upstream, estimate, load, make_plan
from tuningfork.store.db import Store

#: Proposals one screen may hold. Each costs n samples, and each widens the correction.
MAX_ARMS = 6
BASELINE = "original"


@dataclass(slots=True)
class ScreenArm:
    """One proposal in a screen: its wording, and the request the model will see."""

    name: str
    text: str
    #: Where it came from: {"proposal", "variant_path", ...}, stored with its arm.
    meta: dict[str, Any]
    request: dict[str, Any]
    spans: list[Any]
    unapplied: list[str]
    changed: int

    def variant_info(self, original_text: str) -> dict[str, Any]:
        from tuningfork.experiment.skill import text_sha256

        return {
            **self.meta,
            "original_sha256": text_sha256(original_text),
            "variant_sha256": text_sha256(self.text),
            "spans": [asdict(span) for span in self.spans],
            "unapplied": self.unapplied,
        }


@dataclass(slots=True)
class ScreenPlan:
    checkpoint: Any
    plan: Any
    base_url: str
    original_text: str
    arms: list[ScreenArm]
    target: Any
    normalizers: list[str]
    alpha: float
    refinement: str | None = None
    #: Before any sample: what an arm must reach to survive, against what the baseline did.
    power: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def family(self) -> int:
        return len(self.arms)

    @property
    def total(self) -> int:
        return self.plan.n * (self.family + 1)

    def annotations(self) -> dict[str, Any]:
        """Stored with every arm: the screen as planned, before its first sample."""
        return {
            "target": self.target.spec,
            "refinement": self.refinement,
            "screen": {
                "arms": [arm.name for arm in self.arms],
                "per_arm": self.plan.n,
                "alpha": self.alpha,
                "family": self.family,
                "proposals": {arm.name: arm.meta.get("proposal") for arm in self.arms},
            },
        }

    def to_json(self) -> dict[str, Any]:
        guess = estimate(self.checkpoint, self.plan, arms=self.family + 1)
        return {
            "checkpoint": self.checkpoint.to_json(),
            "refinement": self.refinement,
            "base_url": self.base_url,
            "n": self.plan.n,
            "total": self.total,
            "target": self.target.description,
            "normalizers": self.normalizers,
            "alpha": self.alpha,
            "family": self.family,
            "arms": [
                {"name": arm.name, **{k: v for k, v in arm.meta.items() if k in ("proposal", "variant_path")},
                 "changed_lines": arm.changed, "unapplied": arm.unapplied}
                for arm in self.arms
            ],
            "power": self.power,
            "estimate": guess.to_json() if guess else None,
            "warnings": self.warnings,
        }


def prepare_screen(
    store: Store,
    run: str,
    step: str,
    *,
    original_text: str,
    arms: Sequence[tuple[str, str, dict[str, Any]]],
    target_spec: dict[str, Any],
    normalizers: Sequence[str] = (),
    alpha: float = 0.05,
    base_url: str | None = None,
    refinement: str | None = None,
    **plan: Any,
) -> ScreenPlan:
    """A screen at a checkpoint, checked before any sample is drawn: every arm must change
    what the model sees there, and at least one must be able to survive the correction.

    `arms` are (name, the proposal's whole text, where it came from), in the order the
    family is planned."""
    from tuningfork.analysis.distribution import target_from_spec
    from tuningfork.analysis.screen import needed_to_survive
    from tuningfork.endpoints import check_endpoint
    from tuningfork.experiment import experiment_outcomes
    from tuningfork.experiment.sampling import step_resample
    from tuningfork.experiment.skill import NothingToCompare, diagnose, locate, substitute, text_sha256, unapplied
    from tuningfork.ops.diff import line_diff
    from tuningfork.ops.trash import refuse_trashed

    checkpoint = load(store, run, step)
    refuse_trashed(checkpoint.run)
    if not arms:
        raise OperationError("a screen needs at least one proposal", code="bad_params", field="proposals")
    if len(arms) > MAX_ARMS:
        raise OperationError(
            f"a screen holds at most {MAX_ARMS} proposals: each costs n samples and widens the correction",
            code="bad_params", field="proposals",
        )
    names = [name for name, _, _ in arms]
    if len(set(names)) != len(names) or BASELINE in names:
        raise OperationError("each proposal needs its own name, and none may be called original", field="proposals")
    if not 0 < alpha < 1:
        raise OperationError("alpha is between 0 and 1", code="bad_params", field="alpha")
    if not locate(checkpoint.request, original_text):
        raise OperationError(
            f"the skill's text is not in the context at step {checkpoint.step.idx}, so no wording "
            "of it can change what the model saw there",
            detail=diagnose(checkpoint.request, original_text),
            field="step",
        )
    seen = {text_sha256(original_text): BASELINE}
    prepared: list[ScreenArm] = []
    warnings: list[str] = []
    for name, text, meta in arms:
        digest = text_sha256(text)
        if digest in seen:
            raise OperationError(f"{name} is the same text as {seen[digest]}", field="proposals")
        seen[digest] = name
        try:
            request, spans = substitute(checkpoint.request, original_text, text)
        except NothingToCompare:
            raise OperationError(
                f"{name} changes nothing the model sees at step {checkpoint.step.idx}",
                hint="an edit to the frontmatter, or to text the harness never sends, cannot be screened",
                field="proposals",
            ) from None
        left = unapplied(original_text, text, spans)
        if left:
            warnings.append(f"{name}: part of the edit cannot reach the model here — " + "; ".join(left))
        changed = line_diff(original_text, text)["changed"]
        prepared.append(ScreenArm(name, text, dict(meta), request, spans, left, changed))

    target = target_from_spec(target_spec)
    sampling = make_plan(normalizers=normalizers, **plan)
    n, family = sampling.n, len(prepared)
    # How the baseline did at this step before: a resample's count of the target, if any.
    power: dict[str, Any] = {"family": family, "alpha": alpha, "n": n}
    resample = step_resample(store, checkpoint)
    if resample is not None:
        outcomes = [o for o in experiment_outcomes(store, resample, list(normalizers)) if not o.is_error]
        if outcomes:
            hits = sum(1 for outcome in outcomes if target.matches(outcome))
            expected = round(hits / len(outcomes) * n)
            power.update(baseline={"hits": hits, "of": len(outcomes), "resample": resample.id},
                         needed=needed_to_survive(expected, n, family, alpha))
            if power["needed"] is None:
                raise OperationError(
                    f"no proposal could survive: the step already does this {hits} times in "
                    f"{len(outcomes)}, so even {n} of {n} would not pass the correction over {family}",
                    hint="screen fewer proposals, draw more samples per arm, or declare a target "
                    "the step seldom reaches",
                    field="n",
                )
    if "needed" not in power:
        power["needed"] = needed_to_survive(0, n, family, alpha)
        power["assumed"] = "a baseline that never reaches the target: the most an arm could be helped"
    if n < 20:
        warnings.append(f"at {n} samples an arm, only a large shift can survive the correction")
    return ScreenPlan(
        checkpoint=checkpoint,
        plan=sampling,
        base_url=check_endpoint(base_url or default_upstream(), home=store.home, purpose="sampling"),
        original_text=original_text,
        arms=prepared,
        target=target,
        normalizers=list(normalizers),
        alpha=alpha,
        refinement=refinement,
        power=power,
        warnings=warnings,
    )


@dataclass(slots=True)
class ScreenResult:
    group_id: str
    experiments: dict[str, Any]
    report: Any
    elapsed: float = 0.0

    @property
    def notes(self) -> list[str]:
        """What qualifies the samples, each said once though every arm keeps it: on Claude
        Code's Messages calls, that llama.cpp drew them from its cache of the prompt."""
        return list(dict.fromkeys(note for arm in self.experiments.values() for note in arm.notes or []))

    def to_json(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "experiments": {name: experiment.id for name, experiment in self.experiments.items()},
            "screen": self.report.to_json(),
            "notes": self.notes,
            "elapsed": round(self.elapsed, 1),
        }


async def execute_screen(
    store: Store,
    prepared: ScreenPlan,
    *,
    progress: Progress | None = None,
    cancel: Cancel | None = None,
) -> ScreenResult:
    import httpx

    from tuningfork.endpoints import guard_hooks
    from tuningfork.experiment import check_parity
    from tuningfork.experiment.sampling import finished, run_arm

    started = time.monotonic()
    created: list[Any] = []
    group_id = uuid.uuid4().hex
    annotations = prepared.annotations()

    async def work() -> None:
        async with httpx.AsyncClient(timeout=prepared.plan.timeout, event_hooks=guard_hooks(home=store.home)) as http:
            parity = await check_parity(http, prepared.base_url, prepared.checkpoint, prepared.plan)
            # Every arm stays running until the whole screen has: a finished baseline with
            # its proposals still to come is a screen under way, not one cut off.
            with finished(store, _tracking(created, progress)) as ended:
                for name, request, variant in (
                    (BASELINE, prepared.checkpoint.request, None),
                    *((arm.name, arm.request, arm.variant_info(prepared.original_text)) for arm in prepared.arms),
                ):
                    await run_arm(
                        store, prepared.checkpoint, prepared.plan,
                        base_url=prepared.base_url, request=request, kind="screen", arm=name,
                        client=http, group_id=group_id, variant=variant, parity=parity,
                        annotations=annotations, progress=ended,
                    )

    try:
        await _cancellable(work(), cancel)
    except Cancelled as stopped:
        _stopped(store, created, stopped.reason)
        # What was sampled is stored; the report says it is incomplete, with no verdict.
        raise Cancelled(stopped.reason, partial=screen_result(store, group_id) if created else None) from None
    result = screen_result(store, group_id)
    result.elapsed = time.monotonic() - started
    return result


def screen_result(store: Store, reference: str) -> ScreenResult:
    """A screen's report from the store, by its group or any of its arms: recomputed against
    the arms, target and alpha planned before its first sample."""
    from tuningfork.analysis.distribution import target_from_spec
    from tuningfork.analysis.screen import compare_to_baseline
    from tuningfork.experiment import experiment_outcomes

    found = store.get_experiment(reference)
    group_id = found.group_id if found is not None and found.group_id else reference
    arms = {arm.arm: arm for arm in store.experiment_group(group_id)}
    if BASELINE not in arms:
        raise OperationError(f"no screen matching {reference!r}", code="not_found")
    planned = arms[BASELINE].settings.get("screen") or {}
    target = target_from_spec(arms[BASELINE].settings["target"])
    report = compare_to_baseline(
        experiment_outcomes(store, arms[BASELINE]),
        {name: experiment_outcomes(store, arms[name]) if name in arms else [] for name in planned.get("arms", [])},
        target=target,
        alpha=float(planned.get("alpha", 0.05)),
        planned=int(planned.get("per_arm") or arms[BASELINE].settings.get("n") or 0),
    )
    return ScreenResult(group_id=group_id, experiments=arms, report=report)
