"""Reading recordings: what every front end shows of a run, computed once.

The CLI prints these and the HTTP API serves them as JSON; neither computes a result of its
own, so the two cannot disagree about one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from tuningfork.ops.common import OperationError
from tuningfork.ops.experiments import resolve_run, step_index
from tuningfork.store.db import Run, Step, Store


# --- runs -----------------------------------------------------------------------------------


@dataclass(slots=True)
class StepLine:
    """One step of a run, with what the model did there."""

    step: Step
    outcome: Any
    #: "none" where the model answered without reasoning though it reasons at other steps of
    #: the run, "reasoned", or None (`analysis.reasoning`).
    reasoning: str | None = None


def step_lines(store: Store, run: Run) -> list[StepLine]:
    from tuningfork.analysis.normalize import recorded_outcome
    from tuningfork.analysis.reasoning import ACTED, reasons, status

    steps = store.steps(run.id)
    run_reasons = reasons(step.reasoning_chars for step in steps if not step.served)
    lines = []
    for step in steps:
        outcome = recorded_outcome(store.response(step), step.error)
        lines.append(StepLine(step, outcome, status(
            step.reasoning_chars, reasons=run_reasons, acted=outcome.kind in ACTED and not step.served,
        )))
    return lines


def runs_judged(runs: list[Run], verdict: str) -> list[Run]:
    """The runs the task's check `verdict` ("passed" or "failed"), counted. Found writing the
    design-md guide: a run to study is a failing one, and nothing picked one. A run no check
    judged, cancelled, or not counted — a timeout, an exclusion — has no verdict here."""
    wanted = verdict == "passed"

    def judged(run: Run) -> bool:
        result = run.result or {}
        return (result.get("passed") is wanted and result.get("usable", True)
                and not result.get("cancelled"))

    return [run for run in runs if judged(run)]


def runs_calling(
    store: Store, tool: str, *, task: str | None = None, step: int | None = None, limit: int = 20
) -> list[Run]:
    """Runs, newest first, where the model called `tool` — at `step`, if one is given. Not
    every run takes the same path: one that wrote its file with a script never called the
    write tool at all, and one that ran code first may search later."""
    from tuningfork.ops.experiments import first_call, tools_called

    def called(run: Run) -> bool:
        if step is None:
            return first_call(store, run, tool) is not None
        recorded = store.step(run.id, step)
        return recorded is not None and tool in tools_called(store, recorded)

    return [run for run in store.runs(limit=1_000_000, task_id=task) if called(run)][:limit]


def run_detail(store: Store, reference: str) -> dict[str, Any]:
    """A run as its page shows it: each step with what the model did there, its threads and
    the launch that made it."""
    from tuningfork.ops.timeline import launch_json, run_json

    found = resolve_run(store, reference)
    steps = [
        {
            "idx": line.step.idx,
            "outcome": line.outcome.to_json(),
            "finish_reason": line.step.finish_reason,
            "timings": line.step.timings,
            "usage": line.step.usage,
            "reasoning_chars": line.step.reasoning_chars,
            "reasoning": line.reasoning,
            "edited": line.step.edited,
            "error": line.step.error,
            # A fork's replayed step, where a variant was swapped in, the workspace.
            "served": line.step.served,
            "replay": line.step.replay or None,
            "substitution": line.step.substitution or None,
            "workspace_commit": line.step.workspace_commit,
        }
        for line in step_lines(store, found)
    ]
    threads = [] if found.is_thread else store.threads_of(found.id)
    launch = store.get_launch(found.launch_id) if found.launch_id else None
    return {
        "run": run_json(found),
        "steps": steps,
        "threads": [run_json(thread) for thread in threads],
        "launch": launch_json(launch),
    }


# --- tasks ----------------------------------------------------------------------------------


def condition_json(condition: Any) -> dict[str, Any]:
    """One condition a task ran under — harness, model, skill version, server — with its
    runs' check pass rate."""
    import hashlib
    import json

    passed = condition.passed
    parts = [condition.harness, condition.model, condition.skill_version]
    if condition.server_id:
        # Only when known, so a condition recorded before servers were keeps its id.
        parts.append(condition.server_id)
    identity = json.dumps(parts)
    return {
        # Stable across reloads, so a page can link to one condition of a task.
        "id": hashlib.sha256(identity.encode()).hexdigest()[:8],
        "harness": condition.harness,
        "model": condition.model,
        "skill_version": condition.skill_version,
        "server_id": condition.server_id,
        "skill_file": condition.skill_file,
        "runs": len(condition.runs),
        "run_ids": condition.run_ids,
        "passed": passed.to_json() if passed else None,
        "uncounted": condition.uncounted,
        "first": condition.first,
        "latest": condition.latest,
    }


def task_summaries(runs: Sequence[Run], checks: dict[str, list[Any]] | None = None) -> list[dict[str, Any]]:
    """Runs grouped by task, with a check pass rate per condition.

    A task run under two wordings, models or harnesses has one rate for each: `passed` is
    given for the task as a whole only when all its runs share one condition. `checks` are
    each task's checks, the latest first (`Store.task_checks`): how its runs pass.
    """
    from tuningfork.analysis.conditions import conditions
    from tuningfork.specs import check_kind, describe_check

    checks = checks or {}
    groups: dict[str | None, list[Run]] = {}
    for found in runs:
        groups.setdefault(found.task_id, []).append(found)
    summaries = []
    for task_id, members in groups.items():
        found_conditions = [condition_json(found) for found in conditions(members)]
        summaries.append(
            {
                "task_id": task_id,
                "runs": len(members),
                "latest": max(found.started_at for found in members),
                "models": sorted({found.model for found in members if found.model}),
                "passed": found_conditions[0]["passed"] if len(found_conditions) == 1 else None,
                "uncounted": sum(found["uncounted"] for found in found_conditions),
                "conditions": found_conditions,
                "check": (
                    {"kind": check_kind(checks[task_id][0]), "text": describe_check(checks[task_id][0])}
                    if checks.get(task_id) else None
                ),
                "checks": len(checks.get(task_id) or []),
            }
        )
    return sorted(summaries, key=lambda summary: summary["latest"], reverse=True)


def tasks(store: Store) -> list[dict[str, Any]]:
    """Every task recorded, newest first, with its pass rate under each condition."""
    return task_summaries(store.runs(limit=100_000), store.task_checks())


# --- divergence -----------------------------------------------------------------------------


def divergence_runs(
    store: Store, *, refs: Sequence[str] = (), task: str | None = None, limit: int = 20
) -> tuple[list[str], str]:
    """The runs a divergence report compares, and what to call them."""
    if refs:
        return [resolve_run(store, ref).id for ref in refs], "selected runs"
    if task:
        return [run.id for run in store.runs(limit=limit, task_id=task)], task
    raise OperationError("pass run ids, or a task to compare every run of")


def divergence(
    store: Store, run_ids: Sequence[str], normalizers: Sequence[str] = ()
) -> tuple[Any, dict[str, dict[str, Any]]]:
    """The divergence report, and each run's stored verdict, for which behaviour passed."""
    from tuningfork.analysis.divergence import compare_runs

    if len(run_ids) < 2:
        raise OperationError("need at least two runs to compare")
    results = {}
    for run_id in run_ids:
        found = store.get_run(run_id)
        results[run_id] = found.result if found else {}
    return compare_runs(store, list(run_ids), normalizers), results


def divergence_report(
    store: Store,
    *,
    refs: Sequence[str] = (),
    task: str | None = None,
    normalizers: Sequence[str] = (),
    limit: int = 20,
) -> dict[str, Any]:
    """Where the runs named, or a task's latest runs, stopped agreeing, as JSON: each step's
    behaviours, and how often the runs behind each went on to pass."""
    from tuningfork.analysis.normalize import validate_normalizers

    try:
        lens = validate_normalizers(normalizers)
    except ValueError as error:
        raise OperationError(str(error), field="normalizers") from None
    if not refs and not task:
        raise OperationError("name a task, or the runs to compare", field="task")
    run_ids, _ = divergence_runs(store, refs=refs, task=task, limit=limit)
    if len(run_ids) < 2:
        raise OperationError(
            "a comparison needs at least two runs",
            hint="run the task again: `tuningfork batch <spec> -n 5`",
        )
    report, results = divergence(store, run_ids, lens)
    return report.to_json(results)


def behaviour_pass_rates(comparison: Any, results: dict[str, dict[str, Any]]) -> list[Any]:
    """For each behaviour at a step, how often the runs that did it went on to pass."""
    from tuningfork.analysis.conditions import pass_rate

    return [
        pass_rate(results.get(run_id, {}) for run_id in group.run_ids)
        for group in comparison.groups
    ]


# --- experiments ----------------------------------------------------------------------------


@dataclass(slots=True)
class ExperimentReport:
    """A stored experiment's result, as it reads again without drawing a sample.

    `kind` is how to show it: `resample`, `ab`, `interrupted` (an A/B with one arm),
    `ab_running` (an A/B whose second arm has not begun), `to_end` or `ab_to_end` (forks
    run to the end, judged by the task's check). `status` is how the experiment as a whole
    stands: running, done, cancelled, interrupted or failed.
    """

    experiment: Any
    kind: str
    checkpoint: Any = None
    distribution: Any = None
    comparison: Any = None
    #: Why a target declared before sampling cannot be re-tested now, if it cannot.
    target_note: str | None = None
    arms: dict[str, Any] = field(default_factory=dict)
    #: Arms that stopped short of the samples they were meant to draw: (arm, drawn, meant).
    partial: list[tuple[str, int, int]] = field(default_factory=list)
    to_end: Any = None
    status: str = "done"
    #: Arms still drawing samples: (arm, drawn, meant). Not stopped short — not yet done.
    running: list[tuple[str, int, int]] = field(default_factory=list)
    #: A forking-token probe's report (`tuningfork.analysis.forking`).
    forking: Any = None
    #: A screen's report (`tuningfork.analysis.screen`): each proposal against one baseline.
    screen: Any = None

    @property
    def interrupted_arm(self) -> str | None:
        return next(iter(self.arms), None) if self.kind == "interrupted" else None

    def to_json(self) -> dict[str, Any]:
        """The report as the API serves it; the web UI and, later, MCP read this."""
        report, found = self, self.experiment
        base = {
            "id": found.id,
            "kind": found.kind,
            "run_id": found.run_id,
            "step_idx": found.step_idx,
            "notes": found.notes,
            "changes": found.changes,
            "settings": found.settings,
            "reach": found.settings.get("reach", "step"),
            # Arms that stopped short of the samples they were meant to draw.
            "partial": [
                {"arm": arm, "drawn": drawn, "meant": meant}
                for arm, drawn, meant in report.partial
            ],
            "status": report.status,
            # Arms still drawing: not stopped short, not done yet.
            "running": [
                {"arm": arm, "drawn": drawn, "meant": meant}
                for arm, drawn, meant in report.running
            ],
            "job_id": found.job_id,
            "hold_id": found.hold_id,
            "origin": found.settings.get("origin"),
            "trashed_at": found.trashed_at,
        }
        if report.kind == "to_end":
            base["to_end"] = report.to_end.to_json()
            return base
        if report.kind == "screen":
            base["screen"] = report.screen.to_json()
            base["arms"] = {
                name: {
                    "id": arm.id,
                    "proposal": (arm.variant or {}).get("proposal"),
                    "variant_path": (arm.variant or {}).get("variant_path"),
                    "samples": arm.sample_count,
                    "status": arm_status(arm),
                }
                for name, arm in report.arms.items()
            }
            return base
        if report.kind == "forking":
            base["forking"] = report.forking.to_json()
            return base
        if report.kind == "resample":
            base["distribution"] = report.distribution.to_json()
            return base
        if report.kind in ("interrupted", "ab_running"):
            # Stopped before the second arm, or not at it yet: show what was sampled.
            if report.kind == "interrupted":
                base["interrupted"] = report.interrupted_arm
            else:
                base["running_arm"] = report.only_arm
            if report.to_end is not None:
                base["to_end"] = report.to_end.to_json()
            else:
                base["distribution"] = report.distribution.to_json()
            return base
        base["variant"] = report.arms["variant"].variant
        if report.kind == "ab_to_end":
            result = report.to_end
            base["to_end"] = {
                "original": result.original.to_json(),
                "variant": result.variant.to_json(),
                "p_value": result.p_value,
                "significant": result.significant,
                "alpha": result.alpha,
            }
            return base
        if report.target_note:
            base["notes"] = [*found.notes, report.target_note]
        base["comparison"] = report.comparison.to_json()
        return base
    @property
    def only_arm(self) -> str | None:
        """The one arm an A/B has so far: interrupted before its second, or still in its
        first."""
        return next(iter(self.arms), None) if self.kind in ("interrupted", "ab_running") else None


def experiment_report(
    store: Store, reference: str, *, lens: Sequence[str] | None = None, alpha: float = 0.05
) -> ExperimentReport:
    from tuningfork.analysis.distribution import compare_arms
    from tuningfork.experiment import (
        CheckpointError,
        declared_target,
        experiment_distribution,
        experiment_outcomes,
        experiment_reasoning,
        load_checkpoint,
    )
    from tuningfork.ops.forks import arm_summary, compare_to_end

    found = store.get_experiment(reference)
    if found is None:
        raise OperationError(f"no experiment matching {reference!r}")
    try:
        # None while it samples a call held at a breakpoint that is not recorded yet.
        checkpoint = (
            load_checkpoint(store, found.run_id, found.step_idx) if found.run_id is not None else None
        )
    except CheckpointError:
        checkpoint = None  # the run it sampled has gone; its samples have not
    arms = (
        {arm.arm: arm for arm in store.experiment_group(found.group_id)}
        if found.group_id
        else {found.arm: found}
    )
    statuses = {name: arm_status(arm) for name, arm in arms.items()}
    counts = {
        name: (arm.arm, arm.sample_count, int(arm.settings.get("n") or 0))
        for name, arm in arms.items()
    }
    running = [counts[name] for name in arms if statuses[name] == "running"]
    partial = [
        counts[name] for name in arms
        if statuses[name] != "running" and counts[name][1] < counts[name][2]
    ]
    report = ExperimentReport(
        experiment=found, kind="resample", checkpoint=checkpoint, arms=arms, partial=partial,
        status=overall_status(statuses.values()), running=running,
    )
    lens = list(lens) if lens else None
    if found.kind == "screen":
        from tuningfork.ops.screen import screen_result

        if lens:
            raise OperationError(
                "a screen is reported under the normalizers it was planned with: its target was "
                "declared under them"
            )
        report.kind, report.screen = "screen", screen_result(store, found.id).report
        return report
    if found.kind == "forking-tokens":
        from tuningfork.ops.forking import forking_result

        # Its lens and target are the ones declared before sampling: the report reads them.
        report.kind, report.forking = "forking", forking_result(store, found)
        return report
    to_end = found.settings.get("reach") == "end"
    if found.kind != "ab" or not found.group_id:
        if to_end:
            report.kind, report.to_end = "to_end", arm_summary(store, found)
        else:
            report.distribution = experiment_distribution(store, found, lens)
        return report
    if set(arms) != {"original", "variant"}:
        (arm,) = arms.values()
        report.kind = "ab_running" if statuses[arm.arm] == "running" else "interrupted"
        if to_end:
            report.to_end = arm_summary(store, arm)
        else:
            report.distribution = experiment_distribution(store, arm, lens)
        return report
    if to_end:
        report.kind = "ab_to_end"
        report.to_end = compare_to_end(
            arm_summary(store, arms["original"]), arm_summary(store, arms["variant"]), alpha
        )
        return report
    declared, report.target_note = declared_target(arms["original"], lens)
    report.kind = "ab"
    report.comparison = compare_arms(
        experiment_outcomes(store, arms["original"], lens),
        experiment_outcomes(store, arms["variant"], lens),
        target=declared,
        alpha=alpha,
        reasoning=(experiment_reasoning(store, arms["original"]), experiment_reasoning(store, arms["variant"])),
    )
    return report

def arm_status(experiment: Any) -> str:
    """How one experiment stands.

    Its stored status; for an experiment recorded before statuses existed, done if it drew
    every sample it was meant to and interrupted if not. A running one whose process has
    gone was cut off: it reads as interrupted.
    """
    from tuningfork.owners import owner_alive

    status = experiment.status
    if status is None:
        meant = int(experiment.settings.get("n") or 0)
        return "done" if experiment.sample_count >= meant else "interrupted"
    if status == "running" and owner_alive(experiment.owner) is False:
        return "interrupted"
    return status


def overall_status(statuses: Any) -> str:
    """An A/B's status from its arms': running while any arm is, else the worst ending."""
    statuses = list(statuses)
    for status in ("running", "failed", "cancelled", "interrupted"):
        if status in statuses:
            return status
    return "done"


def sample_view(store: Store, experiment: str, idx: int) -> dict[str, Any]:
    """One sample as the model gave it: what it said, the tools it called, its reasoning —
    which often shows the instruction it was weighing — and what that counts as."""
    found = store.get_experiment(experiment)
    if found is None:
        raise OperationError(f"no experiment matching {experiment!r}", code="not_found")
    sample = next((each for each in store.samples(found.id) if each.idx == idx), None)
    if sample is None:
        raise OperationError(f"experiment {found.id[:8]} has no sample {idx}", code="not_found")
    response = store.sample_response(sample) or {}
    message = ((response.get("choices") or [{}])[0].get("message")) or {}
    content = message.get("content")
    if isinstance(content, list):
        content = "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
    return {
        "experiment": found.id,
        "kind": found.kind,
        "arm": found.arm,
        "idx": sample.idx,
        "seed": sample.seed,
        "outcome": {"kind": sample.outcome_kind, "label": sample.outcome_label},
        "finish_reason": sample.finish_reason,
        "error": sample.error,
        "content": content or None,
        "tool_calls": [
            {"name": (call.get("function") or {}).get("name"), "arguments": (call.get("function") or {}).get("arguments")}
            for call in message.get("tool_calls") or []
        ],
        "reasoning": message.get("reasoning_content") or message.get("reasoning") or None,
        "usage": sample.usage or None,
        # Run to the end, a sample is a fork: whether the task's check passed there.
        "run_id": sample.run_id,
        "passed": sample.passed,
        "forced": sample.forced,
    }


def experiment_list(store: Store, run: str | None = None, *, limit: int = 200) -> list[dict[str, Any]]:
    """Experiments, newest first — every one, or those at a run's steps."""
    run_id = resolve_run(store, run).id if run else None
    return [
        {
            "id": experiment.id,
            "kind": experiment.kind,
            "reach": experiment.settings.get("reach", "step"),
            "arm": experiment.arm,
            "group_id": experiment.group_id,
            "run_id": experiment.run_id,
            "step_idx": experiment.step_idx,
            "hold_id": experiment.hold_id,
            "samples": experiment.sample_count,
            "n": experiment.settings.get("n"),
            "target": (experiment.settings.get("target") or {}).get("description"),
            "created_at": experiment.created_at,
            "status": arm_status(experiment),
            "job_id": experiment.job_id,
        }
        for experiment in store.experiments(run_id=run_id, limit=limit)
    ]


#: Names for the workspace at the ends of a launch, rather than before one of its calls.
START, END = ("start", "initial"), ("end", "final")


@dataclass(slots=True)
class WorkspaceView:
    """A run's workspace at one point, and what changed since the snapshot before it."""

    run: Run
    #: The step whose call found the workspace like this; None at the start or the end.
    step: int | None
    #: Where, in words: "before step 2", "at the start", "at the end".
    where: str
    commit: str
    files: list[Any]
    #: The snapshot before this one, and each file that changed since, as (status, path).
    before: str | None = None
    changes: list[tuple[str, str]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run.id,
            "step": self.step,
            "where": self.where,
            "commit": self.commit,
            "files": [{"path": e.path, "mode": e.mode, "size": e.size} for e in self.files],
            "before": self.before,
            "changes": [{"status": s, "path": p} for s, p in self.changes],
        }


def workspace_at(store: Store, run: str, where: str) -> WorkspaceView:
    """The workspace a run's call at `where` found — a step, by number or tool — or the one
    it started with (`start`) or ended with (`end`)."""
    from tuningfork.snapshots import Snapshots, SnapshotError, available

    found = resolve_run(store, run)
    launch = store.get_launch(found.launch_id) if found.launch_id else None
    hint = "workspaces are snapshotted for runs tuningfork launches, from Phase 4 on"
    if not available():
        raise OperationError("workspace snapshots need git, which is not on PATH")
    step: int | None = None
    if where in START:
        commit, before, words = (launch.initial_commit if launch else None), None, "at the start"
    elif where in END:
        steps = store.steps(found.id)
        commit = launch.final_commit if launch else None
        before = next(
            (s.workspace_commit for s in reversed(steps) if s.workspace_commit),
            launch.initial_commit if launch else None,
        )
        words = "at the end"
    else:
        step = step_index(store, found, where)
        recorded = store.step(found.id, step)
        if recorded is None:
            raise OperationError(f"run {found.id[:8]} has no step {step}")
        commit, words = recorded.workspace_commit, f"before step {step}"
        previous = store.step(found.id, step - 1) if step > 0 else None
        before = (previous.workspace_commit if previous else None) or (
            launch.initial_commit if launch else None
        )
    if not commit:
        raise OperationError(f"run {found.id[:8]} has no snapshot of its workspace {words}", hint=hint)
    snapshots = Snapshots(store.home)
    try:
        return WorkspaceView(
            run=found,
            step=step,
            where=words,
            commit=commit,
            files=snapshots.files(commit),
            before=before,
            changes=snapshots.diff(before, commit) if before else [],
        )
    except SnapshotError as exc:
        raise OperationError(f"cannot read the snapshot: {exc}") from None


def workspace_patch(store: Store, view: WorkspaceView, path: str | None = None) -> str:
    """What changed since the snapshot before, as a unified diff."""
    from tuningfork.snapshots import Snapshots

    return Snapshots(store.home).patch(view.before, view.commit, path)


def restore_workspace(store: Store, view: WorkspaceView, target: Path) -> None:
    """Write the snapshot's files into `target`."""
    from tuningfork.snapshots import Snapshots

    target = Path(target)
    if target.exists() and any(target.iterdir()):
        raise OperationError(f"{target} is not empty", hint="restore into a new or empty folder")
    Snapshots(store.home).restore(view.commit, target)
