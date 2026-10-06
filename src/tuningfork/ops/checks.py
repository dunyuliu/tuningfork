"""Trying a check before trusting it.

A task's check decides every verdict a batch or a fork counts, and a check that is subtly
wrong — a key misspelled, a pattern that matches too much, a command that passes whatever
happens — skews every pass rate built on it, silently. So a check can be tried as it stands,
unsaved: against what a recorded run left behind (its workspace as the run ended, restored
from its snapshot, and its final reply), or against a reply and files a person writes. It
runs in a throwaway folder, and nothing is recorded: a run's own verdict never changes.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from tuningfork.ops.common import OperationError
from tuningfork.store.db import Run, Store

#: How long a tried check may run, unless asked otherwise: one that takes longer is worth
#: knowing about before a batch runs it once a run.
DEFAULT_TIMEOUT = 60.0
#: How much of a reply, and of a workspace's file list, a trial reports.
REPLY_SHOWN = 4000
FILES_SHOWN = 200


@dataclass(slots=True)
class CheckTrialPlan:
    """A check, and what it will be tried against: decided before anything runs."""

    check: Any
    #: The recorded run it is tried against; None for a reply and files written here.
    run: Run | None = None
    #: The run's workspace as it ended, in the snapshot store; None when it kept none.
    commit: str | None = None
    #: What is judged as the reply: the run's final reply, or the one written.
    reply: str = ""
    #: Files for the workspace, for a trial without a run.
    files: list[dict[str, str]] = field(default_factory=list)
    timeout: float = DEFAULT_TIMEOUT
    #: The verdict the run got when it ran, set beside this one.
    recorded: dict[str, Any] | None = None
    #: Whether the check is the run's own, as it was launched.
    own: bool = False
    #: The folder of the spec the run was launched from: `$TUNINGFORK_SPEC_DIR` for the check.
    spec_dir: str | None = None

    def to_json(self) -> dict[str, Any]:
        from tuningfork.specs import check_kind, describe_check

        return {
            "check": self.check,
            "check_kind": check_kind(self.check),
            "check_text": describe_check(self.check),
            "own": self.own,
            "run": {
                "id": self.run.id, "label": self.run.label, "task_id": self.run.task_id,
                "harness": self.run.harness,
            } if self.run else None,
            "workspace": "run" if self.commit else "files" if self.files else "empty",
            "files": [entry["path"] for entry in self.files],
            "reply_chars": len(self.reply),
            "recorded": self.recorded,
            "timeout": self.timeout,
        }


@dataclass(slots=True)
class CheckTrial:
    """How a check judged what it was tried against."""

    plan: CheckTrialPlan
    passed: bool
    #: The command's exit code; None for a reply check, which runs nothing.
    returncode: int | None
    output: str
    #: The workspace's files as the check found them.
    found: list[str]

    @property
    def agrees(self) -> bool | None:
        """Whether this is the verdict the run got when it ran; None when it got none."""
        recorded = (self.plan.recorded or {}).get("passed")
        return None if recorded is None else recorded == self.passed

    def to_json(self) -> dict[str, Any]:
        reply = self.plan.reply
        return {
            **self.plan.to_json(),
            "passed": self.passed,
            "returncode": self.returncode,
            "output": self.output,
            "found": self.found[:FILES_SHOWN],
            "found_more": max(0, len(self.found) - FILES_SHOWN),
            "reply": reply if len(reply) <= REPLY_SHOWN else f"{reply[:REPLY_SHOWN]}… (+{len(reply) - REPLY_SHOWN} chars)",
            "agrees": self.agrees,
        }


def prepare_check_trial(
    store: Store,
    check: Any = None,
    *,
    run: str | None = None,
    reply: str = "",
    files: Sequence[dict[str, str]] = (),
    timeout: float = DEFAULT_TIMEOUT,
) -> CheckTrialPlan:
    """What a check will be tried against, checked before it runs: a recorded run (with no
    check named, the run's own), or a reply and files written here."""
    from tuningfork.ops.experiments import resolve_run
    from tuningfork.runner import final_reply
    from tuningfork.snapshots import available
    from tuningfork.specs import Eval, SpecError, check_kind, check_problems, inside

    found: Run | None = None
    commit = None
    recorded = None
    own = False
    spec_dir = None
    if run:
        found = resolve_run(store, run)
        if found.is_thread:
            raise OperationError(
                f"run {found.id[:8]} is a {found.role} thread: a check judges the run its launch made",
                hint=f"try it on run {found.parent_run_id[:8]}" if found.parent_run_id else None,
                field="run",
            )
        if files:
            raise OperationError(
                "a trial against a run uses the run's own workspace: files are for a trial without one",
                field="files",
            )
        launch = store.get_launch(found.launch_id) if found.launch_id else None
        if check is None:
            if launch is None or not launch.eval:
                raise OperationError(
                    f"run {found.id[:8]} was not launched by tuningfork, so it has no check of its own",
                    hint="name the check to try", field="check",
                )
            check = Eval.from_dict(launch.eval).tuningfork.check
            if check is None:
                raise OperationError(f"run {found.id[:8]}'s task has no check", hint="name one to try", field="check")
            own = True
        commit = launch.final_commit if launch else None
        # A fork's launch names no spec file; its eval keeps the folder, as a batch's does.
        if launch is not None and launch.spec_path:
            spec_dir = str(Path(launch.spec_path).parent)
        elif launch is not None and isinstance(launch.eval, dict) and launch.eval.get("spec_dir"):
            spec_dir = str(launch.eval["spec_dir"])
        reply = final_reply(store, found.id)
        if found.result:
            recorded = {
                "passed": found.result.get("passed"),
                "usable": found.result.get("usable"),
                "check_output": (found.result.get("check_output") or "")[-2000:],
            }
    problems = check_problems(check)
    if problems:
        raise OperationError(f"the check {problems[0].message}", field="check")
    kind = check_kind(check)
    if kind is None:
        raise OperationError("name the check to try: a shell command, or what the reply should hold", field="check")
    if kind == "shell" and found is not None and commit is None:
        raise OperationError(
            f"run {found.id[:8]} kept no workspace for a command to run in",
            hint="only runs tuningfork launched keep one, from Phase 4 on; a reply check needs none",
            field="run",
        )
    if commit and not available():
        raise OperationError("a run's workspace is restored with git, which is not on PATH", field="run")
    written = []
    for entry in files:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not isinstance(entry.get("content", ""), str):
            raise OperationError("each file is {path, content}, both text", field="files")
        try:
            inside(Path("/workspace"), entry["path"], what="file")
        except SpecError as exc:
            raise OperationError(str(exc), field="files") from None
        written.append({"path": entry["path"], "content": entry.get("content", "")})
    return CheckTrialPlan(
        check=check, run=found, commit=commit, reply=reply or "", files=written,
        timeout=timeout, recorded=recorded, own=own, spec_dir=spec_dir,
    )


def execute_check_trial(store: Store, plan: CheckTrialPlan) -> CheckTrial:
    """Run the check in a throwaway folder laid out as a run's: the workspace, and the reply
    in a file beside it (`$TUNINGFORK_REPLY`). Nothing is kept, and nothing recorded."""
    from tuningfork.snapshots import Snapshots
    from tuningfork.specs import inside, run_check

    with tempfile.TemporaryDirectory(prefix="tuningfork-check-") as scratch:
        root = Path(scratch)
        workspace = root / "workspace"
        if plan.commit:
            Snapshots(store.home).restore(plan.commit, workspace)
        else:
            workspace.mkdir()
            for entry in plan.files:
                target = inside(workspace, entry["path"], what="file")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(entry["content"], encoding="utf-8")
        found = sorted(path.relative_to(workspace).as_posix() for path in workspace.rglob("*") if path.is_file())
        reply_file = root / "reply.txt"
        reply_file.write_text(plan.reply, encoding="utf-8")
        result = run_check(plan.check, workspace, plan.timeout, reply=plan.reply, reply_file=reply_file,
                           spec_dir=Path(plan.spec_dir) if plan.spec_dir else None)
    return CheckTrial(plan=plan, passed=result.passed, returncode=result.returncode, output=result.output, found=found)


def try_check(store: Store, check: Any = None, **options: Any) -> CheckTrial:
    """Prepare a trial and run it: what the CLI does in one go."""
    return execute_check_trial(store, prepare_check_trial(store, check, **options))
