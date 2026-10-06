"""The tools an agent is offered: what each is for, its arguments, and what it answers.

Reads answer from the recordings, in this process — no proxy needed. Each `start_*` tool
is a job kind from the catalog (`mcp.schema`), sent to a proxy's job runner; `server.py`
starts it and waits. Every answer is a compact view (`ops.compact`) opened by sentences
that say it statistics first: an agent should never mistake noise for an improvement.

Nothing here needs the MCP SDK: the registry and its handlers are plain Python, and
`server.py` puts them on the wire.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from tuningfork.ops import compact
from tuningfork.ops.catalog import RULE_LIST, Kind
from tuningfork.ops.common import OperationError, params_schema, parse_params

STEP_HELP = "The step: its index, or a tool's name for the first step whose model called it."
RUN_HELP = "The run: its id, an 8-character prefix of it, or its label."
#: A job id or a prefix of one: hexadecimal, as every id tuningfork makes.
JOB_ID = re.compile(r"[0-9a-f]{4,32}")


def _help(text: str, **extra: Any) -> dict[str, Any]:
    return {"help": text, **extra}


@dataclass(slots=True)
class Answer:
    """What a tool answers: JSON, and the sentences a reader should see first."""

    data: dict[str, Any]
    lines: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Context:
    """What a tool can use: the recordings, and where work is sent."""

    store: Any
    jobs: Any  # mcp.backend.Jobs
    max_wait: int = 120


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    title: str
    description: str
    #: Its arguments, strictly parsed (a dataclass), for every tool but `start_*`.
    args: type | None = None
    #: A read, answered in a worker thread: (context, parsed arguments) -> Answer.
    read: Callable[[Context, Any], Answer] | None = None
    #: A write that calls no model, in a worker thread: (context, arguments, who) -> Answer.
    act: Callable[[Context, Any, dict[str, Any]], Answer] | None = None
    #: The job kind a `start_*` tool starts.
    kind: Kind | None = None
    #: Calls the model, runs nothing on the machine — or (`machine`) runs a harness or check.
    machine: bool = False
    #: Changes something: starts or stops work.
    acts: bool = False

    def annotations(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "readOnlyHint": not self.acts,
            "destructiveHint": False,
            "idempotentHint": not self.acts or self.kind is not None,
            "openWorldHint": self.machine,
        }


def read_args(cls: type, arguments: dict[str, Any] | None) -> Any:
    """A tool's arguments, strictly: unknown names and wrong types refused, by field. A
    model writes a step as 3; the parser takes "3"."""
    data = dict(arguments or {})
    if isinstance(data.get("step"), int) and not isinstance(data.get("step"), bool):
        data["step"] = str(data["step"])
    return parse_params(cls, data)


def args_schema(cls: type) -> dict[str, Any]:
    schema = params_schema(cls)
    if "step" in schema["properties"]:
        schema["properties"]["step"] = {"type": ["integer", "string"], "description": STEP_HELP}
    return schema


# --- arguments ---------------------------------------------------------------------------------


@dataclass
class NoArgs:
    pass


@dataclass
class RunsArgs:
    task: str | None = field(default=None, metadata=_help("Only this task's runs, by its id."))
    called: str | None = field(default=None, metadata=_help("Only runs whose model called this tool somewhere."))
    at_step: int | None = field(default=None, metadata=_help("With `called`: only runs that called it at this step.", minimum=0))
    limit: int = field(default=20, metadata=_help("How many, newest first.", minimum=1, maximum=100))


@dataclass
class RunArgs:
    run: str = field(metadata=_help(RUN_HELP))


@dataclass
class StepArgs:
    run: str = field(metadata=_help(RUN_HELP))
    step: str = field(metadata=_help(STEP_HELP))
    full: bool = field(default=False, metadata=_help("Every message of the context, clipped less: much longer."))
    last: int = field(default=6, metadata=_help("How many messages before the answer to show.", minimum=1, maximum=50))


@dataclass
class DivergenceArgs:
    task: str | None = field(default=None, metadata=_help("Compare this task's latest runs."))
    runs: list[str] = field(default_factory=list, metadata=_help("Or compare these runs, by id or prefix."))
    normalizers: list[str] = field(default_factory=list, metadata=_help(
        "How behaviours are compared: tool_call_json, python_ast, json_content, or tool_name "
        "(by which tools were called, ignoring arguments). None named: the defaults."))
    limit: int = field(default=20, metadata=_help("With `task`: how many of its latest runs.", minimum=2, maximum=200))
    all_steps: bool = field(default=False, metadata=_help("Every step, not only where the runs split."))


@dataclass
class BehavioursArgs:
    run: str = field(metadata=_help(RUN_HELP))
    step: str = field(metadata=_help(STEP_HELP))
    normalizers: list[str] = field(default_factory=list, metadata=_help("How behaviours are compared; none named: the defaults."))
    target_match: str | None = field(default=None, metadata=_help(
        "A regular expression over each behaviour's label: which ones a target would count."))
    target_rules: list[dict[str, str]] = field(default_factory=list, metadata=_help(
        "Or a target described: conditions that must all hold, each {if: calls | not_calls | contains | "
        "not_contains, value}.", schema=RULE_LIST))


@dataclass
class ExperimentsArgs:
    run: str | None = field(default=None, metadata=_help("Only experiments at this run's steps."))
    limit: int = field(default=30, metadata=_help("How many, newest first.", minimum=1, maximum=200))


@dataclass
class ExperimentArgs:
    experiment: str = field(metadata=_help("The experiment: its id or an 8-character prefix."))


@dataclass
class TokensArgs:
    run: str = field(metadata=_help(RUN_HELP))
    step: str = field(metadata=_help(STEP_HELP))
    positions: list[int] = field(default_factory=list, metadata=_help("Token positions to show in full, as the hotspots number them."))
    hotspots: int = field(default=10, metadata=_help("How many of the least certain tokens to show.", minimum=1, maximum=50))
    stream: str | None = field(default=None, metadata=_help("A sample's token stream, by its full id. Default: the answer's own."))


@dataclass
class ScenariosArgs:
    name: str | None = field(default=None, metadata=_help("One scenario in full, by name."))


@dataclass
class RefinementArgs:
    run: str = field(metadata=_help(RUN_HELP))
    step: str = field(metadata=_help(STEP_HELP))
    target: str | None = field(default=None, metadata=_help("What the model should do here: a behaviour's letter from this step's latest resample."))
    target_match: str | None = field(default=None, metadata=_help("Or a regular expression over what the model did."))
    target_rules: list[dict[str, str]] = field(default_factory=list, metadata=_help(
        "Or conditions in words that must all hold, each {if: calls | not_calls | contains | not_contains, value}.",
        schema=RULE_LIST))


@dataclass
class RefinementRef:
    refinement: str = field(metadata=_help("The refinement: its id or an 8-character prefix."))
    evidence: bool = field(default=True, metadata=_help("With the packet of evidence a proposal should cite."))


@dataclass
class ProposeEditArgs:
    refinement: str = field(metadata=_help("The refinement: its id or an 8-character prefix."))
    edits: list[dict[str, str]] = field(default_factory=list, metadata=_help(
        "1 to 3 replacements, each {old, new}: old is copied exactly from the skill and occurs once in it.",
        schema={"type": "array", "minItems": 1, "maxItems": 3, "items": {
            "type": "object", "properties": {"old": {"type": "string"}, "new": {"type": "string"}},
            "required": ["old", "new"], "additionalProperties": False}}))
    rationale: str = field(default="", metadata=_help("Why: cite the evidence in the packet."))
    packet: str | None = field(default=None, metadata=_help("The packet the edit came from, as show_refinement named it."))


@dataclass
class JobArgs:
    job: str = field(metadata=_help("The job: its id or an 8-character prefix."))
    wait_seconds: int = field(default=0, metadata=_help("How long to wait for it to end.", minimum=0, maximum=120))
    after: int | None = field(default=None, metadata=_help("Also its progress events after this sequence number.", minimum=0))


@dataclass
class JobsArgs:
    limit: int = field(default=20, metadata=_help("How many, newest first.", minimum=1, maximum=100))


@dataclass
class CancelArgs:
    job: str = field(metadata=_help("The job: its id or an 8-character prefix."))


# --- what an agent is never shown ------------------------------------------------------------------
#
# Runs of a held-out task judge an edit. A person's validation never records them, but runs
# recorded before it existed, or by a person's explicit --include-held-out, may be in the
# recordings: every read below leaves them out, and refuses to open one by name.


def hidden_tasks(context: Context) -> set[str]:
    return context.store.held_out_tasks()


def refuse_hidden(context: Context, run: Any, hidden: set[str] | None = None) -> Any:
    """The run, unless it is of a held-out task."""
    if run is not None and run.task_id in (hidden_tasks(context) if hidden is None else hidden):
        raise OperationError(
            f"run {run.id[:8]} is of a held-out task: its runs are the person's evidence for judging "
            "an edit, never an agent's",
            code="held_out", field="run",
        )
    return run


def visible_run(context: Context, reference: str) -> Any:
    from tuningfork.ops.experiments import resolve_run

    return refuse_hidden(context, resolve_run(context.store, reference))


def hidden_run_ids(context: Context, run_ids: Any, hidden: set[str]) -> set[str]:
    found = set()
    for run_id in {run_id for run_id in run_ids if run_id}:
        run = context.store.get_run(run_id)
        if run is not None and run.task_id in hidden:
            found.add(run_id)
    return found


def without_held_out_evals(spec: Any) -> Any:
    """A spec as an agent sees it: its held-out evals left out, but for a count."""
    if not isinstance(spec, dict) or not isinstance(spec.get("evals"), list):
        return spec
    kept = [item for item in spec["evals"]
            if not (isinstance(item, dict) and (item.get("tuningfork") or {}).get("held_out") is True)]
    shown = {**spec, "evals": kept}
    if len(kept) != len(spec["evals"]):
        shown["held_out_evals"] = len(spec["evals"]) - len(kept)
    return shown


# --- reads -------------------------------------------------------------------------------------


def list_tasks(context: Context, args: NoArgs) -> Answer:
    from tuningfork.ops import inspect

    hidden = hidden_tasks(context)
    found = [compact.task(summary) for summary in inspect.tasks(context.store)]
    found = [summary for summary in found if summary.get("task_id") not in hidden]
    if not found:
        return Answer({"tasks": []}, [
            "no runs are recorded yet: `tuningfork batch <spec> -n 5`, or the web UI's Run a task, makes some",
        ])
    lines = [line for summary in found[:10] for line in compact.task_lines(summary)]
    if len(found) > 10:
        lines.append(f"… and {len(found) - 10} more task(s) below")
    return Answer({"tasks": compact.capped(found, 30)}, lines)


def list_runs(context: Context, args: RunsArgs) -> Answer:
    from tuningfork.ops.inspect import runs_calling
    from tuningfork.ops.timeline import run_json

    if args.at_step is not None and not args.called:
        raise OperationError("at_step narrows `called`: name the tool too", code="bad_params", field="at_step")
    hidden = hidden_tasks(context)
    if args.task in hidden:
        raise OperationError(f"{args.task} is a held-out task: its runs are the person's evidence, never an agent's",
                             code="held_out", field="task")
    if args.called:
        runs = runs_calling(context.store, args.called, task=args.task, step=args.at_step, limit=args.limit)
    else:
        runs = context.store.runs(limit=args.limit, task_id=args.task)
    views = [compact.run(run_json(found)) for found in runs if found.task_id not in hidden]
    judged = [view for view in views if (view.get("result") or {}).get("passed") is not None]
    passed = sum(1 for view in judged if view["result"]["passed"])
    lines = [f"{len(views)} run(s), newest first" + (f"; of those judged by their check, {passed}/{len(judged)} passed" if judged else "")]
    return Answer({"runs": views}, lines)


def show_run(context: Context, args: RunArgs) -> Answer:
    from tuningfork.ops import inspect

    found = visible_run(context, args.run)
    view = compact.run_detail(inspect.run_detail(context.store, found.id))
    # Where the spec is would lead an agent with file access to the evals held out from it.
    if view.get("launch"):
        view["launch"].pop("spec", None)
    run = view["run"]
    result = run.get("result") or {}
    verdict = {True: "passed its check", False: "failed its check"}.get(result.get("passed"), "has no verdict")
    steps = view["steps"]
    count = steps["total"] if isinstance(steps, dict) else len(steps)
    return Answer(view, [f"run {run['id']} of {run.get('task_id') or 'no task'}: {count} step(s); it {verdict}"])


def show_step(context: Context, args: StepArgs) -> Answer:
    from tuningfork.ops.experiments import step_index
    from tuningfork.ops.timeline import step_detail

    run = visible_run(context, args.run)
    index = step_index(context.store, run, args.step)
    view = compact.step(step_detail(context.store, run.id, index).to_json(), full=args.full, last=args.last)
    lines = [f"step {index} of run {run.id[:8]}: {view['context']['messages']} message(s) in its context;"
             f" the model did: {(view.get('did') or {}).get('label')}"]
    for span in view["context"]["skills"]:
        where = f"{span['file']}, from line {span['first_line']}" if span.get("file") else "not matched to a file"
        lines.append(f"  skill {span['name']!r}: {span['lines']} line(s) in message {span['message_index']} ({where})")
    return Answer(view, lines)


def find_divergence(context: Context, args: DivergenceArgs) -> Answer:
    from tuningfork.ops import inspect

    hidden = hidden_tasks(context)
    if args.task in hidden:
        raise OperationError(f"{args.task} is a held-out task: its runs are the person's evidence, never an agent's",
                             code="held_out", field="task")
    for reference in args.runs or []:
        visible_run(context, reference)
    report = inspect.divergence_report(
        context.store, refs=args.runs, task=args.task, normalizers=args.normalizers, limit=args.limit
    )
    view = compact.divergence(report, all_steps=args.all_steps)
    if view.get("task_id") in hidden or view.get("task") in hidden:
        raise OperationError("those runs are of a held-out task: the person's evidence, never an agent's",
                             code="held_out", field="task")
    lines = compact.divergence_lines(view)
    if view.get("divergence_index") is not None:
        lines.append("next: show_step at that step, then start_resample there to measure each behaviour")
    return Answer(view, lines)


def step_behaviours(context: Context, args: BehavioursArgs) -> Answer:
    from tuningfork.ops.experiments import step_targets
    from tuningfork.ops.targets import seen_behaviours

    visible_run(context, args.run)
    seen = seen_behaviours(
        context.store, args.run, args.step, normalizers=args.normalizers,
        target_match=args.target_match, target_rules=args.target_rules or None,
    ).to_json()
    try:
        targets = step_targets(context.store, args.run, args.step).to_json()
    except OperationError:
        targets = None
    view = compact.behaviours(seen, targets)
    found = seen.get("behaviours") or []
    lines = [f"{len(found)} distinct behaviour(s) seen at step {seen.get('step')} (in runs, forks and samples)"]
    if seen.get("target"):
        counted = sum(1 for entry in found if entry.get("matches"))
        lines.append(f"the target {seen['target']!r} counts {counted} of them")
    if view.get("letters"):
        lines.append(f"resample {view['letters']['resample']} lettered them: an A/B can declare `target` as one of those letters")
    else:
        lines.append("no resample at this step yet: start_resample letters its behaviours, for an A/B's target")
    return Answer(view, lines)


def show_skill(context: Context, args: RunArgs) -> Answer:
    from tuningfork.ops.experiments import recorded_skill
    from tuningfork.ops.variants import list_variants

    visible_run(context, args.run)
    recorded = recorded_skill(context.store, args.run).to_json()
    variants = [variant.to_json() for variant in list_variants(context.store.home, recorded.get("skill"))]
    view = compact.skill(recorded, variants)
    lines = [f"skill {recorded.get('skill')!r} as run {view['run']} was given it: {recorded.get('path') or 'no file kept'}"]
    if variants:
        lines.append(f"{len(variants)} saved variant(s): " + ", ".join(v.get("name", "?") for v in view["variants"]))
    lines.append("to test an edit: start_ab with variant_text (the whole edited file) and variant_name")
    return Answer(view, lines)


def list_experiments(context: Context, args: ExperimentsArgs) -> Answer:
    from tuningfork.ops import inspect

    if args.run:
        visible_run(context, args.run)
    listed = inspect.experiment_list(context.store, args.run, limit=args.limit)
    hidden = hidden_tasks(context)
    if hidden:
        held = hidden_run_ids(context, (entry.get("run_id") for entry in listed), hidden)
        listed = [entry for entry in listed if entry.get("run_id") not in held]
    shown = compact.experiments(listed, limit=args.limit)
    count = shown["total"] if isinstance(shown, dict) else len(shown)
    return Answer({"experiments": shown}, [f"{count} experiment(s), newest first; an A/B's two arms are one"])


def show_experiment(context: Context, args: ExperimentArgs) -> Answer:
    from tuningfork.ops.inspect import experiment_report

    report = experiment_report(context.store, args.experiment).to_json()
    if report.get("run_id"):
        refuse_hidden(context, context.store.get_run(report["run_id"]))
    view = compact.experiment(report)
    return Answer(view, compact.experiment_lines(view))


def show_tokens(context: Context, args: TokensArgs) -> Answer:
    from tuningfork.ops.tokens import step_tokens

    visible_run(context, args.run)
    found = step_tokens(context.store, args.run, args.step, stream=args.stream).to_json(hotspots=args.hotspots)
    view = compact.tokens(found, positions=args.positions, hotspots=args.hotspots)
    if not view.get("scored"):
        return Answer(view, [view["hint"]])
    spots = view.get("hotspots") or []
    lines = [f"{len(spots)} least certain token(s) of step {view.get('step')}'s answer, the runner-up's chance first"]
    for spot in spots[:5]:
        chosen = spot.get("chosen") or {}
        lines.append(f"  position {spot.get('position')} ({spot.get('part')}): {chosen.get('text')!r} at "
                     f"{chosen.get('p', 0):.0%}, runner-up {spot.get('runner_up', 0):.0%}")
    lines.append("start_forking_tokens forces alternatives there and tests which change what the model does")
    return Answer(view, lines)


def server_status(context: Context, args: NoArgs) -> Answer:
    store = context.store
    data: dict[str, Any] = {}
    lines = []
    try:
        health = context.jobs.status()
    except OperationError as error:
        health = None
        data["proxy"] = None
        lines.append(f"no proxy to start work through: {error.message}" + (f" — {error.hint}" if error.hint else ""))
    if health:
        server = health.get("server") or {}
        data["proxy"] = {key: health.get(key) for key in ("version", "upstream", "paused", "inflight", "max_inflight", "home")}
        data["server"] = {
            "id": compact.short(server.get("id")),
            "models": server.get("models"),
            "build": server.get("build"),
            "n_ctx": server.get("n_ctx"),
        }
        data["problem"] = health.get("server_problem")
        data["checks"] = compact.fit(store.server_checks(server.get("id")) or {}, 6000) or None
        lines.append(f"the proxy sends model calls to {health.get('upstream')}"
                     + (" — paused" if health.get("paused") else ""))
        if health.get("server_problem"):
            lines.append(f"the server could not be read: {health['server_problem']}")
        elif server.get("models"):
            lines.append(f"it serves {', '.join(map(str, server['models']))}")
    data["recorded_servers"] = [
        {"id": compact.short(server_id), "models": identity.get("models"), "build": identity.get("build"),
         "first_seen": compact.when(first_seen), "runs": runs}
        for server_id, identity, first_seen, runs in store.servers()[:5]
    ]
    return Answer(data, lines)


def show_scenarios(context: Context, args: ScenariosArgs) -> Answer:
    from tuningfork.ops.scenarios import list_scenarios, read_scenario

    def shown(found: dict[str, Any]) -> dict[str, Any]:
        return {**found, "spec": without_held_out_evals(found.get("spec"))} if "spec" in found else found

    if args.name:
        found = shown(read_scenario(context.store.home, args.name).to_json())
        return Answer(compact.fit(found, 16_000), [f"scenario {args.name!r}: a task spec start_batch can run"])
    listed = [shown(scenario.to_json()) for scenario in list_scenarios(context.store)]
    return Answer({"scenarios": compact.fit(listed, 16_000)}, [f"{len(listed)} saved scenario(s)"])


# --- the refinement loop -----------------------------------------------------------------------------

#: Proposals an agent may make to one refinement: a screen tests a few, not a flood.
AGENT_PROPOSALS = 12


def start_refinement(context: Context, args: RefinementArgs, who: dict[str, Any]) -> Answer:
    from tuningfork.ops import refine

    made = refine.create(
        context.store, args.run, args.step, target=args.target, target_match=args.target_match,
        target_rules=args.target_rules or None, created_by=who,
    )
    data = made.to_json()
    source = data.get("source") or {}
    lines = [
        f"refinement {made.id[:8]} at run {made.run_id[:8]} step {made.step_idx}, toward: "
        f"{data['target'].get('description')}",
        "next: show_refinement for the evidence, then propose_edit, then start_screen",
    ]
    if source and not source.get("matches"):
        lines.append("note: the skill's file has changed since this run was given it; proposals edit the run's copy")
    return Answer({"refinement": compact.short(made.id), "run": compact.short(made.run_id), "step": made.step_idx,
                   "target": data["target"].get("description"), "skill": made.skill}, lines)


def show_refinement(context: Context, args: RefinementRef) -> Answer:
    from tuningfork.ops import refine
    from tuningfork.ops.packet import build_packet

    shown = refine.view(context.store, args.refinement)
    found = shown["refinement"]
    data: dict[str, Any] = {
        "refinement": compact.short(found["id"]),
        "status": found["status"],
        "in_trash": True if found.get("trashed_at") else None,
        "target": found["target"].get("description"),
        "skill": found["skill"],
        "proposals": capped_proposals(shown["proposals"]),
        "screens": [
            {"experiment": compact.short(screen["experiment"]), "status": screen["status"],
             **(compact.screen(screen["report"]) or {})}
            for screen in shown["screens"][:5]
        ],
        # Without the held-out verdicts, or the gate they feed: those are the person's.
        "validations": [
            {"validation": compact.short(each["validation"]), "proposal": each["proposal"][:8],
             "status": each["status"], "forward": _arms(each["forward"]), "training": _arms(each["training"]),
             "held_out": each["held_out"]}
            for each in shown["validations"][:5]
        ] or None,
    }
    lines = [f"refinement {compact.short(found['id'])}: {len(shown['proposals'])} proposal(s), "
             f"{len(shown['screens'])} screen(s); target: {data['target']}"]
    if found.get("trashed_at"):
        lines.append("it is in the trash: it takes no new proposals, screens or validations until the person restores it")
    for screen in shown["screens"][:1]:
        lines += compact.screen_lines(compact.screen(screen["report"]))
    if shown["validations"]:
        lines.append(f"{len(shown['validations'])} validation(s); their held-out verdicts and the gate are the "
                     "person's to read, and approving is theirs")
    if args.evidence and found["status"] == "open" and not found.get("trashed_at"):
        packet = build_packet(context.store, found["id"])
        data["packet"] = {"sha": packet.sha, **packet.data}
        lines.append(f"packet {packet.sha}: {packet.data.get('answers_here', {}).get('summary', '')}")
        lines.append("propose_edit with exact {old, new} copied from the skill; cite this packet")
    return Answer(data, lines)


def _arms(part: dict[str, Any] | None) -> dict[str, Any] | None:
    """A validation part's pass rates, the edit's and the original's."""
    if not part:
        return None
    return {arm: compact.rate(part[arm].get("rate")) for arm in ("edit", "original")}


def capped_proposals(items: list[dict[str, Any]]) -> Any:
    return compact.capped([
        {
            "id": item["id"][:8],
            "changed": item["changed"],
            "diff": [line for hunk in item["diff"] for line in hunk["lines"] if line["op"] != " "][:12],
            "proposers": [{key: p.get(key) for key in ("via", "client", "model") if p.get(key)} for p in item["proposers"]],
            "rationale": compact.clip(item["proposers"][0].get("rationale"), 300),
            "screens": [
                {"survives": v.get("survives"), "worse": v.get("worse"), "complete": v.get("complete"),
                 "target": compact.rate(v.get("target")), "baseline": compact.rate(v.get("baseline")),
                 "adjusted_p": v.get("adjusted_p")}
                for v in item["screens"]
            ] or None,
        }
        for item in items
    ], 20)


def propose_edit(context: Context, args: ProposeEditArgs, who: dict[str, Any]) -> Answer:
    from tuningfork.ops import refine

    mine = [p for p in refine.proposals(context.store, args.refinement)
            if any(entry.get("via") == "mcp" for entry in p.data.get("proposers", []))]
    if len(mine) >= AGENT_PROPOSALS:
        raise OperationError(
            f"agents have made {len(mine)} proposals to this refinement already",
            hint="screen the ones there: start_screen with their ids",
            code="too_many_proposals", field="refinement",
        )
    made = refine.add_proposal(
        context.store, args.refinement, edits=args.edits, rationale=args.rationale,
        proposer=who, packet=args.packet,
    )
    data = made.to_json()
    lines = [
        f"proposal {made.id[:8]}: {data['changed']} line(s) changed"
        + (" — the same text was proposed before; you are added to its proposers" if made.duplicate else ""),
        f"a hypothesis until screened: start_screen(refinement, proposals=[\"{made.id[:8]}\", …])",
    ]
    return Answer({"proposal": made.id[:8], "changed": data["changed"], "duplicate": made.duplicate,
                   "variant": data["variant"]}, lines)


# --- jobs, as answers ------------------------------------------------------------------------------


def job_brief(job: dict[str, Any]) -> dict[str, Any]:
    """A job as a list shows it: what, how far, who asked."""
    progress = job.get("progress") or {}
    error = job.get("error") or None
    data = {
        "id": compact.short(job.get("id")),
        "kind": job.get("kind"),
        "status": job.get("status"),
        "progress": {key: progress.get(key) for key in ("done", "total", "waiting") if progress.get(key) is not None} or None,
        "queue_position": job.get("queue_position"),
        "started_by": job.get("started_by"),
        "created": compact.when(job.get("created_at")),
        "ended": compact.when(job.get("ended_at")),
        "error": {key: error.get(key) for key in ("message", "hint", "field", "code") if error.get(key)} if error else None,
    }
    return {key: value for key, value in data.items() if value is not None}


def job_answer(context: Context, job: dict[str, Any]) -> Answer:
    """A job with its result, read the way `show_experiment` reads it, when it has one."""
    data = job_brief(job)
    if job.get("deduplicated"):
        data["deduplicated"] = True
    links = {name: [compact.short(found) for found in ids] for name, ids in (job.get("links") or {}).items() if ids}
    if links:
        data["links"] = links
    progress = job.get("progress") or {}
    head = f"job {data['id']} ({job.get('kind')}): {job.get('status')}"
    if progress.get("total"):
        head += f", {progress.get('done') or 0} of {progress['total']}"
    lines = [head + (" — the same work was already under way, so it was not started again" if job.get("deduplicated") else "")]
    status = job.get("status")
    if status in ("queued", "running"):
        lines.append(f"still {status}: job_status(job=\"{data['id']}\", wait_seconds=…) follows it")
        return Answer(data, lines)
    if job.get("error"):
        error = job["error"]
        lines.append(f"failed: {error.get('message')}" + (f" — {error['hint']}" if error.get("hint") else ""))
    experiments = (job.get("links") or {}).get("experiments") or []
    hidden = hidden_tasks(context)
    if hidden:
        reference = (job.get("params") or {}).get("run")
        runs = [context.store.get_run(str(reference))] if reference else []
        runs += [context.store.get_run(found.run_id) for found in
                 (context.store.get_experiment(experiment) for experiment in experiments[:1])
                 if found is not None and found.run_id]
        if any(run is not None and run.task_id in hidden for run in runs):
            data.pop("links", None)
            lines.append("its results are of a held-out task: the person's evidence, never an agent's")
            return Answer(data, lines)
    result = job.get("result")
    if experiments and status in ("done", "cancelled", "interrupted"):
        from tuningfork.ops.inspect import experiment_report

        view = compact.experiment(experiment_report(context.store, experiments[0]).to_json())
        data["result"] = view
        lines += compact.experiment_lines(view)[1:]
    elif job.get("kind") == "tokens" and isinstance(result, dict):
        data["result"] = compact.tokens(result)
        lines.append("step_tokens shows where the model was least sure; start_forking_tokens tests those tokens")
    elif result is not None:
        data["result"] = compact.fit(result, 12_000)
    if status in ("cancelled", "interrupted"):
        lines.append("it stopped before the end: what it finished is kept and counted")
    return Answer(data, lines)


#: What a job's total counts, by kind, for a dry run's first sentence.
UNITS = {
    "resample": "samples", "ab": "samples", "screen": "samples", "forking-tokens": "samples",
    "resample-to-end": "forks, each run to the end", "ab-to-end": "forks, each run to the end",
    "fork": "forks", "batch": "runs", "server-check": "checks",
}


def plan_answer(kind: Kind, planned: dict[str, Any]) -> Answer:
    """A dry run: what the job would do, and what it would cost."""
    plan = planned.get("plan") or {}
    total = planned.get("total")
    made = f"{total} {UNITS.get(kind.name, 'units of work')}" if total is not None else "its work"
    lines = [f"{kind.name} would make {made}; nothing was started"]
    estimate = plan.get("estimate") if isinstance(plan, dict) else None
    if isinstance(estimate, dict):
        # A resample's or an A/B's bound is the recorded step's own time, on a cold cache;
        # the forking probe's is decode time over the tokens it would generate, with each
        # continuation's prompt processing when the step's timing gave it.
        if estimate.get("fresh"):
            tokens = estimate.get("prompt_tokens")
            lines.append("each sample processes its whole prompt" + (f", {tokens:,} tokens" if isinstance(tokens, int) else "")
                         + ", then answers: allow for that on top of the step's own time")
        elif isinstance(estimate.get("worst_minutes"), (int, float)):
            lines.append(f"at most about {max(1, round(estimate['worst_minutes']))} minute(s): the recorded "
                         "step's time for every sample" + ("; a warm cache is usually faster" if estimate.get("cached") else ""))
        elif isinstance(estimate.get("minutes"), (int, float)):
            lines.append(f"about {max(1, round(estimate['minutes']))} minute(s) at the step's recorded speed"
                         + (", its whole prompt processed for each continuation" if estimate.get("prefill_seconds")
                            else "; each continuation also processes the whole prompt first"
                            if estimate.get("prompt_tokens") else ""))
    for warning in (plan.get("warnings") or []) if isinstance(plan, dict) else []:
        lines.append(f"warning: {warning}")
    return Answer({"kind": kind.name, "total": total, "plan": compact.fit(plan, 12_000)}, lines)


# --- the registry -----------------------------------------------------------------------------


READS = (
    Tool("list_tasks", "Tasks and pass rates",
         "Start here. Every task recorded, with its check's pass rate (and 95% interval) under each "
         "condition it ran under: harness, model, skill version, server.", NoArgs, list_tasks),
    Tool("list_runs", "Runs",
         "Recorded runs, newest first: of a task, or where the model called a tool (at a step).",
         RunsArgs, list_runs),
    Tool("show_run", "A run step by step",
         "One run step by step: what the model did at each, its verdict, its threads.", RunArgs, show_run),
    Tool("find_divergence", "Where runs split",
         "The first step where a task's runs stopped agreeing: the behaviours there, how many runs "
         "took each, and how often those runs went on to pass. Where to look first.",
         DivergenceArgs, find_divergence),
    Tool("show_step", "One step's context",
         "One step as the model saw it: where the skill's text sits (file and lines), the last "
         "messages before it answered, and its answer.", StepArgs, show_step),
    Tool("step_behaviours", "Behaviours at a step",
         "What the model did at a step, across runs, forks and samples — each marked as a target "
         "would count it — and the letters of the step's latest resample, for declaring an A/B's target.",
         BehavioursArgs, step_behaviours),
    Tool("show_skill", "The skill a run was given",
         "The SKILL.md a run was given, its lines numbered, and the variants saved of it: the text "
         "an edit for start_ab starts from.", RunArgs, show_skill),
    Tool("list_experiments", "Experiments",
         "Resamples, A/Bs and probes, newest first: of a run, or all.", ExperimentsArgs, list_experiments),
    Tool("show_experiment", "An experiment's result",
         "An experiment's result again: behaviours with their rates and intervals; an A/B's declared "
         "target, Fisher's p and its verdict, or that it was exploratory; a probe's flipping tokens.",
         ExperimentArgs, show_experiment),
    Tool("step_tokens", "Where the model wavered",
         "A step's answer as tokens, once scored (start_tokens): the tokens the model was least sure "
         "of, and its alternatives there.", TokensArgs, show_tokens),
    Tool("server_status", "The inference server",
         "The proxy and the server answering it: what it serves, whether it can be read, what it "
         "can do (token analysis needs llama.cpp), and the servers recorded before.", NoArgs, server_status),
    Tool("list_scenarios", "Scenarios",
         "Task specs saved from the web UI, or one in full: what start_batch can run.",
         ScenariosArgs, show_scenarios),
)

REFINE_READS = (
    "list_tasks", "list_runs", "show_run", "find_divergence", "show_step", "step_behaviours",
    "show_skill", "list_experiments", "show_experiment",
)
LOOP_READ = Tool(
    "show_refinement", "A refinement and its evidence",
    "A refinement's proposals with their line diffs and screen verdicts, and the evidence packet a "
    "proposal should cite: the step's context, answers that did and did not reach the target, how "
    "runs fared, the task's criteria, edits tried before.", RefinementRef, show_refinement,
)
LOOP_ACTS = (
    Tool("start_refinement", "Start refining a skill",
         "Start refining the skill a run was given, at one of its steps, toward a target declared now: "
         "what the model should do there. Returns the open one if it exists.",
         RefinementArgs, act=start_refinement, acts=True),
    Tool("propose_edit", "Propose an edit",
         "Propose a small edit to the refinement's skill: 1 to 3 exact {old, new} replacements and a "
         "rationale citing the packet. A hypothesis until start_screen tests it.",
         ProposeEditArgs, act=propose_edit, acts=True),
)

JOB_TOOLS = (
    Tool("job_status", "A job's progress and result",
         "A job's progress, and its result once it ends — read like show_experiment. Wait on a "
         "running job here; never start the same work again.", JobArgs, acts=False),
    Tool("list_jobs", "Jobs", "Recent jobs, newest first, with who started each.", JobsArgs),
    Tool("cancel_job", "Cancel a job",
         "Stop a job an agent started. What it finished is kept.", CancelArgs, acts=True),
)

#: What each kind is for in the loop, for its tool's description.
KIND_GUIDE = {
    "resample": "How often each behaviour happens at a step, its history held fixed: N samples, "
                "each behaviour lettered with its rate and 95% interval. Run it at the divergence step.",
    "ab": "Does an edited wording change what the model does at a step? Give the whole edited skill as "
          "variant_text with a variant_name. Declare the target before sampling — `target` (a letter "
          "from that step's resample), `target_match` or `target_rules` — or the result is exploratory.",
    "tokens": "Score a step's answer token by token, so step_tokens can show where the model wavered.",
    "forking-tokens": "At the step's least certain tokens, force each likely alternative and let the "
                      "model go on: which tokens change what it does (Holm-corrected).",
    "server-check": "Measure what the server can do: harness needs, token analysis.",
    "batch": "Run a task N times through its harness, each run in a fresh workspace: pass rates with "
             "intervals. Runs a harness, and the task's check, on this machine.",
    "fork": "Replay a run up to a step and go live from there. Runs a harness on this machine.",
    "resample-to-end": "Fork a run from a step N times, each to the end: how often the task's check "
                       "passes from there. Runs a harness and the check on this machine.",
    "ab-to-end": "The A/B run to the end: each wording's forks judged by the task's check, compared "
                 "with Fisher's test. Runs a harness and the check on this machine.",
    "check-trial": "Try a check on a recorded run's end state, or on a reply: nothing is recorded.",
    "screen": "Screen proposals at the refinement's checkpoint: a fresh sample of the original and of "
              "each proposal (n each), each tested against the original on the target, Holm-corrected "
              "over the proposals named. A survivor is evidence to confirm with full runs, not proof.",
}


def job_tool(kind: Kind, *, max_wait: int) -> Tool:
    from tuningfork.mcp.schema import HARNESS, tool_name

    guide = KIND_GUIDE.get(kind.name, kind.summary)
    return Tool(
        tool_name(kind.name),
        kind.summary.split(":")[0].split(",")[0][:60],
        f"{guide} A job: returns its result if it ends within wait_seconds, else its id for "
        "job_status. dry_run plans it without starting.",
        kind=kind, machine=kind.name in HARNESS, acts=True,
    )


def registry(toolset: str = "evidence", *, max_wait: int = 120) -> dict[str, Tool]:
    """Every tool an agent is offered, in a fixed order: reads, the loop's own tools (with
    the refine and all toolsets), the jobs it may start, and the tools that follow jobs up."""
    from tuningfork.mcp.schema import offered

    kinds = offered(toolset)
    reads = [tool for tool in READS if toolset != "refine" or tool.name in REFINE_READS]
    loop = [LOOP_READ, *LOOP_ACTS] if toolset in ("refine", "all") else []
    tools = [*reads, *loop, *(job_tool(kind, max_wait=max_wait) for kind in kinds), *JOB_TOOLS]
    return {tool.name: tool for tool in tools}


def input_schema(tool: Tool, *, max_wait: int) -> dict[str, Any]:
    from tuningfork.mcp.schema import job_schema

    if tool.kind is not None:
        return job_schema(tool.kind, max_wait=max_wait)
    schema = args_schema(tool.args or NoArgs)
    if "wait_seconds" in schema["properties"]:
        schema["properties"]["wait_seconds"]["maximum"] = max_wait
    return schema


def check_job_id(value: str) -> str:
    if not JOB_ID.fullmatch(value):
        raise OperationError(f"no job matching {value!r}: job ids are hexadecimal", code="bad_params", field="job")
    return value


def same_params(kind: Kind, params: dict[str, Any]) -> dict[str, Any]:
    """Parameters as the job runner stores them — parsed, defaults filled, redacted — so an
    identical job already under way can be found."""
    from tuningfork.config import redact_json

    return redact_json(dataclasses.asdict(kind.parse(params)))
