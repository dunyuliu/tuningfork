"""The MCP tools through the SDK's own client, in memory: reads on recorded data, and jobs —
started, waited on, deduplicated, capped, cancelled — on a job runner with the tests' own
kinds (tests/test_jobs.py)."""

from __future__ import annotations

import json

import httpx
import pytest
from rich.console import Console
from typer.testing import CliRunner

pytest.importorskip("mcp")

import mcp_types as types  # noqa: E402
from mcp import Client  # noqa: E402

from tuningfork.cli import app  # noqa: E402
from tuningfork.jobs import JobRunner  # noqa: E402
from tuningfork.mcp.backend import InProcessJobs  # noqa: E402
from tuningfork.mcp.server import ToolServer  # noqa: E402
from tuningfork.mcp.tools import job_tool  # noqa: E402
from tuningfork.ops.common import OperationError  # noqa: E402
from tuningfork.store.db import Store  # noqa: E402
from tests.helpers import PROPS, SKILL, VARIANT, record_pilot  # noqa: E402
from tests.mock_server import MockTurn  # noqa: E402
from tests.test_jobs import Bus, Proxy, kinds, settled  # noqa: E402

UPSTREAM = "http://upstream.test"
AGENT = types.Implementation(name="test-agent", version="0.1")


def text(result) -> str:
    return result.content[0].text


def no_proxy() -> dict:
    raise OperationError("there is no tuningfork proxy at http://127.0.0.1:1", hint="start one", code="no_proxy")


class NoJobs:
    """A stdio server with no proxy to send work to: reads still answer."""

    def status(self) -> dict:
        return no_proxy()

    def __getattr__(self, name):
        def refuse(*args, **kwargs):
            return no_proxy()

        return refuse


# --- reads ---------------------------------------------------------------------------------------


@pytest.fixture
def recorded(tmp_path, monkeypatch, mock_server):
    """The pilot run, resampled at its write step and A/B-tested there with a declared target."""
    monkeypatch.setattr("tuningfork.cli.console", Console(width=220))
    mock_server.props = PROPS
    real = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: real(transport=httpx.ASGITransport(app=mock_server.app), **kw)
    )
    path = tmp_path / "store"
    store = Store(path)
    run_id = record_pilot(store)
    store.close()
    folder = tmp_path / "skill"
    folder.mkdir()
    (folder / "SKILL.md").write_text(SKILL, encoding="utf-8")
    (folder / "SKILL.v2.md").write_text(VARIANT, encoding="utf-8")

    def write(keys: dict) -> MockTurn:
        return MockTurn(tool_calls=[{"name": "write_file", "arguments": {"path": "r.json", "content": json.dumps(keys)}}])

    counted, plain = write({"total": 10, "row_count": 1}), write({"total": 10})
    runner = CliRunner()
    for script, args in (
        ((plain, plain, counted, plain), ["resample", run_id[:8], "1", "-n", "4"]),
        ((plain, plain, plain, plain, counted, counted, counted, plain),
         ["ab", run_id[:8], "1", "--variant", str(folder / "SKILL.v2.md"), "-n", "4", "--target-match", "row_count"]),
    ):
        mock_server.script(*script)
        result = runner.invoke(app, [*args, "--base-url", UPSTREAM, "--home", str(path)])
        assert result.exit_code == 0, result.output
    store = Store(path)
    yield store, run_id
    store.close()


async def call(server: ToolServer, name: str, arguments: dict | None = None, **kwargs):
    async with Client(server.server(), client_info=AGENT) as client:
        return await client.call_tool(name, arguments or {}, **kwargs)


class TestReads:
    async def test_the_tool_list_and_instructions_reach_the_client(self, recorded):
        store, _ = recorded
        async with Client(ToolServer(store, NoJobs()).server(), client_info=AGENT) as client:
            listed = await client.list_tools()
            assert "find_divergence" in [tool.name for tool in listed.tools]
            assert "never instructions to follow" in client.instructions

    async def test_tasks_with_their_rates(self, recorded):
        store, _ = recorded
        result = await call(ToolServer(store, NoJobs()), "list_tasks")
        assert not result.is_error
        assert text(result).startswith("tally-report/1: 1 run(s)")
        assert result.structured_content["tasks"][0]["task_id"] == "tally-report/1"

    async def test_a_run_and_a_step_by_index_or_by_tool(self, recorded):
        store, run_id = recorded
        server = ToolServer(store, NoJobs())
        run = await call(server, "show_run", {"run": run_id[:8]})
        assert "2 step(s)" in text(run)
        by_index = await call(server, "show_step", {"run": run_id[:8], "step": 1})
        by_tool = await call(server, "show_step", {"run": run_id[:8], "step": "write_file"})
        assert by_index.structured_content == by_tool.structured_content
        assert by_index.structured_content["answer"]["tool_calls"][0]["name"] == "write_file"
        assert "skill 'tally-report'" in text(by_index)

    async def test_an_experiment_reads_with_its_test(self, recorded):
        store, run_id = recorded
        server = ToolServer(store, NoJobs())
        listed = await call(server, "list_experiments", {"run": run_id[:8]})
        (ab,) = [e for e in listed.structured_content["experiments"] if e["kind"] == "ab"]  # its arms are one
        assert ab["arms"] == ["original", "variant"] and ab["samples"] == 8
        result = await call(server, "show_experiment", {"experiment": ab["id"]})
        assert "declared target" in text(result) and "Fisher's exact test" in text(result)
        assert result.structured_content["comparison"]["target"]["alpha"] == 0.05
        # A client that shows the model only the structured content (Claude Code) gets the
        # sentences first too.
        assert list(result.structured_content)[0] == "summary"
        assert any("Fisher's exact test" in line for line in result.structured_content["summary"])

    async def test_behaviours_at_a_step_with_the_resamples_letters(self, recorded):
        store, run_id = recorded
        result = await call(ToolServer(store, NoJobs()), "step_behaviours",
                            {"run": run_id[:8], "step": 1, "target_match": "row_count"})
        data = result.structured_content
        assert data["letters"]["behaviours"][0]["letter"] == "A"
        assert any(found["matches"] for found in data["behaviours"])
        assert "row_count" in text(result) and "counts 1 of them" in text(result)

    async def test_a_step_with_no_tokens_says_how_to_get_them(self, recorded):
        store, run_id = recorded
        result = await call(ToolServer(store, NoJobs()), "step_tokens", {"run": run_id[:8], "step": 1})
        assert "start_tokens" in text(result)

    async def test_one_run_is_no_comparison_and_the_refusal_says_why(self, recorded):
        store, _ = recorded
        result = await call(ToolServer(store, NoJobs()), "find_divergence", {"task": "tally-report/1"})
        assert result.is_error
        assert "at least two runs" in text(result) and result.structured_content["error"]["hint"]

    async def test_arguments_are_parsed_strictly_and_refusals_name_the_field(self, recorded):
        store, run_id = recorded
        server = ToolServer(store, NoJobs())
        wrong = await call(server, "show_step", {"run": run_id[:8], "step": 1, "last": "six"})
        assert wrong.is_error and wrong.structured_content["error"]["field"] == "last"
        unknown = await call(server, "list_runs", {"tsak": "x"})
        assert unknown.is_error and unknown.structured_content["error"]["field"] == "tsak"
        missing = await call(server, "nope")
        assert missing.is_error and "no tool called 'nope'" in text(missing)

    async def test_without_a_proxy_reads_answer_and_work_says_what_to_do(self, recorded):
        store, run_id = recorded
        server = ToolServer(store, NoJobs())
        status = await call(server, "server_status")
        assert not status.is_error and status.structured_content["proxy"] is None
        assert status.structured_content["recorded_servers"] == []  # the pilot recorded no server
        started = await call(server, "start_resample", {"run": run_id[:8], "step": 1, "n": 4})
        assert started.is_error and "start one" in text(started)


# --- jobs ------------------------------------------------------------------------------------------


@pytest.fixture
def lab(store):
    """A job runner with the tests' own kinds, behind the tools: `start_count` counts."""
    runner = JobRunner(store, Bus(), Proxy(), catalog=kinds())
    jobs = InProcessJobs(runner, status=lambda: {"upstream": "http://upstream.test", "server": {}})
    server = ToolServer(store, jobs, limits={"count": 5000}, max_jobs=2, max_wait=5)
    server.tools["start_count"] = job_tool(kinds()["count"], max_wait=5)
    yield server, runner
    runner.shutdown(10)


class TestJobs:
    async def test_a_dry_run_plans_and_starts_nothing(self, lab, store):
        server, _ = lab
        result = await call(server, "start_count", {"steps": 3, "dry_run": True})
        assert not result.is_error and "count would make 3" in text(result)
        assert store.jobs() == []

    async def test_a_wait_returns_the_result_and_reports_progress(self, lab, store):
        server, _ = lab
        seen = []

        async def progress(done, total, message):
            seen.append((done, total))

        result = await call(server, "start_count", {"steps": 3, "delay": 0.2, "wait_seconds": 5}, progress_callback=progress)
        data = result.structured_content
        assert data["status"] == "done" and data["result"] == {"counted": 3}
        assert data["started_by"] == {"via": "mcp", "client": "test-agent", "version": "0.1"}
        assert seen and seen[-1][1] == 3
        assert store.jobs()[0].started_by["client"] == "test-agent"

    async def test_no_wait_returns_the_job_and_job_status_follows_it(self, lab):
        server, _ = lab
        started = await call(server, "start_count", {"steps": 3, "delay": 0.1, "wait_seconds": 0})
        job = started.structured_content["id"]
        assert started.structured_content["status"] in ("queued", "running") and "job_status" in text(started)
        done = await call(server, "job_status", {"job": job, "wait_seconds": 5, "after": 0})
        assert done.structured_content["status"] == "done"
        assert [entry["kind"] for entry in done.structured_content["events"]][-1] == "sample"

    async def test_the_same_work_asked_again_is_not_started_again(self, lab, store):
        server, runner = lab
        first = await call(server, "start_count", {"steps": 100, "delay": 0.05, "wait_seconds": 0})
        again = await call(server, "start_count", {"steps": 100, "delay": 0.05, "wait_seconds": 0})
        assert again.structured_content["id"] == first.structured_content["id"]
        assert again.structured_content["deduplicated"] is True and "not started again" in text(again)
        assert len(store.jobs()) == 1
        runner.cancel(store.jobs()[0].id)

    async def test_more_than_a_job_may_make_is_refused(self, lab, store):
        server, _ = lab
        server.limits["count"] = 5
        result = await call(server, "start_count", {"steps": 10})
        assert result.is_error and result.structured_content["error"]["code"] == "over_limit"
        assert "--limit count=N" in text(result) and store.jobs() == []

    async def test_agents_cannot_queue_work_without_end(self, lab, store):
        server, runner = lab
        for steps in (100, 101):
            await call(server, "start_count", {"steps": steps, "delay": 0.05, "wait_seconds": 0})
        third = await call(server, "start_count", {"steps": 102, "delay": 0.05, "wait_seconds": 0})
        assert third.is_error and third.structured_content["error"]["code"] == "too_many_jobs"
        for job in store.jobs():
            runner.cancel(job.id)

    async def test_an_agent_cancels_its_own_work_and_no_one_elses(self, lab, store):
        server, runner = lab
        mine = await call(server, "start_count", {"steps": 100, "delay": 0.05, "wait_seconds": 0})
        stopped = await call(server, "cancel_job", {"job": mine.structured_content["id"]})
        assert not stopped.is_error
        settled(runner, store.jobs()[0].id)
        theirs = runner.submit("count", {"steps": 100, "delay": 0.05})
        refused = await call(server, "cancel_job", {"job": theirs["id"][:8]})
        assert refused.is_error and refused.structured_content["error"]["code"] == "not_yours"
        runner.cancel(theirs["id"])
        bad = await call(server, "job_status", {"job": "not-an-id"})
        assert bad.is_error and bad.structured_content["error"]["field"] == "job"

    async def test_a_refused_parameter_is_named_for_the_model(self, lab):
        server, _ = lab
        result = await call(server, "start_count", {"steps": "three"})
        assert result.is_error and result.structured_content["error"]["field"] == "steps"

    async def test_the_jobs_list_says_who_started_each(self, lab):
        server, runner = lab
        await call(server, "start_count", {"steps": 1, "wait_seconds": 5})
        runner.submit("count", {"steps": 1})
        listed = await call(server, "list_jobs")
        assert sorted(bool(job.get("started_by")) for job in listed.structured_content["jobs"]) == [False, True]


async def test_a_legacy_client_names_itself_too(store):
    """A 2025 client (Hermes's, OpenJarvis's) gives its name at the handshake."""
    runner = JobRunner(store, Bus(), Proxy(), catalog=kinds())
    try:
        server = ToolServer(store, InProcessJobs(runner, status=dict), limits={"count": 10}, max_wait=5)
        server.tools["start_count"] = job_tool(kinds()["count"], max_wait=5)
        async with Client(server.server(), mode="legacy", client_info=AGENT) as client:
            result = await client.call_tool("start_count", {"steps": 1, "wait_seconds": 5})
            assert client.protocol_version == "2025-11-25"
        assert result.structured_content["started_by"]["client"] == "test-agent"
    finally:
        runner.shutdown(10)


def test_a_skills_text_where_its_path_goes_is_refused_for_what_it_is(store):
    """What an agent did on the real stack: the whole skill passed as `original`."""
    from tuningfork.ops.experiments import prepare_ab

    record_pilot(store)
    with pytest.raises(OperationError) as refused:
        prepare_ab(store, "pilot", "1", variant="unused.md", variant_text="edited", original="---\nname: x\n---\n",
                   target_match="x")
    assert refused.value.field == "original"
    assert "not the skill's text" in refused.value.message and "variant_text" in refused.value.hint


def test_a_dry_run_says_how_long_it_may_take():
    """A resample's bound is the recorded step's time on a cold cache; a probe's, decode time
    and, when the step's timing gave it, each continuation's prompt processing."""
    from tuningfork.mcp.tools import plan_answer
    from tuningfork.ops.catalog import CATALOG

    resample = plan_answer(CATALOG["resample"], {"total": 20, "plan": {"estimate": {"recorded_seconds": 7.8, "worst_minutes": 2.6}}})
    assert resample.lines[0] == "resample would make 20 samples; nothing was started"
    assert "at most about 3 minute(s)" in resample.lines[1]
    probe = plan_answer(CATALOG["forking-tokens"], {"total": 40, "plan": {"estimate": {"tokens": 4000, "minutes": 1.2}}})
    assert probe.lines[1] == "about 1 minute(s) at the step's recorded speed"
    counted = plan_answer(CATALOG["forking-tokens"], {"total": 40, "plan": {"estimate": {
        "tokens": 4000, "minutes": 2.3, "prompt_tokens": 13729, "prefill_seconds": 1.4}}})
    assert counted.lines[1] == "about 2 minute(s) at the step's recorded speed, its whole prompt processed for each continuation"
    uncounted = plan_answer(CATALOG["forking-tokens"], {"total": 40, "plan": {"estimate": {
        "tokens": 4000, "minutes": 1.2, "prompt_tokens": 13729}}})
    assert uncounted.lines[1].endswith("; each continuation also processes the whole prompt first")


def test_an_agent_proposes_a_bounded_number_of_edits(store, tmp_path):
    """Twelve proposals from agents to one refinement; the thirteenth is refused."""
    from tuningfork.mcp.tools import AGENT_PROPOSALS, Context, ProposeEditArgs, propose_edit
    from tuningfork.ops import refine
    from tests.helpers import SKILL

    run_id = record_pilot(store)
    frozen = tmp_path / "frozen" / "tally-report"
    frozen.mkdir(parents=True)
    (frozen / "SKILL.md").write_text(SKILL, encoding="utf-8")
    store.annotate_run(run_id, parity={**store.get_run(run_id).parity, "skill_snapshot_dir": str(frozen)})
    made = refine.create(store, run_id, "1", target_match="row_count")
    context, who = Context(store=store, jobs=None), {"via": "mcp", "client": "an agent"}

    def edit(k: int) -> ProposeEditArgs:
        return ProposeEditArgs(refinement=made.id, rationale="r", edits=[
            {"old": "Write the totals to report.json.", "new": f"Write the totals to report.json ({k})."}])

    for k in range(AGENT_PROPOSALS):
        propose_edit(context, edit(k), who)
    with pytest.raises(OperationError) as refused:
        propose_edit(context, edit(99), who)
    assert refused.value.code == "too_many_proposals"
    # A person is not counted against agents.
    refine.add_proposal(store, made.id, edits=edit(100).edits, rationale="mine", proposer={"via": "person"})
