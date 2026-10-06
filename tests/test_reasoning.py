"""A step the model answered without reasoning (`analysis.reasoning`): marked where the same
run reasons elsewhere — and never in a run with no reasoning at all — with whether a tool call
had just failed; counted per behaviour in an experiment; and a breakpoint condition."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from tuningfork.analysis.distribution import compare_arms, distribution
from tuningfork.analysis.normalize import Outcome
from tuningfork.analysis.reasoning import after_failed_call, chars_in, reasons, status
from tuningfork.breakpoints.conditions import Breakpoint, BreakpointError, ExchangeContext, response_matches
from tuningfork.ops import compact
from tuningfork.ops.inspect import run_detail
from tuningfork.ops.timeline import run_timeline, step_detail
from tests.helpers import as_response, tool_call_message


def call(name: str, call_id: str, reasoning: str | None = None) -> dict:
    message = tool_call_message(name, {"path": "data.csv"}, call_id)
    if reasoning:
        message["reasoning_content"] = reasoning
    return message


def a_run(store, *, failed: bool = True, reasoning: bool = True, reply: str | None = None) -> str:
    """Three steps: a read (reasoned), a retry right after the read failed (no reasoning), and
    a write (reasoned) — as the user saw models do — then, given one, a reply in text."""
    store.begin_run("r", "t/1")
    messages = [{"role": "user", "content": "summarize data.csv"}]
    answers = [call("read_file", "c1", "I should read the file." if reasoning else None),
               call("search_files", "c2"),
               call("write_file", "c3", "Now the report." if reasoning else None)]
    results = [json.dumps({"error": "no such file: data.csv"}) if failed else "id,amount\n1,10",
               "data/data.csv", "written"]
    for answer, result in zip(answers, results):
        store.record_step(request={"model": "m", "messages": list(messages)}, response=as_response(answer))
        messages += [answer, {"role": "tool", "tool_call_id": answer["tool_calls"][0]["id"], "content": result}]
    if reply:
        store.record_step(request={"model": "m", "messages": list(messages)},
                          response=as_response({"role": "assistant", "content": reply}))
    return store.get_run("r").id


class TestTheRule:
    def test_none_only_where_the_model_reasons_elsewhere(self):
        assert status(0, reasons=True) == "none" and status(12, reasons=True) == "reasoned"
        assert status(0, reasons=False) is None  # nothing to compare with
        assert status(0, reasons=True, acted=False) is None  # a reply in text, an error, a replayed call
        assert reasons([0, None, 3]) and not reasons([0, None])

    def test_a_failed_tool_result_is_read_as_the_harness_reports_it(self):
        failed = [{"role": "assistant", "content": ""}, {"role": "tool", "content": '{"status": "error", "output": ""}'}]
        assert after_failed_call(failed)
        assert after_failed_call([{"role": "tool", "content": "fine", "is_error": True}])  # Anthropic's own mark
        assert after_failed_call([{"role": "tool", "content": "Error: no such file"}])
        # Hermes's terminal tool says it by the exit code, its error left empty: the case the
        # user saw, a command not found and another way tried at once, without reasoning.
        assert after_failed_call([{"role": "tool", "content": json.dumps(
            {"output": "/usr/bin/bash: line 3: pdftotext: command not found", "exit_code": 127, "error": None})}])
        assert not after_failed_call([{"role": "tool", "content": json.dumps({"output": "ok", "exit_code": 0, "error": None})}])
        assert not after_failed_call([{"role": "tool", "content": "id,amount"}])
        # Only what the model is about to read: a failure before its last turn is history.
        assert not after_failed_call([{"role": "tool", "content": "Error"}, {"role": "assistant", "content": ""}])
        assert chars_in(as_response({"role": "assistant", "content": "x", "reasoning_content": "abc"})) == 3


class TestARun:
    def test_the_step_without_reasoning_is_marked_with_the_failure_before_it(self, store):
        run_id = a_run(store)
        steps = run_timeline(store, run_id).to_json()["steps"]
        assert [step["reasoning"] for step in steps] == ["reasoned", "none", "reasoned"]
        assert [step["after_failure"] for step in steps] == [False, True, False]
        detail = step_detail(store, run_id, 1).to_json()
        assert detail["reasoning"] == "none" and detail["after_failure"]
        assert [step["reasoning"] for step in run_detail(store, run_id)["steps"]] == ["reasoned", "none", "reasoned"]

    def test_a_run_that_never_reasons_marks_nothing(self, store):
        run_id = a_run(store, reasoning=False)
        assert {step["reasoning"] for step in run_timeline(store, run_id).to_json()["steps"]} == {None}

    def test_a_reply_in_text_without_reasoning_is_not_marked(self, store):
        # As Hermes on Gemma 4 ended three runs of four: it reasoned before every tool call,
        # then wrote its summary straight away.
        run_id = a_run(store, failed=False, reply="The report is written.")
        steps = run_timeline(store, run_id).to_json()["steps"]
        assert [step["reasoning"] for step in steps] == ["reasoned", "none", "reasoned", None]

    def test_an_agent_reads_it_too(self, store):
        run_id = a_run(store)
        answer = compact.step(step_detail(store, run_id, 1).to_json())["answer"]
        assert answer["no_reasoning"] and answer["after_failed_tool_call"]
        steps = compact.run_detail(run_detail(store, run_id))["steps"]
        assert [step.get("no_reasoning") for step in steps] == [None, True, None]

    def test_the_terminal_says_it(self, store, monkeypatch):
        from tuningfork.cli import app as cli

        run_id = a_run(store)
        shown = CliRunner().invoke(cli, ["show", run_id[:8], "--home", str(store.home)])
        assert shown.exit_code == 0 and "none" in shown.output, shown.output
        step = CliRunner().invoke(cli, ["step", run_id[:8], "1", "--home", str(store.home)])
        assert "no reasoning — right after a tool call failed" in step.output, step.output
        timeline = CliRunner().invoke(cli, ["timeline", run_id[:8], "--home", str(store.home)])
        assert "no reasoning — right after a tool call failed" in " ".join(timeline.output.split()), timeline.output


class TestExperiments:
    def test_each_behaviour_says_how_many_of_its_samples_reasoned(self):
        read, search = Outcome("tool_call", "read", "read_file"), Outcome("tool_call", "search", "search_files")
        found = distribution([read, read, search, read], reasoning=[30, 0, 0, 12])
        clusters = {cluster.outcome.key: cluster for cluster in found.clusters}
        assert (clusters["read"].reasoned, clusters["search"].reasoned) == (2, 0)
        assert found.to_json()["clusters"][0]["reasoned"] == 2
        none = distribution([read, search], reasoning=[0, 0])
        assert all(cluster.reasoned is None for cluster in none.clusters)  # nothing reasoned: nothing to say
        assert "reasoned" not in none.to_json()["clusters"][0]
        # An agent is told only where some samples went without: every sample reasoning says nothing.
        assert [b.get("reasoned") for b in compact.distribution(found.to_json())["behaviours"]] == [2, 0]
        every = distribution([read, read], reasoning=[5, 7])
        assert "reasoned" not in compact.distribution(every.to_json())["behaviours"][0]

    def test_an_ab_says_it_per_arm(self):
        read, search = Outcome("tool_call", "read", "read_file"), Outcome("tool_call", "search", "search_files")
        compared = compare_arms([read, search], [read, read], reasoning=([10, 0], [0, 7]))
        shift = next(shift for shift in compared.shifts if shift.outcome.key == "read")
        assert (shift.original_reasoned, shift.variant_reasoned) == (1, 1)
        assert compared.to_json()["shifts"][0]["original_reasoned"] is not None


class TestABreakpoint:
    def context(self) -> ExchangeContext:
        return ExchangeContext(request={"messages": []}, run_id=None, task_id=None, step=0)

    def test_it_stops_where_the_model_called_a_tool_without_reasoning(self):
        bp = Breakpoint(phase="response", no_reasoning=True)
        assert "calling a tool with no reasoning first" in bp.describe()
        assert response_matches(bp, self.context(), as_response(call("read_file", "c1")))
        assert not response_matches(bp, self.context(), as_response(call("read_file", "c1", "Let me think.")))
        # A reply in text without reasoning is the usual end of a run: never held for it.
        assert not response_matches(bp, self.context(), as_response({"role": "assistant", "content": "Done."}))

    def test_it_is_a_condition_on_the_answer(self):
        with pytest.raises(BreakpointError, match="response phase"):
            Breakpoint(phase="request", no_reasoning=True)
        with pytest.raises(BreakpointError, match="true or false"):
            Breakpoint.from_dict({"phase": "response", "no_reasoning": "yes"})
        assert Breakpoint(phase="request", no_reasoning=False).describe().startswith("before the model")
