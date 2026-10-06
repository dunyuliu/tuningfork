"""A step's tokens, in every front end: the operation, the API route, and the command."""

from __future__ import annotations

import json
import math

import httpx
import pytest
from rich.console import Console
from typer.testing import CliRunner

from tuningfork.analysis.tokens import from_entries
from tuningfork.backends.formats import get_format
from tuningfork.cli import app
from tuningfork.experiment import SamplingPlan, load_checkpoint, run_resample
from tuningfork.ops.common import OperationError
from tuningfork.ops.tokens import sample_tokens, step_tokens
from tests.helpers import PROPS, record_pilot
from tests.mock_server import MockTurn

API = "/_tuningfork/api"
UPSTREAM = "http://upstream.test"
GEMMA_PROPS = {**PROPS, "chat_template": "{{ '<|tool_call>call:' }}"}


def entry(token_id: int, token: str, p: float = 1.0, alternatives=()) -> dict:
    top = sorted([(token_id, token, p), *alternatives], key=lambda item: -item[2])
    return {
        "id": token_id, "token": token, "bytes": list(token.encode()), "logprob": math.log(p),
        "top_logprobs": [
            {"id": i, "token": t, "bytes": list(t.encode()), "logprob": math.log(q)} for i, t, q in top
        ],
    }


#: The write step's answer: a short thought, then the call, where the key name wavers.
ANSWER = [
    entry(100, "<|channel>"), entry(1, "thought"), entry(2, "\n"),
    entry(3, "Write", 0.7, [(4, "Save", 0.3)]),
    entry(101, "<channel|>"), entry(48, "<|tool_call>"), entry(5, "call"), entry(6, ":"),
    entry(7, "write_file"), entry(8, "{"), entry(9, "count", 0.6, [(10, "row", 0.4)]),
    entry(11, ":4}"), entry(49, "<tool_call|>"), entry(50, ""),
]


def stream_json(entries=ANSWER) -> dict:
    return from_entries(entries, top_k=2).mark_parts(get_format("gemma4")).to_json()


class TestStepTokens:
    def test_a_step_without_token_data(self, store):
        run_id = record_pilot(store)
        found = step_tokens(store, run_id, "1")
        assert found.view is None and found.streams == []
        assert found.to_json()["view"] is None

    def test_the_answers_own_stream(self, store):
        run_id = record_pilot(store)
        step = store.step(run_id, 1)
        store.add_token_stream(stream_json(), source="scored", step_id=step.id, server_id=step.server_id,
                               method="grammar", exact=True)

        found = step_tokens(store, run_id, "write_file").to_json()

        view = found["view"]
        assert view["stream"]["source"] == "scored" and view["stream"]["exact"] is True
        assert view["text"].endswith("call:write_file{count:4}<tool_call|>")
        key = next(token for token in view["tokens"] if token["text"] == "count")
        assert key["part"] == "tool_call"
        assert [text for text, _ in key["alternatives"]] == ["count", "row"]
        assert [p for _, p in key["alternatives"]] == pytest.approx([0.6, 0.4], abs=1e-4)
        assert key["heat"] == 3 and key["entropy"] == pytest.approx(0.971, abs=1e-3)
        # ranked by how likely the step's sampler was to draw the runner-up
        assert [(h["position"], h["part"]) for h in view["hotspots"]] == [(10, "tool_call"), (3, "reasoning")]
        assert [kind for kind, _, _ in view["parts"]] == ["marker", "reasoning", "marker", "tool_call", "marker"]

    async def test_a_samples_stream(self, store, mock_server):
        mock_server.props = GEMMA_PROPS
        mock_server.script(MockTurn(content="", logprobs={"content": ANSWER}))
        run_id = record_pilot(store)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_server.app)) as client:
            experiment = await run_resample(
                store, load_checkpoint(store, run_id, 1), SamplingPlan(n=1, logprobs=2),
                base_url=UPSTREAM, client=client,
            )

        listing = step_tokens(store, run_id, "1")
        assert listing.view is None  # the answer's own was never scored
        (listed,) = listing.streams
        assert (listed["experiment_id"], listed["sample_idx"]) == (experiment.id, 0)

        found = sample_tokens(store, experiment.id[:8], 0)
        assert found.view.sample["experiment_id"] == experiment.id
        assert found.view.stream.format == "gemma4"
        assert step_tokens(store, run_id, "1", stream=listed["id"][:8]).view.record.id == listed["id"]
        with pytest.raises(OperationError, match="no token stream"):
            step_tokens(store, run_id, "1", stream="ffffffff")

    async def test_a_sample_that_kept_no_tokens(self, store, mock_server):
        run_id = record_pilot(store)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_server.app)) as client:
            experiment = await run_resample(
                store, load_checkpoint(store, run_id, 1), SamplingPlan(n=1), base_url=UPSTREAM, client=client,
            )
        with pytest.raises(OperationError, match="kept no tokens") as refused:
            sample_tokens(store, experiment.id, 0)
        assert "--logprobs" in refused.value.hint


class TestFrontEnds:
    async def test_the_api(self, proxy_client, recorder):
        run_id = record_pilot(recorder.store)
        step = recorder.store.step(run_id, 1)
        recorder.store.add_token_stream(stream_json(), source="scored", step_id=step.id, method="grammar")

        found = (await proxy_client.get(f"{API}/runs/{run_id}/steps/1/tokens")).json()
        assert found["view"]["stream"]["method"] == "grammar"
        empty = (await proxy_client.get(f"{API}/runs/{run_id}/steps/0/tokens")).json()
        assert empty["view"] is None
        assert (await proxy_client.get(f"{API}/runs/{run_id}/steps/9/tokens")).status_code == 404
        assert (await proxy_client.get(f"{API}/experiments/nothing/samples/0")).status_code == 404

    def test_the_command(self, store, monkeypatch):
        monkeypatch.setattr("tuningfork.cli.console", Console(width=220))
        run_id = record_pilot(store)
        step = store.step(run_id, 1)
        store.add_token_stream(stream_json(), source="scored", step_id=step.id, method="grammar", exact=True)
        runner = CliRunner()

        shown = runner.invoke(app, ["tokens", run_id[:8], "1", "--home", str(store.home)])
        assert shown.exit_code == 0, shown.output
        assert "scored · grammar · top 2 · 14 tokens · rebuilt exactly" in shown.output
        assert "reasoning" in shown.output and "tool call" in shown.output
        assert "[10] tool call 'count' 60% | 'row' 40%" in shown.output
        assert "a lower bound" in shown.output

        as_json = runner.invoke(app, ["tokens", run_id[:8], "1", "--json", "--home", str(store.home)])
        assert json.loads(as_json.output)["view"]["stream"]["tokens"] == 14

        none = runner.invoke(app, ["tokens", run_id[:8], "0", "--home", str(store.home)])
        assert "has not been scored yet" in none.output and "--score" in none.output
