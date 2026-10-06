"""Which server answered: recorded with every step, and part of what a pass rate is of."""

from __future__ import annotations

import httpx
import pytest
from rich.console import Console
from typer.testing import CliRunner

from tuningfork.analysis.conditions import conditions
from tuningfork.cli import app
from tuningfork.proxy.recorder import fetch_server_identity
from tuningfork.store.db import Store, identity_id
from tests.helpers import record_run, text_message

PROPS = {
    "build_info": "b11108-abc",
    "model_path": "/models/lmstudio-community/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q4_K_M.gguf",
    "total_slots": 4,
    "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}",
    "default_generation_settings": {"n_ctx": 65536, "params": {"temperature": 1.0, "top_p": 0.95}},
}


@pytest.fixture
def client(mock_server):
    mock_server.props = dict(PROPS)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_server.app))


class TestIdentity:
    async def test_what_the_server_says_about_itself(self, client):
        identity = await fetch_server_identity(client, "http://upstream.test/v1")
        await client.aclose()

        assert identity["build"] == "b11108-abc"
        assert identity["model_file"] == "gemma-4-31B-it-Q4_K_M.gguf"
        assert identity["n_ctx"] == 65536 and identity["slots"] == 4
        assert len(identity["chat_template_sha256"]) == 16
        assert identity["defaults"]["temperature"] == 1.0
        assert identity["upstream"] == "http://upstream.test"

    async def test_a_changed_template_or_context_is_another_server(self, client, mock_server):
        before = identity_id(await fetch_server_identity(client, "http://upstream.test"))
        mock_server.props = {**PROPS, "chat_template": "{{ something else }}"}
        template = identity_id(await fetch_server_identity(client, "http://upstream.test"))
        mock_server.props = {**PROPS, "default_generation_settings": {"n_ctx": 32768}}
        context = identity_id(await fetch_server_identity(client, "http://upstream.test"))
        await client.aclose()
        assert len({before, template, context}) == 3

    async def test_where_it_was_reached_is_not_what_it_is(self, client):
        """`localhost` and `127.0.0.1`, or two ports serving alike, are one condition."""
        one = await fetch_server_identity(client, "http://127.0.0.1:8081")
        other = await fetch_server_identity(client, "http://localhost:9000/v1")
        await client.aclose()
        assert one["upstream"] != other["upstream"]
        assert identity_id(one) == identity_id(other)


class TestConditions:
    def test_runs_answered_by_different_servers_are_different_samples(self, store):
        a = store.add_server({"build": "b1", "model_file": "gemma.gguf"})
        b = store.add_server({"build": "lmstudio", "model_file": "gemma.gguf"})
        for index, server in enumerate([a, a, b]):
            run_id = record_run(store, f"r{index}", [text_message("x")], task_id="t/1")
            with store._lock:
                store._conn.execute("UPDATE runs SET server_id = ? WHERE id = ?", (server, run_id))
                store._conn.commit()

        found = conditions(store.runs())

        assert sorted(len(c.runs) for c in found) == [1, 2]
        assert {c.server_id for c in found} == {a, b}

    def test_tasks_names_the_server_only_when_it_varies(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tuningfork.cli.console", Console(width=220))
        home = tmp_path / "store"
        store = Store(home)
        a = store.add_server({"build": "b1"})
        b = store.add_server({"build": "b2"})
        for index, server in enumerate([a, b]):
            run_id = record_run(store, f"r{index}", [text_message("x")], task_id="t/1")
            store.annotate_run(run_id, harness="hermes", result={"passed": True, "usable": True})
            with store._lock:
                store._conn.execute("UPDATE runs SET server_id = ? WHERE id = ?", (server, run_id))
                store._conn.commit()
        store.close()

        tasks = CliRunner().invoke(app, ["tasks", "--home", str(home)])
        servers = CliRunner().invoke(app, ["servers", "--home", str(home)])

        assert "server" in tasks.output and a[:8] in tasks.output and b[:8] in tasks.output
        assert "b1" in servers.output and "b2" in servers.output

    def test_a_recorded_step_is_filed_under_its_server(self, store):
        server = store.add_server({"build": "b1"})
        step = store.record_step(
            request={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            server_id=server,
        )
        assert step.server_id == server
        assert store.get_run(step.run_id).server_id == server
        assert store.server(server) == {"build": "b1"}
