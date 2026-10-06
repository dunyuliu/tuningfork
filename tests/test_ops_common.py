"""What every operation shares: cancellation, progress events, typed parameters, owners."""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from dataclasses import dataclass, field
from typing import Literal

import pytest

from tuningfork.ops import Cancel, Cancelled, Event, OperationError, emit, params_schema, parse_params
from tuningfork.ops.common import current_owner, owner_alive
from tuningfork.ops.experiments import _cancellable


class TestCancel:
    def test_a_failing_callback_does_not_stop_the_others(self, caplog):
        """Found in review: one callback raising left every later one — a harness to kill —
        uncalled."""
        cancel, calls = Cancel(), []

        def broken():
            raise RuntimeError("boom")

        cancel.on_cancel(broken)
        cancel.on_cancel(lambda: calls.append("killed"))
        cancel.cancel()

        assert calls == ["killed"]
        assert "cancellation callback failed" in caplog.text

    def test_a_callback_can_be_unregistered(self):
        cancel, calls = Cancel(), []
        unregister = cancel.on_cancel(lambda: calls.append(1))
        unregister()
        cancel.cancel()
        assert calls == []

    def test_a_callback_registered_after_the_cancel_runs_at_once(self):
        cancel, calls = Cancel(), []
        cancel.cancel("shutdown")
        cancel.on_cancel(lambda: calls.append(1))
        assert calls == [1]

    def test_check_raises_with_the_reason_it_was_given(self):
        cancel = Cancel()
        cancel.check()
        cancel.cancel("shutdown")
        cancel.cancel("cancelled")  # only the first counts
        with pytest.raises(Cancelled) as raised:
            cancel.check()
        assert raised.value.reason == "shutdown"

    def test_wait_wakes_when_cancelled(self):
        cancel = Cancel()
        assert cancel.wait(0.01) is False
        threading.Timer(0.05, cancel.cancel).start()
        assert cancel.wait(5.0) is True


class TestCancellableWork:
    async def test_a_cancel_stops_the_work_as_cancelled(self):
        cancel = Cancel()

        async def forever():
            await asyncio.sleep(60)

        asyncio.get_running_loop().call_later(0.05, cancel.cancel)
        with pytest.raises(Cancelled):
            await _cancellable(forever(), cancel)

    async def test_its_callback_is_gone_once_the_work_is_done(self):
        cancel = Cancel()

        async def quick():
            return 7

        assert await _cancellable(quick(), cancel) == 7
        cancel.cancel()  # would schedule on a loop that may be gone; must not


class TestEvents:
    def test_the_subject_never_leaves_the_process(self):
        seen = []
        emit(seen.append, "sample", subject=object(), index=3)
        (event,) = seen
        assert isinstance(event, Event)
        assert event.to_json() == {"kind": "sample", "data": {"index": 3}}

    def test_emit_without_a_listener_does_nothing(self):
        emit(None, "sample", index=1)


@dataclass
class Params:
    run: str
    n: int = field(default=20, metadata={"help": "samples to draw", "minimum": 1})
    temperature: float | None = None
    normalizers: list[str] = field(default_factory=list)
    overrides: dict[str, float] = field(default_factory=dict)
    held_out: bool = False
    reach: Literal["step", "end"] = "step"


class TestParams:
    def test_a_well_formed_request(self):
        params = parse_params(Params, {"run": "abc", "n": 5, "temperature": 1, "normalizers": ["tool_name"]})
        assert params == Params(run="abc", n=5, temperature=1.0, normalizers=["tool_name"])

    @pytest.mark.parametrize(
        "data, field_name",
        [
            ({"run": "a", "held_out": "false"}, "held_out"),  # the string "false" is truthy
            ({"run": "a", "n": True}, "n"),  # a bool is an int to Python
            ({"run": "a", "n": 2.5}, "n"),
            ({"run": "a", "normalizers": "tool_name"}, "normalizers"),
            ({"run": "a", "overrides": {"temperature": "hot"}}, "overrides"),
            ({"run": "a", "reach": "forever"}, "reach"),
            ({"run": "a", "tempreature": 1.0}, "tempreature"),
            ({}, "run"),
        ],
    )
    def test_near_misses_are_refused_and_named(self, data, field_name):
        with pytest.raises(OperationError) as refused:
            parse_params(Params, data)
        assert refused.value.field == field_name
        assert refused.value.code == "bad_params"

    def test_the_schema_is_what_a_form_needs(self):
        schema = params_schema(Params)
        assert schema["required"] == ["run"]
        assert schema["properties"]["n"] == {
            "type": "integer", "description": "samples to draw", "default": 20, "minimum": 1,
        }
        # Optional: its plain type, null by default — the parser takes null for it too.
        assert schema["properties"]["temperature"] == {"type": "number", "default": None}
        assert schema["properties"]["reach"] == {"type": "string", "enum": ["step", "end"], "default": "step"}
        assert schema["properties"]["normalizers"]["default"] == []
        assert schema["additionalProperties"] is False


class TestCatalogSchemas:
    """Every job kind's schema is real JSON Schema, and says what the parser accepts: an
    agent's tool list is built from it, and llama.cpp builds a tool call's grammar from that."""

    @staticmethod
    def kinds():
        from tuningfork.ops.catalog import CATALOG

        return list(CATALOG.values())

    def test_each_is_valid_json_schema(self):
        jsonschema = pytest.importorskip("jsonschema")
        for kind in self.kinds():
            schema = params_schema(kind.params)
            jsonschema.Draft202012Validator.check_schema(schema)
            assert schema["additionalProperties"] is False, kind.name

    def test_each_kinds_defaults_are_what_its_parser_takes(self):
        filler = {"string": "x", "integer": 1, "number": 1.0, "boolean": True}
        for kind in self.kinds():
            schema = params_schema(kind.params)
            data = {name: entry["default"] for name, entry in schema["properties"].items() if "default" in entry}
            for name in schema["required"]:
                data[name] = filler[schema["properties"][name]["type"]]
            parse_params(kind.params, data)  # refuses anything its own schema allowed

    def test_shapes_a_type_cannot_say_are_spelled_out(self):
        from tuningfork.analysis.distribution import RULES
        from tuningfork.ops.catalog import CATALOG

        rules = params_schema(CATALOG["ab"].params)["properties"]["target_rules"]
        assert rules["items"]["properties"]["if"]["enum"] == list(RULES)
        assert rules["items"]["required"] == ["if", "value"]
        trial = params_schema(CATALOG["check-trial"].params)["properties"]
        assert trial["files"]["items"]["required"] == ["path", "content"]
        assert [option["type"] for option in trial["check"]["anyOf"]] == ["string", "object"]


class TestOwners:
    @pytest.mark.skipif(sys.platform == "win32", reason="probing a pid ends it on Windows")
    def test_this_process_is_alive_and_a_gone_one_is_not(self):
        me = current_owner()
        assert owner_alive(me) is True
        assert owner_alive({**me, "pid": 2**22 + 12345}) is False

    def test_another_host_cannot_be_told(self):
        assert owner_alive({"pid": os.getpid(), "host": "elsewhere", "start": None}) is None
        assert owner_alive(None) is None
