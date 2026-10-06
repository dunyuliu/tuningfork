"""Model output formats, checked against the server's own parse of real output.

The fixtures (`tests/fixtures/formats/<format>/`) are raw generated text as llama.cpp's
logprobs gave it, token by token, beside the message the same server parsed from it; see
`tests/fixtures/formats/capture.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tuningfork.analysis.normalize import outcome_for
from tuningfork.backends.formats import FormatError, format_for, get_format
from tuningfork.backends.formats.gemma4 import Gemma4

FIXTURES = sorted((Path(__file__).parent / "fixtures" / "formats" / "gemma4").glob("*.json"))
GEMMA = Gemma4()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def as_seen(message: dict) -> dict:
    """A message as the fixtures keep the server's: ids and empty fields aside."""
    return {
        "content": message.get("content") or "",
        "reasoning_content": message.get("reasoning_content") or "",
        "tool_calls": [
            {"name": call["function"]["name"], "arguments": call["function"]["arguments"]}
            for call in message.get("tool_calls") or []
        ],
    }


def response(message: dict, finish: str) -> dict:
    return {"choices": [{"finish_reason": finish, "message": message}]}


def server_message(fixture: dict) -> dict:
    seen = fixture["message"]
    message = {"role": "assistant", "content": seen["content"]}
    if seen["tool_calls"]:
        message["tool_calls"] = [
            {"id": f"x{i}", "type": "function", "function": call}
            for i, call in enumerate(seen["tool_calls"])
        ]
    return message


def test_there_are_fixtures_for_every_case():
    assert len(FIXTURES) >= 20
    assert {load(path)["case"] for path in FIXTURES} >= {"text", "parallel", "braces", "unicode", "text-nothink"}


@pytest.mark.parametrize("path", FIXTURES, ids=lambda path: path.stem)
class TestGemma4Fixtures:
    def test_parses_as_the_server_did(self, path):
        fixture = load(path)
        raw = "".join(fixture["pieces"])
        parsed = GEMMA.parse(raw, tools=fixture["tools"], parallel=fixture["parallel"])
        assert as_seen(parsed) == fixture["message"]

    def test_same_behaviour_as_the_server_saw(self, path):
        fixture = load(path)
        parsed = GEMMA.parse("".join(fixture["pieces"]), tools=fixture["tools"])
        finish = fixture["finish_reason"]
        assert outcome_for(response(parsed, finish)).key == outcome_for(response(server_message(fixture), finish)).key

    def test_renders_back_to_the_raw_text(self, path):
        fixture = load(path)
        parsed = GEMMA.parse("".join(fixture["pieces"]), tools=fixture["tools"])
        rendered = GEMMA.render(parsed, thinking=fixture["thinking"])
        # the model's end token has an empty piece: the rest is the text rendered back
        assert fixture["pieces"][-1] == ""
        assert rendered == "".join(fixture["pieces"][:-1])

    def test_parts_follow_the_markers(self, path):
        fixture = load(path)
        pieces = fixture["pieces"]
        parts = GEMMA.parts(pieces)
        assert parts[0][1] == 0 and parts[-1][2] == len(pieces)
        assert all(a[2] == b[1] for a, b in zip(parts, parts[1:]))  # contiguous
        kind = {position: part for part, start, end in parts for position in range(start, end)}
        for position, piece in enumerate(pieces):
            if piece in ("<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>", ""):
                assert kind[position] == "marker", (position, piece)
        assert kind[len(pieces) - 1] == "marker"  # the end token
        reasoning = "".join(p for i, p in enumerate(pieces) if kind[i] == "reasoning")
        assert reasoning == fixture["message"]["reasoning_content"]
        if fixture["message"]["tool_calls"]:
            call = "".join(p for i, p in enumerate(pieces) if kind[i] == "tool_call")
            assert call.startswith("call:")


class TestGemma4:
    def test_detected_by_the_servers_own_rule(self):
        assert format_for("{% if tools %}<|tool_call>call:{{ name }}{% endif %}").name == "gemma4"
        assert format_for("<|im_start|>assistant") is None
        assert format_for(None) is None
        assert get_format("gemma4").name == "gemma4"

    def test_thinking_is_on_when_the_prompt_says_so(self):
        assert GEMMA.thinking("<|turn>system\n<|think|>\nSYS<turn|>")
        assert not GEMMA.thinking("<|turn>user\nhi<turn|>")

    def test_a_number_is_written_back_as_it_was(self):
        message = {"tool_calls": [{"function": {"name": "record_expense", "arguments": '{"amount":61.50,"note":null,"ok":true}'}}]}
        text = GEMMA.render(message, thinking=False)
        assert text == '<|tool_call>call:record_expense{amount:61.50,note:null,ok:true}<tool_call|>'
        assert GEMMA.parse(text, tools=True)["tool_calls"][0]["function"]["arguments"] == '{"amount":61.50,"note":null,"ok":true}'

    def test_an_empty_thought_is_opened_when_thinking_is_on(self):
        assert GEMMA.render({"content": "hi"}, thinking=True) == "<|channel>thought\n<channel|>hi"
        assert GEMMA.render({"content": "hi"}, thinking=False) == "hi"

    def test_one_call_only_when_parallel_calls_are_off(self):
        text = '<|tool_call>call:a{x:1}<tool_call|><|tool_call>call:b{y:2}<tool_call|>'
        assert len(GEMMA.parse(text, tools=True)["tool_calls"]) == 2
        assert len(GEMMA.parse(text, tools=True, parallel=False)["tool_calls"]) == 1

    def test_without_tools_a_call_is_content(self):
        text = "<|channel>thought\nhm<channel|>see <|tool_call>call:a{x:1}<tool_call|>"
        parsed = GEMMA.parse(text, tools=False)
        assert parsed["reasoning_content"] == "hm"
        assert parsed["content"] == "see <|tool_call>call:a{x:1}<tool_call|>"

    @pytest.mark.parametrize("text", [
        '<|tool_call>call:a{x:<|"|>open}<tool_call|>',
        "<|tool_call>call:a{x:1<tool_call|>",
        "<|tool_call>call:a{x:}<tool_call|>",
        "<|tool_call>call:a{x:1}",
    ])
    def test_a_malformed_call_is_an_error(self, text):
        with pytest.raises(FormatError):
            GEMMA.parse(text, tools=True)

    def test_cut_off_while_thinking(self):
        parsed = GEMMA.parse("<|channel>thought\nstill going", tools=True)
        assert parsed["reasoning_content"] == "still going"
        assert parsed["content"] == ""
