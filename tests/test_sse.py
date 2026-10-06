"""SSE parsing and chunk reassembly.

The shapes asserted here are the ones observed on the real stack, not invented: the
`reasoning_content` delta field, a tool call whose arguments arrive whole in a later chunk,
and a trailing usage chunk with an empty `choices` list.
"""

from __future__ import annotations

import json

import pytest

from tuningfork.proxy.sse import (
    CUT_OFF,
    ChunkAssembler,
    SSEParser,
    reassemble,
    render_sse,
    upstream_error,
)


def sse(obj: object) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode()


def chunk(delta: dict | None = None, **kwargs) -> dict:
    choice = {"index": 0, "delta": delta if delta is not None else {}}
    choice.update(kwargs)
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "model": "m",
        "choices": [choice],
    }


class TestParser:
    def test_splits_events_on_blank_lines(self):
        parser = SSEParser()
        events = parser.feed(b'data: {"a": 1}\n\ndata: {"a": 2}\n\n')
        assert [e.json() for e in events] == [{"a": 1}, {"a": 2}]

    def test_reassembles_a_line_split_across_reads(self):
        parser = SSEParser()
        assert parser.feed(b'data: {"a"') == []
        assert parser.feed(b": 1}\n") == []
        events = parser.feed(b"\n")
        assert [e.json() for e in events] == [{"a": 1}]

    def test_ignores_comment_keepalives(self):
        parser = SSEParser()
        events = parser.feed(b': keepalive\n\ndata: {"a": 1}\n\n')
        assert [e.json() for e in events] == [{"a": 1}]

    def test_handles_crlf_and_missing_space_after_colon(self):
        parser = SSEParser()
        events = parser.feed(b'data:{"a": 1}\r\n\r\n')
        assert [e.json() for e in events] == [{"a": 1}]

    def test_joins_multiple_data_lines(self):
        parser = SSEParser()
        events = parser.feed(b'data: {"a":\ndata: 1}\n\n')
        assert [e.json() for e in events] == [{"a": 1}]

    def test_done_sentinel(self):
        parser = SSEParser()
        (event,) = parser.feed(b"data: [DONE]\n\n")
        assert event.is_done
        assert event.json() is None

    def test_non_json_payload_does_not_raise(self):
        parser = SSEParser()
        (event,) = parser.feed(b"data: not json\n\n")
        assert event.json() is None


class TestAssembler:
    def test_accumulates_content_and_reasoning_separately(self):
        body = (
            sse(chunk({"role": "assistant"}))
            + sse(chunk({"reasoning_content": "The user"}))
            + sse(chunk({"reasoning_content": " wants a greeting."}))
            + sse(chunk({"content": "hel"}))
            + sse(chunk({"content": "lo"}))
            + sse(chunk({}, finish_reason="stop"))
            + b"data: [DONE]\n\n"
        )
        message = reassemble(body)["choices"][0]["message"]
        assert message["content"] == "hello"
        assert message["reasoning_content"] == "The user wants a greeting."
        assert message["role"] == "assistant"

    def test_tool_call_arguments_arrive_in_a_later_chunk(self):
        opening = {
            "index": 0,
            "id": "385339560",
            "type": "function",
            "function": {"name": "get_weather", "arguments": ""},
        }
        arguments = {
            "index": 0,
            "type": "function",
            "function": {"arguments": '{"city":"Paris"}'},
        }
        body = (
            sse(chunk({"role": "assistant"}))
            + sse(chunk({"tool_calls": [opening]}))
            + sse(chunk({"tool_calls": [arguments]}))
            + sse(chunk({}, finish_reason="tool_calls"))
            + b"data: [DONE]\n\n"
        )
        choice = reassemble(body)["choices"][0]
        (call,) = choice["message"]["tool_calls"]
        assert call["id"] == "385339560"
        assert call["function"] == {
            "name": "get_weather",
            "arguments": '{"city":"Paris"}',
        }
        assert choice["finish_reason"] == "tool_calls"

    def test_parallel_tool_calls_stay_separate(self):
        body = (
            sse(chunk({"tool_calls": [{"index": 0, "id": "a", "function": {"name": "f"}}]}))
            + sse(chunk({"tool_calls": [{"index": 1, "id": "b", "function": {"name": "g"}}]}))
            + sse(chunk({"tool_calls": [{"index": 1, "function": {"arguments": "{}"}}]}))
            + sse(chunk({"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}))
            + b"data: [DONE]\n\n"
        )
        calls = reassemble(body)["choices"][0]["message"]["tool_calls"]
        assert [c["id"] for c in calls] == ["a", "b"]
        assert [c["function"]["name"] for c in calls] == ["f", "g"]

    def test_usage_chunk_has_no_choices(self):
        usage = {"prompt_tokens": 20, "completion_tokens": 36, "total_tokens": 56}
        body = (
            sse(chunk({"content": "hi"}))
            + sse(chunk({}, finish_reason="stop"))
            + sse(
                {
                    "id": "chatcmpl-1",
                    "object": "chat.completion.chunk",
                    "model": "m",
                    "choices": [],
                    "usage": usage,
                }
            )
            + b"data: [DONE]\n\n"
        )
        assert reassemble(body)["usage"] == usage

    def test_unknown_string_delta_fields_survive(self):
        """A vendor extension must be captured without this module knowing its name."""
        body = sse(chunk({"some_future_channel": "abc"})) + b"data: [DONE]\n\n"
        message = reassemble(body)["choices"][0]["message"]
        assert message["some_future_channel"] == "abc"

    def test_content_key_always_present(self):
        body = sse(chunk({"reasoning_content": "thinking"})) + b"data: [DONE]\n\n"
        assert reassemble(body)["choices"][0]["message"]["content"] == ""

    def test_logprobs_accumulate_across_chunks(self):
        body = (
            sse(chunk({"content": "a"}, logprobs={"content": [{"token": "a"}]}))
            + sse(chunk({"content": "b"}, logprobs={"content": [{"token": "b"}]}))
            + b"data: [DONE]\n\n"
        )
        entries = reassemble(body)["choices"][0]["logprobs"]["content"]
        assert [e["token"] for e in entries] == ["a", "b"]

    def test_tracks_done_and_chunk_count(self):
        parser, assembler = SSEParser(), ChunkAssembler()
        for event in parser.feed(sse(chunk({"content": "x"})) + b"data: [DONE]\n\n"):
            assembler.add_event(event)
        assert assembler.chunk_count == 1
        assert assembler.saw_done is True

    def test_truncated_stream_still_assembles(self):
        """A stream cut off mid-flight must still yield what was received."""
        body = sse(chunk({"content": "par"})) + sse(chunk({"content": "tial"}))
        result = reassemble(body)
        assert result["choices"][0]["message"]["content"] == "partial"
        assert result["choices"][0]["finish_reason"] is None


class TestShapesServersSend:
    """Shapes the assembler did not expect once raised, and a raise in the proxy's copy of
    a stream ended the harness's stream with it."""

    def test_arguments_sent_as_an_object_become_json_text(self):
        body = (
            sse(chunk({"tool_calls": [{"index": 0, "id": "a", "function": {"name": "f", "arguments": ""}}]}))
            + sse(chunk({"tool_calls": [{"index": 0, "function": {"arguments": {"path": "x.json"}}}]}))
            + sse(chunk({}, finish_reason="tool_calls"))
            + b"data: [DONE]\n\n"
        )
        (call,) = reassemble(body)["choices"][0]["message"]["tool_calls"]
        # The common shape carries arguments as text, as OpenAI's own responses do.
        assert json.loads(call["function"]["arguments"]) == {"path": "x.json"}

    def test_a_null_index_is_the_first_choice(self):
        body = sse(chunk({"content": "a"}, index=None)) + sse(chunk({"content": "b"}))
        (choice,) = reassemble(body)["choices"]
        assert choice["message"]["content"] == "ab"

    def test_a_function_that_is_not_an_object_is_passed_over(self):
        body = (
            sse(chunk({"tool_calls": [{"index": 0, "id": "a", "function": "f"}]}))
            + sse(chunk({"tool_calls": [{"index": 0, "function": {"name": "f", "arguments": "{}"}}]}))
        )
        (call,) = reassemble(body)["choices"][0]["message"]["tool_calls"]
        assert call["function"] == {"name": "f", "arguments": "{}"}

    def test_choices_that_are_not_a_list_are_passed_over(self):
        body = sse({"id": "c", "choices": 5}) + sse(chunk({"content": "ok"}))
        assert reassemble(body)["choices"][0]["message"]["content"] == "ok"


class TestStreamFailure:
    """Whether a stream was a whole answer — a partial one is not behaviour."""

    @staticmethod
    def assembled(body: bytes) -> ChunkAssembler:
        assembler = ChunkAssembler()
        for event in SSEParser().feed(body):
            assembler.add_event(event)
        return assembler

    @pytest.mark.parametrize(
        "ending",
        [
            sse(chunk({}, finish_reason="stop")) + b"data: [DONE]\n\n",
            # Some servers never send [DONE]; the finish reason is the end.
            sse(chunk({}, finish_reason="stop")),
            # ...and [DONE] is the server's word that it is finished, finish reason or not.
            b"data: [DONE]\n\n",
        ],
    )
    def test_a_stream_that_ended_properly_is_whole(self, ending):
        assert self.assembled(sse(chunk({"content": "hi"})) + ending).failure() is None

    def test_neither_done_nor_a_finish_reason_means_it_was_cut_off(self):
        assembler = self.assembled(sse(chunk({"content": "par"})) + sse(chunk({"content": "tial"})))
        assert assembler.failure() == CUT_OFF

    def test_an_error_mid_stream_is_a_failure_and_stays_visible(self):
        error = {"message": "the model crashed", "type": "server_error", "code": 500}
        assembler = self.assembled(sse(chunk({"content": "par"})) + sse({"error": error}))

        assert assembler.failure() == "upstream_error: the model crashed"
        result = assembler.result()
        assert result["error"] == error
        assert result["choices"][0]["message"]["content"] == "par"  # what did arrive


class TestUpstreamError:
    @pytest.mark.parametrize(
        "error, expected",
        [
            ({"type": "overloaded_error", "message": "Overloaded"}, "upstream_error: Overloaded"),
            ({"type": "overloaded_error"}, "upstream_error: overloaded_error"),
            ("plain text", "upstream_error: plain text"),
            (None, "upstream_error"),
            ({"message": {"nested": True}}, "upstream_error"),
        ],
    )
    def test_the_message_is_kept_when_there_is_one(self, error, expected):
        assert upstream_error(error) == expected


class TestRender:
    """An edited response must round-trip: rendered, then reassembled, it is unchanged."""

    RESPONSE = {
        "id": "chatcmpl-1",
        "model": "m",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "I should write the report.",
                    "tool_calls": [
                        {
                            "id": "call_7",
                            "type": "function",
                            "index": 0,
                            "function": {
                                "name": "write_file",
                                "arguments": '{"path": "report.json", "content": "{}"}',
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }

    def test_round_trip(self):
        assert reassemble(render_sse(self.RESPONSE)) == self.RESPONSE

    def test_the_stream_ends_the_way_clients_expect(self):
        body = render_sse(self.RESPONSE)
        assert body.endswith(b"data: [DONE]\n\n")
        assert b'"finish_reason": "tool_calls"' in body

    def test_usage_is_only_sent_when_asked_for(self):
        assert b'"usage"' not in render_sse(self.RESPONSE, include_usage=False)

    def test_plain_text_answer(self):
        response = {
            "choices": [
                {"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": "hello"}}
            ]
        }
        result = reassemble(render_sse(response))
        assert result["choices"][0]["message"]["content"] == "hello"
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_a_missing_finish_reason_is_inferred(self):
        response = {"choices": [{"message": {"role": "assistant", "content": "x"}}]}
        assert reassemble(render_sse(response))["choices"][0]["finish_reason"] == "stop"
