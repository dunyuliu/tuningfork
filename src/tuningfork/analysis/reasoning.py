"""Whether the model reasoned before it acted, and when its not reasoning is worth saying.

Every response records how much reasoning came with it (`Step.reasoning_chars`, counted from
the text; `Sample.reasoning_chars` the same). Zero means one of two things: the model went
straight to its answer, or the model — or the server, or the harness — never shows any. Only
the first says something about the step, so a step is marked "none" only where the same run
has reasoning at another step; in a run with none anywhere there is nothing to compare with.

Only a step where the model acted — called a tool — is marked. A final reply, or a question,
written without reasoning is what models usually do: Hermes on Gemma 4 ended three runs of
four that way, each having reasoned before every tool call.

A model that skips its reasoning often does so right after a tool call failed: it tries the
other way at once. `after_failed_call` says when that is the case, so a page can say both.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

REASONED, NONE = "reasoned", "none"
#: Outcomes where the model acted, so acting without reasoning is worth saying. Not a reply
#: in text, an error, an empty or a cut-off answer; nor a call replayed from a recording.
ACTED = frozenset({"tool_call"})


def reasons(chars: Iterable[int | None]) -> bool:
    """Whether reasoning shows at all here: any response with some."""
    return any((count or 0) > 0 for count in chars)


def status(chars: int | None, *, reasons: bool, acted: bool = True) -> str | None:
    """`reasoned`, `none` — acted without reasoning, where reasoning shows elsewhere — or
    None, when there is nothing to say."""
    if not acted:
        return None
    if (chars or 0) > 0:
        return REASONED
    return NONE if reasons else None


def chars_in(response: dict[str, Any] | None) -> int:
    """The reasoning in a response, in characters, as the store counts it."""
    message = ((response or {}).get("choices") or [{}])[0].get("message") or {}
    return sum(len(message[key]) for key in ("reasoning_content", "reasoning") if isinstance(message.get(key), str))


#: How a JSON tool result gives a command's exit status: Hermes's terminal tool says
#: `{"output": "…: command not found", "exit_code": 127, "error": null}`.
EXIT_KEYS = ("exit_code", "returncode", "exitCode")


def _failed(message: dict[str, Any]) -> bool:
    """Whether a tool result reports failure: Anthropic marks it (`is_error`); for others, a
    JSON result with an error, a failed status or a command's non-zero exit status, or a text
    that starts by saying so."""
    if message.get("is_error"):
        return True
    content = message.get("content")
    if isinstance(content, list):
        content = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    if not isinstance(content, str):
        return False
    text = content.strip()
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except ValueError:
            data = None
        if isinstance(data, dict):
            exits = [data[key] for key in EXIT_KEYS if isinstance(data.get(key), int) and not isinstance(data.get(key), bool)]
            return (bool(data.get("error")) or str(data.get("status", "")).lower() in ("error", "failed", "failure")
                    or data.get("success") is False or any(code != 0 for code in exits))
    return text.lower().startswith(("error", "<tool_use_error>"))


def after_failed_call(messages: list[dict[str, Any]]) -> bool:
    """Whether a tool result the model is about to read — any since its last turn — reports
    failure. `messages` is a request's view, in the OpenAI shape every wire converts to."""
    for message in reversed(messages):
        role = message.get("role")
        if role == "assistant":
            return False
        if role == "tool" and _failed(message):
            return True
    return False
