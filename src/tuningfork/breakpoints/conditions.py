"""Breakpoint conditions: when to pause a model call.

Two phases:

- **request** — before the model sees the context. The context can be inspected and edited.
- **response** — after the model has generated, before the harness acts on it. The
  model's decision can be inspected and edited: this is where "stop before the first
  write_file of a *.json file" lives, because that decision only exists once the model has
  made it.

Every condition set on a breakpoint must hold for it to fire. Request-side conditions can
also be used in the response phase, since the request is known then too; response-side
conditions in the request phase are rejected, because there is nothing yet to match them
against.
"""

from __future__ import annotations

import fnmatch
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable

PHASES = ("request", "response")
REQUEST_CONDITIONS = ("step_min", "step_max", "after_tool", "context_regex")
RESPONSE_CONDITIONS = ("tool", "path_glob", "args_regex", "output_regex", "diverges_from", "no_reasoning")
PATH_KEYS = ("path", "file_path", "filename", "file")


#: Fields the engine owns. A client that could set them could break the engine's own
#: bookkeeping — and `id` would let it shadow or impersonate another breakpoint.
ENGINE_FIELDS = frozenset({"id", "hits", "created_at"})
_WHOLE_NUMBERS = ("step_min", "step_max")
_FLAGS = ("enabled", "once")
#: True or false where given; absent, no condition.
_OPTIONAL_FLAGS = ("no_reasoning",)


class BreakpointError(ValueError):
    """A breakpoint definition that cannot work."""


def _check_type(name: str, value: Any) -> None:
    if name in _FLAGS:
        if not isinstance(value, bool):
            raise BreakpointError(f"{name} must be true or false")
    elif value is None:
        return
    elif name in _OPTIONAL_FLAGS:
        if not isinstance(value, bool):
            raise BreakpointError(f"{name} must be true or false")
    elif name in _WHOLE_NUMBERS:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BreakpointError(f"{name} must be a whole number, 0 or more")
    elif not isinstance(value, str):
        raise BreakpointError(f"{name} must be a string")


@dataclass(slots=True)
class ExchangeContext:
    """Everything known about a model call before its response exists."""

    request: dict[str, Any]
    #: The run this request would join, and its step index — predicted, since the step is
    #: only recorded after the response. None for a request that would start a new run.
    run_id: str | None
    task_id: str | None
    step: int
    #: Node path of the request as the harness sent it; used to arm "step" breaks.
    path: list[str] = field(default_factory=list)
    #: The conversation as OpenAI-style messages, whatever the call's wire format — what
    #: conditions read. None means the request is already in that shape.
    view: list[dict[str, Any]] | None = None
    wire: str = "openai-chat"
    #: The launch the call belongs to, for a harness tuningfork started.
    launch_id: str | None = None

    @property
    def messages(self) -> list[dict[str, Any]]:
        if self.view is not None:
            return self.view
        return self.request.get("messages") or []


@dataclass(slots=True)
class Breakpoint:
    phase: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    enabled: bool = True
    label: str | None = None
    # scope
    run: str | None = None
    task: str | None = None
    #: Only the calls of one launch: a fork started to stop at its step, and nothing else.
    launch: str | None = None
    # request-side conditions
    step_min: int | None = None
    step_max: int | None = None
    after_tool: str | None = None
    context_regex: str | None = None
    # response-side conditions
    tool: str | None = None
    path_glob: str | None = None
    args_regex: str | None = None
    output_regex: str | None = None
    diverges_from: str | None = None
    #: The model answered without reasoning (`analysis.reasoning`) — often right after a tool
    #: call failed, when it tries the other way at once.
    no_reasoning: bool | None = None
    # behaviour
    once: bool = False
    hits: int = 0
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.phase not in PHASES:
            raise BreakpointError(f"phase must be one of {', '.join(PHASES)}")
        if self.phase == "request":
            misplaced = [name for name in RESPONSE_CONDITIONS if getattr(self, name) not in (None, False)]
            if misplaced:
                raise BreakpointError(
                    f"{', '.join(misplaced)} can only be matched in the response phase: "
                    "before the model has answered, there is no tool call or output yet"
                )
        for name in ("context_regex", "args_regex", "output_regex"):
            pattern = getattr(self, name)
            if pattern is not None:
                try:
                    re.compile(pattern)
                except re.error as exc:
                    raise BreakpointError(f"{name} is not a valid regex: {exc}") from None
        if (
            self.step_min is not None
            and self.step_max is not None
            and self.step_min > self.step_max
        ):
            raise BreakpointError("step_min is greater than step_max")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Breakpoint:
        """A breakpoint from a client's definition, checked field by field.

        Checked here because nothing downstream expects a wrong type: a `created_at` of
        "x" once broke the sort behind every listing, and with the bad breakpoint already
        installed, every model call through the proxy failed until it was restarted.
        """
        if not isinstance(data, dict):
            raise BreakpointError("a breakpoint definition is a JSON object")
        internal = sorted(set(data) & ENGINE_FIELDS)
        if internal:
            raise BreakpointError(f"{', '.join(internal)} are set by tuningfork, not by a client")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise BreakpointError(f"unknown breakpoint field(s): {', '.join(unknown)}")
        if "phase" not in data:
            raise BreakpointError("a breakpoint needs a phase: request or response")
        for name, value in data.items():
            _check_type(name, value)
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> dict[str, Any]:
        """For the API and the UI: every field, plus the one-line description."""
        return {**asdict(self), "description": self.describe()}

    def describe(self) -> str:
        """A one-line summary for lists and hold reasons."""
        parts = []
        if self.tool or self.path_glob:
            call = self.tool or "any tool"
            parts.append(f"{call} on {self.path_glob}" if self.path_glob else call)
        if self.args_regex:
            parts.append(f"arguments ~ /{self.args_regex}/")
        if self.output_regex:
            parts.append(f"output ~ /{self.output_regex}/")
        if self.diverges_from:
            parts.append(f"diverging from run {self.diverges_from[:8]}")
        if self.no_reasoning:
            parts.append("calling a tool with no reasoning first")
        if self.after_tool:
            parts.append(f"after {self.after_tool} returns")
        if self.context_regex:
            parts.append(f"context ~ /{self.context_regex}/")
        if self.step_min is not None or self.step_max is not None:
            if self.step_min == self.step_max:
                parts.append(f"step {self.step_min}")
            elif self.step_max is None:
                parts.append(f"step >= {self.step_min}")
            elif self.step_min is None:
                parts.append(f"step <= {self.step_max}")
            else:
                parts.append(f"steps {self.step_min}-{self.step_max}")
        if self.run:
            parts.append(f"run {self.run[:8]}")
        if self.task:
            parts.append(f"task {self.task}")
        if self.launch:
            parts.append(f"launch {self.launch[:8]}")
        when = "before the model" if self.phase == "request" else "after the model"
        return f"{when}: {', '.join(parts) or 'every call'}" + (" (once)" if self.once else "")


# --- extracting what conditions look at ----------------------------------------------


def tool_calls(response: dict[str, Any] | None) -> list[tuple[str, Any]]:
    """(name, parsed arguments) for every tool call in a response."""
    if not response:
        return []
    choices = response.get("choices") or [{}]
    message = choices[0].get("message") or {}
    calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        raw = function.get("arguments")
        try:
            arguments = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            arguments = raw
        calls.append((str(function.get("name") or ""), arguments))
    return calls


def _output_text(response: dict[str, Any] | None) -> str:
    if not response:
        return ""
    message = (response.get("choices") or [{}])[0].get("message") or {}
    parts = [
        str(message.get(key))
        for key in ("reasoning_content", "reasoning", "content")
        if message.get(key)
    ]
    parts += [f"{name}({arguments_text(arguments)})" for name, arguments in tool_calls(response)]
    return "\n".join(parts)


def _context_text(messages: list[dict[str, Any]]) -> str:
    parts = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts += [p.get("text", "") for p in content if isinstance(p, dict)]
        for call in message.get("tool_calls") or []:
            parts.append(json.dumps(call.get("function") or {}, sort_keys=True))
    return "\n".join(parts)


def last_tool_result(messages: list[dict[str, Any]]) -> str | None:
    """Name of the tool whose result the model is about to read: the last tool result
    since the model's last turn.

    Not simply "the context ends with a tool result": a harness may add text to the turn
    after a result — Claude Code follows a loaded skill's result with the skill itself.
    """
    call_id = None
    for message in reversed(messages):
        role = message.get("role")
        if role == "assistant":
            break
        if role == "tool":
            call_id = message.get("tool_call_id")
            break
    if call_id is None:
        return None
    for message in reversed(messages):
        for call in message.get("tool_calls") or []:
            if call.get("id") == call_id:
                return (call.get("function") or {}).get("name")
    return None


def arguments_text(arguments: Any) -> str:
    """Tool arguments as a person reads them, for regex matching.

    Deliberately not a JSON dump: a string argument that itself contains JSON — the content
    of a written `.json` file — would appear escaped (`{\\"count\\": 4}`), and a pattern
    like `"count"` would silently never match. Each value is rendered raw instead.
    """
    if isinstance(arguments, dict):
        return "\n".join(
            f"{key}={value if isinstance(value, str) else json.dumps(value, sort_keys=True)}"
            for key, value in sorted(arguments.items())
        )
    return arguments if isinstance(arguments, str) else json.dumps(arguments, sort_keys=True)


def _path_values(arguments: Any) -> list[str]:
    if not isinstance(arguments, dict):
        return []
    return [str(arguments[key]) for key in PATH_KEYS if isinstance(arguments.get(key), str)]


# --- matching -----------------------------------------------------------------------


def request_matches(bp: Breakpoint, ctx: ExchangeContext) -> bool:
    """Whether the request-side conditions (and scope) hold for this call."""
    if not bp.enabled:
        return False
    if bp.run and not (ctx.run_id and ctx.run_id.startswith(bp.run)):
        return False
    if bp.task and ctx.task_id != bp.task:
        return False
    if bp.launch and ctx.launch_id != bp.launch:
        return False
    if bp.step_min is not None and ctx.step < bp.step_min:
        return False
    if bp.step_max is not None and ctx.step > bp.step_max:
        return False
    if bp.after_tool:
        name = last_tool_result(ctx.messages)
        if name is None or not fnmatch.fnmatchcase(name, bp.after_tool):
            return False
    if bp.context_regex and not re.search(bp.context_regex, _context_text(ctx.messages)):
        return False
    return True


def response_matches(
    bp: Breakpoint,
    ctx: ExchangeContext,
    response: dict[str, Any] | None,
    diverged: Callable[[str, int, dict[str, Any] | None], bool] | None = None,
) -> bool:
    """Whether every condition holds, once the model's response exists."""
    if not request_matches(bp, ctx):
        return False

    if bp.tool or bp.path_glob or bp.args_regex:
        # Tool, path and arguments must all hold for the *same* call: "write_file on
        # *.json" must not fire for a read_file of a .json plus a write_file of a .py.
        def call_matches(name: str, arguments: Any) -> bool:
            if bp.tool and not fnmatch.fnmatchcase(name, bp.tool):
                return False
            if bp.path_glob and not any(
                fnmatch.fnmatchcase(value, bp.path_glob)
                or fnmatch.fnmatchcase(value.rsplit("/", 1)[-1], bp.path_glob)
                for value in _path_values(arguments)
            ):
                return False
            if bp.args_regex and not re.search(bp.args_regex, arguments_text(arguments)):
                return False
            return True

        if not any(call_matches(name, arguments) for name, arguments in tool_calls(response)):
            return False

    if bp.output_regex and not re.search(bp.output_regex, _output_text(response)):
        return False

    if bp.no_reasoning:
        from tuningfork.analysis.reasoning import chars_in

        # A tool call made without reasoning: a reply in text without any is the usual thing.
        if response is None or chars_in(response) > 0 or not tool_calls(response):
            return False

    if bp.diverges_from:
        if diverged is None or not diverged(bp.diverges_from, ctx.step, response):
            return False
    return True
