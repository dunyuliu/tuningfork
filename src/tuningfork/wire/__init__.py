"""Wire formats: how tuningfork reads and writes each dialect of model call.

See `base.py`. Adding a dialect means adding a module here and listing it in `WIRES`.
"""

from __future__ import annotations

from tuningfork.wire.anthropic import AnthropicMessages
from tuningfork.wire.base import Assembler, WireFormat
from tuningfork.wire.completion import RawCompletion
from tuningfork.wire.openai import OpenAIChat

#: Chat dialects first: `…/chat/completions` must never be read as a raw completion.
WIRES: tuple[WireFormat, ...] = (OpenAIChat(), AnthropicMessages(), RawCompletion())
DEFAULT = WIRES[0]


def wire_for(method: str, path: str) -> WireFormat | None:
    """The format of a model call to `path`, or None if it is not a model call."""
    return next((wire for wire in WIRES if wire.matches(method, path)), None)


def get_wire(name: str | None) -> WireFormat:
    """A format by name. Steps recorded before formats existed are OpenAI chat."""
    return next((wire for wire in WIRES if wire.name == name), DEFAULT)


__all__ = ["DEFAULT", "WIRES", "Assembler", "WireFormat", "get_wire", "wire_for"]
