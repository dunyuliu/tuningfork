"""Model output formats: how a model writes its answer as raw text, and how to read it back.

A chat server parses the text a model generates into reasoning, content and tool calls, and
its clients only ever see the parsed message. Token-level work sees the raw text: a recorded
answer is scored as the model's own tokens, and a continuation forced from the middle of an
answer comes back unparsed. So tuningfork reads raw text the way the server would, and writes
a parsed answer back out as the text the model generated.

A format is model knowledge, not server knowledge — Gemma 4 writes the same text whichever
server runs it — and is chosen by the chat template the server reports, by the same rule the
server uses. Supporting another model means adding one here; nothing else changes.
"""

from __future__ import annotations

from tuningfork.backends.formats.base import Format, FormatError
from tuningfork.backends.formats.gemma4 import Gemma4

FORMATS: tuple[Format, ...] = (Gemma4(),)


def format_for(template: str | None) -> Format | None:
    """The format whose chat template this is, or None if none is known."""
    if not template:
        return None
    return next((found for found in FORMATS if found.detect(template)), None)


def get_format(name: str | None) -> Format | None:
    return next((found for found in FORMATS if found.name == name), None)


__all__ = ["FORMATS", "Format", "FormatError", "format_for", "get_format"]
