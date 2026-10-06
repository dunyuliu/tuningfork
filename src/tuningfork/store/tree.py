"""The content-addressed message tree.

Chat-completions requests are stateless, so every request carries the model's entire
context. Each message is hashed by content, and a node is `hash(parent_node + message)`.
A request's message list is therefore a *path* through a tree, and:

- a request extending a known path continues that run;
- a request that branches off a known path is a fork;
- a request sharing no prefix starts a new run.

Two hashes are kept per message, and the distinction matters:

`exact_hash` includes everything, so a recorded request rebuilds byte-identically. It is
what path identity is built on, because a harness echoes tool-call ids back and those ids
must line up within a run.

`normal_hash` masks volatile fields — tool-call ids are a fresh number per call on the
servers measured in Phase 0. Comparing runs needs this: on exact hashes, two runs that did
exactly the same thing look completely different from the first tool call onward.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Sequence

#: Fields that change between identical behaviours and must be masked before comparing.
VOLATILE_KEYS = frozenset({"id", "tool_call_id"})

MASK = "<masked>"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def exact_hash(message: dict[str, Any]) -> str:
    """Hash of the message exactly as it appeared."""
    return _digest(_canonical(message))


def mask_volatile(value: Any) -> Any:
    """Recursively replace volatile identifiers with a constant.

    Applied to tool calls and tool results, whose ids are generated per call. Everything
    else is left alone, including message content.
    """
    if isinstance(value, dict):
        return {
            key: MASK if key in VOLATILE_KEYS and isinstance(inner, str) else mask_volatile(inner)
            for key, inner in value.items()
        }
    if isinstance(value, list):
        return [mask_volatile(item) for item in value]
    return value


def normal_hash(message: dict[str, Any]) -> str:
    """Hash of the message with volatile identifiers masked, for cross-run comparison."""
    return _digest(_canonical(mask_volatile(message)))


def node_id(parent_id: str | None, message_hash: str) -> str:
    """A node's identity is its position in the tree, not just its message.

    The same message appearing after a different history is a different node, which is what
    makes a path through the tree unambiguous.
    """
    return _digest(f"{parent_id or ''}:{message_hash}")


def node_path(messages: Sequence[dict[str, Any]]) -> list[str]:
    """Node ids for a message list, root first."""
    path: list[str] = []
    parent: str | None = None
    for message in messages:
        parent = node_id(parent, exact_hash(message))
        path.append(parent)
    return path


def common_prefix_length(left: Sequence[str], right: Sequence[str]) -> int:
    length = 0
    for a, b in zip(left, right):
        if a != b:
            break
        length += 1
    return length


def describe_message(message: dict[str, Any], limit: int = 80) -> str:
    """One-line summary for CLI output and divergence reports."""
    role = message.get("role", "?")
    calls = message.get("tool_calls") or []
    if calls:
        names = ", ".join(
            (call.get("function") or {}).get("name", "?") for call in calls
        )
        return f"{role}: tool_calls({names})"
    content = message.get("content")
    if isinstance(content, list):  # multimodal parts
        content = " ".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    content = (content or "").strip().replace("\n", " ")
    if len(content) > limit:
        content = f"{content[:limit]}…"
    return f"{role}: {content}"


def iter_messages(request: dict[str, Any]) -> Iterable[dict[str, Any]]:
    messages = request.get("messages")
    return messages if isinstance(messages, list) else ()
