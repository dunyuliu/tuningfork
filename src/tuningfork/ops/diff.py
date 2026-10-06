"""A skill's text, edited and compared line by line.

A proposed change to a skill is a few exact replacements — `{old, new}`, each `old` found
exactly once in the text — rather than line numbers, which drift as soon as anything above
them changes, and which a model miscounts. What changed is then shown as a line diff: the
same JSON for the CLI, the web UI and an agent.
"""

from __future__ import annotations

import difflib
from typing import Any, Sequence

from tuningfork.ops.common import OperationError


def line_diff(before: str, after: str, *, context: int = 2) -> dict[str, Any]:
    """What changed from `before` to `after`, line by line.

    Each hunk lists its lines with `op` " " (unchanged), "-" (removed) or "+" (added), and
    their numbers in the old and new text. `changed` is the larger of the lines added and
    removed: a line reworded counts once.
    """
    old, new = before.splitlines(), after.splitlines()
    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    hunks = []
    for group in matcher.get_grouped_opcodes(context):
        lines: list[dict[str, Any]] = []
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                lines.extend(
                    {"op": " ", "old": i1 + k + 1, "new": j1 + k + 1, "text": old[i1 + k]} for k in range(i2 - i1)
                )
                continue
            lines.extend({"op": "-", "old": k + 1, "text": old[k]} for k in range(i1, i2))
            lines.extend({"op": "+", "new": k + 1, "text": new[k]} for k in range(j1, j2))
        hunks.append({"old_start": group[0][1] + 1, "new_start": group[0][3] + 1, "lines": lines})
    opcodes = matcher.get_opcodes()
    removed = sum(i2 - i1 for tag, i1, i2, _, _ in opcodes if tag in ("replace", "delete"))
    added = sum(j2 - j1 for tag, _, _, j1, j2 in opcodes if tag in ("replace", "insert"))
    return {"hunks": hunks, "added": added, "removed": removed, "changed": max(added, removed)}


def apply_edits(text: str, edits: Sequence[Any]) -> str:
    """`text` with each edit's `old` replaced by its `new`, in order.

    Each `old` must occur exactly once in the text as it stands by then — the way an editor
    that replaces by text, not by line number, refuses an ambiguous edit. A text written
    with Windows line endings takes edits written with plain ones.
    """
    if not isinstance(edits, (list, tuple)) or not edits:
        raise OperationError("give at least one edit: {old, new}", code="bad_params", field="edits")
    crlf = "\r\n" in text
    for number, edit in enumerate(edits, 1):
        if (
            not isinstance(edit, dict)
            or set(edit) != {"old", "new"}
            or not isinstance(edit["old"], str)
            or not isinstance(edit["new"], str)
        ):
            raise OperationError(
                f"edit {number} must be {{old, new}}, both text", code="bad_params", field="edits"
            )
        old, new = edit["old"], edit["new"]
        if crlf:
            old = old.replace("\r\n", "\n").replace("\n", "\r\n")
            new = new.replace("\r\n", "\n").replace("\n", "\r\n")
        if not old.strip():
            raise OperationError(
                f"edit {number} names no text to replace: `old` is the exact text it changes",
                code="bad_params", field="edits",
            )
        if old == new:
            raise OperationError(f"edit {number} changes nothing", code="bad_params", field="edits")
        found = text.count(old)
        if found == 0:
            raise OperationError(
                f"edit {number}: its old text is not in the skill",
                hint=_closest(text, old),
                code="bad_params", field="edits",
            )
        if found > 1:
            raise OperationError(
                f"edit {number}: its old text occurs {found} times",
                hint="include more of the text around it, so that it occurs once",
                code="bad_params", field="edits",
            )
        text = text.replace(old, new, 1)
    return text


def _closest(text: str, old: str) -> str:
    wanted = next((line for line in old.splitlines() if line.strip()), old).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    close = difflib.get_close_matches(wanted, lines, n=2, cutoff=0.5)
    if not close:
        return "copy `old` from the skill exactly, whitespace and punctuation included"
    return "the closest lines in the skill: " + " | ".join(repr(line) for line in close)
