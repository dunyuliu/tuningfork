"""Edits to a skill's text, and the line diff that shows them (`ops.diff`)."""

from __future__ import annotations

import pytest

from tuningfork.ops.common import OperationError
from tuningfork.ops.diff import apply_edits, line_diff

SKILL = "---\nname: tally-report\n---\n\n# Tally report\n\n## Steps\n\n1. Read the CSV file.\n2. Work out the totals.\n3. Write report.json.\n"


class TestEdits:
    def test_each_old_text_is_replaced_once_in_order(self):
        edited = apply_edits(SKILL, [
            {"old": "3. Write report.json.", "new": "3. Write report.json with the keys row_count and amount_total."},
            {"old": "1. Read the CSV file.", "new": "1. Read the CSV file with read_file."},
        ])
        assert "row_count and amount_total" in edited and "with read_file" in edited
        assert edited.count("\n") == SKILL.count("\n")

    def test_an_old_text_that_is_not_there_names_the_closest_lines(self):
        with pytest.raises(OperationError) as refused:
            apply_edits(SKILL, [{"old": "3. Write the report.json.", "new": "x"}])
        assert refused.value.field == "edits" and "not in the skill" in refused.value.message
        assert "3. Write report.json." in refused.value.hint

    def test_an_ambiguous_old_text_is_refused(self):
        with pytest.raises(OperationError, match="occurs 2 times"):
            apply_edits("a\nb\na\n", [{"old": "a", "new": "c"}])

    @pytest.mark.parametrize("edit", [
        {"old": "", "new": "x"}, {"old": "  ", "new": "x"}, {"old": "1. Read the CSV file.", "new": "1. Read the CSV file."},
        {"old": "x"}, {"old": "x", "new": "y", "why": "z"}, "replace it", {"old": 1, "new": "y"},
    ])
    def test_a_malformed_edit_is_refused_by_field(self, edit):
        with pytest.raises(OperationError) as refused:
            apply_edits(SKILL, [edit])
        assert refused.value.field == "edits"

    def test_no_edits_is_refused(self):
        with pytest.raises(OperationError):
            apply_edits(SKILL, [])

    def test_a_windows_file_takes_plain_line_endings(self):
        windows = SKILL.replace("\n", "\r\n")
        edited = apply_edits(windows, [{"old": "2. Work out the totals.\n3. Write", "new": "2. Count the rows.\n3. Write"}])
        assert "2. Count the rows.\r\n3. Write" in edited and "\n" not in edited.replace("\r\n", "")


class TestDiff:
    def test_a_reworded_line_counts_once_with_its_numbers(self):
        after = SKILL.replace("3. Write report.json.", "3. Write report.json with row_count.")
        diff = line_diff(SKILL, after)
        assert (diff["added"], diff["removed"], diff["changed"]) == (1, 1, 1)
        (hunk,) = diff["hunks"]
        changed = [line for line in hunk["lines"] if line["op"] != " "]
        assert changed == [
            {"op": "-", "old": 11, "text": "3. Write report.json."},
            {"op": "+", "new": 11, "text": "3. Write report.json with row_count."},
        ]
        assert all(line["op"] == " " for line in hunk["lines"][:2])  # its context

    def test_added_lines_and_far_apart_changes(self):
        after = SKILL.replace("# Tally report\n", "# Tally report\n\nUse the keys row_count and amount_total.\n")
        after = after.replace("3. Write report.json.", "3. Write report.json, nothing else.")
        diff = line_diff(SKILL, after)
        assert diff["added"] == 3 and diff["removed"] == 1 and diff["changed"] == 3
        # Four unchanged lines apart: one hunk with two lines of context, two with one.
        assert len(diff["hunks"]) == 1 and len(line_diff(SKILL, after, context=1)["hunks"]) == 2

    def test_no_change_is_no_hunk(self):
        assert line_diff(SKILL, SKILL) == {"hunks": [], "added": 0, "removed": 0, "changed": 0}
