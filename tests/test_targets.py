"""A target described in conditions, and what a target would count before it is declared.

Found in use: writing the target as a regular expression was hard to get right. A pattern meets
a label truncated at 90 characters or a JSON-escaped key, so "does not mention" cannot be one,
and "mentions A and B" needs lookaheads that stopped at a line break.
"""

from __future__ import annotations

import json

import pytest

from tuningfork.analysis.distribution import check_rules, target_by_rules, target_from_spec
from tuningfork.analysis.normalize import outcome_for
from tuningfork.ops.common import OperationError
from tuningfork.ops.targets import offered_tools, seen_behaviours
from tests.helpers import as_response, record_run, text_message, tool_call_message


def calling(name: str, **arguments) -> object:
    return outcome_for(as_response(tool_call_message(name, arguments), "tool_calls"))


def replying(text: str) -> object:
    return outcome_for(as_response(text_message(text)))


class TestDescribed:
    def test_a_tool_called_or_not(self):
        search = calling("search_files", pattern="data.csv")
        read = calling("read_file", path="data.csv")
        calls = target_by_rules([{"if": "calls", "value": "search_files"}])
        avoids = target_by_rules([{"if": "not_calls", "value": "search_files"}])
        assert (calls.matches(search), calls.matches(read)) == (True, False)
        assert (avoids.matches(search), avoids.matches(read), avoids.matches(replying("done"))) == (False, True, True)
        assert calls.description == "a behaviour that calls search_files"

    def test_a_phrase_in_written_json_however_it_was_escaped(self):
        """The key holds the file's JSON escaped inside JSON: `\\"row_count\\"`. A phrase is
        looked for in the decoded values, so it is found as a person writes it."""
        wrote = calling("write_file", path="report.json", content='{\n  "row_count": 4,\n  "amount_total": 61.5\n}')
        assert '\\"row_count\\"' in wrote.key
        assert target_by_rules([{"if": "contains", "value": '"row_count":4'}]).matches(wrote)
        assert target_by_rules([{"if": "contains", "value": "ROW_COUNT"}]).matches(wrote)  # case ignored
        assert not target_by_rules([{"if": "not_contains", "value": "amount_total"}]).matches(wrote)

    def test_every_condition_must_hold_whatever_the_line_breaks(self):
        """The real check that failed: "212.00 and process-receipt", on different lines."""
        reply = replying("The total is 212.00.\n\nI filed it with the process-receipt skill.")
        both = target_by_rules([{"if": "contains", "value": "212.00"}, {"if": "contains", "value": "process-receipt"}])
        not_archived = target_by_rules([{"if": "contains", "value": "212.00"}, {"if": "not_contains", "value": "archive"}])
        assert both.matches(reply) and not_archived.matches(reply)
        assert not target_by_rules([{"if": "contains", "value": "212.00"}, {"if": "calls", "value": "x"}]).matches(reply)
        assert both.description == 'a behaviour that mentions "212.00", and mentions "process-receipt"'

    def test_under_the_tool_name_lens(self):
        named = outcome_for(as_response(tool_call_message("read_file", {"path": "a"}), "tool_calls"), ["tool_name"])
        assert target_by_rules([{"if": "calls", "value": "read_file"}]).matches(named)

    def test_stored_and_read_back_as_declared(self):
        target = target_by_rules([{"if": "calls", "value": "read_file"}, {"if": "contains", "value": " data.csv "}])
        again = target_from_spec(json.loads(json.dumps(target.spec)))
        assert again.description == target.description
        assert target.spec["rules"][1]["value"] == "data.csv"


class TestCode:
    """A script is one behaviour whatever its comments say: its key keeps a digest of its
    syntax tree. A target reads the code itself. Found when Hermes wrote its report from a
    script at most steps, and neither an A/B nor a screen could see the keys it wrote — the
    same edit went 0/20 → 18/20 run to the end."""

    SCRIPT = "import json\njson.dump({'row_count': 4, 'amount_total': 61.5}, open('report.json', 'w'))\n"

    def test_a_target_reads_what_a_script_writes(self):
        script = calling("execute_code", code=self.SCRIPT)
        assert "<ast:" in script.key and "row_count" not in script.key
        from tuningfork.analysis.distribution import target_by_pattern

        assert target_by_pattern("row_count").matches(script)
        assert target_by_rules([{"if": "contains", "value": "row_count"}, {"if": "contains", "value": "amount_total"}]).matches(script)
        assert not target_by_rules([{"if": "not_contains", "value": "row_count"}]).matches(script)
        assert not target_by_pattern("row_count").matches(calling("execute_code", code="print(4)"))

    def test_two_scripts_differing_in_a_comment_are_still_one_behaviour(self):
        first = calling("execute_code", code=self.SCRIPT)
        second = calling("execute_code", code="# the report\n" + self.SCRIPT)
        assert first == second and first.text != second.text

    def test_a_target_stored_before_matches_as_it_was_declared(self):
        """Its meaning was fixed before sampling: one stored without `code` reads labels and keys."""
        script = calling("execute_code", code=self.SCRIPT)
        pattern = {"kind": "pattern", "value": "row_count", "description": "behaviour matching /row_count/"}
        rules = {"kind": "rules", "rules": [{"if": "contains", "value": "row_count"}], "description": "…"}
        assert not target_from_spec(pattern).matches(script) and not target_from_spec(rules).matches(script)
        assert target_from_spec({**pattern, "code": True}).matches(script)
        from tuningfork.analysis.distribution import target_by_pattern

        assert target_by_pattern("x").spec["code"] is True and target_by_rules([{"if": "calls", "value": "x"}]).spec["code"] is True

    @pytest.mark.parametrize(
        ("rules", "said"),
        [
            ([], "at least one condition"),
            ([{"if": "loves", "value": "x"}], "is not one of"),
            ([{"if": "calls", "value": "  "}], "needs a value"),
            ("calls x", "at least one condition"),
        ],
    )
    def test_a_condition_that_cannot_be_judged_is_named(self, rules, said):
        with pytest.raises(ValueError, match=said):
            check_rules(rules)


class TestSeen:
    def test_what_a_target_would_count_at_a_step(self, store):
        ran = [
            record_run(store, f"r{index}", [tool_call_message(name, {"pattern" if name == "search_files" else "path": "data.csv"})], task_id="t/1")
            for index, name in enumerate(["search_files", "read_file", "read_file"])
        ]

        seen = seen_behaviours(store, ran[0], "0", target_rules=[{"if": "calls", "value": "search_files"}]).to_json()

        by_label = {entry["label"].split("(")[0]: entry for entry in seen["behaviours"]}
        assert by_label["search_files"]["matches"] is True and by_label["search_files"]["seen"] == {"this run": 1}
        assert by_label["read_file"]["matches"] is False and by_label["read_file"]["seen"] == {"other runs": 2}
        assert seen["target"] == "a behaviour that calls search_files"
        # the run's own behaviour, for a probe's default target
        own = seen_behaviours(store, ran[1], "0", base=True).to_json()
        assert {entry["label"].split("(")[0]: entry["matches"] for entry in own["behaviours"]} == {"read_file": True, "search_files": False}

    def test_a_bad_description_or_pattern_names_its_field(self, store):
        run = record_run(store, "r", [text_message("x")], task_id="t/1")
        with pytest.raises(OperationError) as described:
            seen_behaviours(store, run, "0", target_rules=[{"if": "calls"}])
        with pytest.raises(OperationError) as matched:
            seen_behaviours(store, run, "0", target_match="(unclosed")
        assert (described.value.field, matched.value.field) == ("target_rules", "target_match")

    def test_the_tools_a_step_offered(self):
        assert offered_tools({"tools": [{"type": "function", "function": {"name": "read_file"}}, {"name": "Write"}]}) == ["read_file", "Write"]
