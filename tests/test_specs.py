"""Task specs as files: what tuningfork writes back is what it read, and a spec with a
problem is refused — every problem named — before anything runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tuningfork.runner import eval_record
from tuningfork.specs import Eval, SpecError, TaskSpec, TuningforkOptions, load_spec, validate_spec

EXAMPLE = Path(__file__).parents[1] / "examples" / "tally-report.evals.json"


def spec(**eval_fields) -> dict:
    item = {"id": "1", "prompt": "Summarize data.csv."}
    item.update(eval_fields)
    return {"skill_name": "tally", "evals": [item]}


def where(data) -> list[str]:
    return [problem.where for problem in validate_spec(data)]


class TestRoundTrip:
    def test_the_example_spec_is_written_back_as_it_was_read(self):
        data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        assert validate_spec(data) == []
        again = TaskSpec.from_dict(data).to_dict()
        assert TaskSpec.from_dict(again).to_dict() == again
        assert again["evals"][0]["prompt"] == data["evals"][0]["prompt"]
        assert again["tuningfork"] == data["tuningfork"]

    def test_keys_this_version_does_not_know_are_kept(self):
        """A spec shared with other tools (the evals.json convention) carries their keys; a
        spec saved from the browser must not drop them."""
        data = {
            "skill_name": "tally",
            "owner": "someone else's tool",
            "tuningfork": {"skill_dir": "skill", "future_option": 3},
            "evals": [
                {
                    "id": 7,
                    "prompt": "Do it.",
                    "grader": {"kind": "llm"},
                    "tuningfork": {"check": "true", "retries": 2},
                }
            ],
        }
        written = TaskSpec.from_dict(data).to_dict()
        assert written["owner"] == "someone else's tool"
        assert written["tuningfork"] == {"skill_dir": "skill", "future_option": 3}
        item = written["evals"][0]
        assert item["id"] == "7" and item["grader"] == {"kind": "llm"}
        assert item["tuningfork"]["retries"] == 2 and item["tuningfork"]["check"] == "true"

    def test_a_launch_record_reads_back_as_the_eval_it_was(self, tmp_path):
        item = Eval(
            id="1", prompt="p", files=["data.csv", {"path": "in/b.bin", "source": "inputs/b.bin"}],
            tuningfork=TuningforkOptions(check="true", normalizers=["tool_name"], extra={"x": 1}),
        )
        record = json.loads(json.dumps(eval_record(item, tmp_path)))
        again = Eval.from_dict(record)
        assert again.tuningfork == item.tuningfork
        assert again.files == item.files

    def test_a_launch_record_from_before_the_serializer_still_loads(self):
        """Launches stored their eval with `dataclasses.asdict`: every option, `check`
        included when it was None."""
        record = {
            "id": "1", "prompt": "p", "expected_output": "", "files": [], "expectations": [],
            "tuningfork": {
                "check": None, "normalizers": [], "held_out": False, "skills": ["s"],
                "timeout": 900.0, "max_turns": 12,
            },
            "spec_dir": "/somewhere",
        }
        item = Eval.from_dict(record)
        assert item.tuningfork.check is None and item.tuningfork.skills == ["s"]
        assert item.tuningfork.extra == {}


class TestValidation:
    def test_a_good_spec_has_no_problems(self):
        assert validate_spec(spec(files=["a.csv", {"path": "b/c.txt", "content": "x"}])) == []

    @pytest.mark.parametrize(
        ("data", "expected"),
        [
            ([], [""]),
            ({"skill_name": "t"}, ["evals"]),
            ({"skill_name": "t", "evals": []}, ["evals"]),
            ({"skill_name": 3, "evals": [{"id": "1", "prompt": "p"}]}, ["skill_name"]),
            ({"tuningfork": [], "evals": [{"id": "1", "prompt": "p"}]}, ["tuningfork"]),
            ({"tuningfork": {"skill_dir": 1}, "evals": [{"id": "1", "prompt": "p"}]}, ["tuningfork.skill_dir"]),
            ({"evals": ["not an eval"]}, ["evals[0]"]),
        ],
    )
    def test_the_spec_as_a_whole(self, data, expected):
        assert where(data) == expected

    def test_ids_are_required_and_unique(self):
        data = {"evals": [
            {"id": "1", "prompt": "p"},
            {"id": 1, "prompt": "p"},
            {"prompt": "p"},
            {"id": True, "prompt": "p"},
        ]}
        problems = validate_spec(data)
        assert [p.where for p in problems] == ["evals[1].id", "evals[2].id", "evals[3].id"]
        assert "used by another eval" in problems[0].message

    @pytest.mark.parametrize(
        ("fields", "expected"),
        [
            ({"prompt": "  "}, "evals[0].prompt"),
            ({"expected_output": 3}, "evals[0].expected_output"),
            ({"expectations": "one"}, "evals[0].expectations"),
            ({"files": "a.csv"}, "evals[0].files"),
            ({"tuningfork": "fast"}, "evals[0].tuningfork"),
            ({"tuningfork": {"check": ["python3", "c.py"]}}, "evals[0].tuningfork.check"),
            # Read as true before, since "false" is a non-empty string.
            ({"tuningfork": {"held_out": "false"}}, "evals[0].tuningfork.held_out"),
            ({"tuningfork": {"skills": "tally"}}, "evals[0].tuningfork.skills"),
            ({"tuningfork": {"timeout": 0}}, "evals[0].tuningfork.timeout"),
            ({"tuningfork": {"timeout": "60"}}, "evals[0].tuningfork.timeout"),
            ({"tuningfork": {"max_turns": 0}}, "evals[0].tuningfork.max_turns"),
            ({"tuningfork": {"max_turns": 2.5}}, "evals[0].tuningfork.max_turns"),
            ({"tuningfork": {"max_turns": True}}, "evals[0].tuningfork.max_turns"),
        ],
    )
    def test_each_field_of_an_eval(self, fields, expected):
        assert where(spec(**fields)) == [expected]

    @pytest.mark.parametrize(
        "check",
        [{"reply_contains": "61.5"}, {"reply_matches": r"(?i)total\D+61\.5"}, "true"],
    )
    def test_a_check_can_judge_the_reply(self, check):
        assert validate_spec(spec(tuningfork={"check": check})) == []

    @pytest.mark.parametrize(
        ("check", "expected"),
        [
            ({"reply_contains": ""}, "evals[0].tuningfork.check.reply_contains"),
            ({"reply_contains": 61.5}, "evals[0].tuningfork.check.reply_contains"),
            ({"reply_matches": "(unclosed"}, "evals[0].tuningfork.check.reply_matches"),
            ({"reply_equals": "x"}, "evals[0].tuningfork.check"),
            ({"reply_contains": "a", "reply_matches": "b"}, "evals[0].tuningfork.check"),
        ],
    )
    def test_a_reply_check_that_cannot_judge_is_named(self, check, expected):
        assert where(spec(tuningfork={"check": check})) == [expected]

    def test_checks_read_as_a_person_would(self):
        from tuningfork.specs import check_kind, describe_check

        assert (check_kind("true"), describe_check("true")) == ("shell", "true")
        assert describe_check({"reply_contains": "61.5"}) == 'the reply contains "61.5"'
        assert describe_check({"reply_matches": "a+"}) == "the reply matches /a+/"
        assert (check_kind(None), check_kind("  "), check_kind({"x": "y"})) == (None, None, None)

    def test_a_reply_pattern_crosses_lines(self, tmp_path):
        """Found on a real check: (?=.*212.00)(?=.*process-receipt) failed on a reply that
        held both, because they were on different lines."""
        from tuningfork.specs import run_check

        reply = "The total is 212.00.\n\nFiled with the process-receipt skill."
        both = run_check({"reply_matches": r"(?=.*212\.00)(?=.*process-receipt)"}, tmp_path, reply=reply)
        missing = run_check({"reply_matches": r"(?=.*212\.00)(?=.*archive)"}, tmp_path, reply=reply)
        assert both.passed and not missing.passed

    def test_an_unknown_normalizer_is_named_with_the_known_ones(self):
        (problem,) = validate_spec(spec(tuningfork={"normalizers": ["tool_name", "vibes"]}))
        assert problem.where == "evals[0].tuningfork.normalizers"
        assert "vibes" in problem.message and "tool_call_json" in problem.message

    @pytest.mark.parametrize(
        "entry",
        [
            "..",
            {"path": "/etc/passwd", "content": "x"},
            {"path": "../outside.txt", "content": "x"},
            {"path": "in/../../outside.txt", "content": "x"},
            {"path": "C:/Windows/x.txt", "content": "x"},
            {"path": "", "content": "x"},
            {"path": "a.bin", "content": b"x".hex(), "source": ""},
            {"path": "a.txt", "content": 3},
            3,
        ],
    )
    def test_a_file_that_would_land_outside_the_workspace_or_is_malformed(self, entry):
        assert where(spec(files=[entry])) == ["evals[0].files[0]"]


class TestLoading:
    def test_a_spec_with_problems_is_refused_with_each_named(self, tmp_path):
        path = tmp_path / "evals.json"
        path.write_text(json.dumps(spec(tuningfork={"held_out": "no", "max_turns": 0})))
        with pytest.raises(SpecError) as refused:
            load_spec(path)
        assert [p.where for p in refused.value.problems] == [
            "evals[0].tuningfork.held_out", "evals[0].tuningfork.max_turns",
        ]
        assert "held_out" in str(refused.value) and "max_turns" in str(refused.value)

    def test_not_json_at_all(self, tmp_path):
        path = tmp_path / "evals.json"
        path.write_text("{ not json")
        with pytest.raises(SpecError, match="not valid JSON"):
            load_spec(path)

    def test_leniency_is_still_there_when_asked_for(self, tmp_path):
        path = tmp_path / "evals.json"
        path.write_text(json.dumps(spec(tuningfork={"max_turns": 0})))
        assert load_spec(path, strict=False).evals[0].tuningfork.max_turns == 0
