"""The Hermes design-md demo (`examples/hermes-design-md/`): Hermes Agent's own design-md skill,
a task it is refined on and one held out, and the check beside the spec that judges them —
found by `$TUNINGFORK_SPEC_DIR`, out of the agent's sight. Where PyYAML is installed, the check
passes the skill's own starter template and fails each way a model was seen to get it wrong."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tuningfork.specs import load_spec

DEMO = Path(__file__).parents[1] / "examples" / "hermes-design-md"
STARTER = (DEMO / "design-md" / "templates" / "starter.md").read_text(encoding="utf-8")


def test_the_spec_names_a_task_to_refine_on_and_one_held_out():
    spec = load_spec(DEMO / "evals.json")
    assert spec.skill_name == "design-md" and spec.resolved_skill_dir() == (DEMO / "design-md").resolve()
    assert [(item.id, item.tuningfork.held_out) for item in spec.evals] == [("1", False), ("2", True)]
    for item in spec.evals:
        assert '"$TUNINGFORK_SPEC_DIR/checks/design.py"' in item.tuningfork.check
    assert (DEMO / "checks" / "design.py").is_file()
    # The skill as Hermes ships it, its licence beside it; the check out of what every run is given.
    assert sorted(p.relative_to(DEMO / "design-md").as_posix() for p in (DEMO / "design-md").rglob("*") if p.is_file()) == [
        "SKILL.md", "templates/starter.md"]
    assert "MIT License" in (DEMO / "LICENSE-hermes").read_text(encoding="utf-8")


@pytest.mark.parametrize("fix, lines", [
    ("SKILL.md", 2),  # two lines appended to the write step
    ("SKILL-section.md", 5),  # a section of its own, before the file's anatomy
])
def test_each_fix_only_adds_the_lint(fix, lines):
    original = (DEMO / "design-md" / "SKILL.md").read_text(encoding="utf-8").splitlines()
    fixed = (DEMO / "fix" / fix).read_text(encoding="utf-8").splitlines()
    added = [line for line in fixed if line not in original]
    assert len(fixed) == len(original) + lines
    assert [line for line in original if line not in fixed] == []
    assert any("npx -y @google/design.md lint DESIGN.md" in line for line in added)


def judge(tmp_path: Path, design: str) -> subprocess.CompletedProcess:
    (tmp_path / "DESIGN.md").write_text(design, encoding="utf-8")
    return subprocess.run([sys.executable, str(DEMO / "checks" / "design.py")], cwd=tmp_path,
                          capture_output=True, text=True)


@pytest.mark.parametrize("change, passes, says", [
    (lambda text: text, True, "ok"),
    # What the original skill's runs left: white text on a light teal button.
    (lambda text: text.replace('tertiary: "#2563EB"', 'tertiary: "#20B2AA"'), False, "2.62:1"),
    (lambda text: text.replace('"{colors.tertiary}"', '"{colors.teal}"', 1), False, "does not resolve"),
    (lambda text: text.replace('primary: "#0F172A"', "primary: navy", 1), False, "neither a hex color"),
    (lambda text: text.split("---\n", 2)[2], False, "no YAML front matter"),
    (lambda text: text + "\n## Colors\n\nAgain.\n", False, "appears twice"),
    (lambda text: text.replace("## Overview", "## Overview-moved", 1) + "\n## Overview\n\nLast.\n", False, "out of order"),
    # A color by reference is a color: the spec allows references to primitives.
    (lambda text: text.replace('neutral: "#F8FAFC"', 'neutral: "#F8FAFC"\n  surface: "{colors.neutral}"', 1), True, "ok"),
    # What the linter only warns about passes: a property outside the component list.
    (lambda text: text.replace("    padding: 24px\n", "    padding: 24px\n    textDecoration: underline\n", 1), True, "ok"),
])
def test_the_check(tmp_path, change, passes, says):
    pytest.importorskip("yaml")
    done = judge(tmp_path, change(STARTER))
    assert (done.returncode == 0) is passes, done.stdout
    assert says in done.stdout


def test_the_tasks_ask_for_the_same_file_from_two_brands():
    tasks = json.loads((DEMO / "evals.json").read_text(encoding="utf-8"))["evals"]
    assert [t["prompt"].split(":")[0].rsplit(" ", 1)[1] for t in tasks] == ["Tidewater", "Skyline"]
    assert tasks[0]["tuningfork"]["check"] == tasks[1]["tuningfork"]["check"]
