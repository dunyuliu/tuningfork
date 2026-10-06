"""The design-md skill's own bar, on the DESIGN.md a run left in its project folder.

It fails a file without the YAML front matter the skill's anatomy opens with, or without a
`name` and `colors`; a color that is neither a hex value nor a reference to one; a token
reference that does not resolve; a section twice, or out of the canonical order; and a WCAG
contrast failure, which the skill says to fix before returning: any component whose text
and background colors fall below 4.5:1, WCAG AA for normal text. What the skill's linter
(`npx @google/design.md lint`) only warns about otherwise, such as a component property
outside its list, passes.

Run in the run's workspace; needs PyYAML.
"""

import re
import sys
from pathlib import Path

import yaml

#: The canonical sections, in order, with the aliases the skill names.
CANON = [
    ("overview", "brand & style"),
    ("colors",),
    ("typography",),
    ("layout", "layout & spacing"),
    ("elevation & depth", "elevation"),
    ("shapes",),
    ("components",),
    ("do's and don'ts", "dos and don'ts"),
]
HEX = re.compile(r"#([0-9A-Fa-f]{3}|[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})")


def fail(why: str) -> None:
    print(why)
    sys.exit(1)


path = Path("DESIGN.md")
if not path.exists():
    fail("no DESIGN.md in the project folder")
text = path.read_text(encoding="utf-8")
match = re.match(r"---\n(.*?)\n---\n(.*)", text, flags=re.S)
if not match:
    fail("no YAML front matter between --- lines at the top")
try:
    tokens = yaml.safe_load(match.group(1))
except yaml.YAMLError as error:
    fail(f"the front matter is not YAML: {error}")
if not isinstance(tokens, dict):
    fail("the front matter is not a mapping")
if not tokens.get("name"):
    fail("no name")
colors = tokens.get("colors")
if not isinstance(colors, dict) or not colors:
    fail("no colors")


def lookup(ref: str):
    node = tokens
    for part in ref.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


for ref in re.findall(r"\{([^{}]+)\}", match.group(1)):
    if lookup(ref.strip()) is None:
        fail(f"{{{ref}}} does not resolve by dotted path")


def color_of(value):
    """A hex color, following references to one; None for anything else."""
    for _ in range(5):
        if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
            value = lookup(value[1:-1].strip())
    return value if isinstance(value, str) and HEX.fullmatch(value) else None


for key, value in colors.items():
    if not color_of(value):
        fail(f"colors.{key} is neither a hex color nor a reference to one: {value!r}")


def luminance(hex_color: str) -> float:
    digits = hex_color.lstrip("#")
    digits = "".join(c * 2 for c in digits) if len(digits) == 3 else digits[:6]
    channels = [int(digits[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


for name, props in (tokens.get("components") or {}).items():
    if not isinstance(props, dict):
        continue
    text_color, background = color_of(props.get("textColor")), color_of(props.get("backgroundColor"))
    if text_color and background:
        light, dark = sorted([luminance(text_color), luminance(background)], reverse=True)
        ratio = (light + 0.05) / (dark + 0.05)
        if ratio < 4.5:
            fail(f"component {name}: {text_color} on {background} is {ratio:.2f}:1, below WCAG AA's 4.5:1")
seen, order = set(), []
for heading in re.findall(r"^## (.+?)\s*$", match.group(2), flags=re.M):
    key = heading.strip().lower()
    if key in seen:
        fail(f"## {heading} appears twice")
    seen.add(key)
    for index, names in enumerate(CANON):
        if key in names:
            order.append(index)
if order != sorted(order):
    fail("the canonical sections are out of order")
print("ok")
