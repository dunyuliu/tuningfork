"""Paths as a person types them, read the way tuningfork will open them.

tuningfork often runs in WSL while the person types into a browser or a shell on Windows.
A path copied from Explorer — `"C:\\Users\\me\\skills\\tally"`, quotes and all — names
nothing inside WSL as it stands; there it is `/mnt/c/Users/me/skills/tally`. Every file or
folder a person names (a skill folder, a spec to import, a variant) goes through
`resolve_path`, which also says what it did — translated from what, relative to what — so a
page can show the path that will actually be used rather than leave a person guessing.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: `C:\…` or `C:/…`: a path on a Windows drive.
_DRIVE = re.compile(r"^([A-Za-z]):[\\/]")
#: `\\wsl.localhost\<distro>\…` or `\\wsl$\<distro>\…`: a WSL distro's files seen from Windows.
_WSL_SHARE = re.compile(r"^[\\/]{2}(?:wsl\.localhost|wsl\$)[\\/]([^\\/]+)(.*)$", re.IGNORECASE)


def under_wsl() -> bool:
    """Whether this process runs inside WSL, where Windows drives are mounted as folders."""
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        return "microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower()
    except OSError:
        return False


def drive_root() -> str:
    """Where WSL mounts Windows drives: `/mnt/`, unless `/etc/wsl.conf` moves it."""
    try:
        text = Path("/etc/wsl.conf").read_text()
    except OSError:
        return "/mnt/"
    section = ""
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().lower()
        elif section == "automount" and "=" in line:
            key, value = (part.strip() for part in line.split("=", 1))
            if key.lower() == "root" and value:
                value = value.strip("\"'")
                return value if value.endswith("/") else value + "/"
    return "/mnt/"


def host_path(
    text: str,
    *,
    wsl: bool | None = None,
    distro: str | None = None,
    mounts: str | None = None,
) -> str:
    """`text` as a path this machine can open.

    Surrounding quotes and spaces go (Explorer's "Copy as path" adds quotes) and a leading
    `~` becomes the home folder. Under WSL, a Windows drive path becomes the folder WSL
    mounts it at, a `\\\\wsl.localhost\\<this distro>\\…` path becomes the path inside the
    distro, and backslashes become slashes. Anything else is returned as it came.
    """
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    if wsl is None:
        wsl = under_wsl()
    if wsl:
        drive = _DRIVE.match(text)
        if drive:
            root = mounts if mounts is not None else drive_root()
            rest = text[drive.end():].replace("\\", "/").strip("/")
            return f"{root}{drive.group(1).lower()}" + (f"/{rest}" if rest else "")
        share = _WSL_SHARE.match(text)
        this = distro if distro is not None else os.environ.get("WSL_DISTRO_NAME", "")
        if share and this and share.group(1).lower() == this.lower():
            return "/" + share.group(2).replace("\\", "/").strip("/")
        if "\\" in text and not share:
            text = text.replace("\\", "/")
    return os.path.expanduser(text)


def windows_form(
    path: Path | str,
    *,
    wsl: bool | None = None,
    distro: str | None = None,
    mounts: str | None = None,
) -> str | None:
    """`path` as Windows names it, for a person about to open it there: a drive path for a
    folder WSL mounts (`/mnt/c/Users/me` is `C:\\Users\\me`), else the distro's share
    (`\\\\wsl.localhost\\<distro>\\home\\me`). None outside WSL, where a path is already its
    own name. The reverse of `host_path`."""
    if wsl is None:
        wsl = under_wsl()
    text = str(path)
    if not wsl or not text.startswith("/"):
        return None
    root = mounts if mounts is not None else drive_root()
    if text.startswith(root):
        drive, _, rest = text[len(root):].partition("/")
        if len(drive) == 1 and drive.isalpha():
            return f"{drive.upper()}:\\" + rest.replace("/", "\\")
    this = distro if distro is not None else os.environ.get("WSL_DISTRO_NAME", "")
    return f"\\\\wsl.localhost\\{this}" + text.replace("/", "\\") if this else None


def windows_path(text: str) -> bool:
    """Whether `text` names a path on a Windows drive, or a Windows share."""
    text = text.strip().strip("\"'")
    return bool(_DRIVE.match(text) or text.startswith("\\\\"))


@dataclass(frozen=True, slots=True)
class ResolvedPath:
    """A path as a person gave it, and as tuningfork will open it."""

    given: str
    path: Path
    #: What a relative path was resolved against; None for an absolute one.
    base: Path | None
    #: The text as given, when it had to be translated to be opened here.
    translated_from: str | None

    @property
    def exists(self) -> bool:
        return self.path.exists()

    def to_json(self) -> dict[str, Any]:
        return {
            "given": self.given,
            "path": str(self.path),
            "relative_to": str(self.base) if self.base is not None else None,
            "translated_from": self.translated_from,
            "exists": self.exists,
        }


def resolve_path(text: str, base: Path | str | None = None, **options: Any) -> ResolvedPath:
    """`text` as an absolute path: translated by `host_path`, then, when relative, resolved
    against `base` (the current folder when none is given — say which, to whoever reads it)."""
    given = text
    translated = host_path(text, **options)
    stripped = text.strip().strip("\"'").strip()
    path = Path(translated)
    anchor = None
    if not path.is_absolute():
        anchor = Path(base) if base is not None else Path.cwd()
        path = anchor / path
    return ResolvedPath(
        given=given,
        path=path.resolve(),
        base=anchor,
        translated_from=given if translated not in (stripped, os.path.expanduser(stripped)) else None,
    )
