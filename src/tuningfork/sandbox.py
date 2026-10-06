"""A private view of the machine for every launch, with bubblewrap.

What a harness tuningfork launches sees is part of what is measured: the model reads the
paths in its prompt. In folders of its own under the recordings folder, every launch names
a different `.../batches/<id>/<n>/workspace`, and a few digits of that path were seen to
decide whether the model wrote a file or only said it did. So where
bubblewrap can run, every launch sees what an installation would, and the same every time:
its home is `/home/user`, it works in `/home/user/project`, and its harness keeps its files
where it installs them there.

Everything else is the machine as it is, read-only: the person's own home too, since the
harness and its tools live there. What tuningfork keeps is not in it: the recordings folder,
the folder the runs are made in, and the task's spec folder — its check, and any variant
kept beside it — are empty inside. `/tmp` is the launch's own, and so are its processes: when
the launch ends, nothing it started lives on.

`TUNINGFORK_SANDBOX=off` launches without it, as where bubblewrap cannot run: each launch
then sees its own folders.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

from tuningfork.harness.base import LaunchSpec

#: What every launch's harness sees as its home, and as the folder it works in.
HOME = PurePosixPath("/home/user")
WORKSPACE = HOME / "project"

#: Every launch's own, and empty: nothing of the machine's there is seen inside.
TMP = Path("/tmp")

#: The environment variable that turns the sandbox off.
SWITCH = "TUNINGFORK_SANDBOX"

#: A probe gives up after this long: bubblewrap that hangs must not hang a batch's plan.
PROBE_TIMEOUT = 10.0


@dataclass(frozen=True, slots=True)
class Status:
    """Whether launches run in the sandbox here, and if not, why not."""

    executable: str | None
    version: str | None
    reason: str | None = None

    @property
    def available(self) -> bool:
        return self.reason is None

    def to_json(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "kind": "bubblewrap",
            "executable": self.executable,
            "version": self.version,
            "home": str(HOME),
            "workspace": str(WORKSPACE),
            "reason": self.reason,
        }


def turned_off() -> bool:
    """Whether the person turned the sandbox off: then no plan warns that it is."""
    return os.environ.get(SWITCH, "").strip().lower() in {"off", "0", "no", "false"}


def status() -> Status:
    """Whether a launch can run in the sandbox now: bubblewrap found, able to make the
    namespaces it needs, and the person's own home not where a launch's would go."""
    if turned_off():
        return Status(None, None, f"turned off ({SWITCH}=off)")
    return _probe(shutil.which("bwrap"), str(Path.home()))


@lru_cache(maxsize=8)
def _probe(executable: str | None, real_home: str) -> Status:
    if os.name != "posix" or executable is None:
        return Status(None, None, "bubblewrap is not installed (Linux: the bubblewrap package)")
    version = None
    try:
        done = subprocess.run(
            [executable, "--version"], capture_output=True, text=True, timeout=PROBE_TIMEOUT,
            stdin=subprocess.DEVNULL,
        )
        version = done.stdout.strip().removeprefix("bubblewrap").strip() or None
        tried = subprocess.run(
            [executable, *_base_arguments(Path(real_home)), "--", "true"],
            capture_output=True, text=True, timeout=PROBE_TIMEOUT, stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return Status(executable, version, f"bubblewrap could not run: {error}")
    if tried.returncode != 0:
        said = (tried.stderr or tried.stdout).strip().splitlines()
        return Status(executable, version, f"bubblewrap could not make a sandbox here: {said[0] if said else f'exit {tried.returncode}'}")
    return Status(executable, version, home_conflict(real_home))


def home_conflict(real_home: str | Path) -> str | None:
    """Why the person's own home rules the sandbox out, if it does: a launch's home is seen
    where it is, and the person's — where the harness and its tools live — would be hidden."""
    home = PurePosixPath(real_home)
    if home == HOME or HOME in home.parents or home in HOME.parents:
        return f"your home, {home}, is where a launch's home would go ({HOME})"
    return None


def _base_arguments(real_home: Path) -> list[str]:
    """The machine as it is, read-only; a fresh /dev, /proc and /tmp; every home hidden but
    the person's own, read-only; the launch's processes its own, and gone with it."""
    arguments = [
        "--die-with-parent", "--unshare-pid",
        "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
        "--tmpfs", "/home",
    ]
    if PurePosixPath("/home") in PurePosixPath(real_home).parents:
        arguments += ["--ro-bind", str(real_home), str(real_home)]
    return arguments


@dataclass(frozen=True, slots=True)
class Sandbox:
    """Launches in bubblewrap, each seeing `HOME` and `WORKSPACE`. `hide`: folders of
    tuningfork's that no launch may read, made empty inside."""

    executable: str
    version: str | None
    hide: tuple[Path, ...] = ()

    def to_json(self) -> dict[str, Any]:
        """As a launch's settings keep it: a fork of the run is sandboxed the same way."""
        return {"kind": "bubblewrap", "version": self.version, "home": str(HOME), "workspace": str(WORKSPACE)}

    def wrap(self, spec: LaunchSpec, *, home: Path, workspace: Path) -> LaunchSpec:
        """The harness's command, run in the sandbox: `home` seen as `HOME`, `workspace` as
        `WORKSPACE`, and every path to either in its arguments and environment written as
        the harness will see it."""
        if spec.argv:
            refuse_under_tmp(spec.argv[0], spec.env.get("PATH"))
        mounts = [(Path(workspace), WORKSPACE), (Path(home), HOME)]
        env = {key: translate(value, mounts) for key, value in spec.env.items()}
        real_home = Path.home()
        executable = _resolve(spec.argv[0], env.get("PATH")) if spec.argv else None
        hidden = hideable(self.hide, real_home=real_home, keep=[executable] if executable else [])
        if "PATH" in env:
            # tuningfork's own tools are not the harness's: a folder that is empty inside
            # leaves the harness's PATH.
            env["PATH"] = os.pathsep.join(
                entry for entry in env["PATH"].split(os.pathsep)
                if not any(_within(Path(entry), folder) for folder in hidden)
            )
        env.update({"HOME": str(HOME), "TMPDIR": "/tmp", "PWD": str(WORKSPACE)})
        arguments = [self.executable, *_base_arguments(real_home)]
        for folder in hidden:
            arguments += ["--tmpfs", str(folder)]
        arguments += [
            "--bind", str(Path(home).resolve()), str(HOME),
            "--bind", str(Path(workspace).resolve()), str(WORKSPACE),
            "--chdir", str(WORKSPACE), "--",
        ]
        return LaunchSpec(
            argv=(*arguments, *(translate(part, mounts) for part in spec.argv)),
            env=env,
            cwd=Path(workspace),
        )


def for_launches(hide: Iterable[Path | str | None] = ()) -> tuple[Sandbox | None, str | None]:
    """The sandbox launches run in here, or None and why not."""
    found = status()
    if not found.available or found.executable is None:
        return None, found.reason
    folders = tuple(dict.fromkeys(Path(folder).resolve() for folder in hide if folder))
    return Sandbox(found.executable, found.version, folders), None


def mount_point(home: Path) -> Path:
    """Where the workspace is mounted inside a launch's home: made before the launch, so
    nothing is created in the home behind tuningfork's back."""
    return Path(home) / WORKSPACE.relative_to(HOME)


def translate(text: str, mounts: Sequence[tuple[Path, PurePosixPath]]) -> str:
    """`text` with every path to a mounted folder written as the sandbox shows it — the
    folder as given, and as resolved — ending at a path's end, not inside a longer name."""
    for host, inside in mounts:
        for form in dict.fromkeys((str(host), str(Path(host).resolve()))):
            text = re.sub(re.escape(form) + r"(?![\w.-])", str(inside), text)
    return text


def refuse_under_tmp(program: str, path: str | None) -> None:
    """Refused, with the reason, when the harness's program is under `/tmp`, which every
    launch sees empty. Found when a checkout cloned under /tmp failed its sandboxed runs with
    bubblewrap's own words: "execvp …: No such file or directory"."""
    from tuningfork.ops.common import OperationError

    found = shutil.which(program, path=path) if os.sep not in program else program
    for form in dict.fromkeys(Path(each) for each in (found, found and os.path.realpath(found)) if each):
        if _within(form, TMP):
            raise OperationError(
                f"the harness's program, {form}, is under {TMP}, which every sandboxed launch "
                "sees empty",
                hint=f"install the harness, and any Python it runs on, outside {TMP}; or launch "
                f"without the sandbox: {SWITCH}=off",
                code="sandbox_tmp",
            )


def hideable(
    folders: Iterable[Path], *, real_home: Path, keep: Sequence[Path] = ()
) -> list[Path]:
    """Of the folders tuningfork would hide, the ones a launch could otherwise read: not one
    the sandbox hides anyway (under `/tmp`, or another person's home), not one holding the
    person's home or the harness's program (`keep`), and not one inside another hidden."""
    found: list[Path] = []
    real_home = Path(real_home).resolve()
    for folder in sorted({Path(folder).resolve() for folder in folders}, key=lambda path: len(path.parts)):
        if not folder.is_dir() or folder == Path("/"):
            continue
        if _within(folder, TMP) or _within(real_home, folder):
            continue
        if _within(folder, Path("/home")) and not _within(folder, real_home):
            continue
        if any(_within(kept, folder) for kept in keep) or any(_within(folder, done) for done in found):
            continue
        found.append(folder)
    return found


def _within(path: Path, folder: Path) -> bool:
    """`path` is `folder` or inside it."""
    try:
        Path(path).relative_to(folder)
    except ValueError:
        return False
    return True


def _resolve(program: str, path: str | None) -> Path | None:
    """The file a program name runs, its links followed."""
    found = shutil.which(program, path=path) if os.sep not in program else program
    return Path(found).resolve() if found else None
