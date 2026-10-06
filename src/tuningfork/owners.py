"""Who is running a piece of work, and whether that process is still alive.

Stored with a job or an experiment when it starts, so a later process can tell work that is
still going from work that was cut off when its process died.
"""

from __future__ import annotations

import os
import socket
import sys
from typing import Any


def _start_ticks(pid: int) -> str | None:
    """A process's start time in clock ticks since boot (Linux), which tells a live
    process apart from a later one that reused its pid."""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as stat:
            return stat.read().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def current_owner() -> dict[str, Any]:
    return {"pid": os.getpid(), "host": socket.gethostname(), "start": _start_ticks(os.getpid())}


def owner_alive(owner: dict[str, Any] | None) -> bool | None:
    """True or False when it can be told, None when it cannot: another host, or a platform
    where probing a pid is unsafe — on Windows, `os.kill(pid, 0)` would end the process."""
    if not owner or owner.get("host") != socket.gethostname():
        return None
    pid = owner.get("pid")
    if not isinstance(pid, int) or sys.platform == "win32":
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    except OSError:
        return None
    start = owner.get("start")
    if start is not None:
        now = _start_ticks(pid)
        if now is not None and now != start:
            return False  # the pid was reused by another process
    return True
