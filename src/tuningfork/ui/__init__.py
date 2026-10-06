"""The web UI: static files served by tuningfork itself, no build step.

Plain HTML, CSS and JavaScript on purpose. A contributor needs nothing beyond Python to
work on it, and there is no toolchain to keep in step with the Python package.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

from starlette.responses import RedirectResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

STATIC = Path(__file__).parent / "static"
UI_PATH = "/_tuningfork/ui"

# Static files are typed from the `mimetypes` table, which on Windows is read from the
# registry — where `.js` is often `text/plain`, and a browser refuses to run a module
# served that way. Pin the types the UI depends on.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")


def ui_routes() -> list:
    async def redirect(request):
        return RedirectResponse(f"{UI_PATH}/")

    return [
        Route(UI_PATH, redirect, methods=["GET"]),
        Mount(UI_PATH, app=StaticFiles(directory=STATIC, html=True), name="ui"),
    ]
