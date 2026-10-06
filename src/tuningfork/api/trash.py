"""The trash over HTTP: what is in it, and moving things in, back out, and away for good.

Reads stay open; every change needs the access token, checked by `api.guard.LocalOnly` like
any other state change under `/_tuningfork/`. The logic is `ops.trash`.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from tuningfork.api.control import CONTROL_HEADER
from tuningfork.ops import trash
from tuningfork.ops.common import OperationError

PREFIX = "/_tuningfork/api/trash"
STATUS = {"not_found": 404, "busy": 409, "trashed": 409}


def _refusal(error: OperationError) -> JSONResponse:
    return JSONResponse(
        {"error": error.to_json()}, status_code=STATUS.get(error.code or "", 400),
        headers={CONTROL_HEADER: "control"},
    )


async def _item(request: Request) -> tuple[str, str]:
    """The (kind, id) a request names, from its JSON body."""
    raw = await request.body()
    try:
        body = json.loads(raw) if raw else {}
    except ValueError:
        raise OperationError("the body is not JSON", code="bad_params") from None
    if not isinstance(body, dict):
        raise OperationError("the body must be a JSON object", code="bad_params")
    kind, item = body.get("kind"), body.get("id")
    if not isinstance(kind, str) or not isinstance(item, str) or not item:
        raise OperationError("name the item: {kind, id}", code="bad_params", field="id")
    return kind, item


def trash_routes(*, store: Any, writable: bool) -> list[Route]:
    async def answer(function: Any, *args: Any) -> Response:
        try:
            found = await asyncio.to_thread(function, *args)
        except OperationError as error:
            return _refusal(error)
        return JSONResponse(found.to_json() if hasattr(found, "to_json") else found)

    async def listing(request: Request) -> Response:
        return await answer(trash.listing, store)

    async def preview(request: Request) -> Response:
        query = request.query_params
        return await answer(trash.preview, store, query.get("kind") or "", query.get("id") or "")

    routes = [
        Route(PREFIX, listing, methods=["GET"]),
        Route(f"{PREFIX}/preview", preview, methods=["GET"]),
    ]
    if not writable:
        return routes

    def acting(function: Any) -> Any:
        async def route(request: Request) -> Response:
            try:
                kind, item = await _item(request)
            except OperationError as error:
                return _refusal(error)
            return await answer(function, store, kind, item)

        return route

    async def empty(request: Request) -> Response:
        return await answer(trash.empty, store)

    return routes + [
        Route(PREFIX, acting(trash.trash), methods=["POST"]),
        Route(f"{PREFIX}/restore", acting(trash.restore), methods=["POST"]),
        Route(f"{PREFIX}/purge", acting(trash.purge), methods=["POST"]),
        Route(f"{PREFIX}/empty", empty, methods=["POST"]),
    ]
