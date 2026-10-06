"""Keep tuningfork's own endpoints for the person at the machine.

The proxy listens on loopback, but a loopback port is reachable from every web page open in
the user's browser. Such a page cannot read the answers — there are no CORS headers — yet
it can still send requests, and some of these endpoints act: continuing a held call with
an edited response would let a page put a tool call into a running agent. Hold ids even
match by prefix, so guessing one takes a handful of tries.

Three checks close that, on `/_tuningfork/` paths only:

- **Host** must be a loopback name, or the specific address the proxy was bound to. A page
  using DNS rebinding reaches the port under its own hostname, and is refused — which also
  keeps recordings, which may hold sensitive data, from being read that way.
- **Origin**, when a browser sends one on a state-changing request, must be the proxy's
  own. Browsers attach it to every cross-site POST, so a forged request is refused before
  it acts. The CLI and scripts send none.
- **The access token** (`tuningfork.access`) must come with every state-changing request —
  as `x-tuningfork-token`, or as a bearer token, which is all most MCP clients can send.
  Checked here, once, so no endpoint can forget it: missing is 401, wrong is 403. A server
  with no token — `tuningfork ui`, which only browses — refuses every change with 405. The
  MCP endpoint takes its own token instead, the MCP token, on every request, a GET included:
  it is a channel to the tools, not a page of recordings. Neither token opens the other's
  door, so an MCP client can never call what its tools leave out.

Model traffic passing through is deliberately not guarded: it belongs to the harness, and
must behave exactly as if tuningfork were not there.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from starlette.datastructures import Headers
from starlette.responses import JSONResponse

from tuningfork.access import HEADER, Access

PREFIX = "/_tuningfork"
UI_PREFIX = "/_tuningfork/ui"
MCP_PATH = "/_tuningfork/mcp"
LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})
#: Wildcard binds name no address a browser would use, so they add nothing to allow.
WILDCARD_BINDS = frozenset({"", "0.0.0.0", "::"})
SAFE_METHODS = frozenset({"GET", "HEAD"})
#: Marks a refusal as tuningfork's own; see `api.control.CONTROL_HEADER`.
CONTROL_HEADER = "x-tuningfork"


def hostname(value: str) -> str:
    """The host in a Host header or netloc, without its port or IPv6 brackets."""
    value = value.strip().lower()
    if value.startswith("["):
        end = value.find("]")
        return value[1:end] if end != -1 else value
    if value.count(":") == 1:
        return value.split(":", 1)[0]
    return value


#: The page runs its own scripts and nothing else. From Phase 5 the UI can start work that
#: runs shell commands — a scenario's check, a harness — so text injected into the page
#: must never run: no inline script, no eval, nothing loaded from elsewhere. Nothing may
#: frame it either: a framed UI makes its requests from its own origin, and one click
#: steered onto Continue would release a call the person deliberately stopped.
UI_POLICY = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)
#: Everything else under the prefix is data, never a page.
API_POLICY = "default-src 'none'; frame-ancestors 'none'"


def _hardened(send: Any, policy: str) -> Any:
    headers = (
        (b"x-frame-options", b"DENY"),
        (b"content-security-policy", policy.encode()),
        # The UI's address carries the token the first time it is opened.
        (b"referrer-policy", b"no-referrer"),
        (b"x-content-type-options", b"nosniff"),
        # Checked again on every load (a cheap 304 locally): a browser that kept the page's
        # modules from before an upgrade would otherwise run old code against a new API.
        (b"cache-control", b"no-cache"),
    )

    async def wrapped(message: dict) -> None:
        if message["type"] == "http.response.start":
            present = {name.lower() for name, _ in message.get("headers", [])}
            extra = [header for header in headers if header[0] not in present]
            message = {**message, "headers": [*message.get("headers", []), *extra]}
        await send(message)

    return wrapped


def bearer(value: str | None) -> str | None:
    """The token in `Authorization: Bearer <token>`. A browser never sends one on its own, so
    it is as safe as the custom header."""
    if not value:
        return None
    scheme, _, token = value.strip().partition(" ")
    if scheme.lower() != "bearer":
        return None
    return token.strip() or None


class Refusal(Exception):
    def __init__(self, status: int, message: str, code: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.status, self.message, self.code, self.hint = status, message, code, hint

    def response(self) -> JSONResponse:
        error = {"message": self.message, "code": self.code}
        if self.hint:
            error["hint"] = self.hint
        headers = {CONTROL_HEADER: "control"} if self.code.startswith("token") else {}
        if self.status == 405:
            headers["allow"] = "GET, HEAD"
        return JSONResponse({"error": error}, status_code=self.status, headers=headers)


class LocalOnly:
    """ASGI middleware refusing requests to tuningfork's endpoints from anywhere but here.

    `access` is the running proxy's token; without one, the server changes nothing.
    """

    def __init__(self, app: Any, *, bind_host: str | None = None, access: Access | None = None) -> None:
        self.app = app
        self.access = access
        self.allowed = set(LOOPBACK)
        if bind_host and bind_host not in WILDCARD_BINDS:
            self.allowed.add(hostname(bind_host))

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        path = scope.get("path", "")
        if scope["type"] == "http" and path.startswith(PREFIX):
            try:
                self.check(Headers(scope=scope), scope["method"], path)
            except Refusal as refusal:
                await refusal.response()(scope, receive, send)
                return
            policy = UI_POLICY if path.startswith(UI_PREFIX) else API_POLICY
            await self.app(scope, receive, _hardened(send, policy))
            return
        await self.app(scope, receive, send)

    def check(self, headers: Headers, method: str, path: str = "") -> None:
        host = headers.get("host", "")
        if hostname(host) not in self.allowed:
            raise Refusal(
                403,
                f"tuningfork answers only to a local address, not {host!r}; open it as "
                "http://127.0.0.1:<port> or http://localhost:<port>",
                "host",
            )
        if method in SAFE_METHODS and path != MCP_PATH:
            return
        origin = headers.get("origin")
        if origin is not None and urlsplit(origin).netloc.lower() != host.strip().lower():
            raise Refusal(
                403, "refused a cross-site request: only tuningfork's own page may change it",
                "origin",
            )
        if self.access is None:
            raise Refusal(
                405,
                "this server only browses recordings: nothing here can be changed",
                "read_only",
                hint="live controls need `tuningfork serve`",
            )
        given = headers.get(HEADER) or bearer(headers.get("authorization"))
        if path == MCP_PATH:
            self.check_mcp(given)
            return
        if not given:
            raise Refusal(
                401,
                "changing tuningfork needs its access token, sent as the x-tuningfork-token "
                "header or as `Authorization: Bearer <token>`",
                "token_missing",
                hint="`tuningfork token` prints it; the UI address `tuningfork serve` printed "
                "carries it",
            )
        if not self.access.allows(given):
            raise Refusal(
                403,
                "that is the MCP token: it opens /_tuningfork/mcp and nothing else"
                if self.access.allows_mcp(given) else
                "that access token is not this proxy's: a new one is made each time "
                "`tuningfork serve` starts",
                "token_wrong",
                hint="`tuningfork token` prints the current one",
            )

    def check_mcp(self, given: str | None) -> None:
        """The MCP endpoint takes the MCP token, and only it: a client holding it can call the
        tools, and nothing they leave out."""
        assert self.access is not None
        if not given:
            raise Refusal(
                401,
                "the MCP endpoint needs the MCP token, as `Authorization: Bearer <token>` or the "
                "x-tuningfork-token header",
                "token_missing",
                hint="`tuningfork token --mcp` prints it; `--header-json` prints the header",
            )
        if not self.access.allows_mcp(given):
            raise Refusal(
                403,
                "that is the access token, which this endpoint does not take: an MCP client is given "
                "the MCP token, which opens the tools and nothing else"
                if self.access.allows(given) else
                "that MCP token is not this proxy's: a new one is made each time `tuningfork serve` "
                "starts, unless TUNINGFORK_MCP_TOKEN pins it",
                "token_wrong",
                hint="`tuningfork token --mcp` prints the current one",
            )
