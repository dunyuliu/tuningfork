"""tuningfork's own endpoints answer the local user, not whatever web page is open.

A loopback port is reachable from any page in the user's browser. These tests are the
attacks that matter: a forged cross-site request continuing a held call with a response
of the page's choosing, and a DNS-rebinding page reading recordings.
"""

from __future__ import annotations

import asyncio

import pytest

from tuningfork.access import HEADER as TOKEN_HEADER
from tuningfork.api.guard import LocalOnly, bearer, hostname
from tests.helpers import record_pilot, until

API = "/_tuningfork/api"
CHAT = "/v1/chat/completions"
EVIL = "http://evil.example"


def chat(content: str = "hi") -> dict:
    return {"model": "m", "messages": [{"role": "user", "content": content}]}


@pytest.mark.parametrize(
    "value, expected",
    [
        ("localhost:8787", "localhost"),
        ("127.0.0.1", "127.0.0.1"),
        ("[::1]:8787", "::1"),
        ("::1", "::1"),
        ("Evil.Example:80", "evil.example"),
    ],
)
def test_hostname(value, expected):
    assert hostname(value) == expected


def test_a_wildcard_bind_allows_no_extra_host():
    assert LocalOnly(None, bind_host="0.0.0.0").allowed == {"localhost", "127.0.0.1", "::1"}
    assert "10.0.0.5" in LocalOnly(None, bind_host="10.0.0.5").allowed


class TestCrossSiteRequests:
    async def test_a_forged_decision_cannot_reach_a_held_call(
        self, proxy_client, proxy_app, mock_server
    ):
        await proxy_client.post(f"{API}/breakpoints", json={"phase": "request"})
        call = asyncio.create_task(proxy_client.post(CHAT, json=chat("original")))
        (hold,) = await until(lambda: proxy_app.state.engine.holds())

        forged = await proxy_client.post(
            f"{API}/holds/{hold.id[:1]}/continue",
            headers={"origin": EVIL},
            json={"request": chat("injected")},
        )

        assert forged.status_code == 403
        assert hold.pending
        # The page's own origin is still welcome: that is the UI.
        mine = await proxy_client.post(
            f"{API}/holds/{hold.id}/continue", headers={"origin": "http://localhost:8787"}
        )
        assert mine.status_code == 200
        await call
        assert mock_server.requests[-1]["messages"][0]["content"] == "original"

    async def test_a_forged_pause_is_refused(self, proxy_client, proxy_app):
        response = await proxy_client.post(f"{API}/pause", headers={"origin": EVIL})
        assert response.status_code == 403
        assert proxy_app.state.gate.paused is False

    async def test_an_opaque_origin_is_refused(self, proxy_client):
        # Sandboxed iframes and file:// pages send the literal origin "null".
        response = await proxy_client.post(f"{API}/pause", headers={"origin": "null"})
        assert response.status_code == 403

    async def test_the_cli_sends_no_origin_and_is_unaffected(self, proxy_client, proxy_app):
        assert (await proxy_client.post(f"{API}/pause")).status_code == 200
        assert proxy_app.state.gate.paused is True

    async def test_reading_is_not_affected_by_origin(self, proxy_client):
        # A cross-site page cannot read the answer anyway: no CORS headers are sent.
        response = await proxy_client.get(f"{API}/status", headers={"origin": EVIL})
        assert response.status_code == 200
        assert "access-control-allow-origin" not in response.headers


class TestRebinding:
    async def test_recordings_are_not_served_under_a_foreign_hostname(
        self, proxy_client, recorder
    ):
        record_pilot(recorder.store)
        response = await proxy_client.get(f"{API}/runs", headers={"host": "evil.example:8787"})
        assert response.status_code == 403

    async def test_the_page_is_not_served_under_a_foreign_hostname(self, proxy_client):
        response = await proxy_client.get("/_tuningfork/ui/", headers={"host": "evil.example"})
        assert response.status_code == 403


class TestTheNamespaceStaysHome:
    async def test_a_preflight_is_never_relayed_upstream(self, proxy_client, mock_server):
        response = await proxy_client.options(
            f"{API}/pause",
            headers={"origin": EVIL, "access-control-request-method": "POST"},
        )
        assert response.status_code in (403, 404, 405)
        assert "access-control-allow-origin" not in response.headers
        assert mock_server.seen == []

    async def test_an_unknown_endpoint_is_not_relayed_either(self, proxy_client, mock_server):
        response = await proxy_client.get(f"{API}/no-such-thing")
        assert response.status_code == 404
        assert mock_server.seen == []

    async def test_model_traffic_is_left_alone(self, proxy_client, mock_server):
        """The guard covers tuningfork's endpoints only; the harness's calls pass as ever."""
        response = await proxy_client.post(CHAT, json=chat(), headers={"origin": EVIL})
        assert response.status_code == 200
        assert len(mock_server.requests) == 1


class TestFraming:
    """A framed UI makes its requests from its own origin, passing the Origin check; one
    click steered onto Continue would release a call the person deliberately stopped."""

    @pytest.mark.parametrize("path", ["/_tuningfork/ui/", f"{API}/status"])
    async def test_tuningforks_pages_refuse_to_be_framed(self, proxy_client, path):
        response = await proxy_client.get(path)
        assert response.headers["x-frame-options"] == "DENY"
        assert "frame-ancestors 'none'" in response.headers["content-security-policy"]

    async def test_model_traffic_gets_no_extra_headers(self, proxy_client):
        response = await proxy_client.post(CHAT, json=chat())
        assert "x-frame-options" not in response.headers


class TestTheAccessToken:
    """Every state change needs the token; reading does not. Checked in the middleware, so
    an endpoint added later cannot forget it."""

    async def test_a_change_without_the_token_is_refused_before_it_acts(self, proxy_client, proxy_app):
        response = await proxy_client.post(f"{API}/pause", headers={TOKEN_HEADER: ""})
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "token_missing"
        assert "tuningfork token" in response.json()["error"]["hint"]
        assert proxy_app.state.gate.paused is False

    async def test_a_wrong_token_is_refused(self, proxy_client, proxy_app):
        await proxy_client.post(f"{API}/breakpoints", json={"phase": "request"})
        call = asyncio.create_task(proxy_client.post(CHAT, json=chat("original")))
        (hold,) = await until(lambda: proxy_app.state.engine.holds())
        forged = await proxy_client.post(
            f"{API}/holds/{hold.id}/continue", headers={TOKEN_HEADER: "guess"},
            json={"request": chat("injected")},
        )
        assert forged.status_code == 403 and forged.json()["error"]["code"] == "token_wrong"
        assert hold.pending
        await proxy_client.post(f"{API}/holds/{hold.id}/continue")
        await call

    @pytest.mark.parametrize("path", [f"{API}/status", f"{API}/runs", "/_tuningfork/ui/", "/_tuningfork/health"])
    async def test_reading_needs_no_token(self, proxy_client, path):
        assert (await proxy_client.get(path, headers={TOKEN_HEADER: ""})).status_code == 200

    async def test_a_bearer_token_is_the_token(self, proxy_client, proxy_app):
        """All most MCP clients can send. A browser never adds one by itself."""
        token = proxy_app.state.access.token
        paused = await proxy_client.post(f"{API}/pause", headers={TOKEN_HEADER: "", "authorization": f"Bearer {token}"})
        assert paused.status_code == 200 and proxy_app.state.gate.paused
        await proxy_client.post(f"{API}/resume")
        basic = await proxy_client.post(f"{API}/pause", headers={TOKEN_HEADER: "", "authorization": f"Basic {token}"})
        assert basic.status_code == 401

    async def test_the_mcp_channel_needs_the_token_even_to_read(self, proxy_client):
        refused = await proxy_client.get("/_tuningfork/mcp", headers={TOKEN_HEADER: ""})
        assert refused.status_code == 401 and refused.json()["error"]["code"] == "token_missing"

    async def test_the_mcp_token_opens_the_mcp_channel_alone(self, proxy_client, proxy_app):
        """An MCP client holds it: the tools, and nothing they leave out."""
        paused = await proxy_client.post(f"{API}/pause", headers={TOKEN_HEADER: proxy_app.state.access.mcp_token})
        assert paused.status_code == 403 and "the MCP token" in paused.json()["error"]["message"]
        assert proxy_app.state.gate.paused is False
        # The client's own header is the access token: no key to the tools.
        refused = await proxy_client.get("/_tuningfork/mcp")
        assert refused.status_code == 403 and refused.json()["error"]["code"] == "token_wrong"


@pytest.mark.parametrize(
    "value, expected",
    [("Bearer abc", "abc"), ("bearer  abc ", "abc"), ("Basic abc", None), ("Bearer", None), ("", None), (None, None)],
)
def test_bearer(value, expected):
    assert bearer(value) == expected

    async def test_the_token_never_travels_upstream(self, proxy_client, mock_server):
        """Model traffic is the harness's, and the server is someone else's."""
        response = await proxy_client.post(CHAT, json=chat())
        assert response.status_code == 200
        (_, _, headers), = [seen for seen in mock_server.seen if seen[1] == CHAT]
        assert TOKEN_HEADER not in {name.lower() for name in headers}

    async def test_a_browsing_server_changes_nothing(self, store):
        import httpx

        from tuningfork.proxy.app import create_browse_app

        app = create_browse_app(store)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost:8788"
        ) as client:
            refused = await client.post(f"{API}/pause", headers={TOKEN_HEADER: "anything"})
            status = await client.get(f"{API}/status")
        assert refused.status_code == 405 and refused.json()["error"]["code"] == "read_only"
        assert status.json()["auth"] == {"required": False, "valid": False}


class TestThePagePolicy:
    """From Phase 5 the page can start work that runs shell commands, so text injected into
    it must never run: only the page's own scripts, nothing inline, nothing from elsewhere."""

    async def test_the_ui_runs_only_its_own_scripts(self, proxy_client):
        response = await proxy_client.get("/_tuningfork/ui/")
        policy = response.headers["content-security-policy"]
        assert "script-src 'self'" in policy and "unsafe" not in policy
        assert "default-src 'none'" in policy and "frame-ancestors 'none'" in policy
        assert response.headers["referrer-policy"] == "no-referrer"
        # Checked again on every load: after an upgrade, never the old modules from cache.
        assert response.headers["cache-control"] == "no-cache"

    async def test_the_browsing_ui_has_the_same_policy(self, store):
        import httpx

        from tuningfork.proxy.app import create_browse_app

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_browse_app(store)), base_url="http://localhost:8788"
        ) as client:
            response = await client.get("/_tuningfork/ui/")
        assert "script-src 'self'" in response.headers["content-security-policy"]
        assert response.headers["referrer-policy"] == "no-referrer"
