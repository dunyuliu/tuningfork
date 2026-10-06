"""llama.cpp's own endpoints, for what chat completions cannot do.

Token-level work needs a server that renders a prompt without generating (`/apply-template`),
tokenizes (`/tokenize`), and continues from token ids with each token's probabilities
(`/completion`), and a chat template to tell the model's output format from (`/props`). That
is llama.cpp's server; knowledge of it lives here, behind `tuningfork.backends.raw`.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from tuningfork.config import server_root


async def fetch_props(client: httpx.AsyncClient, base_url: str) -> dict[str, Any] | None:
    """The server's `/props`, or None if it has none."""
    try:
        response = await client.get(f"{server_root(base_url)}/props", timeout=15)
        if response.status_code != 200:
            return None
        body = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    return body if isinstance(body, dict) else None


#: llama.cpp's endpoints that take `cache_prompt` — false processes the whole prompt instead
#: of reusing what is cached of an earlier one — as measured on b11108: its own completion
#: and its OpenAI chat completions. Its Anthropic Messages endpoint converts a request and
#: drops the field, so a prompt sent there comes from the cache whatever it says.
CACHE_PROMPT_ENDPOINTS = ("/v1/chat/completions", "/completion")


#: How long a measurement of a server's prompt cache is kept, per server and endpoint: a
#: server is rarely restarted with another flag, and a measurement costs two calls.
REUSE_KEPT = 300.0
_reuse: dict[tuple[str, str], tuple[float, bool | None]] = {}

#: A prompt long enough that a copy answered from the cache is unmistakable.
_REUSE_PROMPT = "Reference list: " + " ".join(f"entry-{i}" for i in range(300)) + "\nReply with: ok"


async def reuses_prompts(
    client: httpx.AsyncClient, base_url: str, endpoint: str = "/v1/messages", *,
    model: str = "tuningfork", fresh: bool = False,
) -> bool | None:
    """Whether the server answers a prompt it has just processed from its cache — llama.cpp's
    default, which `--no-cache-prompt` turns off — measured: the same short call sent twice,
    and the second one's cached tokens read. `cache_prompt` is left out, as a harness leaves
    it out. None when the server says nothing to tell from. Kept a while (`fresh` measures
    again)."""
    root = server_root(base_url)
    key = (root, endpoint)
    kept = _reuse.get(key)
    if kept and not fresh and time.monotonic() - kept[0] < REUSE_KEPT:
        return kept[1]
    found: bool | None = None
    try:
        cached: list[Any] = []
        for _ in range(2):
            if endpoint == "/v1/messages":
                response = await client.post(
                    f"{root}/v1/messages",
                    json={"model": model, "max_tokens": 1, "messages": [{"role": "user", "content": _REUSE_PROMPT}]},
                    headers={"anthropic-version": "2023-06-01", "x-api-key": "sk-tuningfork-probe"},
                    timeout=60,
                )
                usage = response.json().get("usage") or {}
                cached.append(usage.get("cache_read_input_tokens"))
            else:
                response = await client.post(
                    f"{root}/v1/chat/completions",
                    json={"model": model, "max_tokens": 1, "messages": [{"role": "user", "content": _REUSE_PROMPT}]},
                    timeout=60,
                )
                body = response.json()
                details = (body.get("usage") or {}).get("prompt_tokens_details") or {}
                cached.append(details.get("cached_tokens", (body.get("timings") or {}).get("cache_n")))
        if isinstance(cached[1], int):
            found = cached[1] > 0
    except (httpx.HTTPError, ValueError, AttributeError, TypeError):
        found = None
    _reuse[key] = (time.monotonic(), found)
    return found


async def is_llamacpp(client: httpx.AsyncClient, base_url: str) -> bool:
    """Whether the server is llama.cpp's: its `/props` reports `default_generation_settings`,
    which no other server's does."""
    props = await fetch_props(client, base_url)
    return isinstance((props or {}).get("default_generation_settings"), dict)


async def fetch_chat_template(client: httpx.AsyncClient, base_url: str) -> str | None:
    props = await fetch_props(client, base_url)
    template = (props or {}).get("chat_template")
    return template if isinstance(template, str) else None


async def server_format(client: httpx.AsyncClient, base_url: str):
    """The output format of the model the server runs, by its chat template, or None."""
    from tuningfork.backends.formats import format_for

    return format_for(await fetch_chat_template(client, base_url))
