"""A Messages request as llama.cpp's server converts it to a chat-completions one.

`server_chat_convert_anthropic_to_oai` in `tools/server/server-chat.cpp` (b11108), mirrored
rule for rule, because the prompt is rendered from what it produces: a Claude Code step can
only be scored, or forked token by token, if this renders the prompt the model was given.
Measured: on every Claude Code step of the tally-report runs, the prompt this renders has the
recorded number of tokens, and llama.cpp's prompt cache takes it for the one it processed.

Not `AnthropicMessages.view`, which is for reading: it orders a turn's tool results before its
text and spaces its JSON. Here a turn's text comes first, tool inputs are compact JSON in key
order, text blocks stay blocks, and thinking is joined into `reasoning_content`.
"""

from __future__ import annotations

import json
from typing import Any

BILLING = "x-anthropic-billing-header:"


def _billing(text: str) -> str:
    """Claude Code stamps a changing `cch=xxxxx` into its system prompt; the server sets it to
    `fffff` so its prompt cache survives."""
    if not text.startswith(BILLING):
        return text
    at = text.find("cch=", len(BILLING))
    if at < 0:
        return text
    at += len("cch=")
    if at + 5 < len(text) and text[at + 5] == ";":
        return text[:at] + "fffff" + text[at + 5 :]
    return text


def _image(source: dict[str, Any]) -> dict[str, Any] | None:
    if source.get("type") == "base64":
        media = source.get("media_type") or "image/jpeg"
        return {"type": "image_url", "image_url": {"url": f"data:{media};base64,{source.get('data', '')}"}}
    if source.get("type") == "url":
        return {"type": "image_url", "image_url": {"url": source.get("url", "")}}
    return None


def to_chat(body: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    messages: list[Any] = []
    system = body.get("system")
    if system is not None:
        text = ""
        if isinstance(system, str):
            text = _billing(system)
        elif isinstance(system, list):
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    text += _billing(block.get("text", ""))
        messages.append({"role": "system", "content": text})
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            messages.append(message)
            continue
        role = message.get("role", "")
        if "content" not in message:
            if role != "assistant":
                messages.append(message)
            continue
        content = message["content"]
        if not isinstance(content, list):
            messages.append(message)
            continue
        calls, converted, results, reasoning = [], [], [], ""
        for block in content:
            kind = block.get("type", "") if isinstance(block, dict) else ""
            if kind == "text":
                converted.append(block)
            elif kind == "thinking":
                reasoning += block.get("thinking", "")
            elif kind == "image":
                image = _image(block.get("source") or {})
                if image is not None:
                    converted.append(image)
            elif kind == "tool_use":
                calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {}), ensure_ascii=False, separators=(",", ":")),
                    },
                })
            elif kind == "tool_result":
                use_id = block.get("tool_use_id", "")
                result = block.get("content")
                if isinstance(result, str):
                    results.append({"role": "tool", "tool_call_id": use_id, "content": result})
                elif isinstance(result, list):
                    text, parts, images = "", [], False
                    for item in result:
                        if not isinstance(item, dict):
                            continue
                        if item.get("type") == "text":
                            text += item.get("text", "")
                            parts.append({"type": "text", "text": item.get("text", "")})
                        elif item.get("type") == "image":
                            images = True
                            image = _image(item.get("source") or {})
                            if image is not None:
                                parts.append(image)
                    results.append({"role": "tool", "tool_call_id": use_id, "content": parts if images else text})
                else:
                    results.append({"role": "tool", "tool_call_id": use_id, "content": ""})
        if converted or calls or reasoning:
            new: dict[str, Any] = {"role": role, "content": converted if converted else ""}
            if calls:
                new["tool_calls"] = calls
            if reasoning:
                new["reasoning_content"] = reasoning
            messages.append(new)
        messages.extend(results)
    out["messages"] = messages
    if isinstance(body.get("tools"), list):
        out["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            }
            for tool in body["tools"]
            if isinstance(tool, dict)
        ]
    choice = body.get("tool_choice")
    if isinstance(choice, dict):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") in ("any", "tool"):
            out["tool_choice"] = "required"
    if "stop_sequences" in body:
        out["stop"] = body["stop_sequences"]
    out["max_tokens"] = body.get("max_tokens", 4096)
    for key in ("temperature", "top_p", "top_k", "stream", "chat_template_kwargs"):
        if key in body:
            out[key] = body[key]
    thinking = body.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "enabled":
        out["thinking_budget_tokens"] = thinking.get("budget_tokens", 10000)
    return out
