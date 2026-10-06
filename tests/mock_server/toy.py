"""A toy llama.cpp, for the raw path: a tokenizer, a chat template, and a model small enough to
reason about — what `/apply-template`, `/tokenize`, `/detokenize` and `/completion` answer.

The model is a weighted set of answers, written as the raw text a Gemma 4 model would generate
after its turn opens. Its next-token distribution after some generated tokens is the weight of
the answers that begin with those tokens, by the token each has next; an answer that is done
ends with its end token. So a test decides exactly where the model wavers: two answers that
share a beginning and part at one token make a fork at that token, and forcing either side
there leads to that answer.
"""

from __future__ import annotations

import math
import random
import re
from typing import Any

SPECIALS = {
    "<eos>": 1, "<bos>": 2, '<|"|>': 52, "<|tool_call>": 48, "<tool_call|>": 49,
    "<|tool_response>": 50, "<tool_response|>": 51, "<|think|>": 98, "<|channel>": 100,
    "<channel|>": 101, "<|turn>": 105, "<turn|>": 106,
}
#: End-of-generation tokens: an answer that calls a tool ends with <|tool_response>.
EOG = {1, 50, 106}
MODEL_TURN = "<|turn>model\n"
#: What `/props` serves as the chat template: detected as Gemma 4, as llama.cpp detects it.
TEMPLATE = "{# a toy Gemma 4 template #}{{ '<|tool_call>call:' }}"
_TOKEN = re.compile(
    "|".join(re.escape(s) for s in sorted(SPECIALS, key=len, reverse=True))
    + r"|[ \t]?[A-Za-z0-9_]+|\n|[ \t]+|.",
    re.DOTALL,
)


def _text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content if isinstance(content, str) else ""


class ToyLlama:
    def __init__(self) -> None:
        self.vocab: dict[str, int] = dict(SPECIALS)
        self.pieces: dict[int, str] = {token: piece for piece, token in SPECIALS.items()}
        #: (raw answer text, weight): the model. Empty: it only ever ends at once.
        self.answers: list[tuple[str, float]] = []
        #: Every /completion body received, for asserting on what was forced.
        self.completions: list[dict[str, Any]] = []
        #: Answer a grammar with an error, as a server without token grammars would.
        self.refuse_grammar = False
        self._last_prompt: list[int] = []

    # --- tokens ---------------------------------------------------------------------------

    def tokenize(self, text: str) -> list[int]:
        ids = []
        for match in _TOKEN.finditer(text):
            piece = match.group(0)
            if piece not in self.vocab:
                self.vocab[piece] = 1000 + len(self.vocab)
                self.pieces[self.vocab[piece]] = piece
            ids.append(self.vocab[piece])
        return ids

    def piece(self, token: int) -> str:
        return "" if token in EOG else self.pieces.get(token, "")

    # --- the chat template ----------------------------------------------------------------

    def apply_template(self, body: dict[str, Any]) -> str:
        messages = [m for m in body.get("messages") or [] if isinstance(m, dict)]
        thinking = (body.get("chat_template_kwargs") or {}).get("enable_thinking", True)
        system = ""
        if messages and messages[0].get("role") == "system":
            system, messages = _text(messages[0].get("content")), messages[1:]
        tools = [
            (tool.get("function") or {}).get("name", "")
            for tool in body.get("tools") or []
            if isinstance(tool, dict)
        ]
        head = ("<|think|>\n" if thinking else "") + system + (f"\ntools: {','.join(tools)}" if tools else "")
        prompt = f"<|turn>system\n{head}<turn|>\n"
        for message in messages:
            role = {"assistant": "model", "tool": "user"}.get(message.get("role"), message.get("role"))
            text = _text(message.get("content"))
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                text += f"<|tool_call>call:{function.get('name')}{function.get('arguments')}<tool_call|>"
            prompt += f"<|turn>{role}\n{text}<turn|>\n"
        return prompt + MODEL_TURN

    def input_tokens(self, body: dict[str, Any]) -> int:
        return 1 + len(self.tokenize(self.apply_template(body)))

    # --- the model ------------------------------------------------------------------------

    def distribution(self, generated: list[int]) -> dict[int, float]:
        weights: dict[int, float] = {}
        for text, weight in self.answers:
            ids = self.tokenize(text)
            if len(generated) > len(ids) or ids[: len(generated)] != generated:
                continue
            nxt = ids[len(generated)] if len(generated) < len(ids) else (50 if text.endswith("<tool_call|>") else 106)
            weights[nxt] = weights.get(nxt, 0.0) + weight
        if not weights:
            return {106: 1.0}
        total = sum(weights.values())
        return {token: weight / total for token, weight in weights.items()}

    def _entry(self, token: int, p: float) -> dict[str, Any]:
        piece = self.piece(token)
        return {"id": token, "token": piece, "bytes": list(piece.encode()),
                "logprob": math.log(p) if p > 0 else -3.4e38}

    def complete(self, body: dict[str, Any]) -> dict[str, Any]:
        self.completions.append(body)
        prompt = body.get("prompt")
        ids = list(prompt) if isinstance(prompt, list) else [2, *self.tokenize(str(prompt or ""))]
        turn = self.tokenize(MODEL_TURN)
        starts = [i + len(turn) for i in range(len(ids) - len(turn) + 1) if ids[i : i + len(turn)] == turn]
        prefix = ids[starts[-1]:] if starts else ids
        forced = [int(found) for found in re.findall(r"<\[(\d+)\]>", body.get("grammar") or "")]
        n_probs = int(body.get("n_probs") or 0)
        temperature = body.get("temperature")
        rng = random.Random(body.get("seed", 0))
        generated: list[int] = []
        entries = []
        for step in range(int(body.get("n_predict") or 64)):
            dist = self.distribution(prefix + generated)
            if body.get("grammar"):
                if step < len(forced):
                    chosen = forced[step]
                else:  # a finished grammar allows only an end token
                    chosen = max((t for t in dist if t in EOG), key=lambda t: dist[t], default=106)
            elif temperature is not None and temperature <= 0:
                chosen = max(dist, key=lambda t: (dist[t], -t))
            else:
                chosen = rng.choices(list(dist), weights=list(dist.values()))[0]
            generated.append(chosen)
            if n_probs:
                ranked = sorted(dist.items(), key=lambda item: (-item[1], item[0]))[:n_probs]
                entry = self._entry(chosen, dist.get(chosen, 0.0))
                entry["top_logprobs"] = [self._entry(t, p) for t, p in ranked]
                entries.append(entry)
            if chosen in EOG:
                break
        cached = 0
        for a, b in zip(ids, self._last_prompt):
            if a != b:
                break
            cached += 1
        self._last_prompt = ids
        result = {
            "content": "".join(self.piece(t) for t in generated),
            "stop_type": "eos" if generated and generated[-1] in EOG else "limit",
            "tokens": generated if body.get("return_tokens") else [],
            "tokens_evaluated": len(ids),
            "tokens_predicted": len(generated),
            "tokens_cached": cached,
            "timings": {"cache_n": cached, "prompt_n": len(ids) - cached, "predicted_n": len(generated)},
        }
        if n_probs:
            result["completion_probabilities"] = entries
        return result

    # --- /tokenize and friends --------------------------------------------------------------

    def tokenize_body(self, body: dict[str, Any]) -> dict[str, Any]:
        ids = self.tokenize(body.get("content") or "")
        if body.get("add_special"):
            ids = [2, *ids]
        if body.get("with_pieces"):
            return {"tokens": [{"id": t, "piece": self.pieces.get(t, "")} for t in ids]}
        return {"tokens": ids}

    def detokenize_body(self, body: dict[str, Any]) -> dict[str, Any]:
        return {"content": "".join(self.pieces.get(t, "") for t in body.get("tokens") or [])}


def props(**extra: Any) -> dict[str, Any]:
    """A `/props` as llama.cpp serves it, with the toy's Gemma 4 template."""
    return {
        "chat_template": TEMPLATE,
        "chat_template_caps": {"supports_parallel_tool_calls": True},
        "default_generation_settings": {"n_ctx": 65536, "params": {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.05}},
        "build_info": "toy",
        **extra,
    }


__all__ = ["EOG", "SPECIALS", "TEMPLATE", "ToyLlama", "props"]
