"""What an inference server can and cannot do.

tuningfork targets whatever OpenAI-compatible server the user already runs, and those differ
in ways that matter: some return logprobs, some ignore `n`, some expose a chat-template
endpoint. Rather than branching on a server name, features declare the capabilities they
need and a probed profile decides whether they can run.

A profile is also the natural unit of contribution: run the probe on your stack, commit the
JSON, and the next person knows what to expect.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable


#: Home-directory shapes across the platforms tuningfork runs on. A probe result embeds
#: model paths, and a profile is only contributable if it carries no one's username.
#: Most specific first: `/mnt/c/Users/name` must be matched before the bare `/Users/name`
#: inside it, or the mount prefix is left behind.
_HOME_PATTERNS = (
    re.compile(r"/mnt/[a-z]/Users/[^/\"\s]+", re.IGNORECASE),
    re.compile(r"[A-Za-z]:\\\\?Users\\\\?[^\\\"\s]+"),
    re.compile(r"/home/[^/\"\s]+"),
    re.compile(r"/Users/[^/\"\s]+"),
)


def anonymize_paths(value: Any) -> Any:
    """Replace home directories with `<home>` throughout a nested structure."""
    if isinstance(value, dict):
        return {key: anonymize_paths(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [anonymize_paths(item) for item in value]
    if isinstance(value, str):
        for pattern in _HOME_PATTERNS:
            value = pattern.sub("<home>", value)
        return value
    return value


class Capability(StrEnum):
    TOOL_CALLS = "tool_calls"
    STREAMING_TOOL_CALLS = "streaming_tool_calls"
    REASONING_CONTENT = "reasoning_content"
    LOGPROBS = "logprobs"
    TOP_LOGPROBS = "top_logprobs"
    SERVER_SIDE_N = "server_side_n"
    SEED_DETERMINISM = "seed_determinism"
    USAGE_IN_STREAM = "usage_in_stream"
    PREFIX_CACHE = "prefix_cache"
    PARTIAL_ASSISTANT_CONTINUATION = "partial_assistant_continuation"
    CHAT_TEMPLATE_ENDPOINT = "chat_template_endpoint"
    ANTHROPIC_MESSAGES = "anthropic_messages"
    LOGPROBS_ALL_TOKENS = "logprobs_all_tokens"
    PROMPT_EQUIVALENCE = "prompt_equivalence"
    RAW_COMPLETION = "raw_completion"
    KNOWN_OUTPUT_FORMAT = "known_output_format"
    FRESH_PROMPTS = "fresh_prompts"


#: Which tuningfork features each capability gates, for error messages that say something
#: useful instead of "unsupported".
CAPABILITY_USES: dict[str, str] = {
    Capability.TOOL_CALLS: "recording and comparing tool-call outcomes",
    Capability.STREAMING_TOOL_CALLS: "recording tool calls from a streaming harness",
    Capability.REASONING_CONTENT: "showing why two samples diverged",
    Capability.LOGPROBS: "token-level analysis of samples (resample --logprobs)",
    Capability.TOP_LOGPROBS: "the alternatives a heatmap shows at each token",
    Capability.LOGPROBS_ALL_TOKENS: "a sample's heatmap over its reasoning and tool calls, not only its text",
    Capability.PROMPT_EQUIVALENCE: "scoring an answer against the exact prompt the model was given",
    Capability.RAW_COMPLETION: "scoring an answer and forcing a token (heatmaps, forking tokens)",
    Capability.KNOWN_OUTPUT_FORMAT: "reading the model's raw output as the server does (heatmap parts, forking tokens)",
    Capability.FRESH_PROMPTS: "samples and forks that do not depend on what the server processed before",
    Capability.SERVER_SIDE_N: "cheap fan-out during resampling",
    Capability.SEED_DETERMINISM: "reproducing an individual sample exactly",
    Capability.USAGE_IN_STREAM: "per-step token accounting",
    Capability.PREFIX_CACHE: "cheap forks from a shared checkpoint prefix",
    Capability.PARTIAL_ASSISTANT_CONTINUATION: "continuing a partial assistant message",
    Capability.CHAT_TEMPLATE_ENDPOINT: "verifying prompt rendering matches another server",
    Capability.ANTHROPIC_MESSAGES: "running Claude Code, which speaks only /v1/messages",
}


class Support(StrEnum):
    YES = "yes"
    NO = "no"
    #: Probed, but the answer needs a human. Treated as unsupported by `require`.
    UNKNOWN = "unknown"


class UnsupportedCapability(RuntimeError):
    """Raised when a feature needs something the server cannot do."""


@dataclass(slots=True)
class CapabilityResult:
    capability: str
    support: Support
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.support is Support.YES


@dataclass(slots=True)
class BackendProfile:
    name: str
    base_url: str
    server: str
    model: str
    probed_at: str
    capabilities: dict[str, CapabilityResult] = field(default_factory=dict)
    performance: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def supports(self, capability: str) -> bool:
        result = self.capabilities.get(str(capability))
        return bool(result and result.ok)

    def require(self, *capabilities: str) -> None:
        """Fail early, and explain what would have to change."""
        missing = [c for c in capabilities if not self.supports(str(c))]
        if not missing:
            return
        lines = [f"{self.server} at {self.base_url} cannot support this operation."]
        for capability in missing:
            result = self.capabilities.get(str(capability))
            use = CAPABILITY_USES.get(str(capability), "this feature")
            state = result.support.value if result else "never probed"
            detail = f" ({result.detail})" if result and result.detail else ""
            lines.append(f"  - {capability}: {state}{detail} — needed for {use}")
        raise UnsupportedCapability("\n".join(lines))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["capabilities"] = {
            key: {**asdict(value), "support": value.support.value}
            for key, value in self.capabilities.items()
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BackendProfile:
        capabilities = {
            key: CapabilityResult(
                capability=value["capability"],
                support=Support(value["support"]),
                detail=value.get("detail", ""),
                evidence=value.get("evidence", {}),
            )
            for key, value in (data.get("capabilities") or {}).items()
        }
        return cls(
            name=data["name"],
            base_url=data["base_url"],
            server=data["server"],
            model=data["model"],
            probed_at=data["probed_at"],
            capabilities=capabilities,
            performance=data.get("performance", {}),
            notes=data.get("notes", []),
        )

    def save(self, path: Path, anonymize: bool = True) -> Path:
        """Write the profile as JSON.

        Anonymized by default. A probe result embeds the model's absolute path, and these
        files are meant to be shared — the whole point of a profile is that someone runs the
        probe on their stack and contributes the result.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = anonymize_paths(self.to_dict()) if anonymize else self.to_dict()
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> BackendProfile:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def summary_rows(self) -> Iterable[tuple[str, str, str]]:
        for key in sorted(self.capabilities):
            result = self.capabilities[key]
            yield key, result.support.value, result.detail
