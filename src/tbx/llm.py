"""OpenAI-compatible chat client for Meta, Baseten and Modal (CONTRACTS.md section 1).

Keys come from the environment and are never logged, traced or echoed in errors.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

PROVIDERS = {
    "meta": ("META_BASE_URL", "META_API_KEY", "https://api.meta.ai/v1", "muse-spark-1.3"),
    "baseten": ("BASETEN_BASE_URL", "BASETEN_API_KEY", "https://inference.baseten.co/v1", "meta-models/muse-spark-open"),
    "modal": ("MODAL_BASE_URL", "MODAL_API_KEY", None, "muse-spark-open"),
}
TEMPERATURE = 0.1
MAX_TOKENS = 8192  # reasoning tokens count against this; 800 was not enough (spike S1)
_THINK = re.compile(r"(?is)<think>.*?(</think>|$)")


class ProviderError(RuntimeError):
    """Network or HTTP failure talking to the model provider (CLI exit 3)."""


class ConfigError(ValueError):
    """Bad provider/model configuration (CLI exit 1)."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class ChatResponse:
    content: str | None
    reasoning: str | None
    finish_reason: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    latency_ms: int = 0


def strip_reasoning(text: str | None) -> str | None:
    if text is None:
        return None
    out = _THINK.sub("", text).strip()
    return out or None


def resolve(provider: str | None = None, model: str | None = None) -> tuple[str, str]:
    """Provider and model from args, then TBX_PROVIDER/TBX_MODEL, then per-provider defaults."""
    env_provider = os.environ.get("TBX_PROVIDER") or "meta"
    p = (provider or env_provider).strip().lower()
    if p not in PROVIDERS:
        raise ConfigError(f"unknown provider {p!r} (use one of {', '.join(PROVIDERS)})")
    m = model or (os.environ.get("TBX_MODEL") if p == env_provider else None) or PROVIDERS[p][3]
    if "contributor" in m:
        raise ConfigError(f"refusing {m!r}: -contributor models are a different tier (CONTRACTS section 1)")
    return p, m


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None, timeout: float = 180.0,
                 max_retries: int = 2):
        import openai

        self.provider, self.model = resolve(provider, model)
        url_var, key_var, default_url, _ = PROVIDERS[self.provider]
        base_url = os.environ.get(url_var) or default_url
        api_key = os.environ.get(key_var)
        if not base_url:
            raise ConfigError(f"{url_var} is not set")
        if not api_key:
            raise ConfigError(f"{key_var} is not set")
        self.reasoning_effort = os.environ.get("TBX_REASONING_EFFORT") if self.provider == "meta" else None
        self._openai = openai
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries)

    def chat(self, messages: list[dict], tools: list[dict]) -> ChatResponse:
        kwargs: dict = {"model": self.model, "messages": messages, "tools": tools, "tool_choice": "auto",
                        "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS}
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        t0 = time.monotonic()
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except self._openai.APIStatusError as e:
            raise ProviderError(f"{self.provider} returned HTTP {e.status_code}: {_safe_message(e)}") from None
        except self._openai.APIError as e:  # connection errors, timeouts
            raise ProviderError(f"{self.provider} request failed: {type(e).__name__}") from None
        latency = int((time.monotonic() - t0) * 1000)
        if not resp.choices:
            raise ProviderError(f"{self.provider} returned no choices")
        choice = resp.choices[0]
        msg = choice.message
        extra = getattr(msg, "model_extra", None) or {}
        reasoning = extra.get("reasoning_content") or getattr(msg, "reasoning_content", None)
        calls = [ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "")
                 for tc in (msg.tool_calls or []) if getattr(tc, "function", None)]
        u = resp.usage
        usage = {"prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                 "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
                 "cached_tokens": getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0}
        return ChatResponse(content=strip_reasoning(msg.content), reasoning=reasoning,
                            finish_reason=choice.finish_reason, tool_calls=calls, usage=usage, latency_ms=latency)


def _safe_message(e: Exception) -> str:
    body = getattr(e, "body", None)
    msg = body.get("message") if isinstance(body, dict) else None
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        msg = body["error"].get("message")
    text = str(msg or "")[:300]
    for var in ("META_API_KEY", "BASETEN_API_KEY", "MODAL_API_KEY"):
        key = os.environ.get(var)
        if key:
            text = text.replace(key, "<redacted>")
    return text or type(e).__name__
