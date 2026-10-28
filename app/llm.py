"""Provider adapters. All providers share one neutral message format:

    {"role": "system" | "user" | "assistant", "content": str,
     "tool_calls": [{"id": str, "name": str, "args": dict}]}          # assistant only
    {"role": "tool", "tool_call_id": str, "name": str, "content": str}   # tool result (JSON string)

and return an LLMResponse(text, tool_calls). Temperature is always 0.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import settings


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient:
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:  # pragma: no cover
        raise NotImplementedError


def _parse_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------- Groq
class GroqClient(LLMClient):
    """Groq exposes an OpenAI-compatible chat-completions API with tool calling."""

    def __init__(self) -> None:
        from openai import OpenAI

        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set (see .env.example).")
        self.client = OpenAI(api_key=settings.groq_api_key, base_url=settings.groq_base_url)
        self.model = settings.llm_model or "llama-3.3-70b-versatile"
        self.extra: dict[str, Any] = {"parallel_tool_calls": False}  # one tool at a time keeps proposal step simple

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        oa_msgs: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "assistant":
                entry: dict[str, Any] = {"role": "assistant", "content": m.get("content") or None}
                if m.get("tool_calls"):
                    entry["tool_calls"] = [
                        {"id": tc["id"], "type": "function",
                         "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}}
                        for tc in m["tool_calls"]
                    ]
                oa_msgs.append(entry)
            elif m["role"] == "tool":
                oa_msgs.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
            else:
                oa_msgs.append({"role": m["role"], "content": m["content"]})
        oa_tools = [{"type": "function", "function": t} for t in tools]
        resp = self.client.chat.completions.create(
            model=self.model, messages=oa_msgs, tools=oa_tools, tool_choice="auto", temperature=0, **self.extra
        )
        msg = resp.choices[0].message
        calls = [ToolCall(tc.id, tc.function.name, _parse_args(tc.function.arguments)) for tc in (msg.tool_calls or [])]
        return LLMResponse(text=msg.content or "", tool_calls=calls)


# ---------------------------------------------------------------- Mock (no API key)
class MockClient(LLMClient):
    """A deterministic stand-in so the service runs end-to-end without any API key.

    It is deliberately simple: it understands the assessment's example intents,
    calls the same tools a real model would, and phrases replies from the tool results.
    All business decisions still come from the Python tools, exactly as with a real LLM.
    """

    ORDER_RE = re.compile(r"\bSS-\d{3,}\b", re.I)
    ITEM_RE = re.compile(r"\bSS-\d{3,}-\d+\b", re.I)

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        from .mock_brain import MockBrain  # local import keeps this module light

        return MockBrain(messages).step()


def get_client() -> LLMClient:
    p = settings.llm_provider
    if p == "groq":
        return GroqClient()
    if p == "mock":
        return MockClient()
    raise RuntimeError(f"Unknown LLM_PROVIDER '{p}'. Use 'groq' or 'mock'.")


def extract_json(text: str) -> Optional[dict[str, Any]]:
    """Pulls the first JSON object out of a model reply (tolerates ``` fences)."""
    if not text:
        return None
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [fence.group(1)] if fence else []
    start = text.find("{")
    if start != -1:
        candidates.append(text[start: text.rfind("}") + 1])
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None
