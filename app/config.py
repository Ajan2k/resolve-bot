"""Settings read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date

from dotenv import load_dotenv

load_dotenv()


def _today() -> date:
    raw = os.getenv("APP_TODAY", "2026-07-01").strip()
    return date.fromisoformat(raw)


@dataclass(frozen=True)
class Settings:
    policies_path: str = field(default_factory=lambda: os.getenv("POLICIES_PATH", "./data/policies.md"))
    manifest_path: str = field(default_factory=lambda: os.getenv("MANIFEST_PATH", "./data/manifest.json"))
    db_path: str = field(default_factory=lambda: os.getenv("DB_PATH", "./data/shopsphere.db"))
    app_today: date = field(default_factory=_today)
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "groq").strip().lower())
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "").strip())
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_base_url: str = field(default_factory=lambda: os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"))
    max_tool_calls: int = field(default_factory=lambda: int(os.getenv("MAX_TOOL_CALLS", "6")))


settings = Settings()
