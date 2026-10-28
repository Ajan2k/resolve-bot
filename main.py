"""ResolveBot API: POST /chat on port 8000.

Run:  uvicorn main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
import threading
from typing import Any, Optional

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

from app.agent import Agent
from app.config import settings
from app.db import Repository
from app.llm import get_client
from app.rag import PolicyIndex
from app.sessions import SessionStore

app = FastAPI(title="ResolveBot", version="0.1.0")

_index = PolicyIndex(settings.policies_path, settings.manifest_path, settings.app_today)
_repo = Repository()
_llm = get_client()
_agent = Agent(_llm, _repo, _index)
_sessions = SessionStore()
_db_lock = threading.Lock()  # one SQLite connection; serialise request handling


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "app_today": settings.app_today.isoformat(), "llm_provider": settings.llm_provider,
            "active_policy_docs": [d.doc_id for d in _index.active]}


@app.post("/chat")
async def chat(request: Request, x_customer_id: Optional[str] = Header(default=None)) -> JSONResponse:
    # 400: invalid JSON or missing fields
    try:
        body = json.loads(await request.body())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _error(400, "Body must be valid JSON.")
    if not isinstance(body, dict):
        return _error(400, "Body must be a JSON object.")
    session_id = body.get("session_id")
    message = body.get("message")
    if not isinstance(session_id, str) or not session_id.strip() or not isinstance(message, str) or not message.strip():
        return _error(400, "session_id and message are required.")

    # 401: missing or unknown customer
    customer_id = (x_customer_id or "").strip()
    if not customer_id or not _repo.customer_exists(customer_id):
        return _error(401, "X-Customer-Id header is missing or unknown.")

    # 403: session owned by another customer
    try:
        session = _sessions.get_or_create(session_id.strip(), customer_id)
    except PermissionError:
        return _error(403, "This session belongs to another customer.")

    with _db_lock:
        result = _agent.handle(session, message.strip())
    return JSONResponse(status_code=200, content=result)
