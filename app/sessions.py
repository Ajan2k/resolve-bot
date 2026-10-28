"""In-memory chat sessions keyed by session_id. A session belongs to the customer who started it."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Proposal:
    tool: str
    args: dict[str, Any]
    description: str


@dataclass
class Session:
    session_id: str
    customer_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)  # neutral LLM message format
    pending: Optional[Proposal] = None


class SessionStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, Session] = {}

    def get_or_create(self, session_id: str, customer_id: str) -> Session:
        """Raises PermissionError if the session belongs to another customer."""
        with self._lock:
            s = self._sessions.get(session_id)
            if s is None:
                s = Session(session_id=session_id, customer_id=customer_id)
                self._sessions[session_id] = s
            elif s.customer_id != customer_id:
                raise PermissionError("session belongs to another customer")
            return s
