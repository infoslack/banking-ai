"""Sessões de conversa em memória (persistência robusta está fora de escopo)."""

from dataclasses import dataclass, field
from uuid import UUID

from groq.types.chat import ChatCompletionMessageParam

from banking_ai.models import Account


@dataclass
class Session:
    id: str
    user: Account
    history: list[ChatCompletionMessageParam] = field(default_factory=list)
    pending_intent: UUID | None = None


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def get_or_create(self, session_id: str, user: Account) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            session = Session(id=session_id, user=user)
            self._sessions[session_id] = session
        return session

    def clear(self) -> None:
        self._sessions.clear()
