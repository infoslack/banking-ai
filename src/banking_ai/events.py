"""Eventos do log de tool calls: o público vê a máquina por dentro."""

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

EventKind = Literal[
    "user_message",
    "input_guard",
    "llm",
    "tool_call",
    "tool_result",
    "output_guard",
    "template",
    "reply",
    "typing",
    "error",
]


class LogEvent(BaseModel):
    kind: EventKind
    title: str
    detail: str = ""
    ok: bool = True
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))


Emitter = Callable[[LogEvent], Awaitable[None]]
