"""Contratos da API HTTP (o que o front e o Swagger enxergam)."""

from pydantic import BaseModel

from banking_ai.events import LogEvent
from banking_ai.models.domain import ConfirmationResult, StatementEntry
from banking_ai.models.reply import ExtractedDocument


class SessionInfo(BaseModel):
    user: str
    pix_key: str
    model: str


class TranscriptionResponse(BaseModel):
    text: str


class DocumentResponse(BaseModel):
    document: ExtractedDocument


class StatementResponse(BaseModel):
    user: str
    balance_cents: int
    entries: list[StatementEntry]


class ResendResponse(BaseModel):
    result: ConfirmationResult
    message: str
    events: list[LogEvent]


class ResetResponse(BaseModel):
    ok: bool
    user: str
