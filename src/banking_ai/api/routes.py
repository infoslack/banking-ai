"""Endpoints HTTP, magros: validação de borda e uma chamada de serviço cada."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, File, HTTPException, UploadFile

from banking_ai import ledger
from banking_ai.api.deps import Deps
from banking_ai.db import reset_demo
from banking_ai.events import LogEvent
from banking_ai.llm.client import UnreadableDocument
from banking_ai.models.api import (
    DocumentResponse,
    ResendResponse,
    ResetResponse,
    SessionInfo,
    StatementResponse,
    TranscriptionResponse,
)
from banking_ai.render import execution_text

router = APIRouter(prefix="/api")


@router.get("/session")
async def session_info(deps: Deps) -> SessionInfo:
    return SessionInfo(user=deps.user.nome, pix_key=deps.user.chave_pix, model=deps.llm.model)


@router.post("/audio")
async def transcribe(deps: Deps, file: Annotated[UploadFile, File()]) -> TranscriptionResponse:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="áudio vazio")
    text = await deps.llm.transcribe(content, file.filename or "audio.webm")
    return TranscriptionResponse(text=text)


@router.post("/document")
async def read_document(deps: Deps, file: Annotated[UploadFile, File()]) -> DocumentResponse:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="imagem vazia")
    try:
        document = await deps.llm.read_document(content, file.content_type or "image/png")
    except UnreadableDocument as exc:
        raise HTTPException(status_code=422, detail=f"não consegui ler o documento: {exc}") from exc
    return DocumentResponse(document=document)


@router.get("/statement")
async def statement(deps: Deps) -> StatementResponse:
    async with deps.pool.connection() as conn:
        balance = await ledger.get_balance(conn, deps.user.id)
        entries = await ledger.statement(conn, deps.user.id)
    return StatementResponse(user=deps.user.nome, balance_cents=balance.saldo_centavos, entries=entries)


@router.post("/intents/{intent_id}/resend-confirmation")
async def resend_confirmation(deps: Deps, intent_id: UUID) -> ResendResponse:
    """Simula a requisição duplicada da demo: reexecuta a confirmação da mesma intent."""
    events = [LogEvent(kind="tool_call", title="requisição duplicada · confirm_intent", detail=str(intent_id))]
    async with deps.pool.connection() as conn:
        result = await ledger.confirm_intent(conn, intent_id, deps.user.id)
    events.append(
        LogEvent(
            kind="tool_result",
            title=f"ledger · {result.status}",
            detail=result.model_dump_json(indent=2),
            ok=result.status == "ja_executada",
        )
    )
    return ResendResponse(result=result, message=execution_text(result), events=events)


@router.post("/admin/reset")
async def reset(deps: Deps) -> ResetResponse:
    async with deps.pool.connection() as conn:
        await reset_demo(conn)
    deps.sessions.clear()
    return ResetResponse(ok=True, user=deps.user.nome)
