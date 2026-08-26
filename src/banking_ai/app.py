"""FastAPI: composição dos serviços, endpoints HTTP e o front estático."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel

from banking_ai import ledger
from banking_ai.agent import Agent
from banking_ai.channels.web import WebChannel
from banking_ai.config import Settings, load_settings
from banking_ai.db import create_pool, prepare_database, reset_demo
from banking_ai.events import LogEvent
from banking_ai.guards import Guards
from banking_ai.llm import GroqLLM, UnreadableDocument
from banking_ai.models import Account, ConfirmationResult, ExtractedDocument, StatementEntry
from banking_ai.render import execution_text
from banking_ai.session import SessionManager

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"


@dataclass
class Services:
    settings: Settings
    pool: AsyncConnectionPool
    llm: GroqLLM
    agent: Agent
    sessions: SessionManager
    user: Account


_services: Services | None = None


def services() -> Services:
    if _services is None:
        raise HTTPException(status_code=503, detail="serviços ainda não inicializados")
    return _services


Deps = Annotated[Services, Depends(services)]


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


class SessionInfo(BaseModel):
    user: str
    pix_key: str
    model: str


async def _load_user(pool: AsyncConnectionPool, pix_key: str) -> Account:
    async with pool.connection() as conn:
        account = await ledger.account_by_pix_key(conn, pix_key)
    if account is None:
        raise RuntimeError(f"usuário da demo não encontrado nos seeds: {pix_key}")
    return account


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _services
    settings = load_settings()
    pool = create_pool(settings.database_url)
    await pool.open()
    await prepare_database(pool)
    llm = GroqLLM(settings)
    guards = Guards(scorer=llm.score_injection, threshold=settings.injection_threshold)
    agent = Agent(llm, guards, pool, max_iterations=settings.max_tool_iterations)
    user = await _load_user(pool, settings.demo_user_pix_key)
    _services = Services(settings=settings, pool=pool, llm=llm, agent=agent, sessions=SessionManager(), user=user)
    app.include_router(WebChannel(agent, _services.sessions, user).router)
    try:
        yield
    finally:
        await pool.close()
        _services = None


def create_app() -> FastAPI:
    app = FastAPI(title="banking-ai", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))

    @app.get("/api/session")
    async def session_info(deps: Deps) -> SessionInfo:
        return SessionInfo(user=deps.user.nome, pix_key=deps.user.chave_pix, model=deps.llm.model)

    @app.post("/api/audio")
    async def transcribe(deps: Deps, file: Annotated[UploadFile, File()]) -> TranscriptionResponse:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="áudio vazio")
        text = await deps.llm.transcribe(content, file.filename or "audio.webm")
        return TranscriptionResponse(text=text)

    @app.post("/api/document")
    async def read_document(deps: Deps, file: Annotated[UploadFile, File()]) -> DocumentResponse:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="imagem vazia")
        try:
            document = await deps.llm.read_document(content, file.content_type or "image/png")
        except UnreadableDocument as exc:
            raise HTTPException(status_code=422, detail=f"não consegui ler o documento: {exc}") from exc
        return DocumentResponse(document=document)

    @app.get("/api/statement")
    async def statement(deps: Deps) -> StatementResponse:
        async with deps.pool.connection() as conn:
            balance = await ledger.get_balance(conn, deps.user.id)
            entries = await ledger.statement(conn, deps.user.id)
        return StatementResponse(user=deps.user.nome, balance_cents=balance.saldo_centavos, entries=entries)

    @app.post("/api/intents/{intent_id}/resend-confirmation")
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

    @app.post("/api/admin/reset")
    async def reset(deps: Deps) -> ResetResponse:
        async with deps.pool.connection() as conn:
            await reset_demo(conn)
        deps.sessions.clear()
        return ResetResponse(ok=True, user=deps.user.nome)

    return app


app = create_app()
