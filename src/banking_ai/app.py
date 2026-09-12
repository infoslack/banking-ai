"""Composição do app: lifespan monta os serviços, o resto são camadas plugadas."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from psycopg_pool import AsyncConnectionPool

from banking_ai import ledger
from banking_ai.agent import Agent
from banking_ai.api.deps import Services, services, set_services
from banking_ai.api.routes import router as api_router
from banking_ai.channels.web import WebChannel
from banking_ai.config import load_settings
from banking_ai.db import create_pool, prepare_database
from banking_ai.guards import Guards
from banking_ai.llm.client import GroqLLM
from banking_ai.models.domain import Account
from banking_ai.session import SessionManager

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"


async def _load_user(pool: AsyncConnectionPool, pix_key: str) -> Account:
    async with pool.connection() as conn:
        account = await ledger.account_by_pix_key(conn, pix_key)
    if account is None:
        raise RuntimeError(f"usuário da demo não encontrado nos seeds: {pix_key}")
    return account


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = load_settings()
    pool = create_pool(settings.database_url)
    await pool.open()
    await prepare_database(pool)
    llm = GroqLLM(settings)
    guards = Guards(scorer=llm.score_injection, threshold=settings.injection_threshold)
    agent = Agent(llm, guards, pool, max_iterations=settings.max_tool_iterations)
    user = await _load_user(pool, settings.demo_user_pix_key)
    set_services(Services(settings=settings, pool=pool, llm=llm, agent=agent, sessions=SessionManager(), user=user))
    app.include_router(WebChannel(agent, services().sessions, user).router)
    try:
        yield
    finally:
        await pool.close()
        set_services(None)


def create_app() -> FastAPI:
    app = FastAPI(title="banking-ai", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.include_router(api_router)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))

    return app


app = create_app()
