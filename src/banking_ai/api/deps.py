"""Serviços compartilhados pelos endpoints, montados no lifespan do app."""

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException
from psycopg_pool import AsyncConnectionPool

from banking_ai.agent import Agent
from banking_ai.config import Settings
from banking_ai.llm.client import GroqLLM
from banking_ai.models.domain import Account
from banking_ai.session import SessionManager


@dataclass
class Services:
    settings: Settings
    pool: AsyncConnectionPool
    llm: GroqLLM
    agent: Agent
    sessions: SessionManager
    user: Account


_services: Services | None = None


def set_services(value: Services | None) -> None:
    global _services
    _services = value


def services() -> Services:
    if _services is None:
        raise HTTPException(status_code=503, detail="serviços ainda não inicializados")
    return _services


Deps = Annotated[Services, Depends(services)]
