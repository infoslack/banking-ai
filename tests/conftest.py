"""Fixtures: Postgres real (o do docker compose) e um LLM fake como seam de dependência."""

import os
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

import pytest
from groq.types.chat import ChatCompletionMessage, ChatCompletionMessageParam, ChatCompletionToolParam
from groq.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
from psycopg import OperationalError
from psycopg_pool import AsyncConnectionPool, PoolTimeout

from banking_ai import ledger
from banking_ai.db import create_pool, prepare_database, reset_demo
from banking_ai.events import LogEvent
from banking_ai.llm import ToolRequested
from banking_ai.models import Account, AgentReply

DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "postgresql://banking:banking@localhost:5433/banking")
DANIEL_PIX_KEY = "daniel@email.com"


@pytest.fixture
async def pool() -> AsyncIterator[AsyncConnectionPool]:
    pool = create_pool(DATABASE_URL)
    try:
        await pool.open(wait=True, timeout=5)
    except (OperationalError, PoolTimeout):
        pytest.skip(f"Postgres indisponível em {DATABASE_URL} (docker compose up -d db)")
    await prepare_database(pool)
    async with pool.connection() as conn:
        await reset_demo(conn)
    yield pool
    await pool.close()


@pytest.fixture
async def daniel(pool: AsyncConnectionPool) -> Account:
    async with pool.connection() as conn:
        account = await ledger.account_by_pix_key(conn, DANIEL_PIX_KEY)
    if account is None:
        raise AssertionError("seed do Daniel ausente")
    return account


@dataclass
class FakeCall:
    name: str
    arguments: str
    id: str = "call_1"


def tool_round(calls: Sequence[FakeCall]) -> ChatCompletionMessage:
    return ChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ChatCompletionMessageToolCall(id=call.id, type="function", function=Function(name=call.name, arguments=call.arguments))
            for call in calls
        ],
    )


def plain_round(text: str = "ok") -> ChatCompletionMessage:
    return ChatCompletionMessage(role="assistant", content=text)


@dataclass
class FakeLLM:
    """Roteiro determinístico: cada chamada consome a próxima rodada / resposta final."""

    rounds: list[ChatCompletionMessage]
    replies: list[AgentReply]
    tool_requests_in_final_phase: list[ChatCompletionMessage] = field(default_factory=list)
    tool_calls_made: int = 0
    structured_calls_made: int = 0

    @property
    def model(self) -> str:
        return "fake"

    async def chat_with_tools(
        self, messages: list[ChatCompletionMessageParam], tools: list[ChatCompletionToolParam]
    ) -> ChatCompletionMessage:
        self.tool_calls_made += 1
        if not self.rounds:
            return plain_round()
        return self.rounds.pop(0)

    async def structured_reply(self, messages: list[ChatCompletionMessageParam]) -> str:
        self.structured_calls_made += 1
        if self.tool_requests_in_final_phase:
            raise ToolRequested(self.tool_requests_in_final_phase.pop(0))
        if not self.replies:
            return AgentReply(acao="responder", mensagem="ok").model_dump_json()
        return self.replies.pop(0).model_dump_json()


class EventCollector:
    def __init__(self) -> None:
        self.events: list[LogEvent] = []

    async def __call__(self, event: LogEvent) -> None:
        self.events.append(event)

    def titles(self) -> list[str]:
        return [event.title for event in self.events]
