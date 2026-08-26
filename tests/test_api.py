"""Endpoints HTTP com o app real (lifespan de verdade: Postgres + settings do .env)."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from banking_ai.app import app
from banking_ai.config import load_settings


@pytest.fixture
def client() -> Iterator[TestClient]:
    try:
        load_settings()
    except ValidationError:
        pytest.skip("GROQ_API_KEY ausente (.env)")
    with TestClient(app) as client:
        yield client


def test_session_and_statement(client: TestClient) -> None:
    client.post("/api/admin/reset")
    session = client.get("/api/session").json()
    assert session["user"] == "Daniel Romero"
    statement = client.get("/api/statement").json()
    assert statement["balance_cents"] == 250_000
    assert statement["entries"][0]["counterparty"] == "Tesouraria"
    assert statement["entries"][0]["direction"] == "credit"


def test_front_is_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "tool calls" in response.text


def test_resend_of_unknown_intent(client: TestClient) -> None:
    response = client.post("/api/intents/00000000-0000-0000-0000-000000000009/resend-confirmation")
    assert response.status_code == 200
    assert response.json()["result"]["status"] == "nao_encontrada"
