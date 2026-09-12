"""Domínio: linhas do Postgres e resultados do ledger (os campos espelham as colunas)."""

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class Account(BaseModel):
    id: UUID
    nome: str
    chave_pix: str


class Balance(BaseModel):
    conta_id: UUID
    saldo_centavos: int


class Contact(BaseModel):
    nome: str
    chave_pix: str


class PixPayload(BaseModel):
    """Payload canônico de uma intent de Pix, gravado no banco na fase 1."""

    tipo: Literal["pix"] = "pix"
    chave_pix: str
    valor_centavos: int = Field(gt=0)
    descricao: str = ""
    nome_recebedor: str
    conta_recebedor_id: UUID


class BoletoPayload(BaseModel):
    tipo: Literal["boleto"] = "boleto"
    linha_digitavel: str
    valor_centavos: int = Field(gt=0)
    beneficiario: str


IntentPayload = Annotated[PixPayload | BoletoPayload, Field(discriminator="tipo")]
IntentStatus = Literal["pendente", "confirmada", "expirada", "cancelada"]


class Intent(BaseModel):
    id: UUID
    user_id: UUID
    tipo: Literal["pix", "boleto"]
    payload: IntentPayload
    status: IntentStatus
    criada_em: datetime
    expira_em: datetime


class LedgerEntry(BaseModel):
    id: int
    intent_id: UUID | None
    conta_debito: UUID
    conta_credito: UUID
    valor_centavos: int
    criado_em: datetime


class StatementEntry(BaseModel):
    """Lançamento visto do ponto de vista de uma conta (API do painel)."""

    id: int
    intent_id: UUID | None
    direction: Literal["debit", "credit"]
    counterparty: str
    amount_cents: int
    created_at: datetime


ConfirmationStatus = Literal[
    "executada",
    "ja_executada",
    "expirada",
    "cancelada",
    "saldo_insuficiente",
    "nao_encontrada",
]


class ConfirmationResult(BaseModel):
    status: ConfirmationStatus
    intent_id: UUID
    lancamento_id: int | None = None
    valor_centavos: int | None = None
    destinatario: str | None = None
    saldo_centavos: int | None = None
    observacao: str = ""


class CancellationResult(BaseModel):
    status: Literal["cancelada", "nao_encontrada", "nao_pendente"]
    intent_id: UUID
