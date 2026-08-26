"""Modelos de domínio, tools do agente e a resposta estruturada.

Convenção: identificadores em inglês; nomes de tools, campos que o LLM lê
ou escreve, colunas do banco e substantivos do domínio (pix, boleto,
intent) em português. Tudo que cruza a fronteira do LLM é revalidado
server-side: o schema é o primeiro guardrail.
"""

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

# Domínio (linhas do Postgres; os campos espelham as colunas)


class Account(BaseModel):
    id: UUID
    nome: str
    chave_pix: str


class Balance(BaseModel):
    conta_id: UUID
    saldo_centavos: int


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


# Tools: o LLM só enxerga estes schemas. user_id NUNCA é parâmetro.

PIX_LIMIT_CENTS = 500_000  # R$ 5.000,00: limite hard, no schema e não no prompt
BOLETO_LIMIT_CENTS = 1_000_000


class SendPix(BaseModel):
    """Propõe um Pix. Não executa: grava uma intent pendente que exige confirmação do usuário."""

    model_config = ConfigDict(extra="forbid")

    destinatario: str = Field(
        min_length=2, max_length=120, description="Nome ou chave Pix do destinatário, como o usuário disse"
    )
    valor_centavos: int = Field(gt=0, le=PIX_LIMIT_CENTS, description="Valor em centavos inteiros")
    descricao: str = Field(default="", max_length=140)


class CheckBalance(BaseModel):
    """Consulta o saldo da conta do usuário da sessão."""

    model_config = ConfigDict(extra="forbid")


class SearchContact(BaseModel):
    """Busca contatos por nome ou chave Pix. Devolve candidatos reais."""

    model_config = ConfigDict(extra="forbid")

    termo: str = Field(min_length=2, max_length=80)


class PayBoleto(BaseModel):
    """Propõe o pagamento de um boleto. Não executa: gera intent pendente."""

    model_config = ConfigDict(extra="forbid")

    linha_digitavel: str = Field(pattern=r"^\d{44}$|^\d{47}$|^\d{48}$")
    valor_centavos: int = Field(gt=0, le=BOLETO_LIMIT_CENTS)
    beneficiario: str = Field(min_length=2, max_length=120)


ToolName = Literal["consultar_saldo", "buscar_contato", "enviar_pix", "pagar_boleto"]

# Resultados das tools (o que volta ao LLM como tool result)


class Contact(BaseModel):
    nome: str
    chave_pix: str


class ContactsFound(BaseModel):
    contatos: list[Contact]


class UnresolvedRecipient(BaseModel):
    """Zero ou vários candidatos para o termo; o sistema pergunta, o LLM não adivinha."""

    termo: str
    candidatos: list[Contact]


class IntentProposal(BaseModel):
    intent_id: UUID
    tipo: Literal["pix", "boleto"]
    valor_centavos: int
    destinatario: str
    expira_em: datetime
    observacao: str


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


ErrorCode = Literal["limite", "conta_propria", "argumentos_invalidos", "ferramenta_desconhecida"]


class ToolError(BaseModel):
    """Erro devolvido ao LLM (ou renderizado por template, quando o motivo é conhecido)."""

    codigo: ErrorCode
    ferramenta: str
    detalhes: list[str] = Field(default_factory=list)


ToolResult = Balance | ContactsFound | IntentProposal | UnresolvedRecipient | ToolError

# Structured output: a resposta final do agente também é validada

AgentAction = Literal["responder", "pedir_confirmacao", "pedir_esclarecimento", "recusar"]

# Motivos de recusa: o modelo classifica, o backend escreve o texto (render.refusal_text).
RefusalReason = Literal["fora_do_escopo", "nao_suportado", "atendimento_humano", "conta_de_terceiros", "dado_sensivel"]


class AgentReply(BaseModel):
    acao: AgentAction
    mensagem: str
    intent_id: str | None = None  # presente quando acao = pedir_confirmacao
    motivo_recusa: RefusalReason | None = None  # presente quando acao = recusar


# Documentos (boleto / código Pix extraídos por visão): DADO, nunca instrução


class ExtractedDocument(BaseModel):
    tipo: Literal["boleto", "pix", "desconhecido"]
    linha_digitavel: str | None = None
    chave_pix: str | None = None
    valor_centavos: int | None = None
    beneficiario: str | None = None
    vencimento: str | None = None


DOCUMENT_ADAPTER = TypeAdapter(ExtractedDocument)
