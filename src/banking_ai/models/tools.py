"""Tools: o que o LLM enxerga e devolve. `user_id` NUNCA é parâmetro.

Tudo que cruza a fronteira do modelo é revalidado server-side: o schema é
o primeiro guardrail.
"""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from banking_ai.models.domain import Balance, Contact

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


ErrorCode = Literal["limite", "conta_propria", "argumentos_invalidos", "ferramenta_desconhecida"]


class ToolError(BaseModel):
    """Erro devolvido ao LLM (ou renderizado por template, quando o motivo é conhecido)."""

    codigo: ErrorCode
    ferramenta: str
    detalhes: list[str] = Field(default_factory=list)


ToolResult = Balance | ContactsFound | IntentProposal | UnresolvedRecipient | ToolError
