"""Registro e execução das tools.

O LLM só produz argumentos; cada um é revalidado aqui contra o schema
Pydantic antes de tocar no ledger. `user_id` vem da sessão, nunca do modelo.
"""

from dataclasses import dataclass
from uuid import UUID

from groq.types.chat import ChatCompletionToolParam
from psycopg import AsyncConnection
from pydantic import BaseModel, ValidationError

from banking_ai import ledger
from banking_ai.models import (
    Account,
    Balance,
    BoletoPayload,
    CheckBalance,
    ContactsFound,
    IntentProposal,
    PayBoleto,
    PixPayload,
    SearchContact,
    SendPix,
    ToolError,
    ToolName,
    ToolResult,
    UnresolvedRecipient,
)


def tool_spec(name: ToolName, model: type[BaseModel]) -> ChatCompletionToolParam:
    """Formato OpenAI (`tools=[...]`), com o JSON Schema gerado direto do model."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": " ".join((model.__doc__ or "").split()),
            "parameters": model.model_json_schema(),
        },
    }


TOOL_SPECS: list[ChatCompletionToolParam] = [
    tool_spec("consultar_saldo", CheckBalance),
    tool_spec("buscar_contato", SearchContact),
    tool_spec("enviar_pix", SendPix),
    tool_spec("pagar_boleto", PayBoleto),
]


@dataclass
class TurnContext:
    """O que o backend sabe deste turno e o LLM não controla."""

    created_intent: UUID | None = None


def validation_error(tool_name: str, exc: ValidationError) -> ToolError:
    errors = exc.errors()
    details = [f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in errors]
    over_limit = any(error["type"] == "less_than_equal" and "valor_centavos" in error["loc"] for error in errors)
    return ToolError(codigo="limite" if over_limit else "argumentos_invalidos", ferramenta=tool_name, detalhes=details)


def is_terminal(result: ToolResult) -> bool:
    """Resultados terminais: o modelo não precisa chamar mais ferramentas depois deles."""
    match result:
        case Balance() | IntentProposal() | UnresolvedRecipient():
            return True
        case ToolError(codigo=code):
            return code in {"limite", "conta_propria", "ferramenta_desconhecida"}
        case ContactsFound():
            return False


class ToolExecutor:
    def __init__(self, conn: AsyncConnection, user: Account, context: TurnContext) -> None:
        self._conn = conn
        self._user = user
        self.context = context

    async def execute(self, name: str, arguments_json: str) -> ToolResult:
        try:
            match name:
                case "consultar_saldo":
                    CheckBalance.model_validate_json(arguments_json or "{}")
                    return await ledger.get_balance(self._conn, self._user.id)
                case "buscar_contato":
                    return await self._search_contact(SearchContact.model_validate_json(arguments_json))
                case "enviar_pix":
                    return await self._send_pix(SendPix.model_validate_json(arguments_json))
                case "pagar_boleto":
                    return await self._pay_boleto(PayBoleto.model_validate_json(arguments_json))
                case _:
                    return ToolError(
                        codigo="ferramenta_desconhecida",
                        ferramenta=name,
                        detalhes=[
                            (
                                "Essa ferramenta não existe. Existem apenas consultar_saldo, buscar_contato, enviar_pix e "
                                "pagar_boleto. Para recusar ou pedir esclarecimento, use o campo acao da resposta final."
                            )
                        ],
                    )
        except ValidationError as exc:
            return validation_error(name, exc)

    async def _search_contact(self, args: SearchContact) -> ContactsFound:
        contacts = await ledger.search_contacts(self._conn, args.termo, exclude_id=self._user.id)
        return ContactsFound(contatos=contacts)

    async def _send_pix(self, args: SendPix) -> ToolResult:
        resolved = await ledger.resolve_recipient(self._conn, args.destinatario, exclude_id=self._user.id)
        match resolved:
            case list():
                return UnresolvedRecipient(termo=args.destinatario, candidatos=resolved)
            case Account() if resolved.id == self._user.id:
                return ToolError(codigo="conta_propria", ferramenta="enviar_pix")
            case Account():
                recipient = resolved
        payload = PixPayload(
            chave_pix=recipient.chave_pix,
            valor_centavos=args.valor_centavos,
            descricao=args.descricao,
            nome_recebedor=recipient.nome,
            conta_recebedor_id=recipient.id,
        )
        return await self._propose(payload, recipient_name=recipient.nome)

    async def _pay_boleto(self, args: PayBoleto) -> IntentProposal:
        payload = BoletoPayload(
            linha_digitavel=args.linha_digitavel,
            valor_centavos=args.valor_centavos,
            beneficiario=args.beneficiario,
        )
        return await self._propose(payload, recipient_name=args.beneficiario)

    async def _propose(self, payload: PixPayload | BoletoPayload, recipient_name: str) -> IntentProposal:
        intent = await ledger.create_intent(self._conn, self._user.id, payload)
        self.context.created_intent = intent.id
        return IntentProposal(
            intent_id=intent.id,
            tipo=payload.tipo,
            valor_centavos=payload.valor_centavos,
            destinatario=recipient_name,
            expira_em=intent.expira_em,
            observacao="Intent pendente; nada foi executado. O sistema pedirá a confirmação ao usuário.",
        )
