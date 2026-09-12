"""Templates com dados do banco. Números, chaves e listas de contatos nunca saem da prosa do LLM."""

from banking_ai.models.domain import Balance, BoletoPayload, ConfirmationResult, Intent, PixPayload
from banking_ai.models.reply import RefusalReason
from banking_ai.models.tools import (
    BOLETO_LIMIT_CENTS,
    PIX_LIMIT_CENTS,
    ToolError,
    ToolResult,
    UnresolvedRecipient,
)

CANAIS_OFICIAIS = "Para isso, use o app ou os canais oficiais do banco."


def format_brl(cents: int) -> str:
    whole, rest = divmod(abs(cents), 100)
    thousands = f"{whole:,}".replace(",", ".")
    sign = "-" if cents < 0 else ""
    return f"{sign}R$ {thousands},{rest:02d}"


def mask_pix_key(key: str) -> str:
    if len(key) <= 6:
        return key
    return f"{key[:3]}…{key[-2:]}"


def balance_text(balance: Balance) -> str:
    return f"Seu saldo é {format_brl(balance.saldo_centavos)}."


def recipient_text(result: UnresolvedRecipient) -> str:
    if not result.candidatos:
        return f"Não encontrei nenhum contato para “{result.termo}”. Me passa a chave Pix?"
    lines = [
        f"{number}) {contact.nome} (chave {mask_pix_key(contact.chave_pix)})"
        for number, contact in enumerate(result.candidatos, start=1)
    ]
    return f"Encontrei {len(lines)} contatos para “{result.termo}”:\n" + "\n".join(lines) + "\nQual deles?"


def error_text(error: ToolError) -> str | None:
    match error.codigo:
        case "limite":
            limit = BOLETO_LIMIT_CENTS if error.ferramenta == "pagar_boleto" else PIX_LIMIT_CENTS
            operation = "boleto" if error.ferramenta == "pagar_boleto" else "Pix"
            return f"Não dá: o limite por {operation} é {format_brl(limit)} por operação."
        case "conta_propria":
            return "Não dá para mandar Pix para a sua própria conta."
        case "argumentos_invalidos" | "ferramenta_desconhecida":
            return None


def tool_result_text(result: ToolResult) -> str | None:
    """Texto determinístico para um tool result terminal, ou None quando o LLM deve responder."""
    match result:
        case Balance():
            return balance_text(result)
        case UnresolvedRecipient():
            return recipient_text(result)
        case ToolError():
            return error_text(result)
        case _:
            return None


def confirmation_text(intent: Intent) -> str:
    match intent.payload:
        case PixPayload() as pix:
            return (
                f"Confirma Pix de {format_brl(pix.valor_centavos)} para {pix.nome_recebedor} "
                f"(chave {mask_pix_key(pix.chave_pix)})? Responda sim ou não."
            )
        case BoletoPayload() as boleto:
            return (
                f"Confirma pagamento de {format_brl(boleto.valor_centavos)} do boleto de "
                f"{boleto.beneficiario} (linha …{boleto.linha_digitavel[-6:]})? Responda sim ou não."
            )


def execution_text(result: ConfirmationResult) -> str:
    amount = format_brl(result.valor_centavos or 0)
    recipient = result.destinatario or "o destinatário"
    match result.status:
        case "executada":
            return f"Feito! {amount} enviados para {recipient}. Seu saldo agora é {format_brl(result.saldo_centavos or 0)}."
        case "ja_executada":
            return f"Essa transferência de {amount} para {recipient} já tinha sido executada; nada foi enviado de novo."
        case "expirada":
            return "A confirmação expirou (5 minutos). Se quiser, me peça de novo."
        case "cancelada":
            return "Essa operação já estava cancelada."
        case "saldo_insuficiente":
            return (
                f"Saldo insuficiente: você tem {format_brl(result.saldo_centavos or 0)} e a operação era de "
                f"{amount}. Cancelei a intenção."
            )
        case "nao_encontrada":
            return "Não encontrei essa operação pendente."


def cancellation_text() -> str:
    return "Cancelado. Nada foi enviado."


def refusal_text(reason: RefusalReason | None) -> str:
    """Recusas têm texto fixo por motivo: o modelo só escolhe o motivo."""
    match reason:
        case "fora_do_escopo":
            return "Eu só cuido da sua conta: saldo, extrato, Pix e boletos. Sobre outros assuntos não consigo ajudar."
        case "atendimento_humano":
            return f"Não faço atendimento humano por aqui. {CANAIS_OFICIAIS}"
        case "conta_de_terceiros":
            return "Só consigo movimentar e consultar a sua própria conta; não dá para agir sobre a conta de outra pessoa."
        case "dado_sensivel":
            return "Não mostro nem envio dados sensíveis (CPF, senhas, número de cartão) por aqui."
        case "nao_suportado" | None:
            return f"Operação não suportada por este assistente: ele faz saldo, extrato, Pix e pagamento de boleto. {CANAIS_OFICIAIS}"
