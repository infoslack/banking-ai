from datetime import UTC, datetime
from uuid import uuid4

from banking_ai.models import Balance, ConfirmationResult, Contact, Intent, PixPayload, ToolError, UnresolvedRecipient
from banking_ai.render import (
    confirmation_text,
    execution_text,
    format_brl,
    mask_pix_key,
    refusal_text,
    tool_result_text,
)


def test_format_brl() -> None:
    assert format_brl(5000) == "R$ 50,00"
    assert format_brl(50) == "R$ 0,50"
    assert format_brl(123456) == "R$ 1.234,56"
    assert format_brl(0) == "R$ 0,00"
    assert format_brl(-250000) == "-R$ 2.500,00"


def test_mask_pix_key() -> None:
    assert mask_pix_key("+5511999990001") == "+55…01"
    assert mask_pix_key("abc") == "abc"


def test_confirmation_uses_database_data_not_prose() -> None:
    now = datetime.now(UTC)
    intent = Intent(
        id=uuid4(),
        user_id=uuid4(),
        tipo="pix",
        payload=PixPayload(chave_pix="+5511999990001", valor_centavos=5000, nome_recebedor="Alberto Souza", conta_recebedor_id=uuid4()),
        status="pendente",
        criada_em=now,
        expira_em=now,
    )
    assert confirmation_text(intent) == "Confirma Pix de R$ 50,00 para Alberto Souza (chave +55…01)? Responda sim ou não."


def test_duplicate_execution_says_nothing_was_sent_again() -> None:
    result = ConfirmationResult(status="ja_executada", intent_id=uuid4(), valor_centavos=5000, destinatario="Alberto Souza")
    assert "já tinha sido executada" in execution_text(result)


def test_balance_by_template() -> None:
    assert tool_result_text(Balance(conta_id=uuid4(), saldo_centavos=245000)) == "Seu saldo é R$ 2.450,00."


def test_ambiguity_lists_candidates_by_template() -> None:
    text = tool_result_text(
        UnresolvedRecipient(
            termo="alberto",
            candidatos=[
                Contact(nome="Alberto Pereira", chave_pix="alberto.pereira@email.com"),
                Contact(nome="Alberto Souza", chave_pix="+5511999990001"),
            ],
        )
    )
    assert text == "Encontrei 2 contatos para “alberto”:\n1) Alberto Pereira (chave alb…om)\n2) Alberto Souza (chave +55…01)\nQual deles?"
    assert tool_result_text(UnresolvedRecipient(termo="fulana", candidatos=[])) == "Não encontrei nenhum contato para “fulana”. Me passa a chave Pix?"


def test_known_errors_by_template_and_unknown_left_to_llm() -> None:
    assert tool_result_text(ToolError(codigo="limite", ferramenta="enviar_pix")) == "Não dá: o limite por Pix é R$ 5.000,00 por operação."
    assert tool_result_text(ToolError(codigo="limite", ferramenta="pagar_boleto")) == "Não dá: o limite por boleto é R$ 10.000,00 por operação."
    assert tool_result_text(ToolError(codigo="conta_propria", ferramenta="enviar_pix")) == "Não dá para mandar Pix para a sua própria conta."
    assert tool_result_text(ToolError(codigo="argumentos_invalidos", ferramenta="enviar_pix", detalhes=["x"])) is None


def test_refusal_text_is_fixed_per_reason() -> None:
    assert refusal_text("conta_de_terceiros").startswith("Só consigo movimentar e consultar a sua própria conta")
    assert refusal_text("atendimento_humano").startswith("Não faço atendimento humano")
    assert refusal_text("fora_do_escopo").startswith("Eu só cuido da sua conta")
    assert refusal_text("dado_sensivel").startswith("Não mostro nem envio dados sensíveis")
    assert refusal_text("nao_suportado").startswith("Operação não suportada")
    assert refusal_text(None) == refusal_text("nao_suportado")
