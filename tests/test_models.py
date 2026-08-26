"""O schema é o primeiro guardrail."""

import pytest
from pydantic import ValidationError

from banking_ai.llm import REPLY_FORMAT, strict_flat_schema
from banking_ai.models import PIX_LIMIT_CENTS, AgentReply, CheckBalance, PayBoleto, SendPix
from banking_ai.tools import TOOL_SPECS


def test_pix_above_hard_limit_is_rejected_by_schema() -> None:
    with pytest.raises(ValidationError) as exc:
        SendPix(destinatario="maria", valor_centavos=100_000_000)  # "1 milhão"
    assert "less than or equal" in str(exc.value)
    assert PIX_LIMIT_CENTS == 500_000


def test_pix_requires_positive_integer_cents() -> None:
    with pytest.raises(ValidationError):
        SendPix(destinatario="maria", valor_centavos=0)
    with pytest.raises(ValidationError):
        SendPix.model_validate_json('{"destinatario": "maria", "valor_centavos": 50.5}')


def test_recipient_is_free_text_resolved_by_backend() -> None:
    assert SendPix(destinatario="alberto", valor_centavos=100).destinatario == "alberto"
    with pytest.raises(ValidationError):
        SendPix(destinatario="a", valor_centavos=100)


def test_user_id_and_pix_key_are_never_tool_parameters() -> None:
    with pytest.raises(ValidationError):
        SendPix.model_validate({"destinatario": "maria", "valor_centavos": 100, "user_id": "x"})
    with pytest.raises(ValidationError):
        SendPix.model_validate({"destinatario": "maria", "valor_centavos": 100, "chave_pix": "11122233344"})
    with pytest.raises(ValidationError):
        CheckBalance.model_validate({"user_id": "10000000-0000-0000-0000-000000000002"})


def test_boleto_requires_digits_only_line() -> None:
    PayBoleto(linha_digitavel="2" * 47, valor_centavos=100, beneficiario="Energia")
    with pytest.raises(ValidationError):
        PayBoleto(linha_digitavel="23793.38128 60000", valor_centavos=100, beneficiario="Energia")


def test_tool_specs_come_from_pydantic_schema() -> None:
    names = [spec["function"]["name"] for spec in TOOL_SPECS]
    assert names == ["consultar_saldo", "buscar_contato", "enviar_pix", "pagar_boleto"]
    assert "confirmar_intent" not in names  # executar é do sistema, não do modelo
    send = next(spec for spec in TOOL_SPECS if spec["function"]["name"] == "enviar_pix")
    schema = SendPix.model_json_schema()
    assert send["function"]["parameters"] == schema
    assert schema["properties"]["valor_centavos"]["maximum"] == PIX_LIMIT_CENTS
    assert schema["additionalProperties"] is False


def test_strict_reply_schema_meets_groq_rules() -> None:
    schema = strict_flat_schema(AgentReply)
    assert schema["additionalProperties"] is False
    assert sorted(schema["required"]) == ["acao", "intent_id", "mensagem", "motivo_recusa"]
    assert "default" not in schema["properties"]["intent_id"]
    assert REPLY_FORMAT["json_schema"]["strict"] is True


def test_salvage_reply_from_groq_json_validate_failed() -> None:
    from banking_ai.llm import GroqErrorDetail, salvage_reply

    failed = (
        '[\n{"acao":"recusar","intent_id":null,"mensagem":"Encaminho sua solicitação.","motivo_recusa":"atendimento_humano"},'
        '\n\n"Check schema: acao enum includes recusar", "ok.",\n\n"Return only JSON.",'
        '{"acao":"recusar","intent_id":null,"mensagem":"Encaminho sua solicitação.","motivo_recusa":"atendimento_humano"}\n]'
    )
    salvaged = salvage_reply(GroqErrorDetail(code="json_validate_failed", failed_generation=failed))
    assert salvaged is not None
    assert AgentReply.model_validate_json(salvaged).motivo_recusa == "atendimento_humano"
    assert salvage_reply(GroqErrorDetail(code="json_validate_failed", failed_generation="[\"só texto\"]")) is None
    assert salvage_reply(GroqErrorDetail(code="json_validate_failed", failed_generation="não é json")) is None
    assert salvage_reply(GroqErrorDetail(code="json_validate_failed", failed_generation=None)) is None
