"""Guardrails AI: a camada probabilística (formato, injection, PII); as garantias duras são código e Postgres."""

import re
from collections.abc import Awaitable, Callable
from typing import Protocol

from groq.types.chat import ChatCompletionMessageParam
from guardrails import AsyncGuard, OnFailAction
from guardrails.errors import ValidationError as GuardrailsValidationError
from guardrails.validator_base import FailResult, PassResult, ValidationResult, Validator, register_validator
from pydantic import BaseModel

from banking_ai.models.reply import AgentReply

SCORE_KEY = "injection_score"

PII_PATTERNS = [
    ("CPF", re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")),
    ("CNPJ", re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b")),
    ("cartão", re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b")),
]


@register_validator(name="banking/sem-prompt-injection", data_type="string")
class NoPromptInjection(Validator):
    """Falha quando o score do classificador (passado em metadata) passa do limiar."""

    def __init__(self, threshold: float, on_fail: OnFailAction) -> None:
        super().__init__(on_fail=on_fail, threshold=threshold)
        self._threshold = threshold

    def validate(self, value: str, metadata: dict[str, float]) -> ValidationResult:
        score = metadata.get(SCORE_KEY, 1.0)
        if score >= self._threshold:
            return FailResult(error_message=f"prompt injection detectada (score {score:.3f})")
        return PassResult()


@register_validator(name="banking/sem-pii", data_type="string")
class NoPII(Validator):
    """Mascara CPF, CNPJ e cartão na mensagem que vai ao usuário."""

    def __init__(self, on_fail: OnFailAction) -> None:
        super().__init__(on_fail=on_fail)

    def validate(self, value: str, metadata: dict[str, float]) -> ValidationResult:
        found = [name for name, pattern in PII_PATTERNS if pattern.search(value)]
        if not found:
            return PassResult()
        masked = value
        for _name, pattern in PII_PATTERNS:
            masked = pattern.sub("[dado mascarado]", masked)
        return FailResult(error_message=f"PII na mensagem: {', '.join(found)}", fix_value=masked)


class InputGuardResult(BaseModel):
    approved: bool
    score: float
    reason: str = ""


class OutputGuardResult(BaseModel):
    validation_passed: bool
    validated_output: AgentReply | None = None
    error: str | None = None


class InvalidOutput(Exception):
    """A resposta do LLM não passou no guard de saída nem após o reask."""


class Reask(Protocol):
    """Reask no formato do Guardrails: `messages` keyword-only e **kwargs; o contexto real vem do callable."""

    async def __call__(self, *, messages: list[ChatCompletionMessageParam], **llm_params: float | str | None) -> str: ...


# O histórico com tool calls não cabe no modelo do Guardrails; vai pelo callable de reask.
PLACEHOLDER_MESSAGES: list[ChatCompletionMessageParam] = [
    {"role": "user", "content": "Gere a resposta final conforme o schema."}
]


class Guards:
    def __init__(self, scorer: Callable[[str], Awaitable[float]], threshold: float) -> None:
        self._scorer = scorer
        self._input = AsyncGuard().use(NoPromptInjection(threshold=threshold, on_fail=OnFailAction.EXCEPTION))
        self._output = AsyncGuard.for_pydantic(AgentReply).use(NoPII(on_fail=OnFailAction.FIX), on="$.mensagem")

    async def validate_input(self, text: str) -> InputGuardResult:
        score = await self._scorer(text)
        try:
            await self._input.validate(text, metadata={SCORE_KEY: score})
        except GuardrailsValidationError as exc:
            return InputGuardResult(approved=False, score=score, reason=str(exc))
        return InputGuardResult(approved=True, score=score)

    async def validate_output(self, raw: str, reask: Reask) -> AgentReply:
        outcome = await self._output.parse(raw, llm_api=reask, num_reasks=1, messages=PLACEHOLDER_MESSAGES)
        result = OutputGuardResult.model_validate(outcome, from_attributes=True)
        if not result.validation_passed or result.validated_output is None:
            raise InvalidOutput(result.error or "saída reprovada pelo guard")
        return result.validated_output
