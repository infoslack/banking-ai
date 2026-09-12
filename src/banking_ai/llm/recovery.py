"""Recuperação dos 400 da Groq: aproveita o `failed_generation` antes de desistir."""

import json

from groq import BadRequestError
from groq.types.chat import ChatCompletionMessage
from groq.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
from pydantic import BaseModel, Field, JsonValue, TypeAdapter, ValidationError, field_validator

from banking_ai.models.reply import AgentReply

RECOVERED_PREFIX = "recovered-"


class GroqErrorDetail(BaseModel):
    code: str | None = None
    message: str = ""
    failed_generation: str | None = None


class GroqError(BaseModel):
    error: GroqErrorDetail


class RecoveredToolCall(BaseModel):
    """Tool call que o parser da Groq rejeitou mas que ainda dá para validar do nosso lado."""

    name: str = Field(min_length=1)
    arguments: str

    @field_validator("arguments", mode="before")
    @classmethod
    def _as_string(cls, value: JsonValue) -> str:
        match value:
            case str():
                return value
            case _:
                return json.dumps(value, ensure_ascii=False)


class ToolRequested(Exception):
    """O modelo quis uma ferramenta na fase estruturada; carrega a chamada para o agente executar."""

    def __init__(self, message: ChatCompletionMessage) -> None:
        super().__init__("modelo solicitou ferramenta na fase estruturada")
        self.message = message


JSON_LIST_ADAPTER: TypeAdapter[list[JsonValue]] = TypeAdapter(list[JsonValue])


def groq_error(exc: BadRequestError) -> GroqErrorDetail:
    try:
        return GroqError.model_validate(exc.response.json()).error
    except (ValidationError, ValueError):
        return GroqErrorDetail(message=str(exc))


def recover_tool_call(detail: GroqErrorDetail, index: int) -> ChatCompletionMessage:
    """Devolve a tool call recuperável, ou uma mensagem vazia ("acabaram as ferramentas")."""
    if detail.failed_generation is None:
        return ChatCompletionMessage(role="assistant", content=None)
    try:
        call = RecoveredToolCall.model_validate_json(detail.failed_generation)
    except ValidationError:
        return ChatCompletionMessage(role="assistant", content=None)
    return ChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ChatCompletionMessageToolCall(
                id=f"{RECOVERED_PREFIX}{index}",
                type="function",
                function=Function(name=call.name, arguments=call.arguments),
            )
        ],
    )


def salvage_reply(detail: GroqErrorDetail) -> str | None:
    """Recupera um AgentReply válido de dentro de um `json_validate_failed`."""
    if detail.failed_generation is None:
        return None
    try:
        return AgentReply.model_validate_json(detail.failed_generation).model_dump_json()
    except ValidationError:
        pass
    try:
        items = JSON_LIST_ADAPTER.validate_json(detail.failed_generation)
    except ValidationError:
        return None
    for item in items:
        try:
            return AgentReply.model_validate(item).model_dump_json()
        except ValidationError:
            continue
    return None
