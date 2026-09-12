"""Cliente Groq em duas fases: loop de tools, depois resposta final com schema strict
(a Groq recusa `tools` e `response_format` na mesma chamada)."""

import base64
from typing import Protocol

from groq import AsyncGroq, BadRequestError
from groq.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from pydantic import ValidationError

from banking_ai.config import Settings
from banking_ai.llm.formats import DOCUMENT_FORMAT, REPLY_FORMAT
from banking_ai.llm.recovery import GroqErrorDetail, ToolRequested, groq_error, recover_tool_call, salvage_reply
from banking_ai.models.reply import DOCUMENT_ADAPTER, ExtractedDocument
from banking_ai.prompts import JSON_ONLY_NUDGE, NO_TOOLS_NUDGE, VISION_PROMPT


class LLM(Protocol):
    """O que o agente precisa de um modelo; os testes usam um fake em vez de mockar módulos."""

    @property
    def model(self) -> str: ...

    async def chat_with_tools(
        self, messages: list[ChatCompletionMessageParam], tools: list[ChatCompletionToolParam]
    ) -> ChatCompletionMessage: ...

    async def structured_reply(self, messages: list[ChatCompletionMessageParam]) -> str: ...


class UnreadableDocument(Exception):
    """A visão não devolveu um documento conforme o schema."""


def assistant_message(msg: ChatCompletionMessage) -> ChatCompletionAssistantMessageParam:
    """Converte a resposta do modelo no formato aceito de volta no histórico."""
    param: ChatCompletionAssistantMessageParam = {"role": "assistant", "content": msg.content}
    if msg.tool_calls:
        calls: list[ChatCompletionMessageToolCallParam] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in msg.tool_calls
        ]
        param["tool_calls"] = calls
    return param


class GroqLLM:
    def __init__(self, settings: Settings) -> None:
        # Tier on-demand da Groq tem TPM baixo; o SDK respeita o retry-after do 429.
        self._client = AsyncGroq(api_key=settings.groq_api_key.get_secret_value(), max_retries=6)
        self._model = settings.groq_model
        self._vision_model = settings.groq_vision_model
        self._whisper_model = settings.groq_whisper_model
        self._guard_model = settings.groq_guard_model
        self._recovered = 0

    @property
    def model(self) -> str:
        return self._model

    async def chat_with_tools(
        self, messages: list[ChatCompletionMessageParam], tools: list[ChatCompletionToolParam]
    ) -> ChatCompletionMessage:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0,
            )
        except BadRequestError as exc:
            return recover_tool_call(self._recoverable(exc), self._recovered)
        return response.choices[0].message

    async def structured_reply(self, messages: list[ChatCompletionMessageParam]) -> str:
        """Resposta final no schema de AgentReply; um 400 recuperável vira tool call, resposta aproveitada ou repetição."""
        try:
            return await self._structured_completion(messages)
        except BadRequestError as exc:
            detail = self._recoverable(exc)
            recovered = recover_tool_call(detail, self._recovered)
            if recovered.tool_calls:
                raise ToolRequested(recovered) from exc
            salvaged = salvage_reply(detail)
            if salvaged is not None:
                return salvaged
            return await self._structured_completion([*messages, JSON_ONLY_NUDGE, NO_TOOLS_NUDGE])

    def _recoverable(self, exc: BadRequestError) -> GroqErrorDetail:
        """400 sem `failed_generation` não tem o que aproveitar: sobe como veio."""
        detail = groq_error(exc)
        if detail.failed_generation is None:
            raise exc
        self._recovered += 1
        return detail

    async def _structured_completion(self, messages: list[ChatCompletionMessageParam]) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            response_format=REPLY_FORMAT,
            temperature=0,
        )
        return response.choices[0].message.content or ""

    async def score_injection(self, text: str) -> float:
        """Score de injection do Prompt Guard; qualquer falha devolve 1.0 (falha fechado)."""
        request: ChatCompletionUserMessageParam = {"role": "user", "content": text}
        response = await self._client.chat.completions.create(model=self._guard_model, messages=[request])
        content = (response.choices[0].message.content or "").strip()
        try:
            return min(max(float(content), 0.0), 1.0)
        except ValueError:
            return 1.0

    async def transcribe(self, audio: bytes, filename: str) -> str:
        transcription = await self._client.audio.transcriptions.create(
            model=self._whisper_model,
            file=(filename, audio),
            language="pt",
            temperature=0,
        )
        return transcription.text.strip()

    async def read_document(self, image: bytes, mime_type: str) -> ExtractedDocument:
        data_url = f"data:{mime_type};base64,{base64.b64encode(image).decode('ascii')}"
        request: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": [
                {"type": "text", "text": VISION_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
        response = await self._client.chat.completions.create(
            model=self._vision_model,
            messages=[request],
            response_format=DOCUMENT_FORMAT,
            temperature=0,
        )
        raw = response.choices[0].message.content or ""
        try:
            return DOCUMENT_ADAPTER.validate_json(raw)
        except ValidationError as exc:
            raise UnreadableDocument(str(exc)) from exc
