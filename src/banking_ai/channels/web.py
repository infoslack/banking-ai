"""Canal web estilo WhatsApp: WebSocket para a conversa + log de tool calls."""

from typing import Annotated, Literal

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from groq import RateLimitError
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from banking_ai.channels.base import DocumentMessage, Processor, TextMessage
from banking_ai.events import LogEvent
from banking_ai.models.domain import Account
from banking_ai.models.reply import AgentReply
from banking_ai.session import SessionManager

ClientMessage = Annotated[TextMessage | DocumentMessage, Field(discriminator="kind")]
CLIENT_ADAPTER: TypeAdapter[TextMessage | DocumentMessage] = TypeAdapter(ClientMessage)


class EventFrame(BaseModel):
    kind: Literal["event"] = "event"
    event: LogEvent


class ReplyFrame(BaseModel):
    kind: Literal["reply"] = "reply"
    reply: AgentReply


class ErrorFrame(BaseModel):
    kind: Literal["error"] = "error"
    message: str


class WebChannel:
    def __init__(self, processor: Processor, sessions: SessionManager, user: Account) -> None:
        self._processor = processor
        self._sessions = sessions
        self._user = user
        self.router = APIRouter()
        self.router.add_api_websocket_route("/ws/{session_id}", self.websocket)

    async def websocket(self, websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        session = self._sessions.get_or_create(session_id, self._user)

        async def emit(event: LogEvent) -> None:
            await websocket.send_text(EventFrame(event=event).model_dump_json())

        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    message = CLIENT_ADAPTER.validate_json(raw)
                except ValidationError as exc:
                    await websocket.send_text(ErrorFrame(message=f"mensagem inválida: {exc.error_count()} erro(s)").model_dump_json())
                    continue
                await emit(LogEvent(kind="typing", title="digitando"))
                try:
                    match message:
                        case TextMessage():
                            reply = await self._processor.handle(session, message.text, emit)
                        case DocumentMessage():
                            reply = await self._processor.handle_document(session, message.document, emit)
                except RateLimitError as exc:
                    await emit(LogEvent(kind="error", title="rate limit da Groq (429)", detail=str(exc), ok=False))
                    reply = AgentReply(
                        acao="responder", mensagem="Estou recebendo muitas mensagens agora. Espera alguns segundos e manda de novo?"
                    )
                except Exception as exc:  # noqa: BLE001 - intencional: a demo não pode derrubar o socket, o erro vai para o painel
                    await emit(LogEvent(kind="error", title=f"falha no agente: {type(exc).__name__}", detail=str(exc), ok=False))
                    reply = AgentReply(acao="responder", mensagem="Tive um problema aqui do meu lado. Pode tentar de novo?")
                await websocket.send_text(ReplyFrame(reply=reply).model_dump_json())
        except WebSocketDisconnect:
            return
