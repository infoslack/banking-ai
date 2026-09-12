"""Contrato entre um canal (web, WhatsApp, Telegram) e o agente.

Um adaptador de canal só precisa: obter/criar a sessão do usuário,
entregar mensagens ao `Processor` e renderizar a `AgentReply`.
"""

from typing import Literal, Protocol

from pydantic import BaseModel

from banking_ai.events import Emitter
from banking_ai.models.reply import AgentReply, ExtractedDocument
from banking_ai.session import Session


class TextMessage(BaseModel):
    kind: Literal["text"] = "text"
    text: str
    source: Literal["keyboard", "audio"] = "keyboard"


class DocumentMessage(BaseModel):
    kind: Literal["document"] = "document"
    document: ExtractedDocument


class Processor(Protocol):
    async def handle(self, session: Session, text: str, emit: Emitter) -> AgentReply: ...

    async def handle_document(self, session: Session, document: ExtractedDocument, emit: Emitter) -> AgentReply: ...
