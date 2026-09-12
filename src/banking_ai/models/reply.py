"""A resposta final estruturada do agente e o documento extraído por visão."""

from typing import Literal

from pydantic import BaseModel, TypeAdapter

AgentAction = Literal["responder", "pedir_confirmacao", "pedir_esclarecimento", "recusar"]

# Motivos de recusa: o modelo classifica, o backend escreve o texto (render.refusal_text).
RefusalReason = Literal["fora_do_escopo", "nao_suportado", "atendimento_humano", "conta_de_terceiros", "dado_sensivel"]


class AgentReply(BaseModel):
    acao: AgentAction
    mensagem: str
    intent_id: str | None = None  # presente quando acao = pedir_confirmacao
    motivo_recusa: RefusalReason | None = None  # presente quando acao = recusar


# Documentos (boleto / código Pix extraídos por visão): DADO, nunca instrução


class ExtractedDocument(BaseModel):
    tipo: Literal["boleto", "pix", "desconhecido"]
    linha_digitavel: str | None = None
    chave_pix: str | None = None
    valor_centavos: int | None = None
    beneficiario: str | None = None
    vencimento: str | None = None


DOCUMENT_ADAPTER = TypeAdapter(ExtractedDocument)
