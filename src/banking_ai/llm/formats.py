"""Formatos de saída estruturada no modo strict da Groq."""

from groq.types.chat.completion_create_params import ResponseFormatResponseFormatJsonSchema
from pydantic import BaseModel
from pydantic.json_schema import JsonSchemaValue

from banking_ai.models.reply import AgentReply, ExtractedDocument


def strict_flat_schema(model: type[BaseModel]) -> JsonSchemaValue:
    """Strict da Groq exige: todo campo obrigatório (opcional = null), objeto fechado, sem defaults."""
    schema = model.model_json_schema()
    properties = schema["properties"]
    for prop in properties.values():
        prop.pop("default", None)
    schema["required"] = list(properties)
    schema["additionalProperties"] = False
    return schema


REPLY_FORMAT: ResponseFormatResponseFormatJsonSchema = {
    "type": "json_schema",
    "json_schema": {
        "name": "resposta_agente",
        "strict": True,
        "schema": strict_flat_schema(AgentReply),
    },
}

DOCUMENT_FORMAT: ResponseFormatResponseFormatJsonSchema = {
    "type": "json_schema",
    "json_schema": {
        "name": "documento_extraido",
        "schema": ExtractedDocument.model_json_schema(),
    },
}
