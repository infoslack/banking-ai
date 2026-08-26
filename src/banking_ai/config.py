"""Configuração via variáveis de ambiente (.env em desenvolvimento)."""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    groq_api_key: SecretStr
    groq_model: str = "openai/gpt-oss-120b"
    groq_vision_model: str = "qwen/qwen3.6-27b"
    groq_whisper_model: str = "whisper-large-v3-turbo"
    groq_guard_model: str = "meta-llama/llama-prompt-guard-2-86m"
    database_url: str = "postgresql://banking:banking@localhost:5433/banking"
    demo_user_pix_key: str = "daniel@email.com"
    injection_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_tool_iterations: int = Field(default=6, ge=1, le=20)
    app_host: str = "0.0.0.0"
    app_port: int = 8000


def load_settings() -> Settings:
    return Settings()
