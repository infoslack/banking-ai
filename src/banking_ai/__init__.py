"""Demo de banking conversacional com IA: o LLM nunca toca no dinheiro."""


def main() -> None:
    import uvicorn

    from banking_ai.config import load_settings

    settings = load_settings()
    uvicorn.run("banking_ai.app:app", host=settings.app_host, port=settings.app_port)
