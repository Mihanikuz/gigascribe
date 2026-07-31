from __future__ import annotations

from .base import LLMProvider, extract_json


def create_provider(*, engine: str, model_path: str, context_length: int) -> LLMProvider:
    if engine == "llama_cpp":
        from .llama_cpp import LlamaCppProvider
        return LlamaCppProvider(model_path=model_path, context_length=context_length)
    if engine == "ollama":
        from .ollama import OllamaProvider
        return OllamaProvider(model_path=model_path, context_length=context_length)
    raise ValueError(f"Unknown protocol engine: {engine}")


__all__ = ["LLMProvider", "extract_json", "create_provider"]
