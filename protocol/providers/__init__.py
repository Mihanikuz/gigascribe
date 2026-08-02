from __future__ import annotations

from typing import Any

from .base import LLMProvider, extract_json


def create_provider(*, engine: str, model_path: str, context_length: int, **engine_config: Any) -> LLMProvider:
    if engine == "llama_cpp":
        from .llama_cpp import LlamaCppProvider
        return LlamaCppProvider(
            model_path=model_path, context_length=context_length,
            n_gpu_layers=engine_config.get("n_gpu_layers") if engine_config.get("n_gpu_layers") is not None else -1,
            no_think=bool(engine_config.get("no_think")),
            extra_launch_params=engine_config.get("extra_launch_params") or {},
        )
    if engine == "ollama":
        from .ollama import OllamaProvider
        return OllamaProvider(
            model_path=model_path, context_length=context_length,
            base_url=engine_config.get("ollama_url"), keep_alive=engine_config.get("ollama_keep_alive"),
        )
    raise ValueError(f"Unknown protocol engine: {engine}")


def validate_engine_config(*, engine: str, model_path: str, **engine_config: Any) -> None:
    """Cheap, no-network sanity check run before a job is created (item 4):
    catches an obviously broken configuration early rather than failing deep
    inside the pipeline after the GPU has already been claimed."""
    if engine not in ("llama_cpp", "ollama"):
        raise ValueError(f"Unknown protocol engine: {engine}")
    if not model_path:
        raise ValueError("Model path/tag is not configured")
    if engine == "llama_cpp":
        from pathlib import Path
        if not Path(model_path).is_file():
            raise ValueError(f"GGUF file not found: {model_path}")
    if engine == "ollama":
        url = engine_config.get("ollama_url")
        if url and not (url.startswith("http://") or url.startswith("https://")):
            raise ValueError("ollama_url must start with http:// or https://")


__all__ = ["LLMProvider", "extract_json", "create_provider", "validate_engine_config"]
