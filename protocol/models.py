"""Static registry of the LLM protocol models GigaScribe supports.

This only describes *what* is supported (id, label, engine, context size,
etc.); actual install state (path, size, installed flag) lives in the
protocol_models table (see store.py) and is seeded from this registry on
first use. Source repo_id/filename are intentionally left for the admin to
fill in (see README): the exact current GGUF Q4_K_M quantization filenames
on Hugging Face change over time and this module must not silently point at
a URL nobody has verified.
"""
from __future__ import annotations

from .schemas import ModelSpec

SUPPORTED_PROTOCOL_MODELS: dict[str, ModelSpec] = {
    "qwen3-14b": ModelSpec(
        id="qwen3-14b", label="Qwen3-14B (Q4_K_M)", engine="llama_cpp",
        format="gguf", quantization="Q4_K_M", context_length=32768,
        temperature=0.2, max_output_tokens=2048,
    ),
    "qwen3-8b": ModelSpec(
        id="qwen3-8b", label="Qwen3-8B (Q4_K_M)", engine="llama_cpp",
        format="gguf", quantization="Q4_K_M", context_length=32768,
        temperature=0.2, max_output_tokens=2048,
    ),
    "gemma3-12b-it": ModelSpec(
        id="gemma3-12b-it", label="Gemma 3 12B IT (Q4_K_M)", engine="llama_cpp",
        format="gguf", quantization="Q4_K_M", context_length=8192,
        temperature=0.2, max_output_tokens=2048,
    ),
    "ministral3-8b-instruct": ModelSpec(
        id="ministral3-8b-instruct", label="Ministral 3 8B Instruct (Q4_K_M)", engine="llama_cpp",
        format="gguf", quantization="Q4_K_M", context_length=32768,
        temperature=0.2, max_output_tokens=2048,
    ),
}


def get_model_spec(model_id: str) -> ModelSpec | None:
    return SUPPORTED_PROTOCOL_MODELS.get(model_id)
