"""Engine-agnostic LLM provider interface.

Business logic (chunking, prompts, glossary, merging) never talks to
llama.cpp or Ollama directly -- it only calls this interface, so swapping
or adding an engine never touches service.py.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

from ..schemas import ProtocolValidationError

# Defense in depth against a runaway/misbehaving model: even though
# max_output_tokens is passed to the engine, also cap the raw response we'll
# ever try to parse.
MAX_RESPONSE_CHARS = 200_000

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(raw: str) -> dict[str, Any]:
    """Best-effort extraction of a JSON object from an LLM response that may
    be wrapped in markdown fences or have leading/trailing prose."""
    if len(raw) > MAX_RESPONSE_CHARS:
        raise ProtocolValidationError(f"LLM response too large ({len(raw)} chars)")
    candidates = [raw]
    fence_match = _FENCE_RE.search(raw)
    if fence_match:
        candidates.insert(0, fence_match.group(1))
    for candidate in candidates:
        text = candidate.strip()
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end < start:
            continue
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            continue
    raise ProtocolValidationError("No valid JSON object found in LLM response")


class LLMProvider(ABC):
    def __init__(self, *, model_path: str, context_length: int):
        self.model_path = model_path
        self.context_length = context_length
        self.loaded = False

    @abstractmethod
    async def load(self) -> None:
        """Load the model into (V)RAM. Idempotent if already loaded."""

    @abstractmethod
    async def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        """Return the raw text completion for prompt."""

    @abstractmethod
    async def unload(self) -> None:
        """Release all GPU/CPU memory held by this provider. Idempotent."""

    async def generate_json(self, prompt: str, *, temperature: float, max_tokens: int,
                             max_retries: int = 2) -> dict[str, Any]:
        last_error: Exception | None = None
        attempt_prompt = prompt
        for attempt in range(max_retries + 1):
            raw = await self.generate(attempt_prompt, temperature=temperature, max_tokens=max_tokens)
            try:
                return extract_json(raw)
            except ProtocolValidationError as exc:
                last_error = exc
                attempt_prompt = (
                    prompt
                    + "\n\nТвой предыдущий ответ не был валидным JSON "
                    + f"({exc}). Верни ТОЛЬКО валидный JSON-объект, без пояснений и без markdown-разметки."
                )
        raise ProtocolValidationError(f"LLM did not return valid JSON after {max_retries + 1} attempt(s): {last_error}")
