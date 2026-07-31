"""Ollama provider: talks to a local Ollama daemon over HTTP.

Uses ``requests`` (already a core GigaScribe dependency) rather than adding
a new package -- Ollama itself is external local infrastructure the admin
runs and points this at, not something this module downloads or manages.
Ollama applies the model's own chat template server-side whenever a
`system` prompt is supplied (i.e. non-`raw` mode), so no template handling
is needed on this side the way llama.cpp requires.
"""
from __future__ import annotations

import asyncio
import os
from typing import Optional

from .base import LLMProvider

_DEFAULT_SYSTEM_PROMPT = "Ты — ассистент для протоколирования деловых встреч. Отвечай строго на русском языке."


class OllamaProvider(LLMProvider):
    def __init__(self, *, model_path: str, context_length: int, base_url: Optional[str] = None,
                 keep_alive: Optional[str] = None):
        # For Ollama, model_path is the Ollama model tag (e.g. "qwen3:14b"),
        # not a filesystem path. base_url is admin-only configuration (set
        # via the protocol model's ollama_url column, never accepted from a
        # regular user/request) -- see server.py's admin-only model params route.
        super().__init__(model_path=model_path, context_length=context_length)
        self.base_url = base_url or os.getenv("GIGASCRIBE_OLLAMA_URL", "http://127.0.0.1:11434")
        self.keep_alive = keep_alive

    async def load(self) -> None:
        if self.loaded:
            return
        loop = asyncio.get_running_loop()

        def _warm_up():
            import requests
            payload: dict = {"model": self.model_path, "prompt": "", "stream": False}
            if self.keep_alive:
                payload["keep_alive"] = self.keep_alive
            resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=60)
            resp.raise_for_status()

        await loop.run_in_executor(None, _warm_up)
        self.loaded = True

    async def generate(self, prompt: str, *, temperature: float, max_tokens: int,
                        system_prompt: str | None = None) -> str:
        if not self.loaded:
            raise RuntimeError("OllamaProvider.generate() called before load()")
        loop = asyncio.get_running_loop()

        def _run():
            import requests
            payload = {
                "model": self.model_path, "prompt": prompt, "system": system_prompt or _DEFAULT_SYSTEM_PROMPT,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": self.context_length},
            }
            if self.keep_alive:
                payload["keep_alive"] = self.keep_alive
            resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=600)
            resp.raise_for_status()
            return resp.json().get("response", "")

        return await loop.run_in_executor(None, _run)

    async def unload(self) -> None:
        if not self.loaded:
            return
        loop = asyncio.get_running_loop()

        def _unload():
            import requests
            try:
                requests.post(f"{self.base_url}/api/generate",
                               json={"model": self.model_path, "prompt": "", "keep_alive": 0}, timeout=30)
            except Exception:
                pass  # best-effort; Ollama will evict it on its own idle timeout regardless

        await loop.run_in_executor(None, _unload)
        self.loaded = False
