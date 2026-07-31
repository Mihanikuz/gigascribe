"""llama.cpp provider (GGUF models, local inference, no network)."""
from __future__ import annotations

import asyncio
import gc

from .base import LLMProvider


class LlamaCppProvider(LLMProvider):
    def __init__(self, *, model_path: str, context_length: int, n_gpu_layers: int = -1):
        super().__init__(model_path=model_path, context_length=context_length)
        self.n_gpu_layers = n_gpu_layers
        self._llm = None

    async def load(self) -> None:
        if self.loaded:
            return
        loop = asyncio.get_running_loop()

        def _load():
            from llama_cpp import Llama  # heavy import; only required when this provider is used
            return Llama(
                model_path=self.model_path,
                n_ctx=self.context_length,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
            )

        self._llm = await loop.run_in_executor(None, _load)
        self.loaded = True

    async def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        if not self.loaded or self._llm is None:
            raise RuntimeError("LlamaCppProvider.generate() called before load()")
        loop = asyncio.get_running_loop()

        def _run():
            out = self._llm(prompt, max_tokens=max_tokens, temperature=temperature)
            return out["choices"][0]["text"]

        return await loop.run_in_executor(None, _run)

    async def unload(self) -> None:
        self._llm = None
        self.loaded = False
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
