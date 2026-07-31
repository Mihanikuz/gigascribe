"""llama.cpp provider (GGUF models, local inference, no network).

Uses create_chat_completion() rather than a raw text completion call so
each supported model (Qwen3, Gemma 3, Ministral 3, ...) is prompted through
its own chat template instead of one hardcoded format. llama-cpp-python
reads that template out of the GGUF's own metadata (tokenizer.chat_template)
-- if a model was quantized/exported without one, load() fails loudly with
a clear error instead of silently falling back to a guessed format.
"""
from __future__ import annotations

import asyncio
import gc
from typing import Any, Optional

from ..schemas import ProtocolValidationError
from .base import LLMProvider

_DEFAULT_SYSTEM_PROMPT = "Ты — ассистент для протоколирования деловых встреч. Отвечай строго на русском языке."


class LlamaCppProvider(LLMProvider):
    def __init__(self, *, model_path: str, context_length: int, n_gpu_layers: int = -1,
                 no_think: bool = False, extra_launch_params: Optional[dict[str, Any]] = None):
        super().__init__(model_path=model_path, context_length=context_length)
        self.n_gpu_layers = n_gpu_layers
        self.no_think = no_think
        self.extra_launch_params = dict(extra_launch_params or {})
        self._llm = None

    async def load(self) -> None:
        if self.loaded:
            return
        loop = asyncio.get_running_loop()

        def _load():
            from llama_cpp import Llama  # heavy import; only required when this provider is used
            llm = Llama(
                model_path=self.model_path,
                n_ctx=self.context_length,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
                **self.extra_launch_params,
            )
            try:
                template = (llm.metadata or {}).get("tokenizer.chat_template")
            except Exception:
                template = None
            if not template:
                raise ProtocolValidationError(
                    f"GGUF at {self.model_path!r} has no tokenizer.chat_template in its metadata. "
                    "Refusing to run it as a raw completion model (that would silently produce an "
                    "incorrectly-formatted prompt for an instruct/chat model). Use a GGUF export that "
                    "embeds its chat template."
                )
            return llm

        self._llm = await loop.run_in_executor(None, _load)
        self.loaded = True

    async def generate(self, prompt: str, *, temperature: float, max_tokens: int,
                        system_prompt: str | None = None) -> str:
        if not self.loaded or self._llm is None:
            raise RuntimeError("LlamaCppProvider.generate() called before load()")
        loop = asyncio.get_running_loop()
        system = system_prompt or _DEFAULT_SYSTEM_PROMPT
        if self.no_think:
            system += " Не показывай ход рассуждений и не используй теги <think>. Сразу дай финальный ответ."

        def _run():
            out = self._llm.create_chat_completion(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                max_tokens=max_tokens, temperature=temperature,
            )
            return out["choices"][0]["message"]["content"] or ""

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
