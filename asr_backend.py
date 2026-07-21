"""CUDA-only adapter for the documented GigaAM v3 E2E RNNT public API."""
from __future__ import annotations

import gc
from typing import Any

from model_store import SUPPORTED_GIGAAM_MODELS


class ASRBackend:
    """Keep model-name selection in one place; do not use longform/VAD APIs."""
    def __init__(self, cache_dir: str):
        self.cache_dir, self.model, self.model_id, self.actual_device = cache_dir, None, None, None

    def load(self, model_id: str, device: str) -> None:
        import gigaam
        if model_id not in SUPPORTED_GIGAAM_MODELS:
            raise ValueError(f"Unsupported ASR model: {model_id}")
        if device != "cuda":
            raise RuntimeError(f"{model_id} is CUDA-only")
        # Keep the call here so a changed v3 RNNT API fails explicitly.
        self.model = gigaam.load_model(
            SUPPORTED_GIGAAM_MODELS[model_id]["official_model_name"],
            device=device, download_root=self.cache_dir,
        )
        if self.model is None or not hasattr(self.model, "transcribe"):
            raise RuntimeError(f"GigaAM returned no transcribable {model_id} model")
        self.model_id, self.actual_device = model_id, device

    def transcribe(self, audio_path: str) -> str:
        if self.model is None: raise RuntimeError("ASR backend is not loaded")
        result = self.model.transcribe(audio_path)
        if isinstance(result, str): return result
        if isinstance(result, (list, tuple)): return " ".join(self._text(x) for x in result).strip()
        return self._text(result)

    @staticmethod
    def _text(value: Any) -> str:
        if isinstance(value, str): return value
        if isinstance(value, dict): return str(value.get("text", value.get("transcript", "")))
        return str(getattr(value, "text", getattr(value, "transcript", value)))

    def unload(self) -> None:
        self.model = self.model_id = self.actual_device = None; gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception: pass
