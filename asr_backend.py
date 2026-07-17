"""Small adapter around the version-specific public GigaAM loading API."""
from __future__ import annotations
import gc
from typing import Any
from model_store import SUPPORTED_GIGAAM_MODELS

class ASRBackend:
    def __init__(self, cache_dir: str): self.cache_dir, self.model, self.model_id, self.actual_device = cache_dir, None, None, None
    def load(self, model_id: str, device: str) -> None:
        import gigaam
        if model_id not in SUPPORTED_GIGAAM_MODELS: raise ValueError(f"Unsupported ASR model: {model_id}")
        meta=SUPPORTED_GIGAAM_MODELS[model_id]
        # The common official entry point owns v2/v3 construction differences.
        self.model=gigaam.load_model(meta["model_name"], device=device, download_root=self.cache_dir)
        self.model_id, self.actual_device=model_id,device
    def transcribe(self, audio_path: str) -> str:
        if self.model is None: raise RuntimeError("ASR backend is not loaded")
        result=self.model.transcribe(audio_path)
        # v3 e2e APIs may return text directly, a result object, or a list.
        if isinstance(result,str): return result
        if isinstance(result,(list,tuple)): return " ".join(self._text(x) for x in result).strip()
        return self._text(result)
    @staticmethod
    def _text(value: Any) -> str:
        if isinstance(value,str): return value
        if isinstance(value,dict): return str(value.get("text", value.get("transcript", "")))
        return str(getattr(value,"text",getattr(value,"transcript",value)))
    def unload(self) -> None:
        self.model=None; self.model_id=None; gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception: pass
