from __future__ import annotations
import gc, logging, os, threading
from pathlib import Path
from typing import Any
from asr_backend import ASRBackend
from diarization_backend import DiarizationBackend
from errors import GigaScribeError
from model_store import ASR_MODEL_ID, ASR_MODEL_NAME, DIARIZATION_MODEL_ID, assert_gigaam_ready, gigaam_cache_dir, is_pyannote_ready, load_settings, validate_gigaam_files, validate_pyannote_files
logger=logging.getLogger(__name__)
class ModelManager:
    def __init__(self, models_dir: Path):
        self.models_dir=Path(models_dir); self.asr=None; self.diarization=None; self.loaded_diarization="none"; self.lock=threading.Lock(); self.last_error=None
    def check_cuda(self) -> dict[str, Any]:
        try:
            import torch
            if not torch.cuda.is_available(): raise GigaScribeError("CUDA_NOT_AVAILABLE", "torch.cuda.is_available() is false")
            x=torch.ones((1,), device="cuda"); _=(x+1).item()
            props=torch.cuda.get_device_properties(0); total=props.total_memory/(1024**3)
            if total < float(os.getenv("GIGASCRIBE_MIN_VRAM_GB", "8")): raise GigaScribeError("GPU_OUT_OF_MEMORY", f"GPU has only {total:.1f} GB VRAM")
            return {"cuda":True,"gpu":torch.cuda.get_device_name(0),"vram_gb":total,"pytorch":torch.__version__,"cuda_runtime":torch.version.cuda}
        except GigaScribeError: raise
        except Exception as exc: raise GigaScribeError("CUDA_SMOKE_TEST_FAILED", str(exc)) from exc
    def load_models(self, diarization_model: str|None=None) -> None:
        diarization_model = diarization_model if diarization_model is not None else load_settings(self.models_dir)["diarization_model"]
        with self.lock:
            self.check_cuda(); assert_gigaam_ready(self.models_dir, ASR_MODEL_NAME)
            if diarization_model != "none" and not is_pyannote_ready(self.models_dir, diarization_model): raise GigaScribeError("DIARIZATION_MODEL_NOT_FOUND", validate_pyannote_files(self.models_dir, diarization_model)[1] or "not ready")
            try:
                if self.asr is None:
                    self.asr=ASRBackend(str(gigaam_cache_dir(self.models_dir))); self.asr.load(ASR_MODEL_ID, "cuda")
                if diarization_model != "none" and (self.diarization is None or self.loaded_diarization != diarization_model):
                    if self.diarization: self.diarization.unload()
                    self.diarization=DiarizationBackend(str(self.models_dir)); self.diarization.load(DIARIZATION_MODEL_ID, "cuda"); self.loaded_diarization=diarization_model
                logger.info("Модели загружены")
            except GigaScribeError as exc: self.last_error=str(exc); raise
            except Exception as exc: self.last_error=str(exc); raise GigaScribeError("ASR_MODEL_LOAD_FAILED", str(exc)) from exc
    def unload_models(self) -> None:
        if self.asr: self.asr.unload(); del self.asr
        if self.diarization: self.diarization.unload(); del self.diarization
        self.asr=None; self.diarization=None; self.loaded_diarization="none"; gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception: pass
    def get_asr(self):
        if self.asr is None: raise GigaScribeError("ASR_MODEL_LOAD_FAILED", "ASR model is not loaded")
        return self.asr
    def get_diarization(self): return self.diarization
    def health_status(self) -> dict[str, Any]:
        status={"status":"ready","cuda":False,"asr":{"model":ASR_MODEL_ID},"diarization":{"model":load_settings(self.models_dir)["diarization_model"]}}
        try: status.update(self.check_cuda())
        except Exception as exc: status["status"]="not_ready"; status["cuda_error"]=str(exc)
        aok,aerr=validate_gigaam_files(self.models_dir); status["asr"].update({"ready":aok,"error":aerr})
        dok,derr=validate_pyannote_files(self.models_dir, status["diarization"]["model"]); status["diarization"].update({"ready":dok,"error":derr})
        if not (status.get("cuda") and aok and dok): status["status"]="not_ready"
        return status
