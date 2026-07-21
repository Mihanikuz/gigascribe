from __future__ import annotations
import gc, logging, os, threading
from pathlib import Path
from typing import Any
from asr_backend import ASRBackend
from diarization_backend import DiarizationBackend
from errors import GigaScribeError
from model_store import ASR_MODEL_ID, ASR_MODEL_NAME, DIARIZATION_MODEL_ID, assert_gigaam_ready, gigaam_cache_dir, is_pyannote_ready, load_settings, validate_gigaam_files, validate_pyannote_files, pyannote_target_for
logger=logging.getLogger(__name__)

class ModelManager:
    """Single CUDA model owner. RNNT is shared; diarization is loaded only on demand."""
    def __init__(self, models_dir: Path):
        self.models_dir=Path(models_dir); self.asr:ASRBackend|None=None; self.diarization:DiarizationBackend|None=None
        self.loaded_diarization="none"; self.lock=threading.RLock(); self.last_error=None; self._health_cache=None

    def check_cuda(self) -> dict[str, Any]:
        try:
            import torch
            if not torch.cuda.is_available(): raise GigaScribeError("CUDA_NOT_AVAILABLE", "torch.cuda.is_available() is false")
            x=torch.ones((1,), device="cuda"); _=(x+1).item(); torch.cuda.synchronize()
            props=torch.cuda.get_device_properties(0); total=props.total_memory/(1024**3)
            if total < float(os.getenv("GIGASCRIBE_MIN_VRAM_GB", "8")): raise GigaScribeError("GPU_OUT_OF_MEMORY", f"GPU has only {total:.1f} GB VRAM")
            return {"available":True,"device":torch.cuda.get_device_name(0),"vram_gb":total,"pytorch":torch.__version__,"cuda_runtime":torch.version.cuda}
        except GigaScribeError: raise
        except Exception as exc: raise GigaScribeError("CUDA_SMOKE_TEST_FAILED", str(exc)) from exc

    def ensure_loaded(self, settings: dict[str, Any]|None=None) -> None:
        settings=settings or load_settings(self.models_dir)
        if settings.get("asr_model") != ASR_MODEL_ID or settings.get("device") != "cuda":
            raise GigaScribeError("UNSUPPORTED_CONFIGURATION", "Only gigaam-v3-e2e-rnnt on cuda is supported")
        diar=settings.get("diarization_model", "none")
        with self.lock:
            self.check_cuda(); self.load_asr()
            if diar == "none": self.unload_diarization()
            else: self.load_diarization(diar)
            self._health_cache=None

    def load_asr(self) -> None:
        assert_gigaam_ready(self.models_dir, ASR_MODEL_NAME)
        if self.asr is None:
            self.asr=ASRBackend(str(gigaam_cache_dir(self.models_dir))); self.asr.load(ASR_MODEL_ID, "cuda")
            logger.info("Loaded ASR model %s from %s on cuda", ASR_MODEL_ID, gigaam_cache_dir(self.models_dir))

    def load_diarization(self, model_id: str=DIARIZATION_MODEL_ID) -> None:
        if model_id != DIARIZATION_MODEL_ID: raise GigaScribeError("UNSUPPORTED_CONFIGURATION", f"Unsupported diarization model: {model_id}")
        if not is_pyannote_ready(self.models_dir, model_id): raise GigaScribeError("DIARIZATION_MODEL_NOT_READY", validate_pyannote_files(self.models_dir, model_id)[1] or "not ready")
        if self.diarization is None or self.loaded_diarization != model_id:
            self.unload_diarization(); self.diarization=DiarizationBackend(str(self.models_dir)); self.diarization.load(model_id, "cuda"); self.loaded_diarization=model_id

    def unload_diarization(self) -> None:
        if self.diarization: self.diarization.unload()
        self.diarization=None; self.loaded_diarization="none"; gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception: pass

    def unload_all(self) -> None:
        if self.asr: self.asr.unload()
        self.asr=None; self.unload_diarization()

    # Backward-compatible aliases
    load_models=ensure_loaded; unload_models=unload_all
    def get_asr(self):
        if self.asr is None: raise GigaScribeError("ASR_MODEL_LOAD_FAILED", "ASR model is not loaded")
        return self.asr
    def get_diarization(self): return self.diarization

    def health_status(self, *, deep: bool=False, settings: dict[str,Any]|None=None) -> dict[str, Any]:
        settings=settings or load_settings(self.models_dir); status={"status":"ready","cuda":{},"asr":{"model":ASR_MODEL_ID},"diarization":{"model":settings["diarization_model"]}}
        try: status["cuda"]=self.check_cuda()
        except GigaScribeError as exc: status["status"]="not_ready"; status["cuda"]={"available":False,"error_code":exc.code,"error":str(exc)}
        aok,aerr=validate_gigaam_files(self.models_dir); status["asr"].update({"ready":aok,"error":aerr})
        if deep and aok:
            try: self.load_asr(); status["asr"].update({"loadable":True})
            except Exception as exc: status["asr"].update({"ready":False,"loadable":False,"error_code":getattr(exc,"code","ASR_MODEL_LOAD_FAILED"),"error":str(exc)})
        dok,derr=validate_pyannote_files(self.models_dir, settings["diarization_model"]); status["diarization"].update({"ready":dok,"error":derr,"path":str(pyannote_target_for(self.models_dir, settings["diarization_model"]))})
        if deep and settings["diarization_model"] != "none" and dok:
            try: self.load_diarization(settings["diarization_model"]); status["diarization"].update({"loadable":True})
            except Exception as exc: status["diarization"].update({"ready":False,"loadable":False,"error_code":getattr(exc,"code","DIARIZATION_MODEL_LOAD_FAILED"),"error":str(exc)})
        if status["cuda"].get("available") is not True or not status["asr"].get("ready") or not status["diarization"].get("ready"): status["status"]="not_ready"
        return status
