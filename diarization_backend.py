"""Local CUDA-only pyannote Community-1 backend and result normalization."""
from __future__ import annotations
import gc, logging, os
from pathlib import Path
from typing import Any
from model_store import DIARIZATION_MODEL_ID, SUPPORTED_DIARIZATION_MODELS, pyannote_target_for
from errors import GigaScribeError
logger = logging.getLogger(__name__)

def normalize_diarization(output: Any) -> list[dict[str, float | str]]:
    annotation = getattr(output, "exclusive_speaker_diarization", None) or getattr(output, "speaker_diarization", output)
    if not hasattr(annotation, "itertracks"):
        raise GigaScribeError("DIARIZATION_INVALID_RESULT", "Unknown pyannote result format")
    segments=[]
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        start,end=float(turn.start),float(turn.end)
        segments.append({"start":start,"end":end,"speaker":str(speaker)})
    return validate_diarization_segments(segments)

def validate_diarization_segments(segments: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    if not segments: raise GigaScribeError("DIARIZATION_EMPTY_RESULT", "pyannote returned no speaker segments")
    ordered=sorted(segments, key=lambda s:(float(s["start"]), float(s["end"]), str(s["speaker"])))
    for s in ordered:
        if float(s["start"]) < 0 or float(s["end"]) <= float(s["start"]):
            raise GigaScribeError("DIARIZATION_INVALID_RESULT", f"Invalid interval: {s}")
    speakers={str(s["speaker"]) for s in ordered}
    duration=max(float(s["end"]) for s in ordered)-min(float(s["start"]) for s in ordered)
    logger.info("Diarization completed: speakers=%d, segments=%d, duration=%.3f", len(speakers), len(ordered), duration)
    return ordered

def map_speakers_to_asr(asr_segments: list[dict[str, Any]], diarization_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels: dict[str, str] = {}
    def display(label: str) -> str:
        if label not in labels: labels[label]=f"Спикер {len(labels)+1}"
        return labels[label]
    out=[]
    for asr in asr_segments:
        a0,a1=float(asr["start"]),float(asr["end"]); center=(a0+a1)/2
        overlaps: dict[str, float] = {}
        center_label=None
        for d in diarization_segments:
            d0,d1=float(d["start"]),float(d["end"]); label=str(d["speaker"])
            inter=max(0.0, min(a1,d1)-max(a0,d0))
            if inter: overlaps[label]=overlaps.get(label,0.0)+inter
            if d0 <= center <= d1 and center_label is None: center_label=label
        if overlaps:
            max_overlap=max(overlaps.values()); tied=[k for k,v in overlaps.items() if v == max_overlap]
            chosen=center_label if center_label in tied else sorted(tied)[0]
        else:
            chosen=center_label or "SPEAKER_00"
        out.append({**asr,"speaker":display(chosen),"speaker_cluster":chosen})
    return out

class DiarizationBackend:
    def __init__(self, models_dir: str): self.models_dir, self.pipeline, self.model_id, self.actual_device = models_dir, None, None, None
    def load(self, model_id: str, device: str) -> None:
        if model_id != DIARIZATION_MODEL_ID or model_id not in SUPPORTED_DIARIZATION_MODELS: raise ValueError(f"Unsupported diarization model: {model_id}")
        if device != "cuda": raise GigaScribeError("CUDA_NOT_AVAILABLE", "Diarization is CUDA-only")
        os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1"); os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0"); os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        try:
            import torch
            from pyannote.audio import Pipeline
            path=pyannote_target_for(Path(self.models_dir), model_id)
            logger.info("Loading pyannote Community-1 from %s on cuda", path)
            self.pipeline = Pipeline.from_pretrained(str(path / "config.yaml"))
            self.pipeline = self.pipeline.to(torch.device("cuda"))
            # Smoke a real device check where possible.
            params=[]
            if hasattr(self.pipeline, "parameters"):
                params=list(self.pipeline.parameters())
            if params and any(getattr(p, "device", None).type != "cuda" for p in params):
                raise RuntimeError("not all pyannote parameters are on cuda")
            self.model_id, self.actual_device = model_id, "cuda"
            logger.info("Loaded pyannote Community-1 path=%s device=cuda", path)
        except GigaScribeError: raise
        except Exception as exc: raise GigaScribeError("DIARIZATION_MODEL_LOAD_FAILED", str(exc)) from exc
    def diarize(self, audio_path: str, **speaker_constraints: Any) -> list[dict[str, float | str]]:
        if self.pipeline is None: raise GigaScribeError("DIARIZATION_MODEL_LOAD_FAILED", "Diarization backend is not loaded")
        try:
            import torch
            with torch.inference_mode():
                return normalize_diarization(self.pipeline(audio_path, **{k:v for k,v in speaker_constraints.items() if v is not None}))
        except GigaScribeError: raise
        except Exception as exc: raise GigaScribeError("DIARIZATION_INVALID_RESULT", str(exc)) from exc
    def unload(self) -> None:
        if self.pipeline is not None: del self.pipeline
        self.pipeline = self.model_id = self.actual_device = None; gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception: pass
