"""Normalize legacy and Community-1 pyannote output without guessing formats."""
from __future__ import annotations
import gc
from typing import Any
from model_store import SUPPORTED_DIARIZATION_MODELS, pyannote_target_for


def normalize_diarization(output: Any) -> list[dict[str, float | str]]:
    """Return Annotation-like outputs as stable JSON-ready speaker turns."""
    annotation = getattr(output, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = getattr(output, "speaker_diarization", output)
    if not hasattr(annotation, "itertracks"):
        raise RuntimeError("Unknown pyannote result format: expected Annotation or speaker_diarization")
    result = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        start, end = float(turn.start), float(turn.end)
        result.append({"start": start, "end": end, "speaker": str(speaker), "duration": end - start})
    return result


class DiarizationBackend:
    def __init__(self, models_dir: str): self.models_dir, self.pipeline, self.model_id, self.actual_device = models_dir, None, None, None
    def load(self, model_id: str, device: str) -> None:
        if model_id not in SUPPORTED_DIARIZATION_MODELS or model_id == "none":
            raise ValueError(f"Unsupported diarization model: {model_id}")
        from pyannote.audio import Pipeline
        self.pipeline = Pipeline.from_pretrained(str(pyannote_target_for(__import__('pathlib').Path(self.models_dir), model_id) / "config.yaml"))
        if device == "cuda": self.pipeline = self.pipeline.to(device)
        self.model_id, self.actual_device = model_id, device
    def diarize(self, audio_path: str, **speaker_constraints: Any) -> list[dict[str, float | str]]:
        if self.pipeline is None: raise RuntimeError("Diarization backend is not loaded")
        return normalize_diarization(self.pipeline(audio_path, **{k:v for k,v in speaker_constraints.items() if v is not None}))
    def unload(self) -> None: self.pipeline = self.model_id = self.actual_device = None; gc.collect()
