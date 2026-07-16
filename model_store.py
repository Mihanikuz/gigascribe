"""Persistent model cache helpers for GigaScribe."""
from __future__ import annotations

import hashlib, json, os, shutil, tempfile
from pathlib import Path
from typing import Any

PYANNOTE_REPO_ID = "pyannote/speaker-diarization-3.1"
PYANNOTE_SEGMENTATION_REPO_ID = "pyannote/segmentation-3.0"
PYANNOTE_DIRNAME = "pyannote-speaker-diarization-3.1"
GIGAAM_MARKER_NAME = ".gigaam-ready.json"
PYANNOTE_MARKER_NAME = ".pyannote-ready.json"
GIGAAM_CHECKPOINTS = {"v2_ctc": "v2_ctc.ckpt", "v3_ctc": "v3_ctc.ckpt"}


def models_dir() -> Path:
    return Path(os.getenv("GIGASCRIBE_MODELS_DIR", "./models")).expanduser().resolve()


def gigaam_cache_dir(base: Path | None = None) -> Path:
    return (base or models_dir()) / "gigaam-cache"


def gigaam_checkpoint_path(base: Path, model_name: str) -> Path:
    return gigaam_cache_dir(base) / GIGAAM_CHECKPOINTS.get(model_name, f"{model_name}.ckpt")


def gigaam_marker_path(base: Path, model_name: str) -> Path:
    return base / "gigaam" / model_name / GIGAAM_MARKER_NAME


def pyannote_target(base: Path) -> Path:
    return pyannote_target_for(base, "pyannote-3.1")


def pyannote_marker_path(base: Path, model_id: str = "pyannote-3.1") -> Path:
    return pyannote_target_for(base, model_id) / PYANNOTE_MARKER_NAME


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
            fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def checkpoint_info(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)}


def is_gigaam_ready(base: Path, model_name: str) -> bool:
    ckpt = gigaam_checkpoint_path(base, model_name)
    marker = _read_json(gigaam_marker_path(base, model_name))
    if not marker or marker.get("model_name") != model_name: return False
    if marker.get("cache_dir") != str(gigaam_cache_dir(base)): return False
    if marker.get("checkpoint") != str(ckpt): return False
    if not ckpt.is_file() or ckpt.stat().st_size <= 0: return False
    if marker.get("checkpoint_size") != ckpt.stat().st_size: return False
    expected = marker.get("checkpoint_sha256")
    return not expected or expected == sha256_file(ckpt)


def write_gigaam_marker(base: Path, model_name: str) -> None:
    ckpt = gigaam_checkpoint_path(base, model_name)
    if not ckpt.is_file() or ckpt.stat().st_size <= 0:
        raise RuntimeError(f"GigaAM checkpoint is missing or empty: {ckpt}")
    atomic_write_json(gigaam_marker_path(base, model_name), {
        "model_name": model_name, "cache_dir": str(gigaam_cache_dir(base)),
        "checkpoint": str(ckpt), "checkpoint_size": ckpt.stat().st_size,
        "checkpoint_sha256": sha256_file(ckpt),
    })


def assert_gigaam_ready(base: Path, model_name: str) -> None:
    if not is_gigaam_ready(base, model_name):
        raise RuntimeError(f"GigaAM model '{model_name}' is not prepared at {gigaam_checkpoint_path(base, model_name)}. Run scripts/download_models.py first.")


def is_pyannote_ready(base: Path, model_id: str = "pyannote-3.1") -> bool:
    if model_id == "none": return True
    meta = SUPPORTED_DIARIZATION_MODELS.get(model_id)
    if not meta: return False
    marker = _read_json(pyannote_marker_path(base, model_id))
    target = pyannote_target_for(base, model_id)
    if not marker or marker.get("repo_id") != meta["repo_id"]: return False
    if marker.get("path") != str(target): return False
    if not (target / "config.yaml").is_file(): return False
    nested = marker.get("nested_repos") or []
    return PYANNOTE_SEGMENTATION_REPO_ID in nested and bool(marker.get("offline_verified"))


def write_pyannote_marker(base: Path, model_id: str = "pyannote-3.1") -> None:
    if model_id == "none": return
    target = pyannote_target_for(base, model_id)
    if not (target / "config.yaml").is_file():
        raise RuntimeError(f"Pyannote snapshot is incomplete: {target}")
    atomic_write_json(pyannote_marker_path(base, model_id), {"repo_id": SUPPORTED_DIARIZATION_MODELS[model_id]["repo_id"], "path": str(target), "nested_repos": [PYANNOTE_SEGMENTATION_REPO_ID], "offline_verified": True})

SUPPORTED_GIGAAM_MODELS = {
    "gigaam-v2-ctc": {"kind":"asr","label":"GigaAM v2_ctc","model_name":"v2_ctc","version":"v2_ctc","speed":"быстро","vram":"низкая","hf_token_required":False,"terms_required":False},
    "gigaam-v3-ctc": {"kind":"asr","label":"GigaAM v3_ctc","model_name":"v3_ctc","version":"v3_ctc","speed":"средне","vram":"средняя","hf_token_required":False,"terms_required":False},
}
SUPPORTED_DIARIZATION_MODELS = {
    "none": {"kind":"diarization","label":"Без диаризации","model_name":"none","version":"none","speed":"максимальная","vram":"0","hf_token_required":False,"terms_required":False},
    "pyannote-3.1": {"kind":"diarization","label":"pyannote speaker-diarization-3.1","model_name":"speaker-diarization-3.1","repo_id":"pyannote/speaker-diarization-3.1","version":"3.1","speed":"средне","vram":"средняя","hf_token_required":True,"terms_required":True},
    "pyannote-community-1": {"kind":"diarization","label":"pyannote speaker-diarization-community-1","model_name":"speaker-diarization-community-1","repo_id":"pyannote/speaker-diarization-community-1","version":"community-1","speed":"средне","vram":"средняя","hf_token_required":True,"terms_required":True},
}
PROFILES = {
    "max_speed": {"label":"Максимальная скорость","asr_model":"gigaam-v2-ctc","diarization_model":"none","device":"cuda"},
    "balanced": {"label":"Сбалансированный","asr_model":"gigaam-v3-ctc","diarization_model":"pyannote-community-1","device":"cuda"},
    "compatibility": {"label":"Совместимость","asr_model":"gigaam-v2-ctc","diarization_model":"pyannote-3.1","device":"cuda"},
    "cpu": {"label":"CPU","asr_model":"gigaam-v2-ctc","diarization_model":"none","device":"cpu","warning":"CPU режим медленнее; параллелизм ограничен."},
}

def settings_path(base: Path | None = None) -> Path:
    return (base or models_dir()) / "settings.json"

def default_settings() -> dict[str, Any]:
    return {"asr_model":"gigaam-v2-ctc","diarization_model":"none","device":os.getenv("GIGASCRIBE_DEVICE","cpu")}

def load_settings(base: Path | None = None) -> dict[str, Any]:
    data = _read_json(settings_path(base)) or {}
    settings = {**default_settings(), **data}
    if settings["asr_model"] not in SUPPORTED_GIGAAM_MODELS: settings["asr_model"] = "gigaam-v2-ctc"
    if settings["diarization_model"] not in SUPPORTED_DIARIZATION_MODELS: settings["diarization_model"] = "none"
    if settings["device"] not in {"cpu","cuda"}: settings["device"] = "cpu"
    return settings

def save_settings(settings: dict[str, Any], base: Path | None = None) -> dict[str, Any]:
    current = load_settings(base); current.update(settings)
    if current["asr_model"] not in SUPPORTED_GIGAAM_MODELS: raise ValueError("Unsupported ASR model")
    if current["diarization_model"] not in SUPPORTED_DIARIZATION_MODELS: raise ValueError("Unsupported diarization model")
    if current["device"] not in {"cpu","cuda"}: raise ValueError("Unsupported device")
    atomic_write_json(settings_path(base), current)
    return current

def pyannote_target_for(base: Path, model_id: str) -> Path:
    if model_id == "pyannote-community-1": return base / "pyannote-speaker-diarization-community-1"
    return base / PYANNOTE_DIRNAME
