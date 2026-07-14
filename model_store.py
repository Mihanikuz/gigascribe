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
GIGAAM_CHECKPOINTS = {"v2_ctc": "v2_ctc.ckpt"}


def models_dir() -> Path:
    return Path(os.getenv("GIGASCRIBE_MODELS_DIR", "./models")).expanduser().resolve()


def gigaam_cache_dir(base: Path | None = None) -> Path:
    return (base or models_dir()) / "gigaam-cache"


def gigaam_checkpoint_path(base: Path, model_name: str) -> Path:
    return gigaam_cache_dir(base) / GIGAAM_CHECKPOINTS.get(model_name, f"{model_name}.ckpt")


def gigaam_marker_path(base: Path, model_name: str) -> Path:
    return base / "gigaam" / model_name / GIGAAM_MARKER_NAME


def pyannote_target(base: Path) -> Path:
    return base / PYANNOTE_DIRNAME


def pyannote_marker_path(base: Path) -> Path:
    return pyannote_target(base) / PYANNOTE_MARKER_NAME


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


def is_pyannote_ready(base: Path) -> bool:
    marker = _read_json(pyannote_marker_path(base))
    target = pyannote_target(base)
    if not marker or marker.get("repo_id") != PYANNOTE_REPO_ID: return False
    if marker.get("path") != str(target): return False
    if not (target / "config.yaml").is_file(): return False
    nested = marker.get("nested_repos") or []
    return PYANNOTE_SEGMENTATION_REPO_ID in nested and bool(marker.get("offline_verified"))


def write_pyannote_marker(base: Path) -> None:
    target = pyannote_target(base)
    if not (target / "config.yaml").is_file():
        raise RuntimeError(f"Pyannote snapshot is incomplete: {target}")
    atomic_write_json(pyannote_marker_path(base), {"repo_id": PYANNOTE_REPO_ID, "path": str(target), "nested_repos": [PYANNOTE_SEGMENTATION_REPO_ID], "offline_verified": True})
