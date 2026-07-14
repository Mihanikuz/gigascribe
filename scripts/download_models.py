#!/usr/bin/env python3
"""Download/warm up GigaScribe models into a persistent models directory."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PYANNOTE_REPO_ID = "pyannote/speaker-diarization-3.1"
PYANNOTE_DIRNAME = "pyannote-speaker-diarization-3.1"
GIGAAM_MARKER_NAME = ".gigaam-ready.json"
PYANNOTE_MARKER_NAME = ".pyannote-ready.json"
PYANNOTE_REQUIRED_FILES = ("config.yaml",)


def configure_model_dirs(models_dir: Path, *, offline: bool = False) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    os.environ["GIGASCRIBE_MODELS_DIR"] = str(models_dir)
    os.environ.setdefault("HF_HOME", str(models_dir / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(models_dir / "torch"))
    os.environ.setdefault("XDG_CACHE_HOME", str(models_dir / "cache"))
    os.environ["HF_HUB_OFFLINE"] = "1" if offline else "0"
    for key in ("HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def _write_marker(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_marker(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def pyannote_target(models_dir: Path) -> Path:
    return models_dir / PYANNOTE_DIRNAME


def is_pyannote_ready(models_dir: Path) -> bool:
    target = pyannote_target(models_dir)
    marker = _read_marker(target / PYANNOTE_MARKER_NAME)
    if not marker or marker.get("repo_id") != PYANNOTE_REPO_ID:
        return False
    return all((target / rel).is_file() for rel in PYANNOTE_REQUIRED_FILES)


def is_gigaam_ready(models_dir: Path, model_name: str) -> bool:
    marker = _read_marker(models_dir / "gigaam" / model_name / GIGAAM_MARKER_NAME)
    return bool(marker and marker.get("model_name") == model_name)


def download_pyannote(models_dir: Path, token: str | None, *, force: bool = False) -> Path:
    target = pyannote_target(models_dir)
    if is_pyannote_ready(models_dir) and not force:
        print(f"Pyannote already present: {target}")
        return target
    if not token:
        raise RuntimeError(
            "HF_TOKEN is required to download pyannote/speaker-diarization-3.1. "
            "Set HF_TOKEN after accepting the model terms on Hugging Face, or skip with --skip-pyannote."
        )
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=PYANNOTE_REPO_ID, local_dir=target, local_dir_use_symlinks=False, token=token, force_download=force)
    if not all((target / rel).is_file() for rel in PYANNOTE_REQUIRED_FILES):
        raise RuntimeError(f"Pyannote download did not produce required files in {target}")
    _write_marker(target / PYANNOTE_MARKER_NAME, {"repo_id": PYANNOTE_REPO_ID, "path": str(target)})
    return target


def warm_up_gigaam(models_dir: Path, model_name: str, *, force: bool = False) -> Path:
    if is_gigaam_ready(models_dir, model_name) and not force:
        path = models_dir / "gigaam" / model_name
        print(f"GigaAM already present: {path}")
        return path
    import gigaam

    # gigaam.load_model currently accepts model aliases such as v2_ctc. It does not
    # expose a stable local-path API, so we pin all generic cache locations to models_dir
    # before loading and record a marker after the official loader succeeds.
    model = gigaam.load_model(model_name, device="cpu")
    if model is None or not hasattr(model, "transcribe"):
        raise RuntimeError(f"GigaAM loader returned an invalid model for {model_name}")
    target = models_dir / "gigaam" / model_name
    target.mkdir(parents=True, exist_ok=True)
    _write_marker(target / GIGAAM_MARKER_NAME, {"model_name": model_name, "cache_roots": {k: os.environ.get(k) for k in ("HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME")}})
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Download GigaScribe models to a persistent directory")
    parser.add_argument("--models-dir", default=os.getenv("GIGASCRIBE_MODELS_DIR", "./models"))
    parser.add_argument("--gigaam-model", default=os.getenv("GIGASCRIBE_GIGAAM_MODEL", "v2_ctc"))
    parser.add_argument("--skip-gigaam", action="store_true")
    parser.add_argument("--skip-pyannote", action="store_true")
    parser.add_argument("--force", action="store_true", help="download/warm up models again even if readiness markers exist")
    args = parser.parse_args()

    models_dir = Path(args.models_dir).expanduser().resolve()
    configure_model_dirs(models_dir, offline=False)
    paths: dict[str, str] = {}
    try:
        if not args.skip_gigaam:
            print(f"Preparing GigaAM model '{args.gigaam_model}' in {models_dir}")
            paths["gigaam"] = str(warm_up_gigaam(models_dir, args.gigaam_model, force=args.force))
        if not args.skip_pyannote:
            print(f"Preparing pyannote model in {models_dir}")
            paths["pyannote"] = str(download_pyannote(models_dir, os.getenv("HF_TOKEN") or None, force=args.force))
        print("Model paths:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
