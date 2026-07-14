#!/usr/bin/env python3
"""Download/warm up GigaScribe models into a persistent models directory."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from model_store import (
    PYANNOTE_REPO_ID, PYANNOTE_SEGMENTATION_REPO_ID, GIGAAM_MARKER_NAME,
    PYANNOTE_MARKER_NAME, gigaam_cache_dir, gigaam_checkpoint_path,
    is_gigaam_ready, is_pyannote_ready, pyannote_target, write_gigaam_marker,
    write_pyannote_marker,
)


def configure_model_dirs(models_dir: Path, *, offline: bool = False) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    os.environ["GIGASCRIBE_MODELS_DIR"] = str(models_dir)
    os.environ["HF_HOME"] = str(models_dir / "huggingface")
    os.environ["HF_HUB_OFFLINE"] = "1" if offline else "0"
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    gigaam_cache_dir(models_dir).mkdir(parents=True, exist_ok=True)


def download_pyannote(models_dir: Path, token: str | None, *, force: bool = False) -> Path:
    target = pyannote_target(models_dir)
    if is_pyannote_ready(models_dir) and not force:
        print(f"Pyannote already present and offline-verified: {target}")
        return target
    if not token and not is_pyannote_ready(models_dir):
        raise RuntimeError(
            "HF_TOKEN is required to download complete pyannote cache. Accept terms for "
            f"{PYANNOTE_REPO_ID} and gated dependency {PYANNOTE_SEGMENTATION_REPO_ID}."
        )
    from huggingface_hub import snapshot_download
    from pyannote.audio import Pipeline

    os.environ["HF_HUB_OFFLINE"] = "0"
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=PYANNOTE_REPO_ID, local_dir=target, local_dir_use_symlinks=False, token=token, force_download=force)
    # Materialize gated nested dependency used by pipeline config.
    snapshot_download(repo_id=PYANNOTE_SEGMENTATION_REPO_ID, token=token, force_download=force)
    Pipeline.from_pretrained(str(target), use_auth_token=token)
    os.environ["HF_HUB_OFFLINE"] = "1"
    Pipeline.from_pretrained(str(target))
    write_pyannote_marker(models_dir)
    return target


def _load_gigaam(model_name: str, cache_dir: Path, device: str = "cpu"):
    import gigaam
    return gigaam.load_model(model_name, device=device, download_root=str(cache_dir))


def warm_up_gigaam(models_dir: Path, model_name: str, *, force: bool = False) -> Path:
    cache_dir = gigaam_cache_dir(models_dir)
    target = models_dir / "gigaam" / model_name
    if is_gigaam_ready(models_dir, model_name) and not force:
        print(f"GigaAM already present: {gigaam_checkpoint_path(models_dir, model_name)}")
        return target
    if force:
        tmp_parent = Path(tempfile.mkdtemp(prefix="gigaam-force-", dir=str(models_dir)))
        try:
            tmp_cache = gigaam_cache_dir(tmp_parent); tmp_cache.mkdir(parents=True, exist_ok=True)
            model = _load_gigaam(model_name, tmp_cache)
            if model is None or not hasattr(model, "transcribe"):
                raise RuntimeError(f"GigaAM loader returned an invalid model for {model_name}")
            tmp_ckpt = gigaam_checkpoint_path(tmp_parent, model_name)
            if not tmp_ckpt.is_file() or tmp_ckpt.stat().st_size <= 0:
                raise RuntimeError(f"GigaAM loader did not create checkpoint: {tmp_ckpt}")
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.replace(tmp_ckpt, gigaam_checkpoint_path(models_dir, model_name))
            write_gigaam_marker(models_dir, model_name)
        finally:
            shutil.rmtree(tmp_parent, ignore_errors=True)
        return target
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = _load_gigaam(model_name, cache_dir)
    if model is None or not hasattr(model, "transcribe"):
        raise RuntimeError(f"GigaAM loader returned an invalid model for {model_name}")
    write_gigaam_marker(models_dir, model_name)
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
