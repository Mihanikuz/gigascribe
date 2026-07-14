#!/usr/bin/env python3
"""Download/warm up GigaScribe models into a dedicated models directory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def configure_model_dirs(models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("GIGASCRIBE_MODELS_DIR", str(models_dir))
    os.environ.setdefault("HF_HOME", str(models_dir / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(models_dir / "torch"))
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)


def download_pyannote(models_dir: Path, token: str | None) -> Path:
    from huggingface_hub import snapshot_download

    target = models_dir / "pyannote-speaker-diarization-3.1"
    snapshot_download(
        repo_id="pyannote/speaker-diarization-3.1",
        local_dir=target,
        local_dir_use_symlinks=False,
        token=token,
    )
    return target


def warm_up_gigaam(model_name: str) -> None:
    import gigaam

    gigaam.load_model(model_name, device="cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GigaScribe models to a separate directory")
    parser.add_argument("--models-dir", default=os.getenv("GIGASCRIBE_MODELS_DIR", "./models"))
    parser.add_argument("--gigaam-model", default=os.getenv("GIGASCRIBE_GIGAAM_MODEL", "v2_ctc"))
    parser.add_argument("--skip-gigaam", action="store_true")
    parser.add_argument("--skip-pyannote", action="store_true")
    args = parser.parse_args()

    models_dir = Path(args.models_dir).expanduser().resolve()
    configure_model_dirs(models_dir)

    if not args.skip_gigaam:
        print(f"Warming up GigaAM model '{args.gigaam_model}' into {models_dir}")
        warm_up_gigaam(args.gigaam_model)

    if not args.skip_pyannote:
        print(f"Downloading pyannote model into {models_dir}")
        target = download_pyannote(models_dir, os.getenv("HF_TOKEN") or None)
        print(f"Set GIGASCRIBE_PYANNOTE_MODEL={target}")

    print("Done")


if __name__ == "__main__":
    main()
