#!/usr/bin/env python3
"""Download/verify the only supported GigaScribe models: RNNT and pyannote Community-1."""
from __future__ import annotations
import argparse, os
from pathlib import Path
from model_store import ASR_MODEL_ID, ASR_MODEL_NAME, DIARIZATION_MODEL_ID, SUPPORTED_GIGAAM_MODELS, SUPPORTED_DIARIZATION_MODELS, gigaam_cache_dir, gigaam_checkpoint_path, gigaam_tokenizer_path, is_gigaam_ready, is_pyannote_ready, models_dir as default_models_dir, pyannote_target_for, write_gigaam_marker, write_pyannote_marker

def _allow_online_downloads() -> None:
    # The serving container always sets HF_HUB_OFFLINE=1/TRANSFORMERS_OFFLINE=1
    # (see compose.yaml), so os.environ.setdefault() here would be a no-op
    # when this script runs inside that same environment -- which is the
    # normal way to invoke it (``docker compose run ... download_models.py``).
    # Force both off unconditionally for the duration of the download.
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

def warm_up_gigaam(models_dir: Path, model_name: str = ASR_MODEL_NAME, *, force: bool=False) -> Path:
    if model_name != ASR_MODEL_NAME: raise RuntimeError("Only GigaAM v3 E2E RNNT is supported")
    if is_gigaam_ready(models_dir, model_name) and not force: return gigaam_cache_dir(models_dir)
    _allow_online_downloads()
    import gigaam
    model = gigaam.load_model(ASR_MODEL_NAME, device="cuda", download_root=str(gigaam_cache_dir(models_dir)))
    if model is None or not hasattr(model, "transcribe"): raise RuntimeError("GigaAM RNNT load verification failed")
    ckpt, tok = gigaam_checkpoint_path(models_dir), gigaam_tokenizer_path(models_dir)
    if not ckpt.is_file(): raise RuntimeError(f"Missing checkpoint after download: {ckpt}")
    if not tok.is_file(): raise RuntimeError(f"Missing tokenizer after download: {tok}")
    write_gigaam_marker(models_dir, ASR_MODEL_NAME, offline_reload_verified=True)
    return gigaam_cache_dir(models_dir)

def download_pyannote(models_dir: Path, token: str|None, *, model_id: str = DIARIZATION_MODEL_ID, force: bool=False) -> Path:
    if model_id != DIARIZATION_MODEL_ID: raise RuntimeError("Only pyannote Community-1 is supported")
    target=pyannote_target_for(models_dir, model_id)
    if is_pyannote_ready(models_dir, model_id) and not force: return target
    _allow_online_downloads()
    from huggingface_hub import snapshot_download
    snapshot_download(SUPPORTED_DIARIZATION_MODELS[model_id]["repo_id"], local_dir=target, token=token, local_dir_use_symlinks=False)
    from pyannote.audio import Pipeline
    import torch
    pipe=Pipeline.from_pretrained(str(target/"config.yaml")); pipe.to(torch.device("cuda"))
    write_pyannote_marker(models_dir, model_id)
    return target

def main() -> None:
    ap=argparse.ArgumentParser()
    # Default to GIGASCRIBE_MODELS_DIR, same as every other component --
    # "./models" alone resolves relative to the container's WORKDIR (/app),
    # not the /opt/gigascribe/models the rest of the app actually uses.
    ap.add_argument("--models-dir", type=Path, default=default_models_dir())
    ap.add_argument("--skip-gigaam", action="store_true"); ap.add_argument("--skip-pyannote", action="store_true"); ap.add_argument("--check", action="store_true"); ap.add_argument("--force", action="store_true"); ap.add_argument("--repair", action="store_true"); ap.add_argument("--list", action="store_true")
    args=ap.parse_args(); models=args.models_dir.resolve(); models.mkdir(parents=True, exist_ok=True)
    if args.list:
        print(f"asr {ASR_MODEL_ID}: {SUPPORTED_GIGAAM_MODELS[ASR_MODEL_ID]['model_name']}"); print(f"diarization {DIARIZATION_MODEL_ID}"); return
    print(f"Models directory: {models}")
    if args.check:
        gigaam_ok = args.skip_gigaam or is_gigaam_ready(models)
        pyannote_ok = args.skip_pyannote or is_pyannote_ready(models, DIARIZATION_MODEL_ID)
        print(f"GigaAM: {'SKIPPED' if args.skip_gigaam else ('READY' if gigaam_ok else 'MISSING')}")
        print(f"Pyannote: {'SKIPPED' if args.skip_pyannote else ('READY' if pyannote_ok else 'MISSING')}")
        if not gigaam_ok: raise RuntimeError("GigaAM RNNT offline verification failed")
        if not pyannote_ok: raise RuntimeError("Pyannote Community-1 offline verification failed")
        return
    if not args.skip_gigaam:
        print("Downloading GigaAM v3 E2E RNNT...")
        warm_up_gigaam(models, force=args.force or args.repair)
        print(f"GigaAM: READY ({gigaam_cache_dir(models)})")
    if not args.skip_pyannote:
        print("Downloading pyannote Community-1...")
        target = download_pyannote(models, os.getenv("HF_TOKEN") or None, force=args.force or args.repair)
        print(f"Pyannote: READY ({target})")
    print("Verifying files...")
    if not args.skip_gigaam and not is_gigaam_ready(models): raise RuntimeError("GigaAM RNNT offline verification failed after download")
    if not args.skip_pyannote and not is_pyannote_ready(models, DIARIZATION_MODEL_ID): raise RuntimeError("Pyannote Community-1 offline verification failed after download")
    print("All requested models are ready.")
if __name__ == "__main__": main()
