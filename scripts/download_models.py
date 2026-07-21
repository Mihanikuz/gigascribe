#!/usr/bin/env python3
"""Download/verify the only supported GigaScribe models: RNNT and pyannote Community-1."""
from __future__ import annotations
import argparse, os
from pathlib import Path
from model_store import ASR_MODEL_ID, ASR_MODEL_NAME, DIARIZATION_MODEL_ID, SUPPORTED_GIGAAM_MODELS, SUPPORTED_DIARIZATION_MODELS, gigaam_cache_dir, gigaam_checkpoint_path, gigaam_tokenizer_path, is_gigaam_ready, is_pyannote_ready, pyannote_target_for, write_gigaam_marker, write_pyannote_marker

def warm_up_gigaam(models_dir: Path, model_name: str = ASR_MODEL_NAME, *, force: bool=False) -> Path:
    if model_name != ASR_MODEL_NAME: raise RuntimeError("Only GigaAM v3 E2E RNNT is supported")
    if is_gigaam_ready(models_dir, model_name) and not force: return gigaam_cache_dir(models_dir)
    import gigaam
    os.environ.setdefault("HF_HUB_OFFLINE", "0")
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
    from huggingface_hub import snapshot_download
    snapshot_download(SUPPORTED_DIARIZATION_MODELS[model_id]["repo_id"], local_dir=target, token=token, local_dir_use_symlinks=False)
    from pyannote.audio import Pipeline
    import torch
    pipe=Pipeline.from_pretrained(str(target/"config.yaml")); pipe.to(torch.device("cuda"))
    write_pyannote_marker(models_dir, model_id)
    return target

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--models-dir", type=Path, default=Path("./models")); ap.add_argument("--skip-gigaam", action="store_true"); ap.add_argument("--skip-pyannote", action="store_true"); ap.add_argument("--check", action="store_true"); ap.add_argument("--force", action="store_true"); ap.add_argument("--repair", action="store_true"); ap.add_argument("--list", action="store_true")
    args=ap.parse_args(); models=args.models_dir.resolve(); models.mkdir(parents=True, exist_ok=True)
    if args.list:
        print(f"asr {ASR_MODEL_ID}: {SUPPORTED_GIGAAM_MODELS[ASR_MODEL_ID]['model_name']}"); print(f"diarization {DIARIZATION_MODEL_ID}"); return
    if args.check:
        if not args.skip_gigaam and not is_gigaam_ready(models): raise RuntimeError("GigaAM RNNT offline verification failed")
        if not args.skip_pyannote and not is_pyannote_ready(models, DIARIZATION_MODEL_ID): raise RuntimeError("Pyannote Community-1 offline verification failed")
        return
    if not args.skip_gigaam: warm_up_gigaam(models, force=args.force or args.repair)
    if not args.skip_pyannote: download_pyannote(models, os.getenv("HF_TOKEN") or None, force=args.force or args.repair)
if __name__ == "__main__": main()
