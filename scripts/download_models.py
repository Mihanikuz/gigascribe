#!/usr/bin/env python3
"""Download/warm up GigaScribe models into a persistent models directory."""

from __future__ import annotations

import argparse
import os, json, socket, subprocess
import shutil
import sys
import tempfile
from pathlib import Path

from model_store import (
    PYANNOTE_REPO_ID, PYANNOTE_SEGMENTATION_REPO_ID, GIGAAM_MARKER_NAME, SUPPORTED_DIARIZATION_MODELS,
    PYANNOTE_MARKER_NAME, gigaam_cache_dir, gigaam_checkpoint_path,
    is_gigaam_ready, is_pyannote_ready, pyannote_target, pyannote_target_for, write_gigaam_marker,
    write_pyannote_marker, SUPPORTED_GIGAAM_MODELS, MIN_CHECKPOINT_BYTES,
    LEGACY_PYANNOTE_PYTHON,
)


def configure_model_dirs(models_dir: Path, *, offline: bool = False) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    os.environ["GIGASCRIBE_MODELS_DIR"] = str(models_dir)
    os.environ["HF_HOME"] = str(models_dir / "huggingface")
    os.environ["HF_HUB_OFFLINE"] = "1" if offline else "0"
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    gigaam_cache_dir(models_dir).mkdir(parents=True, exist_ok=True)


def download_pyannote(models_dir: Path, token: str | None, *, model_id: str = "pyannote-3.1", force: bool = False) -> Path:
    if model_id not in SUPPORTED_DIARIZATION_MODELS or model_id == "none": raise RuntimeError("Unsupported pyannote model")
    repo_id = SUPPORTED_DIARIZATION_MODELS[model_id]["repo_id"]
    target = pyannote_target_for(models_dir, model_id)
    if is_pyannote_ready(models_dir, model_id) and not force:
        print(f"Pyannote already present and offline-verified: {target}")
        return target
    if not token and not is_pyannote_ready(models_dir, model_id):
        raise RuntimeError(
            "HF_TOKEN is required to download complete pyannote cache. Accept terms for "
            f"{repo_id} and any dependencies referenced by its snapshot."
        )
    from huggingface_hub import snapshot_download

    os.environ["HF_HUB_OFFLINE"] = "0"
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=target, local_dir_use_symlinks=False, token=token, force_download=force)
    # Only legacy 3.1 declares segmentation-3.0. Community-1 dependencies are
    # snapshot metadata, not a global hard-coded legacy repository.
    dependencies = list(SUPPORTED_DIARIZATION_MODELS[model_id]["nested_dependencies"])
    for dependency in dependencies:
        snapshot_download(repo_id=dependency, token=token, force_download=force)
    config = target / "config.yaml"
    if model_id == "pyannote-3.1":
        # Never import 3.x into this (4.x Community-1) interpreter.
        worker = Path(__file__).resolve().parents[1] / "workers" / "pyannote_legacy_worker.py"
        python = os.getenv("GIGASCRIBE_LEGACY_PYTHON", LEGACY_PYANNOTE_PYTHON)
        completed = subprocess.run([python, str(worker)], input=json.dumps({"command":"load_test", "snapshot_path":str(target), "device":"cpu"}) + "\n", text=True, capture_output=True, env={**os.environ, "HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1"})
        if completed.returncode:
            raise RuntimeError(f"legacy pyannote offline load-test failed: {completed.stderr.strip()}")
        report = json.loads(completed.stdout)
        if not str(report.get("pyannote_audio_version", "")).startswith("3."):
            raise RuntimeError("legacy worker did not use pyannote.audio 3.x")
        write_pyannote_marker(models_dir, model_id, nested_repos=dependencies, runtime_version=report["pyannote_audio_version"], torch_version=report.get("torch_version"))
    else:
        from pyannote.audio import Pipeline
        Pipeline.from_pretrained(str(config), token=token)
        os.environ["HF_HUB_OFFLINE"] = "1"
        Pipeline.from_pretrained(str(config))
        write_pyannote_marker(models_dir, model_id, nested_repos=dependencies)
    return target


def _load_gigaam(model_name: str, cache_dir: Path, device: str = "cpu"):
    from asr_backend import ASRBackend
    model_id=next(k for k,v in SUPPORTED_GIGAAM_MODELS.items() if v["model_name"] == model_name)
    backend=ASRBackend(str(cache_dir)); backend.load(model_id, device)
    return backend.model


def warm_up_gigaam(models_dir: Path, model_name: str, *, force: bool = False) -> Path:
    cache_dir = gigaam_cache_dir(models_dir)
    target = models_dir / "gigaam" / model_name
    if is_gigaam_ready(models_dir, model_name) and not force:
        print(f"GigaAM already present: {gigaam_checkpoint_path(models_dir, model_name)}")
        return target
    # Always warm into an isolated cache.  A failed HTTP response must never
    # become the live checkpoint or receive a readiness marker.
    if force or not is_gigaam_ready(models_dir, model_name):
        tmp_parent = Path(tempfile.mkdtemp(prefix="gigaam-force-", dir=str(models_dir)))
        try:
            tmp_cache = gigaam_cache_dir(tmp_parent); tmp_cache.mkdir(parents=True, exist_ok=True)
            model = _load_gigaam(model_name, tmp_cache)
            if model is None or not hasattr(model, "transcribe"):
                raise RuntimeError(f"GigaAM loader returned an invalid model for {model_name}")
            tmp_ckpt = gigaam_checkpoint_path(tmp_parent, model_name)
            if not tmp_ckpt.is_file() or tmp_ckpt.stat().st_size < MIN_CHECKPOINT_BYTES:
                raise RuntimeError(f"GigaAM loader produced a corrupt checkpoint (<{MIN_CHECKPOINT_BYTES} bytes): {tmp_ckpt}")
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.replace(tmp_ckpt, gigaam_checkpoint_path(models_dir, model_name))
            # Prove the final cache works in a fresh backend while all common
            # download routes are offline and sockets are blocked.
            del model
            old_env = {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
            old_connect, old_create = socket.socket.connect, socket.create_connection
            def blocked(*_a, **_kw): raise RuntimeError("network attempted during GigaAM offline reload")
            try:
                os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
                socket.socket.connect = blocked
                socket.create_connection = blocked
                model_id = next(k for k,v in SUPPORTED_GIGAAM_MODELS.items() if v["model_name"] == model_name)
                backend = __import__("asr_backend", fromlist=["ASRBackend"]).ASRBackend(str(cache_dir))
                backend.load(model_id, "cpu")
                if backend.model is None or not hasattr(backend.model, "transcribe") or backend.model_id != model_id or backend.actual_device != "cpu":
                    raise RuntimeError("GigaAM final-cache offline reload verification failed")
                backend.unload()
            except Exception:
                # Do not bless a checkpoint which could only be loaded online.
                gigaam_checkpoint_path(models_dir, model_name).unlink(missing_ok=True)
                raise
            finally:
                socket.socket.connect, socket.create_connection = old_connect, old_create
                for key, value in old_env.items():
                    if value is None: os.environ.pop(key, None)
                    else: os.environ[key] = value
            write_gigaam_marker(models_dir, model_name, offline_reload_verified=True)
        finally:
            shutil.rmtree(tmp_parent, ignore_errors=True)
        return target
    return target

def main() -> int:
    parser = argparse.ArgumentParser(description="Download GigaScribe models to a persistent directory")
    parser.add_argument("--models-dir", default=os.getenv("GIGASCRIBE_MODELS_DIR", "./models"))
    parser.add_argument("--gigaam-model", default=None, choices=list(SUPPORTED_GIGAAM_MODELS))
    parser.add_argument("--asr", choices=list(SUPPORTED_GIGAAM_MODELS), help="ASR registry ID")
    parser.add_argument("--diarization", choices=[k for k in SUPPORTED_DIARIZATION_MODELS if k != "none"])
    parser.add_argument("--all", action="store_true", help="download every ASR and diarization model")
    parser.add_argument("--list", action="store_true", help="list registry IDs without downloading")
    parser.add_argument("--repair", action="store_true", help="replace invalid checkpoints/snapshots")
    parser.add_argument("--offline", "--offline-verify", dest="offline", action="store_true", help="verify existing markers without network downloads")
    parser.add_argument("--pyannote-model", default="pyannote-3.1", choices=["pyannote-3.1", "pyannote-community-1"])
    parser.add_argument("--skip-gigaam", action="store_true")
    parser.add_argument("--skip-pyannote", action="store_true")
    parser.add_argument("--force", action="store_true", help="download/warm up models again even if readiness markers exist")
    args = parser.parse_args()
    if args.list:
        for mid, meta in SUPPORTED_GIGAAM_MODELS.items(): print(f"asr {mid}: {meta['model_name']}")
        for mid, meta in SUPPORTED_DIARIZATION_MODELS.items(): print(f"diarization {mid}: {meta.get('repo_id', 'none')}")
        return 0

    models_dir = Path(args.models_dir).expanduser().resolve()
    configure_model_dirs(models_dir, offline=args.offline)
    paths: dict[str, str] = {}; failures: dict[str, str] = {}
    try:
        if args.offline:
            asr_ids = list(SUPPORTED_GIGAAM_MODELS) if args.all else [args.asr or args.gigaam_model or "gigaam-v2-ctc"]
            diar_ids = [k for k in SUPPORTED_DIARIZATION_MODELS if k != "none"] if args.all else [args.diarization or args.pyannote_model]
            if not args.skip_gigaam and any(not is_gigaam_ready(models_dir, SUPPORTED_GIGAAM_MODELS[x]["model_name"]) for x in asr_ids):
                raise RuntimeError("GigaAM offline verification failed")
            if not args.skip_pyannote and any(not is_pyannote_ready(models_dir, x) for x in diar_ids):
                raise RuntimeError("Pyannote offline verification failed")
            print("Offline verification passed")
            return 0
        asr_ids = list(SUPPORTED_GIGAAM_MODELS) if args.all else [args.asr or args.gigaam_model or "gigaam-v2-ctc"]
        diar_ids = [k for k in SUPPORTED_DIARIZATION_MODELS if k != "none"] if args.all else [args.diarization or args.pyannote_model]
        if not args.skip_gigaam:
            for mid in asr_ids:
                try: paths[mid] = str(warm_up_gigaam(models_dir, SUPPORTED_GIGAAM_MODELS[mid]["model_name"], force=args.force or args.repair))
                except Exception as exc: failures[mid] = str(exc)
        if not args.skip_pyannote:
            for mid in diar_ids:
                try: paths[mid] = str(download_pyannote(models_dir, os.getenv("HF_TOKEN") or None, model_id=mid, force=args.force or args.repair))
                except Exception as exc: failures[mid] = str(exc)
        print("Model status:")
        for name in [*asr_ids, *diar_ids]: print(f"  {name}: {'ok ' + paths[name] if name in paths else 'failed: ' + failures.get(name, 'skipped')}")
        return 1 if failures else 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
