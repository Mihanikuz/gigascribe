#!/usr/bin/env python3
"""Download/verify a protocol (meeting-minutes) LLM model.

Offline by design, same as scripts/download_models.py: this script is the
*only* place that reaches the network for the protocol module, and only
for the duration of the download. The serving container always runs with
HF_HUB_OFFLINE=1/TRANSFORMERS_OFFLINE=1, so those flags are forced off here
unconditionally before any network call.

Source repo_id/filename are not hardcoded in this repository: the current
GGUF Q4_K_M quantization filenames for these models on Hugging Face change
over time, and shipping a guessed URL nobody has verified would be worse
than requiring the admin to supply one. Pass --repo-id/--filename, or set
them first via the admin API/UI (POST /api/protocol/models/{id}/params).
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protocol.models import SUPPORTED_PROTOCOL_MODELS
from protocol.store import ProtocolStore


def _allow_online_downloads() -> None:
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"


def download_gguf_model(store: ProtocolStore, *, model_id: str, repo_id: str, filename: str,
                         models_dir: Path, token: str | None = None, force: bool = False) -> Path:
    if model_id not in SUPPORTED_PROTOCOL_MODELS:
        raise RuntimeError(f"Unsupported protocol model: {model_id}")
    state = store.get_model_state(model_id)
    target_dir = models_dir / model_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename
    if state and state.get("installed") and target_path.exists() and not force:
        print(f"{model_id}: already installed at {target_path}")
        return target_path

    _allow_online_downloads()
    from huggingface_hub import hf_hub_download
    print(f"Downloading {model_id} from {repo_id}/{filename} ...")
    downloaded = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(target_dir), token=token)
    downloaded_path = Path(downloaded)
    if downloaded_path != target_path and downloaded_path.exists():
        downloaded_path.replace(target_path)
    if not target_path.is_file() or target_path.stat().st_size == 0:
        raise RuntimeError(f"Download did not produce a usable file: {target_path}")

    store.update_model_state(
        model_id, repo_id=repo_id, filename=filename, local_path=str(target_path),
        size_bytes=target_path.stat().st_size, installed=1, last_check_status="downloaded",
    )
    print(f"{model_id}: installed ({target_path.stat().st_size / (1024**3):.2f} GB) -> {target_path}")
    return target_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", choices=sorted(SUPPORTED_PROTOCOL_MODELS))
    ap.add_argument("--repo-id", help="Hugging Face repo id, e.g. 'someorg/Qwen3-14B-GGUF'")
    ap.add_argument("--filename", help="GGUF filename inside the repo, e.g. 'qwen3-14b-q4_k_m.gguf'")
    ap.add_argument("--data-dir", type=Path, default=Path(os.getenv("GIGASCRIBE_DATA_DIR", "./data")))
    ap.add_argument("--models-dir", type=Path, default=Path(os.getenv("GIGASCRIBE_MODELS_DIR", "./models")))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for mid, spec in SUPPORTED_PROTOCOL_MODELS.items():
            print(f"{mid}: {spec.label} ({spec.quantization}, context={spec.context_length})")
        return

    if not args.model_id:
        raise SystemExit("--model-id is required (use --list to see supported ids)")
    if not args.repo_id or not args.filename:
        raise SystemExit("--repo-id and --filename are required (see this script's module docstring for why)")

    store = ProtocolStore(args.data_dir.resolve() / "jobs.sqlite3")
    for spec in SUPPORTED_PROTOCOL_MODELS.values():
        store.seed_model(spec)
    download_gguf_model(
        store, model_id=args.model_id, repo_id=args.repo_id, filename=args.filename,
        models_dir=(args.models_dir.resolve() / "protocol"), token=os.getenv("HF_TOKEN") or None,
        force=args.force,
    )


if __name__ == "__main__":
    main()
