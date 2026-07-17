#!/usr/bin/env python3
"""One-shot, offline pyannote.audio 3.x worker (JSON lines on stdin/stdout)."""
from __future__ import annotations

import json, os, socket, sys, traceback
from collections import defaultdict
from pathlib import Path
from typing import Any


def deny_network() -> None:
    def blocked(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("network is disabled for the offline pyannote worker")
    socket.socket.connect = blocked  # type: ignore[assignment]
    socket.create_connection = blocked  # type: ignore[assignment]


def normalize(annotation: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    segments, durations = [], defaultdict(float)
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        start, end, speaker = float(turn.start), float(turn.end), str(speaker)
        duration = end - start
        segments.append({"start": start, "end": end, "speaker": speaker, "duration": duration})
        durations[speaker] += duration
    return segments, {"speaker_count": len(durations), "segment_count": len(segments), "speaker_durations": dict(durations)}


def run(request: dict[str, Any]) -> dict[str, Any]:
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    deny_network()
    if request.get("command", "diarize") not in {"diarize", "load_test"}:
        raise ValueError("unsupported legacy worker command")
    snapshot = Path(request["snapshot_path"]).resolve()
    config = snapshot / "config.yaml"
    if not config.is_file():
        raise RuntimeError("local pyannote snapshot has no config.yaml")
    from pyannote.audio import Pipeline
    import torch
    pipeline = Pipeline.from_pretrained(str(config))
    device = request.get("device", "cpu")
    if device not in {"cpu", "cuda"}: raise ValueError("unsupported device")
    if device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("legacy CUDA requested but unavailable")
    pipeline.to(torch.device(device))
    if request.get("command") == "load_test":
        return {"loaded": True, "pyannote_audio_version": __import__("pyannote.audio", fromlist=["__version__"]).__version__, "torch_version": torch.__version__, "device": device}
    constraints = {key: request[key] for key in ("num_speakers", "min_speakers", "max_speakers") if request.get(key) is not None}
    annotation = pipeline(request["audio_path"], **constraints)
    segments, diagnostics = normalize(annotation)
    return {"segments": segments, "diagnostics": diagnostics}


def main() -> int:
    try:
        result = run(json.loads(sys.stdin.readline()))
        print(json.dumps(result), flush=True)
        return 0
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"error": "legacy pyannote worker failed"}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
