"""The single, persistent registry for downloadable GigaScribe models.

Nothing in this module downloads a model.  Consequently it is safe to import in
the offline web runtime and is also shared by the downloader and readiness API.
"""
from __future__ import annotations

import hashlib, json, os, tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GIGAAM_MARKER_NAME = ".gigaam-ready.json"
PYANNOTE_MARKER_NAME = ".pyannote-ready.json"
MIN_CHECKPOINT_BYTES = 1024  # rejects HTML errors and the observed two-byte file
PYANNOTE_REPO_ID = "pyannote/speaker-diarization-3.1"
PYANNOTE_SEGMENTATION_REPO_ID = "pyannote/segmentation-3.0"
PYANNOTE_DIRNAME = "pyannote-speaker-diarization-3.1"

# Official GigaAM names passed verbatim to gigaam.load_model().  v3 RNNT is
# named e2e_rnnt in the public GigaAM API, not an alias of v2 CTC.
SUPPORTED_GIGAAM_MODELS: dict[str, dict[str, Any]] = {
    "gigaam-v2-ctc": {"kind":"asr", "id":"gigaam-v2-ctc", "label":"GigaAM v2 CTC", "model_name":"v2_ctc", "family":"v2", "architecture":"ctc", "e2e":False, "devices":["cpu","cuda"], "checkpoint":"v2_ctc.ckpt", "expected_files":["v2_ctc.ckpt"], "download_method":"gigaam.load_model", "safe_segment_seconds":24.0, "punctuation":False, "normalization":False, "speed":"быстро", "vram":"низкая"},
    "gigaam-v3-ctc": {"kind":"asr", "id":"gigaam-v3-ctc", "label":"GigaAM v3 CTC", "model_name":"v3_ctc", "family":"v3", "architecture":"ctc", "e2e":False, "devices":["cpu","cuda"], "checkpoint":"v3_ctc.ckpt", "expected_files":["v3_ctc.ckpt"], "download_method":"gigaam.load_model", "safe_segment_seconds":24.0, "punctuation":False, "normalization":False, "speed":"средне", "vram":"средняя"},
    "gigaam-v3-e2e-rnnt": {"kind":"asr", "id":"gigaam-v3-e2e-rnnt", "label":"GigaAM v3 e2e RNNT", "model_name":"v3_e2e_rnnt", "family":"v3", "architecture":"rnnt", "e2e":True, "devices":["cpu","cuda"], "checkpoint":"v3_e2e_rnnt.ckpt", "expected_files":["v3_e2e_rnnt.ckpt"], "download_method":"gigaam.load_model", "safe_segment_seconds":24.0, "punctuation":True, "normalization":True, "speed":"средне", "vram":"средняя"},
}
SUPPORTED_DIARIZATION_MODELS: dict[str, dict[str, Any]] = {
    "none": {"kind":"diarization", "id":"none", "label":"Без диаризации", "repo_id":None, "local_path":None, "nested_dependencies":[], "offline_check":"none", "pyannote_audio_versions":["3.1.*"], "devices":["cpu","cuda"], "speed":"максимальная", "vram":"0"},
    "pyannote-3.1": {"kind":"diarization", "id":"pyannote-3.1", "label":"pyannote speaker-diarization-3.1", "model_name":"speaker-diarization-3.1", "repo_id":PYANNOTE_REPO_ID, "local_path":PYANNOTE_DIRNAME, "nested_dependencies":[PYANNOTE_SEGMENTATION_REPO_ID], "offline_check":"Pipeline.from_pretrained(local config)", "pyannote_audio_versions":["3.1.*"], "devices":["cpu","cuda"], "hf_token_required":True, "terms_required":True, "speed":"средне", "vram":"средняя"},
    "pyannote-community-1": {"kind":"diarization", "id":"pyannote-community-1", "label":"pyannote speaker-diarization-community-1", "model_name":"speaker-diarization-community-1", "repo_id":"pyannote/speaker-diarization-community-1", "local_path":"pyannote-speaker-diarization-community-1", "nested_dependencies":[], "offline_check":"Pipeline.from_pretrained(local config)", "pyannote_audio_versions":["3.1.*"], "devices":["cpu","cuda"], "hf_token_required":True, "terms_required":True, "speed":"средне", "vram":"средняя"},
}
GIGAAM_CHECKPOINTS = {v["model_name"]: v["checkpoint"] for v in SUPPORTED_GIGAAM_MODELS.values()}
PROFILES = {"max_speed":{"label":"Максимальная скорость","asr_model":"gigaam-v2-ctc","diarization_model":"none","device":"cuda"}, "balanced":{"label":"Сбалансированный","asr_model":"gigaam-v3-ctc","diarization_model":"pyannote-community-1","device":"cuda"}, "compatibility":{"label":"Совместимость","asr_model":"gigaam-v2-ctc","diarization_model":"pyannote-3.1","device":"cuda"}, "cpu":{"label":"CPU","asr_model":"gigaam-v2-ctc","diarization_model":"none","device":"cpu"}}

def models_dir() -> Path: return Path(os.getenv("GIGASCRIBE_MODELS_DIR", "./models")).expanduser().resolve()
def gigaam_cache_dir(base: Path | None=None) -> Path: return (base or models_dir()) / "gigaam-cache"
def gigaam_checkpoint_path(base: Path, model_name: str) -> Path: return gigaam_cache_dir(base) / GIGAAM_CHECKPOINTS.get(model_name, f"{model_name}.ckpt")
def gigaam_marker_path(base: Path, model_name: str) -> Path: return base / "gigaam" / model_name / GIGAAM_MARKER_NAME
def pyannote_target_for(base: Path, model_id: str) -> Path: return base / (SUPPORTED_DIARIZATION_MODELS[model_id].get("local_path") or "none")
def pyannote_target(base: Path) -> Path: return pyannote_target_for(base, "pyannote-3.1")
def pyannote_marker_path(base: Path, model_id: str="pyannote-3.1") -> Path: return pyannote_target_for(base, model_id) / PYANNOTE_MARKER_NAME
def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()
def _read_json(path: Path) -> dict[str,Any] | None:
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return None
def atomic_write_json(path: Path, payload: dict[str,Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f: json.dump(payload,f,ensure_ascii=False,indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
def is_gigaam_ready(base: Path, model_name: str) -> bool:
    ckpt=gigaam_checkpoint_path(base,model_name); m=_read_json(gigaam_marker_path(base,model_name))
    return bool(m and m.get("model_name")==model_name and m.get("checkpoint")==str(ckpt) and m.get("offline_verified") is True and ckpt.is_file() and ckpt.stat().st_size >= MIN_CHECKPOINT_BYTES and m.get("checkpoint_size")==ckpt.stat().st_size and m.get("checkpoint_sha256")==sha256_file(ckpt))
def write_gigaam_marker(base: Path, model_name: str, *, package_version: str="unknown", torch_version: str="unknown") -> None:
    ckpt=gigaam_checkpoint_path(base,model_name)
    if not ckpt.is_file() or ckpt.stat().st_size < MIN_CHECKPOINT_BYTES: raise RuntimeError(f"GigaAM checkpoint is corrupt or too small: {ckpt}")
    atomic_write_json(gigaam_marker_path(base,model_name), {"model_id":next((k for k,v in SUPPORTED_GIGAAM_MODELS.items() if v["model_name"]==model_name),model_name),"model_name":model_name,"checkpoint":str(ckpt),"checkpoint_size":ckpt.stat().st_size,"checkpoint_sha256":sha256_file(ckpt),"gigaam_version":package_version,"torch_version":torch_version,"offline_verified":True,"downloaded_at":datetime.now(timezone.utc).isoformat()})
def assert_gigaam_ready(base: Path, model_name: str) -> None:
    if not is_gigaam_ready(base,model_name): raise RuntimeError(f"GigaAM checkpoint for '{model_name}' is missing, corrupt, or not offline verified. Run scripts/download_models.py --repair.")
def is_pyannote_ready(base: Path, model_id: str="pyannote-3.1") -> bool:
    if model_id=="none": return True
    meta=SUPPORTED_DIARIZATION_MODELS.get(model_id); target=pyannote_target_for(base,model_id); m=_read_json(pyannote_marker_path(base,model_id))
    return bool(meta and m and m.get("repo_id")==meta["repo_id"] and m.get("path")==str(target) and m.get("offline_verified") is True and (target/"config.yaml").is_file() and set(meta["nested_dependencies"]).issubset(set(m.get("nested_repos",[]))))
def write_pyannote_marker(base: Path, model_id: str="pyannote-3.1", *, nested_repos: list[str] | None=None) -> None:
    target=pyannote_target_for(base,model_id)
    if model_id=="none": return
    if not (target/"config.yaml").is_file(): raise RuntimeError(f"Pyannote snapshot is incomplete: {target}")
    atomic_write_json(pyannote_marker_path(base,model_id),{"model_id":model_id,"repo_id":SUPPORTED_DIARIZATION_MODELS[model_id]["repo_id"],"path":str(target),"nested_repos":nested_repos if nested_repos is not None else SUPPORTED_DIARIZATION_MODELS[model_id]["nested_dependencies"],"offline_verified":True,"downloaded_at":datetime.now(timezone.utc).isoformat()})
def settings_path(base: Path|None=None)->Path:return (base or models_dir())/"settings.json"
def default_settings()->dict[str,Any]:return {"asr_model":"gigaam-v2-ctc","diarization_model":"none","device":os.getenv("GIGASCRIBE_DEVICE","cpu")}
def load_settings(base: Path|None=None)->dict[str,Any]:
    s={**default_settings(),**(_read_json(settings_path(base)) or {})}; s["asr_model"]=s["asr_model"] if s["asr_model"] in SUPPORTED_GIGAAM_MODELS else "gigaam-v2-ctc"; s["diarization_model"]=s["diarization_model"] if s["diarization_model"] in SUPPORTED_DIARIZATION_MODELS else "none"; s["device"]=s["device"] if s["device"] in {"cpu","cuda"} else "cpu"; return s
def save_settings(settings:dict[str,Any],base:Path|None=None)->dict[str,Any]:
    s={**load_settings(base),**settings}
    if s["asr_model"] not in SUPPORTED_GIGAAM_MODELS or s["diarization_model"] not in SUPPORTED_DIARIZATION_MODELS or s["device"] not in {"cpu","cuda"}: raise ValueError("Unsupported model or device")
    atomic_write_json(settings_path(base),s); return s
