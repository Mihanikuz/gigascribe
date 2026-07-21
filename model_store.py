"""Offline-only model registry for the single supported GigaScribe runtime."""
from __future__ import annotations
import hashlib, json, os, tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ASR_MODEL_ID = "gigaam-v3-e2e-rnnt"
ASR_MODEL_NAME = "v3_e2e_rnnt"
DIARIZATION_MODEL_ID = "pyannote-community-1"
DEVICE = "cuda"
GIGAAM_MARKER_NAME = ".gigaam-ready.json"
PYANNOTE_MARKER_NAME = ".pyannote-ready.json"
MIN_CHECKPOINT_BYTES = 1024
MIN_TOKENIZER_BYTES = 128
GIGAAM_SOURCE = "https://github.com/salute-developers/GigaAM.git"
GIGAAM_BACKEND_REF = "v0.3.0"
SUPPORTED_DEVICES = ("cuda",)
SUPPORTED_GIGAAM_MODELS: dict[str, dict[str, Any]] = {
    ASR_MODEL_ID: {"kind":"asr","id":ASR_MODEL_ID,"label":"GigaAM v3 E2E RNNT","model_name":ASR_MODEL_NAME,"official_model_name":ASR_MODEL_NAME,"family":"v3","architecture":"rnnt","e2e":True,"devices":[DEVICE],"checkpoint":"v3_e2e_rnnt.ckpt","tokenizer":"v3_e2e_rnnt_tokenizer.model","expected_files":["v3_e2e_rnnt.ckpt","v3_e2e_rnnt_tokenizer.model"],"download_method":"gigaam.load_model","safe_segment_seconds":24.0,"punctuation":True,"normalization":True,"vram_gb_min":8}
}
SUPPORTED_DIARIZATION_MODELS: dict[str, dict[str, Any]] = {
    "none": {"kind":"diarization","id":"none","label":"Без диаризации","repo_id":None,"local_path":None,"devices":[DEVICE]},
    DIARIZATION_MODEL_ID: {"kind":"diarization","id":DIARIZATION_MODEL_ID,"label":"pyannote Community-1","model_name":"speaker-diarization-community-1","repo_id":"pyannote/speaker-diarization-community-1","local_path":"pyannote/community-1","runtime":"community","pyannote_audio_versions":["4.*"],"devices":[DEVICE],"vram_gb_min":4},
}
PROFILES: dict[str, Any] = {}

def models_dir() -> Path: return Path(os.getenv("GIGASCRIBE_MODELS_DIR", "./models")).expanduser().resolve()
def gigaam_model_dir(base: Path|None=None) -> Path: return (base or models_dir())/"gigaam"/"v3_e2e_rnnt"
def gigaam_cache_dir(base: Path|None=None) -> Path: return gigaam_model_dir(base)
def gigaam_checkpoint_path(base: Path, model_name: str=ASR_MODEL_NAME) -> Path: return gigaam_model_dir(base)/"v3_e2e_rnnt.ckpt"
def gigaam_tokenizer_path(base: Path) -> Path: return gigaam_model_dir(base)/"v3_e2e_rnnt_tokenizer.model"
def gigaam_marker_path(base: Path, model_name: str=ASR_MODEL_NAME) -> Path: return gigaam_model_dir(base)/GIGAAM_MARKER_NAME
def pyannote_target_for(base: Path, model_id: str=DIARIZATION_MODEL_ID) -> Path:
    if model_id == "none": return base/"none"
    if model_id != DIARIZATION_MODEL_ID: raise ValueError(f"Unsupported diarization model: {model_id}")
    return base/"pyannote"/"community-1"
def pyannote_target(base: Path) -> Path: return pyannote_target_for(base, DIARIZATION_MODEL_ID)
def pyannote_marker_path(base: Path, model_id: str=DIARIZATION_MODEL_ID) -> Path: return pyannote_target_for(base, model_id)/PYANNOTE_MARKER_NAME
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
def _torch_version() -> str:
    try:
        import torch; return str(torch.__version__)
    except Exception: return "unavailable"
def validate_gigaam_files(base: Path) -> tuple[bool, str|None]:
    ckpt, tok = gigaam_checkpoint_path(base), gigaam_tokenizer_path(base)
    if not ckpt.is_file() or ckpt.stat().st_size < MIN_CHECKPOINT_BYTES: return False, "ASR_MODEL_NOT_FOUND: checkpoint missing or too small"
    if not tok.is_file() or tok.stat().st_size < MIN_TOKENIZER_BYTES: return False, "ASR_MODEL_NOT_FOUND: tokenizer missing or too small"
    m=_read_json(gigaam_marker_path(base))
    if not m: return False, "ASR marker missing"
    if m.get("model_id") != ASR_MODEL_ID or m.get("model_name") != ASR_MODEL_NAME: return False, "ASR marker model_id mismatch"
    if m.get("checkpoint_path") != str(ckpt) or m.get("tokenizer_path") != str(tok): return False, "ASR marker path mismatch"
    if m.get("backend_commit_or_tag") != GIGAAM_BACKEND_REF or m.get("offline_verified") is not True: return False, "ASR marker backend/offline mismatch"
    if m.get("size") != ckpt.stat().st_size or m.get("sha256") != sha256_file(ckpt): return False, "ASR checkpoint checksum mismatch"
    return True, None
def is_gigaam_ready(base: Path, model_name: str=ASR_MODEL_NAME) -> bool: return model_name == ASR_MODEL_NAME and validate_gigaam_files(base)[0]
def assert_gigaam_ready(base: Path, model_name: str=ASR_MODEL_NAME) -> None:
    ok, err = validate_gigaam_files(base)
    if not ok: raise RuntimeError(err or "ASR_MODEL_NOT_FOUND")
def write_gigaam_marker(base: Path, model_name: str=ASR_MODEL_NAME, *, offline_reload_verified: bool=False, **_: Any) -> None:
    if model_name != ASR_MODEL_NAME: raise ValueError("Only RNNT is supported")
    ckpt,tok=gigaam_checkpoint_path(base),gigaam_tokenizer_path(base)
    if not offline_reload_verified: raise RuntimeError("Refusing to create GigaAM marker before offline reload")
    if not ckpt.is_file() or not tok.is_file(): raise RuntimeError("ASR files are incomplete")
    atomic_write_json(gigaam_marker_path(base), {"model_id":ASR_MODEL_ID,"model_name":ASR_MODEL_NAME,"official_model_name":ASR_MODEL_NAME,"checkpoint_path":str(ckpt),"tokenizer_path":str(tok),"size":ckpt.stat().st_size,"sha256":sha256_file(ckpt),"backend_source":GIGAAM_SOURCE,"backend_commit_or_tag":GIGAAM_BACKEND_REF,"torch_version":_torch_version(),"device_used_for_test":"cuda","offline_verified":True,"offline_reload_verified":True,"created_at":datetime.now(timezone.utc).isoformat()})
def validate_pyannote_files(base: Path, model_id: str=DIARIZATION_MODEL_ID) -> tuple[bool, str|None]:
    if model_id == "none": return True, None
    target=pyannote_target_for(base, model_id); marker=_read_json(pyannote_marker_path(base, model_id))
    if not (target/"config.yaml").is_file(): return False, "DIARIZATION_MODEL_NOT_FOUND: config.yaml missing"
    if not marker: return False, "pyannote marker missing"
    if marker.get("model_id") != model_id or marker.get("repo_id") != SUPPORTED_DIARIZATION_MODELS[model_id]["repo_id"]: return False, "pyannote marker model_id mismatch"
    if marker.get("snapshot_path", marker.get("path")) != str(target): return False, "pyannote marker path mismatch"
    if marker.get("runtime_type") != "community" or marker.get("offline_verified") is not True: return False, "pyannote marker offline/runtime mismatch"
    return True, None
def is_pyannote_ready(base: Path, model_id: str=DIARIZATION_MODEL_ID) -> bool: return validate_pyannote_files(base, model_id)[0]
def write_pyannote_marker(base: Path, model_id: str=DIARIZATION_MODEL_ID, **_: Any) -> None:
    target=pyannote_target_for(base, model_id)
    if not (target/"config.yaml").is_file(): raise RuntimeError(f"Pyannote snapshot is incomplete: {target}")
    atomic_write_json(pyannote_marker_path(base, model_id), {"model_id":model_id,"repo_id":SUPPORTED_DIARIZATION_MODELS[model_id]["repo_id"],"snapshot_path":str(target),"runtime_type":"community","pyannote_audio_version":"4.x","torch_version":_torch_version(),"device_used_for_test":"cuda","offline_verified":True,"created_at":datetime.now(timezone.utc).isoformat()})
def settings_path(base: Path|None=None)->Path:return (base or models_dir())/"settings.json"
def default_settings()->dict[str,Any]:return {"asr_model":ASR_MODEL_ID,"diarization_model":"none","device":DEVICE}
def load_settings(base: Path|None=None)->dict[str,Any]:
    raw=_read_json(settings_path(base)) or {}; diar=raw.get("diarization_model", "none")
    return {"asr_model":ASR_MODEL_ID,"diarization_model":diar if diar in SUPPORTED_DIARIZATION_MODELS else "none","device":DEVICE}
def save_settings(settings:dict[str,Any],base:Path|None=None)->dict[str,Any]:
    diar=settings.get("diarization_model", load_settings(base)["diarization_model"])
    if diar not in SUPPORTED_DIARIZATION_MODELS: raise ValueError("Unsupported diarization model")
    s={"asr_model":ASR_MODEL_ID,"diarization_model":diar,"device":DEVICE}; atomic_write_json(settings_path(base),s); return s
