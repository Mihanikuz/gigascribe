"""Local multi-user web server for GigaScribe on Ubuntu 24.04.

Features:
- local username/password authentication with optional LDAP/AD bind;
- per-user asynchronous transcription jobs;
- live progress polling from the browser;
- downloads for transcript, job log, original audio, and M4A/FLAC exports;
- local-only model loading configured by environment variables.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import secrets
import hashlib
import shutil
import time
import uuid
import logging
import importlib
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, JSONResponse
from starlette.middleware.sessions import SessionMiddleware

APP_MODULE_PATH = Path(__file__).with_name("app.py")
spec = importlib.util.spec_from_file_location("gigascribe_gradio_app", APP_MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load {APP_MODULE_PATH}")
giga_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(giga_app)

MODELS_DIR = Path(os.getenv("GIGASCRIBE_MODELS_DIR", "./models")).resolve()
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

BASE_DIR = Path(os.getenv("GIGASCRIBE_DATA_DIR", "./data")).resolve()
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
USER_DB = Path(os.getenv("GIGASCRIBE_USERS_FILE", BASE_DIR / "users.json"))
SECRET_KEY = os.getenv("GIGASCRIBE_SECRET_KEY", secrets.token_urlsafe(32))
SECRET_KEY_CONFIGURED = bool(os.getenv("GIGASCRIBE_SECRET_KEY"))
MAX_UPLOAD_BYTES = int(os.getenv("GIGASCRIBE_MAX_UPLOAD_BYTES", str(1024 * 1024 * 1024)))
ORIGINAL_RETENTION_DAYS = int(os.getenv("GIGASCRIBE_ORIGINAL_RETENTION_DAYS", "14"))
# Explicit allowlist of downloadable artefacts: WAV is deliberately not
# exposed here (it's only an internal intermediate for ASR), and the
# download endpoint below rejects anything not in this mapping outright
# rather than building a column name from the requested `kind`.
DOWNLOAD_KINDS = {"transcript": "transcript_path", "log": "log_path", "original": "original_path", "m4a": "m4a_path", "flac": "flac_path"}
for directory in (UPLOAD_DIR, RESULT_DIR, USER_DB.parent):
    directory.mkdir(parents=True, exist_ok=True)

class SafeBcryptContext:
    def hash(self, password: str) -> str:
        import bcrypt
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")
    def verify(self, password: str, hashed: str) -> bool:
        try:
            if len(password.encode("utf-8")) > 72:
                return False
            import bcrypt
            return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("ascii"))
        except Exception:
            return False
pwd_context = SafeBcryptContext()
logger = logging.getLogger(__name__)
app = FastAPI(title="GigaScribe Local Server")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, same_site=os.getenv("GIGASCRIBE_SESSION_SAMESITE", "lax"), https_only=os.getenv("GIGASCRIBE_SESSION_SECURE", "0") == "1")

@app.on_event("startup")
async def resume_persistent_queue() -> None:
    # Validate GIGASCRIBE_ADMIN_PASSWORD (and create the admin account) at
    # startup instead of lazily on the first login attempt, so a missing or
    # placeholder password fails loudly in the container logs rather than as
    # an unexplained 500 the first time someone tries to log in.
    _load_users()
    await asyncio.get_running_loop().run_in_executor(None, cleanup_expired_originals)
    asyncio.create_task(periodic_original_cleanup())
    for row in job_store.list(active_only=True):
        if row["status"] == "queued": schedule_job(row["id"])
    if PROTOCOL_SERVICE is not None:
        # Never leave a protocol job claiming to hold the GPU after a
        # restart; the user must explicitly retry it.
        resumed = await asyncio.get_running_loop().run_in_executor(None, PROTOCOL_SERVICE.resume_after_restart)
        if resumed:
            logger.info("Marked %d interrupted protocol job(s) as failed after restart", resumed)


from job_store import JobStore
job_store = JobStore(BASE_DIR / "jobs.sqlite3")
job_store.recover()
# GPU work is serialized: one CUDA worker per container.
gpu_worker = giga_app.GPU_JOB_LOCK
scheduled_jobs: set[str] = set()
model_download_locks: dict[str, asyncio.Lock] = {}
MODEL_MANAGER = giga_app.MODEL_MANAGER

# --- Protocol (meeting-minutes LLM) module: fully optional and isolated.
# The env var is checked with a plain string compare *before* importing
# anything from protocol/, so protocol/ is genuinely never imported when
# disabled -- a missing or broken protocol/ package cannot stop GigaScribe
# from starting in that case. When enabled but broken, we log and degrade
# to disabled rather than take down transcription with it.
PROTOCOL_ENABLED = os.getenv("GIGASCRIBE_PROTOCOL_ENABLED", "0") == "1"
PROTOCOL_SERVICE = None
if PROTOCOL_ENABLED:
    try:
        from protocol import ProtocolConfig, ProtocolService
        from protocol.store import ProtocolStore
        _protocol_config = ProtocolConfig.from_env(data_dir=BASE_DIR, models_dir=MODELS_DIR)
        _protocol_store = ProtocolStore(BASE_DIR / "jobs.sqlite3")
        PROTOCOL_SERVICE = ProtocolService(config=_protocol_config, store=_protocol_store,
                                            gpu_lock=gpu_worker, unload_asr=MODEL_MANAGER.unload_all)
    except Exception:
        logger.exception("Protocol module failed to initialize; continuing with it disabled")
        PROTOCOL_ENABLED = False
        PROTOCOL_SERVICE = None


def _validate_password(password: str) -> None:
    byte_len = len(password.encode("utf-8"))
    if len(password) < 6:
        raise HTTPException(400, detail="Пароль должен содержать не менее 6 символов")
    if byte_len > 72:
        raise HTTPException(400, detail="Пароль не должен превышать 72 байта для bcrypt")
    if byte_len > 4096:
        raise HTTPException(400, detail="Пароль слишком длинный")

FORBIDDEN_ADMIN_PASSWORDS = {
    "admin", "password", "change-me", "changeme",
    "replace-with-a-strong-password",
}

def _load_users() -> dict[str, Any]:
    if not USER_DB.exists():
        admin_password = os.getenv("GIGASCRIBE_ADMIN_PASSWORD")
        if not admin_password or admin_password.lower() in FORBIDDEN_ADMIN_PASSWORDS:
            raise RuntimeError("GIGASCRIBE_ADMIN_PASSWORD must be set to a real password, not a placeholder like 'admin' or 'change-me'")
        _validate_password(admin_password)
        USER_DB.write_text(
            json.dumps({"admin": {"password_hash": pwd_context.hash(admin_password), "disabled": False}}, indent=2),
            encoding="utf-8",
        )
    return json.loads(USER_DB.read_text(encoding="utf-8"))


def _verify_local(username: str, password: str) -> bool:
    users = _load_users()
    user = users.get(username)
    return bool(user and not user.get("disabled") and pwd_context.verify(password, user.get("password_hash", "")))


def _verify_ldap(username: str, password: str) -> bool:
    server_uri = os.getenv("GIGASCRIBE_LDAP_SERVER")
    domain = os.getenv("GIGASCRIBE_AD_DOMAIN")
    if not server_uri or not password:
        return False
    try:
        from ldap3 import ALL, Connection, Server

        bind_user = f"{domain}\\{username}" if domain and "@" not in username and "\\" not in username else username
        server = Server(server_uri, get_info=ALL)
        with Connection(server, user=bind_user, password=password, auto_bind=True):
            return True
    except Exception:
        return False


def authenticate(username: str, password: str) -> bool:
    return _verify_local(username, password) or _verify_ldap(username, password)


def current_user(request: Request) -> str:
    username = request.session.get("user")
    if not username:
        raise HTTPException(status_code=401)
    return username


def safe_name(name: str) -> str:
    cleaned = "".join(ch for ch in Path(name).name if ch.isalnum() or ch in "._- ()").strip()
    return cleaned or "audio"


def safe_user_id(username: str) -> str:
    digest = hashlib.sha256(username.encode("utf-8")).hexdigest()[:24]
    return f"user-{digest}"


def ensure_inside(path: Path, base: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(base):
        raise HTTPException(status_code=400, detail="Invalid storage path")
    return resolved


def ensure_inside_base(path: Path) -> Path:
    return ensure_inside(path, BASE_DIR)


def cleanup_expired_originals() -> int:
    """Delete raw audio (uploaded original and the internal normalized WAV)
    older than ORIGINAL_RETENTION_DAYS.

    Only original_path/wav_path are removed; transcript/log/M4A/FLAC and the
    job row itself are untouched. Queued/running jobs are never touched.
    """
    if ORIGINAL_RETENTION_DAYS <= 0:
        return 0
    cutoff = time.time() - ORIGINAL_RETENTION_DAYS * 86400
    deleted = 0
    for job in job_store.list_deletable_originals(cutoff):
        changes: dict[str, Any] = {}
        for column in ("original_path", "wav_path"):
            raw = job.get(column)
            if not raw:
                continue
            try:
                resolved = ensure_inside_base(Path(raw))
            except HTTPException:
                logger.warning("Refusing to delete %s outside data dir job_id=%s path=%s", column, job["id"], raw)
                continue
            resolved.unlink(missing_ok=True)
            changes[column] = None
        if not changes:
            continue
        job_store.update(job["id"], **changes)
        deleted += 1
    if deleted:
        logger.info("Cleaned raw audio for %d expired job(s) (original/normalized WAV) older than %d day(s)", deleted, ORIGINAL_RETENTION_DAYS)
    return deleted


async def periodic_original_cleanup() -> None:
    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(24 * 3600)
        await loop.run_in_executor(None, cleanup_expired_originals)


def is_admin(username: str) -> bool:
    users = _load_users()
    return bool(username == "admin" and users.get(username) and not users[username].get("disabled"))

def require_admin(username: str = Depends(current_user)) -> str:
    if not is_admin(username): raise HTTPException(403, detail="Administrator authorization required")
    return username


def validate_upload_name(filename: str) -> str:
    safe = safe_name(filename or "audio")
    if Path(safe).suffix.lower() not in giga_app.SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail="Unsupported file extension")
    return safe


@app.get("/health/live")
def health_live():
    return {"status": "live"}


def _writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _import_ok(name: str) -> bool:
    try:
        importlib.import_module(name); return True
    except Exception: return False

def _model_checks(load: bool = False) -> dict[str, Any]:
    return MODEL_MANAGER.health_status(deep=load)

@app.get("/health/ready")
def health_ready():
    # Deep: actually load ASR/diarization once (cheap on later polls, since
    # ModelManager reuses already-loaded models) rather than only checking
    # that files exist. A broken import (e.g. torchcodec) or GPU placement
    # failure should make the container not-ready, not surface as the first
    # user's job failing at 1%.
    status = MODEL_MANAGER.health_status(deep=True)
    checks = {"data_writable": _writable_dir(BASE_DIR), "models_writable": _writable_dir(MODELS_DIR), "secret_key_configured": SECRET_KEY_CONFIGURED, "ffmpeg": shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None}
    ready = status["status"] == "ready" and all(checks.values())
    status["checks"] = checks
    status["status"] = "ready" if ready else "not_ready"
    return JSONResponse(status, status_code=200 if ready else 503)

@app.get("/health/models")
def health_models(): return _model_checks(load=True)

@app.get("/health/gpu")
def health_gpu():
    from system_info import gpu_info
    return gpu_info(real_test=True)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    if not request.session.get("user"):
        return RedirectResponse("/login")
    return HTMLResponse(INDEX_HTML)


@app.get("/login", response_class=HTMLResponse)
def login_page():
    return HTMLResponse(LOGIN_HTML)


@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if not authenticate(username, password):
        return HTMLResponse(LOGIN_HTML.replace("<!--ERROR-->", "<p class='err'>Неверные учетные данные</p>"), status_code=401)
    request.session["user"] = username
    return RedirectResponse("/", status_code=303)


@app.post("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...), username: str = Depends(current_user)):
    filename = validate_upload_name(file.filename or "audio")
    job_id = uuid.uuid4().hex
    user_dir = safe_user_id(username)
    user_upload_dir = ensure_inside_base(UPLOAD_DIR / user_dir / job_id)
    user_result_dir = ensure_inside_base(RESULT_DIR / user_dir / job_id)
    user_upload_dir.mkdir(parents=True, exist_ok=True)
    user_result_dir.mkdir(parents=True, exist_ok=True)
    original_path = ensure_inside_base(user_upload_dir / filename)
    written = 0
    try:
        with original_path.open("wb") as fh:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Uploaded file is too large")
                fh.write(chunk)
    except Exception:
        if original_path.exists():
            original_path.unlink()
        raise
    from model_store import load_settings
    job_store.create(id=job_id, username=username, filename=original_path.name, original_path=str(original_path), log_path=str(user_result_dir / "job.log"), settings_snapshot=load_settings(MODELS_DIR), message="В очереди", timeout_seconds=int(os.getenv("GIGASCRIBE_JOB_TIMEOUT_SECONDS", "0")) or None)
    schedule_job(job_id)
    return {"job_id": job_id}


def schedule_job(job_id: str) -> None:
    if job_id not in scheduled_jobs:
        scheduled_jobs.add(job_id)
        asyncio.create_task(run_job(job_id))


async def run_job(job_id: str) -> None:
    job = job_store.claim(job_id)
    if not job: return
    result_dir = Path(job["log_path"]).parent
    def log(line: str) -> None:
        with Path(job["log_path"]).open("a", encoding="utf-8") as fh: fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")
    def progress(value: float, message: str) -> None:
        value = round(max(0.0, min(1.0, value)), 3); job_store.update(job_id, progress=value, message=message); log(f"{value:.0%} {message}")
    try:
        snapshot = job["settings_snapshot"]; progress(.01, "Инициализация задания")
        import torch
        log(f"job_id={job_id}")
        log(f"ASR model: {snapshot.get('asr_model')}")
        log(f"Diarization model: {snapshot.get('diarization_model')}")
        log(f"Device: {snapshot.get('device')}")
        log(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable'}")
        log(f"PyTorch version: {torch.__version__}")
        log(f"CUDA runtime: {torch.version.cuda}")
        from model_store import gigaam_cache_dir, pyannote_target_for
        log(f"ASR model path: {gigaam_cache_dir(MODELS_DIR)}")
        log(f"Diarization model path: {pyannote_target_for(MODELS_DIR, snapshot.get('diarization_model', 'none'))}")
        t0=time.perf_counter(); processor = giga_app.AudioProcessor(snapshot=snapshot, model_manager=MODEL_MANAGER)
        if job_store.get(job_id)["cancel_requested"]:
            job_store.update(job_id,status="cancelled",finished_at=time.time(),message="Отменено"); return
        async with gpu_worker:
            if job_store.get(job_id)["cancel_requested"]:
                job_store.update(job_id,status="cancelled",finished_at=time.time(),message="Отменено"); return
            await processor.initialize_models(lambda p,m: progress(p*.1,m)); log(f"model_initialization_seconds={time.perf_counter()-t0:.3f}")
            if job_store.get(job_id)["cancel_requested"]:
                job_store.update(job_id,status="cancelled",finished_at=time.time(),message="Отменено"); return
            work = processor.process_single_file(job["original_path"], progress, original_filename=job["filename"], artifacts_dir=str(result_dir))
            result = await asyncio.wait_for(work, job["timeout_seconds"]) if job["timeout_seconds"] else await work
            log(f"total_seconds={time.perf_counter()-t0:.3f}")
        if job_store.get(job_id)["cancel_requested"]:
            job_store.update(job_id,status="cancelled",finished_at=time.time(),message="Отменено"); return
        transcript = result_dir / f"{Path(job['filename']).stem}.txt"; transcript.write_text(result.full_text, encoding="utf-8")
        stem = Path(job['filename']).stem
        wav=result_dir/"normalized.wav"; m4a=result_dir/f"{stem}.m4a"; flac=result_dir/f"{stem}.flac"
        job_store.update(job_id,status="completed",finished_at=time.time(),progress=1,message="Готово",transcript_path=str(transcript),wav_path=str(wav) if wav.exists() else None,m4a_path=str(m4a) if m4a.exists() else None,flac_path=str(flac) if flac.exists() else None,actual_device=str(processor.device),actual_models={"asr_model":snapshot.get("asr_model"),"diarization_model":snapshot.get("diarization_model")})
    except asyncio.TimeoutError:
        job_store.update(job_id,status="failed",finished_at=time.time(),error="Job timed out",message="Превышено время выполнения")
    except Exception as exc:
        job_store.update(job_id,status="failed",finished_at=time.time(),error=str(exc),message="Ошибка обработки")
        logger.exception("transcription failed job_id=%s model=%s device=%s", job_id, job["settings_snapshot"], job["settings_snapshot"].get("device"))
    finally: scheduled_jobs.discard(job_id)


def can_control_job(job: dict[str, Any], username: str) -> bool:
    return job["username"] == username or is_admin(username)


@app.get("/api/jobs")
async def list_jobs(username: str = Depends(current_user)):
    # Transcription history is shared across all users of this deployment.
    return [serialize_job(j) for j in job_store.list()]

@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str, username: str = Depends(current_user)):
    job=job_store.get(job_id)
    if not job: raise HTTPException(404)
    return serialize_job(job)

@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, username: str = Depends(current_user)):
    job=job_store.get(job_id)
    if not job: raise HTTPException(404)
    if not can_control_job(job, username): raise HTTPException(403, detail="Only the job owner or an administrator can cancel it")
    if job["status"] in {"completed","failed","cancelled"}: raise HTTPException(409, detail="Job is already terminal")
    job_store.request_cancel(job_id)
    if job["status"] == "queued": job_store.update(job_id,status="cancelled",finished_at=time.time())
    return serialize_job(job_store.get(job_id))

@app.post("/api/jobs/{job_id}/retry")
async def retry_job(job_id: str, username: str = Depends(current_user)):
    job=job_store.get(job_id)
    if not job: raise HTTPException(404)
    if not can_control_job(job, username): raise HTTPException(403, detail="Only the job owner or an administrator can retry it")
    if job["status"] not in {"failed","cancelled"}: raise HTTPException(409, detail="Only failed or cancelled jobs can be retried")
    if not job["original_path"] or not Path(job["original_path"]).exists():
        raise HTTPException(410, detail="Исходный файл удалён по сроку хранения")
    try: job_store.retry(job_id)
    except ValueError as exc: raise HTTPException(409, detail=str(exc))
    # Preserve upload and log but retire other result artefacts: stale
    # downloads cannot become the result of the next attempt. The log is kept
    # (run_job appends to it) so earlier attempts stay visible after a retry.
    result_dir = Path(job["log_path"]).parent
    log_path = Path(job["log_path"])
    for artifact in result_dir.iterdir() if result_dir.exists() else ():
        if artifact.is_file() and artifact != log_path: artifact.unlink(missing_ok=True)
    if log_path.exists():
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} ==== Retry requested (previous status: {job['status']}) ====\n")
    schedule_job(job_id)
    return serialize_job(job_store.get(job_id))

def _protocol_summary(job: dict[str, Any]) -> dict[str, Any]:
    if not PROTOCOL_ENABLED or PROTOCOL_SERVICE is None:
        return {"enabled": False}
    latest = None
    existing = PROTOCOL_SERVICE.store.list_protocol_jobs_for_job(job["id"])
    if existing:
        latest = existing[0]
    return {
        "enabled": True,
        "can_create": job["status"] == "completed" and bool(job.get("transcript_path")),
        "protocol_job_id": latest["id"] if latest else None,
        "status": latest["status"] if latest else None,
        "progress": latest["progress"] if latest else None,
        "message": latest["message"] if latest else None,
        "model_id": latest["model_id"] if latest else None,
        "chunk_current": latest["chunk_current"] if latest else None,
        "chunk_total": latest["chunk_total"] if latest else None,
        "ready": bool(latest and latest["status"] == "completed"),
        "failed": bool(latest and latest["status"] in ("failed", "cancelled")),
    }


def serialize_job(job: dict[str, Any]) -> dict[str, Any]:
    return {"id":job["id"],"owner":job["username"],"filename":job["filename"],"status":job["status"],"progress":job["progress"],"message":job["message"],"error":job["error"],"settings":job["settings_snapshot"],"requested_models":job["requested_models"],"requested_device":job["requested_device"],"actual_device":job["actual_device"],"actual_models":job["actual_models"],"attempts":job["attempts"],"correlation_id":job["correlation_id"],"downloads":{k:bool(job.get(col)) for k,col in DOWNLOAD_KINDS.items()},"protocol":_protocol_summary(job)}

@app.get("/api/jobs/{job_id}/download/{kind}")
def download(job_id: str, kind: str, username: str = Depends(current_user)):
    # Transcription history (including downloads) is shared across users.
    column = DOWNLOAD_KINDS.get(kind)
    if not column: raise HTTPException(404)
    job=job_store.get(job_id)
    if not job: raise HTTPException(404)
    path=job.get(column)
    if not path: raise HTTPException(404)
    try:
        resolved = ensure_inside_base(Path(path))
    except HTTPException:
        raise HTTPException(404)
    if not resolved.exists(): raise HTTPException(404)
    return FileResponse(resolved, filename=resolved.name)


def _require_protocol_enabled() -> None:
    if not PROTOCOL_ENABLED or PROTOCOL_SERVICE is None:
        raise HTTPException(404, detail="Protocol module is disabled")


def _latest_completed_protocol(job_id: str) -> dict[str, Any] | None:
    for p in PROTOCOL_SERVICE.store.list_protocol_jobs_for_job(job_id):
        if p["status"] == "completed":
            return p
    return None


def _resolved_protocol_path(path_str: str) -> Path:
    try:
        resolved = ensure_inside_base(Path(path_str))
    except HTTPException:
        raise HTTPException(404)
    if not resolved.exists():
        raise HTTPException(404)
    return resolved


@app.get("/api/protocol/status")
def api_protocol_status(username: str = Depends(current_user)):
    if not PROTOCOL_ENABLED or PROTOCOL_SERVICE is None:
        return {"enabled": False, "has_installed_model": False, "active_model_id": None}
    models = PROTOCOL_SERVICE.store.list_model_states()
    return {
        "enabled": True,
        "has_installed_model": any(m["installed"] for m in models),
        "active_model_id": PROTOCOL_SERVICE.store.active_model_id(),
    }


@app.post("/api/jobs/{job_id}/protocol")
async def create_protocol(job_id: str, username: str = Depends(current_user)):
    _require_protocol_enabled()
    job = job_store.get(job_id)
    if not job: raise HTTPException(404)
    if not can_control_job(job, username): raise HTTPException(403, detail="Only the job owner or an administrator can create a protocol")
    if job["status"] != "completed" or not job.get("transcript_path"):
        raise HTTPException(409, detail="Transcript is not ready yet")
    from protocol.schemas import ProtocolOptions
    try:
        result = await PROTOCOL_SERVICE.create_protocol(transcript_path=Path(job["transcript_path"]), job_id=job_id,
                                                          username=username, options=ProtocolOptions())
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc))
    except FileNotFoundError:
        raise HTTPException(404, detail="Transcript file not found")
    return {"protocol_job_id": result.protocol_job_id, "status": result.status}


@app.post("/api/jobs/{job_id}/protocol/retry")
async def retry_protocol(job_id: str, username: str = Depends(current_user)):
    _require_protocol_enabled()
    job = job_store.get(job_id)
    if not job: raise HTTPException(404)
    if not can_control_job(job, username): raise HTTPException(403, detail="Only the job owner or an administrator can retry it")
    if not job.get("transcript_path"):
        raise HTTPException(409, detail="Transcript is not ready yet")
    existing = PROTOCOL_SERVICE.store.list_protocol_jobs_for_job(job_id)
    if not existing: raise HTTPException(404, detail="No protocol job to retry")
    try:
        result = await PROTOCOL_SERVICE.retry(existing[0]["id"], Path(job["transcript_path"]), job_id)
    except ValueError as exc:
        raise HTTPException(409, detail=str(exc))
    return {"protocol_job_id": result.protocol_job_id, "status": result.status}


@app.post("/api/jobs/{job_id}/protocol/cancel")
def cancel_protocol(job_id: str, username: str = Depends(current_user)):
    _require_protocol_enabled()
    job = job_store.get(job_id)
    if not job: raise HTTPException(404)
    if not can_control_job(job, username): raise HTTPException(403)
    existing = PROTOCOL_SERVICE.store.list_protocol_jobs_for_job(job_id)
    if not existing: raise HTTPException(404)
    PROTOCOL_SERVICE.request_cancel(existing[0]["id"])
    return {"ok": True}


@app.get("/api/jobs/{job_id}/protocol/status")
def protocol_job_status(job_id: str, username: str = Depends(current_user)):
    _require_protocol_enabled()
    job = job_store.get(job_id)
    if not job: raise HTTPException(404)
    return _protocol_summary(job)


@app.get("/api/jobs/{job_id}/protocol", response_class=HTMLResponse)
def view_protocol(job_id: str, username: str = Depends(current_user)):
    _require_protocol_enabled()
    job = job_store.get(job_id)
    if not job: raise HTTPException(404)
    completed = _latest_completed_protocol(job_id)
    if not completed or not completed.get("html_path"):
        raise HTTPException(404, detail="Protocol is not ready")
    resolved = _resolved_protocol_path(completed["html_path"])
    return HTMLResponse(resolved.read_text(encoding="utf-8"))


@app.get("/api/jobs/{job_id}/protocol.json")
def download_protocol_json(job_id: str, username: str = Depends(current_user)):
    _require_protocol_enabled()
    job = job_store.get(job_id)
    if not job: raise HTTPException(404)
    completed = _latest_completed_protocol(job_id)
    if not completed or not completed.get("json_path"):
        raise HTTPException(404)
    resolved = _resolved_protocol_path(completed["json_path"])
    return FileResponse(resolved, filename=resolved.name, media_type="application/json")


@app.get("/api/jobs/{job_id}/protocol.html")
def download_protocol_html(job_id: str, username: str = Depends(current_user)):
    _require_protocol_enabled()
    job = job_store.get(job_id)
    if not job: raise HTTPException(404)
    completed = _latest_completed_protocol(job_id)
    if not completed or not completed.get("html_path"):
        raise HTTPException(404)
    resolved = _resolved_protocol_path(completed["html_path"])
    return FileResponse(resolved, filename=resolved.name, media_type="text/html")


@app.get("/api/jobs/{job_id}/protocol.xml")
def download_protocol_xml(job_id: str, username: str = Depends(current_user)):
    _require_protocol_enabled()
    job = job_store.get(job_id)
    if not job: raise HTTPException(404)
    completed = _latest_completed_protocol(job_id)
    if not completed or not completed.get("xml_path"):
        raise HTTPException(404)
    resolved = _resolved_protocol_path(completed["xml_path"])
    return FileResponse(resolved, filename=resolved.name, media_type="application/xml")


@app.post("/api/jobs/{job_id}/glossary-suggestion")
def create_glossary_suggestion(job_id: str, payload: dict[str, Any], username: str = Depends(current_user)):
    """Item 10: a user (owner or admin, matching every other job-control
    check in this file) can propose a correction seen on the transcript or
    protocol page. This only ever creates a `status='proposed'` suggestion
    -- an admin must confirm it before it can affect anything, exactly like
    an LLM-proposed term (see glossary.py's module docstring)."""
    _require_protocol_enabled()
    job = job_store.get(job_id)
    if not job: raise HTTPException(404)
    if not can_control_job(job, username):
        raise HTTPException(403, detail="Only the job owner or an administrator can suggest a correction")
    wrong_text = str(payload.get("wrong_text") or "").strip()
    suggested_text = str(payload.get("suggested_text") or "").strip()
    if not wrong_text or not suggested_text:
        raise HTTPException(400, detail="wrong_text and suggested_text are required")
    from protocol import glossary as glossary_mod
    suggestion = glossary_mod.propose_from_user_correction(
        PROTOCOL_SERVICE.store, wrong_text=wrong_text, suggested_text=suggested_text,
        job_id=job_id, context=str(payload.get("context") or ""),
    )
    return {"ok": True, "suggestion_id": suggestion.id, "status": suggestion.status}


@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request):
    username = request.session.get("user")
    if not username:
        return RedirectResponse("/login")
    if not is_admin(username):
        return HTMLResponse(ADMIN_DENIED_HTML, status_code=403)
    return HTMLResponse(ADMIN_HTML)


@app.get("/admin/users")
def list_local_users(admin: str = Depends(require_admin)):
    users = _load_users()
    return {"users": [
        {"username": name, "disabled": bool(info.get("disabled")), "is_admin": name == "admin"}
        for name, info in sorted(users.items())
    ]}


@app.post("/admin/users")
def create_local_user(admin: str = Depends(require_admin), username: str = Form(...), password: str = Form(...)):
    if not username or len(username) > 128:
        raise HTTPException(400, detail="Invalid username")
    users = _load_users()
    if username in users:
        raise HTTPException(409, detail="User already exists")
    _validate_password(password)
    users[username] = {"password_hash": pwd_context.hash(password), "disabled": False, "storage_id": safe_user_id(username)}
    USER_DB.write_text(json.dumps(users, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"ok": True}


def _set_user_disabled(target: str, disabled: bool) -> None:
    if target == "admin":
        raise HTTPException(400, detail="Cannot disable the built-in admin account")
    users = _load_users()
    if target not in users:
        raise HTTPException(404, detail="Unknown user")
    users[target]["disabled"] = disabled
    USER_DB.write_text(json.dumps(users, indent=2, ensure_ascii=False), encoding="utf-8")


@app.post("/admin/users/{target}/disable")
def disable_local_user(target: str, admin: str = Depends(require_admin)):
    _set_user_disabled(target, True)
    return {"ok": True}


@app.post("/admin/users/{target}/enable")
def enable_local_user(target: str, admin: str = Depends(require_admin)):
    _set_user_disabled(target, False)
    return {"ok": True}


@app.get("/api/whoami")
def whoami(username: str = Depends(current_user)):
    return {"username": username, "is_admin": is_admin(username)}


@app.get("/api/models")
def api_models(username: str = Depends(current_user)):
    from model_store import SUPPORTED_GIGAAM_MODELS, SUPPORTED_DIARIZATION_MODELS, load_settings
    from system_info import gpu_info
    return {"asr": SUPPORTED_GIGAAM_MODELS, "diarization": SUPPORTED_DIARIZATION_MODELS, "profiles": {}, "settings": load_settings(MODELS_DIR), "cuda_available": bool(gpu_info().get("cuda_available"))}

@app.get("/api/models/status")
def api_models_status(username: str = Depends(current_user)):
    from model_store import SUPPORTED_GIGAAM_MODELS, SUPPORTED_DIARIZATION_MODELS, load_settings, gigaam_checkpoint_path, pyannote_target_for, is_gigaam_ready, is_pyannote_ready
    settings = load_settings(MODELS_DIR); out=[]
    for mid, meta in SUPPORTED_GIGAAM_MODELS.items():
        ckpt = gigaam_checkpoint_path(MODELS_DIR, meta["model_name"]); installed = is_gigaam_ready(MODELS_DIR, meta["model_name"])
        out.append({**meta,"id":mid,"installed":installed,"path":str(ckpt),"size":ckpt.stat().st_size if ckpt.exists() else 0,"active":settings["asr_model"]==mid,"integrity":"ok" if installed else "missing","loadable": installed and _import_ok("gigaam"),"test_inference":"not_run","cpu_compatible":False,"gpu_compatible":True,"download_status":"idle","error":None})
    for mid, meta in SUPPORTED_DIARIZATION_MODELS.items():
        path = pyannote_target_for(MODELS_DIR, mid); installed = mid == "none" or is_pyannote_ready(MODELS_DIR, mid)
        out.append({**meta,"id":mid,"installed":installed,"path":str(path),"size":sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) if path.exists() else 0,"active":settings["diarization_model"]==mid,"integrity":"ok" if installed else "missing","loadable": mid == "none" or (installed and _import_ok("pyannote.audio")),"test_inference":"not_run","cpu_compatible":False,"gpu_compatible":mid != "none","download_status":"idle","error":None})
    return {"models": out}

@app.post("/api/models/select")
async def api_models_select(payload: dict[str, Any], username: str = Depends(require_admin)):
    if payload.get("profile"):
        raise HTTPException(400, detail="Profiles are not supported in CUDA-only mode")
    from model_store import save_settings, is_pyannote_ready
    diar = payload.get("diarization_model")
    if diar and diar != "none" and not is_pyannote_ready(MODELS_DIR, diar):
        return JSONResponse({"saved":False,"error_code":"DIARIZATION_MODEL_NOT_READY","error":"pyannote Community-1 is not ready locally"}, status_code=422)
    try: settings = save_settings(payload, MODELS_DIR)
    except ValueError as exc: raise HTTPException(400, detail=str(exc))
    return {"saved": True, "applies_to": "next_job", "settings": settings}

@app.delete("/api/models/{model_id}")
def api_models_delete(model_id: str, username: str = Depends(require_admin)):
    from model_store import SUPPORTED_GIGAAM_MODELS, SUPPORTED_DIARIZATION_MODELS, gigaam_checkpoint_path, pyannote_target_for
    if any(model_id in {j["settings_snapshot"].get("asr_model"), j["settings_snapshot"].get("diarization_model")} for j in job_store.list(active_only=True)): raise HTTPException(409, detail="Model is used by an active job")
    if model_id in SUPPORTED_GIGAAM_MODELS: gigaam_checkpoint_path(MODELS_DIR, SUPPORTED_GIGAAM_MODELS[model_id]["model_name"]).unlink(missing_ok=True); return {"ok": True}
    if model_id in SUPPORTED_DIARIZATION_MODELS and model_id != "none": shutil.rmtree(pyannote_target_for(MODELS_DIR, model_id), ignore_errors=True); return {"ok": True}
    raise HTTPException(400, detail="Unsupported model")

@app.post("/api/models/{model_id}/test")
def api_models_test(model_id: str, username: str = Depends(require_admin)):
    from model_store import SUPPORTED_GIGAAM_MODELS, SUPPORTED_DIARIZATION_MODELS, is_gigaam_ready, is_pyannote_ready, gigaam_checkpoint_path, pyannote_target_for
    try:
        if model_id in SUPPORTED_GIGAAM_MODELS:
            meta=SUPPORTED_GIGAAM_MODELS[model_id]
            if not is_gigaam_ready(MODELS_DIR, meta["model_name"]): raise RuntimeError("Model is not installed or integrity verification failed")
            import gigaam, wave
            # A generated, valid one-frame WAV proves inference for *this* model
            # without shipping a binary fixture in the repository.
            probe = MODELS_DIR / ".model-test.wav"
            with wave.open(str(probe), "wb") as wav:
                wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000); wav.writeframes(b"\0\0" * 1600)
            try:
                model = gigaam.load_model(meta["model_name"], device="cuda", download_root=str(MODELS_DIR / "gigaam" / "v3_e2e_rnnt"))
                result = model.transcribe(str(probe))
                if result is None: raise RuntimeError("Inference returned no result")
            finally: probe.unlink(missing_ok=True)
        elif model_id in SUPPORTED_DIARIZATION_MODELS and model_id != "none":
            if not is_pyannote_ready(MODELS_DIR, model_id): raise RuntimeError("Model is not installed or integrity verification failed")
            from pyannote.audio import Pipeline
            pipeline = Pipeline.from_pretrained(str(pyannote_target_for(MODELS_DIR, model_id)))
            # Pipeline invocation is intentional; loading alone is not a smoke test.
            import wave
            probe = MODELS_DIR / ".model-test.wav"
            with wave.open(str(probe), "wb") as wav:
                wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000); wav.writeframes(b"\0\0" * 1600)
            try: pipeline(str(probe))
            finally: probe.unlink(missing_ok=True)
        elif model_id == "none": return {"ok": True, "model_id": model_id, "load_test": "not_applicable"}
        else: raise HTTPException(404, detail="Unknown model")
        return {"ok": True, "model_id": model_id, "load_test": "ok", "inference_test": "ok"}
    except HTTPException: raise
    except Exception as exc:
        logger.exception("model test failed model_id=%s", model_id)
        return JSONResponse({"ok": False, "model_id": model_id, "load_test": "failed", "error": "Model test failed"}, status_code=422)

@app.get("/api/system")
def api_system(username: str = Depends(current_user)):
    from system_info import gpu_info
    return {"gpu": gpu_info(real_test=False), "active_gpu_job": any(j["status"] == "running" and j["settings_snapshot"].get("device") == "cuda" for j in job_store.list(active_only=True))}

@app.get("/api/system/gpu")
def api_system_gpu(username: str = Depends(current_user)):
    from system_info import gpu_info
    return gpu_info(real_test=True)


# ---------------------------------------------------------------------
# Protocol admin API (models, prompts, glossary, settings, history).
# Every handler starts with _require_protocol_enabled(); none of this is
# reachable, and none of it is registered against real protocol state,
# when the module is disabled.
# ---------------------------------------------------------------------

@app.get("/api/protocol/models")
def api_protocol_models(admin: str = Depends(require_admin)):
    _require_protocol_enabled()
    return {"models": PROTOCOL_SERVICE.store.list_model_states(), "active_model_id": PROTOCOL_SERVICE.store.active_model_id()}


@app.post("/api/protocol/models/{model_id}/select")
def api_protocol_select_model(model_id: str, admin: str = Depends(require_admin)):
    _require_protocol_enabled()
    state = PROTOCOL_SERVICE.store.get_model_state(model_id)
    if not state: raise HTTPException(404, detail="Unknown protocol model")
    if not state.get("installed"): raise HTTPException(422, detail="Model is not installed")
    PROTOCOL_SERVICE.store.set_active_model_id(model_id)
    return {"ok": True, "active_model_id": model_id}


@app.post("/api/protocol/models/{model_id}/params")
def api_protocol_model_params(model_id: str, payload: dict[str, Any], admin: str = Depends(require_admin)):
    # Engine choice and its config (item 4) are admin-only, same as every
    # other route in this block -- there is no user-facing way to reach an
    # arbitrary ollama_url; only require_admin can set it.
    _require_protocol_enabled()
    if not PROTOCOL_SERVICE.store.get_model_state(model_id): raise HTTPException(404)
    allowed = {
        "repo_id", "filename", "context_length", "temperature", "max_output_tokens", "system_prompt_override",
        "engine", "n_gpu_layers", "extra_launch_params", "ollama_url", "ollama_keep_alive",
    }
    changes = {k: v for k, v in payload.items() if k in allowed}
    if not changes: raise HTTPException(400, detail="No recognized parameters in payload")
    if changes.get("engine") not in (None,) and changes.get("engine") not in ("llama_cpp", "ollama"):
        raise HTTPException(400, detail="engine must be 'llama_cpp' or 'ollama'")
    if changes.get("ollama_url") and not str(changes["ollama_url"]).startswith(("http://", "https://")):
        raise HTTPException(400, detail="ollama_url must start with http:// or https://")
    if "context_length" in changes or "max_output_tokens" in changes:
        # item 4: catch an invalid combination here too, not only when a
        # job is actually created -- immediate feedback for the admin.
        current = PROTOCOL_SERVICE.store.get_model_state(model_id)
        context_length = changes.get("context_length", current["context_length"])
        max_output_tokens = changes.get("max_output_tokens", current["max_output_tokens"])
        if context_length - max_output_tokens < 512:
            raise HTTPException(400, detail=(
                f"context_length ({context_length}) leaves too little room for input given "
                f"max_output_tokens ({max_output_tokens}); need at least 512 tokens free"
            ))
    try:
        return PROTOCOL_SERVICE.store.update_model_state(model_id, **changes)
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc))


@app.post("/api/protocol/models/{model_id}/test")
async def api_protocol_test_model(model_id: str, admin: str = Depends(require_admin)):
    _require_protocol_enabled()
    state = PROTOCOL_SERVICE.store.get_model_state(model_id)
    if not state: raise HTTPException(404)
    if not state.get("installed"): raise HTTPException(422, detail="Model is not installed")
    from protocol.models import SUPPORTED_PROTOCOL_MODELS
    from protocol.providers import create_provider, validate_engine_config
    try:
        validate_engine_config(engine=state["engine"], model_path=state["local_path"], ollama_url=state.get("ollama_url"))
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=422)
    no_think = bool(SUPPORTED_PROTOCOL_MODELS.get(model_id) and SUPPORTED_PROTOCOL_MODELS[model_id].no_think)
    provider = create_provider(
        engine=state["engine"], model_path=state["local_path"], context_length=state["context_length"],
        n_gpu_layers=state.get("n_gpu_layers"), no_think=no_think, extra_launch_params=state.get("extra_launch_params") or {},
        ollama_url=state.get("ollama_url"), ollama_keep_alive=state.get("ollama_keep_alive"),
    )
    try:
        async with gpu_worker:
            MODEL_MANAGER.unload_all()
            await provider.load()
            text = await provider.generate("Ответь одним словом: тест.", temperature=0.1, max_tokens=16)
        ok = bool(text and text.strip())
        PROTOCOL_SERVICE.store.update_model_state(model_id, last_check_status="ok" if ok else "empty_response", last_check_at=time.time())
        return {"ok": ok, "sample": text[:200] if text else None}
    except Exception as exc:
        logger.exception("protocol model test failed model_id=%s", model_id)
        PROTOCOL_SERVICE.store.update_model_state(model_id, last_check_status="failed", last_check_at=time.time())
        return JSONResponse({"ok": False, "error": "Model test failed"}, status_code=422)
    finally:
        try: await provider.unload()
        except Exception: pass


@app.delete("/api/protocol/models/{model_id}")
def api_protocol_delete_model(model_id: str, admin: str = Depends(require_admin)):
    _require_protocol_enabled()
    state = PROTOCOL_SERVICE.store.get_model_state(model_id)
    if not state: raise HTTPException(404)
    if state.get("local_path"):
        path = Path(state["local_path"])
        try:
            resolved = ensure_inside(path, PROTOCOL_SERVICE.config.models_dir)
            if resolved.is_file(): resolved.unlink(missing_ok=True)
            elif resolved.is_dir(): shutil.rmtree(resolved, ignore_errors=True)
        except Exception:
            logger.exception("failed to delete protocol model file model_id=%s", model_id)
    PROTOCOL_SERVICE.store.update_model_state(model_id, local_path=None, installed=0, size_bytes=0,
                                               last_check_status=None, last_check_at=None)
    if PROTOCOL_SERVICE.store.active_model_id() == model_id:
        PROTOCOL_SERVICE.store.set_setting("active_model_id", "")
    return {"ok": True}


@app.get("/api/protocol/prompts/{kind}")
def api_protocol_get_prompt(kind: str, model_id: str | None = None, admin: str = Depends(require_admin)):
    _require_protocol_enabled()
    from protocol.prompts import DEFAULT_PROMPTS
    if kind not in DEFAULT_PROMPTS: raise HTTPException(404, detail="Unknown prompt kind")
    active = PROTOCOL_SERVICE.store.get_active_prompt(kind, model_id)
    history = PROTOCOL_SERVICE.store.list_prompt_history(kind, model_id)
    return {"active": active.__dict__, "history": [h.__dict__ for h in history]}


@app.post("/api/protocol/prompts/{kind}")
def api_protocol_save_prompt(kind: str, payload: dict[str, Any], model_id: str | None = None, admin: str = Depends(require_admin)):
    _require_protocol_enabled()
    from protocol.prompts import DEFAULT_PROMPTS, validate_prompt_content
    if kind not in DEFAULT_PROMPTS: raise HTTPException(404, detail="Unknown prompt kind")
    content = payload.get("content", "")
    try:
        validate_prompt_content(kind, content)
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc))
    spec = PROTOCOL_SERVICE.store.save_prompt_version(kind, content, model_id=model_id, updated_by=admin)
    return spec.__dict__


@app.post("/api/protocol/prompts/{kind}/restore-default")
def api_protocol_restore_prompt(kind: str, model_id: str | None = None, admin: str = Depends(require_admin)):
    _require_protocol_enabled()
    from protocol.prompts import DEFAULT_PROMPTS
    if kind not in DEFAULT_PROMPTS: raise HTTPException(404, detail="Unknown prompt kind")
    spec = PROTOCOL_SERVICE.store.restore_default_prompt(kind, model_id=model_id, updated_by=admin)
    return spec.__dict__


@app.post("/api/protocol/prompts/{kind}/preview")
def api_protocol_preview_prompt(kind: str, payload: dict[str, Any], admin: str = Depends(require_admin)):
    _require_protocol_enabled()
    from protocol.prompts import DEFAULT_PROMPTS, validate_prompt_content
    if kind not in DEFAULT_PROMPTS: raise HTTPException(404, detail="Unknown prompt kind")
    content = payload.get("content", "")
    try:
        validate_prompt_content(kind, content)
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc))
    if kind == "html_template":
        from protocol.renderer import render_html
        from protocol.schemas import ProtocolDocument
        sample = ProtocolDocument(
            meeting_title="Пример: синхрон команды", processed_at=time.strftime("%Y-%m-%d %H:%M"),
            source_filename="example.mp3", duration_seconds=930, participants=("Спикер 1", "Спикер 2"),
            summary="Пример резюме встречи для предпросмотра шаблона.", topics=("Пример темы",),
            decisions=(), tasks=(), owners=(), deadlines=(), open_questions=(), risks=(),
            disagreements=(), next_steps=(), unverified_items=(), timestamp_refs=(),
            model_id="preview", prompt_versions={},
        )
        return {"preview_html": render_html(sample, template=content)}
    return {"preview_text": content}


@app.get("/api/protocol/glossary/terms")
def api_glossary_list(scope: str | None = None, status: str | None = None, admin: str = Depends(require_admin)):
    _require_protocol_enabled()
    terms = PROTOCOL_SERVICE.store.list_terms(scope=scope, status=status)
    return {"terms": [t.__dict__ for t in terms]}


@app.post("/api/protocol/glossary/terms")
def api_glossary_add(payload: dict[str, Any], admin: str = Depends(require_admin)):
    _require_protocol_enabled()
    from protocol.schemas import GlossaryTerm
    if not payload.get("canonical"): raise HTTPException(400, detail="canonical is required")
    term = PROTOCOL_SERVICE.store.add_term(GlossaryTerm(
        id=None, canonical=payload["canonical"], aliases=tuple(payload.get("aliases") or ()),
        category=payload.get("category", ""), context=payload.get("context", ""),
        scope=payload.get("scope", "global"), project=payload.get("project"),
        status=payload.get("status", "confirmed"),
    ))
    return term.__dict__


@app.post("/api/protocol/glossary/terms/{term_id}/disable")
def api_glossary_disable(term_id: int, admin: str = Depends(require_admin)):
    _require_protocol_enabled()
    PROTOCOL_SERVICE.store.disable_term(term_id)
    return {"ok": True}


@app.post("/api/protocol/glossary/terms/merge")
def api_glossary_merge(payload: dict[str, Any], admin: str = Depends(require_admin)):
    _require_protocol_enabled()
    keep_id, remove_id = payload.get("keep_id"), payload.get("remove_id")
    if not keep_id or not remove_id: raise HTTPException(400, detail="keep_id and remove_id are required")
    PROTOCOL_SERVICE.store.merge_terms(keep_id, remove_id)
    return {"ok": True}


@app.get("/api/protocol/glossary/export")
def api_glossary_export(format: str = "json", admin: str = Depends(require_admin)):
    _require_protocol_enabled()
    from protocol import glossary as glossary_mod
    terms = PROTOCOL_SERVICE.store.list_terms()
    if format == "csv":
        return JSONResponse({"content": glossary_mod.export_terms_csv(terms), "format": "csv"})
    return JSONResponse({"content": glossary_mod.export_terms_json(terms), "format": "json"})


@app.post("/api/protocol/glossary/import")
def api_glossary_import(payload: dict[str, Any], admin: str = Depends(require_admin)):
    _require_protocol_enabled()
    from protocol import glossary as glossary_mod
    fmt, content = payload.get("format", "json"), payload.get("content", "")
    try:
        n = glossary_mod.import_terms_csv(PROTOCOL_SERVICE.store, content) if fmt == "csv" \
            else glossary_mod.import_terms_json(PROTOCOL_SERVICE.store, content)
    except Exception as exc:
        raise HTTPException(400, detail=f"Import failed: {exc}")
    return {"ok": True, "imported": n}


@app.get("/api/protocol/glossary/suggestions")
def api_glossary_suggestions(status: str = "proposed", admin: str = Depends(require_admin)):
    _require_protocol_enabled()
    return {"suggestions": [s.__dict__ for s in PROTOCOL_SERVICE.store.list_suggestions(status)]}


@app.post("/api/protocol/glossary/suggestions/{suggestion_id}/resolve")
def api_glossary_resolve(suggestion_id: int, payload: dict[str, Any], admin: str = Depends(require_admin)):
    _require_protocol_enabled()
    from protocol import glossary as glossary_mod
    action = payload.get("action")
    if action == "confirm":
        suggestion = PROTOCOL_SERVICE.store.get_suggestion(suggestion_id)
        if not suggestion: raise HTTPException(404)
        term = glossary_mod.confirm_suggestion_as_term(
            PROTOCOL_SERVICE.store, suggestion, resolved_by=admin,
            scope=payload.get("scope", "global"), project=payload.get("project"),
        )
        return {"ok": True, "term": term.__dict__}
    if action == "reject":
        PROTOCOL_SERVICE.store.resolve_suggestion(suggestion_id, status="rejected", resolved_by=admin)
        return {"ok": True}
    raise HTTPException(400, detail="action must be 'confirm' or 'reject'")


@app.get("/api/protocol/jobs")
def api_protocol_job_history(admin: str = Depends(require_admin)):
    _require_protocol_enabled()
    return {"jobs": PROTOCOL_SERVICE.store.list_all_protocol_jobs()}


@app.get("/api/protocol/settings")
def api_protocol_settings(admin: str = Depends(require_admin)):
    _require_protocol_enabled()
    cfg = PROTOCOL_SERVICE.config
    stored = PROTOCOL_SERVICE.store.all_settings()
    defaults = {
        "chunk_minutes": cfg.default_chunk_minutes, "chunk_overlap_seconds": cfg.chunk_overlap_seconds,
        "topic_split_enabled": cfg.topic_split_enabled, "temperature": cfg.default_temperature,
        "max_output_tokens": cfg.default_max_output_tokens, "max_retries": cfg.max_retries,
        "glossary_suggestions_enabled": cfg.glossary_suggestions_enabled,
        "default_glossary_scope": cfg.default_glossary_scope,
    }
    return {**defaults, **stored, "active_model_id": PROTOCOL_SERVICE.store.active_model_id()}


@app.post("/api/protocol/settings")
def api_protocol_save_settings(payload: dict[str, Any], admin: str = Depends(require_admin)):
    _require_protocol_enabled()
    allowed = {
        "chunk_minutes", "chunk_overlap_seconds", "topic_split_enabled", "temperature",
        "max_output_tokens", "max_retries", "glossary_suggestions_enabled", "default_glossary_scope", "engine",
    }
    for key, value in payload.items():
        if key in allowed:
            PROTOCOL_SERVICE.store.set_setting(key, str(value))
    return {"ok": True}


BASE_CSS = """
:root{color-scheme:light dark}
body{font-family:system-ui,-apple-system,sans-serif;max-width:960px;margin:24px auto;padding:0 16px;line-height:1.45}
header{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:18px;border-bottom:1px solid #8884;padding-bottom:12px}
header h1{font-size:1.25rem;margin:0}
nav{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.card{border:1px solid #8884;border-radius:10px;padding:16px;margin:14px 0}
.muted{color:#888;font-size:.9em}
progress{width:100%;height:10px}
input,select,button{padding:8px 10px;margin:4px 0;font-size:1em;box-sizing:border-box}
input,select{width:100%}
button{cursor:pointer;width:auto}
form label{display:block;margin:8px 0 2px;font-size:.9em}
table{border-collapse:collapse;width:100%}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid #8883;vertical-align:middle}
.badge{display:inline-block;padding:2px 9px;border-radius:10px;font-size:.8em;white-space:nowrap}
.badge.queued{background:#dbe4ff;color:#1d3a8a}
.badge.running{background:#ffe6a8;color:#6b4a00}
.badge.completed{background:#d3f3d8;color:#12572a}
.badge.failed{background:#fbd6d6;color:#7a1414}
.badge.cancelled{background:#e5e5e5;color:#444}
.err{color:#b00020}
.actions{display:flex;gap:6px;flex-wrap:wrap}
.actions button{width:auto}
.job .row{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;align-items:center}
"""

LOGIN_HTML = ("<!doctype html><meta charset='utf-8'><title>GigaScribe — вход</title><style>" + BASE_CSS +
              "body{max-width:380px;margin-top:12vh}header{border:none;justify-content:center}</style>"
              "<header><h1>🎙️ GigaScribe</h1></header>"
              "<!--ERROR-->"
              "<form method='post' class='card'>"
              "<label>Пользователь<input name='username' placeholder='Логин' autofocus required></label>"
              "<label>Пароль<input name='password' type='password' placeholder='Пароль' required></label>"
              "<button type='submit'>Войти</button>"
              "</form>")

INDEX_HTML = ("<!doctype html><meta charset='utf-8'><title>GigaScribe</title><style>" + BASE_CSS + """
.upload-progress{display:none;margin-top:8px}
</style>
<header>
  <h1>🎙️ GigaScribe</h1>
  <nav>
    <span id='whoami' class='muted'></span>
    <a id='admin-link' href='/admin' style='display:none'>⚙ Администрирование</a>
    <form method='post' action='/logout' style='display:inline'><button>Выйти</button></form>
  </nav>
</header>

<section class='card'>
  <h2>Новая транскрипция</h2>
  <form id='up'>
    <input type='file' name='file' id='file-input' required>
    <button type='submit' id='up-btn'>🚀 Запустить</button>
  </form>
  <div id='upload-progress-wrap' class='upload-progress'>
    <progress id='upload-progress' value='0' max='100'></progress>
    <span id='upload-progress-text' class='muted'></span>
  </div>
  <p id='diarization-info' class='muted'></p>
</section>

<h2>История транскрипций <span class='muted'>(общая для всех пользователей)</span></h2>
<div id='jobs'></div>

<script>
function escapeHtml(s){return (s==null?'':String(s)).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
const STATUS_LABELS={queued:'В очереди',running:'Обрабатывается',completed:'Готово',failed:'Ошибка',cancelled:'Отменено'};
const DOWNLOAD_LABELS={transcript:'транскрипт',log:'лог',original:'оригинал',m4a:'M4A',flac:'FLAC'};
const PROTOCOL_STATUS_LABELS={queued:'в очереди',waiting_for_gpu:'ожидание GPU',unloading_asr:'выгрузка ASR/диаризации',loading_llm:'загрузка модели',splitting:'разбиение на блоки',processing_chunks:'обработка блоков',merging:'сведение протокола',fact_checking:'проверка фактов',rendering:'формирование HTML',completed:'готово',failed:'ошибка',cancelled:'отменено'};
let ME={username:null,is_admin:false};
let PROTOCOL_STATUS={enabled:false,has_installed_model:false};

async function loadWhoAmI(){
  let r=await fetch('/api/whoami'); if(!r.ok) return;
  ME=await r.json();
  document.getElementById('whoami').textContent='Вы вошли как: '+ME.username;
  if(ME.is_admin) document.getElementById('admin-link').style.display='';
}

async function loadProtocolStatus(){
  let r=await fetch('/api/protocol/status'); if(!r.ok) return;
  PROTOCOL_STATUS=await r.json();
}

async function loadModelsInfo(){
  let r=await fetch('/api/models'); if(!r.ok) return;
  let m=await r.json();
  let diar=m.diarization[m.settings.diarization_model];
  let text='Диаризация: '+escapeHtml(diar?diar.label:m.settings.diarization_model);
  if(ME.is_admin) text+=' — изменить можно в администрировании';
  document.getElementById('diarization-info').textContent=text;
}

function renderProtocolSection(j, canControl){
  let p=j.protocol;
  if(!p||!p.enabled) return '';
  if(p.ready){
    return `<div class='row protocol-row'><a href='/api/jobs/${j.id}/protocol' target='_blank'>📄 Протокол</a></div>`;
  }
  if(p.status && !['completed','failed','cancelled'].includes(p.status)){
    let label=PROTOCOL_STATUS_LABELS[p.status]||p.status;
    let chunkInfo=(p.chunk_total)?` (блок ${p.chunk_current||0}/${p.chunk_total})`:'';
    let pct=p.progress!=null?Math.round(p.progress*100)+'%':'';
    return `<div class='muted protocol-row'>Протокол (${escapeHtml(p.model_id||'')}): ${label}${chunkInfo} ${pct}</div>`;
  }
  if(p.failed && canControl){
    return `<div class='row protocol-row'><span class='err'>Протокол: ошибка</span><button onclick="retryProtocol('${j.id}')">Повторить протокол</button></div>`;
  }
  if(p.can_create && canControl && PROTOCOL_STATUS.has_installed_model){
    return `<div class='row protocol-row'><button onclick="createProtocol('${j.id}')">Создать протокол</button></div>`;
  }
  return '';
}

async function submitCorrection(jobId, ev){
  ev.preventDefault();
  let fd=new FormData(ev.target);
  let payload={wrong_text: fd.get('wrong_text'), suggested_text: fd.get('suggested_text')};
  let r=await fetch(`/api/jobs/${jobId}/glossary-suggestion`, {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify(payload)});
  let msg=ev.target.querySelector('.correction-message');
  if(r.ok){ ev.target.reset(); msg.textContent='Предложение отправлено администратору на подтверждение'; }
  else { let d=await r.json().catch(()=>({})); msg.textContent='Ошибка: '+(d.detail||r.status); }
  return false;
}

function renderJob(j){
  let canControl=(j.owner===ME.username)||ME.is_admin;
  let statusLabel=STATUS_LABELS[j.status]||j.status;
  let downloads=Object.keys(DOWNLOAD_LABELS).filter(k=>j.downloads[k]).map(k=>`<a href='/api/jobs/${j.id}/download/${k}'>${DOWNLOAD_LABELS[k]}</a>`).join(' · ');
  let actions='';
  if(canControl&&(j.status==='queued'||j.status==='running')) actions+=`<button onclick="cancelJob('${j.id}')">Отменить</button>`;
  if(canControl&&(j.status==='failed'||j.status==='cancelled')) actions+=`<button onclick="retryJob('${j.id}')">Повторить</button>`;
  let correctionForm=(canControl && PROTOCOL_STATUS.enabled) ? `
    <details class='correction-details'><summary>Предложить исправление термина</summary>
      <form onsubmit="return submitCorrection('${j.id}', event)">
        <input name='wrong_text' placeholder='Как распознано в тексте' required>
        <input name='suggested_text' placeholder='Как должно быть написано' required>
        <button type='submit'>Предложить</button>
        <span class='muted correction-message'></span>
      </form>
    </details>` : '';
  return `<div class='job card'>
    <div class='row'><b>${escapeHtml(j.filename)}</b><span class='badge ${j.status}'>${statusLabel}</span></div>
    <div class='muted'>Загрузил(а): ${escapeHtml(j.owner)}${j.attempts>1?` · попытка ${j.attempts}`:''}</div>
    <progress value='${j.progress}' max='1'></progress> ${Math.round(j.progress*100)}%
    <div class='muted'>${escapeHtml(j.message)}</div>
    ${j.error?`<div class='err'>${escapeHtml(j.error)}</div>`:''}
    <div class='row'><span>${downloads}</span><span class='actions'>${actions}</span></div>
    ${renderProtocolSection(j, canControl)}
    ${correctionForm}
  </div>`;
}

async function refresh(){
  let r=await fetch('/api/jobs'); if(!r.ok) return;
  let js=await r.json();
  document.getElementById('jobs').innerHTML=js.map(renderJob).join('')||"<p class='muted'>Пока нет ни одной транскрипции.</p>";
}

async function cancelJob(id){await fetch(`/api/jobs/${id}/cancel`,{method:'POST'});refresh()}
async function retryJob(id){await fetch(`/api/jobs/${id}/retry`,{method:'POST'});refresh()}
async function createProtocol(id){await fetch(`/api/jobs/${id}/protocol`,{method:'POST'});refresh()}
async function retryProtocol(id){await fetch(`/api/jobs/${id}/protocol/retry`,{method:'POST'});refresh()}

let poll=3000;
async function tick(){await refresh();setTimeout(tick,document.hidden?10000:poll)}

document.getElementById('up').onsubmit=function(e){
  e.preventDefault();
  let fileInput=document.getElementById('file-input');
  if(!fileInput.files.length) return;
  let fd=new FormData(); fd.append('file', fileInput.files[0]);
  let wrap=document.getElementById('upload-progress-wrap');
  let bar=document.getElementById('upload-progress');
  let text=document.getElementById('upload-progress-text');
  let btn=document.getElementById('up-btn');
  wrap.style.display='block'; btn.disabled=true; bar.value=0; text.textContent='Загрузка файла: 0%';
  let xhr=new XMLHttpRequest();
  xhr.upload.onprogress=function(ev){
    if(ev.lengthComputable){
      let pct=Math.round(ev.loaded/ev.total*100);
      bar.value=pct; text.textContent='Загрузка файла: '+pct+'%';
    }
  };
  xhr.onload=function(){
    btn.disabled=false;
    if(xhr.status>=200&&xhr.status<300){
      text.textContent='Загрузка завершена, обработка начата';
      document.getElementById('up').reset();
      refresh();
    } else {
      let msg='Ошибка загрузки';
      try{ msg=JSON.parse(xhr.responseText).detail||msg }catch(e){}
      text.textContent=msg;
    }
    setTimeout(()=>{wrap.style.display='none'},4000);
  };
  xhr.onerror=function(){btn.disabled=false;text.textContent='Ошибка сети при загрузке'};
  xhr.open('POST','/api/jobs');
  xhr.send(fd);
};

(async()=>{ await loadWhoAmI(); await loadModelsInfo(); await loadProtocolStatus(); tick(); })();
</script>""")

ADMIN_DENIED_HTML = ("<!doctype html><meta charset='utf-8'><title>GigaScribe — доступ запрещён</title><style>" + BASE_CSS +
                      "body{max-width:420px;margin-top:15vh;text-align:center}</style>"
                      "<h1>⛔ Доступ запрещён</h1>"
                      "<p>Эта страница доступна только администратору.</p>"
                      "<p><a href='/'>На главную</a></p>")

if PROTOCOL_ENABLED:
    PROTOCOL_ADMIN_SECTION = """
<section class='card' id='protocol-section'>
  <h2>Протоколирование</h2>
  <nav class='protocol-tabs'>
    <button data-tab='p-models' class='tab-btn'>Модели</button>
    <button data-tab='p-prompts' class='tab-btn'>Промпты</button>
    <button data-tab='p-glossary' class='tab-btn'>Словарь</button>
    <button data-tab='p-suggestions' class='tab-btn'>Предложения</button>
    <button data-tab='p-history' class='tab-btn'>История</button>
    <button data-tab='p-settings' class='tab-btn'>Настройки</button>
  </nav>

  <div id='p-models' class='tab-panel'>
    <table><thead><tr><th>Модель</th><th>Движок</th><th>Статус</th><th>Активна</th><th>Размер</th><th></th></tr></thead><tbody id='protocol-models-body'></tbody></table>
    <p class='muted'>Установка выполняется скриптом <code>scripts/download_protocol_model.py</code> (см. README) — здесь можно выбрать активную модель, проверить загрузку или удалить файлы.</p>
    <p id='protocol-model-message' class='muted'></p>
    <h3>Настройка модели</h3>
    <form id='engine-config-form'>
      <label>Модель<select name='model_id' id='engine-config-model'></select></label>
      <label>Движок<select name='engine'><option value='llama_cpp'>llama.cpp (GGUF)</option><option value='ollama'>Ollama</option></select></label>
      <label>Контекст, токенов<input type='number' name='context_length' min='1' step='1'></label>
      <label>Температура<input type='number' name='temperature' min='0' max='2' step='0.1'></label>
      <label>Максимум выходных токенов<input type='number' name='max_output_tokens' min='1' step='1'></label>
      <label>Число GPU-слоёв (llama.cpp, -1 = все)<input type='number' name='n_gpu_layers' step='1'></label>
      <label>URL локального Ollama (только для Ollama)<input name='ollama_url' placeholder='http://127.0.0.1:11434'></label>
      <label>keep_alive Ollama (например, 5m)<input name='ollama_keep_alive' placeholder='5m'></label>
      <button type='submit'>Сохранить параметры модели</button>
    </form>
    <p id='engine-config-message' class='muted'></p>
  </div>

  <div id='p-prompts' class='tab-panel' style='display:none'>
    <label>Промпт<select id='prompt-kind-select'>
      <option value='chunk_analysis'>Анализ блока</option>
      <option value='topic_split'>Тематическое разбиение</option>
      <option value='merge'>Итоговое сведение</option>
      <option value='fact_check'>Проверка фактов</option>
      <option value='html_template'>Шаблон HTML-протокола</option>
    </select></label>
    <label>Модель<select id='prompt-model-select'>
      <option value=''>Общий промпт (для всех моделей)</option>
      <option value='qwen3-14b'>Qwen3-14B</option>
      <option value='qwen3-8b'>Qwen3-8B</option>
      <option value='gemma3-12b-it'>Gemma 3 12B IT</option>
      <option value='ministral3-8b-instruct'>Ministral 3 8B Instruct</option>
    </select></label>
    <textarea id='prompt-content' rows='14' style='width:100%;font-family:monospace;font-size:.85em'></textarea>
    <div class='actions'>
      <button id='prompt-save' type='button'>Сохранить</button>
      <button id='prompt-restore' type='button'>Вернуть значение по умолчанию</button>
      <button id='prompt-preview' type='button'>Предпросмотр</button>
    </div>
    <p id='prompt-message' class='muted'></p>
    <p class='muted'>Порядок выбора при обработке: активный промпт выбранной модели → общий активный промпт → встроенный промпт по умолчанию.</p>
    <div id='prompt-preview-box'></div>
  </div>

  <div id='p-glossary' class='tab-panel' style='display:none'>
    <table><thead><tr><th>Термин</th><th>Синонимы</th><th>Область</th><th>Статус</th><th></th></tr></thead><tbody id='glossary-body'></tbody></table>
    <h3>Добавить термин</h3>
    <form id='glossary-form'>
      <label>Канонический термин<input name='canonical' required></label>
      <label>Синонимы (через запятую)<input name='aliases' placeholder='постгрес, постгресс'></label>
      <label>Область<select name='scope'><option value='global'>Общий словарь</option><option value='department'>Подразделение</option><option value='project'>Проект</option><option value='job'>Задание</option></select></label>
      <button type='submit'>Добавить</button>
    </form>
    <div class='actions'>
      <button id='glossary-export-json' type='button'>Экспорт JSON</button>
      <button id='glossary-export-csv' type='button'>Экспорт CSV</button>
    </div>
    <form id='glossary-import-form'>
      <label>Импорт (JSON или CSV, как при экспорте)<textarea name='content' rows='4' style='width:100%'></textarea></label>
      <label><input type='radio' name='format' value='json' checked> JSON</label>
      <label><input type='radio' name='format' value='csv'> CSV</label>
      <button type='submit'>Импортировать</button>
    </form>
    <p id='glossary-message' class='muted'></p>
  </div>

  <div id='p-suggestions' class='tab-panel' style='display:none'>
    <p class='muted'>Предложения из пользовательских исправлений и от модели. Ничего не применяется глобально без подтверждения.</p>
    <table><thead><tr><th>Источник</th><th>Было</th><th>Стало</th><th>Уверенность</th><th></th></tr></thead><tbody id='suggestions-body'></tbody></table>
  </div>

  <div id='p-history' class='tab-panel' style='display:none'>
    <table><thead><tr><th>ID задания</th><th>Статус</th><th>Модель</th><th>Блок</th><th>Создано</th></tr></thead><tbody id='protocol-history-body'></tbody></table>
  </div>

  <div id='p-settings' class='tab-panel' style='display:none'>
    <form id='protocol-settings-form'>
      <label>Размер блока, минут (10–15)<input type='number' name='chunk_minutes' min='10' max='15' step='1'></label>
      <label>Перекрытие блоков, секунд<input type='number' name='chunk_overlap_seconds' min='0'></label>
      <label><input type='checkbox' name='topic_split_enabled'> Сначала пробовать тематическое разбиение</label>
      <label>Температура<input type='number' name='temperature' min='0' max='2' step='0.1'></label>
      <label>Максимум выходных токенов<input type='number' name='max_output_tokens' min='256'></label>
      <label>Повторов при невалидном JSON<input type='number' name='max_retries' min='0' max='5'></label>
      <label><input type='checkbox' name='glossary_suggestions_enabled'> Принимать предложения терминов от модели</label>
      <label>Область словаря по умолчанию<select name='default_glossary_scope'><option value='global'>Общий</option><option value='department'>Подразделение</option><option value='project'>Проект</option></select></label>
      <button type='submit'>Сохранить настройки</button>
    </form>
    <p id='protocol-settings-message' class='muted'></p>
  </div>
</section>
"""
    PROTOCOL_ADMIN_SCRIPT = """
function showProtocolTab(name){
  document.querySelectorAll('.tab-panel').forEach(el=>el.style.display=(el.id===name?'':'none'));
  document.querySelectorAll('.tab-btn').forEach(el=>el.classList.toggle('active', el.dataset.tab===name));
}
document.querySelectorAll('.tab-btn').forEach(btn=>btn.onclick=()=>showProtocolTab(btn.dataset.tab));
showProtocolTab('p-models');

let PROTOCOL_MODELS_CACHE=[];
async function loadProtocolModels(){
  let body=document.getElementById('protocol-models-body');
  let r=await fetch('/api/protocol/models'); if(!r.ok) return;
  let data=await r.json();
  PROTOCOL_MODELS_CACHE=data.models;
  body.innerHTML=data.models.map(m=>{
    let active=m.id===data.active_model_id;
    return `<tr>
      <td>${escapeHtml(m.label)}</td>
      <td>${escapeHtml(m.engine)}</td>
      <td>${m.installed?"<span class='badge completed'>установлена</span>":"<span class='badge failed'>не установлена</span>"}${m.last_check_status?` <span class='muted'>(${escapeHtml(m.last_check_status)})</span>`:''}</td>
      <td>${active?'✅':(m.installed?`<button data-select-protocol="${escapeAttr(m.id)}" class='select-protocol-model'>Выбрать</button>`:'—')}</td>
      <td>${fmtSize(m.size_bytes)}</td>
      <td class='actions'>${m.installed?`<button data-test-protocol="${escapeAttr(m.id)}" class='test-protocol-model'>Тест</button> <button data-delete-protocol="${escapeAttr(m.id)}" class='delete-protocol-model'>Удалить</button>`:''}</td>
    </tr>`;
  }).join('');
  body.querySelectorAll('.select-protocol-model').forEach(b=>b.onclick=()=>selectProtocolModel(b.dataset.selectProtocol));
  body.querySelectorAll('.test-protocol-model').forEach(b=>b.onclick=()=>testProtocolModel(b.dataset.testProtocol));
  body.querySelectorAll('.delete-protocol-model').forEach(b=>b.onclick=()=>deleteProtocolModel(b.dataset.deleteProtocol));
  let sel=document.getElementById('engine-config-model');
  let prevValue=sel.value;
  sel.innerHTML=data.models.map(m=>`<option value="${escapeAttr(m.id)}">${escapeHtml(m.label)}</option>`).join('');
  sel.value=prevValue||data.models[0]?.id||'';
  loadEngineConfig();
}
function loadEngineConfig(){
  let id=document.getElementById('engine-config-model').value;
  let m=PROTOCOL_MODELS_CACHE.find(x=>x.id===id); if(!m) return;
  let form=document.getElementById('engine-config-form');
  form.elements['engine'].value=m.engine||'llama_cpp';
  form.elements['context_length'].value=m.context_length||'';
  form.elements['temperature'].value=m.temperature==null?'':m.temperature;
  form.elements['max_output_tokens'].value=m.max_output_tokens||'';
  form.elements['n_gpu_layers'].value=(m.n_gpu_layers===null||m.n_gpu_layers===undefined)?'':m.n_gpu_layers;
  form.elements['ollama_url'].value=m.ollama_url||'';
  form.elements['ollama_keep_alive'].value=m.ollama_keep_alive||'';
}
document.getElementById('engine-config-model').onchange=loadEngineConfig;
document.getElementById('engine-config-form').onsubmit=async(e)=>{
  e.preventDefault();
  let id=document.getElementById('engine-config-model').value;
  let form=e.target;
  let payload={engine:form.elements['engine'].value};
  let ctx=form.elements['context_length'].value; if(ctx!=='') payload.context_length=parseInt(ctx,10);
  let temp=form.elements['temperature'].value; if(temp!=='') payload.temperature=parseFloat(temp);
  let mot=form.elements['max_output_tokens'].value; if(mot!=='') payload.max_output_tokens=parseInt(mot,10);
  let ngl=form.elements['n_gpu_layers'].value; if(ngl!=='') payload.n_gpu_layers=parseInt(ngl,10);
  let ourl=form.elements['ollama_url'].value; if(ourl) payload.ollama_url=ourl;
  let oka=form.elements['ollama_keep_alive'].value; if(oka) payload.ollama_keep_alive=oka;
  let r=await fetch(`/api/protocol/models/${id}/params`, {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify(payload)});
  let d=await r.json().catch(()=>({}));
  document.getElementById('engine-config-message').textContent=r.ok?'Параметры модели сохранены':('Ошибка: '+(d.detail||r.status));
  if(r.ok) loadProtocolModels();
};
async function selectProtocolModel(id){
  let msg=document.getElementById('protocol-model-message');
  let r=await fetch(`/api/protocol/models/${id}/select`, {method:'POST'});
  let d=await r.json().catch(()=>({}));
  msg.textContent=r.ok?'Активная модель сохранена':('Ошибка: '+(d.detail||r.status));
  loadProtocolModels();
}
async function testProtocolModel(id){
  let msg=document.getElementById('protocol-model-message');
  msg.textContent='Тестируем '+id+'… (модели транскрибации будут временно выгружены)';
  let r=await fetch(`/api/protocol/models/${id}/test`, {method:'POST'});
  let d=await r.json().catch(()=>({}));
  msg.textContent=(r.ok&&d.ok)?('Модель '+id+': тест пройден. Ответ: '+(d.sample||'')):('Модель '+id+': тест не пройден');
  loadProtocolModels();
}
async function deleteProtocolModel(id){
  if(!confirm('Удалить файлы модели '+id+'?')) return;
  await fetch(`/api/protocol/models/${id}`, {method:'DELETE'});
  loadProtocolModels();
}

const PROMPT_KIND_VARS={chunk_analysis:'{transcript_chunk}',topic_split:'{transcript}',merge:'{chunk_analyses}',fact_check:'{document} и {transcript}',html_template:'$body (можно также $title)'};
function promptQuery(){
  let modelId=document.getElementById('prompt-model-select').value;
  return modelId?`?model_id=${encodeURIComponent(modelId)}`:'';
}
async function loadPrompt(){
  let kind=document.getElementById('prompt-kind-select').value;
  let r=await fetch(`/api/protocol/prompts/${kind}${promptQuery()}`); if(!r.ok) return;
  let d=await r.json();
  document.getElementById('prompt-content').value=d.active.content;
  let scopeLabel=d.active.model_id?`модель ${d.active.model_id}`:'общий';
  document.getElementById('prompt-message').textContent=`Версия ${d.active.version} (${scopeLabel})${d.active.is_default?' (по умолчанию)':''}. Обязательная переменная: ${PROMPT_KIND_VARS[kind]}`;
  document.getElementById('prompt-preview-box').innerHTML='';
}
document.getElementById('prompt-kind-select').onchange=loadPrompt;
document.getElementById('prompt-model-select').onchange=loadPrompt;
document.getElementById('prompt-save').onclick=async()=>{
  let kind=document.getElementById('prompt-kind-select').value;
  let content=document.getElementById('prompt-content').value;
  let r=await fetch(`/api/protocol/prompts/${kind}${promptQuery()}`, {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify({content})});
  let d=await r.json().catch(()=>({}));
  document.getElementById('prompt-message').textContent=r.ok?`Сохранено как версия ${d.version}`:('Ошибка: '+(d.detail||r.status));
};
document.getElementById('prompt-restore').onclick=async()=>{
  let kind=document.getElementById('prompt-kind-select').value;
  await fetch(`/api/protocol/prompts/${kind}/restore-default${promptQuery()}`, {method:'POST'});
  loadPrompt();
};
document.getElementById('prompt-preview').onclick=async()=>{
  let kind=document.getElementById('prompt-kind-select').value;
  let content=document.getElementById('prompt-content').value;
  let r=await fetch(`/api/protocol/prompts/${kind}/preview`, {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify({content})});
  let d=await r.json().catch(()=>({}));
  if(!r.ok){ document.getElementById('prompt-message').textContent='Ошибка: '+(d.detail||r.status); return; }
  document.getElementById('prompt-preview-box').innerHTML = d.preview_html
    ? `<iframe style='width:100%;height:400px;border:1px solid #8884' srcdoc="${escapeAttr(d.preview_html)}"></iframe>`
    : `<pre style='white-space:pre-wrap'>${escapeHtml(d.preview_text||'')}</pre>`;
};

async function loadGlossary(){
  let body=document.getElementById('glossary-body');
  let r=await fetch('/api/protocol/glossary/terms'); if(!r.ok) return;
  let d=await r.json();
  body.innerHTML=d.terms.map(t=>`<tr>
    <td>${escapeHtml(t.canonical)}</td>
    <td>${escapeHtml(t.aliases.join(', '))}</td>
    <td>${escapeHtml(t.scope)}</td>
    <td>${t.status==='confirmed'?"<span class='badge completed'>подтверждён</span>":"<span class='badge cancelled'>отключён</span>"}</td>
    <td>${t.status==='confirmed'?`<button data-disable-term="${t.id}" class='disable-term'>Отключить</button>`:''}</td>
  </tr>`).join('');
  body.querySelectorAll('.disable-term').forEach(b=>b.onclick=async()=>{await fetch(`/api/protocol/glossary/terms/${b.dataset.disableTerm}/disable`,{method:'POST'});loadGlossary();});
}
document.getElementById('glossary-form').onsubmit=async(e)=>{
  e.preventDefault();
  let fd=new FormData(e.target);
  let payload={canonical:fd.get('canonical'), aliases:(fd.get('aliases')||'').split(',').map(s=>s.trim()).filter(Boolean), scope:fd.get('scope')};
  let r=await fetch('/api/protocol/glossary/terms', {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify(payload)});
  document.getElementById('glossary-message').textContent=r.ok?'Термин добавлен':'Ошибка добавления';
  if(r.ok){ e.target.reset(); loadGlossary(); }
};
document.getElementById('glossary-export-json').onclick=async()=>{
  let r=await fetch('/api/protocol/glossary/export?format=json'); let d=await r.json();
  navigator.clipboard?.writeText(d.content).catch(()=>{});
  document.getElementById('glossary-message').textContent='JSON скопирован в буфер обмена ('+d.content.length+' симв.)';
};
document.getElementById('glossary-export-csv').onclick=async()=>{
  let r=await fetch('/api/protocol/glossary/export?format=csv'); let d=await r.json();
  navigator.clipboard?.writeText(d.content).catch(()=>{});
  document.getElementById('glossary-message').textContent='CSV скопирован в буфер обмена ('+d.content.length+' симв.)';
};
document.getElementById('glossary-import-form').onsubmit=async(e)=>{
  e.preventDefault();
  let fd=new FormData(e.target);
  let r=await fetch('/api/protocol/glossary/import', {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify({content:fd.get('content'), format:fd.get('format')})});
  let d=await r.json().catch(()=>({}));
  document.getElementById('glossary-message').textContent=r.ok?`Импортировано терминов: ${d.imported}`:('Ошибка: '+(d.detail||r.status));
  if(r.ok) loadGlossary();
};

async function loadSuggestions(){
  let body=document.getElementById('suggestions-body');
  let r=await fetch('/api/protocol/glossary/suggestions?status=proposed'); if(!r.ok) return;
  let d=await r.json();
  body.innerHTML=d.suggestions.map(s=>`<tr>
    <td>${escapeHtml(s.source)}</td><td>${escapeHtml(s.wrong_text)}</td><td>${escapeHtml(s.suggested_text)}</td>
    <td>${(s.confidence*100).toFixed(0)}%</td>
    <td class='actions'><button data-confirm="${s.id}">Подтвердить</button><button data-reject="${s.id}">Отклонить</button></td>
  </tr>`).join('')||"<tr><td colspan='5' class='muted'>Нет новых предложений</td></tr>";
  body.querySelectorAll('[data-confirm]').forEach(b=>b.onclick=async()=>{await fetch(`/api/protocol/glossary/suggestions/${b.dataset.confirm}/resolve`,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({action:'confirm'})});loadSuggestions();loadGlossary();});
  body.querySelectorAll('[data-reject]').forEach(b=>b.onclick=async()=>{await fetch(`/api/protocol/glossary/suggestions/${b.dataset.reject}/resolve`,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({action:'reject'})});loadSuggestions();});
}

async function loadProtocolHistory(){
  let body=document.getElementById('protocol-history-body');
  let r=await fetch('/api/protocol/jobs'); if(!r.ok) return;
  let d=await r.json();
  body.innerHTML=d.jobs.map(j=>`<tr>
    <td>${escapeHtml(j.job_id)}</td>
    <td><span class='badge ${j.status==="completed"?"completed":(j.status==="failed"||j.status==="cancelled"?"failed":"running")}'>${escapeHtml(j.status)}</span></td>
    <td>${escapeHtml(j.model_id||'')}</td>
    <td>${j.chunk_total?`${j.chunk_current||0}/${j.chunk_total}`:'—'}</td>
    <td>${new Date(j.created_at*1000).toLocaleString()}</td>
  </tr>`).join('')||"<tr><td colspan='5' class='muted'>Заданий протоколирования ещё не было</td></tr>";
}

async function loadProtocolSettings(){
  let r=await fetch('/api/protocol/settings'); if(!r.ok) return;
  let s=await r.json();
  let form=document.getElementById('protocol-settings-form');
  for(let [k,v] of Object.entries(s)){
    let el=form.elements[k]; if(!el) continue;
    if(el.type==='checkbox') el.checked=(v===true||v==='true'); else el.value=v;
  }
}
document.getElementById('protocol-settings-form').onsubmit=async(e)=>{
  e.preventDefault();
  let fd=new FormData(e.target);
  let payload={};
  for(let el of e.target.elements){
    if(!el.name) continue;
    payload[el.name]=el.type==='checkbox'?el.checked:el.value;
  }
  let r=await fetch('/api/protocol/settings', {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify(payload)});
  document.getElementById('protocol-settings-message').textContent=r.ok?'Настройки сохранены':'Ошибка сохранения';
};

loadPrompt();
loadProtocolModels();
loadGlossary();
loadSuggestions();
loadProtocolHistory();
loadProtocolSettings();
"""
else:
    PROTOCOL_ADMIN_SECTION = ""
    PROTOCOL_ADMIN_SCRIPT = ""

ADMIN_HTML = ("<!doctype html><meta charset='utf-8'><title>GigaScribe — администрирование</title><style>" + BASE_CSS + """
.protocol-tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px}
.tab-btn{padding:6px 12px;border-radius:6px;border:1px solid #8884;background:transparent;cursor:pointer;color:inherit}
.tab-btn.active{background:#8882}
.tab-panel textarea{font-family:monospace}
</style>
<header>
  <h1>⚙ GigaScribe — Администрирование</h1>
  <nav><a href='/'>← К транскрибации</a><form method='post' action='/logout' style='display:inline'><button>Выйти</button></form></nav>
</header>

<section class='card'>
  <h2>Пользователи</h2>
  <table><thead><tr><th>Логин</th><th>Статус</th><th></th></tr></thead><tbody id='users-body'></tbody></table>
  <h3>Добавить пользователя</h3>
  <form id='new-user-form'>
    <label>Логин<input name='username' required></label>
    <label>Пароль<input name='password' type='password' minlength='6' required placeholder='Минимум 6 символов'></label>
    <button type='submit'>Создать</button>
  </form>
  <p id='user-message' class='muted'></p>
</section>

<section class='card'>
  <h2>Модели</h2>
  <table><thead><tr><th>Модель</th><th>Тип</th><th>Статус</th><th>Активна</th><th>Размер</th><th></th></tr></thead><tbody id='models-body'></tbody></table>
  <p id='model-message' class='muted'></p>
</section>
""" + PROTOCOL_ADMIN_SECTION + """
<script>
function escapeHtml(s){return (s==null?'':String(s)).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function escapeAttr(s){return escapeHtml(s).replace(/`/g,'&#96;')}
function fmtSize(bytes){if(!bytes) return '—';let units=['Б','КБ','МБ','ГБ'];let i=0;while(bytes>=1024&&i<units.length-1){bytes/=1024;i++}return bytes.toFixed(1)+' '+units[i]}

async function loadUsers(){
  let body=document.getElementById('users-body');
  let r=await fetch('/admin/users');
  if(!r.ok){ body.innerHTML="<tr><td colspan='3'>Ошибка загрузки списка пользователей</td></tr>"; return; }
  let data=await r.json();
  body.innerHTML=data.users.map(u=>`<tr>
    <td>${escapeHtml(u.username)}${u.is_admin?" <span class='badge completed'>админ</span>":''}</td>
    <td>${u.disabled?"<span class='badge cancelled'>отключён</span>":"<span class='badge completed'>активен</span>"}</td>
    <td>${u.is_admin?'':`<button data-user="${escapeAttr(u.username)}" data-disable="${u.disabled?'0':'1'}" class='toggle-user'>${u.disabled?'Включить':'Отключить'}</button>`}</td>
  </tr>`).join('');
  body.querySelectorAll('.toggle-user').forEach(btn=>btn.onclick=()=>toggleUser(btn.dataset.user, btn.dataset.disable==='1'));
}

async function toggleUser(username, disable){
  await fetch(`/admin/users/${encodeURIComponent(username)}/${disable?'disable':'enable'}`, {method:'POST'});
  loadUsers();
}

document.getElementById('new-user-form').onsubmit=async function(e){
  e.preventDefault();
  let msg=document.getElementById('user-message');
  let r=await fetch('/admin/users', {method:'POST', body:new FormData(e.target)});
  if(r.ok){ msg.textContent='Пользователь создан'; e.target.reset(); loadUsers(); }
  else { let d=await r.json().catch(()=>({})); msg.textContent='Ошибка: '+(d.detail||r.status); }
};

async function loadModels(){
  let body=document.getElementById('models-body');
  let r=await fetch('/api/models/status'); if(!r.ok) return;
  let data=await r.json();
  body.innerHTML=data.models.map(m=>{
    let activeCell=m.active?'✅':(m.kind==='diarization'?`<button data-select="${escapeAttr(m.id)}" ${m.installed?'':'disabled'} class='select-model'>Выбрать</button>`:'—');
    let actionCell='';
    if(m.installed && (m.kind==='asr' || m.id!=='none')) actionCell+=`<button data-test="${escapeAttr(m.id)}" class='test-model'>Тест</button> `;
    if(m.installed && m.kind==='diarization' && m.id!=='none') actionCell+=`<button data-delete="${escapeAttr(m.id)}" class='delete-model'>Удалить</button>`;
    return `<tr>
      <td>${escapeHtml(m.label)}</td>
      <td>${m.kind==='asr'?'ASR':'Диаризация'}</td>
      <td>${m.installed?"<span class='badge completed'>установлена</span>":"<span class='badge failed'>не установлена</span>"}</td>
      <td>${activeCell}</td>
      <td>${fmtSize(m.size)}</td>
      <td class='actions'>${actionCell}</td>
    </tr>`;
  }).join('');
  body.querySelectorAll('.select-model').forEach(btn=>btn.onclick=()=>selectDiarization(btn.dataset.select));
  body.querySelectorAll('.test-model').forEach(btn=>btn.onclick=()=>testModel(btn.dataset.test));
  body.querySelectorAll('.delete-model').forEach(btn=>btn.onclick=()=>deleteModel(btn.dataset.delete));
}

async function selectDiarization(id){
  let msg=document.getElementById('model-message');
  let r=await fetch('/api/models/select', {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify({diarization_model:id})});
  let d=await r.json().catch(()=>({}));
  msg.textContent=r.ok?'Сохранено, применится к следующему заданию':('Ошибка: '+(d.error||d.detail||r.status));
  loadModels();
}
async function testModel(id){
  let msg=document.getElementById('model-message');
  msg.textContent='Тестируем '+id+'…';
  let r=await fetch(`/api/models/${id}/test`, {method:'POST'});
  let d=await r.json().catch(()=>({}));
  msg.textContent=(r.ok&&d.ok)?('Модель '+id+': тест пройден'):('Модель '+id+': тест не пройден — '+(d.error||r.status));
}
async function deleteModel(id){
  if(!confirm('Удалить модель '+id+' с диска?')) return;
  let msg=document.getElementById('model-message');
  let r=await fetch(`/api/models/${id}`, {method:'DELETE'});
  let d=await r.json().catch(()=>({}));
  msg.textContent=r.ok?'Модель удалена':('Ошибка: '+(d.detail||r.status));
  loadModels();
}

loadUsers();
loadModels();
""" + PROTOCOL_ADMIN_SCRIPT + "</script>")
