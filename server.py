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


from job_store import JobStore
job_store = JobStore(BASE_DIR / "jobs.sqlite3")
job_store.recover()
# GPU work is serialized: one CUDA worker per container.
gpu_worker = giga_app.GPU_JOB_LOCK
scheduled_jobs: set[str] = set()
model_download_locks: dict[str, asyncio.Lock] = {}
MODEL_MANAGER = giga_app.MODEL_MANAGER


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


def ensure_inside_base(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(BASE_DIR):
        raise HTTPException(status_code=400, detail="Invalid storage path")
    return resolved


def cleanup_expired_originals() -> int:
    """Delete uploaded originals older than ORIGINAL_RETENTION_DAYS.

    Only the original upload is removed; transcript/log/M4A/FLAC and the job
    row itself are untouched. Queued/running jobs are never touched.
    """
    if ORIGINAL_RETENTION_DAYS <= 0:
        return 0
    cutoff = time.time() - ORIGINAL_RETENTION_DAYS * 86400
    deleted = 0
    for job in job_store.list_deletable_originals(cutoff):
        path = Path(job["original_path"])
        try:
            resolved = ensure_inside_base(path)
        except HTTPException:
            logger.warning("Refusing to delete original outside data dir job_id=%s path=%s", job["id"], path)
            continue
        resolved.unlink(missing_ok=True)
        job_store.update(job["id"], original_path=None)
        deleted += 1
    if deleted:
        logger.info("Deleted %d expired original upload(s) older than %d day(s)", deleted, ORIGINAL_RETENTION_DAYS)
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

def serialize_job(job: dict[str, Any]) -> dict[str, Any]:
    return {"id":job["id"],"owner":job["username"],"filename":job["filename"],"status":job["status"],"progress":job["progress"],"message":job["message"],"error":job["error"],"settings":job["settings_snapshot"],"requested_models":job["requested_models"],"requested_device":job["requested_device"],"actual_device":job["actual_device"],"actual_models":job["actual_models"],"attempts":job["attempts"],"correlation_id":job["correlation_id"],"downloads":{k:bool(job.get(col)) for k,col in DOWNLOAD_KINDS.items()}}

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
let ME={username:null,is_admin:false};

async function loadWhoAmI(){
  let r=await fetch('/api/whoami'); if(!r.ok) return;
  ME=await r.json();
  document.getElementById('whoami').textContent='Вы вошли как: '+ME.username;
  if(ME.is_admin) document.getElementById('admin-link').style.display='';
}

async function loadModelsInfo(){
  let r=await fetch('/api/models'); if(!r.ok) return;
  let m=await r.json();
  let diar=m.diarization[m.settings.diarization_model];
  let text='Диаризация: '+escapeHtml(diar?diar.label:m.settings.diarization_model);
  if(ME.is_admin) text+=' — изменить можно в администрировании';
  document.getElementById('diarization-info').textContent=text;
}

function renderJob(j){
  let canControl=(j.owner===ME.username)||ME.is_admin;
  let statusLabel=STATUS_LABELS[j.status]||j.status;
  let downloads=Object.keys(DOWNLOAD_LABELS).filter(k=>j.downloads[k]).map(k=>`<a href='/api/jobs/${j.id}/download/${k}'>${DOWNLOAD_LABELS[k]}</a>`).join(' · ');
  let actions='';
  if(canControl&&(j.status==='queued'||j.status==='running')) actions+=`<button onclick="cancelJob('${j.id}')">Отменить</button>`;
  if(canControl&&(j.status==='failed'||j.status==='cancelled')) actions+=`<button onclick="retryJob('${j.id}')">Повторить</button>`;
  return `<div class='job card'>
    <div class='row'><b>${escapeHtml(j.filename)}</b><span class='badge ${j.status}'>${statusLabel}</span></div>
    <div class='muted'>Загрузил(а): ${escapeHtml(j.owner)}${j.attempts>1?` · попытка ${j.attempts}`:''}</div>
    <progress value='${j.progress}' max='1'></progress> ${Math.round(j.progress*100)}%
    <div class='muted'>${escapeHtml(j.message)}</div>
    ${j.error?`<div class='err'>${escapeHtml(j.error)}</div>`:''}
    <div class='row'><span>${downloads}</span><span class='actions'>${actions}</span></div>
  </div>`;
}

async function refresh(){
  let r=await fetch('/api/jobs'); if(!r.ok) return;
  let js=await r.json();
  document.getElementById('jobs').innerHTML=js.map(renderJob).join('')||"<p class='muted'>Пока нет ни одной транскрипции.</p>";
}

async function cancelJob(id){await fetch(`/api/jobs/${id}/cancel`,{method:'POST'});refresh()}
async function retryJob(id){await fetch(`/api/jobs/${id}/retry`,{method:'POST'});refresh()}

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

(async()=>{ await loadWhoAmI(); await loadModelsInfo(); tick(); })();
</script>""")

ADMIN_DENIED_HTML = ("<!doctype html><meta charset='utf-8'><title>GigaScribe — доступ запрещён</title><style>" + BASE_CSS +
                      "body{max-width:420px;margin-top:15vh;text-align:center}</style>"
                      "<h1>⛔ Доступ запрещён</h1>"
                      "<p>Эта страница доступна только администратору.</p>"
                      "<p><a href='/'>На главную</a></p>")

ADMIN_HTML = ("<!doctype html><meta charset='utf-8'><title>GigaScribe — администрирование</title><style>" + BASE_CSS + """
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
</script>""")
