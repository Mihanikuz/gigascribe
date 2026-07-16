"""Local multi-user web server for GigaScribe on Ubuntu 24.04.

Features:
- local username/password authentication with optional LDAP/AD bind;
- per-user asynchronous transcription jobs;
- live progress polling from the browser;
- downloads for transcript, job log, original audio, and normalized WAV;
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
import subprocess
import time
import uuid
import logging
import importlib
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from passlib.context import CryptContext
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
MODELS_DIR.mkdir(parents=True, exist_ok=True)

BASE_DIR = Path(os.getenv("GIGASCRIBE_DATA_DIR", "./data")).resolve()
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
USER_DB = Path(os.getenv("GIGASCRIBE_USERS_FILE", BASE_DIR / "users.json"))
SECRET_KEY = os.getenv("GIGASCRIBE_SECRET_KEY", secrets.token_urlsafe(32))
SECRET_KEY_CONFIGURED = bool(os.getenv("GIGASCRIBE_SECRET_KEY"))
MAX_UPLOAD_BYTES = int(os.getenv("GIGASCRIBE_MAX_UPLOAD_BYTES", str(1024 * 1024 * 1024)))
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
security = HTTPBasic()
logger = logging.getLogger(__name__)
app = FastAPI(title="GigaScribe Local Server")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, same_site=os.getenv("GIGASCRIBE_SESSION_SAMESITE", "lax"), https_only=os.getenv("GIGASCRIBE_SESSION_SECURE", "0") == "1")

@app.on_event("startup")
async def resume_persistent_queue() -> None:
    for row in job_store.list(active_only=True):
        if row["status"] == "queued": schedule_job(row["id"])


from job_store import JobStore
job_store = JobStore(BASE_DIR / "jobs.sqlite3")
job_store.recover()
# Model instances are isolated by immutable snapshot.  GPU work is serialized;
# CPU concurrency is explicitly configurable.
gpu_worker = asyncio.Semaphore(max(1, int(os.getenv("GIGASCRIBE_GPU_WORKERS", "1"))))
cpu_worker = asyncio.Semaphore(max(1, int(os.getenv("GIGASCRIBE_CPU_WORKERS", "2"))))
scheduled_jobs: set[str] = set()
model_download_locks: dict[str, asyncio.Lock] = {}


def _validate_password(password: str) -> None:
    byte_len = len(password.encode("utf-8"))
    if len(password) < 6:
        raise HTTPException(400, detail="Пароль должен содержать не менее 6 символов")
    if byte_len > 72:
        raise HTTPException(400, detail="Пароль не должен превышать 72 байта для bcrypt")
    if byte_len > 4096:
        raise HTTPException(400, detail="Пароль слишком длинный")

def _load_users() -> dict[str, Any]:
    if not USER_DB.exists():
        admin_password = os.getenv("GIGASCRIBE_ADMIN_PASSWORD")
        if not admin_password or admin_password == "admin":
            raise RuntimeError("GIGASCRIBE_ADMIN_PASSWORD must be set and must not be the default 'admin'")
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
    """Report every prerequisite separately; never hide an import failure as "false"."""
    from model_store import load_settings, SUPPORTED_GIGAAM_MODELS, is_gigaam_ready, is_pyannote_ready
    st = load_settings(MODELS_DIR); asr_name = SUPPORTED_GIGAAM_MODELS[st["asr_model"]]["model_name"]
    gigaam_import = _import_ok("gigaam")
    pyannote_import = st["diarization_model"] == "none" or _import_ok("pyannote.audio")
    checks = {
        "gigaam_files": is_gigaam_ready(MODELS_DIR, asr_name),
        "gigaam_import": gigaam_import,
        "gigaam_load": False,
        "pyannote_files": st["diarization_model"] == "none" or is_pyannote_ready(MODELS_DIR, st["diarization_model"]),
        "pyannote_import": pyannote_import,
        "pyannote_load": st["diarization_model"] == "none",
    }
    # A load test is deliberately opt-in: /health/ready must remain cheap and
    # must not allocate a second model instance during normal request probes.
    # Presence is deliberately not a load result.  Explicit /health/models or
    # model test records a real load attempt; ready stays inexpensive.
    checks["gigaam_load"] = False
    checks["pyannote_load"] = st["diarization_model"] == "none"
    if load and checks["gigaam_load"]:
        try:
            import gigaam
            gigaam.load_model(asr_name, device=st["device"], download_root=str(MODELS_DIR / "gigaam-cache"))
        except Exception as exc:
            checks["gigaam_load"] = False
            checks["gigaam_error"] = str(exc)
    if load and st["diarization_model"] != "none" and checks["pyannote_load"]:
        try:
            from pyannote.audio import Pipeline
            from model_store import pyannote_target_for
            Pipeline.from_pretrained(str(pyannote_target_for(MODELS_DIR, st["diarization_model"])))
        except Exception as exc:
            checks["pyannote_load"] = False
            checks["pyannote_error"] = str(exc)
    checks["gigaam"] = {
        "import": "ok" if gigaam_import else "failed", "dependencies": "ok" if gigaam_import else "failed",
        "checkpoint": "exists" if checks["gigaam_files"] else "missing", "load": "ok" if checks["gigaam_load"] else "not_run",
        "device": "ok" if st["device"] == "cpu" or checks["gigaam_load"] else "failed",
    }
    return checks

@app.get("/health/ready")
def health_ready():
    from model_store import load_settings
    from system_info import gpu_info
    settings = load_settings(MODELS_DIR)
    checks = {"data_writable": _writable_dir(BASE_DIR), "models_writable": _writable_dir(MODELS_DIR), "secret_key_configured": SECRET_KEY_CONFIGURED, "ffmpeg": shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None, **_model_checks()}
    checks["gigaam"] = checks["gigaam_files"] and checks["gigaam_import"]
    checks["pyannote"] = checks["pyannote_files"] and checks["pyannote_import"] and checks["pyannote_load"]
    gpu = gpu_info(real_test=settings["device"] == "cuda")
    checks["device_works"] = settings["device"] == "cpu" or bool(gpu.get("real_cuda_test",{}).get("ok"))
    required = all(checks.values())
    return JSONResponse({"status":"ready" if required else "not_ready", "checks":checks, "settings":settings}, status_code=200 if required else 503)

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
    job = job_store.get(job_id)
    if not job or job["status"] != "queued": return
    result_dir = Path(job["log_path"]).parent
    def log(line: str) -> None:
        with Path(job["log_path"]).open("a", encoding="utf-8") as fh: fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")
    def progress(value: float, message: str) -> None:
        value = round(max(0.0, min(1.0, value)), 3); job_store.update(job_id, progress=value, message=message); log(f"{value:.0%} {message}")
    try:
        job_store.update(job_id, status="running", started_at=time.time(), attempts=job["attempts"] + 1)
        snapshot = job["settings_snapshot"]; progress(.01, "Инициализация задания")
        processor = giga_app.AudioProcessor(snapshot=snapshot)
        worker = gpu_worker if snapshot.get("device") == "cuda" else cpu_worker
        async with worker:
            await processor.initialize_models(lambda p,m: progress(p*.1,m))
            work = processor.process_single_file(job["original_path"], progress, original_filename=job["filename"], artifacts_dir=str(result_dir))
            result = await asyncio.wait_for(work, job["timeout_seconds"]) if job["timeout_seconds"] else await work
        if job_store.get(job_id)["cancel_requested"]:
            job_store.update(job_id,status="cancelled",finished_at=time.time(),message="Отменено"); return
        transcript = result_dir / f"{Path(job['filename']).stem}.txt"; transcript.write_text(result.full_text, encoding="utf-8")
        wav=result_dir/"normalized.wav"
        job_store.update(job_id,status="completed",finished_at=time.time(),progress=1,message="Готово",transcript_path=str(transcript),wav_path=str(wav) if wav.exists() else None,actual_device=str(processor.device),actual_models={"asr_model":snapshot.get("asr_model"),"diarization_model":snapshot.get("diarization_model")})
    except asyncio.TimeoutError:
        job_store.update(job_id,status="failed",finished_at=time.time(),error="Job timed out",message="Превышено время выполнения")
    except Exception:
        job_store.update(job_id,status="failed",finished_at=time.time(),error="Processing failed",message="Ошибка обработки")
        logger.exception("transcription failed job_id=%s model=%s device=%s", job_id, job["settings_snapshot"], job["settings_snapshot"].get("device"))
    finally: scheduled_jobs.discard(job_id)


@app.get("/api/jobs")
async def list_jobs(username: str = Depends(current_user)):
    return [serialize_job(j) for j in job_store.list(username)]

@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str, username: str = Depends(current_user)):
    job=job_store.get(job_id)
    if not job or job["username"] != username: raise HTTPException(404)
    return serialize_job(job)

@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, username: str = Depends(current_user)):
    job=job_store.get(job_id)
    if not job or job["username"] != username: raise HTTPException(404)
    if job["status"] in {"completed","failed","cancelled"}: raise HTTPException(409, detail="Job is already terminal")
    job_store.request_cancel(job_id)
    if job["status"] == "queued": job_store.update(job_id,status="cancelled",finished_at=time.time())
    return serialize_job(job_store.get(job_id))

@app.post("/api/jobs/{job_id}/retry")
async def retry_job(job_id: str, username: str = Depends(current_user)):
    job=job_store.get(job_id)
    if not job or job["username"] != username: raise HTTPException(404)
    if job["status"] not in {"failed","cancelled"}: raise HTTPException(409, detail="Only failed or cancelled jobs can be retried")
    job_store.update(job_id,status="queued",progress=0,message="В очереди",error=None,started_at=None,finished_at=None,cancel_requested=0); schedule_job(job_id)
    return serialize_job(job_store.get(job_id))

def serialize_job(job: dict[str, Any]) -> dict[str, Any]:
    return {"id":job["id"],"filename":job["filename"],"status":job["status"],"progress":job["progress"],"message":job["message"],"error":job["error"],"settings":job["settings_snapshot"],"actual_device":job["actual_device"],"actual_models":job["actual_models"],"downloads":{k:bool(job[f"{k}_path"]) for k in ("transcript","log","original","wav")}}

@app.get("/api/jobs/{job_id}/download/{kind}")
def download(job_id: str, kind: str, username: str = Depends(current_user)):
    job=job_store.get(job_id)
    if not job or job["username"] != username: raise HTTPException(404)
    path=job.get(f"{kind}_path")
    if not path or not Path(path).exists(): raise HTTPException(404)
    return FileResponse(path, filename=Path(path).name)


@app.post("/admin/users")
def create_local_user(credentials: HTTPBasicCredentials = Depends(security), username: str = Form(...), password: str = Form(...)):
    if not _verify_local(credentials.username, credentials.password) or not is_admin(credentials.username):
        raise HTTPException(403)
    if not username or len(username) > 128:
        raise HTTPException(400, detail="Invalid username")
    _validate_password(password)
    users = _load_users()
    users[username] = {"password_hash": pwd_context.hash(password), "disabled": False, "storage_id": safe_user_id(username)}
    USER_DB.write_text(json.dumps(users, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"ok": True}


@app.get("/api/models")
def api_models(username: str = Depends(current_user)):
    from model_store import SUPPORTED_GIGAAM_MODELS, SUPPORTED_DIARIZATION_MODELS, PROFILES, load_settings
    return {"asr": SUPPORTED_GIGAAM_MODELS, "diarization": SUPPORTED_DIARIZATION_MODELS, "profiles": PROFILES, "settings": load_settings(MODELS_DIR)}

@app.get("/api/models/status")
def api_models_status(username: str = Depends(current_user)):
    from model_store import SUPPORTED_GIGAAM_MODELS, SUPPORTED_DIARIZATION_MODELS, load_settings, gigaam_checkpoint_path, pyannote_target_for, is_gigaam_ready, is_pyannote_ready
    settings = load_settings(MODELS_DIR); out=[]
    for mid, meta in SUPPORTED_GIGAAM_MODELS.items():
        ckpt = gigaam_checkpoint_path(MODELS_DIR, meta["model_name"]); installed = is_gigaam_ready(MODELS_DIR, meta["model_name"])
        out.append({**meta,"id":mid,"installed":installed,"path":str(ckpt),"size":ckpt.stat().st_size if ckpt.exists() else 0,"active":settings["asr_model"]==mid,"integrity":"ok" if installed else "missing","loadable": installed and _import_ok("gigaam"),"test_inference":"not_run","cpu_compatible":True,"gpu_compatible":True,"download_status":"idle","error":None})
    for mid, meta in SUPPORTED_DIARIZATION_MODELS.items():
        path = pyannote_target_for(MODELS_DIR, mid); installed = mid == "none" or is_pyannote_ready(MODELS_DIR, mid)
        out.append({**meta,"id":mid,"installed":installed,"path":str(path),"size":sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) if path.exists() else 0,"active":settings["diarization_model"]==mid,"integrity":"ok" if installed else "missing","loadable": mid == "none" or (installed and _import_ok("pyannote.audio")),"test_inference":"not_run","cpu_compatible":True,"gpu_compatible":mid != "none","download_status":"idle","error":None})
    return {"models": out}

@app.post("/api/models/select")
async def api_models_select(payload: dict[str, Any], username: str = Depends(require_admin)):
    from model_store import save_settings, PROFILES
    if profile := payload.get("profile"):
        if profile not in PROFILES: raise HTTPException(400, detail="Unknown profile")
        payload = {k:v for k,v in PROFILES[profile].items() if k in {"asr_model","diarization_model","device"}}
    try: settings = save_settings(payload, MODELS_DIR)
    except ValueError as exc: raise HTTPException(400, detail=str(exc))
    return {"ok": True, "settings": settings}

@app.post("/api/models/verify")
def api_models_verify(username: str = Depends(require_admin)): return {"checks": _model_checks()}

@app.post("/api/models/download")
async def api_models_download(payload: dict[str, Any], username: str = Depends(require_admin)):
    model_id = str(payload.get("model_id", "")); force = bool(payload.get("force", False))
    from model_store import SUPPORTED_GIGAAM_MODELS, SUPPORTED_DIARIZATION_MODELS
    if model_id not in {*SUPPORTED_GIGAAM_MODELS, *SUPPORTED_DIARIZATION_MODELS} or model_id == "none": raise HTTPException(400, detail="Unsupported model")
    lock = model_download_locks.setdefault(model_id, asyncio.Lock())
    if lock.locked(): raise HTTPException(409, detail="Model download is already running")
    async with lock:
        import scripts.download_models as dl
        await asyncio.to_thread(dl.configure_model_dirs, MODELS_DIR, offline=False)
        if model_id in SUPPORTED_GIGAAM_MODELS:
            await asyncio.to_thread(dl.warm_up_gigaam, MODELS_DIR, SUPPORTED_GIGAAM_MODELS[model_id]["model_name"], force=force)
        else:
            await asyncio.to_thread(dl.download_pyannote, MODELS_DIR, os.getenv("HF_TOKEN") or None, model_id=model_id, force=force)
    return {"ok": True, "model_id": model_id}

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
            import gigaam; gigaam.load_model(meta["model_name"], device="cpu", download_root=str(MODELS_DIR / "gigaam-cache"))
        elif model_id in SUPPORTED_DIARIZATION_MODELS and model_id != "none":
            if not is_pyannote_ready(MODELS_DIR, model_id): raise RuntimeError("Model is not installed or integrity verification failed")
            from pyannote.audio import Pipeline; Pipeline.from_pretrained(str(pyannote_target_for(MODELS_DIR, model_id) / "config.yaml"))
        elif model_id == "none": return {"ok": True, "model_id": model_id, "load_test": "not_applicable"}
        else: raise HTTPException(404, detail="Unknown model")
        return {"ok": True, "model_id": model_id, "load_test": "ok", "inference_test": "not_run"}
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

LOGIN_HTML = """<!doctype html><meta charset='utf-8'><title>GigaScribe login</title><style>body{font-family:sans-serif;max-width:420px;margin:10vh auto}.err{color:#b00}input,button{width:100%;padding:10px;margin:6px 0}</style><h1>GigaScribe</h1><!--ERROR--><form method='post'><input name='username' placeholder='Пользователь'><input name='password' type='password' placeholder='Пароль'><button>Войти</button></form>"""
INDEX_HTML = """<!doctype html><meta charset='utf-8'><title>GigaScribe</title><style>body{font-family:sans-serif;max-width:900px;margin:30px auto}.job{border:1px solid #ddd;padding:12px;margin:10px 0}progress{width:100%}label,select{margin:4px}</style><h1>Локальная транскрибация</h1><form id='up'><input type='file' name='file' required><button>Запустить</button></form><form id='model-settings'><label>ASR <select id='asr-model'></select></label><label>Диаризация <select id='diarization-model'></select></label><button>Сохранить модели</button><span id='model-message'></span></form><form method='post' action='/logout'><button>Выйти</button></form><div id='jobs'></div><script>
async function refresh(){let r=await fetch('/api/jobs');let js=await r.json();jobs.innerHTML=js.reverse().map(j=>`<div class='job'><b>${j.filename}</b> — ${j.status}<br><progress value='${j.progress}' max='1'></progress> ${Math.round(j.progress*100)}%<br>${j.message}<br>${['transcript','log','original','wav'].filter(k=>j.downloads[k]).map(k=>`<a href='/api/jobs/${j.id}/download/${k}'>скачать ${k}</a>`).join(' | ')}</div>`).join('')}
async function loadModels(){let r=await fetch('/api/models'), m=await r.json(); for(let [target,values,current] of [['asr-model',m.asr,m.settings.asr_model],['diarization-model',m.diarization,m.settings.diarization_model]]){let s=document.getElementById(target);s.innerHTML=Object.entries(values).map(([id,v])=>`<option value='${id}'>${v.label}</option>`).join('');s.value=current}}
modelSettings=document.getElementById('model-settings'); modelSettings.onsubmit=async(e)=>{e.preventDefault();let r=await fetch('/api/models/select',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({asr_model:document.getElementById('asr-model').value,diarization_model:document.getElementById('diarization-model').value})});document.getElementById('model-message').textContent=r.ok?'Сохранено':'Ошибка сохранения'};
up.onsubmit=async(e)=>{e.preventDefault();let fd=new FormData(up);await fetch('/api/jobs',{method:'POST',body:fd});up.reset();refresh()};loadModels();setInterval(refresh,1500);refresh();</script>"""
