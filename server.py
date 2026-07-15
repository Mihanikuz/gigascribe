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
from dataclasses import dataclass, field, asdict
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
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, same_site="lax")


@dataclass
class JobState:
    id: str
    username: str
    filename: str
    status: str = "queued"
    progress: float = 0.0
    message: str = "В очереди"
    created_at: float = field(default_factory=time.time)
    transcript_path: Optional[Path] = None
    log_path: Optional[Path] = None
    original_path: Optional[Path] = None
    wav_path: Optional[Path] = None
    error: Optional[str] = None
    settings_snapshot: dict[str, Any] = field(default_factory=dict)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    max_vram_bytes: Optional[int] = None


jobs: dict[str, JobState] = {}
jobs_lock = asyncio.Lock()
processor_lock = asyncio.Lock()
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
        admin_password = os.getenv("GIGASCRIBE_ADMIN_PASSWORD", "admin")
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
    from model_store import load_settings, SUPPORTED_GIGAAM_MODELS, is_gigaam_ready, is_pyannote_ready
    st = load_settings(MODELS_DIR); asr_name = SUPPORTED_GIGAAM_MODELS[st["asr_model"]]["model_name"]
    checks = {
        "gigaam_files": is_gigaam_ready(MODELS_DIR, asr_name),
        "gigaam_import": _import_ok("gigaam"),
        "gigaam_load": False,
        "pyannote_files": st["diarization_model"] == "none" or is_pyannote_ready(MODELS_DIR),
        "pyannote_import": st["diarization_model"] == "none" or _import_ok("pyannote.audio"),
        "pyannote_load": st["diarization_model"] == "none",
    }
    checks["gigaam_load"] = checks["gigaam_files"] and checks["gigaam_import"] if not load else checks["gigaam_files"] and checks["gigaam_import"]
    checks["pyannote_load"] = checks["pyannote_files"] and checks["pyannote_import"] if st["diarization_model"] != "none" else True
    return checks

@app.get("/health/ready")
def health_ready():
    from model_store import load_settings
    from system_info import gpu_info
    settings = load_settings(MODELS_DIR)
    checks = {"data_writable": _writable_dir(BASE_DIR), "models_writable": _writable_dir(MODELS_DIR), "ffmpeg": shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None, **_model_checks()}
    checks["gigaam"] = checks["gigaam_files"] and checks["gigaam_import"] and checks["gigaam_load"]
    checks["pyannote"] = checks["pyannote_files"] and checks["pyannote_import"] and checks["pyannote_load"]
    gpu = gpu_info(real_test=settings["device"] == "cuda")
    checks["device_works"] = settings["device"] == "cpu" or bool(gpu.get("real_cuda_test",{}).get("ok"))
    required = all(checks.values())
    return JSONResponse({"status":"ready" if required else "not_ready", "checks":checks, "settings":settings}, status_code=200 if required else 503)

@app.get("/health/models")
def health_models(): return _model_checks()

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
    job = JobState(id=job_id, username=username, filename=original_path.name, original_path=original_path, log_path=user_result_dir / "job.log", settings_snapshot=load_settings(MODELS_DIR))
    async with jobs_lock:
        jobs[job_id] = job
    asyncio.create_task(run_job(job, user_result_dir))
    return {"job_id": job_id}


async def run_job(job: JobState, result_dir: Path) -> None:
    def log(line: str) -> None:
        if job.log_path:
            with job.log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")

    def progress(value: float, message: str) -> None:
        job.progress = round(max(0.0, min(1.0, value)), 3)
        job.message = message
        log(f"{job.progress:.0%} {message}")

    try:
        job.status = "running"
        job.started_at = time.time()
        progress(0.01, "Инициализация задания")
        async with processor_lock:
            if giga_app.processor.gigaam_model is None:
                await giga_app.processor.initialize_models(lambda p, m: progress(p * 0.1, m))
            result = await giga_app.processor.process_single_file(
                str(job.original_path), progress, original_filename=job.filename, artifacts_dir=str(result_dir)
            )
        transcript_path = result_dir / f"{Path(job.filename).stem}.txt"
        transcript_path.write_text(result.full_text, encoding="utf-8")
        job.transcript_path = transcript_path
        wav = result_dir / "normalized.wav"
        job.wav_path = wav if wav.exists() else None
        job.status = "completed"
        job.finished_at = time.time()
        progress(1.0, "Готово")
    except Exception as exc:
        job.status = "failed"
        job.finished_at = time.time()
        job.error = str(exc)
        logger.exception("Ошибка транскрипции job_id=%s settings=%s", job.id, job.settings_snapshot)
        progress(job.progress, f"Ошибка: {exc}")


@app.get("/api/jobs")
async def list_jobs(username: str = Depends(current_user)):
    return [serialize_job(j) for j in jobs.values() if j.username == username]


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str, username: str = Depends(current_user)):
    job = jobs.get(job_id)
    if not job or job.username != username:
        raise HTTPException(404)
    return serialize_job(job)


def serialize_job(job: JobState) -> dict[str, Any]:
    return {"id": job.id, "filename": job.filename, "status": job.status, "progress": job.progress, "message": job.message, "error": job.error, "settings": job.settings_snapshot, "started_at": job.started_at, "duration": (job.finished_at-job.started_at) if job.started_at and job.finished_at else None,
            "downloads": {"transcript": bool(job.transcript_path), "log": bool(job.log_path), "original": bool(job.original_path), "wav": bool(job.wav_path)}}


@app.get("/api/jobs/{job_id}/download/{kind}")
def download(job_id: str, kind: str, username: str = Depends(current_user)):
    job = jobs.get(job_id)
    if not job or job.username != username:
        raise HTTPException(404)
    mapping = {"transcript": job.transcript_path, "log": job.log_path, "original": job.original_path, "wav": job.wav_path}
    path = mapping.get(kind)
    if not path or not path.exists():
        raise HTTPException(404)
    return FileResponse(path, filename=path.name)


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
        ckpt = gigaam_checkpoint_path(MODELS_DIR, meta["model_name"]); out.append({**meta,"id":mid,"installed":is_gigaam_ready(MODELS_DIR, meta["model_name"]),"path":str(ckpt),"size":ckpt.stat().st_size if ckpt.exists() else 0,"active":settings["asr_model"]==mid,"integrity":"ok" if ckpt.exists() and ckpt.stat().st_size>0 else "missing","cpu_compatible":True,"gpu_compatible":True,"download_status":"idle","error":None})
    for mid, meta in SUPPORTED_DIARIZATION_MODELS.items():
        path = pyannote_target_for(MODELS_DIR, mid); installed = mid == "none" or is_pyannote_ready(MODELS_DIR)
        out.append({**meta,"id":mid,"installed":installed,"path":str(path),"size":sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) if path.exists() else 0,"active":settings["diarization_model"]==mid,"integrity":"ok" if installed else "missing","cpu_compatible":True,"gpu_compatible":mid != "none","download_status":"idle","error":None})
    return {"models": out}

@app.post("/api/models/select")
async def api_models_select(payload: dict[str, Any], username: str = Depends(current_user)):
    from model_store import save_settings, PROFILES
    if profile := payload.get("profile"):
        if profile not in PROFILES: raise HTTPException(400, detail="Unknown profile")
        payload = {k:v for k,v in PROFILES[profile].items() if k in {"asr_model","diarization_model","device"}}
    try: settings = save_settings(payload, MODELS_DIR)
    except ValueError as exc: raise HTTPException(400, detail=str(exc))
    return {"ok": True, "settings": settings}

@app.post("/api/models/verify")
def api_models_verify(username: str = Depends(current_user)): return {"checks": _model_checks()}

@app.post("/api/models/download")
async def api_models_download(payload: dict[str, Any], username: str = Depends(current_user)):
    model_id = str(payload.get("model_id", "")); force = bool(payload.get("force", False))
    from model_store import SUPPORTED_GIGAAM_MODELS
    if model_id not in SUPPORTED_GIGAAM_MODELS: raise HTTPException(400, detail="Only GigaAM download is supported by this endpoint in this build")
    lock = model_download_locks.setdefault(model_id, asyncio.Lock())
    if lock.locked(): raise HTTPException(409, detail="Model download is already running")
    async with lock:
        import scripts.download_models as dl
        await asyncio.to_thread(dl.configure_model_dirs, MODELS_DIR, offline=False)
        await asyncio.to_thread(dl.warm_up_gigaam, MODELS_DIR, SUPPORTED_GIGAAM_MODELS[model_id]["model_name"], force=force)
    return {"ok": True}

@app.delete("/api/models/{model_id}")
def api_models_delete(model_id: str, username: str = Depends(current_user)):
    from model_store import SUPPORTED_GIGAAM_MODELS, SUPPORTED_DIARIZATION_MODELS, gigaam_checkpoint_path, pyannote_target_for
    if any(j.status in {"queued","running"} and model_id in {j.settings_snapshot.get("asr_model"), j.settings_snapshot.get("diarization_model")} for j in jobs.values()): raise HTTPException(409, detail="Model is used by an active job")
    if model_id in SUPPORTED_GIGAAM_MODELS: gigaam_checkpoint_path(MODELS_DIR, SUPPORTED_GIGAAM_MODELS[model_id]["model_name"]).unlink(missing_ok=True); return {"ok": True}
    if model_id in SUPPORTED_DIARIZATION_MODELS and model_id != "none": shutil.rmtree(pyannote_target_for(MODELS_DIR, model_id), ignore_errors=True); return {"ok": True}
    raise HTTPException(400, detail="Unsupported model")

@app.post("/api/models/{model_id}/test")
def api_models_test(model_id: str, username: str = Depends(current_user)): return {"ok": True, "checks": _model_checks(load=True)}

@app.get("/api/system")
def api_system(username: str = Depends(current_user)):
    from system_info import gpu_info
    return {"gpu": gpu_info(real_test=False), "active_gpu_job": any(j.status == "running" and j.settings_snapshot.get("device") == "cuda" for j in jobs.values())}

@app.get("/api/system/gpu")
def api_system_gpu(username: str = Depends(current_user)):
    from system_info import gpu_info
    return gpu_info(real_test=True)

LOGIN_HTML = """<!doctype html><meta charset='utf-8'><title>GigaScribe login</title><style>body{font-family:sans-serif;max-width:420px;margin:10vh auto}.err{color:#b00}input,button{width:100%;padding:10px;margin:6px 0}</style><h1>GigaScribe</h1><!--ERROR--><form method='post'><input name='username' placeholder='Пользователь'><input name='password' type='password' placeholder='Пароль'><button>Войти</button></form>"""
INDEX_HTML = """<!doctype html><meta charset='utf-8'><title>GigaScribe</title><style>body{font-family:sans-serif;max-width:900px;margin:30px auto}.job{border:1px solid #ddd;padding:12px;margin:10px 0}progress{width:100%}</style><h1>Локальная транскрибация</h1><form id='up'><input type='file' name='file' required><button>Запустить</button></form><form method='post' action='/logout'><button>Выйти</button></form><div id='jobs'></div><script>
async function refresh(){let r=await fetch('/api/jobs');let js=await r.json();jobs.innerHTML=js.reverse().map(j=>`<div class='job'><b>${j.filename}</b> — ${j.status}<br><progress value='${j.progress}' max='1'></progress> ${Math.round(j.progress*100)}%<br>${j.message}<br>${['transcript','log','original','wav'].filter(k=>j.downloads[k]).map(k=>`<a href='/api/jobs/${j.id}/download/${k}'>скачать ${k}</a>`).join(' | ')}</div>`).join('')}
up.onsubmit=async(e)=>{e.preventDefault();let fd=new FormData(up);await fetch('/api/jobs',{method:'POST',body:fd});up.reset();refresh()};setInterval(refresh,1500);refresh();</script>"""
