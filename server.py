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
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from passlib.context import CryptContext
from starlette.middleware.sessions import SessionMiddleware

APP_MODULE_PATH = Path(__file__).with_name("app (1).py")
spec = importlib.util.spec_from_file_location("gigascribe_gradio_app", APP_MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load {APP_MODULE_PATH}")
giga_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(giga_app)

MODELS_DIR = Path(os.getenv("GIGASCRIBE_MODELS_DIR", "./models")).resolve()
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "torch"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

BASE_DIR = Path(os.getenv("GIGASCRIBE_DATA_DIR", "./data")).resolve()
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
USER_DB = Path(os.getenv("GIGASCRIBE_USERS_FILE", BASE_DIR / "users.json"))
SECRET_KEY = os.getenv("GIGASCRIBE_SECRET_KEY", secrets.token_urlsafe(32))
for directory in (UPLOAD_DIR, RESULT_DIR, USER_DB.parent):
    directory.mkdir(parents=True, exist_ok=True)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBasic()
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


jobs: dict[str, JobState] = {}
jobs_lock = asyncio.Lock()
processor_lock = asyncio.Lock()


def _load_users() -> dict[str, Any]:
    if not USER_DB.exists():
        admin_password = os.getenv("GIGASCRIBE_ADMIN_PASSWORD", "admin")
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
    cleaned = "".join(ch for ch in name if ch.isalnum() or ch in "._- ()").strip()
    return cleaned or "audio"


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
    job_id = uuid.uuid4().hex
    user_upload_dir = UPLOAD_DIR / username / job_id
    user_result_dir = RESULT_DIR / username / job_id
    user_upload_dir.mkdir(parents=True, exist_ok=True)
    user_result_dir.mkdir(parents=True, exist_ok=True)
    original_path = user_upload_dir / safe_name(file.filename or "audio")
    with original_path.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    job = JobState(id=job_id, username=username, filename=original_path.name, original_path=original_path, log_path=user_result_dir / "job.log")
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
        job.status = "done"
        progress(1.0, "Готово")
    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
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
    return {"id": job.id, "filename": job.filename, "status": job.status, "progress": job.progress, "message": job.message, "error": job.error,
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
    if not _verify_local(credentials.username, credentials.password):
        raise HTTPException(401)
    users = _load_users()
    users[username] = {"password_hash": pwd_context.hash(password), "disabled": False}
    USER_DB.write_text(json.dumps(users, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"ok": True}


LOGIN_HTML = """<!doctype html><meta charset='utf-8'><title>GigaScribe login</title><style>body{font-family:sans-serif;max-width:420px;margin:10vh auto}.err{color:#b00}input,button{width:100%;padding:10px;margin:6px 0}</style><h1>GigaScribe</h1><!--ERROR--><form method='post'><input name='username' placeholder='Пользователь'><input name='password' type='password' placeholder='Пароль'><button>Войти</button></form>"""
INDEX_HTML = """<!doctype html><meta charset='utf-8'><title>GigaScribe</title><style>body{font-family:sans-serif;max-width:900px;margin:30px auto}.job{border:1px solid #ddd;padding:12px;margin:10px 0}progress{width:100%}</style><h1>Локальная транскрибация</h1><form id='up'><input type='file' name='file' required><button>Запустить</button></form><form method='post' action='/logout'><button>Выйти</button></form><div id='jobs'></div><script>
async function refresh(){let r=await fetch('/api/jobs');let js=await r.json();jobs.innerHTML=js.reverse().map(j=>`<div class='job'><b>${j.filename}</b> — ${j.status}<br><progress value='${j.progress}' max='1'></progress> ${Math.round(j.progress*100)}%<br>${j.message}<br>${['transcript','log','original','wav'].filter(k=>j.downloads[k]).map(k=>`<a href='/api/jobs/${j.id}/download/${k}'>скачать ${k}</a>`).join(' | ')}</div>`).join('')}
up.onsubmit=async(e)=>{e.preventDefault();let fd=new FormData(up);await fetch('/api/jobs',{method:'POST',body:fd});up.reset();refresh()};setInterval(refresh,1500);refresh();</script>"""
