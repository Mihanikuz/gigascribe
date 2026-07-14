import importlib
import os
from pathlib import Path

import pytest
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


@pytest.fixture()
def srv(tmp_path, monkeypatch):
    monkeypatch.setenv("GIGASCRIBE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GIGASCRIBE_USERS_FILE", str(tmp_path / "data" / "users.json"))
    monkeypatch.setenv("GIGASCRIBE_ADMIN_PASSWORD", "adminpw")
    monkeypatch.setenv("GIGASCRIBE_MAX_UPLOAD_BYTES", "4")
    import server
    importlib.reload(server)
    return server


def test_non_admin_cannot_create_users(srv):
    users = srv._load_users()
    users["bob"] = {"password_hash": srv.pwd_context.hash("pw"), "disabled": False}
    srv.USER_DB.write_text(__import__('json').dumps(users), encoding="utf-8")
    c = TestClient(srv.app)
    r = c.post("/admin/users", auth=("bob", "pw"), data={"username": "alice", "password": "pw"})
    assert r.status_code == 403


def test_username_cannot_escape_data_dir(srv):
    uid = srv.safe_user_id("../evil")
    path = srv.ensure_inside_base(srv.UPLOAD_DIR / uid / "job")
    assert path.is_relative_to(srv.BASE_DIR)
    assert ".." not in uid


def test_upload_size_limit_removes_partial_file(srv, monkeypatch):
    c = TestClient(srv.app)
    c.post("/login", data={"username":"admin","password":"adminpw"})
    r = c.post("/api/jobs", files={"file": ("a.wav", b"12345", "audio/wav")})
    assert r.status_code == 413
    assert not list(srv.UPLOAD_DIR.rglob("*.wav"))


def test_bad_extension_rejected_before_processing(srv):
    c = TestClient(srv.app)
    c.post("/login", data={"username":"admin","password":"adminpw"})
    r = c.post("/api/jobs", files={"file": ("a.exe", b"1", "application/octet-stream")})
    assert r.status_code == 400


def test_health_live_and_ready(srv, tmp_path):
    import model_store as ms
    ckpt=ms.gigaam_checkpoint_path(srv.MODELS_DIR, "v2_ctc")
    ckpt.parent.mkdir(parents=True, exist_ok=True); ckpt.write_bytes(b"ok")
    ms.write_gigaam_marker(srv.MODELS_DIR, "v2_ctc")
    c = TestClient(srv.app)
    assert c.get("/health/live").status_code == 200
    r = c.get("/health/ready")
    assert r.status_code in (200, 503)
    assert "pyannote" in r.json()["checks"]
