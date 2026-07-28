"""Session auth for /admin, user management, and the shared job history."""
import os
import tempfile
import time

_TMP = tempfile.mkdtemp(prefix="gigascribe-test-")
os.environ["GIGASCRIBE_DATA_DIR"] = os.path.join(_TMP, "data")
os.environ["GIGASCRIBE_MODELS_DIR"] = os.path.join(_TMP, "models")
os.environ["GIGASCRIBE_ADMIN_PASSWORD"] = "SuperSecretAdminPass123"
os.environ["GIGASCRIBE_SECRET_KEY"] = "test-secret-key"

import pytest
from fastapi.testclient import TestClient

import server


@pytest.fixture()
def client():
    with TestClient(server.app) as c:
        yield c


def _login(client, username, password):
    return client.post("/login", data={"username": username, "password": password}, follow_redirects=False)


def _make_job(job_id: str, username: str, status: str = "completed") -> None:
    server.job_store.create(
        id=job_id, username=username, filename=f"{job_id}.wav",
        original_path=f"/tmp/{job_id}.wav", log_path=f"/tmp/{job_id}.log",
        settings_snapshot={"asr_model": "gigaam-v3-e2e-rnnt", "diarization_model": "none", "device": "cuda"},
        message="Готово",
    )
    if status != "queued":
        server.job_store.update(job_id, status="running", started_at=time.time())
        server.job_store.update(job_id, status=status, finished_at=time.time(), progress=1)


def test_anonymous_is_redirected_to_login(client):
    assert client.get("/", follow_redirects=False).status_code == 307
    assert client.get("/admin", follow_redirects=False).headers["location"] == "/login"


def test_admin_page_requires_admin_session(client):
    _login(client, "admin", "SuperSecretAdminPass123")
    assert client.get("/admin").status_code == 200
    assert client.get("/api/whoami").json() == {"username": "admin", "is_admin": True}


def test_non_admin_is_denied_admin_page_and_api(client):
    client.post("/admin/users", data={"username": "bob", "password": "bobpassword"})  # unauthenticated -> 401
    _login(client, "admin", "SuperSecretAdminPass123")
    assert client.post("/admin/users", data={"username": "bob", "password": "bobpassword"}).status_code == 200

    bob = TestClient(server.app)
    _login(bob, "bob", "bobpassword")
    assert bob.get("/admin").status_code == 403
    assert bob.get("/admin/users").status_code == 403
    assert bob.post("/admin/users", data={"username": "carol", "password": "carolpassword"}).status_code == 403


def test_duplicate_username_is_rejected(client):
    _login(client, "admin", "SuperSecretAdminPass123")
    client.post("/admin/users", data={"username": "dave", "password": "davepassword"})
    r = client.post("/admin/users", data={"username": "dave", "password": "otherpassword"})
    assert r.status_code == 409


def test_admin_account_cannot_be_disabled(client):
    _login(client, "admin", "SuperSecretAdminPass123")
    r = client.post("/admin/users/admin/disable")
    assert r.status_code == 400


def test_disabled_user_cannot_log_in(client):
    _login(client, "admin", "SuperSecretAdminPass123")
    client.post("/admin/users", data={"username": "erin", "password": "erinpassword"})
    client.post("/admin/users/erin/disable")

    erin = TestClient(server.app)
    r = _login(erin, "erin", "erinpassword")
    assert r.status_code == 401


def test_job_history_is_shared_across_users(client):
    _login(client, "admin", "SuperSecretAdminPass123")
    client.post("/admin/users", data={"username": "frank", "password": "frankpassword"})
    _make_job("job-shared-1", "admin")

    frank = TestClient(server.app)
    _login(frank, "frank", "frankpassword")
    job_ids = [j["id"] for j in frank.get("/api/jobs").json()]
    assert "job-shared-1" in job_ids
    assert frank.get("/api/jobs/job-shared-1").status_code == 200


def test_only_owner_or_admin_can_cancel_or_retry(client):
    _login(client, "admin", "SuperSecretAdminPass123")
    client.post("/admin/users", data={"username": "grace", "password": "gracepassword"})
    _make_job("job-owned-by-admin", "admin", status="failed")

    grace = TestClient(server.app)
    _login(grace, "grace", "gracepassword")
    assert grace.post("/api/jobs/job-owned-by-admin/retry").status_code == 403
    # Admin can act on anyone's job.
    assert client.post("/api/jobs/job-owned-by-admin/retry").status_code == 200
