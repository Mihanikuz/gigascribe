"""Priority 7: admin-only api_tokens issuance/revocation, the
current_user_via_token dependency, and the /api/v1/meeting-protocol routes.

The hard constraint from the spec: this API must never be able to change any
settings (model/engine/prompts/glossary) -- configuration stays
admin-panel-only. Several tests below verify that directly.

Every test spawns its own subprocess (see test_protocol_server_integration.py
for the rationale): `server` is imported once and cached for the whole
pytest session, so setting env vars like GIGASCRIBE_PROTOCOL_ENABLED or
GIGASCRIBE_DATA_DIR at this file's module scope would leak into every other
test file collected afterwards.
"""
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_script(body: str, *, env_extra: dict[str, str]) -> subprocess.CompletedProcess:
    import os
    env = {**os.environ, **env_extra}
    script = "import sys; sys.path.insert(0, %r)\n" % str(REPO_ROOT) + textwrap.dedent(body)
    return subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT, env=env,
                           capture_output=True, text=True, timeout=60)


def _base_env(tmp_path) -> dict[str, str]:
    return {
        "GIGASCRIBE_DATA_DIR": str(tmp_path / "data"),
        "GIGASCRIBE_MODELS_DIR": str(tmp_path / "models"),
        "GIGASCRIBE_ADMIN_PASSWORD": "ARealStrongPassword123",
        "GIGASCRIBE_SECRET_KEY": "test-secret-key",
        "GIGASCRIBE_PROTOCOL_ENABLED": "1",
    }


def _run(tmp_path, body: str) -> None:
    result = _run_script(body, env_extra=_base_env(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_token_creation_requires_admin(tmp_path):
    _run(tmp_path, """
        import server
        from fastapi.testclient import TestClient

        with TestClient(server.app) as c:
            assert c.post("/admin/api-tokens", data={"label": "bot", "username": "admin"}).status_code == 401

            c.post("/login", data={"username": "admin", "password": "ARealStrongPassword123"})
            c.post("/admin/users", data={"username": "bob", "password": "bobpassword"})

            bob = TestClient(server.app)
            bob.post("/login", data={"username": "bob", "password": "bobpassword"})
            r = bob.post("/admin/api-tokens", data={"label": "bot", "username": "bob"})
            assert r.status_code == 403, "no self-service token creation -- admin only"
        print("OK")
    """)


def test_token_creation_rejects_unknown_username(tmp_path):
    _run(tmp_path, """
        import server
        from fastapi.testclient import TestClient

        with TestClient(server.app) as c:
            c.post("/login", data={"username": "admin", "password": "ARealStrongPassword123"})
            r = c.post("/admin/api-tokens", data={"label": "bot", "username": "nobody"})
            assert r.status_code == 400
        print("OK")
    """)


def test_token_is_shown_once_and_authenticates(tmp_path):
    _run(tmp_path, """
        import server
        from fastapi.testclient import TestClient

        with TestClient(server.app) as c:
            c.post("/login", data={"username": "admin", "password": "ARealStrongPassword123"})
            r = c.post("/admin/api-tokens", data={"label": "telegram-bot", "username": "admin"})
            assert r.status_code == 200, r.text
            token = r.json()["token"]
            assert token.startswith("gsp_")

            listed = c.get("/admin/api-tokens").json()["tokens"]
            assert len(listed) == 1
            assert "token" not in listed[0] and "secret_hash" not in listed[0]
            assert listed[0]["label"] == "telegram-bot"
            assert listed[0]["username"] == "admin"

            r2 = c.get("/api/v1/meeting-protocol/does-not-exist", headers={"Authorization": f"Bearer {token}"})
            assert r2.status_code == 404  # authenticated, just no such job
        print("OK")
    """)


def test_missing_wrong_and_malformed_tokens_are_rejected(tmp_path):
    _run(tmp_path, """
        import server
        from fastapi.testclient import TestClient

        with TestClient(server.app) as c:
            c.post("/login", data={"username": "admin", "password": "ARealStrongPassword123"})
            token = c.post("/admin/api-tokens", data={"label": "bot", "username": "admin"}).json()["token"]

            assert c.get("/api/v1/meeting-protocol/x").status_code == 401, "no Authorization header"
            assert c.get("/api/v1/meeting-protocol/x", headers={"Authorization": "Bearer nonsense"}).status_code == 401
            tampered = token[:-1] + ("a" if token[-1] != "a" else "b")
            assert c.get("/api/v1/meeting-protocol/x", headers={"Authorization": f"Bearer {tampered}"}).status_code == 401
        print("OK")
    """)


def test_revoked_token_is_rejected(tmp_path):
    _run(tmp_path, """
        import server
        from fastapi.testclient import TestClient

        with TestClient(server.app) as c:
            c.post("/login", data={"username": "admin", "password": "ARealStrongPassword123"})
            r = c.post("/admin/api-tokens", data={"label": "bot", "username": "admin"})
            token, token_id = r.json()["token"], r.json()["id"]

            ok = c.get("/api/v1/meeting-protocol/x", headers={"Authorization": f"Bearer {token}"})
            assert ok.status_code == 404  # authenticated, unknown job

            rev = c.post(f"/admin/api-tokens/{token_id}/revoke")
            assert rev.status_code == 200
            assert c.post(f"/admin/api-tokens/{token_id}/revoke").status_code == 404, "revoking twice is a clean 404"

            after = c.get("/api/v1/meeting-protocol/x", headers={"Authorization": f"Bearer {token}"})
            assert after.status_code == 401
        print("OK")
    """)


def test_v1_upload_creates_job_owned_by_the_tokens_username(tmp_path):
    _run(tmp_path, """
        import server
        from fastapi.testclient import TestClient

        server.schedule_job = lambda job_id: None

        with TestClient(server.app) as c:
            c.post("/login", data={"username": "admin", "password": "ARealStrongPassword123"})
            c.post("/admin/users", data={"username": "carol", "password": "carolpassword"})
            token = c.post("/admin/api-tokens", data={"label": "bridge", "username": "carol"}).json()["token"]

            r2 = c.post("/api/v1/meeting-protocol", headers={"Authorization": f"Bearer {token}"},
                        files={"file": ("m.wav", b"fake-audio", "audio/wav")})
            assert r2.status_code == 200, r2.text
            job_id = r2.json()["job_id"]
            job = server.job_store.get(job_id)
            assert job["username"] == "carol"
            assert job["auto_protocol"] is True, "the v1 API always auto-triggers the protocol"
        print("OK")
    """)


def test_v1_status_is_scoped_to_the_tokens_own_jobs(tmp_path):
    _run(tmp_path, """
        import server
        from fastapi.testclient import TestClient

        server.schedule_job = lambda job_id: None

        with TestClient(server.app) as c:
            c.post("/login", data={"username": "admin", "password": "ARealStrongPassword123"})
            c.post("/admin/users", data={"username": "dave", "password": "davepassword"})
            c.post("/admin/users", data={"username": "erin", "password": "erinpassword"})
            dave_token = c.post("/admin/api-tokens", data={"label": "d", "username": "dave"}).json()["token"]
            erin_token = c.post("/admin/api-tokens", data={"label": "e", "username": "erin"}).json()["token"]

            r = c.post("/api/v1/meeting-protocol", headers={"Authorization": f"Bearer {dave_token}"},
                       files={"file": ("m.wav", b"fake-audio", "audio/wav")})
            job_id = r.json()["job_id"]

            own = c.get(f"/api/v1/meeting-protocol/{job_id}", headers={"Authorization": f"Bearer {dave_token}"})
            assert own.status_code == 200
            assert own.json()["job_id"] == job_id

            other = c.get(f"/api/v1/meeting-protocol/{job_id}", headers={"Authorization": f"Bearer {erin_token}"})
            assert other.status_code == 404, "a token must never see another token's/user's job"
        print("OK")
    """)


def test_v1_status_reports_transcription_and_protocol_state(tmp_path):
    _run(tmp_path, """
        import time
        import server
        from fastapi.testclient import TestClient

        with TestClient(server.app) as c:
            c.post("/login", data={"username": "admin", "password": "ARealStrongPassword123"})
            token = c.post("/admin/api-tokens", data={"label": "f", "username": "admin"}).json()["token"]

            server.job_store.create(id="v1-job-1", username="admin", filename="m.wav",
                                     original_path=None, log_path=None, settings_snapshot={}, auto_protocol=True)
            server.job_store.update("v1-job-1", status="running", started_at=time.time())
            server.job_store.update("v1-job-1", status="completed", finished_at=time.time(), progress=1,
                                     transcript_path="/nonexistent.txt")

            r = c.get("/api/v1/meeting-protocol/v1-job-1", headers={"Authorization": f"Bearer {token}"})
            assert r.status_code == 200
            body = r.json()
            assert body["status"] == "completed"
            assert body["protocol"]["enabled"] is True
            assert "protocol_data" not in body  # nothing completed yet
        print("OK")
    """)


def test_v1_routes_accept_no_settings_parameters(tmp_path):
    """Hard constraint: the external API cannot change model/engine/prompts/
    glossary. It has no parameters for any of that at all -- extra fields in
    the multipart body are simply ignored by FastAPI, they cannot reach
    load_settings()/the admin-only config routes."""
    _run(tmp_path, """
        import server
        from fastapi.testclient import TestClient

        server.schedule_job = lambda job_id: None

        with TestClient(server.app) as c:
            c.post("/login", data={"username": "admin", "password": "ARealStrongPassword123"})
            token = c.post("/admin/api-tokens", data={"label": "g", "username": "admin"}).json()["token"]

            r = c.post("/api/v1/meeting-protocol", headers={"Authorization": f"Bearer {token}"},
                       files={"file": ("m.wav", b"fake-audio", "audio/wav")},
                       data={"asr_model": "some-other-model", "diarization_model": "none", "device": "cpu"})
            assert r.status_code == 200, r.text
            job = server.job_store.get(r.json()["job_id"])
            assert job["settings_snapshot"].get("asr_model") != "some-other-model"

            assert c.post("/api/v1/models/select", headers={"Authorization": f"Bearer {token}"},
                          json={"diarization_model": "none"}).status_code == 404
        print("OK")
    """)


def test_v1_route_requires_protocol_module_enabled(tmp_path):
    """When the protocol module is disabled, the v1 API must 404 rather than
    accept uploads it can never act on."""
    env = _base_env(tmp_path)
    env["GIGASCRIBE_PROTOCOL_ENABLED"] = "0"
    result = _run_script("""
        import server
        from fastapi.testclient import TestClient

        assert server.PROTOCOL_ENABLED is False

        with TestClient(server.app) as c:
            c.post("/login", data={"username": "admin", "password": "ARealStrongPassword123"})
            token = c.post("/admin/api-tokens", data={"label": "h", "username": "admin"}).json()["token"]
            r = c.post("/api/v1/meeting-protocol", headers={"Authorization": f"Bearer {token}"},
                       files={"file": ("m.wav", b"fake-audio", "audio/wav")})
            assert r.status_code == 404
        print("OK")
    """, env_extra=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
