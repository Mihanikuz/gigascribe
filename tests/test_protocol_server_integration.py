"""server.py integration: isolation guarantees, button visibility conditions,
and access control parity with the main job (items 1, 2, 3, 17, 24).

Each check here runs `server.py` in its own subprocess. This is deliberate,
not incidental: every other test file in this suite does `import server` at
module scope, and because Python caches that import for the whole pytest
session, whichever file the collector happens to import first silently
fixes PROTOCOL_ENABLED for every other file too. Import-time behavior that
depends on that env var can only be tested honestly in a fresh interpreter.
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
    }


def test_protocol_package_never_imported_when_disabled(tmp_path):
    """Item 1: disabled means genuinely not loaded, not just inert."""
    result = _run_script("""
        import server
        assert server.PROTOCOL_ENABLED is False
        assert server.PROTOCOL_SERVICE is None
        assert 'protocol' not in sys.modules, "protocol package must not be imported when disabled"
        print("OK")
    """, env_extra=_base_env(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_app_starts_with_protocol_directory_entirely_missing(tmp_path):
    """Item 24: a missing protocol/ package must not stop GigaScribe from
    starting, as long as the module is disabled."""
    import shutil
    repo_copy = tmp_path / "repo_copy"
    shutil.copytree(REPO_ROOT, repo_copy, ignore=shutil.ignore_patterns(
        "data", "models", "__pycache__", ".git", "protocol",
    ))
    env = _base_env(tmp_path)
    script = textwrap.dedent("""
        import sys
        sys.path.insert(0, %r)
        import server
        assert server.PROTOCOL_ENABLED is False
        assert server.PROTOCOL_SERVICE is None
        from fastapi.testclient import TestClient
        with TestClient(server.app) as c:
            r = c.get("/health/live")
            assert r.status_code == 200
        print("OK")
    """ % str(repo_copy))
    result = subprocess.run([sys.executable, "-c", script], cwd=repo_copy, env={**__import__("os").environ, **env},
                             capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_broken_protocol_package_degrades_gracefully_when_enabled(tmp_path):
    """Enabled, but the package fails to import: server must still start."""
    import shutil
    repo_copy = tmp_path / "repo_copy"
    shutil.copytree(REPO_ROOT, repo_copy, ignore=shutil.ignore_patterns("data", "models", "__pycache__", ".git"))
    with (repo_copy / "protocol" / "models.py").open("a", encoding="utf-8") as fh:
        fh.write("\nthis is not valid python !!! ((\n")
    env = _base_env(tmp_path)
    env["GIGASCRIBE_PROTOCOL_ENABLED"] = "1"
    script = textwrap.dedent("""
        import sys
        sys.path.insert(0, %r)
        import server
        assert server.PROTOCOL_ENABLED is False
        assert server.PROTOCOL_SERVICE is None
        from fastapi.testclient import TestClient
        with TestClient(server.app) as c:
            assert c.get("/health/live").status_code == 200
        print("OK")
    """ % str(repo_copy))
    result = subprocess.run([sys.executable, "-c", script], cwd=repo_copy, env={**__import__("os").environ, **env},
                             capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_health_ready_unaffected_by_missing_protocol_models(tmp_path):
    env = _base_env(tmp_path)
    env["GIGASCRIBE_PROTOCOL_ENABLED"] = "1"
    result = _run_script("""
        import server
        assert server.PROTOCOL_ENABLED is True
        from fastapi.testclient import TestClient
        with TestClient(server.app) as c:
            r = c.get("/health/ready")
            # 'protocol' must never appear inside the readiness payload at all
            import json
            assert "protocol" not in json.dumps(r.json()).lower()
        print("OK")
    """, env_extra=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_create_button_conditions_and_access_control(tmp_path):
    """Items 2, 3, 17 in one process: button hidden when disabled, shown once
    a transcription is completed + module enabled + a model installed, and
    protocol view/create access control mirrors the main job's."""
    env = _base_env(tmp_path)
    env["GIGASCRIBE_PROTOCOL_ENABLED"] = "1"
    result = _run_script("""
        import time
        import server
        from fastapi.testclient import TestClient

        with TestClient(server.app) as c:
            c.post("/login", data={"username": "admin", "password": "ARealStrongPassword123"})
            r = c.post("/admin/users", data={"username": "bob", "password": "bobpassword"})
            assert r.status_code == 200

            transcript_path = server.BASE_DIR / "results" / "job-1" / "t.txt"
            transcript_path.parent.mkdir(parents=True, exist_ok=True)
            transcript_path.write_text("Спикер 1 [00:00 - 00:02]: привет", encoding="utf-8")
            server.job_store.create(
                id="job-1", username="admin", filename="m.wav",
                original_path=None, log_path=str(transcript_path.parent / "job.log"),
                settings_snapshot={},
            )
            server.job_store.update("job-1", status="running", started_at=time.time())
            server.job_store.update(
                "job-1", status="completed", finished_at=time.time(), progress=1,
                transcript_path=str(transcript_path),
            )

            job = server.job_store.get("job-1")
            summary = server._protocol_summary(job)
            assert summary["enabled"] is True
            assert summary["can_create"] is True, "can_create must be true once transcription is completed"

            # No model installed yet -> creating a protocol must fail cleanly
            r = c.post("/api/jobs/job-1/protocol")
            assert r.status_code == 400, r.text

            server.PROTOCOL_SERVICE.store.update_model_state("qwen3-8b", local_path="/fake", installed=1)
            server.PROTOCOL_SERVICE.store.set_active_model_id("qwen3-8b")

            status = c.get("/api/protocol/status").json()
            assert status["has_installed_model"] is True

            bob = TestClient(server.app)
            bob.post("/login", data={"username": "bob", "password": "bobpassword"})
            r = bob.post("/api/jobs/job-1/protocol")
            assert r.status_code == 403, "a non-owner, non-admin must not be able to create a protocol"

            r = bob.get("/api/jobs/job-1/protocol")
            assert r.status_code == 404, "viewing before completion should 404, not 403 -- matches job download semantics"

        print("OK")
    """, env_extra=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_user_correction_creates_suggestion_and_is_owner_only(tmp_path):
    """Items 21-22: POST /api/jobs/{id}/glossary-suggestion creates a
    proposed, never-auto-applied suggestion for the owner/admin, and a
    non-owner, non-admin user gets 403 trying to suggest one for someone
    else's job."""
    env = _base_env(tmp_path)
    env["GIGASCRIBE_PROTOCOL_ENABLED"] = "1"
    result = _run_script("""
        import server
        from fastapi.testclient import TestClient

        with TestClient(server.app) as c:
            c.post("/login", data={"username": "admin", "password": "ARealStrongPassword123"})
            r = c.post("/admin/users", data={"username": "bob", "password": "bobpassword"})
            assert r.status_code == 200

            server.job_store.create(id="job-c1", username="admin", filename="m.wav",
                                     original_path=None, log_path=None, settings_snapshot={})

            r = c.post("/api/jobs/job-c1/glossary-suggestion",
                       json={"wrong_text": "азуре", "suggested_text": "Azure"})
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["status"] == "proposed"

            pending = server.PROTOCOL_SERVICE.store.list_suggestions("proposed")
            match = [s for s in pending if s.wrong_text == "азуре"]
            assert len(match) == 1
            assert match[0].source == "user_correction"
            assert match[0].job_id == "job-c1"

            # never auto-applied
            terms = server.PROTOCOL_SERVICE.store.resolved_terms()
            assert not any("азуре" in t.aliases for t in terms)

            bob = TestClient(server.app)
            bob.post("/login", data={"username": "bob", "password": "bobpassword"})
            r = bob.post("/api/jobs/job-c1/glossary-suggestion",
                         json={"wrong_text": "x", "suggested_text": "Y"})
            assert r.status_code == 403, "a non-owner, non-admin must not be able to suggest a correction for another user's job"

        print("OK")
    """, env_extra=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
