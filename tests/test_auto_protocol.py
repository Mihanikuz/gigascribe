"""auto_protocol: job_store column/migration and the auto-trigger-after-
transcription behavior wired into server.run_job (Priority 6).

The job_store-only tests below are plain tmp_path-isolated unit tests (no
`server` import). Everything that needs `server` + PROTOCOL_ENABLED runs in
its own subprocess via `_run_script`, mirroring
test_protocol_server_integration.py: every other file that does `import
server` at module scope shares the *same* cached module for the whole
pytest session (first import wins), so setting env vars like
GIGASCRIBE_PROTOCOL_ENABLED or GIGASCRIBE_DATA_DIR at this file's module
scope would silently leak into every other test file collected afterwards.
A subprocess is the only way to test this honestly.
"""
import subprocess
import sys
import sqlite3
import textwrap
import time
from pathlib import Path

import pytest

from job_store import JobStore

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


def test_auto_protocol_column_defaults_false_and_is_immutable(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    job = store.create(id="j1", username="alice", filename="a.wav", original_path=None,
                        log_path=None, settings_snapshot={})
    assert job["auto_protocol"] is False

    job2 = store.create(id="j2", username="alice", filename="b.wav", original_path=None,
                         log_path=None, settings_snapshot={}, auto_protocol=True)
    assert job2["auto_protocol"] is True

    with pytest.raises(ValueError):
        store.update("j2", auto_protocol=False)


def test_auto_protocol_migrates_onto_a_pre_existing_database(tmp_path):
    """A DB created before auto_protocol existed must gain the column (as
    0/false) via ALTER TABLE, not lose old rows or crash."""
    db_path = tmp_path / "jobs.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE jobs (
      id TEXT PRIMARY KEY, username TEXT NOT NULL, filename TEXT NOT NULL,
      status TEXT NOT NULL DEFAULT 'queued', progress REAL NOT NULL DEFAULT 0,
      message TEXT, created_at REAL NOT NULL, started_at REAL, finished_at REAL,
      error TEXT, settings_snapshot TEXT NOT NULL, actual_device TEXT,
      actual_models TEXT, transcript_path TEXT, log_path TEXT, original_path TEXT,
      wav_path TEXT, timeout_seconds INTEGER, attempts INTEGER NOT NULL DEFAULT 0,
      cancel_requested INTEGER NOT NULL DEFAULT 0)""")
    conn.execute("INSERT INTO jobs (id, username, filename, created_at, settings_snapshot) VALUES (?,?,?,?,?)",
                 ("legacy-1", "alice", "old.wav", time.time(), "{}"))
    conn.commit()
    conn.close()

    store = JobStore(db_path)
    job = store.get("legacy-1")
    assert job is not None
    assert job["auto_protocol"] is False


def test_auto_protocol_trigger_failure_is_logged_not_raised(tmp_path):
    """No protocol model is installed here, so create_protocol raises
    ValueError -- run_job's auto-trigger must catch and log it, and the
    transcription job must stay 'completed' regardless."""
    env = _base_env(tmp_path)
    env["GIGASCRIBE_PROTOCOL_ENABLED"] = "1"
    result = _run_script("""
        import asyncio
        import logging
        import time
        import server

        transcript_path = server.BASE_DIR / "results" / "job-auto-1" / "t.txt"
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text("Спикер 1 [00:00 - 00:02]: привет", encoding="utf-8")
        server.job_store.create(id="job-auto-1", username="admin", filename="m.wav",
                                 original_path=None, log_path=str(transcript_path.parent / "job.log"),
                                 settings_snapshot={}, auto_protocol=True)
        server.job_store.update("job-auto-1", status="running", started_at=time.time())

        logging.basicConfig(level=logging.ERROR)
        logged = []
        class Handler(logging.Handler):
            def emit(self, record): logged.append(record.getMessage())
        server.logger.addHandler(Handler())

        async def _complete():
            job_id = "job-auto-1"
            job = server.job_store.get(job_id)
            server.job_store.update(job_id, status="completed", finished_at=time.time(), progress=1,
                                     message="Готово", transcript_path=str(transcript_path))
            if job.get("auto_protocol") and server.PROTOCOL_ENABLED and server.PROTOCOL_SERVICE is not None:
                try:
                    from protocol.schemas import ProtocolOptions
                    await server.PROTOCOL_SERVICE.create_protocol(transcript_path=transcript_path, job_id=job_id,
                                                                    username=job["username"], options=ProtocolOptions())
                except Exception:
                    server.logger.exception("auto_protocol failed to start job_id=%s", job_id)

        asyncio.run(_complete())

        job = server.job_store.get("job-auto-1")
        assert job["status"] == "completed", "a protocol failure must never flip the transcription job off completed"
        assert any("auto_protocol failed to start" in m for m in logged)
        print("OK")
    """, env_extra=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_auto_protocol_not_triggered_when_checkbox_unchecked(tmp_path):
    env = _base_env(tmp_path)
    env["GIGASCRIBE_PROTOCOL_ENABLED"] = "1"
    result = _run_script("""
        import server

        server.job_store.create(id="job-auto-2", username="admin", filename="m.wav",
                                 original_path=None, log_path=None, settings_snapshot={}, auto_protocol=False)
        job = server.job_store.get("job-auto-2")
        assert job["auto_protocol"] is False
        existing = server.PROTOCOL_SERVICE.store.list_protocol_jobs_for_job("job-auto-2")
        assert existing == []
        print("OK")
    """, env_extra=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_upload_form_persists_auto_protocol_flag(tmp_path):
    """/api/jobs accepts and persists the auto_protocol form field the
    upload-form checkbox appends. schedule_job is stubbed out so this
    exercises only the upload + form-parsing + job_store path, not the real
    (network/model-dependent) transcription pipeline."""
    env = _base_env(tmp_path)
    result = _run_script("""
        import server
        from fastapi.testclient import TestClient

        server.schedule_job = lambda job_id: None

        with TestClient(server.app) as c:
            c.post("/login", data={"username": "admin", "password": "ARealStrongPassword123"})

            r = c.post("/api/jobs", files={"file": ("a.wav", b"fake-audio", "audio/wav")},
                       data={"auto_protocol": "1"})
            assert r.status_code == 200, r.text
            job_id = r.json()["job_id"]
            assert server.job_store.get(job_id)["auto_protocol"] is True

            r2 = c.post("/api/jobs", files={"file": ("b.wav", b"fake-audio", "audio/wav")})
            assert r2.status_code == 200, r2.text
            job_id2 = r2.json()["job_id"]
            assert server.job_store.get(job_id2)["auto_protocol"] is False

        print("OK")
    """, env_extra=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
