"""WAV removal, M4A/FLAC exports, newest-first history, and original-file retention."""
import json
import os
import subprocess
import tempfile
import time
import wave
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="gigascribe-test-dl-")
os.environ.setdefault("GIGASCRIBE_DATA_DIR", os.path.join(_TMP, "data"))
os.environ.setdefault("GIGASCRIBE_MODELS_DIR", os.path.join(_TMP, "models"))
os.environ.setdefault("GIGASCRIBE_ADMIN_PASSWORD", "SuperSecretAdminPass123")
os.environ.setdefault("GIGASCRIBE_SECRET_KEY", "test-secret-key")

import pytest
from fastapi.testclient import TestClient

import job_store as job_store_module
import server

# server.py loads app.py itself via importlib under a private module name
# (not the standard "app" import), so we reuse that instance instead of
# doing our own `import app`: a plain top-level `import app` here would
# get cached in sys.modules and make test_cuda_only_architecture.py's
# monkeypatched, stubbed `import app` calls silently return this real,
# unstubbed module instead for the rest of the test session.
giga_app = server.giga_app


@pytest.fixture()
def client():
    with TestClient(server.app) as c:
        yield c


def _login(client, username, password):
    return client.post("/login", data={"username": username, "password": password}, follow_redirects=False)


def _admin(client):
    _login(client, "admin", "SuperSecretAdminPass123")
    return client


def _make_job(job_id: str, username: str = "admin", *, status: str = "completed",
              created_at: float | None = None, original_exists: bool = True,
              extra: dict | None = None) -> dict:
    original_path = server.BASE_DIR / "uploads" / f"{job_id}-original.wav"
    original_path.parent.mkdir(parents=True, exist_ok=True)
    if original_exists:
        original_path.write_bytes(b"fake-audio")
    log_path = server.BASE_DIR / "results" / job_id / "job.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    job = server.job_store.create(
        id=job_id, username=username, filename=f"{job_id}.wav",
        original_path=str(original_path), log_path=str(log_path),
        settings_snapshot={"asr_model": "gigaam-v3-e2e-rnnt", "diarization_model": "none", "device": "cuda"},
        message="Готово", created_at=created_at,
    )
    if status != "queued":
        server.job_store.update(job_id, status="running", started_at=time.time())
        server.job_store.update(job_id, status=status, finished_at=time.time(), progress=1)
    if extra:
        server.job_store.update(job_id, **extra)
    return server.job_store.get(job_id)


def _ffprobe_stream(path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)["streams"][0]


def _write_test_wav(path: str, seconds: float = 1.0, channels: int = 2, rate: int = 44100) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * int(rate * seconds) * channels)


# 1. Newest jobs are returned (and would render) first.
def test_newest_jobs_listed_first():
    now = time.time()
    _make_job("order-old", created_at=now - 100)
    _make_job("order-new", created_at=now)
    ids = [j["id"] for j in server.job_store.list()]
    assert ids.index("order-new") < ids.index("order-old")


def test_index_html_does_not_reverse_job_list():
    assert "js.reverse()" not in server.INDEX_HTML
    assert "js.map(renderJob)" in server.INDEX_HTML


# 2. Queue is still processed oldest-first.
def test_claim_next_returns_oldest_queued_job_first(tmp_path):
    store = job_store_module.JobStore(tmp_path / "queue.sqlite3")
    now = time.time()
    store.create(id="q-new", username="u", filename="a.wav", settings_snapshot={}, created_at=now)
    store.create(id="q-old", username="u", filename="b.wav", settings_snapshot={}, created_at=now - 50)
    claimed = store.claim_next()
    assert claimed["id"] == "q-old"


# 3. WAV is absent from the API and from the rendered download labels.
def test_wav_absent_from_serialized_downloads():
    job = _make_job("no-wav-1", extra={"wav_path": str(server.BASE_DIR / "results" / "no-wav-1" / "normalized.wav")})
    serialized = server.serialize_job(job)
    assert "wav" not in serialized["downloads"]
    assert set(serialized["downloads"]) == {"transcript", "log", "original", "m4a", "flac"}


def test_wav_not_offered_in_ui_labels():
    labels_snippet = server.INDEX_HTML.split("DOWNLOAD_LABELS=")[1].split(";")[0]
    assert "wav" not in labels_snippet
    assert "m4a" in labels_snippet and "flac" in labels_snippet


# 4. A direct request for kind=wav is rejected even if wav_path is set.
def test_direct_wav_download_rejected(client):
    _admin(client)
    wav_path = server.BASE_DIR / "results" / "wav-direct" / "normalized.wav"
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    wav_path.write_bytes(b"RIFF....WAVEfmt ")
    _make_job("wav-direct", extra={"wav_path": str(wav_path)})
    r = client.get("/api/jobs/wav-direct/download/wav")
    assert r.status_code == 404


def test_unknown_kind_rejected(client):
    _admin(client)
    _make_job("unknown-kind-1")
    r = client.get("/api/jobs/unknown-kind-1/download/nope")
    assert r.status_code == 404


# 5 & 6. After processing, M4A/FLAC exist with the specified ffmpeg parameters.
def test_create_m4a_and_flac_produce_expected_streams(tmp_path):
    src = str(tmp_path / "source.wav")
    _write_test_wav(src)
    m4a_out, flac_out = str(tmp_path / "out.m4a"), str(tmp_path / "out.flac")

    class _Owner:  # create_m4a/create_flac don't touch self, so a bare object is enough.
        pass

    giga_app.AudioProcessor.create_m4a(_Owner(), src, m4a_out)
    giga_app.AudioProcessor.create_flac(_Owner(), src, flac_out)

    assert os.path.getsize(m4a_out) > 0
    assert os.path.getsize(flac_out) > 0

    m4a_stream = _ffprobe_stream(m4a_out)
    assert m4a_stream["codec_name"] == "aac"
    assert m4a_stream["sample_rate"] == "48000"
    assert m4a_stream["channels"] == 1

    flac_stream = _ffprobe_stream(flac_out)
    assert flac_stream["codec_name"] == "flac"
    assert flac_stream["sample_rate"] == "16000"
    assert flac_stream["channels"] == 1


def test_create_m4a_raises_clear_error_on_missing_input(tmp_path):
    class _Owner:
        pass
    with pytest.raises(FileNotFoundError):
        giga_app.AudioProcessor.create_m4a(_Owner(), str(tmp_path / "missing.wav"), str(tmp_path / "out.m4a"))


def test_m4a_and_flac_downloadable_once_job_completed(client):
    _admin(client)
    result_dir = server.BASE_DIR / "results" / "m4a-flac-ready"
    result_dir.mkdir(parents=True, exist_ok=True)
    m4a_path, flac_path = result_dir / "clip.m4a", result_dir / "clip.flac"
    m4a_path.write_bytes(b"m4a-bytes"); flac_path.write_bytes(b"flac-bytes")
    _make_job("m4a-flac-ready", extra={"m4a_path": str(m4a_path), "flac_path": str(flac_path)})

    listed = client.get("/api/jobs").json()
    job = next(j for j in listed if j["id"] == "m4a-flac-ready")
    assert job["downloads"]["m4a"] is True and job["downloads"]["flac"] is True

    assert client.get("/api/jobs/m4a-flac-ready/download/m4a").content == b"m4a-bytes"
    assert client.get("/api/jobs/m4a-flac-ready/download/flac").content == b"flac-bytes"


# 7-11. Original retention cleanup.
def test_recent_original_is_not_deleted():
    job = _make_job("retain-recent", created_at=time.time())
    deleted = server.cleanup_expired_originals()
    assert Path(job["original_path"]).exists()
    refreshed = server.job_store.get("retain-recent")
    assert refreshed["original_path"] == job["original_path"]


def test_old_original_is_deleted():
    old_ts = time.time() - (server.ORIGINAL_RETENTION_DAYS + 1) * 86400
    job = _make_job("expire-old", created_at=old_ts)
    assert Path(job["original_path"]).exists()
    server.cleanup_expired_originals()
    assert not Path(job["original_path"]).exists()


def test_active_job_older_than_retention_is_not_deleted():
    old_ts = time.time() - (server.ORIGINAL_RETENTION_DAYS + 1) * 86400
    job = _make_job("expire-but-running", status="queued", created_at=old_ts)
    server.cleanup_expired_originals()
    assert Path(job["original_path"]).exists()
    assert server.job_store.get("expire-but-running")["original_path"] == job["original_path"]


def test_cleanup_keeps_transcript_log_m4a_flac():
    old_ts = time.time() - (server.ORIGINAL_RETENTION_DAYS + 1) * 86400
    result_dir = server.BASE_DIR / "results" / "expire-keep-artifacts"
    result_dir.mkdir(parents=True, exist_ok=True)
    transcript, m4a, flac = result_dir / "t.txt", result_dir / "t.m4a", result_dir / "t.flac"
    for p in (transcript, m4a, flac): p.write_bytes(b"x")
    job = _make_job("expire-keep-artifacts", created_at=old_ts,
                     extra={"transcript_path": str(transcript), "m4a_path": str(m4a), "flac_path": str(flac)})
    server.cleanup_expired_originals()
    assert transcript.exists() and m4a.exists() and flac.exists()
    refreshed = server.job_store.get("expire-keep-artifacts")
    assert refreshed["status"] != "queued"  # job row itself untouched/still present
    assert refreshed["transcript_path"] == str(transcript)


def test_original_path_becomes_null_after_cleanup():
    old_ts = time.time() - (server.ORIGINAL_RETENTION_DAYS + 1) * 86400
    _make_job("expire-null-path", created_at=old_ts)
    server.cleanup_expired_originals()
    assert server.job_store.get("expire-null-path")["original_path"] is None


def test_cleanup_ignores_already_missing_file():
    old_ts = time.time() - (server.ORIGINAL_RETENTION_DAYS + 1) * 86400
    job = _make_job("expire-already-gone", created_at=old_ts, original_exists=False)
    Path(job["original_path"]).unlink(missing_ok=True)
    # Should not raise even though the file is already gone.
    server.cleanup_expired_originals()
    assert server.job_store.get("expire-already-gone")["original_path"] is None


# 12. Retry without an original returns a clear error.
def test_retry_without_original_returns_410(client):
    _admin(client)
    _make_job("retry-no-original", status="failed", original_exists=False)
    r = client.post("/api/jobs/retry-no-original/retry")
    assert r.status_code == 410


def test_retry_after_retention_cleanup_returns_410(client):
    _admin(client)
    old_ts = time.time() - (server.ORIGINAL_RETENTION_DAYS + 1) * 86400
    _make_job("retry-expired-original", status="failed", created_at=old_ts)
    server.cleanup_expired_originals()
    r = client.post("/api/jobs/retry-expired-original/retry")
    assert r.status_code == 410


# 13. Migration works against a pre-existing (older-schema) SQLite database.
def test_migration_adds_m4a_flac_columns_to_existing_db(tmp_path):
    import sqlite3
    db_path = tmp_path / "legacy.sqlite3"
    raw = sqlite3.connect(db_path)
    raw.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    raw.execute("INSERT INTO schema_version VALUES (2)")
    raw.execute("""CREATE TABLE jobs (
      id TEXT PRIMARY KEY, username TEXT NOT NULL, filename TEXT NOT NULL,
      status TEXT NOT NULL CHECK(status IN ('queued','running','completed','failed','cancelled')),
      progress REAL NOT NULL DEFAULT 0, message TEXT, created_at REAL NOT NULL,
      started_at REAL, finished_at REAL, error TEXT, settings_snapshot TEXT NOT NULL,
      actual_device TEXT, actual_models TEXT, transcript_path TEXT, log_path TEXT,
      original_path TEXT, wav_path TEXT, timeout_seconds INTEGER, attempts INTEGER NOT NULL DEFAULT 0,
      cancel_requested INTEGER NOT NULL DEFAULT 0, correlation_id TEXT,
      requested_models TEXT, requested_device TEXT)""")
    raw.execute("""CREATE TABLE job_attempts (
      id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL REFERENCES jobs(id),
      attempt INTEGER NOT NULL, status TEXT NOT NULL, started_at REAL, finished_at REAL,
      error TEXT, transcript_path TEXT, wav_path TEXT, log_path TEXT)""")
    raw.execute(
        "INSERT INTO jobs (id,username,filename,status,created_at,settings_snapshot,original_path,wav_path) "
        "VALUES ('legacy-1','alice','f.wav','completed',100.0,'{}','/tmp/orig.wav','/tmp/norm.wav')"
    )
    raw.commit(); raw.close()

    store = job_store_module.JobStore(db_path)
    job = store.get("legacy-1")
    assert job["id"] == "legacy-1"
    assert job["wav_path"] == "/tmp/norm.wav"
    assert job.get("m4a_path") is None and job.get("flac_path") is None
    store.update("legacy-1", m4a_path="/tmp/x.m4a", flac_path="/tmp/x.flac")
    assert store.get("legacy-1")["m4a_path"] == "/tmp/x.m4a"
