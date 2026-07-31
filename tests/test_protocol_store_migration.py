"""ProtocolStore schema, migration against a pre-existing database (item 25)."""
import sqlite3
import tempfile
import time
from pathlib import Path

import job_store
from protocol.models import SUPPORTED_PROTOCOL_MODELS
from protocol.schemas import ModelSpec
from protocol.store import ProtocolStore


def test_migration_creates_tables_on_fresh_db(tmp_path):
    db_path = tmp_path / "jobs.sqlite3"
    job_store.JobStore(db_path)
    store = ProtocolStore(db_path)
    with sqlite3.connect(db_path) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for expected in ("protocol_jobs", "protocol_models", "protocol_prompts", "protocol_results",
                      "glossary_terms", "glossary_aliases", "glossary_suggestions", "glossary_corrections",
                      "protocol_settings"):
        assert expected in tables


def test_migration_is_safe_on_pre_existing_database_with_data(tmp_path):
    """Simulate a database that already has the main app's `jobs` table
    populated (pre-dating the protocol module) and confirm the migration
    adds protocol_* tables without touching existing rows."""
    db_path = tmp_path / "jobs.sqlite3"
    raw = sqlite3.connect(db_path)
    raw.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    raw.execute("INSERT INTO schema_version VALUES (3)")
    raw.execute("""CREATE TABLE jobs (
      id TEXT PRIMARY KEY, username TEXT NOT NULL, filename TEXT NOT NULL,
      status TEXT NOT NULL CHECK(status IN ('queued','running','completed','failed','cancelled')),
      progress REAL NOT NULL DEFAULT 0, message TEXT, created_at REAL NOT NULL,
      started_at REAL, finished_at REAL, error TEXT, settings_snapshot TEXT NOT NULL,
      actual_device TEXT, actual_models TEXT, transcript_path TEXT, log_path TEXT,
      original_path TEXT, wav_path TEXT, m4a_path TEXT, flac_path TEXT,
      timeout_seconds INTEGER, attempts INTEGER NOT NULL DEFAULT 0,
      cancel_requested INTEGER NOT NULL DEFAULT 0, correlation_id TEXT,
      requested_models TEXT, requested_device TEXT)""")
    raw.execute(
        "INSERT INTO jobs (id,username,filename,status,created_at,settings_snapshot,transcript_path) "
        "VALUES ('existing-job','bob','meeting.wav','completed',?,'{}','/data/results/existing-job/t.txt')",
        (time.time(),),
    )
    raw.commit()
    raw.close()

    store = ProtocolStore(db_path)  # migration must not raise or touch `jobs`
    with sqlite3.connect(db_path) as db:
        db.row_factory = sqlite3.Row
        row = db.execute("SELECT * FROM jobs WHERE id='existing-job'").fetchone()
    assert row["username"] == "bob"
    assert row["status"] == "completed"

    # and the new tables actually work, including the FK against the
    # pre-existing jobs row
    pj = store.create_protocol_job(id="p1", job_id="existing-job", username="bob", model_id="qwen3-8b")
    assert pj["job_id"] == "existing-job"


def test_migration_runs_twice_without_error(tmp_path):
    db_path = tmp_path / "jobs.sqlite3"
    job_store.JobStore(db_path)
    ProtocolStore(db_path)
    ProtocolStore(db_path)  # second instantiation re-runs _migrate(); must be idempotent


def test_foreign_key_rejects_protocol_job_for_unknown_job(tmp_path):
    db_path = tmp_path / "jobs.sqlite3"
    job_store.JobStore(db_path)
    store = ProtocolStore(db_path)
    try:
        store.create_protocol_job(id="orphan", job_id="does-not-exist", username="x", model_id="qwen3-8b")
        assert False, "expected a foreign key violation"
    except sqlite3.IntegrityError:
        pass


def test_seed_model_is_idempotent(tmp_path):
    db_path = tmp_path / "jobs.sqlite3"
    job_store.JobStore(db_path)
    store = ProtocolStore(db_path)
    spec = SUPPORTED_PROTOCOL_MODELS["qwen3-8b"]
    store.seed_model(spec)
    store.update_model_state("qwen3-8b", installed=1)
    store.seed_model(spec)  # must not overwrite the installed flag we just set
    assert store.get_model_state("qwen3-8b")["installed"] == 1
