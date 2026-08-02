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


def test_protocol_jobs_rebuild_migration_preserves_rows_and_adds_new_shape(tmp_path):
    """Follow-up spec item 1/2: protocol_jobs predating settings_snapshot and
    the 'fact_checking' status (as it existed right after the module's first
    release) must upgrade in place, in a single transaction, without losing
    any existing protocol job rows -- SQLite can't ALTER a CHECK constraint
    or add a NOT NULL column without a default to a table that might already
    have rows, so ProtocolStore rebuilds the table instead."""
    db_path = tmp_path / "jobs.sqlite3"
    job_store.JobStore(db_path)
    raw = sqlite3.connect(db_path)
    raw.execute(
        "INSERT INTO jobs (id,username,filename,status,created_at,settings_snapshot,transcript_path) "
        "VALUES ('job-old','carol','m.wav','completed',?,'{}','/data/results/job-old/t.txt')",
        (time.time(),),
    )
    old_status_check = "CHECK(status IN ('queued','waiting_for_gpu','unloading_asr','loading_llm',"\
        "'splitting','processing_chunks','merging','rendering','completed','failed','cancelled'))"
    raw.execute(f"""CREATE TABLE protocol_jobs (
      id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id), username TEXT NOT NULL,
      status TEXT NOT NULL {old_status_check}, progress REAL NOT NULL DEFAULT 0, message TEXT,
      model_id TEXT, chunk_total INTEGER, chunk_current INTEGER, error TEXT, created_at REAL NOT NULL,
      started_at REAL, finished_at REAL, attempts INTEGER NOT NULL DEFAULT 0,
      cancel_requested INTEGER NOT NULL DEFAULT 0, json_path TEXT, html_path TEXT, log_path TEXT
    )""")
    raw.execute(
        "INSERT INTO protocol_jobs (id,job_id,username,status,progress,message,created_at) "
        "VALUES ('pj-old','job-old','carol','completed',1.0,'Готово',?)", (time.time(),),
    )
    raw.commit()
    raw.close()

    store = ProtocolStore(db_path)  # triggers the rebuild

    with sqlite3.connect(db_path) as db:
        db.row_factory = sqlite3.Row
        row = db.execute("SELECT * FROM protocol_jobs WHERE id='pj-old'").fetchone()
    assert row is not None, "existing protocol job row must survive the rebuild"
    assert row["status"] == "completed"
    assert row["username"] == "carol"
    assert row["settings_snapshot"] == "{}", "old rows get the column's default, not NULL"

    # the new shape actually accepts what the old one couldn't
    job = store.get_protocol_job("pj-old")
    assert job["settings_snapshot"] == {}
    store.update_protocol_job("pj-old", status="fact_checking")
    assert store.get_protocol_job("pj-old")["status"] == "fact_checking"

    new_id = store.create_protocol_job(id="pj-new", job_id="job-old", username="carol", model_id="qwen3-8b",
                                        settings_snapshot={"model_id": "qwen3-8b"})
    assert new_id["settings_snapshot"] == {"model_id": "qwen3-8b"}


def test_protocol_jobs_migration_is_idempotent_once_rebuilt(tmp_path):
    db_path = tmp_path / "jobs.sqlite3"
    job_store.JobStore(db_path)
    ProtocolStore(db_path)  # fresh DB: already the new shape
    ProtocolStore(db_path)  # second run must not attempt (or need) a rebuild
    with sqlite3.connect(db_path) as db:
        cols = {r[1] for r in db.execute("PRAGMA table_info(protocol_jobs)")}
    assert "settings_snapshot" in cols
