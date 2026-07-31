"""Crash-safe SQLite job queue.

The database is deliberately the only queue: workers atomically claim rows with
``BEGIN IMMEDIATE`` rather than relying on an in-memory scheduler.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

TERMINAL = {"completed", "failed", "cancelled"}
VALID = {"queued", "running", *TERMINAL}
TRANSITIONS = {
    "queued": {"running", "cancelled"}, "running": TERMINAL,
    "failed": {"queued"}, "cancelled": {"queued"}, "completed": set(),
}
JSON_COLUMNS = {"settings_snapshot", "actual_models", "requested_models"}
MUTABLE_COLUMNS = {
    "status", "progress", "message", "started_at", "finished_at", "error",
    "settings_snapshot", "actual_device", "actual_models", "requested_models",
    "requested_device", "transcript_path", "log_path", "original_path", "wav_path",
    "m4a_path", "flac_path",
    "timeout_seconds", "attempts", "cancel_requested", "correlation_id",
}


class JobStore:
    def __init__(self, path: Path, *, retry_limit: int = 3):
        self.path, self.retry_limit = Path(path), retry_limit
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=30000")
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def _migrate(self) -> None:
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
            if not db.execute("SELECT 1 FROM schema_version").fetchone():
                db.execute("INSERT INTO schema_version VALUES (0)")
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY, username TEXT NOT NULL, filename TEXT NOT NULL,
              status TEXT NOT NULL CHECK(status IN ('queued','running','completed','failed','cancelled')),
              progress REAL NOT NULL DEFAULT 0, message TEXT, created_at REAL NOT NULL,
              started_at REAL, finished_at REAL, error TEXT, settings_snapshot TEXT NOT NULL,
              actual_device TEXT, actual_models TEXT, transcript_path TEXT, log_path TEXT,
              original_path TEXT, wav_path TEXT, m4a_path TEXT, flac_path TEXT,
              timeout_seconds INTEGER, attempts INTEGER NOT NULL DEFAULT 0,
              cancel_requested INTEGER NOT NULL DEFAULT 0, correlation_id TEXT,
              requested_models TEXT, requested_device TEXT)""")
            existing = {r[1] for r in db.execute("PRAGMA table_info(jobs)")}
            # Migration from the first persistent-queue release, plus the
            # M4A/FLAC download columns added later.
            for col, typ in (("correlation_id", "TEXT"), ("requested_models", "TEXT"), ("requested_device", "TEXT"), ("m4a_path", "TEXT"), ("flac_path", "TEXT")):
                if col not in existing: db.execute(f"ALTER TABLE jobs ADD COLUMN {col} {typ}")
            db.execute("""CREATE TABLE IF NOT EXISTS job_attempts (
              id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL REFERENCES jobs(id),
              attempt INTEGER NOT NULL, status TEXT NOT NULL, started_at REAL, finished_at REAL,
              error TEXT, transcript_path TEXT, wav_path TEXT, m4a_path TEXT, flac_path TEXT, log_path TEXT)""")
            existing_attempts = {r[1] for r in db.execute("PRAGMA table_info(job_attempts)")}
            for col, typ in (("m4a_path", "TEXT"), ("flac_path", "TEXT")):
                if col not in existing_attempts: db.execute(f"ALTER TABLE job_attempts ADD COLUMN {col} {typ}")
            db.execute("CREATE INDEX IF NOT EXISTS jobs_user_created ON jobs(username, created_at DESC)")
            db.execute("CREATE INDEX IF NOT EXISTS jobs_claim ON jobs(status, created_at)")
            db.execute("UPDATE schema_version SET version=3")

    def _tx(self):
        db = self._connect(); db.execute("BEGIN IMMEDIATE"); return db

    def recover(self) -> None:
        now = time.time(); db = self._tx()
        try:
            db.execute("UPDATE jobs SET status='queued', message='Восстановлено после перезапуска', started_at=NULL WHERE status='running' AND cancel_requested=0")
            db.execute("UPDATE jobs SET status='cancelled', finished_at=?, message='Отменено при перезапуске' WHERE status='running' AND cancel_requested=1", (now,))
            db.commit()
        except Exception: db.rollback(); raise
        finally: db.close()

    def create(self, **j: Any) -> dict[str, Any]:
        snapshot = j.get("settings_snapshot") or {}
        values = {"id": j["id"], "username": j["username"], "filename": j["filename"], "status": j.get("status") or "queued", "progress": j.get("progress", 0), "message": j.get("message"), "created_at": j.get("created_at") or time.time(), "settings_snapshot": json.dumps(snapshot), "original_path": j.get("original_path"), "log_path": j.get("log_path"), "timeout_seconds": j.get("timeout_seconds"), "correlation_id": j.get("correlation_id", j["id"]), "requested_models": json.dumps(j.get("requested_models") or {k: snapshot.get(k) for k in ("asr_model", "diarization_model")}), "requested_device": j.get("requested_device", snapshot.get("device"))}
        cols = list(values); db = self._connect()
        try: db.execute(f"INSERT INTO jobs ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)})", [values[c] for c in cols])
        finally: db.close()
        return self.get(j["id"])

    def get(self, job_id: str):
        with self._connect() as db: return self._row(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def list(self, username: str | None = None, active_only: bool = False):
        clauses, args = [], []
        if username: clauses.append("username=?"); args.append(username)
        if active_only: clauses.append("status IN ('queued','running')")
        q = "SELECT * FROM jobs" + (" WHERE " + " AND ".join(clauses) if clauses else "") + " ORDER BY created_at DESC"
        with self._connect() as db: return [self._row(r) for r in db.execute(q, args)]

    def claim_next(self) -> dict[str, Any] | None:
        """Claim exactly one queued job. Safe across processes sharing the DB."""
        db = self._tx()
        try:
            row = db.execute("SELECT id FROM jobs WHERE status='queued' AND cancel_requested=0 ORDER BY created_at LIMIT 1").fetchone()
            if not row: db.commit(); return None
            now = time.time()
            changed = db.execute("UPDATE jobs SET status='running', started_at=?, attempts=attempts+1, message='Запущено' WHERE id=? AND status='queued' AND cancel_requested=0", (now, row["id"])).rowcount
            if not changed: db.commit(); return None
            job = db.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone()
            db.execute("INSERT INTO job_attempts(job_id,attempt,status,started_at,log_path) VALUES(?,?,?,?,?)", (row["id"], job["attempts"], "running", now, job["log_path"]))
            db.commit(); return self._row(job)
        except Exception: db.rollback(); raise
        finally: db.close()

    def claim(self, job_id: str) -> dict[str, Any] | None:
        """Atomically claim a particular scheduled job (used by the web worker)."""
        db = self._tx()
        try:
            now = time.time()
            changed = db.execute("UPDATE jobs SET status='running', started_at=?, attempts=attempts+1, message='Запущено' WHERE id=? AND status='queued' AND cancel_requested=0", (now, job_id)).rowcount
            if not changed: db.commit(); return None
            job = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            db.execute("INSERT INTO job_attempts(job_id,attempt,status,started_at,log_path) VALUES(?,?,?,?,?)", (job_id, job["attempts"], "running", now, job["log_path"]))
            db.commit(); return self._row(job)
        except Exception: db.rollback(); raise
        finally: db.close()

    def update(self, job_id: str, **changes: Any):
        if not changes: return self.get(job_id)
        unknown = set(changes) - MUTABLE_COLUMNS
        if unknown: raise ValueError(f"Unsupported job columns: {sorted(unknown)}")
        db = self._tx()
        try:
            old = db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not old: db.commit(); return None
            if "status" in changes and changes["status"] != old["status"] and changes["status"] not in TRANSITIONS[old["status"]]:
                raise ValueError(f"Invalid status transition {old['status']} -> {changes['status']}")
            for k in JSON_COLUMNS & changes.keys(): changes[k] = json.dumps(changes[k])
            sql = ", ".join(f"{k}=?" for k in changes)
            db.execute(f"UPDATE jobs SET {sql} WHERE id=?", [*changes.values(), job_id])
            if changes.get("status") in TERMINAL:
                db.execute("UPDATE job_attempts SET status=?, finished_at=?, error=?, transcript_path=?, wav_path=?, m4a_path=?, flac_path=? WHERE job_id=? AND finished_at IS NULL", (changes["status"], changes.get("finished_at", time.time()), changes.get("error"), changes.get("transcript_path"), changes.get("wav_path"), changes.get("m4a_path"), changes.get("flac_path"), job_id))
            db.commit()
        except Exception: db.rollback(); raise
        finally: db.close()
        return self.get(job_id)

    def request_cancel(self, job_id: str): return self.update(job_id, cancel_requested=1, message="Отмена запрошена")

    def retry(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        if not job or job["status"] not in {"failed", "cancelled"}: raise ValueError("Only failed or cancelled jobs can be retried")
        # ``retry_limit`` counts retries in addition to the original attempt.
        if job["attempts"] > self.retry_limit: raise ValueError("Retry limit exceeded")
        return self.update(job_id, status="queued", progress=0, message="В очереди", error=None, started_at=None, finished_at=None, cancel_requested=0, transcript_path=None, wav_path=None, m4a_path=None, flac_path=None, actual_device=None, actual_models={})

    def attempts(self, job_id: str):
        with self._connect() as db: return [dict(r) for r in db.execute("SELECT * FROM job_attempts WHERE job_id=? ORDER BY attempt", (job_id,))]

    def list_deletable_originals(self, cutoff: float) -> list[dict[str, Any]]:
        """Jobs with a raw-audio artefact (original upload and/or the internal
        normalized WAV) old enough to delete, that are not queued/running."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM jobs WHERE (original_path IS NOT NULL OR wav_path IS NOT NULL) AND created_at < ? AND status NOT IN ('queued','running')",
                (cutoff,),
            )
            return [self._row(r) for r in rows]

    @staticmethod
    def _row(row):
        if not row: return None
        d = dict(row)
        for k in JSON_COLUMNS:
            d[k] = json.loads(d[k]) if d.get(k) else {}
        d["cancel_requested"] = bool(d["cancel_requested"])
        return d
