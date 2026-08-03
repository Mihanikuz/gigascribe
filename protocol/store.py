"""SQLite persistence for the protocol module.

Deliberately opens the *same* database file the main app's JobStore uses
(passed in as a plain Path by the integration layer -- this module never
imports job_store.py) so that ``FOREIGN KEY (job_id) REFERENCES jobs(id)``
is a real, enforced constraint. Every table this module owns is prefixed
``protocol_`` or ``glossary_`` and migration only ever adds to them;
main-app tables (``jobs``, ``job_attempts``) are never touched here.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from .prompts import DEFAULT_PROMPTS
from .schemas import (
    GLOSSARY_SCOPE_PRIORITY, GlossaryCorrection, GlossarySuggestion, GlossaryTerm,
    ModelSpec, PromptSpec,
)

PROTOCOL_TERMINAL = {"completed", "failed", "cancelled"}
PROTOCOL_ACTIVE = {
    "queued", "waiting_for_gpu", "unloading_asr", "loading_llm",
    "splitting", "processing_chunks", "merging", "fact_checking", "rendering",
}
PROTOCOL_STATUSES = PROTOCOL_ACTIVE | PROTOCOL_TERMINAL


class ProtocolStore:
    def __init__(self, db_path: Path):
        self.path = Path(db_path)
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
            status_check = "CHECK(status IN (" + ",".join(f"'{s}'" for s in sorted(PROTOCOL_STATUSES)) + "))"
            db.execute(f"""CREATE TABLE IF NOT EXISTS protocol_jobs (
              id TEXT PRIMARY KEY,
              job_id TEXT NOT NULL REFERENCES jobs(id),
              username TEXT NOT NULL,
              status TEXT NOT NULL {status_check},
              progress REAL NOT NULL DEFAULT 0,
              message TEXT,
              model_id TEXT,
              chunk_total INTEGER,
              chunk_current INTEGER,
              error TEXT,
              created_at REAL NOT NULL,
              started_at REAL,
              finished_at REAL,
              attempts INTEGER NOT NULL DEFAULT 0,
              cancel_requested INTEGER NOT NULL DEFAULT 0,
              json_path TEXT,
              html_path TEXT,
              log_path TEXT,
              settings_snapshot TEXT NOT NULL DEFAULT '{{}}',
              xml_path TEXT
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS protocol_jobs_job_id ON protocol_jobs(job_id)")
            self._rebuild_protocol_jobs_if_stale(db, status_check)
            existing_job_cols = {r[1] for r in db.execute("PRAGMA table_info(protocol_jobs)")}
            if "xml_path" not in existing_job_cols:
                db.execute("ALTER TABLE protocol_jobs ADD COLUMN xml_path TEXT")
            db.execute("""CREATE TABLE IF NOT EXISTS protocol_models (
              id TEXT PRIMARY KEY,
              label TEXT NOT NULL,
              engine TEXT NOT NULL,
              repo_id TEXT,
              filename TEXT,
              local_path TEXT,
              format TEXT,
              quantization TEXT,
              size_bytes INTEGER NOT NULL DEFAULT 0,
              context_length INTEGER NOT NULL DEFAULT 8192,
              temperature REAL NOT NULL DEFAULT 0.2,
              max_output_tokens INTEGER NOT NULL DEFAULT 2048,
              system_prompt_override TEXT,
              installed INTEGER NOT NULL DEFAULT 0,
              last_check_status TEXT,
              last_check_at REAL,
              updated_at REAL,
              n_gpu_layers INTEGER,
              extra_launch_params TEXT,
              ollama_url TEXT,
              ollama_keep_alive TEXT
            )""")
            existing_model_cols = {r[1] for r in db.execute("PRAGMA table_info(protocol_models)")}
            for col, typ in (("n_gpu_layers", "INTEGER"), ("extra_launch_params", "TEXT"),
                              ("ollama_url", "TEXT"), ("ollama_keep_alive", "TEXT")):
                if col not in existing_model_cols:
                    db.execute(f"ALTER TABLE protocol_models ADD COLUMN {col} {typ}")
            db.execute("""CREATE TABLE IF NOT EXISTS protocol_prompts (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              kind TEXT NOT NULL,
              model_id TEXT,
              version INTEGER NOT NULL,
              content TEXT NOT NULL,
              is_default INTEGER NOT NULL DEFAULT 0,
              is_active INTEGER NOT NULL DEFAULT 0,
              updated_by TEXT,
              updated_at REAL NOT NULL
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS protocol_prompts_kind_model ON protocol_prompts(kind, model_id, is_active)")
            db.execute("""CREATE TABLE IF NOT EXISTS protocol_results (
              protocol_job_id TEXT PRIMARY KEY REFERENCES protocol_jobs(id),
              document_json TEXT NOT NULL,
              model_id TEXT,
              prompt_versions TEXT,
              created_at REAL NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS glossary_terms (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              canonical TEXT NOT NULL,
              category TEXT,
              context TEXT,
              scope TEXT NOT NULL DEFAULT 'global',
              project TEXT,
              status TEXT NOT NULL DEFAULT 'confirmed',
              usage_count INTEGER NOT NULL DEFAULT 0,
              created_at REAL NOT NULL,
              updated_at REAL NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS glossary_aliases (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              term_id INTEGER NOT NULL REFERENCES glossary_terms(id),
              alias TEXT NOT NULL
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS glossary_aliases_alias ON glossary_aliases(alias)")
            db.execute("""CREATE TABLE IF NOT EXISTS glossary_suggestions (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              source TEXT NOT NULL,
              wrong_text TEXT NOT NULL,
              suggested_text TEXT NOT NULL,
              context TEXT,
              timestamp TEXT,
              confidence REAL NOT NULL DEFAULT 0,
              job_id TEXT,
              status TEXT NOT NULL DEFAULT 'proposed',
              created_at REAL NOT NULL,
              resolved_at REAL,
              resolved_by TEXT,
              chunk_index INTEGER,
              model_id TEXT
            )""")
            existing_suggestion_cols = {r[1] for r in db.execute("PRAGMA table_info(glossary_suggestions)")}
            for col, typ in (("chunk_index", "INTEGER"), ("model_id", "TEXT")):
                if col not in existing_suggestion_cols:
                    db.execute(f"ALTER TABLE glossary_suggestions ADD COLUMN {col} {typ}")
            db.execute("""CREATE TABLE IF NOT EXISTS protocol_settings (
              key TEXT PRIMARY KEY,
              value TEXT
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS glossary_corrections (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              job_id TEXT,
              original_text TEXT NOT NULL,
              corrected_text TEXT NOT NULL,
              rule TEXT,
              source TEXT,
              timestamp TEXT,
              confidence REAL NOT NULL DEFAULT 0,
              created_at REAL NOT NULL
            )""")
            self._migrate_qwen3_8b_max_output_tokens(db)
            db.commit()

    def _migrate_qwen3_8b_max_output_tokens(self, db: sqlite3.Connection) -> None:
        """Item 4: qwen3-8b's default max_output_tokens moved 2048 -> 12288.
        A one-time, flagged bump for rows already seeded at the old default
        -- an admin who deliberately customized it (to anything other than
        the exact old default) keeps their own value; this never
        overwrites a real customization, and never re-runs."""
        flag = "migrated_qwen3_8b_max_output_tokens_v1"
        already = db.execute("SELECT 1 FROM protocol_settings WHERE key=?", (flag,)).fetchone()
        if already:
            return
        row = db.execute("SELECT max_output_tokens FROM protocol_models WHERE id='qwen3-8b'").fetchone()
        if row is not None and row[0] == 2048:
            db.execute("UPDATE protocol_models SET max_output_tokens=12288, updated_at=? WHERE id='qwen3-8b'",
                       (time.time(),))
        db.execute("INSERT INTO protocol_settings (key, value) VALUES (?,?) "
                   "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (flag, "1"))

    def _rebuild_protocol_jobs_if_stale(self, db: sqlite3.Connection, status_check: str) -> None:
        """SQLite can't ALTER a CHECK constraint or add a NOT NULL column
        without a default to a table that might already have rows with a
        different set of columns, so a protocol_jobs table created before
        'fact_checking' (item 2) or settings_snapshot (item 1) existed is
        rebuilt in place: new table, copy over every column the old table
        actually had, swap names. Data is never dropped -- only tables
        already matching the current shape are left untouched.
        """
        row = db.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='protocol_jobs'").fetchone()
        if not row or not row[0]:
            return
        current_sql = row[0]
        if "fact_checking" in current_sql and "settings_snapshot" in current_sql:
            return  # already current shape
        db.execute("BEGIN IMMEDIATE")
        try:
            db.execute("ALTER TABLE protocol_jobs RENAME TO protocol_jobs_old")
            db.execute(f"""CREATE TABLE protocol_jobs (
              id TEXT PRIMARY KEY,
              job_id TEXT NOT NULL REFERENCES jobs(id),
              username TEXT NOT NULL,
              status TEXT NOT NULL {status_check},
              progress REAL NOT NULL DEFAULT 0,
              message TEXT,
              model_id TEXT,
              chunk_total INTEGER,
              chunk_current INTEGER,
              error TEXT,
              created_at REAL NOT NULL,
              started_at REAL,
              finished_at REAL,
              attempts INTEGER NOT NULL DEFAULT 0,
              cancel_requested INTEGER NOT NULL DEFAULT 0,
              json_path TEXT,
              html_path TEXT,
              log_path TEXT,
              settings_snapshot TEXT NOT NULL DEFAULT '{{}}',
              xml_path TEXT
            )""")
            old_cols = [r[1] for r in db.execute("PRAGMA table_info(protocol_jobs_old)")]
            new_cols = [r[1] for r in db.execute("PRAGMA table_info(protocol_jobs)")]
            common = [c for c in new_cols if c in old_cols]
            col_list = ", ".join(common)
            db.execute(f"INSERT INTO protocol_jobs ({col_list}) SELECT {col_list} FROM protocol_jobs_old")
            db.execute("DROP TABLE protocol_jobs_old")
            db.execute("COMMIT")
        except Exception:
            db.execute("ROLLBACK")
            raise
        db.execute("CREATE INDEX IF NOT EXISTS protocol_jobs_job_id ON protocol_jobs(job_id)")

    # ---- protocol_jobs ----

    def create_protocol_job(self, *, id: str, job_id: str, username: str, model_id: Optional[str],
                             settings_snapshot: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        with self._connect() as db:
            db.execute(
                "INSERT INTO protocol_jobs (id, job_id, username, status, progress, message, model_id, created_at, settings_snapshot) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (id, job_id, username, "queued", 0.0, "В очереди", model_id, time.time(),
                 json.dumps(settings_snapshot or {}, ensure_ascii=False)),
            )
            db.commit()
        return self.get_protocol_job(id)

    def get_protocol_job(self, protocol_job_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM protocol_jobs WHERE id=?", (protocol_job_id,)).fetchone()
            return self._job_row(row)

    def _job_row(self, row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
        if not row:
            return None
        d = dict(row)
        try:
            d["settings_snapshot"] = json.loads(d["settings_snapshot"]) if d.get("settings_snapshot") else {}
        except (TypeError, json.JSONDecodeError):
            d["settings_snapshot"] = {}
        return d

    def get_active_protocol_job_for_job(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as db:
            placeholders = ",".join("?" * len(PROTOCOL_ACTIVE))
            row = db.execute(
                f"SELECT * FROM protocol_jobs WHERE job_id=? AND status IN ({placeholders}) "
                "ORDER BY created_at DESC LIMIT 1",
                (job_id, *PROTOCOL_ACTIVE),
            ).fetchone()
            return self._job_row(row)

    def list_protocol_jobs_for_job(self, job_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM protocol_jobs WHERE job_id=? ORDER BY created_at DESC", (job_id,))
            return [self._job_row(r) for r in rows]

    def list_active_protocol_jobs(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            placeholders = ",".join("?" * len(PROTOCOL_ACTIVE))
            rows = db.execute(f"SELECT * FROM protocol_jobs WHERE status IN ({placeholders})", tuple(PROTOCOL_ACTIVE))
            return [self._job_row(r) for r in rows]

    def list_all_protocol_jobs(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM protocol_jobs ORDER BY created_at DESC LIMIT ?", (limit,))
            return [self._job_row(r) for r in rows]

    def update_protocol_job(self, protocol_job_id: str, **changes: Any) -> Optional[dict[str, Any]]:
        if not changes:
            return self.get_protocol_job(protocol_job_id)
        allowed = {
            "status", "progress", "message", "model_id", "chunk_total", "chunk_current",
            "error", "started_at", "finished_at", "attempts", "cancel_requested",
            "json_path", "html_path", "log_path", "xml_path",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"Unsupported protocol job columns: {sorted(unknown)}")
        with self._connect() as db:
            sql = ", ".join(f"{k}=?" for k in changes)
            db.execute(f"UPDATE protocol_jobs SET {sql} WHERE id=?", [*changes.values(), protocol_job_id])
            db.commit()
        return self.get_protocol_job(protocol_job_id)

    # ---- protocol_models ----

    def seed_model(self, spec: ModelSpec) -> None:
        with self._connect() as db:
            existing = db.execute("SELECT id FROM protocol_models WHERE id=?", (spec.id,)).fetchone()
            if existing:
                return
            db.execute(
                "INSERT INTO protocol_models (id,label,engine,repo_id,filename,format,quantization,"
                "context_length,temperature,max_output_tokens,installed,updated_at,"
                "n_gpu_layers,extra_launch_params,ollama_url,ollama_keep_alive) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (spec.id, spec.label, spec.engine, spec.repo_id, spec.filename, spec.format, spec.quantization,
                 spec.context_length, spec.temperature, spec.max_output_tokens, 0, time.time(),
                 spec.n_gpu_layers, json.dumps(spec.extra_launch_params or {}), spec.ollama_url, spec.ollama_keep_alive),
            )
            db.commit()

    def get_model_state(self, model_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM protocol_models WHERE id=?", (model_id,)).fetchone()
            return self._model_row(row)

    def list_model_states(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            return [self._model_row(r) for r in db.execute("SELECT * FROM protocol_models ORDER BY id")]

    def _model_row(self, row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
        if not row:
            return None
        d = dict(row)
        try:
            d["extra_launch_params"] = json.loads(d["extra_launch_params"]) if d.get("extra_launch_params") else {}
        except (TypeError, json.JSONDecodeError):
            d["extra_launch_params"] = {}
        return d

    def update_model_state(self, model_id: str, **changes: Any) -> Optional[dict[str, Any]]:
        if not changes:
            return self.get_model_state(model_id)
        allowed = {
            "engine", "repo_id", "filename", "local_path", "size_bytes", "context_length", "temperature",
            "max_output_tokens", "system_prompt_override", "installed", "last_check_status",
            "last_check_at", "updated_at", "n_gpu_layers", "extra_launch_params", "ollama_url", "ollama_keep_alive",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"Unsupported protocol model columns: {sorted(unknown)}")
        if "engine" in changes and changes["engine"] not in ("llama_cpp", "ollama"):
            raise ValueError("engine must be 'llama_cpp' or 'ollama'")
        if "extra_launch_params" in changes and not isinstance(changes["extra_launch_params"], str):
            changes = {**changes, "extra_launch_params": json.dumps(changes["extra_launch_params"] or {})}
        changes = {**changes, "updated_at": time.time()}
        with self._connect() as db:
            sql = ", ".join(f"{k}=?" for k in changes)
            db.execute(f"UPDATE protocol_models SET {sql} WHERE id=?", [*changes.values(), model_id])
            db.commit()
        return self.get_model_state(model_id)

    # ---- protocol_settings (admin-tunable runtime knobs; the master
    # enable/disable switch itself stays an env var -- see config.py) ----

    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._connect() as db:
            row = db.execute("SELECT value FROM protocol_settings WHERE key=?", (key,)).fetchone()
            return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO protocol_settings (key, value) VALUES (?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            db.commit()

    def all_settings(self) -> dict[str, str]:
        with self._connect() as db:
            return {r["key"]: r["value"] for r in db.execute("SELECT key, value FROM protocol_settings")}

    def active_model_id(self) -> Optional[str]:
        return self.get_setting("active_model_id")

    def set_active_model_id(self, model_id: str) -> None:
        self.set_setting("active_model_id", model_id)

    # ---- protocol_prompts ----

    def get_active_prompt(self, kind: str, model_id: Optional[str] = None) -> PromptSpec:
        with self._connect() as db:
            row = None
            if model_id:
                row = db.execute(
                    "SELECT * FROM protocol_prompts WHERE kind=? AND model_id=? AND is_active=1 ORDER BY version DESC LIMIT 1",
                    (kind, model_id),
                ).fetchone()
            if not row:
                row = db.execute(
                    "SELECT * FROM protocol_prompts WHERE kind=? AND model_id IS NULL AND is_active=1 ORDER BY version DESC LIMIT 1",
                    (kind,),
                ).fetchone()
        if row:
            return PromptSpec(kind=row["kind"], content=row["content"], version=row["version"],
                               is_default=bool(row["is_default"]), is_active=bool(row["is_active"]),
                               model_id=row["model_id"], updated_by=row["updated_by"], updated_at=row["updated_at"])
        return PromptSpec(kind=kind, content=DEFAULT_PROMPTS[kind], version=0, is_default=True, is_active=True)

    def save_prompt_version(self, kind: str, content: str, *, model_id: Optional[str] = None,
                             updated_by: Optional[str] = None, is_default: bool = False) -> PromptSpec:
        with self._connect() as db:
            last = db.execute(
                "SELECT MAX(version) as v FROM protocol_prompts WHERE kind=? AND model_id IS ?",
                (kind, model_id),
            ).fetchone()
            next_version = (last["v"] or 0) + 1
            db.execute("UPDATE protocol_prompts SET is_active=0 WHERE kind=? AND model_id IS ?", (kind, model_id))
            db.execute(
                "INSERT INTO protocol_prompts (kind, model_id, version, content, is_default, is_active, updated_by, updated_at) "
                "VALUES (?,?,?,?,?,1,?,?)",
                (kind, model_id, next_version, content, int(is_default), updated_by, time.time()),
            )
            db.commit()
        return self.get_active_prompt(kind, model_id)

    def restore_default_prompt(self, kind: str, *, model_id: Optional[str] = None,
                                updated_by: Optional[str] = None) -> PromptSpec:
        return self.save_prompt_version(kind, DEFAULT_PROMPTS[kind], model_id=model_id, updated_by=updated_by, is_default=True)

    def get_prompt_by_version(self, kind: str, model_id: Optional[str], version: int) -> Optional[PromptSpec]:
        """Fetch a specific historical prompt version -- used by retry() so a
        retried job replays the exact prompt text pinned at its original
        creation, not whatever version happens to be active now."""
        if version == 0:
            return PromptSpec(kind=kind, content=DEFAULT_PROMPTS[kind], version=0, is_default=True, is_active=True)
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM protocol_prompts WHERE kind=? AND model_id IS ? AND version=?",
                (kind, model_id, version),
            ).fetchone()
        if not row:
            return None
        return PromptSpec(kind=row["kind"], content=row["content"], version=row["version"],
                           is_default=bool(row["is_default"]), is_active=bool(row["is_active"]),
                           model_id=row["model_id"], updated_by=row["updated_by"], updated_at=row["updated_at"])

    def list_prompt_history(self, kind: str, model_id: Optional[str] = None) -> list[PromptSpec]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM protocol_prompts WHERE kind=? AND model_id IS ? ORDER BY version DESC",
                (kind, model_id),
            )
            return [PromptSpec(kind=r["kind"], content=r["content"], version=r["version"],
                                is_default=bool(r["is_default"]), is_active=bool(r["is_active"]),
                                model_id=r["model_id"], updated_by=r["updated_by"], updated_at=r["updated_at"])
                    for r in rows]

    # ---- protocol_results ----

    def save_result(self, protocol_job_id: str, document_json: dict[str, Any], *,
                     model_id: Optional[str], prompt_versions: dict[str, int]) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO protocol_results (protocol_job_id, document_json, model_id, prompt_versions, created_at) "
                "VALUES (?,?,?,?,?) ON CONFLICT(protocol_job_id) DO UPDATE SET "
                "document_json=excluded.document_json, model_id=excluded.model_id, "
                "prompt_versions=excluded.prompt_versions, created_at=excluded.created_at",
                (protocol_job_id, json.dumps(document_json, ensure_ascii=False), model_id,
                 json.dumps(prompt_versions), time.time()),
            )
            db.commit()

    def get_result(self, protocol_job_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM protocol_results WHERE protocol_job_id=?", (protocol_job_id,)).fetchone()
            if not row:
                return None
            d = dict(row)
            d["document_json"] = json.loads(d["document_json"])
            d["prompt_versions"] = json.loads(d["prompt_versions"]) if d.get("prompt_versions") else {}
            return d

    # ---- glossary ----

    def add_term(self, term: GlossaryTerm) -> GlossaryTerm:
        now = time.time()
        with self._connect() as db:
            cur = db.execute(
                "INSERT INTO glossary_terms (canonical,category,context,scope,project,status,usage_count,created_at,updated_at) "
                "VALUES (?,?,?,?,?,?,0,?,?)",
                (term.canonical, term.category, term.context, term.scope, term.project, term.status, now, now),
            )
            term_id = cur.lastrowid
            for alias in term.aliases:
                db.execute("INSERT INTO glossary_aliases (term_id, alias) VALUES (?,?)", (term_id, alias))
            db.commit()
        return self.get_term(term_id)

    def add_alias(self, term_id: int, alias: str) -> None:
        with self._connect() as db:
            db.execute("INSERT INTO glossary_aliases (term_id, alias) VALUES (?,?)", (term_id, alias))
            db.execute("UPDATE glossary_terms SET updated_at=? WHERE id=?", (time.time(), term_id))
            db.commit()

    def get_term(self, term_id: int) -> Optional[GlossaryTerm]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM glossary_terms WHERE id=?", (term_id,)).fetchone()
            if not row:
                return None
            aliases = [r["alias"] for r in db.execute("SELECT alias FROM glossary_aliases WHERE term_id=?", (term_id,))]
        return GlossaryTerm(id=row["id"], canonical=row["canonical"], aliases=tuple(aliases),
                             category=row["category"] or "", context=row["context"] or "", scope=row["scope"],
                             project=row["project"], status=row["status"], usage_count=row["usage_count"],
                             created_at=row["created_at"], updated_at=row["updated_at"])

    def list_terms(self, *, scope: Optional[str] = None, project: Optional[str] = None,
                    status: Optional[str] = None) -> list[GlossaryTerm]:
        clauses, args = [], []
        if scope: clauses.append("scope=?"); args.append(scope)
        if project: clauses.append("project=?"); args.append(project)
        if status: clauses.append("status=?"); args.append(status)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._connect() as db:
            rows = db.execute(f"SELECT id FROM glossary_terms{where} ORDER BY canonical", args).fetchall()
        return [self.get_term(r["id"]) for r in rows]

    def disable_term(self, term_id: int) -> None:
        with self._connect() as db:
            db.execute("UPDATE glossary_terms SET status='disabled', updated_at=? WHERE id=?", (time.time(), term_id))
            db.commit()

    def merge_terms(self, keep_id: int, remove_id: int) -> None:
        with self._connect() as db:
            db.execute("UPDATE glossary_aliases SET term_id=? WHERE term_id=?", (keep_id, remove_id))
            db.execute("DELETE FROM glossary_terms WHERE id=?", (remove_id,))
            db.commit()

    def bump_usage(self, term_id: int) -> None:
        with self._connect() as db:
            db.execute("UPDATE glossary_terms SET usage_count=usage_count+1, updated_at=? WHERE id=?", (time.time(), term_id))
            db.commit()

    def resolved_terms(self, *, project: Optional[str] = None, job_id: Optional[str] = None) -> list[GlossaryTerm]:
        """All confirmed terms visible to a job, highest-priority scope first
        for any given alias (job > project > department > global)."""
        terms = self.list_terms(status="confirmed")
        by_alias: dict[str, GlossaryTerm] = {}
        for scope in reversed(GLOSSARY_SCOPE_PRIORITY):
            for term in terms:
                if term.scope != scope:
                    continue
                if scope == "project" and project and term.project != project:
                    continue
                for alias in (*term.aliases, term.canonical):
                    by_alias[alias.lower()] = term
        seen_ids = set()
        result = []
        for term in by_alias.values():
            if term.id not in seen_ids:
                seen_ids.add(term.id)
                result.append(term)
        return result

    # -- suggestions --

    def add_suggestion(self, s: GlossarySuggestion) -> GlossarySuggestion:
        with self._connect() as db:
            cur = db.execute(
                "INSERT INTO glossary_suggestions (source,wrong_text,suggested_text,context,timestamp,confidence,job_id,status,created_at,chunk_index,model_id) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (s.source, s.wrong_text, s.suggested_text, s.context, s.timestamp, s.confidence, s.job_id, "proposed",
                 time.time(), s.chunk_index, s.model_id),
            )
            db.commit()
            return self.get_suggestion(cur.lastrowid)

    def has_pending_duplicate_suggestion(self, *, wrong_text: str, suggested_text: str, job_id: Optional[str]) -> bool:
        """Exact-duplicate guard (item 9): same wrong->suggested pair already
        pending for the same job is not re-proposed."""
        with self._connect() as db:
            row = db.execute(
                "SELECT 1 FROM glossary_suggestions WHERE status='proposed' AND wrong_text=? AND suggested_text=? "
                "AND job_id IS ? LIMIT 1",
                (wrong_text, suggested_text, job_id),
            ).fetchone()
            return row is not None

    def get_suggestion(self, suggestion_id: int) -> Optional[GlossarySuggestion]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM glossary_suggestions WHERE id=?", (suggestion_id,)).fetchone()
        if not row:
            return None
        return GlossarySuggestion(id=row["id"], source=row["source"], wrong_text=row["wrong_text"],
                                   suggested_text=row["suggested_text"], context=row["context"] or "",
                                   timestamp=row["timestamp"] or "", confidence=row["confidence"],
                                   job_id=row["job_id"], status=row["status"], created_at=row["created_at"],
                                   chunk_index=row["chunk_index"], model_id=row["model_id"])

    def list_suggestions(self, status: Optional[str] = "proposed") -> list[GlossarySuggestion]:
        with self._connect() as db:
            if status:
                rows = db.execute("SELECT id FROM glossary_suggestions WHERE status=? ORDER BY created_at DESC", (status,)).fetchall()
            else:
                rows = db.execute("SELECT id FROM glossary_suggestions ORDER BY created_at DESC").fetchall()
        return [self.get_suggestion(r["id"]) for r in rows]

    def resolve_suggestion(self, suggestion_id: int, *, status: str, resolved_by: str) -> None:
        if status not in ("confirmed", "rejected"):
            raise ValueError("status must be 'confirmed' or 'rejected'")
        with self._connect() as db:
            db.execute("UPDATE glossary_suggestions SET status=?, resolved_at=?, resolved_by=? WHERE id=?",
                       (status, time.time(), resolved_by, suggestion_id))
            db.commit()

    # -- corrections (audit log of applied replacements) --

    def log_correction(self, c: GlossaryCorrection) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO glossary_corrections (job_id,original_text,corrected_text,rule,source,timestamp,confidence,created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (c.job_id, c.original_text, c.corrected_text, c.rule, c.source, c.timestamp, c.confidence, time.time()),
            )
            db.commit()

    def list_corrections(self, job_id: Optional[str] = None) -> list[dict[str, Any]]:
        with self._connect() as db:
            if job_id:
                rows = db.execute("SELECT * FROM glossary_corrections WHERE job_id=? ORDER BY id", (job_id,))
            else:
                rows = db.execute("SELECT * FROM glossary_corrections ORDER BY id")
            return [dict(r) for r in rows]
