"""SQLite-backed job queue.  SQLite is the source of truth, never process memory."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

TERMINAL = {"completed", "failed", "cancelled"}
VALID = {"queued", "running", *TERMINAL}

class JobStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY, username TEXT NOT NULL, filename TEXT NOT NULL,
              status TEXT NOT NULL, progress REAL NOT NULL DEFAULT 0, message TEXT,
              created_at REAL NOT NULL, started_at REAL, finished_at REAL,
              error TEXT, settings_snapshot TEXT NOT NULL, actual_device TEXT,
              actual_models TEXT, transcript_path TEXT, log_path TEXT, original_path TEXT,
              wav_path TEXT, timeout_seconds INTEGER, attempts INTEGER NOT NULL DEFAULT 0,
              cancel_requested INTEGER NOT NULL DEFAULT 0)""")
            db.execute("CREATE INDEX IF NOT EXISTS jobs_user_created ON jobs(username, created_at DESC)")
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db
    def recover(self):
        """An interrupted worker cannot safely resume inference: requeue it."""
        with self._connect() as db:
            db.execute("UPDATE jobs SET status='queued', message='Восстановлено после перезапуска', started_at=NULL WHERE status='running' AND cancel_requested=0")
            db.execute("UPDATE jobs SET status='cancelled', finished_at=?, message='Отменено при перезапуске' WHERE status='running' AND cancel_requested=1", (time.time(),))
    def create(self, **j):
        cols = ['id','username','filename','status','progress','message','created_at','settings_snapshot','original_path','log_path','timeout_seconds']
        values = [j.get(c) for c in cols]; values[3] = values[3] or 'queued'; values[4] = values[4] or 0; values[6] = values[6] or time.time(); values[7] = json.dumps(values[7] or {})
        with self._connect() as db: db.execute(f"INSERT INTO jobs ({','.join(cols)}) VALUES ({','.join('?'*len(cols))})", values)
        return self.get(j['id'])
    def get(self, job_id):
        with self._connect() as db: row=db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._row(row)
    def list(self, username=None, active_only=False):
        q="SELECT * FROM jobs"; args=[]; where=[]
        if username: where.append("username=?"); args.append(username)
        if active_only: where.append("status IN ('queued','running')")
        if where: q += " WHERE " + " AND ".join(where)
        q += " ORDER BY created_at DESC"
        with self._connect() as db: rows=db.execute(q,args).fetchall()
        return [self._row(r) for r in rows]
    def update(self, job_id, **changes):
        if not changes: return self.get(job_id)
        for k in ('settings_snapshot','actual_models'):
            if k in changes: changes[k]=json.dumps(changes[k])
        sql=', '.join(f'{k}=?' for k in changes)
        with self._connect() as db: db.execute(f"UPDATE jobs SET {sql} WHERE id=?", [*changes.values(),job_id])
        return self.get(job_id)
    def request_cancel(self, job_id): return self.update(job_id, cancel_requested=1, message='Отмена запрошена')
    def _row(self,row):
        if not row:return None
        d=dict(row)
        for k in ('settings_snapshot','actual_models'):
            d[k]=json.loads(d[k]) if d[k] else {}
        d['cancel_requested']=bool(d['cancel_requested'])
        return d
