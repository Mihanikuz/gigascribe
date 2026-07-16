import threading
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from job_store import JobStore


def make(store, ident="job"):
    return store.create(id=ident, username="u", filename="a.wav", settings_snapshot={"asr_model": "gigaam-v2-ctc", "device": "cpu"})


def test_schema_wal_and_claim_is_single_worker(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    make(store)
    claimed = []
    def claim(): claimed.append(store.claim_next())
    workers = [threading.Thread(target=claim) for _ in range(2)]
    for worker in workers: worker.start()
    for worker in workers: worker.join()
    assert sum(row is not None for row in claimed) == 1
    assert store.get("job")["attempts"] == 1
    assert store.attempts("job")[0]["status"] == "running"


def test_restart_recovery_cancel_and_retry_clears_old_results(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    make(store)
    assert store.claim("job")["status"] == "running"
    store.recover()
    assert store.get("job")["status"] == "queued"
    assert store.claim("job")
    store.request_cancel("job")
    store.update("job", status="cancelled", finished_at=1, transcript_path="old.txt", wav_path="old.wav")
    retried = store.retry("job")
    assert retried["status"] == "queued"
    assert retried["transcript_path"] is None and retried["wav_path"] is None
    assert not retried["cancel_requested"]


def test_invalid_transition_and_retry_limit(tmp_path):
    store = JobStore(tmp_path / "jobs.db", retry_limit=1)
    make(store)
    with pytest.raises(ValueError): store.update("job", status="completed")
    store.claim("job")
    store.update("job", status="failed", finished_at=1)
    assert store.retry("job")["status"] == "queued"
    store.claim("job"); store.update("job", status="failed", finished_at=2)
    with pytest.raises(ValueError, match="Retry limit"):
        store.retry("job")
