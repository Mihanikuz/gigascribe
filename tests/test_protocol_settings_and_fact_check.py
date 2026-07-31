"""Settings-snapshot resolution/immutability, engine selection, and the
fact-check stage (items 1-4, 9-10, 13-15, 18-20 of the follow-up spec).

Uses the same fake-provider pattern as test_protocol_service_pipeline.py:
no GPU/torch/real LLM required, only the LLMProvider interface.
"""
import asyncio
import time
from pathlib import Path

import pytest

import job_store
import protocol.service as service_mod
from protocol.config import ProtocolConfig
from protocol.providers.base import LLMProvider
from protocol.schemas import ProtocolOptions, SettingsSnapshot, validate_settings_snapshot, ProtocolValidationError
from protocol.store import ProtocolStore


def _write_transcript(path: Path, n_segments: int = 40) -> None:
    lines = []
    t = 0
    for i in range(n_segments):
        start, end = t, t + 4
        speaker = "Спикер 1" if i % 2 == 0 else "Спикер 2"
        lines.append(f"{speaker} [{start // 60:02d}:{start % 60:02d} - {end // 60:02d}:{end % 60:02d}]: "
                      f"Реплика номер {i} про проект.")
        t = end + 2
    path.write_text("\n".join(lines), encoding="utf-8")


class ScriptedProvider(LLMProvider):
    def __init__(self, *, merge_response=None, chunk_response=None, fact_check_response=None,
                 fail_fact_check=False, on_generate=None):
        super().__init__(model_path="fake", context_length=8192)
        self.load_calls = 0
        self.unload_calls = 0
        self.fact_check_calls = 0
        self._merge_response = merge_response or (
            '{"summary": "ok", "topics": [], "decisions": [], "tasks": [], "open_questions": [], '
            '"risks": [], "disagreements": [], "next_steps": [], "unverified_items": []}'
        )
        self._chunk_response = chunk_response or (
            '{"topic": "t", "summary": "s", "facts": [], "decisions": [], "tasks": [], '
            '"open_questions": [], "risks": [], "disagreements": [], "terms": []}'
        )
        self._fact_check_response = fact_check_response
        self._fail_fact_check = fail_fact_check
        self._on_generate = on_generate

    async def load(self):
        self.load_calls += 1
        self.loaded = True

    async def generate(self, prompt, *, temperature, max_tokens, system_prompt=None):
        if self._on_generate:
            self._on_generate(prompt)
        if "определи, можно ли надёжно разделить" in prompt:
            return '{"topics": []}'
        if "Результаты анализа фрагментов" in prompt:
            return self._merge_response
        if "Проверь каждое решение" in prompt:
            self.fact_check_calls += 1
            if self._fail_fact_check:
                return "not json ever"
            if self._fact_check_response is not None:
                return self._fact_check_response
            return '{"verified": [], "unverified_items": []}'
        return self._chunk_response

    async def unload(self):
        self.unload_calls += 1
        self.loaded = False


@pytest.fixture()
def env(tmp_path):
    from protocol.models import SUPPORTED_PROTOCOL_MODELS
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    data_dir.mkdir(); models_dir.mkdir()
    store_path = data_dir / "jobs.sqlite3"
    js = job_store.JobStore(store_path)
    config = ProtocolConfig.from_env(data_dir=data_dir, models_dir=models_dir)
    store = ProtocolStore(store_path)
    for spec in SUPPORTED_PROTOCOL_MODELS.values():
        store.seed_model(spec)
    return {"data_dir": data_dir, "models_dir": models_dir, "job_store": js, "config": config, "store": store}


def _install_model(store: ProtocolStore, models_dir: Path, model_id: str = "qwen3-8b", engine: str = "llama_cpp") -> None:
    if engine == "llama_cpp":
        path = models_dir / f"{model_id}.gguf"
        path.write_bytes(b"fake-gguf")
        store.update_model_state(model_id, local_path=str(path), installed=1, engine="llama_cpp")
    else:
        store.update_model_state(model_id, local_path="qwen3:8b", installed=1, engine="ollama",
                                  ollama_url="http://127.0.0.1:11434", ollama_keep_alive="5m")
    store.set_active_model_id(model_id)


async def _wait_terminal(store: ProtocolStore, protocol_job_id: str, timeout: float = 15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = store.get_protocol_job(protocol_job_id)
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        await asyncio.sleep(0.02)
    raise TimeoutError("protocol job did not reach a terminal state in time")


def _make_job(env, job_id="job-1", username="alice"):
    env["job_store"].create(id=job_id, username=username, filename="m.wav", settings_snapshot={})
    transcript = env["data_dir"] / f"{job_id}.txt"
    _write_transcript(transcript)
    return transcript


# ---- item 1: settings snapshot resolution & immutability ----------------

def test_settings_from_protocol_settings_table_are_applied(env, monkeypatch):
    """protocol_settings values must actually reach the pipeline, not just
    the admin API response."""
    from protocol.service import ProtocolService
    transcript = _make_job(env)
    _install_model(env["store"], env["models_dir"])
    env["store"].set_setting("chunk_minutes", "11")
    env["store"].set_setting("temperature", "0.05")

    monkeypatch.setattr(service_mod, "create_provider", lambda **k: ScriptedProvider())
    svc = ProtocolService(config=env["config"], store=env["store"], gpu_lock=asyncio.Lock(), unload_asr=lambda: None)

    async def run():
        result = await svc.create_protocol(transcript_path=transcript, job_id="job-1", username="alice")
        return env["store"].get_protocol_job(result.protocol_job_id)

    job = asyncio.run(run())
    snapshot = job["settings_snapshot"]
    assert snapshot["chunk_minutes"] == 11.0
    assert snapshot["temperature"] == 0.05


def test_request_options_take_priority_over_settings(env, monkeypatch):
    """item 1: request > protocol_settings > model > config."""
    from protocol.service import ProtocolService
    transcript = _make_job(env)
    _install_model(env["store"], env["models_dir"])
    env["store"].set_setting("temperature", "0.05")

    monkeypatch.setattr(service_mod, "create_provider", lambda **k: ScriptedProvider())
    svc = ProtocolService(config=env["config"], store=env["store"], gpu_lock=asyncio.Lock(), unload_asr=lambda: None)

    async def run():
        result = await svc.create_protocol(transcript_path=transcript, job_id="job-1", username="alice",
                                            options=ProtocolOptions(temperature=0.9))
        return env["store"].get_protocol_job(result.protocol_job_id)

    job = asyncio.run(run())
    assert job["settings_snapshot"]["temperature"] == 0.9


def test_settings_snapshot_frozen_at_creation_survives_later_admin_changes(env, monkeypatch):
    """item 1: an admin changing settings after a job is created must not
    change what that job (or its retry) actually does."""
    from protocol.service import ProtocolService
    transcript = _make_job(env)
    _install_model(env["store"], env["models_dir"])
    env["store"].set_setting("chunk_minutes", "10")

    seen_chunk_minutes = []

    def fake_create_provider(**kwargs):
        return ScriptedProvider()

    monkeypatch.setattr(service_mod, "create_provider", fake_create_provider)
    svc = ProtocolService(config=env["config"], store=env["store"], gpu_lock=asyncio.Lock(), unload_asr=lambda: None)

    async def run():
        result = await svc.create_protocol(transcript_path=transcript, job_id="job-1", username="alice")
        job_before = env["store"].get_protocol_job(result.protocol_job_id)
        seen_chunk_minutes.append(job_before["settings_snapshot"]["chunk_minutes"])
        # Admin changes settings *after* the job was created but before it finishes.
        env["store"].set_setting("chunk_minutes", "15")
        job_after = await _wait_terminal(env["store"], result.protocol_job_id)
        return job_after

    job = asyncio.run(run())
    assert job["status"] == "completed"
    assert job["settings_snapshot"]["chunk_minutes"] == 10.0, "the frozen snapshot must not pick up the later change"
    assert seen_chunk_minutes == [10.0]


def test_retry_replays_original_snapshot_not_new_settings(env, monkeypatch):
    from protocol.service import ProtocolService
    transcript = _make_job(env)
    _install_model(env["store"], env["models_dir"])
    env["store"].set_setting("chunk_minutes", "10")
    merge_response = (
        '{"summary": "s", "topics": [], '
        '"decisions": [{"text": "X", "speaker": "Спикер 1", "timestamp": "00:01", "confidence": 0.9}], '
        '"tasks": [], "open_questions": [], "risks": [], "disagreements": [], "next_steps": [], "unverified_items": []}'
    )

    monkeypatch.setattr(service_mod, "create_provider",
                         lambda **k: ScriptedProvider(merge_response=merge_response, fail_fact_check=True))
    svc = ProtocolService(config=env["config"], store=env["store"], gpu_lock=asyncio.Lock(), unload_asr=lambda: None)

    async def run():
        result = await svc.create_protocol(transcript_path=transcript, job_id="job-1", username="alice")
        failed = await _wait_terminal(env["store"], result.protocol_job_id)
        assert failed["status"] == "failed"
        env["store"].set_setting("chunk_minutes", "14")
        service_mod.create_provider = lambda **k: ScriptedProvider()
        await svc.retry(result.protocol_job_id, transcript, "job-1")
        return await _wait_terminal(env["store"], result.protocol_job_id)

    job = asyncio.run(run())
    assert job["status"] == "completed"
    assert job["settings_snapshot"]["chunk_minutes"] == 10.0


def test_settings_snapshot_validation_rejects_out_of_range_chunk_minutes(env, monkeypatch):
    from protocol.service import ProtocolService
    transcript = _make_job(env)
    _install_model(env["store"], env["models_dir"])
    monkeypatch.setattr(service_mod, "create_provider", lambda **k: ScriptedProvider())
    svc = ProtocolService(config=env["config"], store=env["store"], gpu_lock=asyncio.Lock(), unload_asr=lambda: None)

    async def run():
        with pytest.raises(ProtocolValidationError):
            await svc.create_protocol(transcript_path=transcript, job_id="job-1", username="alice",
                                       options=ProtocolOptions(chunk_minutes=1))

    asyncio.run(run())


def test_settings_snapshot_round_trips_through_json():
    snap = SettingsSnapshot(
        model_id="qwen3-8b", engine="llama_cpp", model_path="/x.gguf", context_length=8192,
        temperature=0.2, max_output_tokens=1024, chunk_minutes=12, chunk_overlap_seconds=30,
        topic_split_enabled=True, max_retries=2, glossary_suggestions_enabled=True,
        default_glossary_scope="global", glossary_project=None, prompt_versions={"merge": 1}, created_at=1.0,
    )
    validate_settings_snapshot(snap)  # must not raise
    restored = SettingsSnapshot.from_json_dict(snap.to_json_dict())
    assert restored == snap


# ---- item 4: engine selection actually used --------------------------

def test_engine_selection_flows_from_model_state_into_provider(env, monkeypatch):
    transcript = _make_job(env)
    _install_model(env["store"], env["models_dir"], engine="ollama")

    captured = {}

    def fake_create_provider(**kwargs):
        captured.update(kwargs)
        return ScriptedProvider()

    from protocol.service import ProtocolService
    monkeypatch.setattr(service_mod, "create_provider", fake_create_provider)
    svc = ProtocolService(config=env["config"], store=env["store"], gpu_lock=asyncio.Lock(), unload_asr=lambda: None)

    async def run():
        result = await svc.create_protocol(transcript_path=transcript, job_id="job-1", username="alice")
        return await _wait_terminal(env["store"], result.protocol_job_id)

    job = asyncio.run(run())
    assert job["status"] == "completed"
    assert captured["engine"] == "ollama"
    assert captured["ollama_url"] == "http://127.0.0.1:11434"
    assert captured["ollama_keep_alive"] == "5m"


# ---- items 2, 13-14: fact-check stage ---------------------------------

def test_fact_check_is_actually_invoked(env, monkeypatch):
    transcript = _make_job(env)
    _install_model(env["store"], env["models_dir"])
    merge_response = (
        '{"summary": "s", "topics": [], '
        '"decisions": [{"text": "X", "speaker": "Спикер 1", "timestamp": "00:01", "confidence": 0.9}], '
        '"tasks": [], "open_questions": [], "risks": [], "disagreements": [], "next_steps": [], "unverified_items": []}'
    )
    provider = ScriptedProvider(merge_response=merge_response)
    from protocol.service import ProtocolService
    monkeypatch.setattr(service_mod, "create_provider", lambda **k: provider)
    svc = ProtocolService(config=env["config"], store=env["store"], gpu_lock=asyncio.Lock(), unload_asr=lambda: None)

    async def run():
        result = await svc.create_protocol(transcript_path=transcript, job_id="job-1", username="alice")
        return await _wait_terminal(env["store"], result.protocol_job_id)

    job = asyncio.run(run())
    assert job["status"] == "completed"
    assert provider.fact_check_calls == 1


def test_unconfirmed_item_marked_but_not_deleted_and_version_recorded(env, monkeypatch):
    transcript = _make_job(env)
    _install_model(env["store"], env["models_dir"])
    merge_response = (
        '{"summary": "s", "topics": [], '
        '"decisions": [{"text": "Сомнительное решение", "speaker": "Спикер 1", "timestamp": "00:01", "confidence": 0.5}], '
        '"tasks": [], "open_questions": [], "risks": [], "disagreements": [], "next_steps": [], "unverified_items": []}'
    )
    fact_check_response = '{"verified": [{"type": "decision", "index": 0, "confirmed": false, "reason": "не подтверждено в тексте"}], "unverified_items": []}'
    provider = ScriptedProvider(merge_response=merge_response, fact_check_response=fact_check_response)
    from protocol.service import ProtocolService
    monkeypatch.setattr(service_mod, "create_provider", lambda **k: provider)
    svc = ProtocolService(config=env["config"], store=env["store"], gpu_lock=asyncio.Lock(), unload_asr=lambda: None)

    async def run():
        result = await svc.create_protocol(transcript_path=transcript, job_id="job-1", username="alice")
        return await _wait_terminal(env["store"], result.protocol_job_id)

    job = asyncio.run(run())
    assert job["status"] == "completed"
    res = env["store"].get_result(job["id"])
    decision = res["document_json"]["decisions"][0]
    assert decision["text"] == "Сомнительное решение", "content must never be rewritten by fact-check"
    assert decision["verified"] is False
    assert decision["verification_reason"]
    assert "fact_check" in res["document_json"]["prompt_versions"]


def test_fact_check_technical_failure_fails_whole_job(env, monkeypatch):
    """item 2: an unrecoverable fact-check error must fail the job, not
    silently publish an unverified protocol as 'ready'."""
    transcript = _make_job(env)
    _install_model(env["store"], env["models_dir"])
    merge_response = (
        '{"summary": "s", "topics": [], '
        '"decisions": [{"text": "X", "speaker": "Спикер 1", "timestamp": "00:01", "confidence": 0.9}], '
        '"tasks": [], "open_questions": [], "risks": [], "disagreements": [], "next_steps": [], "unverified_items": []}'
    )
    provider = ScriptedProvider(merge_response=merge_response, fail_fact_check=True)
    from protocol.service import ProtocolService
    monkeypatch.setattr(service_mod, "create_provider", lambda **k: provider)
    svc = ProtocolService(config=env["config"], store=env["store"], gpu_lock=asyncio.Lock(), unload_asr=lambda: None)

    async def run():
        result = await svc.create_protocol(transcript_path=transcript, job_id="job-1", username="alice")
        return await _wait_terminal(env["store"], result.protocol_job_id)

    job = asyncio.run(run())
    assert job["status"] == "failed"
    assert env["store"].get_result(job["id"]) is None, "no protocol result must be published on fact-check failure"
    assert provider.unload_calls == 1, "LLM must still be unloaded after a fact-check failure"


# ---- item 9: LLM-proposed glossary terms -------------------------------

def test_llm_proposed_terms_from_chunk_analysis_create_suggestions_without_duplicates(env, monkeypatch):
    transcript = _make_job(env)
    _install_model(env["store"], env["models_dir"])
    chunk_response = (
        '{"topic": "t", "summary": "s", "facts": [], "decisions": [], "tasks": [], '
        '"open_questions": [], "risks": [], "disagreements": [], '
        '"terms": [{"detected": "постгрес скуль", "suggested": "PostgreSQL", "context": "c", '
        '"timestamp": "00:01", "confidence": 0.7}, '
        '{"detected": "PostgreSQL", "suggested": "PostgreSQL", "context": "", "timestamp": "", "confidence": 0.9}]}'
    )
    provider = ScriptedProvider(chunk_response=chunk_response)
    from protocol.service import ProtocolService
    monkeypatch.setattr(service_mod, "create_provider", lambda **k: provider)
    svc = ProtocolService(config=env["config"], store=env["store"], gpu_lock=asyncio.Lock(), unload_asr=lambda: None)

    async def run():
        result = await svc.create_protocol(transcript_path=transcript, job_id="job-1", username="alice",
                                            options=ProtocolOptions(chunk_minutes=10))
        return await _wait_terminal(env["store"], result.protocol_job_id)

    job = asyncio.run(run())
    assert job["status"] == "completed"
    suggestions = env["store"].list_suggestions("proposed")
    matching = [s for s in suggestions if s.wrong_text == "постгрес скуль"]
    assert len(matching) == 1, "self-replacement suggestion must be filtered, real one kept exactly once"
    assert matching[0].source == "llm_proposal"
    assert not any(s.wrong_text == s.suggested_text for s in suggestions), "no term may suggest itself"

    terms = env["store"].resolved_terms()
    assert not any("постгрес скуль" in t.aliases for t in terms), "an LLM proposal must never be auto-applied"


def test_glossary_suggestions_disabled_setting_stops_proposals(env, monkeypatch):
    transcript = _make_job(env)
    _install_model(env["store"], env["models_dir"])
    env["store"].set_setting("glossary_suggestions_enabled", "false")
    chunk_response = (
        '{"topic": "t", "summary": "s", "facts": [], "decisions": [], "tasks": [], '
        '"open_questions": [], "risks": [], "disagreements": [], '
        '"terms": [{"detected": "азуре", "suggested": "Azure", "context": "", "timestamp": "", "confidence": 0.8}]}'
    )
    provider = ScriptedProvider(chunk_response=chunk_response)
    from protocol.service import ProtocolService
    monkeypatch.setattr(service_mod, "create_provider", lambda **k: provider)
    svc = ProtocolService(config=env["config"], store=env["store"], gpu_lock=asyncio.Lock(), unload_asr=lambda: None)

    async def run():
        result = await svc.create_protocol(transcript_path=transcript, job_id="job-1", username="alice")
        return await _wait_terminal(env["store"], result.protocol_job_id)

    job = asyncio.run(run())
    assert job["status"] == "completed"
    assert not env["store"].list_suggestions("proposed")


# ---- item 7: long meeting must not be capped to the first 20000 chars --

def test_long_meeting_topic_split_covers_the_whole_duration_not_just_first_20000_chars(env, monkeypatch):
    transcript_path = env["data_dir"] / "long.txt"
    _write_transcript(transcript_path, n_segments=600)  # far beyond 20000 chars of raw text
    assert len(transcript_path.read_text(encoding="utf-8")) > 20000
    env["job_store"].create(id="job-long", username="alice", filename="m.wav", settings_snapshot={})
    _install_model(env["store"], env["models_dir"])

    seen_topic_split_prompts = []

    def on_generate(prompt):
        if "определи, можно ли надёжно разделить" in prompt:
            seen_topic_split_prompts.append(prompt)

    provider = ScriptedProvider(on_generate=on_generate)
    from protocol.service import ProtocolService
    monkeypatch.setattr(service_mod, "create_provider", lambda **k: provider)
    svc = ProtocolService(config=env["config"], store=env["store"], gpu_lock=asyncio.Lock(), unload_asr=lambda: None)

    async def run():
        result = await svc.create_protocol(transcript_path=transcript_path, job_id="job-long", username="alice",
                                            options=ProtocolOptions(chunk_minutes=10))
        return await _wait_terminal(env["store"], result.protocol_job_id)

    job = asyncio.run(run())
    assert job["status"] == "completed"
    # Two-stage split: several small per-block calls, not one call holding
    # the (truncated) first-20000-chars of raw transcript text.
    assert len(seen_topic_split_prompts) > 1, "expected multiple pre-block topic-split calls, not a single truncated one"
    assert all(len(p) < 20000 for p in seen_topic_split_prompts), "no single topic-split call should carry the raw un-chunked transcript"
