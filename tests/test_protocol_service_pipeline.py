"""End-to-end ProtocolService pipeline with a fake LLM provider (items 4-9, 13-15, 22-23).

No GPU, torch, or real LLM required: the fake provider implements the same
LLMProvider interface a real llama.cpp/Ollama provider would, so this
exercises the real state machine, GPU-lock handoff, chunking dispatch, and
document assembly logic end to end.
"""
import asyncio
import time
from pathlib import Path

import pytest

import job_store
import protocol.service as service_mod
from protocol.config import ProtocolConfig
from protocol.providers.base import LLMProvider
from protocol.schemas import ProtocolOptions, ProtocolValidationError
from protocol.store import ProtocolStore


def _write_transcript(path: Path, n_segments: int = 120) -> None:
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
    """Dispatches canned responses based on which stage's prompt text is seen."""

    def __init__(self, *, merge_response=None, chunk_response=None, always_bad=False,
                 on_generate=None):
        super().__init__(model_path="fake", context_length=8192)
        self.load_calls = 0
        self.unload_calls = 0
        self.generate_calls = 0
        self._merge_response = merge_response or (
            '{"summary": "ok", "topics": [], "decisions": [], "tasks": [], "open_questions": [], '
            '"risks": [], "disagreements": [], "next_steps": [], "unverified_items": []}'
        )
        self._chunk_response = chunk_response or (
            '{"topic": "t", "summary": "s", "facts": [], "decisions": [], "tasks": [], '
            '"open_questions": [], "risks": [], "disagreements": [], "terms": []}'
        )
        self._always_bad = always_bad
        self._on_generate = on_generate

    async def load(self):
        self.load_calls += 1
        self.loaded = True

    async def generate(self, prompt, *, temperature, max_tokens, system_prompt=None):
        self.generate_calls += 1
        if self._on_generate:
            self._on_generate(prompt)
        if self._always_bad:
            return "not json ever"
        if "определи, можно ли надёжно разделить" in prompt:
            return '{"topics": []}'
        if "Результаты анализа фрагментов" in prompt:
            return self._merge_response
        if "Проверь каждое решение" in prompt:
            return '{"verified": [], "unverified_items": []}'
        return self._chunk_response

    async def unload(self):
        self.unload_calls += 1
        self.loaded = False


@pytest.fixture()
def env(tmp_path, monkeypatch):
    from protocol.models import SUPPORTED_PROTOCOL_MODELS
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    data_dir.mkdir(); models_dir.mkdir()
    store_path = data_dir / "jobs.sqlite3"
    js = job_store.JobStore(store_path)
    config = ProtocolConfig.from_env(data_dir=data_dir, models_dir=models_dir)
    store = ProtocolStore(store_path)
    # Seed independently of ProtocolService construction order: tests may
    # need the model installed before constructing the service under test.
    for spec in SUPPORTED_PROTOCOL_MODELS.values():
        store.seed_model(spec)
    return {"data_dir": data_dir, "models_dir": models_dir, "job_store": js, "config": config, "store": store}


def _install_model(store: ProtocolStore, models_dir: Path, model_id: str = "qwen3-8b") -> None:
    # validate_engine_config() checks the GGUF path actually exists (item 4),
    # so tests need a real (if empty) file rather than a made-up path.
    gguf_path = models_dir / f"{model_id}.gguf"
    gguf_path.write_bytes(b"fake-gguf")
    store.update_model_state(model_id, local_path=str(gguf_path), installed=1)
    store.set_active_model_id(model_id)


async def _wait_terminal(store: ProtocolStore, protocol_job_id: str, timeout: float = 15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = store.get_protocol_job(protocol_job_id)
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        await asyncio.sleep(0.02)
    raise TimeoutError("protocol job did not reach a terminal state in time")


def test_create_protocol_requires_completed_transcript(env, monkeypatch, tmp_path):
    """Item 4: protocol must not start before transcription is done -- this
    is enforced by the caller (server.py) checking job status, and here at
    the service level by requiring a real, existing transcript file."""
    from protocol.service import ProtocolService
    monkeypatch.setattr(service_mod, "create_provider", lambda **k: ScriptedProvider())
    _install_model(env["store"], env["models_dir"])
    svc = ProtocolService(config=env["config"], store=env["store"], gpu_lock=asyncio.Lock(), unload_asr=lambda: None)

    async def run():
        with pytest.raises(FileNotFoundError):
            await svc.create_protocol(transcript_path=tmp_path / "missing.txt", job_id="job-x", username="a")

    asyncio.run(run())


def test_asr_unloaded_before_llm_loaded_and_llm_unloaded_on_success(env, monkeypatch):
    """Items 5 & 6."""
    from protocol.service import ProtocolService
    env["job_store"].create(id="job-1", username="alice", filename="m.wav", settings_snapshot={})
    transcript = env["data_dir"] / "t.txt"; _write_transcript(transcript)
    _install_model(env["store"], env["models_dir"])

    call_order = []
    provider = ScriptedProvider()
    provider_load = provider.load
    async def tracked_load():
        call_order.append("llm_load")
        await provider_load()
    provider.load = tracked_load

    monkeypatch.setattr(service_mod, "create_provider", lambda **k: provider)
    def unload_asr():
        call_order.append("asr_unload")
    svc = ProtocolService(config=env["config"], store=env["store"], gpu_lock=asyncio.Lock(), unload_asr=unload_asr)

    async def run():
        result = await svc.create_protocol(transcript_path=transcript, job_id="job-1", username="alice",
                                            options=ProtocolOptions(chunk_minutes=10))
        job = await _wait_terminal(env["store"], result.protocol_job_id)
        return job

    job = asyncio.run(run())
    assert job["status"] == "completed"
    assert call_order.index("asr_unload") < call_order.index("llm_load"), "ASR must unload before LLM loads"
    assert provider.load_calls == 1
    assert provider.unload_calls == 1, "LLM must be unloaded after a successful run"


def test_llm_unloaded_after_failure_and_transcription_job_untouched(env, monkeypatch):
    """Items 6 & 23: LLM unloads on failure too, and the main job row is
    never touched by a protocol failure."""
    from protocol.service import ProtocolService
    env["job_store"].create(id="job-2", username="alice", filename="m.wav", settings_snapshot={})
    transcript = env["data_dir"] / "t.txt"; _write_transcript(transcript)
    _install_model(env["store"], env["models_dir"])

    provider = ScriptedProvider(always_bad=True)
    monkeypatch.setattr(service_mod, "create_provider", lambda **k: provider)
    svc = ProtocolService(config=env["config"], store=env["store"], gpu_lock=asyncio.Lock(), unload_asr=lambda: None)

    async def run():
        result = await svc.create_protocol(transcript_path=transcript, job_id="job-2", username="alice")
        return await _wait_terminal(env["store"], result.protocol_job_id)

    job = asyncio.run(run())
    assert job["status"] == "failed"
    assert job["error"]
    assert provider.unload_calls == 1

    main_job = env["job_store"].get("job-2")
    assert main_job["status"] == "queued"  # untouched by the protocol failure
    assert main_job["error"] is None


def test_gpu_mutual_exclusion_between_transcription_and_protocol(env, monkeypatch):
    """Item 7: transcription and protocol never hold the GPU at once."""
    from protocol.service import ProtocolService
    env["job_store"].create(id="job-3", username="alice", filename="m.wav", settings_snapshot={})
    transcript = env["data_dir"] / "t.txt"; _write_transcript(transcript)
    _install_model(env["store"], env["models_dir"])

    gpu_lock = asyncio.Lock()
    active = {"n": 0}
    overlap = {"seen": False}

    async def fake_transcription():
        async with gpu_lock:
            active["n"] += 1
            if active["n"] > 1: overlap["seen"] = True
            await asyncio.sleep(0.05)
            active["n"] -= 1

    def on_generate(prompt):
        active["n"] += 1
        if active["n"] > 1: overlap["seen"] = True
        active["n"] -= 1

    provider = ScriptedProvider(on_generate=on_generate)
    monkeypatch.setattr(service_mod, "create_provider", lambda **k: provider)
    svc = ProtocolService(config=env["config"], store=env["store"], gpu_lock=gpu_lock, unload_asr=lambda: None)

    async def run():
        result = await svc.create_protocol(transcript_path=transcript, job_id="job-3", username="alice",
                                            options=ProtocolOptions(chunk_minutes=10))
        await asyncio.gather(fake_transcription(), fake_transcription())
        return await _wait_terminal(env["store"], result.protocol_job_id)

    job = asyncio.run(run())
    assert job["status"] == "completed"
    assert not overlap["seen"], "GPU lock did not enforce mutual exclusion between transcription and protocol"


def test_duplicate_create_returns_existing_active_job(env, monkeypatch):
    """Item: 'повторное нажатие не создаёт дубликат активного задания'."""
    from protocol.service import ProtocolService
    env["job_store"].create(id="job-4", username="alice", filename="m.wav", settings_snapshot={})
    transcript = env["data_dir"] / "t.txt"; _write_transcript(transcript)
    _install_model(env["store"], env["models_dir"])
    monkeypatch.setattr(service_mod, "create_provider", lambda **k: ScriptedProvider())
    svc = ProtocolService(config=env["config"], store=env["store"], gpu_lock=asyncio.Lock(), unload_asr=lambda: None)

    async def run():
        r1 = await svc.create_protocol(transcript_path=transcript, job_id="job-4", username="alice")
        r2 = await svc.create_protocol(transcript_path=transcript, job_id="job-4", username="alice")
        return r1, r2

    r1, r2 = asyncio.run(run())
    assert r1.protocol_job_id == r2.protocol_job_id


def test_retry_after_failure_produces_a_completed_result(env, monkeypatch):
    """Item 22: re-generation after a failed attempt succeeds without corrupting state."""
    from protocol.service import ProtocolService
    env["job_store"].create(id="job-5", username="alice", filename="m.wav", settings_snapshot={})
    transcript = env["data_dir"] / "t.txt"; _write_transcript(transcript)
    _install_model(env["store"], env["models_dir"])

    bad_provider = ScriptedProvider(always_bad=True)
    monkeypatch.setattr(service_mod, "create_provider", lambda **k: bad_provider)
    svc = ProtocolService(config=env["config"], store=env["store"], gpu_lock=asyncio.Lock(), unload_asr=lambda: None)

    async def run():
        result = await svc.create_protocol(transcript_path=transcript, job_id="job-5", username="alice")
        failed_job = await _wait_terminal(env["store"], result.protocol_job_id)
        assert failed_job["status"] == "failed"

        good_provider = ScriptedProvider()
        service_mod.create_provider = lambda **k: good_provider
        await svc.retry(result.protocol_job_id, transcript, "job-5")
        return await _wait_terminal(env["store"], result.protocol_job_id)

    job = asyncio.run(run())
    assert job["status"] == "completed"
    res = env["store"].get_result(job["id"])
    assert res is not None


def test_no_fabricated_owner_or_deadline_when_not_stated(env, monkeypatch):
    """Item 14: the model is not allowed to invent owners/deadlines -- verify
    the document assembly keeps whatever the model actually returned
    (including the 'не указан' placeholder) rather than backfilling one."""
    from protocol.service import ProtocolService
    env["job_store"].create(id="job-6", username="alice", filename="m.wav", settings_snapshot={})
    transcript = env["data_dir"] / "t.txt"; _write_transcript(transcript)
    _install_model(env["store"], env["models_dir"])

    merge_response = (
        '{"summary": "s", "topics": [], "decisions": [], '
        '"tasks": [{"task": "Подготовить отчёт", "owner": "не указан", "deadline": "не указан", '
        '"speaker": "Спикер 1", "timestamp": "00:01", "confidence": 0.4}], '
        '"open_questions": [], "risks": [], "disagreements": [], "next_steps": [], '
        '"unverified_items": ["Ответственный за отчёт не указан"]}'
    )
    provider = ScriptedProvider(merge_response=merge_response)
    monkeypatch.setattr(service_mod, "create_provider", lambda **k: provider)
    svc = ProtocolService(config=env["config"], store=env["store"], gpu_lock=asyncio.Lock(), unload_asr=lambda: None)

    async def run():
        result = await svc.create_protocol(transcript_path=transcript, job_id="job-6", username="alice",
                                            options=ProtocolOptions(chunk_minutes=10))
        return await _wait_terminal(env["store"], result.protocol_job_id)

    job = asyncio.run(run())
    assert job["status"] == "completed"
    res = env["store"].get_result(job["id"])
    task = res["document_json"]["tasks"][0]
    assert task["owner"] == "не указан"
    assert task["deadline"] == "не указан"
    assert res["document_json"]["unverified_items"]
    # owners/deadlines summary lists must not contain the placeholder itself
    assert "не указан" not in res["document_json"]["owners"]


def test_final_document_has_timestamp_refs_linking_to_source_chunks(env, monkeypatch):
    """Item 15 / item 16: a decision with a timestamp that lands near a real
    transcript segment gets a resolved anchor and an appendix entry."""
    from protocol.service import ProtocolService
    env["job_store"].create(id="job-7", username="alice", filename="m.wav", settings_snapshot={})
    transcript = env["data_dir"] / "t.txt"; _write_transcript(transcript, n_segments=200)
    _install_model(env["store"], env["models_dir"])
    merge_response = (
        '{"summary": "s", "topics": [], '
        '"decisions": [{"text": "Перейти на новую версию", "speaker": "Спикер 1", "timestamp": "00:01", "confidence": 0.8}], '
        '"tasks": [], "open_questions": [], "risks": [], "disagreements": [], "next_steps": [], "unverified_items": []}'
    )
    monkeypatch.setattr(service_mod, "create_provider", lambda **k: ScriptedProvider(merge_response=merge_response))
    svc = ProtocolService(config=env["config"], store=env["store"], gpu_lock=asyncio.Lock(), unload_asr=lambda: None)

    async def run():
        result = await svc.create_protocol(transcript_path=transcript, job_id="job-7", username="alice",
                                            options=ProtocolOptions(chunk_minutes=10))
        return await _wait_terminal(env["store"], result.protocol_job_id)

    job = asyncio.run(run())
    assert job["status"] == "completed"
    res = env["store"].get_result(job["id"])
    assert res["document_json"]["timestamp_refs"], "final document must reference source timestamps"
    ref = res["document_json"]["timestamp_refs"][0]
    assert ref["anchor"]
    decision = res["document_json"]["decisions"][0]
    assert decision["anchor"] == ref["anchor"]


def test_resume_after_restart_fails_stuck_jobs_without_relaunching(env):
    """Item: server restart recovery -- never leave a phantom GPU hold."""
    from protocol.service import ProtocolService
    env["job_store"].create(id="job-8", username="alice", filename="m.wav", settings_snapshot={})
    env["store"].update_model_state("qwen3-8b", local_path="/fake", installed=1)
    env["store"].create_protocol_job(id="stuck-1", job_id="job-8", username="alice", model_id="qwen3-8b")
    env["store"].update_protocol_job("stuck-1", status="processing_chunks")

    svc = ProtocolService(config=env["config"], store=env["store"], gpu_lock=asyncio.Lock(), unload_asr=lambda: None)
    n = svc.resume_after_restart()
    assert n == 1
    after = env["store"].get_protocol_job("stuck-1")
    assert after["status"] == "failed"
    assert "restart" in after["error"].lower()
