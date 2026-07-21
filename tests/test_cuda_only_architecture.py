import asyncio
import sys
import types
import pytest
import model_store as ms
from diarization_backend import normalize_diarization, map_speakers_to_asr


def _stub_app_imports(monkeypatch):
    class FakeCuda:
        OutOfMemoryError = RuntimeError
        def is_available(self): return True
        def get_device_name(self, *a): return "GPU"
        def get_device_properties(self, *a): return types.SimpleNamespace(total_memory=16 * 1024 ** 3)
        def empty_cache(self): pass
    monkeypatch.setitem(sys.modules, "gradio", types.SimpleNamespace(Progress=lambda: (lambda *a, **k: None)))
    monkeypatch.setitem(sys.modules, "numpy", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=FakeCuda(), device=lambda name: types.SimpleNamespace(type=name), ones=lambda *a, **k: object()))

def test_registry_contains_only_rnnt_and_cuda():
    assert set(ms.SUPPORTED_GIGAAM_MODELS) == {"gigaam-v3-e2e-rnnt"}
    assert all("cpu" not in m["devices"] for m in ms.SUPPORTED_GIGAAM_MODELS.values())
    assert ms.load_settings()["device"] == "cuda"

def test_compose_contains_only_gpu_service():
    text=open("compose.yaml", encoding="utf-8").read()
    assert "  gigascribe-gpu:" in text
    assert "gigascribe-cpu" not in text and "profiles:" not in text
    assert "GIGASCRIBE_DEVICE: cuda" in text and "HF_HUB_OFFLINE: \"1\"" in text

def test_pyannote_4_output_normalizes_exclusive():
    class Turn:
        def __init__(self,s,e): self.start=s; self.end=e
    class Ann:
        def itertracks(self, yield_label=True):
            yield Turn(0,1), None, "SPEAKER_00"; yield Turn(1,2), None, "SPEAKER_01"
    out=types.SimpleNamespace(exclusive_speaker_diarization=Ann(), speaker_diarization=None)
    assert normalize_diarization(out) == [{"start":0.0,"end":1.0,"speaker":"SPEAKER_00"},{"start":1.0,"end":2.0,"speaker":"SPEAKER_01"}]

def test_empty_diarization_is_error():
    class Ann:
        def itertracks(self, yield_label=True): return iter(())
    with pytest.raises(RuntimeError, match="DIARIZATION_EMPTY_RESULT"):
        normalize_diarization(Ann())

def test_rnnt_segments_map_to_multiple_speakers():
    asr=[{"start":0,"end":1,"text":"a"},{"start":1,"end":2,"text":"b"}]
    dia=[{"start":0,"end":1,"speaker":"SPEAKER_00"},{"start":1,"end":2,"speaker":"SPEAKER_01"}]
    mapped=map_speakers_to_asr(asr,dia)
    assert [m["speaker"] for m in mapped] == ["Спикер 1", "Спикер 2"]


def test_audio_processor_uses_passed_snapshot(monkeypatch):
    _stub_app_imports(monkeypatch)
    import app
    snapshot={"asr_model": ms.ASR_MODEL_ID, "diarization_model":"none", "device":"cuda"}
    processor=app.AudioProcessor(snapshot=snapshot, model_manager=object())
    assert processor.snapshot == snapshot
    assert processor.snapshot is not snapshot


def test_audio_processor_snapshot_is_immutable_from_settings_file(monkeypatch, tmp_path):
    _stub_app_imports(monkeypatch)
    import app
    snapshot={"asr_model": ms.ASR_MODEL_ID, "diarization_model":"none", "device":"cuda"}
    processor=app.AudioProcessor(snapshot=snapshot, model_manager=object())
    ms.save_settings({"diarization_model": ms.DIARIZATION_MODEL_ID}, tmp_path)
    assert processor.snapshot["diarization_model"] == "none"


def test_model_manager_configure_none_returns_no_diarization(monkeypatch, tmp_path):
    from model_manager import ModelManager
    m=ModelManager(tmp_path)
    monkeypatch.setattr(m, 'check_cuda', lambda: {"available": True})
    monkeypatch.setattr('model_manager.assert_gigaam_ready', lambda *a, **k: None)
    class A:
        model=object()
        def load(self,*a): pass
    monkeypatch.setattr('model_manager.ASRBackend', lambda *a: A())
    cfg=m.configure({"asr_model":ms.ASR_MODEL_ID,"diarization_model":"none","device":"cuda"})
    assert cfg.diarization_backend is None
    assert cfg.asr_backend is m.asr


def test_model_manager_configure_community_reuses_pipeline_and_rnnt(monkeypatch, tmp_path):
    from model_manager import ModelManager
    m=ModelManager(tmp_path); asr_loads=[]; dia_loads=[]
    monkeypatch.setattr(m, 'check_cuda', lambda: {"available": True})
    monkeypatch.setattr('model_manager.assert_gigaam_ready', lambda *a, **k: None)
    monkeypatch.setattr('model_manager.is_pyannote_ready', lambda *a: True)
    class A:
        model=object()
        def load(self,*a): asr_loads.append(a)
    class D:
        pipeline=object()
        def load(self,*a): dia_loads.append(a)
        def unload(self): pass
    monkeypatch.setattr('model_manager.ASRBackend', lambda *a: A())
    monkeypatch.setattr('model_manager.DiarizationBackend', lambda *a: D())
    snap={"asr_model":ms.ASR_MODEL_ID,"diarization_model":ms.DIARIZATION_MODEL_ID,"device":"cuda"}
    first=m.configure(snap); second=m.configure(snap)
    assert first.diarization_backend is second.diarization_backend is m.diarization
    assert len(dia_loads) == 1
    assert len(asr_loads) == 1


def test_model_manager_configure_community_to_none_unloads_diarization_without_reloading_rnnt(monkeypatch, tmp_path):
    from model_manager import ModelManager
    m=ModelManager(tmp_path); asr_loads=[]; unloads=[]
    monkeypatch.setattr(m, 'check_cuda', lambda: {"available": True})
    monkeypatch.setattr('model_manager.assert_gigaam_ready', lambda *a, **k: None)
    monkeypatch.setattr('model_manager.is_pyannote_ready', lambda *a: True)
    class A:
        model=object()
        def load(self,*a): asr_loads.append(a)
    class D:
        pipeline=object()
        def load(self,*a): pass
        def unload(self): unloads.append(True)
    monkeypatch.setattr('model_manager.ASRBackend', lambda *a: A())
    monkeypatch.setattr('model_manager.DiarizationBackend', lambda *a: D())
    m.configure({"asr_model":ms.ASR_MODEL_ID,"diarization_model":ms.DIARIZATION_MODEL_ID,"device":"cuda"})
    cfg=m.configure({"asr_model":ms.ASR_MODEL_ID,"diarization_model":"none","device":"cuda"})
    assert cfg.diarization_backend is None
    assert unloads == [True]
    assert len(asr_loads) == 1


def test_gpu_job_lock_serializes_two_jobs(monkeypatch):
    _stub_app_imports(monkeypatch)
    import app
    async def run():
        inside = 0
        max_inside = 0
        async def job():
            nonlocal inside, max_inside
            async with app.GPU_JOB_LOCK:
                inside += 1
                max_inside = max(max_inside, inside)
                await asyncio.sleep(0.01)
                inside -= 1
        await asyncio.gather(job(), job())
        assert max_inside == 1
    asyncio.run(run())
