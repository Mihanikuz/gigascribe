import json, types
from pathlib import Path
import pytest

import model_store as ms
from diarization_backend import DiarizationBackend, normalize_diarization, map_speakers_to_asr
from model_manager import ModelManager


def test_only_rnnt_and_cuda_supported():
    assert tuple(ms.SUPPORTED_GIGAAM_MODELS) == (ms.ASR_MODEL_ID,)
    assert ms.SUPPORTED_DEVICES == ("cuda",)
    assert all(m["devices"] == ["cuda"] for m in ms.SUPPORTED_GIGAAM_MODELS.values())


def test_community_1_local_path():
    assert ms.pyannote_target_for(Path("models"), ms.DIARIZATION_MODEL_ID) == Path("models/pyannote/community-1")


def test_local_pyannote_load_uses_directory_not_hf_repo(monkeypatch, tmp_path):
    target = ms.pyannote_target_for(tmp_path, ms.DIARIZATION_MODEL_ID)
    target.mkdir(parents=True)
    (target / "config.yaml").write_text("pipeline: dummy\n")
    ms.write_pyannote_marker(tmp_path, ms.DIARIZATION_MODEL_ID)
    seen = {}
    class DummyPipeline:
        def to(self, device): self.device = device; return self
    class DummyFactory:
        @staticmethod
        def from_pretrained(value): seen["value"] = value; return DummyPipeline()
    monkeypatch.setitem(__import__('sys').modules, 'pyannote', types.SimpleNamespace())
    monkeypatch.setitem(__import__('sys').modules, 'pyannote.audio', types.SimpleNamespace(Pipeline=DummyFactory))
    monkeypatch.setitem(__import__('sys').modules, 'torch', types.SimpleNamespace(device=lambda d: types.SimpleNamespace(type=d)))
    DiarizationBackend(str(tmp_path)).load(ms.DIARIZATION_MODEL_ID, "cuda")
    assert seen["value"] == str(target)
    assert not seen["value"].startswith("pyannote/")
    assert not seen["value"].endswith("config.yaml")
    assert seen["value"].endswith("community-1")


def _ann(*rows):
    class Turn:
        def __init__(self, s, e): self.start=s; self.end=e
    class Ann:
        def itertracks(self, yield_label=True):
            for s,e,l in rows: yield Turn(s,e), None, l
    return Ann()


def test_normalizes_speaker_diarization_and_annotation():
    assert normalize_diarization(types.SimpleNamespace(speaker_diarization=_ann((0,1,"SPEAKER_00"))))[0]["speaker"] == "SPEAKER_00"
    assert normalize_diarization(_ann((1,2,"SPEAKER_01")))[0] == {"start":1.0,"end":2.0,"speaker":"SPEAKER_01"}


def test_speaker_display_order_starts_at_one():
    mapped = map_speakers_to_asr(
        [{"start":0,"end":1,"text":"a"},{"start":1,"end":2,"text":"b"}],
        [{"start":0,"end":1,"speaker":"SPEAKER_00"},{"start":1,"end":2,"speaker":"SPEAKER_03"}],
    )
    assert [m["speaker"] for m in mapped] == ["Спикер 1", "Спикер 2"]
    assert "Спикер 00" not in json.dumps(mapped, ensure_ascii=False)


def test_no_overlap_does_not_default_to_speaker_00():
    mapped = map_speakers_to_asr([{"start":10,"end":11,"text":"x"}], [{"start":0,"end":1,"speaker":"SPEAKER_00"}])
    assert mapped[0]["speaker"] is None
    assert mapped[0]["speaker_cluster"] is None


def test_model_manager_unloads_diarization_when_settings_change(monkeypatch, tmp_path):
    m = ModelManager(tmp_path); calls=[]
    monkeypatch.setattr(m, 'check_cuda', lambda: {"available": True})
    monkeypatch.setattr('model_manager.assert_gigaam_ready', lambda *a, **k: None)
    class A: 
        def load(self,*a): pass
    class D:
        def load(self,*a): pass
        def unload(self): calls.append('unload')
    monkeypatch.setattr('model_manager.ASRBackend', lambda *a: A())
    monkeypatch.setattr('model_manager.is_pyannote_ready', lambda *a: True)
    monkeypatch.setattr('model_manager.DiarizationBackend', lambda *a: D())
    m.ensure_loaded({"asr_model":ms.ASR_MODEL_ID,"diarization_model":ms.DIARIZATION_MODEL_ID,"device":"cuda"})
    m.ensure_loaded({"asr_model":ms.ASR_MODEL_ID,"diarization_model":"none","device":"cuda"})
    assert calls == ['unload']


def test_health_not_ready_when_required_model_missing(monkeypatch, tmp_path):
    m = ModelManager(tmp_path)
    monkeypatch.setattr(m, 'check_cuda', lambda: {"available": True, "device": "GPU"})
    status = m.health_status(settings={"asr_model":ms.ASR_MODEL_ID,"diarization_model":ms.DIARIZATION_MODEL_ID,"device":"cuda"})
    assert status["status"] == "not_ready"
    assert status["diarization"]["ready"] is False
