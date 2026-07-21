import types
import pytest
import model_store as ms
from diarization_backend import normalize_diarization, map_speakers_to_asr

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
