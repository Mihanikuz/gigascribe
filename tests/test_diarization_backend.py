from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diarization_backend import normalize_diarization
from workers.pyannote_legacy_worker import normalize as normalize_legacy


class Turn:
    def __init__(self, start, end): self.start, self.end = start, end


class Annotation:
    def itertracks(self, yield_label=False):
        assert yield_label
        return iter([(Turn(1, 2.5), "track", "SPEAKER_00")])


def test_normalizes_legacy_annotation():
    assert normalize_diarization(Annotation()) == [{"start": 1.0, "end": 2.5, "speaker": "SPEAKER_00", "duration": 1.5}]


def test_normalizes_community_and_prefers_exclusive_result():
    class Output:
        speaker_diarization = Annotation()
        exclusive_speaker_diarization = Annotation()
    assert normalize_diarization(Output())[0]["speaker"] == "SPEAKER_00"


def test_legacy_worker_returns_diagnostics_for_annotation():
    segments, diagnostics = normalize_legacy(Annotation())
    assert segments[0]["duration"] == 1.5
    assert diagnostics == {"speaker_count": 1, "segment_count": 1, "speaker_durations": {"SPEAKER_00": 1.5}}
