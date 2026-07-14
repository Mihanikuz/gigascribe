import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.download_models as dm


def test_repeated_download_skips_ready_gigaam(tmp_path, monkeypatch):
    model = tmp_path / "gigaam" / "v2_ctc"
    model.mkdir(parents=True)
    (model / dm.GIGAAM_MARKER_NAME).write_text('{"model_name":"v2_ctc"}')
    called = False
    def fail_import(*a, **k):
        nonlocal called
        called = True
    assert dm.warm_up_gigaam(tmp_path, "v2_ctc") == model
    assert not called


def test_force_warms_gigaam_again(tmp_path, monkeypatch):
    model = tmp_path / "gigaam" / "v2_ctc"
    model.mkdir(parents=True)
    (model / dm.GIGAAM_MARKER_NAME).write_text('{"model_name":"v2_ctc"}')
    class FakeModel:
        def transcribe(self): pass
    class FakeGiga:
        @staticmethod
        def load_model(name, device="cpu"):
            FakeGiga.called = True
            return FakeModel()
    FakeGiga.called = False
    monkeypatch.setitem(sys.modules, "gigaam", FakeGiga)
    dm.warm_up_gigaam(tmp_path, "v2_ctc", force=True)
    assert FakeGiga.called


def test_missing_hf_token_errors_when_pyannote_not_ready(tmp_path):
    try:
        dm.download_pyannote(tmp_path, None)
    except RuntimeError as exc:
        assert "HF_TOKEN" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
