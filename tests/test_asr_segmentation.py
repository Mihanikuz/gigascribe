import asyncio
import importlib
import logging
import os
import sys
import types
import wave
from pathlib import Path


def load_app(tmp_path, monkeypatch):
    monkeypatch.setenv("GIGASCRIBE_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("GIGASCRIBE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setitem(sys.modules, "gradio", types.SimpleNamespace(Progress=lambda: None))
    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "gigaam", types.SimpleNamespace())
    import app
    return importlib.reload(app)


def wav(path, seconds=0.1):
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1); out.setsampwidth(2); out.setframerate(16000)
        out.writeframes(b"\0\0" * int(seconds * 16000))


def test_long_interval_is_split_with_timestamps_and_remainder(tmp_path, monkeypatch):
    app = load_app(tmp_path, monkeypatch)
    processor = app.AudioProcessor({"asr_model": "gigaam-v2-ctc", "diarization_model": "none", "device": "cpu"})
    source = tmp_path / "source.wav"; wav(source)
    def extract(_, output, start, duration):
        wav(output); return True
    processor.extract_audio_segment = extract
    pieces = processor.prepare_asr_segments(str(source), str(tmp_path), [{"start": 10, "end": 59, "speaker": "SPEAKER_2"}])
    assert [(p["start"], p["end"]) for p in pieces] == [(10, 34), (34, 58), (58, 59)]
    assert all(piece["duration"] <= app.MAX_DURATION_CHUNK for piece in pieces)
    assert pieces[-1]["duration"] == 1


def test_segmented_transcription_uses_normal_api_and_keeps_successes(tmp_path, monkeypatch, caplog):
    app = load_app(tmp_path, monkeypatch)
    class Model:
        def __init__(self): self.calls = []; self.longform_calls = 0
        def transcribe(self, path):
            self.calls.append(path)
            if path.endswith("bad.wav"): raise ValueError("bad signal")
            return "ok"
        def transcribe_longform(self, path): self.longform_calls += 1; raise AssertionError("must not be called")
    processor = app.AudioProcessor({"asr_model": "gigaam-v2-ctc", "diarization_model": "none", "device": "cpu"})
    processor.gigaam_model = Model()
    good, bad = tmp_path / "good.wav", tmp_path / "bad.wav"; wav(good); wav(bad)
    with caplog.at_level(logging.ERROR):
        result = asyncio.run(processor.transcribe_audio_batch([str(good), str(bad)]))
    assert result[0] == "ok" and "bad signal" in result[1]
    assert processor.gigaam_model.longform_calls == 0
    assert len(processor.gigaam_model.calls) == 2
    assert "Traceback" in caplog.text and "duration=" in caplog.text and "model=v2_ctc" in caplog.text


def test_cpu_requirements_and_log_file_are_cpu_safe(tmp_path, monkeypatch):
    app = load_app(tmp_path, monkeypatch)
    requirements = Path("requirements-cpu.txt").read_text()
    assert "--index-url https://download.pytorch.org/whl/cpu" in requirements
    assert "torch==2.5.1+cpu" in requirements
    assert str(app.DATA_DIR / "transcription_app.log") in [getattr(h, "baseFilename", "") for h in app.logger.handlers]
