import asyncio, importlib, json, sys, types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model_store as ms
import scripts.download_models as dm

def stub_app_imports(monkeypatch):
    monkeypatch.setitem(sys.modules, "gradio", types.SimpleNamespace(Progress=lambda: None))
    monkeypatch.setitem(sys.modules, "numpy", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "gigaam", types.SimpleNamespace(load_model=lambda *a, **k: None))

class FakeModel:
    def transcribe(self, *a): return "ok"

def ready_gigaam(base, model="v2_ctc", data=b"ckpt"):
    ckpt = ms.gigaam_checkpoint_path(base, model)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(data + b"x" * max(0, ms.MIN_CHECKPOINT_BYTES - len(data)))
    ms.write_gigaam_marker(base, model)
    return ckpt


def test_gigaam_saved_in_cache_and_load_model_gets_download_root(tmp_path, monkeypatch):
    calls=[]
    def load_model(name, device="cpu", download_root=None):
        calls.append((name, device, download_root))
        p=Path(download_root)/"v2_ctc.ckpt"; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b"x" * ms.MIN_CHECKPOINT_BYTES)
        return FakeModel()
    monkeypatch.setitem(sys.modules,"gigaam", types.SimpleNamespace(load_model=load_model))
    dm.warm_up_gigaam(tmp_path,"v2_ctc")
    assert len(calls) == 1 and calls[0][:2] == ("v2_ctc", "cpu") and calls[0][2].endswith("/gigaam-cache")
    assert (tmp_path/"gigaam-cache"/"v2_ctc.ckpt").is_file()


def test_repeated_download_skips_correct_checkpoint(tmp_path, monkeypatch):
    ready_gigaam(tmp_path)
    monkeypatch.setitem(sys.modules,"gigaam", types.SimpleNamespace(load_model=lambda *a,**k: pytest.fail("called")))
    dm.warm_up_gigaam(tmp_path,"v2_ctc")


def test_marker_without_checkpoint_not_ready(tmp_path):
    (tmp_path/"gigaam"/"v2_ctc").mkdir(parents=True)
    (tmp_path/"gigaam"/"v2_ctc"/ms.GIGAAM_MARKER_NAME).write_text(json.dumps({"model_name":"v2_ctc","cache_dir":str(ms.gigaam_cache_dir(tmp_path)),"checkpoint":str(ms.gigaam_checkpoint_path(tmp_path,"v2_ctc")),"checkpoint_size":1}))
    assert not ms.is_gigaam_ready(tmp_path,"v2_ctc")


def test_empty_or_corrupt_checkpoint_not_ready(tmp_path):
    ckpt=ready_gigaam(tmp_path, data=b"ok")
    ckpt.write_bytes(b"")
    assert not ms.is_gigaam_ready(tmp_path,"v2_ctc")
    ckpt.write_bytes(b"bad")
    assert not ms.is_gigaam_ready(tmp_path,"v2_ctc")


def test_force_redownloads_and_replaces_atomically(tmp_path, monkeypatch):
    old=ready_gigaam(tmp_path, data=b"old")
    def load_model(name, device="cpu", download_root=None):
        p=Path(download_root)/"v2_ctc.ckpt"; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b"new" + b"x" * ms.MIN_CHECKPOINT_BYTES)
        return FakeModel()
    monkeypatch.setitem(sys.modules,"gigaam", types.SimpleNamespace(load_model=load_model))
    dm.warm_up_gigaam(tmp_path,"v2_ctc",force=True)
    assert old.read_bytes().startswith(b"new")


def test_force_error_keeps_working_model(tmp_path, monkeypatch):
    old=ready_gigaam(tmp_path, data=b"old")
    def fail(*a,**k): raise RuntimeError("boom")
    monkeypatch.setitem(sys.modules,"gigaam", types.SimpleNamespace(load_model=fail))
    with pytest.raises(RuntimeError): dm.warm_up_gigaam(tmp_path,"v2_ctc",force=True)
    assert old.read_bytes().startswith(b"old") and ms.is_gigaam_ready(tmp_path,"v2_ctc")


def test_runtime_without_checkpoint_fails_before_network(tmp_path, monkeypatch):
    monkeypatch.setenv("GIGASCRIBE_MODELS_DIR", str(tmp_path))
    stub_app_imports(monkeypatch)
    import app; importlib.reload(app)
    monkeypatch.setattr(app, "GIGAAM_AVAILABLE", True)
    monkeypatch.setattr(app.gigaam, "load_model", lambda *a,**k: pytest.fail("network/loader called"))
    with pytest.raises(RuntimeError): asyncio.run(app.AudioProcessor().initialize_models())


def test_runtime_with_checkpoint_no_network_and_download_root(tmp_path, monkeypatch):
    ready_gigaam(tmp_path)
    monkeypatch.setenv("GIGASCRIBE_MODELS_DIR", str(tmp_path))
    stub_app_imports(monkeypatch)
    import app; importlib.reload(app)
    calls=[]
    monkeypatch.setattr(app, "GIGAAM_AVAILABLE", True)
    monkeypatch.setattr(app, "PYANNOTE_AVAILABLE", False)
    monkeypatch.setattr(app.gigaam, "load_model", lambda n, device, download_root: calls.append(download_root) or FakeModel())
    asyncio.run(app.AudioProcessor().initialize_models())
    assert calls == [str(ms.gigaam_cache_dir(tmp_path))]


def test_pyannote_marker_only_after_offline_check(tmp_path, monkeypatch):
    created=[]
    def snap(repo_id, **kw):
        if repo_id == ms.PYANNOTE_REPO_ID:
            target=Path(kw["local_dir"]); target.mkdir(parents=True, exist_ok=True); (target/"config.yaml").write_text("x")
    class P:
        @staticmethod
        def from_pretrained(path, **kw): created.append((path, kw)); return object()
    monkeypatch.setitem(sys.modules,"huggingface_hub", types.SimpleNamespace(snapshot_download=snap))
    monkeypatch.setitem(sys.modules,"pyannote.audio", types.SimpleNamespace(Pipeline=P))
    dm.download_pyannote(tmp_path,"tok")
    assert ms.is_pyannote_ready(tmp_path) and len(created)==2


def test_incomplete_pyannote_not_ready_and_ready_needs_no_token(tmp_path):
    target=ms.pyannote_target(tmp_path); target.mkdir(parents=True); (target/"config.yaml").write_text("x")
    assert not ms.is_pyannote_ready(tmp_path)
    ms.write_pyannote_marker(tmp_path)
    assert dm.download_pyannote(tmp_path, None) == target
