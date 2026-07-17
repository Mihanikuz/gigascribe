from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model_store as ms
import scripts.download_models as dm


def test_registry_contains_only_documented_model_ids():
    assert set(ms.SUPPORTED_GIGAAM_MODELS) == {"gigaam-v2-ctc", "gigaam-v3-ctc", "gigaam-v3-e2e-rnnt"}
    assert set(ms.SUPPORTED_DIARIZATION_MODELS) == {"none", "pyannote-3.1", "pyannote-community-1"}
    assert all("official_model_name" in model for model in ms.SUPPORTED_GIGAAM_MODELS.values())


def test_community_download_does_not_request_legacy_segmentation(tmp_path, monkeypatch):
    seen = []
    def snapshot_download(repo_id, **kwargs):
        seen.append(repo_id)
        target = Path(kwargs["local_dir"]); target.mkdir(parents=True, exist_ok=True); (target / "config.yaml").write_text("x")
    class Pipeline:
        @staticmethod
        def from_pretrained(*args, **kwargs): return object()
    monkeypatch.setitem(sys.modules, "huggingface_hub", type("Hub", (), {"snapshot_download": snapshot_download}))
    monkeypatch.setitem(sys.modules, "pyannote.audio", type("Audio", (), {"Pipeline": Pipeline}))
    dm.download_pyannote(tmp_path, "token", model_id="pyannote-community-1")
    assert ms.PYANNOTE_SEGMENTATION_REPO_ID not in seen
