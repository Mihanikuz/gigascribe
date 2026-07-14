import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model_store as ms
import scripts.download_models as dm


def _ready(base):
    ckpt=ms.gigaam_checkpoint_path(base,"v2_ctc"); ckpt.parent.mkdir(parents=True, exist_ok=True); ckpt.write_bytes(b"ok")
    ms.write_gigaam_marker(base,"v2_ctc")


def test_repeated_download_skips_ready_gigaam(tmp_path):
    _ready(tmp_path)
    assert dm.warm_up_gigaam(tmp_path, "v2_ctc") == tmp_path / "gigaam" / "v2_ctc"


def test_missing_hf_token_errors_when_pyannote_not_ready(tmp_path):
    try:
        dm.download_pyannote(tmp_path, None)
    except RuntimeError as exc:
        assert "HF_TOKEN" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
