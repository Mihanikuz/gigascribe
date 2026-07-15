from pathlib import Path


def test_required_packages_are_declared():
    text = "\n".join(Path(p).read_text() for p in ["requirements-gigaam.txt", "requirements-models.txt", "requirements-cpu.txt", "requirements-cu128.txt"])
    for package in ["gigaam", "pyannote.audio", "torch", "torchaudio", "librosa", "soundfile"]:
        assert package in text


def test_cu128_pins_torch_27():
    text = Path("requirements-cu128.txt").read_text()
    assert "https://download.pytorch.org/whl/cu128" in text
    assert "torch==2.7.0" in text
    assert "torchaudio==2.7.0" in text
