"""Items 23-25: Docker build gating and CI FFmpeg fix.

No Docker daemon is available in this environment, so these check the
actual Dockerfile/compose/workflow content that governs the build rather
than running docker build itself -- see the final report for a note that a
real `docker compose build` (both variants) still needs to be run once on a
machine with Docker/GPU access to fully confirm this end to end.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_install_protocol_defaults_to_0_and_gates_the_pip_install():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "ARG INSTALL_PROTOCOL=0" in dockerfile
    assert 'if [ "$INSTALL_PROTOCOL" = "1" ]' in dockerfile
    assert "requirements-protocol.txt" in dockerfile
    # the base torch/model/app installs must not depend on the flag at all
    assert "ARG TORCH_REQUIREMENTS" in dockerfile


def test_install_protocol_1_builds_llama_cpp_python_with_cuda():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "GGML_CUDA=on" in dockerfile
    assert "llama-cpp-python" in (REPO_ROOT / "requirements-protocol.txt").read_text(encoding="utf-8")


def test_compose_protocol_override_sets_build_arg_and_env_var():
    override = (REPO_ROOT / "compose.protocol.yaml").read_text(encoding="utf-8")
    assert 'INSTALL_PROTOCOL: "1"' in override
    assert 'GIGASCRIBE_PROTOCOL_ENABLED: "1"' in override
    # base compose.yaml must remain independently usable (no protocol changes)
    base = (REPO_ROOT / "compose.yaml").read_text(encoding="utf-8")
    assert "INSTALL_PROTOCOL" not in base
    assert "GIGASCRIBE_PROTOCOL_ENABLED" not in base


def test_ci_installs_ffmpeg_before_pytest_without_continue_on_error():
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "apt-get install -y ffmpeg" in ci
    assert "continue-on-error" not in ci
    ffmpeg_pos = ci.index("apt-get install -y ffmpeg")
    pytest_pos = ci.index("run: pytest -q")
    assert ffmpeg_pos < pytest_pos, "FFmpeg must be installed before the Pytest step"
