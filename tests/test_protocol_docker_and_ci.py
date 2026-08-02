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
    # CUDA build flags are baked into the Dockerfile itself (item 3) -- the
    # user never has to set CMAKE_ARGS/FORCE_CMAKE by hand.
    assert "FORCE_CMAKE=1" in dockerfile


def test_install_protocol_1_verifies_llama_cpp_importable_after_install():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    # item 3: the build itself must fail if llama_cpp isn't importable
    # afterwards, rather than silently succeeding without it.
    assert 'import llama_cpp' in dockerfile
    install_pos = dockerfile.index("requirements-protocol.txt")
    check_pos = dockerfile.index("import llama_cpp")
    assert install_pos < check_pos


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


def test_ci_confirms_default_image_excludes_llama_cpp():
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "import llama_cpp" in ci
    assert "must NOT be importable" in ci


def test_download_models_script_runs_before_docker_build_in_documented_order():
    script = REPO_ROOT / "scripts" / "download-models.sh"
    assert script.is_file()
    assert script.stat().st_mode & 0o111, "download-models.sh must be executable"
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    download_pos = readme.index("download-models.sh")
    build_pos = readme.index("docker compose build")
    assert download_pos < build_pos, "README must document downloading models before building"
