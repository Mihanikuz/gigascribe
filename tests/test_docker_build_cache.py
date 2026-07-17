from pathlib import Path


DOCKERFILE = Path(__file__).parents[1] / "Dockerfile"


def test_pip_installs_use_a_shared_buildkit_cache_and_network_retries():
    dockerfile = DOCKERFILE.read_text()

    assert "# syntax=docker/dockerfile:1.7" in dockerfile
    assert "PIP_CACHE_DIR=/root/.cache/pip" in dockerfile
    assert "PIP_DEFAULT_TIMEOUT=600 PIP_RETRIES=20" in dockerfile
    assert "--resume-retries" not in dockerfile
    assert dockerfile.count("--mount=type=cache,id=gigascribe-pip-cache,target=/root/.cache/pip,sharing=locked") == 3
    assert "--resume-retries" not in dockerfile


def test_dependency_layers_precede_application_source_copy():
    dockerfile = DOCKERFILE.read_text()

    torch_install = dockerfile.index('-r "${TORCH_REQUIREMENTS}"')
    model_install = dockerfile.index("-r requirements-models.txt")
    base_install = dockerfile.index("-r requirements-base.txt")
    gigaam_install = dockerfile.index("--no-deps gigaam==0.1.0")
    source_copy = dockerfile.index("COPY . .")

    assert torch_install < model_install < base_install < gigaam_install < source_copy
