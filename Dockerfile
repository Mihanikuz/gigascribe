# syntax=docker/dockerfile:1.7
FROM python:3.11-slim
ARG HOST_UID=1000
ARG HOST_GID=1000
ARG TORCH_REQUIREMENTS=requirements-cu128.txt
# Optional protocol (meeting-minutes LLM) module dependencies. 0 by default:
# the main image stays exactly as before, with no llama-cpp-python and no
# size increase. Build with --build-arg INSTALL_PROTOCOL=1 (see
# compose.protocol.yaml) to compile it in with CUDA offload support.
ARG INSTALL_PROTOCOL=0
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_CACHE_DIR=/root/.cache/pip PIP_NO_CACHE_DIR=0 \
    PIP_DEFAULT_TIMEOUT=600 PIP_RETRIES=20 \
    GIGASCRIBE_DATA_DIR=/opt/gigascribe/data GIGASCRIBE_MODELS_DIR=/opt/gigascribe/models \
    HF_HOME=/opt/gigascribe/models/huggingface HF_HUB_OFFLINE=1 PYTHONPATH=/app
ENV HOST_UID=${HOST_UID} HOST_GID=${HOST_GID}
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg build-essential git curl && rm -rf /var/lib/apt/lists/*
COPY requirements-cu128.txt ./
RUN --mount=type=cache,id=gigascribe-pip-cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install --upgrade pip --timeout 600 --retries 20 \
    && python -m pip install --timeout 600 --retries 20 -r "${TORCH_REQUIREMENTS}"
COPY requirements-models.txt constraints.txt ./
RUN --mount=type=cache,id=gigascribe-pip-cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install --timeout 600 --retries 20 \
        -c constraints.txt -r requirements-models.txt
COPY requirements-base.txt requirements-gigaam.txt ./
RUN --mount=type=cache,id=gigascribe-pip-cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install --timeout 600 --retries 20 \
        -c constraints.txt -r requirements-base.txt -r requirements-gigaam.txt \
    && python -m pip check
COPY requirements-protocol.txt ./
# Only reached when INSTALL_PROTOCOL=1: compiles llama-cpp-python with CUDA
# GPU offload (GGML_CUDA) so it can serve the protocol module's GGUF models
# on the same CUDA 12.8 toolchain torch was built against (RTX 5060 Ti and
# other CUDA 12.8-capable cards). With the default INSTALL_PROTOCOL=0 this
# step is a no-op and requirements-protocol.txt is never installed, so the
# main image's size and dependency set are unaffected.
RUN --mount=type=cache,id=gigascribe-pip-cache,target=/root/.cache/pip,sharing=locked \
    if [ "$INSTALL_PROTOCOL" = "1" ]; then \
        CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 python -m pip install --timeout 600 --retries 20 \
            -r requirements-protocol.txt ; \
    else \
        echo "INSTALL_PROTOCOL=0: skipping protocol module dependencies" ; \
    fi
COPY . .
RUN apt-get update && apt-get install -y --no-install-recommends gosu && rm -rf /var/lib/apt/lists/* \
    && mkdir -p "$GIGASCRIBE_DATA_DIR/uploads" "$GIGASCRIBE_DATA_DIR/cache/matplotlib" "$GIGASCRIBE_MODELS_DIR" \
    && chmod +x /app/scripts/container-entrypoint.sh
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD curl -fsS http://127.0.0.1:8000/health/ready >/dev/null || exit 1
ENTRYPOINT ["/app/scripts/container-entrypoint.sh"]
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
