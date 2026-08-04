# syntax=docker/dockerfile:1.7
ARG INSTALL_PROTOCOL=0
# Empty this (--build-arg PROTOCOL_CMAKE_ARGS=) to skip the CUDA toolchain
# entirely and build a plain CPU wheel instead -- the officially-supported
# fallback if the CUDA build below doesn't work on your driver/CUDA
# combination (see README "GPU offload status and the CPU-wheel fallback").
ARG PROTOCOL_CMAKE_ARGS=-DGGML_CUDA=on

# =========================================================================
# protocol-builder-1: compiles llama-cpp-python, with CUDA GPU offload
# (GGML_CUDA) unless PROTOCOL_CMAKE_ARGS was emptied above.
#
# Only reached when INSTALL_PROTOCOL=1, via the ARG-selected `FROM
# protocol-builder-${INSTALL_PROTOCOL}` stage below -- Docker/BuildKit only
# builds whichever numbered stage that ARG actually names, so the default
# build (INSTALL_PROTOCOL=0) never touches this one at all.
#
# Base is python:3.11-slim -- the SAME Debian (bookworm) base as the final
# image, deliberately, not an nvidia/cuda Ubuntu image: a wheel compiled
# against Ubuntu 24.04's newer glibc can fail to load on Debian bookworm's
# older glibc, so the CUDA devel toolchain (nvcc + headers) is installed
# here temporarily via NVIDIA's official Debian apt repo instead (only if
# actually needed for a CUDA build), then discarded -- only the built wheel
# and the runtime .so files it needs are copied into the final image below.
#
# NOT verified against real hardware in this environment (no GPU/Docker
# daemon available here) -- see README's "Protocol module" GPU section for
# how to confirm this and for the officially-supported CPU-wheel fallback
# if it doesn't work on a given driver/CUDA combination.
# =========================================================================
FROM python:3.11-slim AS protocol-builder-1
ARG PROTOCOL_CMAKE_ARGS
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates gnupg wget \
    && if [ -n "$PROTOCOL_CMAKE_ARGS" ]; then \
        wget -qO /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb \
        && dpkg -i /tmp/cuda-keyring.deb && rm -f /tmp/cuda-keyring.deb \
        && apt-get update && apt-get install -y --no-install-recommends \
            cuda-nvcc-12-8 cuda-cudart-dev-12-8 libcublas-dev-12-8 ; \
    fi \
    && mkdir -p /usr/local/cuda/lib64 \
    && rm -rf /var/lib/apt/lists/*
ENV PATH=/usr/local/cuda/bin:$PATH
WORKDIR /wheels
COPY requirements-protocol.txt ./
RUN CMAKE_ARGS="$PROTOCOL_CMAKE_ARGS" FORCE_CMAKE=1 \
    python -m pip wheel --no-cache-dir --wheel-dir /wheels -r requirements-protocol.txt

# No-op alias selected when INSTALL_PROTOCOL=0: mirrors the *paths* the
# real builder stage produces (left empty), so the unconditional COPY
# --from instructions in the final stage always have something valid to
# copy from, without ever pulling in the CUDA toolkit for the default build.
FROM python:3.11-slim AS protocol-builder-0
RUN mkdir -p /wheels /usr/local/cuda/lib64

FROM protocol-builder-${INSTALL_PROTOCOL} AS protocol-builder

# =========================================================================
# main image
# =========================================================================
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
    HF_HOME=/opt/gigascribe/models/huggingface HF_HUB_OFFLINE=1 PYTHONPATH=/app \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64
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
# vendor/gigaam is an empty placeholder (just .gitkeep) unless you clone/
# extract a local copy of https://github.com/salute-developers/GigaAM
# there before building -- see README "Restricted networks" for when a
# build host can't reach GitHub at all. Detected below by the presence of
# a setup.py/pyproject.toml; falls back to the pinned git commit in
# requirements-gigaam.txt otherwise, exactly as before.
COPY vendor/gigaam /app/vendor-gigaam
COPY requirements-base.txt requirements-gigaam.txt ./
RUN --mount=type=cache,id=gigascribe-pip-cache,target=/root/.cache/pip,sharing=locked \
    if [ -f /app/vendor-gigaam/setup.py ] || [ -f /app/vendor-gigaam/pyproject.toml ]; then \
        echo "Installing GigaAM from local vendor/gigaam (no GitHub access needed)" \
        && python -m pip install --timeout 600 --retries 20 \
            -c constraints.txt -r requirements-base.txt /app/vendor-gigaam ; \
    else \
        python -m pip install --timeout 600 --retries 20 \
            -c constraints.txt -r requirements-base.txt -r requirements-gigaam.txt ; \
    fi \
    && python -m pip check \
    && rm -rf /app/vendor-gigaam
COPY requirements-protocol.txt ./
# Only reached when INSTALL_PROTOCOL=1: installs the llama-cpp-python wheel
# protocol-builder-1 compiled with CUDA GPU offload (GGML_CUDA), so it can
# serve the protocol module's GGUF models on the same CUDA 12.8 toolchain
# torch was built against (RTX 5060 Ti and other CUDA 12.8-capable cards).
# With the default INSTALL_PROTOCOL=0 this step is a no-op and
# requirements-protocol.txt is never installed, so the main image's size
# and dependency set are unaffected.
COPY --from=protocol-builder /wheels /tmp/protocol-wheels
COPY --from=protocol-builder /usr/local/cuda/lib64 /usr/local/cuda/lib64
RUN --mount=type=cache,id=gigascribe-pip-cache,target=/root/.cache/pip,sharing=locked \
    if [ "$INSTALL_PROTOCOL" = "1" ]; then \
        python -m pip install --timeout 600 --retries 20 \
            --find-links /tmp/protocol-wheels -r requirements-protocol.txt \
        && python -c "import llama_cpp; print('llama-cpp-python', llama_cpp.__version__)" ; \
    else \
        echo "INSTALL_PROTOCOL=0: skipping protocol module dependencies" ; \
    fi \
    && rm -rf /tmp/protocol-wheels
COPY . .
RUN apt-get update && apt-get install -y --no-install-recommends gosu && rm -rf /var/lib/apt/lists/* \
    && mkdir -p "$GIGASCRIBE_DATA_DIR/uploads" "$GIGASCRIBE_DATA_DIR/cache/matplotlib" "$GIGASCRIBE_DATA_DIR/cache/fontconfig" "$GIGASCRIBE_MODELS_DIR" \
    && chmod +x /app/scripts/container-entrypoint.sh
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD curl -fsS http://127.0.0.1:8000/health/ready >/dev/null || exit 1
ENTRYPOINT ["/app/scripts/container-entrypoint.sh"]
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
