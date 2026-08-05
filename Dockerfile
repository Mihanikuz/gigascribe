# syntax=docker/dockerfile:1.7
ARG INSTALL_PROTOCOL=0
# Empty this (--build-arg PROTOCOL_CMAKE_ARGS=) to skip the CUDA toolchain
# entirely and build a plain CPU wheel instead -- the officially-supported
# fallback if the CUDA build below doesn't work on your driver/CUDA
# combination (see README "GPU offload status and the CPU-wheel fallback").
#
# CMAKE_CUDA_ARCHITECTURES is set explicitly (not left to llama.cpp's own
# "native" default) because "native" probes an actual GPU device at CMake
# configure time -- which doesn't exist here: `docker compose build` runs
# this stage with no GPU passthrough (only `docker compose up`'s `gpus: all`
# reaches a running container), so "native" fails outright with "no GPU was
# detected". The list below matches requirements-cu128.txt's own torch
# build's supported architectures (confirmed via its reported arch_list),
# so a wheel built here targets the same hardware range torch already
# does, including sm_120 (RTX 5060 Ti / Blackwell).
ARG PROTOCOL_CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=70;75;80;86;90;100;120"

# =========================================================================
# protocol-builder-1: compiles llama-cpp-python, with CUDA GPU offload
# (GGML_CUDA) unless PROTOCOL_CMAKE_ARGS was emptied above.
#
# Only reached when INSTALL_PROTOCOL=1, via the ARG-selected `FROM
# protocol-builder-${INSTALL_PROTOCOL}` stage below -- Docker/BuildKit only
# builds whichever numbered stage that ARG actually names, so the default
# build (INSTALL_PROTOCOL=0) never touches this one at all.
#
# Base is explicitly python:3.11-slim-bookworm (Debian 12) -- pinned by
# codename, not the untagged/rolling python:3.11-slim, and matching the
# final image below for the same reason -- not an nvidia/cuda Ubuntu image:
# a wheel compiled against Ubuntu 24.04's newer glibc can fail to load on
# Debian bookworm's older one, so the CUDA devel toolchain (nvcc + headers)
# is installed here temporarily via NVIDIA's official Debian apt repo
# instead (only if actually needed for a CUDA build), then discarded --
# only the built wheel and the runtime .so files it needs are copied into
# the final image below.
#
# The codename pin matters beyond that: python:3.11-slim (untagged) tracks
# whatever Debian is currently "stable" and silently moved from bookworm to
# trixie at some point. Trixie's much newer glibc redeclares cospi/sinpi/
# cospif/sinpif (recent ISO/IEC TS 18661 additions) with a `noexcept` that
# conflicts with CUDA 12.8's own crt/math_functions.h declarations of the
# same names, which fails nvcc's compiler-id detection outright ("exception
# specification is incompatible with that of previous function") before any
# real compilation happens -- confirmed against real hardware after this
# was first written; see README's "Protocol module" GPU section.
# =========================================================================
FROM python:3.11-slim-bookworm AS protocol-builder-1
ARG PROTOCOL_CMAKE_ARGS
# NVIDIA's cuda-keyring (as of this writing) certifies its signing key with
# a SHA1 self-signature, which apt's sqv/sequoia backend on Debian trixie
# rejects outright since SHA1 was declared insecure for this purpose on
# 2026-02-01 ("Policy rejected non-revocation signature ... SHA1 is not
# considered secure"). This is an upstream NVIDIA packaging issue, not
# something fixable from this Dockerfile beyond scoping trust narrowly to
# just this one apt source (still fetched over HTTPS from
# developer.download.nvidia.com) rather than disabling verification globally.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates gnupg wget \
    && if [ -n "$PROTOCOL_CMAKE_ARGS" ]; then \
        wget -qO /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb \
        && dpkg -i /tmp/cuda-keyring.deb && rm -f /tmp/cuda-keyring.deb \
        && apt-get -o Acquire::AllowInsecureRepositories=true update \
        && apt-get install -y --no-install-recommends --allow-unauthenticated \
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
FROM python:3.11-slim-bookworm AS protocol-builder-0
RUN mkdir -p /wheels /usr/local/cuda/lib64

FROM protocol-builder-${INSTALL_PROTOCOL} AS protocol-builder

# =========================================================================
# main image
# =========================================================================
FROM python:3.11-slim-bookworm
ARG HOST_UID=1000
ARG HOST_GID=1000
ARG TORCH_REQUIREMENTS=requirements-cu128.txt
# Optional protocol (meeting-minutes LLM) module dependencies. 0 by default:
# the main image stays exactly as before, with no llama-cpp-python and no
# size increase. Build with --build-arg INSTALL_PROTOCOL=1 (see
# compose.protocol.yaml) to compile it in with CUDA offload support.
ARG INSTALL_PROTOCOL=0
ARG PROTOCOL_CMAKE_ARGS
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
            --find-links /tmp/protocol-wheels -r requirements-protocol.txt ; \
        case "$PROTOCOL_CMAKE_ARGS" in \
            *GGML_CUDA*) echo "Skipping build-time 'import llama_cpp' check: a CUDA-linked build needs libcuda.so.1 from the NVIDIA driver, which is only present once the container actually starts with GPU access (docker compose up's gpus: all), never during docker build. Verified for real the first time a protocol job loads a GGUF model." ;; \
            *) python -c "import llama_cpp; print('llama-cpp-python', llama_cpp.__version__)" ;; \
        esac \
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
