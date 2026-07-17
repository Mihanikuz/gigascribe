# syntax=docker/dockerfile:1.7
FROM python:3.11-slim
ARG HOST_UID=1000
ARG HOST_GID=1000
ARG TORCH_REQUIREMENTS=requirements-cpu.txt
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_CACHE_DIR=/root/.cache/pip PIP_NO_CACHE_DIR=0 \
    PIP_DEFAULT_TIMEOUT=600 PIP_RETRIES=20 \
    GIGASCRIBE_DATA_DIR=/opt/gigascribe/data GIGASCRIBE_MODELS_DIR=/opt/gigascribe/models \
    HF_HOME=/opt/gigascribe/models/huggingface HF_HUB_OFFLINE=1 PYTHONPATH=/app
ENV HOST_UID=${HOST_UID} HOST_GID=${HOST_GID}
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg build-essential git curl && rm -rf /var/lib/apt/lists/*
COPY requirements-cpu.txt requirements-cu128.txt ./
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
    && python -m pip install --timeout 600 --retries 20 \
        --no-deps gigaam==0.1.0
COPY . .
RUN apt-get update && apt-get install -y --no-install-recommends gosu && rm -rf /var/lib/apt/lists/* \
    && mkdir -p "$GIGASCRIBE_DATA_DIR/uploads" "$GIGASCRIBE_DATA_DIR/cache/matplotlib" "$GIGASCRIBE_MODELS_DIR" \
    && chmod +x /app/scripts/container-entrypoint.sh
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD curl -fsS http://127.0.0.1:8000/health/ready >/dev/null || exit 1
ENTRYPOINT ["/app/scripts/container-entrypoint.sh"]
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
