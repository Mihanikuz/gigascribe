FROM python:3.11-slim

ARG HOST_UID=1000
ARG HOST_GID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GIGASCRIBE_DATA_DIR=/opt/gigascribe/data \
    GIGASCRIBE_MODELS_DIR=/opt/gigascribe/models \
    HF_HOME=/opt/gigascribe/models/huggingface \
    HF_HUB_OFFLINE=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-gigaam.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

RUN groupadd --gid "$HOST_GID" gigascribe \
    && useradd --uid "$HOST_UID" --gid "$HOST_GID" --home-dir /opt/gigascribe --create-home gigascribe \
    && mkdir -p "$GIGASCRIBE_DATA_DIR" "$GIGASCRIBE_MODELS_DIR" \
    && chown -R gigascribe:gigascribe /opt/gigascribe /app

USER gigascribe

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health/ready >/dev/null || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
