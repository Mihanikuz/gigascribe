FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GIGASCRIBE_DATA_DIR=/opt/gigascribe/data \
    GIGASCRIBE_MODELS_DIR=/opt/gigascribe/models \
    HF_HOME=/opt/gigascribe/models/huggingface \
    TORCH_HOME=/opt/gigascribe/models/torch

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt "requirements-gigaam.txt" ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

RUN mkdir -p "$GIGASCRIBE_DATA_DIR" "$GIGASCRIBE_MODELS_DIR" \
    && chmod -R 777 "$GIGASCRIBE_DATA_DIR" "$GIGASCRIBE_MODELS_DIR"

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
