#!/usr/bin/env bash
# Downloads and verifies every model GigaScribe needs *before* the
# application is built and started (item 1):
#
#   ./scripts/download-models.sh
#   docker compose build
#   docker compose up -d
#
# GigaAM/pyannote verification loads the model through its real loader (see
# scripts/download_models.py), which needs gigaam/torch/CUDA -- so this
# script builds the base image itself first to get an environment capable
# of running that loader, then downloads models inside it. The `docker
# compose build` you run afterwards is then a fast, cached no-op; the
# ordering above is what matters: docker compose up never starts before
# every model is present and verified.
#
# Safe to re-run: every step below checks for an existing, verified file
# before touching the network again.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Building the base image (needed to run the GigaAM/pyannote loaders for verification)..."
docker compose build

DOWNLOAD_ARGS=()
if [ "${GIGASCRIBE_SKIP_PYANNOTE:-0}" = "1" ]; then
  DOWNLOAD_ARGS+=(--skip-pyannote)
fi

echo "==> Downloading and verifying GigaAM (and pyannote unless GIGASCRIBE_SKIP_PYANNOTE=1)..."
docker compose run --rm gigascribe-gpu python scripts/download_models.py "${DOWNLOAD_ARGS[@]}"

if [ -n "${GIGASCRIBE_PROTOCOL_MODEL_REPO_ID:-}" ] && [ -n "${GIGASCRIBE_PROTOCOL_MODEL_FILENAME:-}" ]; then
  echo "==> Downloading protocol model qwen3-8b..."
  docker compose -f compose.yaml -f compose.protocol.yaml run --rm gigascribe-gpu \
    python scripts/download_protocol_model.py --model-id qwen3-8b \
    --repo-id "$GIGASCRIBE_PROTOCOL_MODEL_REPO_ID" --filename "$GIGASCRIBE_PROTOCOL_MODEL_FILENAME"
else
  echo "==> Skipping the protocol model (qwen3-8b): set GIGASCRIBE_PROTOCOL_MODEL_REPO_ID and"
  echo "    GIGASCRIBE_PROTOCOL_MODEL_FILENAME to a Hugging Face repo/GGUF filename you've verified"
  echo "    yourself to fetch it here, or run scripts/download_protocol_model.py directly later."
fi

echo "==> Verifying all required model files are present..."
docker compose run --rm gigascribe-gpu python scripts/download_models.py --check

echo "All required models are downloaded and verified. Safe to proceed with: docker compose build && docker compose up -d"
