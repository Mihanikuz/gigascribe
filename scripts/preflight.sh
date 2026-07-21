#!/usr/bin/env bash
set -euo pipefail
docker compose config --services | grep -qx gigascribe-gpu
python scripts/gpu_smoke_test.py
