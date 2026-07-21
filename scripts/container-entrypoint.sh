#!/usr/bin/env sh
set -eu

# Bind mounts are created by Docker as root.  Repair them before dropping
# privileges so uploads and the persistent model cache work on first start.
uid="${HOST_UID:-1000}"
gid="${HOST_GID:-1000}"
for dir in "${GIGASCRIBE_DATA_DIR:-/opt/gigascribe/data}" \
           "${GIGASCRIBE_DATA_DIR:-/opt/gigascribe/data}/uploads" \
           "${GIGASCRIBE_DATA_DIR:-/opt/gigascribe/data}/cache/matplotlib" \
           "${GIGASCRIBE_MODELS_DIR:-/opt/gigascribe/models}"; do
    mkdir -p "$dir"
    chown -R "$uid:$gid" "$dir"
done

if [ "${GIGASCRIBE_DEVICE:-cuda}" = "cuda" ]; then
    python /app/scripts/gpu_smoke_test.py --require-cuda
fi

exec gosu "$uid:$gid" "$@"
