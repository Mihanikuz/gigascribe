#!/usr/bin/env bash
set -uo pipefail
MODE=${1:-cpu}; FAIL=0
pass(){ echo "PASS $*"; }; warn(){ echo "WARN $*"; }; fail(){ echo "FAIL $*"; FAIL=1; }
command -v uname >/dev/null && pass "arch $(uname -m)"
[ -r /etc/os-release ] && pass "os $(. /etc/os-release; echo ${PRETTY_NAME:-unknown})" || warn "cannot read os-release"
free -h >/dev/null 2>&1 && pass "RAM $(free -h | awk '/Mem:/ {print $2}')" || warn "free not found"
df -h . | awk 'NR==2{print "PASS disk free "$4}'
command -v docker >/dev/null && pass "docker $(which docker)" || fail "docker not found"
docker compose version >/dev/null 2>&1 && pass "Docker Compose v2" || fail "Docker Compose v2 missing"
if command -v snap >/dev/null && snap list docker >/dev/null 2>&1; then fail "Snap Docker detected; stop GPU installation and migrate to apt Docker"; fi
ROOT=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || true); [ -n "$ROOT" ] && pass "Docker Root Dir: $ROOT" || warn "docker info unavailable"
[[ "$ROOT" == /var/snap/docker* ]] && fail "Snap Docker root detected: $ROOT"
[ -S /run/docker.sock ] && pass "docker socket /run/docker.sock" || warn "docker socket missing"
[ -f .env ] && pass ".env exists" || warn ".env missing"
if [ -f .env ] && grep -q 'GIGASCRIBE_SECRET_KEY=change-me-long-random-string' .env; then fail "insecure GIGASCRIBE_SECRET_KEY"; fi
for d in data models; do mkdir -p "$d" && [ -w "$d" ] && pass "$d writable" || fail "$d not writable"; done
command -v ffmpeg >/dev/null && pass "ffmpeg" || warn "ffmpeg unavailable on host"
if [ "$MODE" = gpu ]; then
  command -v nvidia-smi >/dev/null && pass "nvidia-smi" || fail "nvidia-smi missing"
  command -v nvidia-container-runtime >/dev/null && pass "nvidia-container-runtime" || fail "nvidia-container-runtime missing"
fi
exit $FAIL
