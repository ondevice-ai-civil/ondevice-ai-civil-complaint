#!/usr/bin/env bash

set -euo pipefail

IMAGE="${1:?usage: smoke_test_container.sh <image-ref>}"
CONTAINER_NAME="${CONTAINER_NAME:-govon-runtime-smoke}"
HOST_PORT="${HOST_PORT:-8000}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"
SLEEP_SECONDS="${SLEEP_SECONDS:-2}"
HEALTH_URL="http://127.0.0.1:${HOST_PORT}/health"

cleanup() {
  docker logs "${CONTAINER_NAME}" >/tmp/"${CONTAINER_NAME}".log 2>&1 || true
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}

trap cleanup EXIT

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run -d \
  --name "${CONTAINER_NAME}" \
  -e SERVING_PROFILE="${SERVING_PROFILE:-container}" \
  -e SKIP_MODEL_LOAD="${SKIP_MODEL_LOAD:-true}" \
  -e API_KEY="${API_KEY:-test-key}" \
  -e PORT=8000 \
  -p "${HOST_PORT}:8000" \
  "${IMAGE}" >/dev/null

for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
  if ! docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
    echo "::error::Container exited before readiness probe completed."
    docker logs "${CONTAINER_NAME}" 2>&1 || true
    exit 1
  fi

  response="$(curl -fsS "${HEALTH_URL}" 2>/dev/null || true)"
  if [ -n "${response}" ]; then
    compact_response="$(printf '%s' "${response}" | tr -d '[:space:]')"
    if ! printf '%s' "${compact_response}" | grep -q '"status":"healthy"'; then
      echo "::error::Health endpoint responded without healthy status."
      printf '%s\n' "${response}"
      exit 1
    fi
    if ! printf '%s' "${compact_response}" | grep -q '"profile":"container"'; then
      echo "::error::Container runtime profile mismatch."
      printf '%s\n' "${response}"
      exit 1
    fi

    printf '%s\n' "${response}"
    exit 0
  fi

  sleep "${SLEEP_SECONDS}"
done

echo "::error::Container did not become ready on ${HEALTH_URL}."
docker logs "${CONTAINER_NAME}" 2>&1 || true
exit 1
