#!/usr/bin/env bash

set -euo pipefail

coverage_threshold="${1:-80}"
if [ "$#" -gt 0 ]; then
  shift
fi

export SKIP_MODEL_LOAD="${SKIP_MODEL_LOAD:-true}"

# PR-safe inference suite:
# - auto-discovers tests under tests/test_inference
# - excludes dedicated integration/E2E lanes by filename convention
# - keeps real-FAISS persistence behavior in its own lane until stabilized
# New unit/contract tests should land under tests/test_inference and will be
# picked up automatically without workflow edits.
uv run pytest \
  tests/test_inference \
  --ignore-glob='tests/test_inference/*integration*.py' \
  --ignore-glob='tests/test_inference/*e2e*.py' \
  --ignore='tests/test_inference/test_index_manager.py' \
  -o "addopts=" \
  --cov=src/inference \
  --cov-branch \
  --cov-fail-under="${coverage_threshold}" \
  --cov-report=xml \
  --cov-report=term-missing \
  "$@"
