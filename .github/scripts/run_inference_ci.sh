#!/usr/bin/env bash

set -euo pipefail

coverage_threshold="${1:-80}"
if [ "$#" -gt 0 ]; then
  shift
fi

export SKIP_MODEL_LOAD="${SKIP_MODEL_LOAD:-true}"

# PR-safe runtime suite:
# - explicitly covers the shell-first MVP runtime surfaces
# - avoids broad auto-discovery of legacy retrieval/search suites
# - keeps integration/E2E and storage-heavy indexing flows in dedicated lanes
test_targets=(
  tests/test_inference/test_agent_loop.py
  tests/test_inference/test_api_lookup_capability.py
  tests/test_inference/test_api_server_units.py
  tests/test_inference/test_feature_flags.py
  tests/test_inference/test_graph_smoke.py
  tests/test_inference/test_plan_validator.py
  tests/test_inference/test_response_formatter.py
  tests/test_inference/test_session_context.py
  tests/test_inference/test_tool_router.py
)

coverage_targets=(
  --cov=src.inference.agent_loop
  --cov=src.inference.feature_flags
  --cov=src.inference.graph
  --cov=src.inference.response_formatter
  --cov=src.inference.session_context
  --cov=src.inference.tool_router
)

uv run pytest \
  "${test_targets[@]}" \
  -o "addopts=" \
  "${coverage_targets[@]}" \
  --cov-branch \
  --cov-fail-under="${coverage_threshold}" \
  --cov-report=xml \
  --cov-report=term-missing \
  "$@"
