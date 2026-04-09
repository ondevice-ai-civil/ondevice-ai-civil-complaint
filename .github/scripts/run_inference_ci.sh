#!/usr/bin/env bash

set -euo pipefail

coverage_threshold="${1:-75}"
if [ "$#" -gt 0 ]; then
  shift
fi

export SKIP_MODEL_LOAD="${SKIP_MODEL_LOAD:-true}"

# ---------------------------------------------------------------------------
# Auto-discover tests — exclusion pattern based
#
# Scans all test_*.py files under tests/test_inference/,
# but explicitly excludes only files matching the categories below.
# New runtime/capability/validator tests are auto-included unless matched.
# ---------------------------------------------------------------------------

# Exclusion patterns (by category)
_EXCLUDE=(
  # Legacy search pipeline — separate search CI lane
  "*hybrid_search*"
  "test_retriever*"
  "test_bm25*"
  "test_search_*"
  "test_index_manager*"

  # Database layer — separate DB CI lane
  "test_db_*"

  # Heavy ML / vLLM runtime — separate model CI lane
  "test_vllm*"
  "test_document_processor*"
  "test_tokenizer*"

  # Integration / E2E — separate E2E lane
  "*e2e*"
  "*integration*"

  # Infrastructure utilities (execution environment dependent)
  "test_health_checker*"
  "test_rate_tracker*"
  "test_runtime_config*"

  # API integration (requires running server)
  "test_agent_api*"
  "test_api_logic*"
  "test_agent_manager*"

  # External services (requires real API keys)
  "test_data_go_kr*"

  # Schema / prompt validation — separate static analysis lane
  "test_schemas*"
  "test_prompt_validator*"
)

# Assemble find command exclusion arguments
find_args=( tests/test_inference -maxdepth 1 -name "test_*.py" )
for pat in "${_EXCLUDE[@]}"; do
  find_args+=( ! -name "$pat" )
done

test_targets=()
while IFS= read -r f; do
  test_targets+=( "$f" )
done < <( find "${find_args[@]}" | sort )

if [ "${#test_targets[@]}" -eq 0 ]; then
  echo "ERROR: No test files found to run." >&2
  exit 1
fi

echo "=== Inference CI Test List (${#test_targets[@]} files) ==="
printf '  %s\n' "${test_targets[@]}"
echo ""

# ---------------------------------------------------------------------------
# Coverage targets — core runtime modules
# ---------------------------------------------------------------------------
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
