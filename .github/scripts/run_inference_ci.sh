#!/usr/bin/env bash

set -euo pipefail

coverage_threshold="${1:-80}"
if [ "$#" -gt 0 ]; then
  shift
fi

export SKIP_MODEL_LOAD="${SKIP_MODEL_LOAD:-true}"

# ---------------------------------------------------------------------------
# 테스트 자동 탐색 — 제외 패턴 기반
#
# tests/test_inference/ 아래의 모든 test_*.py를 탐색하되,
# 아래 카테고리에 해당하는 파일만 명시 제외한다.
# 새 런타임/capability/validator 테스트는 제외 패턴에 걸리지 않으면 자동 포함된다.
# ---------------------------------------------------------------------------

# 제외 패턴 (카테고리별)
_EXCLUDE=(
  # 레거시 검색 파이프라인 — 별도 검색 CI 레인
  "*hybrid_search*"
  "test_retriever*"
  "test_bm25*"
  "test_search_*"
  "test_index_manager*"

  # 데이터베이스 레이어 — 별도 DB CI 레인
  "test_db_*"

  # 무거운 ML / vLLM 런타임 — 별도 모델 CI 레인
  "test_vllm*"
  "test_document_processor*"
  "test_tokenizer*"

  # 통합 / E2E — 별도 E2E 레인
  "*e2e*"
  "*integration*"

  # 인프라 유틸리티 (실행 환경 의존)
  "test_health_checker*"
  "test_rate_tracker*"
  "test_runtime_config*"

  # API 통합 (실행 중인 서버 필요)
  "test_agent_api*"
  "test_api_logic*"
  "test_agent_manager*"

  # 외부 서비스 (실제 API 키 필요)
  "test_data_go_kr*"

  # 스키마 / 프롬프트 검증 — 별도 정적 분석 레인
  "test_schemas*"
  "test_prompt_validator*"
)

# find 명령어에 제외 인수 조립
find_args=( tests/test_inference -maxdepth 1 -name "test_*.py" )
for pat in "${_EXCLUDE[@]}"; do
  find_args+=( ! -name "$pat" )
done

test_targets=()
while IFS= read -r f; do
  test_targets+=( "$f" )
done < <( find "${find_args[@]}" | sort )

if [ "${#test_targets[@]}" -eq 0 ]; then
  echo "ERROR: 실행할 테스트 파일을 찾지 못했습니다." >&2
  exit 1
fi

echo "=== 인퍼런스 CI 테스트 목록 (${#test_targets[@]}개) ==="
printf '  %s\n' "${test_targets[@]}"
echo ""

# ---------------------------------------------------------------------------
# 커버리지 대상 — 런타임 핵심 모듈
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
