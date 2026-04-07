# GovOn LangGraph 런타임 튜닝 가이드

## 1. 아키텍처 개요

GovOn 에이전트는 6-node StateGraph 토폴로지를 사용한다:

```
session_load → planner → approval_wait → tool_execute → synthesis → persist
```

- **session_load**: 세션 로드 및 쿼리 컨텍스트 구성
- **planner**: LLM/Regex 기반 실행 계획 생성 (도구 선택 + 순서 결정)
- **approval_wait**: human-in-the-loop 승인 게이트 (`interrupt()`)
- **tool_execute**: capability 실행 (독립 도구 병렬, 의존 도구 순차)
- **synthesis**: tool 결과 종합 → 최종 응답 텍스트 생성
- **persist**: 세션/tool_run/graph_run DB 영속화

### 도구 실행 전략

`tool_execute` 노드는 두 단계로 실행된다:

1. **병렬 실행**: `rag_search`, `api_lookup` (독립 도구) — `asyncio.gather()`
2. **순차 실행**: `draft_civil_response`, `append_evidence` (의존 도구) — 누적 컨텍스트 필요

## 2. 레이턴시 기준선

### 노드별 예상 범위

| 노드 | 일반 범위 | 비고 |
|------|-----------|------|
| session_load | 1-5ms | SQLite I/O |
| planner | 50-500ms | LLM: 200-500ms, Regex(CI): 1-5ms |
| approval_wait | 사용자 응답 대기 | interrupt 메커니즘 |
| tool_execute | 200-2000ms | 도구 조합에 따라 상이 |
| synthesis | 1-10ms | 텍스트 조합, LLM 미사용 |
| persist | 5-20ms | SQLite 쓰기 |

### Capability별 예상 범위

| Capability | 기본 Timeout | 일반 범위 | 비고 |
|-----------|-------------|-----------|------|
| rag_search | 15s | 100-500ms | FAISS + BM25 hybrid |
| api_lookup | 10s | 300-2000ms | 외부 API 의존 |
| draft_civil_response | 30s | 500-5000ms | LLM 생성 |
| append_evidence | 15s | 200-1000ms | RAG + API 조합 |

## 3. 튜닝 포인트

### 3.1 Capability별 Timeout 설정

환경변수로 capability별 timeout을 오버라이드할 수 있다:

```bash
# 기본값 변경
export GOVON_TOOL_TIMEOUT_RAG_SEARCH=20      # 15 → 20초
export GOVON_TOOL_TIMEOUT_API_LOOKUP=15       # 10 → 15초
export GOVON_TOOL_TIMEOUT_DRAFT_CIVIL_RESPONSE=45  # 30 → 45초
export GOVON_TOOL_TIMEOUT_APPEND_EVIDENCE=20  # 15 → 20초
```

코드 위치: `src/inference/graph/capabilities/defaults.py`

### 3.2 임베딩 캐시 크기 (hybrid_search)

`src/inference/hybrid_search.py`의 `HybridSearchEngine._embed_cache`는 `OrderedDict`
기반 LRU 캐시(기본 최대 64개)로 동작한다. 생성자에서 직접 제어하는 상수는 없으며,
`_embed_query()` 내부의 `if len(self._embed_cache) >= 64:` 조건을 수정해 크기를 조정한다:

```python
# 기본: 64
# 높은 트래픽 환경: 256
# 메모리 제한 환경: 32
if len(self._embed_cache) >= 64:   # 이 값을 변경
    self._embed_cache.popitem(last=False)
```

### 3.3 RRF 가중치

Reciprocal Rank Fusion 가중치는 `hybrid_search.py`의 `DEFAULT_RRF_WEIGHTS` dict와
`HybridSearchEngine` 생성자의 `rrf_k` 파라미터로 조정한다:

```python
# rrf_k: RRF smoothing 파라미터 (기본값: 60, 높을수록 순위 차이 완화)
engine = HybridSearchEngine(..., rrf_k=60)

# DEFAULT_RRF_WEIGHTS: 데이터 타입별 dense/sparse 가중치
DEFAULT_RRF_WEIGHTS = {
    IndexType.CASE:   RRFWeightConfig(dense_weight=1.0, sparse_weight=0.7),
    IndexType.LAW:    RRFWeightConfig(dense_weight=0.9, sparse_weight=1.2),
    IndexType.MANUAL: RRFWeightConfig(dense_weight=0.8, sparse_weight=0.8),
    IndexType.NOTICE: RRFWeightConfig(dense_weight=0.6, sparse_weight=0.6),
}
```

### 3.4 top_k

`top_k`는 전역 상수가 아닌 `HybridSearchEngine.search()` 호출 시 per-call 파라미터로 전달된다:

```python
results, mode = await engine.search(query, index_type, top_k=5)
```

- `top_k`를 늘리면 recall이 증가하지만 레이턴시도 증가
- BM25 후보가 50개를 초과하면 성능 저하가 뚜렷함 (BM25Indexer 내부 제한)

## 4. 장애 대응

### 4.1 외부 API 장애 시 Graceful Fallback

`api_lookup` capability는 외부 API(data.go.kr) 의존:

- **timeout**: `GOVON_TOOL_TIMEOUT_API_LOOKUP` 초과 시 `LookupResult(success=False, empty_reason="provider_error")` 반환
- **HTTP 오류**: httpx 에러를 잡아 빈 결과로 fallback
- **graph 계속 실행**: api_lookup 실패 시에도 다른 도구 결과로 synthesis 진행

### 4.2 LLM 서버 무응답 시 Timeout 동작

- `draft_civil_response`의 기본 timeout: 30초
- timeout 초과 시 `LookupResult(success=False)` 반환
- planner가 LLM 기반이면 planner 단계에서도 timeout 적용
- CI 환경(`SKIP_MODEL_LOAD=true`)에서는 `RegexPlannerAdapter`가 fallback

### 4.3 FAISS 인덱스 미로드 시

- `rag_search` 실행 시 인덱스가 없으면 예외 발생
- capability의 `execute()` 메서드가 예외를 잡아 `LookupResult(success=False, error=...)` 반환
- API 서버 레벨에서 503 응답으로 변환 가능

### 4.4 병렬 실행 중 부분 실패

- `asyncio.gather(return_exceptions=True)`로 예외를 격리
- 실패한 도구는 건너뛰고, 성공한 도구 결과만 accumulated_context에 반영
- 로그에 실패 원인 기록 (`logger.error`)

## 5. Benchmark 실행 방법

### 기본 실행

```bash
SKIP_MODEL_LOAD=true python -m benchmarks.bench_graph_latency --repeat 10
```

### JSON 파일로 출력

```bash
SKIP_MODEL_LOAD=true python -m benchmarks.bench_graph_latency --repeat 20 --output bench_results.json
```

### 출력 형식

```json
{
  "repeat": 10,
  "total_ms": {
    "p50": 1250.5,
    "p95": 1580.3,
    "p99": 1620.1,
    "mean": 1300.2,
    "stdev": 85.3
  },
  "runs": [
    {"run_id": 0, "total_ms": 1245.3, "has_final_text": true},
    ...
  ]
}
```

## 6. 회귀 비교 방식

### 기준선 저장

```bash
# 기준선 측정 (main 브랜치)
git checkout main
SKIP_MODEL_LOAD=true python -m benchmarks.bench_graph_latency --repeat 20 --output baseline.json

# 변경 후 측정 (feature 브랜치)
git checkout feat/my-optimization
SKIP_MODEL_LOAD=true python -m benchmarks.bench_graph_latency --repeat 20 --output current.json
```

### 비교 방법

```bash
# p50 비교
python -c "
import json
baseline = json.load(open('baseline.json'))
current = json.load(open('current.json'))
b_p50 = baseline['total_ms']['p50']
c_p50 = current['total_ms']['p50']
diff_pct = ((c_p50 - b_p50) / b_p50) * 100
print(f'Baseline p50: {b_p50}ms')
print(f'Current  p50: {c_p50}ms')
print(f'Diff: {diff_pct:+.1f}%')
"
```

### 회귀 판단 기준

- **p50 증가 > 10%**: 조사 필요
- **p95 증가 > 20%**: 머지 전 원인 분석 필수
- **p99 증가 > 30%**: 반드시 수정 후 머지

### 노드별 계측 활용

각 노드는 `node_latencies: Dict[str, float]` 형태로 레이턴시를 반환한다.
LangGraph state의 `_merge_dicts` reducer가 모든 노드 결과를 하나의 dict로 누적 병합하므로
graph 실행이 완료되면 최종 state에 아래와 같이 모든 노드의 레이턴시가 포함된다:

```python
state["node_latencies"]
# {
#   "session_load":    2.1,
#   "planner":        45.3,
#   "tool_execute":  312.5,
#   "tool:rag_search": 198.7,
#   "tool:api_lookup": 285.4,
#   "synthesis":       1.8,
#   "persist":         8.2,
# }
```

`tool_execute` 노드는 노드 전체 레이턴시(`"tool_execute"` 키)와 함께
개별 도구 레이턴시를 `"tool:<tool_name>"` 접두사 키로 함께 기록한다.
벤치마크 출력(`bench_graph_latency.py`)의 `node_latencies_ms` 섹션에서
노드별 p50/p95/mean을 확인할 수 있다.
