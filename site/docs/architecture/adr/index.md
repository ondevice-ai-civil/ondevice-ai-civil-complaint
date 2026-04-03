# ADR 인덱스

GovOn은 CLI 셸 중심의 제품 구조를 기준으로 기술 결정을 유지한다. 현재 가장 중요한 결정은 `셸 우선`, `로컬 daemon`, `approval-gated task loop`, `civil-response adapter`, `FAISS 기반 RAG`다.

## ADR 목록

| ADR | 상태 | 요약 |
|-----|------|------|
| ADR-003 | Accepted | vLLM을 로컬 FastAPI daemon의 추론 엔진으로 유지 |
| ADR-004 | Accepted | FAISS + BM25를 로컬 RAG 검색 계층으로 유지 |
| ADR-006 | Accepted | GovOn Shell MVP를 CLI + daemon + approval loop 구조로 확정 |

## 현재 우선 문서

아래 루트 문서를 우선 기준으로 본다.

- `docs/architecture/GovOn-shell-mvp-architecture.md`
- `docs/architecture/ADR-006-agentic-architecture.md`

## 읽는 순서

1. ADR-006으로 제품 구조를 먼저 이해한다.
2. ADR-003으로 로컬 추론 런타임 선택을 확인한다.
3. ADR-004로 RAG 검색 계층 선택을 확인한다.
