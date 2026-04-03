# 기술결정기록 (Architecture Decision Records)

GovOn의 핵심 기술 결정을 기록하는 인덱스입니다. 현재 ADR은 두 층으로 나뉩니다.

- `docs/adr/`: 기반 기술 선택
- `docs/architecture/`: 상위 런타임/오케스트레이션 구조

## ADR 인덱스

| ADR | 위치 | 상태 | 설명 |
|-----|------|------|------|
| ADR-003 | [docs/adr/ADR-003-vllm-serving.md](ADR-003-vllm-serving.md) | Accepted | `govon` CLI가 붙는 로컬 FastAPI daemon의 추론 엔진으로 vLLM 유지 |
| ADR-004 | [docs/adr/ADR-004-faiss-vector-search.md](ADR-004-faiss-vector-search.md) | Accepted | 로컬 RAG 검색 계층으로 FAISS + BM25 유지 |
| ADR-006 | [docs/architecture/ADR-006-agentic-architecture.md](../architecture/ADR-006-agentic-architecture.md) | Accepted | GovOn Shell MVP의 전체 아키텍처를 CLI + daemon + approval loop로 확정 |

## 현재 기준선

GovOn의 현재 제품 기준은 다음과 같습니다.

- 제품 본체는 웹 UI가 아니라 `govon` 대화형 CLI 셸이다.
- 내부에는 로컬 FastAPI daemon runtime이 자동 기동된다.
- base model은 의도 파악과 tool 선택을 담당한다.
- 민원 답변 작성 단계에서만 civil-response LoRA adapter를 사용한다.
- tool 실행은 작업 단위 승인 후 진행한다.
- 근거/출처는 기본 표시가 아니라 후속 증강 작업으로 다룬다.

이 기준은 [GovOn Shell MVP 아키텍처 문서](../architecture/GovOn-shell-mvp-architecture.md)와 [ADR-006](../architecture/ADR-006-agentic-architecture.md)을 우선한다.

## 작성 원칙

1. 하나의 ADR은 하나의 결정을 다룬다.
2. 결정 자체보다 `왜 그 결정을 유지하는지`를 중심으로 쓴다.
3. 이후 더 큰 결정으로 대체되면 `Deprecated` 또는 `Superseded` 상태로 남긴다.
4. 구현이 바뀌면 문서도 같은 턴에 함께 갱신한다.
