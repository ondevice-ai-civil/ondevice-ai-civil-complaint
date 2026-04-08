# 기술결정기록 (Architecture Decision Records)

GovOn의 핵심 기술 결정을 기록하는 인덱스입니다. 현재 ADR은 두 층으로 나뉩니다.

- `docs/adr/`: 기반 기술 선택
- `docs/architecture/`: 상위 런타임/오케스트레이션 구조

## ADR 인덱스

| ADR | 위치 | 상태 | 설명 |
|-----|------|------|------|
| TEMPLATE | [docs/adr/TEMPLATE.md](TEMPLATE.md) | - | ADR 작성 템플릿 |
| ADR-003 | [docs/adr/ADR-003-vllm-serving.md](ADR-003-vllm-serving.md) | Accepted | `govon` CLI가 붙는 로컬 FastAPI daemon의 추론 엔진으로 vLLM 유지 |
| ADR-004 | [docs/adr/ADR-004-faiss-vector-search.md](ADR-004-faiss-vector-search.md) | Accepted | 로컬 RAG 검색 계층으로 FAISS + BM25 유지 |
| ADR-006 | [docs/archive/v3-planner-era/ADR-006-agentic-architecture.md](../archive/v3-planner-era/ADR-006-agentic-architecture.md) | Superseded | v3 planner 기반 아키텍처 → v4 ReAct+ToolNode로 대체 |

## 현재 기준선

GovOn의 현재 제품 기준은 다음과 같습니다. (v4 ReAct + ToolNode 아키텍처)

- 제품 본체는 웹 UI가 아니라 `govon` 대화형 CLI 셸이다.
- 내부에는 로컬 FastAPI daemon runtime이 자동 기동된다.
- LangGraph agent 노드(LLM + `bind_tools`)가 의도 파악과 `tool_call`을 자율 결정한다.
- Tier 0 도구(검색/분석)는 자동 실행, Tier 1 도구(어댑터)는 사용자 승인 후 실행한다.
- 어댑터 도구(`public_admin_adapter`, `legal_adapter`) 호출 시에만 해당 LoRA를 per-request attach한다.
- 거부 시 agent가 대안을 제시하는 ReAct 루프 구조이다.

이 기준은 [TO-BE Architecture SVG](https://govon-org.github.io/GovOn/govon-tobe-architecture.svg)를 우선한다.

## 번호 체계

- ADR 번호는 3자리 순번을 사용한다 (ADR-001, ADR-002, ...).
- 결번은 허용한다.
- 기반 기술 선택: `docs/adr/ADR-NNN-slug.md`
- 상위 아키텍처 결정: `docs/architecture/ADR-NNN-slug.md`
- 한 결정이 다른 결정으로 대체되면 원본에 `Superseded by ADR-NNN`을 명시한다.

## 작성 원칙

1. 하나의 ADR은 하나의 결정을 다룬다.
2. 결정 자체보다 `왜 그 결정을 유지하는지`를 중심으로 쓴다.
3. 이후 더 큰 결정으로 대체되면 `Deprecated` 또는 `Superseded` 상태로 남긴다.
4. 구현이 바뀌면 문서도 같은 턴에 함께 갱신한다.
