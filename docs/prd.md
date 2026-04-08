# PRD: GovOn — 에이전틱 행정 보조 CLI 셸
**Status**: Accepted Target | **Author**: umyunsang | **Last Updated**: 2026-04-03 | **Version**: 5.0
**Stakeholders**: Eng Lead, AI Lead, Runtime Lead

---

## 1. Problem Statement (문제 정의)

행정 실무자는 민원 답변을 준비할 때 다음 문제를 동시에 겪는다.

1. 질문의 의도를 빠르게 파악해야 한다.
2. 비슷한 사례와 외부 정보를 여러 시스템에서 확인해야 한다.
3. 실제 답변 초안을 정중하고 일관된 문체로 작성해야 한다.
4. 이미 작성한 답변을 다시 고치거나 근거를 덧붙여야 한다.

현재 문제는 단순히 LLM이 없다는 것이 아니라, **작업 단위로 사고하고 승인하며 이어서 대화할 수 있는 실무형 인터페이스가 없다**는 데 있다.

GovOn MVP는 이 문제를 다음 방식으로 해결한다.

- 사용자는 웹 UI가 아니라 `govon` 셸에서 자연어로 요청한다.
- AI는 한 번의 요청을 하나의 작업으로 해석한다.
- 필요한 검색이나 API 조회가 있으면 먼저 사람말로 설명하고 승인을 받는다.
- 승인된 작업만 실행하고, 거절되면 바로 멈춘다.
- 답변 작성이 필요하면 민원 답변 특화 어댑터를 사용한다.

---

## 2. Goals & Success Metrics (목표 및 성공 지표)
본 프로젝트의 목표는 공무원이 **'행정 엔진의 메인테이너'**로서 고도의 의사결정에만 집중할 수 있는 **에이전틱 행정 환경**을 구축하는 것입니다.

| 목표 (Goal) | 성공 지표 (Metric) | 목표치 (Target) |
|------|--------|--------|
| 셸 중심 업무 진입 | `govon` 실행 후 첫 응답 가능 상태 | 10초 이내 |
| 승인 기반 실행 신뢰성 | 승인 없는 tool 실행 비율 | 0% |
| 민원 답변 초안 생산성 | 답변 초안 생성까지 걸리는 시간 | 60초 이내 |
| 세션 연속성 | `govon --session <id>` 재개 성공률 | 100% |
| 근거 보강 가능성 | 초안 생성 후 evidence augmentation 성공률 | 95% 이상 |

---

## 3. Non-Goals (비목표)
- 공문서 초안 작성
- 민원 분류 기능
- 웹 UI 기반 업무 수행
- 승인 없는 완전 자율 에이전트
- 정적 planner / executor / synthesis 노드 구조
- 정규식/패턴 기반 business tool router

---

## 4. User Personas & Stories (사용자 페르소나 및 스토리)

### Primary Persona: 민원 담당 실무자

*"터미널에서 그냥 자연어로 말하면, 필요한 검색과 조회를 거쳐 답변 초안을 같이 만들어주는 업무 보조 셸이 필요합니다."*

**User Stories:**
1. "나는 민원 답변 초안을 자연어로 요청하고, 필요한 자료 검색은 AI가 대신 제안해주길 원한다."
2. "나는 AI가 도구를 쓰기 전에 왜 필요한지 쉽게 설명하고 승인받길 원한다."
3. "나는 답변을 만든 뒤에도 같은 세션에서 수정 요청이나 근거 추가 요청을 이어서 하고 싶다."

---

## 5. Solution Overview (솔루션 개요)

GovOn MVP는 다음 구조를 가진다.

1. **CLI Surface**
   - `govon`으로 진입하는 대화형 셸
   - 자연어 중심 상호작용
   - 승인/거절 UI

2. **Local Runtime Daemon**
   - FastAPI 기반 로컬 데몬
   - 모델, tool, 세션, RAG를 단일 ownership으로 관리

3. **ReAct Agent Loop**
   - LangGraph StateGraph 위에서 agent 노드가 자율적으로 도구 선택
   - `bind_tools()`로 LLM에 도구 스키마 전달, 네이티브 tool_call로 결정
   - Tier 0 도구(검색/분석)는 자동 실행, Tier 1 도구(어댑터)는 승인 후 실행
   - 거부 시 agent가 대안을 제시하는 루프 구조

4. **Tool Layer**
   - Tier 0: `rag_search`, `api_lookup`, `stats_lookup`, `keyword_analyzer`, `demographics_lookup`, `issue_detector`
   - Tier 1: `public_admin_adapter` (civil LoRA), `legal_adapter` (legal LoRA)
   - `adapters.yaml` 기반 동적 도구 등록

---

## 6. Technical Considerations (기술적 고려사항)
- **FastAPI Local Daemon**: CLI와 모델/도구 실행을 분리해 데몬 재사용과 세션 지속성을 확보한다.
- **LangGraph Agent Runtime**: agent, approval_wait, ToolNode, persist를 StateGraph로 구성하고 ReAct 루프로 동작한다.
- **Model-Driven Tool Selection**: 베이스 LLM이 `bind_tools()`로 전달받은 도구 스키마를 읽고 자율적으로 `tool_call`을 결정한다.
- **Approval-Gated Orchestration**: 자동 tool 연쇄 실행보다 사용자 신뢰와 예측 가능성을 우선한다.
- **Per-Request LoRA Attach**: 어댑터 도구(`public_admin_adapter`, `legal_adapter`) tool_call 시에만 해당 LoRA를 attach한다.
- **Checkpointer Session**: LangGraph checkpointer로 messages를 영속화하고, persist 노드에서 evidence를 DB에 저장한다.

---

## 7. Launch Plan (출시 계획)
- **Phase 1 (MVP)**: CLI + daemon + LangGraph 기반 승인 루프 검증
- **Phase 2**: evidence augmentation, RAG corpus 확장, daemon 운영 고도화
- **Phase 3**: web surface, public-doc adapter, 분류 기능 등 확장

---

## 8. Appendix (부록)
- [TO-BE Architecture SVG](https://govon-org.github.io/GovOn/govon-tobe-architecture.svg)
- [v3 아키텍처 아카이브](archive/v3-planner-era/) — 이전 planner 기반 문서 보관
