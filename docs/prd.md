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
- 정규식/패턴 기반 business tool router 유지
- 분산형 복잡한 graph checkpoint 시스템

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

3. **Approval-Gated Task Loop**
   - LangGraph state graph 위에서 요청을 한 작업으로 정리
   - planner LLM이 tool 선택과 실행 순서를 구조화
   - 실행 전 승인 요청
   - 승인된 경우에만 tool 실행

4. **Execution Layer**
   - base model
   - civil-response adapter
   - unified `api_lookup`
   - local `rag_search`
   - `append_evidence`

---

## 6. Technical Considerations (기술적 고려사항)
- **FastAPI Local Daemon**: CLI와 모델/도구 실행을 분리해 데몬 재사용과 세션 지속성을 확보한다.
- **LangGraph Agent Runtime**: planner, approval interrupt, tool executor, synthesis를 bounded state graph로 고정한다.
- **Model-Driven Tool Selection**: 업무 요청의 도구 선택은 LLM이 session context와 tool metadata를 읽고 결정하며, 정규식 라우터는 shell control 외 정본이 아니다.
- **Approval-Gated Orchestration**: 자동 tool 연쇄 실행보다 사용자 신뢰와 예측 가능성을 우선한다.
- **Single Task Adapter Use**: 민원 답변 작성 단계에서만 adapter를 attach한다.
- **SQLite Session Store**: transcript와 tool log를 단순하고 재개 가능한 형태로 보관한다.

---

## 7. Launch Plan (출시 계획)
- **Phase 1 (MVP)**: CLI + daemon + LangGraph 기반 승인 루프 검증
- **Phase 2**: evidence augmentation, RAG corpus 확장, daemon 운영 고도화
- **Phase 3**: web surface, public-doc adapter, 분류 기능 등 확장

---

## 8. Appendix (부록)
- [GovOn Shell MVP Architecture](architecture/GovOn-shell-mvp-architecture.md)
- [ADR-006: GovOn CLI Shell + Local Daemon MVP Architecture](architecture/ADR-006-agentic-architecture.md)
- [WORKFLOW: 에이전트 오케스트레이터 워크플로우](architecture/WORKFLOW-orchestrator-tool-calling.md)
