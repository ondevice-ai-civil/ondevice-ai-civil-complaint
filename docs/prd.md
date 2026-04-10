# PRD: GovOn — 국가 인프라 AX(Agentic Transformation) 플랫폼
**Status**: Accepted Target | **Author**: umyunsang | **Last Updated**: 2026-04-10 | **Version**: 6.0
**Stakeholders**: Eng Lead, AI Lead, Runtime Lead

---

## 1. Problem Statement (문제 정의)

대한민국 공공 부문은 수십 년에 걸친 DX(Digital Transformation) 투자로 방대한 디지털 인프라를 구축했다. 그러나 이 인프라는 **부처·기관·부서별로 분산되어** 각각의 독립 시스템으로 운영된다.

핵심 문제:
1. 데이터와 API가 기관별로 단절되어 있어 통합 조회가 불가능하다.
2. 실무자는 동일한 민원 하나를 처리하기 위해 여러 시스템에 개별 접근해야 한다.
3. 기존 DX 시스템들은 단독으로는 유능하지만 서로 연결되지 않는다.
4. LLM이 도입되어도 각 시스템의 API가 단절된 상태로는 진정한 통합 서비스가 불가능하다.

**현재 문제는 DX가 부족한 것이 아니라, 분산된 DX 인프라를 하나의 지능형 인터페이스로 통합하는 AX(Agentic Transformation) 레이어가 없다는 데 있다.**

GovOn은 이 문제를 다음 방식으로 해결한다:

- 각 기관의 API 엔드포인트를 LLM이 호출 가능한 도구(tool)로 래핑한다.
- 중앙 LLM이 어느 도구를 호출할지 자율적으로 판단하고 체이닝한다.
- 사용자는 단일 `govon` 셸에서 자연어로 요청하면 에이전트가 필요한 기관 API들을 조율해 결과를 종합한다.
- 에이전트는 승인 필요로 지정된 도구(Tier 1 어댑터)에 대해서만 실행 전 승인을 요청하며, Tier 0 검색/분석 도구는 자동 실행한다.

---

## 2. Goals & Success Metrics (목표 및 성공 지표)
본 프로젝트의 목표는 분산된 국가 DX 인프라 위에 **AX 레이어**를 구축하여, 공무원이 **'행정 엔진의 메인테이너'**로서 고도의 의사결정에만 집중할 수 있는 **에이전틱 통합 환경**을 제공하는 것입니다.

| 목표 (Goal) | 성공 지표 (Metric) | 목표치 (Target) |
|------|--------|--------|
| AX 단일 인터페이스 | 단일 `govon` 셸에서 다중 기관 API 통합 조회 가능 여부 | 100% |
| 셸 중심 업무 진입 | `govon` 실행 후 첫 응답 가능 상태 | 10초 이내 |
| 승인 기반 실행 신뢰성 | 승인 없는 tool 실행 비율 | 0% |
| 도메인 답변 초안 생산성 | 도메인 어댑터를 통한 답변 초안 생성 시간 | 60초 이내 |
| 세션 연속성 | `govon --session <id>` 재개 성공률 | 100% |
| 근거 보강 가능성 | 초안 생성 후 evidence augmentation 성공률 | 95% 이상 |
| 도메인 확장성 | MVP 2개 어댑터 → 16개 국가 도메인 어댑터 확장 경로 확보 | 아키텍처 준비 완료 |

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

GovOn은 DX → AX 전환을 위한 4계층 구조를 가진다.

1. **CLI Surface (AX 진입점)**
   - `govon`으로 진입하는 대화형 셸
   - 자연어 중심 상호작용
   - 승인/거절 UI

2. **AX Runtime**
   - FastAPI 기반 런타임
   - 모델, tool, 세션, RAG를 단일 ownership으로 관리
   - HuggingFace Space 또는 Docker로 배포

3. **ReAct Agent Loop (LLM 오케스트레이터)**
   - LangGraph StateGraph 위에서 agent 노드가 자율적으로 도구 선택
   - `bind_tools()`로 LLM에 도구 스키마 전달, 네이티브 tool_call로 결정
   - Tier 0 도구(검색/분석)는 자동 실행, Tier 1 도구(어댑터)는 승인 후 실행
   - 거부 시 agent가 대안을 제시하는 루프 구조

4. **Domain Adapter Layer (DX 인프라 래핑)**
   - Tier 0: `rag_search`, `api_lookup`, `stats_lookup`, `keyword_analyzer`, `demographics_lookup`, `issue_detector`
   - Tier 1: 도메인 어댑터 — 각 기관 API를 LLM 도구로 래핑, 전문 LoRA로 답변 생성
   - MVP: `public_admin_adapter` (공공행정 LoRA), `legal_adapter` (법률 LoRA)
   - `adapters.yaml` 기반 동적 도구 등록 — 신규 도메인 추가 시 yaml만 수정

---

## 5-1. 16개 국가 도메인 어댑터 로드맵

GovOn은 대한민국 16개 국가 행정 도메인을 순차적으로 통합한다.

| 단계 | 도메인 | 어댑터 | 상태 |
|------|--------|--------|------|
| MVP | 공공행정 | `public_admin_adapter` | 운영 중 |
| MVP | 법률 | `legal_adapter` | 운영 중 |
| R2 | 보건의료 | `healthcare_adapter` | 계획 |
| R2 | 사회복지 | `welfare_adapter` | 계획 |
| R2 | 교육 | `education_adapter` | 계획 |
| R3 | 교통물류 | `transport_adapter` | 계획 |
| R3 | 재정금융 | `finance_adapter` | 계획 |
| R3 | 재난안전 | `disaster_adapter` | 계획 |
| R4 | 환경기상 | `environment_adapter` | 계획 |
| R4 | 농축수산 | `agriculture_adapter` | 계획 |
| R4 | 식품건강 | `food_adapter` | 계획 |
| R4 | 산업고용 | `industry_adapter` | 계획 |
| R5 | 국토관리 | `land_adapter` | 계획 |
| R5 | 문화관광 | `culture_adapter` | 계획 |
| R5 | 과학기술 | `science_adapter` | 계획 |
| R5 | 통일외교안보 | `diplomacy_adapter` | 계획 |

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
