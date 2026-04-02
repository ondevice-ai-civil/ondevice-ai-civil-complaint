# ADR-006: GovOn Shell-First Agentic Architecture for First Release

## Status
Accepted

## Date
2026-04-02

## Context

GovOn은 현재 FastAPI 기반 추론 API, vLLM 서빙, RAG 검색, 패키징 자산을 이미 보유하고 있다. 그러나 기존 문서는 첫 릴리즈의 사용자 표면을 `웹 UI/사이드바`로 설명하고, 에이전트 프레임워크도 `Smolagents Phase 1 -> LangGraph Phase 2`로 분리해 설명하고 있었다.

이 구조는 현재 팀의 릴리즈 결정과 충돌한다.

- 첫 릴리즈의 제품 표면은 `govon` 명령으로 진입하는 대화형 셸이다.
- 첫 릴리즈 안에 graph-based decision framework를 포함한다.
- LLM이 무작정 tool을 호출하지 않도록 state, guardrail, checkpoint, recovery를 런타임 기본 기능으로 둔다.
- 웹/앱 UI는 같은 런타임 위에 추가되는 후속 표면이며, R1의 주제품이 아니다.

따라서 첫 릴리즈 기준의 아키텍처를 `GovOn Agentic Shell + LangGraph-based decision framework`로 재정의해야 한다.

## Decision

### 1. 첫 릴리즈 제품 표면은 Shell-First로 고정한다

R1의 사용자 진입점은 웹 사이드바가 아니라 터미널에서 실행하는 `govon` 셸이다.

- 사용자는 설치 후 shell/bash에서 `govon` 명령으로 세션을 시작한다.
- 공무원은 기존 행정 시스템을 유지한 채 GovOn 셸을 독립형 업무 어시스턴트로 병행 사용한다.
- 웹/앱 UI는 동일한 런타임을 재사용하는 후속 표면으로 취급한다.

### 2. 런타임의 핵심은 LangGraph 기반 상태 그래프다

R1 런타임은 LangGraph 또는 동급의 graph runtime을 사용해 다음을 명시적으로 관리한다.

- 세션 상태
- 사용자 의도와 route 결정
- tool eligibility와 입력 검증
- tool 실행 결과와 citations
- draft 생성과 합성
- checkpoint, retry, recovery
- audit trace

이 decision layer는 단순한 `LLM -> tool -> LLM` 루프가 아니라, 상태 전이와 중단 지점을 가진 agentic runtime이어야 한다.

### 3. R1 아키텍처는 5개 계층으로 구성한다

#### 3.1 User Surface
- `govon` interactive shell
- 자유 입력, 멀티라인 붙여넣기, slash commands
- 스트리밍 응답, sources 확인, 세션 재개

#### 3.2 Runtime Adapter
- FastAPI 기반 API 계층
- shell client와 decision graph 사이의 I/O 어댑터
- 스트리밍, 요청 검증, 환경 상태, 세션 식별자 처리

#### 3.3 Decision Graph
- LangGraph state machine
- route 결정: `no-tool`, `search-first`, `search-then-draft`, `ask-back`, `retry`, `fallback`
- guardrail: 불필요한 tool 호출 억제, 입력 부족 시 재질문, 실패 시 복구 규칙 적용

#### 3.4 Tooling and Model Execution
- 표준 tool interface로 래핑된 검색/외부 API/초안 생성 action
- vLLM OpenAI-compatible endpoint에 연결되는 model adapter
- 필요 시 문서 유형 또는 태스크별 모델 프로필/LoRA 프로필 선택
- 현재 기본 내장 tool catalog는 `classify`, `search_similar`, `generate_public_doc`, `generate_civil_response`, `api_lookup`를 포함한다
- 단, tool registry 자체는 고정 5개로 닫지 않고 후속 요구사항에 맞게 tool을 추가할 수 있어야 한다

#### 3.5 Persistence
- session transcript
- graph checkpoint store
- audit log / execution trace
- recovery metadata

### 4. Graph state를 R1의 공통 계약으로 사용한다

다음 정보는 graph state 또는 연결된 persistence 계층에서 일관되게 유지한다.

- `session_id`
- `messages`
- `user_intent`
- `selected_route`
- `eligible_tools`
- `tool_inputs`
- `tool_results`
- `citations`
- `draft_candidates`
- `final_response`
- `checkpoint_id`
- `audit_trace`
- `error_state`

이 state schema는 shell, runtime, testing, recovery 문서가 공유하는 공통 용어 집합이다.

### 5. Tool 호출은 "의사결정 이후"에만 허용한다

모든 요청은 먼저 decision node를 통과해야 한다.

- 단순 안내/설명 요청이면 `no-tool`
- 사실 근거가 필요하면 `search-first`
- 검색 근거를 바탕으로 답변 초안이 필요하면 `search-then-draft`
- 입력이 부족하거나 모호하면 `ask-back`
- 실패 시 `retry` 또는 `fallback`

즉, tool 호출은 LLM의 자유 행동이 아니라 policy와 state에 의해 제약되는 실행 단계다.

### 6. Checkpoint와 recovery를 R1 기본 기능으로 포함한다

각 주요 graph node 실행 후 checkpoint를 저장한다.

- 세션 재개
- 특정 node부터 재시도
- tool failure 이후 fallback
- human-in-the-loop 승인 또는 확인

이 기능은 R2가 아니라 R1 agentic runtime의 일부로 정의한다.

### 7. 웹/앱 UI는 동일한 런타임 위에 올린다

향후 웹 사이드 패널 또는 데스크톱 앱을 도입하더라도 orchestration은 별도로 만들지 않는다.

- shell과 웹은 동일한 decision graph와 tool layer를 공유한다.
- 표면만 달라지고, 세션/도구/정책/감사 로그는 같은 런타임 계약을 사용한다.

## Consequences

### 장점

- 첫 릴리즈 정의와 아키텍처 문서가 일치한다.
- agentic runtime의 핵심 가치인 guardrail, checkpoint, recovery를 초기에 확보할 수 있다.
- shell, 테스트, 후속 UI가 동일한 orchestration spine을 공유한다.
- tool layer와 model adapter가 분리되어 유지보수와 교체가 쉬워진다.
- 세션/감사 로그/재시도 흐름을 운영 요구사항으로 명확히 다룰 수 있다.

### 단점

- 단순 API 래핑보다 런타임 구조가 복잡해진다.
- state schema, checkpoint 저장소, tool policy를 함께 설계해야 한다.
- shell-first 제품이라 초기 사용자층이 웹 UI보다 제한적일 수 있다.
- 문서, 테스트, 런타임 구현이 같은 계약을 따라야 하므로 정합성 비용이 높다.

## Implementation Notes

- 모델 실행은 vLLM OpenAI-compatible endpoint와 adapter 계층을 통해 연결한다.
- 검색, 민원분석, 초안 생성 기능은 framework-specific decorator가 아니라 표준 tool interface로 노출한다.
- shell client는 runtime API를 직접 호출하되, 내부 orchestration 세부 구현에 결합되지 않는다.
- 문서 기준 canonical workstream은 `#406`, `#407`, `#409`, `#410`, `#415`, `#416`, `#417`, `#418`이다.
