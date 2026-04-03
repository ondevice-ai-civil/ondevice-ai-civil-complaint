# ADR-006: GovOn CLI Shell + Local Daemon MVP Architecture

## Status
Accepted

## Date
2026-04-03

## Context

GovOn은 현재 FastAPI 기반 추론 서버, vLLM 서빙, 외부 API action, RAG 검색 뼈대를 이미 보유하고 있다. 그러나 기존 문서와 이슈는 다음과 같은 충돌을 갖고 있었다.

- 첫 릴리즈의 실제 제품 표면은 웹 UI가 아니라 대화형 CLI 셸이어야 한다.
- 공문서 작성과 분류는 MVP 핵심 흐름이 아니다.
- tool 호출은 자동 연쇄 실행보다 사용자 승인 기반 task execution이 더 중요하다.
- LangGraph를 도입할 수는 있지만, MVP의 본질은 특정 프레임워크가 아니라 승인 기반 task loop다.
- 세션은 인메모리가 아니라 재개 가능한 로컬 저장소가 필요하다.

따라서 첫 릴리즈의 기준 아키텍처를 `GovOn CLI Shell + Local FastAPI Daemon + Approval-Gated Task Loop`로 재정의한다.

## Decision

### 1. 첫 릴리즈 제품 표면은 Shell-First로 고정한다

MVP의 사용자 진입점은 터미널에서 실행하는 `govon` 셸이다.

- 사용자는 설치 후 shell/bash에서 `govon` 명령으로 세션을 시작한다.
- 사용자는 모든 업무 요청을 자연어로 입력한다.
- 웹/앱 UI는 동일한 런타임을 재사용하는 후속 표면으로 취급한다.

### 2. 런타임은 로컬 FastAPI 데몬으로 분리한다

`govon` CLI는 모델과 도구를 직접 들고 있지 않는다. 로컬 FastAPI 데몬에 연결한다.

- `govon` 실행 시 데몬이 없으면 자동 기동한다.
- 이미 떠 있으면 재사용한다.
- 데몬은 모델, tool registry, retrieval, SQLite 세션 저장소를 단일 ownership으로 관리한다.

### 3. MVP의 핵심 오케스트레이션은 승인 기반 task loop다

각 사용자 입력에 대해 런타임은 먼저 `한 작업(task loop)`을 정의한다.

실행 전에는 반드시:

- 무엇을 하려는지
- 왜 필요한지
- 어떤 결과를 기대하는지

를 쉬운 설명으로 보여주고 `승인 / 거절`을 받는다.

여러 tool이 필요해도 한 task loop 안에서 한 번만 승인받는다.

### 4. 거절은 즉시 중단이며 완전 대기 상태를 의미한다

사용자가 거절하면:

- 해당 작업은 즉시 중단한다.
- 다른 tool을 자동 실행하지 않는다.
- 설명성 오류 응답을 덧붙이지 않는다.
- 다음 사용자 입력을 기다린다.

### 5. 민원 답변 작성 시에만 civil-response adapter를 사용한다

Civil-response LoRA adapter는 일반 대화 전체에 항상 붙어 있지 않다.

- 실제 작성 의도가 명확한 task에서만 사용한다.
- adapter activation도 task 승인 범위 안에 포함된다.
- public-doc adapter와 classification adapter는 MVP 범위에서 제외한다.

### 6. 증거 제시는 후속 보강 작업으로 처리한다

초안 작성 시 내부 검색을 사용할 수는 있지만, evidence는 기본적으로 즉시 노출하지 않는다.

사용자가 후속으로:

- "근거를 보여줘"
- "왜 이렇게 답했어?"
- "출처를 붙여줘"

와 같이 요청하면 원 질문과 생성된 답변을 함께 기준으로 다시 검색하고, 기존 답변 아래에 근거/출처 섹션을 추가한다.

### 7. 세션 저장은 SQLite를 기본으로 한다

MVP 세션 저장 범위는 다음으로 제한한다.

- 대화 기록
- tool 사용 기록

전체 draft versioning이나 복잡한 graph checkpoint 저장은 MVP 이후로 미룬다.

## Consequences

### 장점

- 실제 MVP 사용자 표면과 문서가 일치한다.
- tool 실행이 사용자 통제 아래로 들어가 신뢰성과 예측 가능성이 높아진다.
- 공문서/분류/UI를 걷어내고 민원 답변 중심으로 범위를 줄일 수 있다.
- FastAPI 런타임, RAG, API action 등 기존 자산을 재사용할 수 있다.
- 세션 재개와 daemon 재사용으로 CLI 체감 속도를 높일 수 있다.

### 단점

- 승인 UI, task grouping, rejection semantics를 명확히 구현해야 한다.
- daemon과 CLI 사이의 로컬 프로세스 관리가 필요하다.
- evidence를 나중에 붙이는 방식이라 기본 답변과 근거 흐름이 분리된다.
- line-level provenance, classification, public-doc drafting은 별도 후속 범위로 밀린다.

## Implementation Notes

- FastAPI는 외부 공개 API가 아니라 로컬 daemon runtime으로 우선 사용한다.
- shell client는 localhost runtime에만 연결한다.
- tool registry는 MVP에서 `api_lookup`, `rag_search`, `draft_civil_response`, `append_evidence` 중심으로 재구성한다.
- LangGraph는 필수 의존이 아니라 후속 확장 가능 지점으로 취급한다.
