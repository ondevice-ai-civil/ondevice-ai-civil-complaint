# WORKFLOW: GovOn Agentic Shell Runtime

**Version**: 3.0
**Date**: 2026-04-02
**Status**: Active Target for R1
**Implements**: GovOn Agentic Shell + graph-based decision framework

## Overview

이 워크플로우는 첫 릴리즈에서 공무원이 `govon` 셸을 통해 민원성 질의, 유사 사례 검색, 답변 초안 생성을 수행하는 기본 실행 경로를 정의한다.

핵심 원칙은 다음과 같다.

- 모든 요청은 먼저 decision graph를 통과한다.
- tool 호출은 필요성이 확인된 뒤에만 실행한다.
- 세션, checkpoint, audit trace를 turn 단위로 유지한다.
- 실패 시 바로 중단하지 않고 retry, fallback, ask-back 중 하나로 복구한다.

## Actors

| Actor | Role |
|---|---|
| Public Official | 기존 행정 시스템과 병행하여 GovOn 셸에 질의를 입력 |
| GovOn Shell Client | 입력 수집, 스트리밍 표시, slash command, 세션 재개 |
| FastAPI Runtime Adapter | 요청 검증, stream 처리, graph runtime 호출 |
| LangGraph Decision Runtime | state 기반 route 결정, tool orchestration, synthesis |
| Tool Nodes | built-in tool catalog + 확장 가능한 registry 기반 action 실행 |
| Persistence Layer | transcript, checkpoint, audit log 저장 |

## Prerequisites

- `govon` shell client가 설치되어 있다.
- FastAPI runtime이 실행 중이다.
- vLLM 또는 호환 모델 endpoint가 응답 가능하다.
- 표준 tool registry에 기본 built-in tool과 추가 확장 tool이 함께 등록될 수 있다.
- session store와 checkpoint store가 사용 가능하다.
- 외부 API가 필요한 경우 기관 환경변수가 설정되어 있다.

## Trigger

사용자가 shell/bash에서 `govon`을 실행하고 질의를 제출한다.

예시 입력:

```text
민원 내용 붙여넣기:
"아파트 단지 앞 도로 포트홀 보수 지연에 대한 민원이 반복 접수되고 있습니다..."
```

## State Graph

기본 실행 상태는 다음과 같다.

```text
[session_bootstrap]
  -> [input_normalize]
  -> [decision]
    -> [no_tool_response]
    -> [search]
    -> [search_then_draft]
    -> [ask_back]
    -> [fallback]
  -> [synthesis]
  -> [stream_response]
  -> [persist]
  -> [idle]
```

각 주요 node 이후에는 checkpoint를 저장한다.

현재 기본 built-in tool catalog는 다음과 같다.

- `classify`
- `search_similar`
- `generate_public_doc`
- `generate_civil_response`
- `api_lookup`

위 목록은 초기 제공 세트이며, 실제 runtime registry는 기관 업무 요구에 맞게 확장 가능하다.

## Workflow Steps

### STEP 0: Session Bootstrap

**Actor**: GovOn Shell Client + Persistence Layer

**Action**:
- 새 세션을 만들거나 기존 세션을 재개한다.
- 최근 transcript, draft 후보, checkpoint 메타데이터를 로드한다.

**Output**:
- `session_id`
- `session_context`
- `last_checkpoint` (있다면)

### STEP 1: Input Normalize and Policy Precheck

**Actor**: FastAPI Runtime Adapter

**Action**:
- 빈 입력, 과도한 길이, 기본 금지 패턴을 검증한다.
- slash command 여부를 해석한다.
- 일반 질의인 경우 graph runtime에 넘길 `normalized_input`을 만든다.

**Failure / Recovery**:
- validation error면 shell에 구조화된 오류를 반환한다.
- slash command면 해당 제어 흐름으로 바로 전환한다.

### STEP 2: Decision Node

**Actor**: LangGraph Decision Runtime

**Action**:
- 현재 세션 상태와 사용자 입력을 바탕으로 다음 route 중 하나를 선택한다.

**Available routes**:
- `no-tool`: 일반 설명, 정책 안내, 간단한 재작성
- `search-first`: 사실 근거 또는 유사 사례가 먼저 필요한 요청
- `search-then-draft`: 검색 근거를 바탕으로 민원 답변/공문 초안까지 필요한 요청
- `ask-back`: 입력이 부족하거나 범위가 불명확한 요청
- `fallback`: tool 또는 모델 실패 이후 최소 안전 응답

**Guardrails**:
- 입력이 불충분하면 tool을 부르지 않고 `ask-back`
- 동일 turn에서 중복 tool 반복 호출 금지
- 빈 검색 결과에서는 근거 없는 draft 생성 금지
- 외부 API 입력 스키마가 불완전하면 실행 차단

### STEP 3A: No-Tool Response

**Actor**: LangGraph Decision Runtime + Model Adapter

**Action**:
- tool 없이 직접 응답을 생성한다.
- 정책 설명, 요약, 문체 수정, 간단한 후속 질문에 사용한다.

**Output**:
- `final_response`
- `route = no-tool`

### STEP 3B: Search-First

**Actor**: Tool Nodes

**Action**:
- RAG 검색 또는 민원분석 API를 실행한다.
- 결과를 `tool_results`, `citations` 필드에 정규화한다.

**Failure / Recovery**:
- 타임아웃이면 1회 재시도 후 실패 정보를 state에 저장한다.
- 외부 API 실패 시 검색 가능한 내부 근거가 있으면 내부 근거만으로 계속 진행한다.
- 둘 다 실패하면 `fallback`으로 전환한다.

### STEP 3C: Search-Then-Draft

**Actor**: Tool Nodes + Model Adapter

**Action**:
- 먼저 검색/분석 tool을 실행한다.
- 검색 결과와 citations를 바탕으로 답변 초안 또는 공문 초안을 생성한다.
- draft 결과를 `draft_candidates`에 저장한다.

**Policy**:
- draft는 검색 근거가 없으면 생성하지 않는다.
- 근거 부족 시 `ask-back` 또는 `fallback`으로 내린다.

### STEP 4: Synthesis

**Actor**: LangGraph Decision Runtime

**Action**:
- tool 결과, citations, draft 후보를 하나의 사용자 응답으로 합성한다.
- shell에서 바로 복사 가능한 최종본과 참고 근거를 분리한다.

**Output shape**:
- `final_response`
- `sources`
- `next_actions` (예: `/sources`, `/retry`, `/session resume`)

### STEP 5: Stream Response

**Actor**: FastAPI Runtime Adapter + GovOn Shell Client

**Action**:
- 응답을 스트리밍하거나 블록 단위로 출력한다.
- shell은 최소한 다음 상태를 사용자에게 보여준다.
  - `thinking`
  - `searching`
  - `drafting`
  - `recovering`
  - `done`

### STEP 6: Persist and Checkpoint

**Actor**: Persistence Layer

**Action**:
- transcript 저장
- audit trace 저장
- 현재 node 실행 결과 checkpoint 저장

**Saved fields**:
- `session_id`
- `messages`
- `selected_route`
- `tool_results`
- `citations`
- `draft_candidates`
- `final_response`
- `checkpoint_id`
- `error_state`

### STEP 7: Recovery and Human-in-the-loop

**Actor**: GovOn Shell Client + LangGraph Decision Runtime

**Action**:
- 실패한 node부터 재시도할 수 있다.
- 사용자는 `/retry`, `/sources`, `/session resume` 같은 제어 명령을 사용할 수 있다.
- 필요 시 사용자의 확인을 받아 다음 단계로 진행한다.

**Typical cases**:
- 외부 API 타임아웃 후 재시도
- 검색 결과 부족으로 질문 보강 요청
- 이전 checkpoint에서 세션 재개

## Decision Table

| User request pattern | Route | Notes |
|---|---|---|
| 간단한 설명, 요약, 문체 수정 | `no-tool` | 직접 응답 |
| 근거가 필요한 사실성 질문 | `search-first` | citations 우선 |
| 민원 답변/공문 초안 요청 | `search-then-draft` | 검색 후 초안 생성 |
| 입력이 짧거나 모호함 | `ask-back` | 재질문 후 진행 |
| tool 실패 또는 근거 부족 | `fallback` | 안전 응답 + 재시도 경로 제시 |

## Shell Control Surface

R1 셸은 최소한 다음 제어 흐름을 제공한다.

- `/help`
- `/sources`
- `/retry`
- `/session resume`
- `/copy`
- `/exit`

이 명령은 별도의 제품 기능이 아니라 graph runtime의 상태 전이와 연결된 제어면이다.

## Test Cases

| TC | Trigger | Expected behavior |
|---|---|---|
| TC-01 | `govon` 실행 | 새 세션 또는 재개 가능한 세션 선택 |
| TC-02 | 일반 설명 요청 | `no-tool` route로 직접 응답 |
| TC-03 | 유사 민원 검색 요청 | `search-first` route로 citations 포함 응답 |
| TC-04 | 답변 초안 요청 | `search-then-draft` route로 근거 + 초안 제공 |
| TC-05 | 입력 부족 | `ask-back` route로 보강 질문 |
| TC-06 | 외부 API 타임아웃 | retry 또는 fallback 후 세션 유지 |
| TC-07 | 세션 재개 | checkpoint와 transcript를 불러와 후속 대화 가능 |
