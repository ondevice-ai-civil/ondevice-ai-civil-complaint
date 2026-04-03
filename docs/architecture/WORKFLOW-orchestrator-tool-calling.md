# WORKFLOW: GovOn CLI MVP Task Loop

**Version**: 5.0
**Date**: 2026-04-03
**Status**: Accepted MVP Flow
**Implements**: GovOn CLI Shell + Local Daemon + LangGraph Approval-Gated Task Loop

## Overview

이 워크플로우는 GovOn MVP에서 사용자가 `govon` 셸을 통해 민원 답변 초안과 후속 근거 보강을 수행하는 기본 실행 경로를 정의한다.

핵심 원칙은 다음과 같다.

- 모든 요청은 먼저 `이번에 해야 할 한 작업`으로 정규화한다.
- 업무용 tool 선택은 정규식이 아니라 planner LLM이 LangGraph 안에서 결정한다.
- tool 실행 전에는 항상 사람말 승인 요청을 보여준다.
- 하나의 task loop 안에서 여러 tool을 묶어 실행할 수 있다.
- 거절되면 자동 fallback 없이 즉시 idle 상태로 돌아간다.
- 세션에는 대화 기록과 tool 사용 기록만 남긴다.

## Actors

| Actor | Role |
|---|---|
| User | `govon` 셸에 자연어로 요청을 입력 |
| GovOn Shell Client | 입력 수집, 승인 UI, 상태 표시, 세션 재개 |
| Local FastAPI Daemon | 요청 검증, task loop 실행, 모델 및 tool lifecycle 관리 |
| LangGraph Agent Runtime | planner, plan validation, approval wait, tool execution, synthesis |
| Tool Registry | `api_lookup`, `rag_search`, `draft_civil_response`, `append_evidence` 실행 |
| SQLite Store | transcript 및 tool log 저장 |

## Prerequisites

- `govon` shell client가 설치되어 있다.
- 로컬 FastAPI daemon이 실행 중이거나 자동 기동 가능하다.
- base model runtime이 응답 가능하다.
- civil-response adapter가 준비되어 있다.
- SQLite 세션 저장소가 사용 가능하다.
- 외부 API를 쓸 경우 필요한 인증정보가 설정되어 있다.

## Trigger

사용자가 shell/bash에서 `govon`을 실행하고 자연어 질의를 제출한다.

예시 입력:

```text
이 민원에 대한 답변 초안 작성해줘.
```

## Task Loop State

기본 실행 흐름은 다음과 같다.

```text
[shell_bootstrap]
  -> [daemon_attach]
  -> [session_load]
  -> [planner_node]
  -> [plan_validate]
  -> [approval_wait]
    -> [rejected_idle]
    -> [approved_execute]
  -> [synthesis]
  -> [persist]
  -> [idle]
```

## Workflow Steps

### STEP 0: Shell Bootstrap

**Actor**: GovOn Shell Client

**Action**:
- 사용자가 `govon` 또는 `govon --session <id>`로 셸을 시작한다.
- CLI는 로컬 daemon에 연결을 시도한다.
- daemon이 없으면 자동 기동한다.

**Output**:
- `session_id`
- runtime connection

### STEP 1: Session Load

**Actor**: Local FastAPI Daemon + SQLite Store

**Action**:
- 새 세션이면 세션 레코드를 만든다.
- 기존 세션이면 transcript와 tool log를 불러온다.

### STEP 2: Input Handling

**Actor**: GovOn Shell Client + Local FastAPI Daemon

**Action**:
- `/help`, `/clear`, `/exit`는 shell control로 처리한다.
- 일반 자연어 요청은 task loop로 전달한다.

### STEP 3: Planner Node

**Actor**: LangGraph Agent Runtime + Base Model

**Action**:
- 현재 요청과 세션 맥락을 함께 읽는다.
- 이번 요청을 하나의 task loop로 정리한다.
- tool metadata를 읽고 어떤 capability 조합이 필요한지 구조화한다.
- 예:
  - 초안 작성
  - 답변 수정
  - 근거 보강
  - 통계 보강

### STEP 4: Plan Validation

**Actor**: LangGraph Agent Runtime

**Action**:
- planner가 만든 구조화된 plan을 schema로 검증한다.
- 등록되지 않은 tool, 비MVP capability, 승인 불가 step을 제거 또는 차단한다.
- approval prompt와 executor가 같은 plan object를 공유하도록 고정한다.

예시:

- `rag_search`
- `api_lookup`
- `draft_civil_response`
- `append_evidence`

### STEP 5: Approval Prompt

**Actor**: GovOn Shell Client

**Action**:
- 사용자가 이해할 수 있는 쉬운 문장으로 작업 설명을 보여준다.
- 두 버튼만 제공한다:
  - `승인`
  - `거절`

### STEP 6A: Rejected Idle

**Actor**: GovOn Shell Client

**Action**:
- 사용자가 거절하면 아무런 추가 출력 없이 대기 상태로 돌아간다.

### STEP 6B: Approved Execute

**Actor**: LangGraph Agent Runtime

**Action**:
- 필요 tool을 실행한다.
- 답변 작성 task면 civil-response adapter를 함께 attach한다.
- 수정 요청이면 다시 retrieval을 돌릴 수 있다.
- 근거 보강 요청이면 원 질문 + 기존 답변을 함께 검색 기준으로 사용한다.

### STEP 7: Synthesis

**Actor**: LangGraph Agent Runtime

**Action**:
- 초안 작성 task면 다음 순서로 출력한다.
  1. 근거 요약
  2. 최종 초안
- 근거 보강 task면 기존 답변 아래에 `근거/출처` 섹션을 추가한다.

### STEP 8: Persist

**Actor**: SQLite Store

**Action**:
- transcript 저장
- tool log 저장
- 세션 재개 가능 상태 유지

## Routing Policy

- `/help`, `/clear`, `/exit` 같은 shell control만 rule-based로 처리한다.
- 업무 요청의 tool selection은 planner LLM이 session context, 기존 답변 존재 여부, tool metadata를 읽고 결정한다.
- 정규식 패턴 매칭은 테스트 fixture나 임시 fallback일 수는 있어도 MVP 정본 orchestration이 아니다.
- executor는 승인된 plan에 없는 capability를 실행하지 않는다.

## Test Cases

| TC | Trigger | Expected behavior |
|---|---|---|
| TC-01 | `govon` 실행 | 로컬 daemon 자동 연결 또는 자동 기동 |
| TC-02 | 답변 초안 요청 | 승인 요청 후 검색 + 초안 생성 |
| TC-03 | 사용자가 거절 | 아무런 추가 업무 출력 없이 idle 복귀 |
| TC-04 | 사용자가 수정 요청 | 새 task loop로 재검색 후 수정 초안 생성 |
| TC-05 | 사용자가 근거 요청 | 기존 답변 아래에 근거/출처 섹션 추가 |
| TC-06 | `govon --session <id>` | transcript와 tool log를 불러와 세션 재개 |
| TC-07 | 애매한 요청 | planner가 session context를 읽고 tool 필요 여부를 판단 |
