# GovOn Release Refactor: Shell-First First Release

**Updated**: 2026-04-01
**Status**: Active
**Decision**: 첫 릴리즈는 `터미널 설치 + shell/bash 대화형 에이전트 + graph-based 의사결정 프레임워크`를 기준으로 재정의한다.

## Release Goal

첫 공개 릴리즈의 제품 형태는 기존의 웹 UI 또는 행정 시스템 사이드 패널이 아니라 다음과 같이 정의한다.

- 사용자는 `pip install GovOn` 또는 오프라인 번들 설치 후 `govon` 명령으로 진입한다.
- `govon`은 shell/bash 안에서 실행되는 대화형 에이전트 셸이다.
- 첫 릴리즈 안에 LLM의 도구 선택과 중단/재시도/합성을 제어하는 graph-based agentic orchestration 계층을 포함한다.
- GovOn은 기존 행정 시스템을 대체하지 않고, 공무원이 복사/붙여넣기로 병행 사용하는 독립형 업무 어시스턴트다.
- 첫 릴리즈는 현재 이미 구현된 `추론 API + 검색 + 패키징 기반`을 재사용해 가장 빠르게 출하 가능한 형태를 목표로 한다.

## Current Baseline

### 이미 구축된 것

- 데이터 수집 및 전처리 파이프라인
  - `src/data_collection_preprocessing/`
- QLoRA 학습 및 AWQ 양자화 스크립트
  - `src/training/`
  - `src/quantization/`
- FastAPI 기반 추론 API
  - `/v1/classify`
  - `/v1/generate-public-doc`
  - `/v1/generate-civil-response`
  - `/v1/generate` (legacy compatibility)
  - `/v1/stream`
  - `/v1/search`
- FAISS + BM25 기반 하이브리드 검색
  - `src/inference/retriever.py`
  - `src/inference/hybrid_search.py`
  - `src/inference/index_manager.py`
- 기본 에이전트 페르소나 로더
  - `src/inference/agent_manager.py`
- DB 모델, CRUD, Alembic 마이그레이션 초안
  - `src/inference/db/`
- Python 패키지/오프라인 배포 워크플로우
  - `.github/workflows/publish-package.yml`
  - `.github/workflows/offline-package.yml`

### 아직 없는 것

- `govon` 대화형 셸 엔트리포인트
- shell/TUI 세션 UX
- shell에서 재개 가능한 대화 세션 저장/복구 흐름
- shell 전용 출력 포맷과 slash command 체계
- 런타임 부트스트랩 명령 (`serve`, `doctor`, `index`, `health`)
- graph-based agentic decision framework (LangGraph 또는 동급 orchestration 계층)
- tool selection guardrail, checkpoint, recovery 정책
- 브라우저/사이드 패널 웹 UI

## GitHub Issue Findings

### 핵심 문제

- 현재 공개 로드맵과 대부분의 M3 이슈는 `웹 UI`와 `agentic framework`를 한 덩어리로 섞어 관리하고 있다.
- 실제 코드베이스는 `runtime/API/search`는 존재하지만 `shell client`, `session shell UX`, `graph-based decision layer`, `tool guardrail`, `웹 UI`는 아직 미구현이다.
- 오픈 child task 이슈가 중복 생성되어 있어 실행 순서가 흐려져 있다.

### 중복 이슈 클러스터

아래 클러스터는 의미가 겹치므로, 실제 GitHub 정리 전까지는 **Initiative 이슈를 상위 기준**으로 보고 child task는 참조용으로만 사용한다.

- Workstream 1.x
  - `#376-#378`
  - `#389-#391`
  - `#408`
- Workstream 2.x
  - `#379-#380`
  - `#392-#393`
  - `#409-#410`
- Workstream 3.x
  - `#381-#384`
  - `#394-#397`
  - `#416`
- Workstream 4.x
  - `#385-#386`
  - `#398-#399`
  - `#411-#412`
- Workstream 5.x
  - `#387-#388`
  - `#400-#401`
  - `#413-#414`

## Refactor Policy

기존 GitHub 이슈는 당장 삭제하지 않고 아래 정책으로 재분류한다.

### R1에 직접 사용

- `#402` Public Roadmap
- `#367` AI 오케스트레이터 아키텍처
- `#368` Tool 통합
- `#372` 인프라 및 배포
- `#373` 테스트 및 품질 보증
- `#374` 프로젝트 문서화 및 마무리
- `#375` AIOSS 최종 납품
- `#406` 에이전틱 의사결정 프레임워크
- `#407` 에이전틱 아키텍처 문서 동기화
- `#409`, `#410`, `#415`, `#416`, `#417`, `#418`
  - graph 기반 의사결정과 tool orchestration 세부 태스크
- `#129` Agent 세션 관리 시스템
- `#405` 폐쇄망 배포(Offline) 스크립트 최적화

### R1에 부분 전용 또는 재목적화

- `#369` 기존 웹 Chat UI 이슈
  - 첫 릴리즈에서는 `브라우저 UI` 대신 `interactive shell client` 범위로 해석
- `#132` 스트리밍 UI
  - 첫 릴리즈에서는 `streaming shell output`으로 해석
- `#140` 행정 시스템 사이드 패널 UI
  - 첫 릴리즈에서는 `shell session layout + command palette`로 해석
- `#144` 프론트엔드-백엔드 API 연동
  - 첫 릴리즈에서는 `shell client-runtime API adapter`로 해석
- `#161` 출처 표시 API 연동
  - 첫 릴리즈에서는 `shell sources/citations view`로 해석
- `#404` 컨테이너화
  - 첫 릴리즈에서는 `runtime container + optional shell companion packaging`으로 축소

### R1 이후로 defer

- `#370`, `#371`, `#141`, `#133-#139`, `#50`, `#57`, `#63`
  - 웹 UI/디자인/사이드 패널은 R2 이후
- `#414`
  - legacy 공문서 품질 메트릭 이슈는 새 shell-first 품질 평가 태스크(`#401`)로 대체

## Refactored Workstreams

### Workstream A: Runtime Hardening for Shell Release

**Primary Issues**: `#367`, `#368`, `#129`
**Goal**: 현재 있는 FastAPI + 검색 + 생성 기능을 shell client가 안정적으로 사용할 수 있는 런타임으로 정리

#### A.1 API contract 고정

- 분류, 공문서 생성, 민원답변 생성, 스트리밍, 검색 응답을 shell client가 쓰기 쉬운 형태로 정리
- 현재 built-in tool catalog는 존재하지만, runtime의 tool registry는 후속 요구사항에 맞게 확장 가능해야 한다
- retrieved cases / search results / 에러 메시지 포맷 명확화
- health output을 shell `doctor` 명령에서 그대로 재사용 가능하게 정리

#### A.2 Session persistence 최소 구현

- shell session 기준의 대화 저장 구조 정의
- DB 또는 로컬 SQLite 중 개발/배포 기본값 결정
- 세션 생성, 이어쓰기, 최근 세션 조회, 종료 흐름 정의

#### A.3 Source-aware response 개선

- 검색 결과를 shell에서 읽기 쉬운 블록으로 변환
- 근거와 초안의 출력 순서를 고정
- 복사 가능한 최종본과 참고자료 구분

**Acceptance Criteria**:

- shell client가 `/v1/classify`, `/v1/generate-public-doc`, `/v1/generate-civil-response`, `/v1/stream`, `/v1/search`와 extensible tool registry만으로 1차 업무 플로우를 수행
- session id 기준으로 최소 1회 이상 대화 이어쓰기 가능
- 검색/생성 실패 시 shell에 구조화된 에러를 출력

### Workstream B: Agentic Decision Framework

**Primary Issues**: `#406`, `#407`, `#409`, `#410`, `#415`, `#416`, `#417`, `#418`
**Goal**: LLM이 무작정 tool을 호출하지 않도록 stateful graph와 명시적 guardrail을 갖춘 agentic runtime을 첫 릴리즈에 포함

#### B.1 Graph runtime and state schema

- 세션 상태, 사용자 의도, tool 후보, 이전 근거, 최종 초안 버전을 하나의 graph state로 정의
- model adapter와 tool registry를 graph 노드에서 공통 사용
- 노드 실행 trace를 audit log와 shell transcript에 남길 수 있도록 설계

#### B.2 Decision policy and guardrails

- 질문 유형에 따라 `답변 직접 생성`, `검색 우선`, `검색 후 초안 생성`, `재질문` 중 무엇을 할지 결정
- low-confidence 또는 빈 검색 결과에서 바로 tool 남발하지 않도록 stop/retry/ask-back 규칙 정의
- tool별 입력 검증과 호출 가능 조건 정의

#### B.3 Tool execution and synthesis nodes

- 검색/외부 API/초안 생성 action을 graph node로 연결
- tool 결과, citations, 실패 정보를 표준 state 필드로 정규화
- 최종 합성 단계에서 근거와 초안을 함께 생성

#### B.4 Checkpoint, recovery, and human-in-the-loop

- turn 단위 checkpoint 저장
- 실패 시 해당 노드부터 재시도 가능
- shell에서 `/retry`, `/sources`, `/session resume` 같은 제어 흐름과 연결

**Acceptance Criteria**:

- 한 번의 요청에서 multi-step tool orchestration이 state graph 기준으로 동작
- 불필요한 tool 호출을 줄이는 guardrail이 존재
- graph node trace와 checkpoint가 세션 저장소와 연결
- shell client가 graph 기반 runtime과 end-to-end로 동작

### Workstream C: Interactive GovOn Shell

**Primary Issues**: `#369` partial, `#132` partial, `#144` partial
**Goal**: `govon` 명령으로 진입하는 대화형 셸 제공

#### B.1 `govon` 엔트리포인트 추가

- `pyproject.toml`에 제품용 entrypoint 추가
- 초기 진입 화면, 연결 상태, 환경 확인, 기본 도움말 제공

#### B.2 대화형 셸 UX

- 기본 자유 입력
- slash commands:
  - `/help`
  - `/classify`
  - `/search`
  - `/draft`
  - `/sources`
  - `/session`
  - `/copy`
  - `/exit`
- 멀티라인 민원 붙여넣기 지원

#### B.3 스트리밍 출력

- `/v1/stream` 기반 토큰 스트리밍
- thinking/searching/generating 상태 텍스트
- 완료 시 최종 초안과 근거 요약 분리 표시

#### B.4 복사/내보내기

- 최종 초안만 출력
- 세션 transcript 저장
- markdown/text export

**Acceptance Criteria**:

- 사용자가 `govon` 실행 후 별도 웹 UI 없이 대화형으로 민원 분류, 검색, 초안 생성을 수행
- shell에서 민원 본문 붙여넣기 후 한 세션 내 반복 수정 가능
- `govon --help`와 `/help` 모두 동작

### Workstream D: Packaging, Install, Offline Delivery

**Primary Issues**: `#372`, `#404`, `#405`, `#375`
**Goal**: shell-first 제품을 실제 설치/배포 가능한 형태로 출하

#### C.1 Python package 배포 정리

- `pip install GovOn` 또는 extras 기반 설치 경로 정의
- runtime 의존성과 client 의존성 분리
- 버전/릴리즈 태그 정책 정리

#### C.2 Offline package 재구성

- 현재 Docker image 중심 번들에 shell 실행 경로 추가
- 오프라인 환경에서 `govon` client와 runtime을 함께 기동하는 절차 문서화
- smoke-test를 shell flow 기준으로 보강

#### C.3 운영 명령 정의

- `govon serve`
- `govon doctor`
- `govon health`
- `govon session list`
- `govon config init`

**Acceptance Criteria**:

- 온라인 설치: 패키지 설치 후 `govon` 명령 사용 가능
- 오프라인 설치: 번들 해제 후 shell client + runtime 기동 가능
- 운영자 문서만 읽고 환경 구성 가능

### Workstream E: Release QA, Docs, Acceptance

**Primary Issues**: `#373`, `#374`, `#375`, `#400`, `#60`, `#61`, `#59`
**Goal**: shell-first 릴리즈를 실제 납품 가능한 상태로 검증

#### D.1 Shell E2E 테스트

- 세션 시작
- 민원 붙여넣기
- 분류
- 검색
- 초안 생성
- 세션 저장/재개

#### D.2 설치/운영 문서

- online install
- offline install
- runtime bootstrap
- shell command guide
- troubleshooting

#### D.3 사용자 검증

- 공무원 또는 프로젝트 팀 기준으로 복붙 기반 업무 플로우 검증
- 기존 행정 시스템 옆에서 병행 사용 가능한지 확인

**Acceptance Criteria**:

- shell 기준 핵심 E2E 시나리오 통과
- 설치 문서로 재현 가능
- 첫 릴리즈 정의와 README/roadmap/tasklist가 일치

## First Release Scope

### In

- `govon` interactive shell
- graph-based agentic orchestration
- tool selection guardrail and checkpoint
- classify/search/draft workflow
- session persistence
- runtime health/doctor
- online/offline install
- release docs and smoke test

### Out

- 브라우저 기반 Chat UI
- 행정 시스템 내장 사이드 패널
- Figma 기반 디자인 시스템
- 웹/앱 전용 시각 디자인 고도화

## Release Sequence

### R1: GovOn Agentic Shell

- 설치형 대화형 shell
- graph-based decision framework
- tool orchestration + checkpoint
- 복붙 기반 실무 보조

### R2: Side Panel / Web UI

- 브라우저 UI
- 행정 시스템 병행형 패널
- 디자인 시스템

## GitHub Cleanup Status

이번 refactor에서 아래 정리를 반영했다.

1. `#402` roadmap body를 shell-first release 기준으로 수정
2. Initiative canonical set을 `#367`, `#368`, `#369`, `#372`, `#373`, `#374`, `#375`, `#406`, `#407`로 재정의
3. 상세 태스크 canonical set을 `#129`, `#132`, `#140`, `#144`, `#161`, `#392-#397`, `#400-#405`, `#409`, `#410`, `#415-#418`, `#60`, `#62`, `#41`로 재배치
4. `Deferred Post-R1` 표기는 UI 고도화 이슈 중심으로 축소
5. 남은 후속 정리는 실제 구현이 시작된 뒤 중복 이슈를 닫는 cleanup pass로 수행

## Status Summary

- 현재 코드베이스는 `runtime/API/search` 출하 준비도가 높다.
- 현재 GitHub 이슈 구조는 `agentic framework`와 `UI 고도화`가 섞여 있어 우선순위가 흐려져 있다.
- 따라서 첫 릴리즈는 `GovOn Agentic Shell`로 고정하고, shell-first + agentic-first 기준으로 issue 해석과 구현 순서를 재배치한다.
