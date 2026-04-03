# GovOn

GovOn은 행정 업무를 보조하는 **에이전틱 CLI 셸**이다. 사용자는 `govon`을 실행한 뒤 자연어로 요청하고, 셸은 로컬 daemon runtime과 연결되어 검색·조회·작성 도구를 승인 기반으로 사용한다.

[![Docs Portal](https://img.shields.io/badge/Docs-Portal-blue?logo=readthedocs)](https://govon-org.github.io/GovOn/)
[![Public Roadmap](https://img.shields.io/badge/Public_Roadmap-Workstreams-7C3AED)](https://github.com/GovOn-Org/GovOn/issues?q=label%3A%22%F0%9F%A7%AD+Workstream%22+sort%3Aupdated-desc)

## 현재 제품 기준

- 진입점은 웹이 아니라 `govon` 대화형 CLI 셸
- 내부 runtime은 로컬 FastAPI daemon
- LangGraph state graph 안에서 planner LLM이 의도 파악, 작업 계획, tool 선택 담당
- 민원 답변 작성 단계에서만 civil-response adapter 사용
- tool 실행은 작업 단위 승인 후 진행
- 근거/출처는 기본 출력이 아니라 후속 증강 작업으로 처리
- 업무용 tool routing의 정본은 정규식 패턴 매칭이 아니라 model-driven planning

상세 기준 문서는 [docs/architecture/GovOn-shell-mvp-architecture.md](docs/architecture/GovOn-shell-mvp-architecture.md)다.

## MVP 범위

포함:

- 자연어 기반 CLI 셸
- 로컬 daemon 자동 기동 및 재연결
- 민원 답변 작성
- 외부 API lookup
- 로컬 RAG 검색
- 작업 단위 승인 UI
- SQLite 기반 세션 resume
- 후속 근거/출처 증강

제외:

- 공문서 작성
- 분류 기능
- 웹/앱 제품화

## 상위 구조

```mermaid
graph TD
    A[govon CLI] --> B[Local FastAPI daemon]
    B --> C[LangGraph agent runtime]
    C --> D[Approval-gated task loop]
    D --> E[vLLM planner/model]
    D --> F[Civil-response adapter]
    D --> G[Tool registry]
    D --> H[(SQLite session DB)]
    G --> I[api_lookup]
    G --> J[local RAG]
```

## 사용자 흐름

1. 사용자가 `govon`을 실행한다.
2. CLI가 로컬 daemon을 자동 기동하거나 기존 daemon에 재연결한다.
3. 사용자가 자연어로 업무를 요청한다.
4. LangGraph planner node가 이번 턴의 한 작업과 필요한 tool 조합을 구조화한다.
5. 시스템이 쉬운 설명과 함께 `승인 / 거절` UI를 보여준다.
6. 승인되면 graph executor가 필요한 여러 tool과 adapter를 묶어서 실행한다.
7. 결과는 `근거 요약 -> 최종 초안` 순서로 출력한다.
8. 사용자가 후속으로 근거를 요청하면 RAG/API를 다시 사용해 기존 답변 아래에 근거 섹션을 추가한다.
9. 종료 시 세션 ID를 보여주고, `govon --session <id>`로 재개한다.

## 문서

- 제품 아키텍처: [docs/architecture/GovOn-shell-mvp-architecture.md](docs/architecture/GovOn-shell-mvp-architecture.md)
- 오케스트레이션 워크플로우: [docs/architecture/WORKFLOW-orchestrator-tool-calling.md](docs/architecture/WORKFLOW-orchestrator-tool-calling.md)
- ADR: [docs/adr/README.md](docs/adr/README.md)
- PRD: [docs/prd.md](docs/prd.md)
- WBS: [docs/wbs.md](docs/wbs.md)
- 공식 문서: [docs/official](docs/official)

## GitHub 이슈 구조

- root roadmap: `#402`
- roadmap의 하위: `workstream`
- workstream의 하위: `task`
- 세부 작업 내용은 `task` 이슈 본문에만 작성한다.

## 개발 규칙

기여 전 아래 문서를 먼저 본다.

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [site/docs/guide/development.md](site/docs/guide/development.md)

브랜치는 GitHub Flow를 사용하고, 기본 대상 브랜치는 항상 `main`이다.
