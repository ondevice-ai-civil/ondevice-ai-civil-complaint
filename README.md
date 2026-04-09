# GovOn

GovOn은 행정 업무를 보조하는 **에이전틱 CLI 셸**이다. 사용자는 `govon`을 실행한 뒤 자연어로 요청하고, 셸은 로컬 daemon runtime과 연결되어 검색·조회·작성 도구를 승인 기반으로 사용한다.

[![Docs Portal](https://img.shields.io/badge/Docs-Portal-blue?logo=readthedocs)](https://govon-org.github.io/GovOn/)
[![Public Roadmap](https://img.shields.io/badge/Public_Roadmap-Workstreams-7C3AED)](https://github.com/GovOn-Org/GovOn/issues?q=label%3A%22%F0%9F%A7%AD+Workstream%22+sort%3Aupdated-desc)

<!-- DORA-BADGES:START -->
![DORA Grade](https://img.shields.io/badge/DORA-Elite-brightgreen)
![Deploy Freq](https://img.shields.io/badge/Deploy_Freq-30%2Fweek-blue)
![Lead Time](https://img.shields.io/badge/Lead_Time-0.9h-brightgreen)
![CFR](https://img.shields.io/badge/CFR-28.3%2525-yellow)
![MTTR](https://img.shields.io/badge/MTTR-0.0h-brightgreen)
<!-- DORA-BADGES:END -->

## 아키텍처

> ReAct + ToolNode 기반 v4 아키텍처. LLM이 자율적으로 도구를 선택하며, 정적 planner/executor를 제거했다.

<p align="center">
  <a href="https://govon-org.github.io/GovOn/govon-tobe-architecture.svg">
    <img src="docs/architecture/govon-tobe-architecture.svg" alt="GovOn TO-BE Architecture" width="100%"/>
  </a>
</p>

### 모델 구성

베이스 LLM **EXAONE 4.0-32B-AWQ** 단일 모델이 사용자 쿼리의 의도를 분석하고 `tool_call`을 자율 결정한다. 어댑터 도구가 호출되면 해당 LoRA를 per-request로 attach하여 추론한다.

| 구성 요소 | LoRA | tool_call 시 동작 |
|---|---|---|
| 베이스 모델 (EXAONE 4.0-32B-AWQ) | 없음 | 의도 분석 · `bind_tools()` · ReAct 루프 · Tier 0 도구 실행 |
| [**civil-adapter**](https://huggingface.co/umyunsang/govon-civil-adapter) (r16) | 74K 민원-답변 쌍 학습 | `public_admin_adapter` tool_call 시 per-request attach |
| [**legal-adapter**](https://huggingface.co/siwo/govon-legal-adapter) (r16) | 270K 법률 문서 학습 | `legal_adapter` tool_call 시 per-request attach |

## 데이터 파이프라인

```mermaid
flowchart LR
    subgraph Sources["Data Sources"]
        A1["AI Hub 71852\nPublic Civil QA\n29K"]
        A2["AI Hub 71847\nAdmin Law QA\n37K"]
        A3["AI Hub 71841/43/48\nCivil/IP/Criminal\n200K"]
        A4["HF Precedents\nCourt Decisions\n85K"]
    end

    subgraph Civil["Civil Adapter"]
        B1["parsers.py"] --> B2["train 33K"]
        B2 --> B3["HF Hub"]
    end

    subgraph Legal["Legal Adapter"]
        C1["build_dataset.py"] --> C2["train 243K"]
        C2 --> C3["HF Hub"]
    end

    subgraph Training["Unsloth QLoRA"]
        D1["EXAONE 4.0-32B\n4-bit NF4"] --> D2["LoRA r16"]
        D2 --> D3["HF Spaces\nL40S 48GB"]
    end

    A1 --> B1
    A2 --> C1
    A3 --> C1
    A4 --> C1
    B3 --> D1
    C3 --> D1
```

| 데이터셋 | 건수 | HuggingFace Hub |
|---|---|---|
| Civil Response | 33K (train) | [umyunsang/govon-civil-response-data](https://huggingface.co/datasets/umyunsang/govon-civil-response-data) |
| Legal Citation | 243K (train) | [umyunsang/govon-legal-response-data](https://huggingface.co/datasets/umyunsang/govon-legal-response-data) |

## LangGraph Agent Flow

```
START → session_load → agent → [route_agent]
     ├── (no tool_calls)   → persist → END
     ├── (all Tier 0)      → tools → agent → ...  (ReAct loop)
     └── (needs approval)  → approval_wait → [route_after_approval]
                                 ├── (approved) → tools → agent → ...
                                 └── (rejected) → agent → ...  (suggest alternatives)
```

## 현재 제품 기준

- 진입점은 웹이 아니라 `govon` 대화형 CLI 셸
- 내부 runtime은 로컬 FastAPI daemon 또는 원격 서버 (`GOVON_RUNTIME_URL`)
- LangGraph ReAct 루프에서 agent LLM이 자율적으로 도구 호출을 결정
- 도구 선택은 EXAONE 4.0의 `bind_tools()` + 네이티브 tool calling으로 수행
- Tier 0 도구(검색/분석)는 자동 실행, Tier 1 도구(어댑터)는 사용자 승인 후 실행
- 민원 답변 작성 시 `public_admin_adapter`, 법률 근거 시 `legal_adapter` LoRA tool 사용
- 거부 시 agent가 대안을 제시하는 루프 구조
- 서빙은 HuggingFace Spaces ZeroGPU 또는 전용 GPU Space

상세 기준 문서는 [docs/architecture/GovOn-shell-mvp-architecture.md](docs/architecture/GovOn-shell-mvp-architecture.md)다.

## MVP 범위

포함:

- 자연어 기반 CLI 셸 + 로컬 daemon / 원격 서버 연결
- ReAct 루프 기반 LLM 자율 도구 선택 (`bind_tools`)
- Tier 0 자동 실행 도구: `rag_search`, `api_lookup`, `stats_lookup`, `keyword_analyzer`, `demographics_lookup`, `issue_detector`
- Tier 1 승인 필요 도구: `public_admin_adapter` (민원 답변 LoRA), `legal_adapter` (법률 근거 LoRA)
- Human-in-the-loop 승인 게이트 (`interrupt()` + approve/reject)
- 거부 시 agent 대안 제시 루프
- `adapters.yaml` 기반 동적 도구 등록 (코드 변경 없이 어댑터 추가)
- LangGraph checkpointer 기반 세션 resume

제외:

- 공문서 작성
- 웹/앱 제품화

## 사용자 흐름

1. 사용자가 `govon`을 실행하면 CLI가 로컬 daemon에 연결한다.
2. 사용자가 자연어로 업무를 요청한다.
3. `session_load` 노드가 기존 대화 컨텍스트를 복원한다.
4. `agent` 노드(LLM + `bind_tools`)가 요청을 분석하여 도구 호출 또는 직접 답변을 결정한다.
5. Tier 0 도구(검색/분석)만 호출되면 `ToolNode`가 즉시 병렬 실행하고 결과를 agent에 반환한다 (ReAct loop).
6. Tier 1 도구(어댑터)가 포함되면 `approval_wait`에서 `interrupt()` — 사용자에게 승인/거절 UI를 보여준다.
7. 승인 시 `ToolNode`가 실행하고 agent로 재진입, 거절 시 agent가 대안을 제시한다.
8. agent가 충분한 정보를 모으면 최종 답변을 직접 작성한다 (tool_calls 없음 → `persist` → END).
9. `persist` 노드가 대화 턴과 evidence를 DB에 저장하고 세션 ID를 반환한다.

## 문서

- 아키텍처 다이어그램: [TO-BE Architecture SVG](https://govon-org.github.io/GovOn/govon-tobe-architecture.svg)
- ADR: [docs/adr/README.md](docs/adr/README.md)
- PRD: [docs/prd.md](docs/prd.md)
- WBS: [docs/wbs.md](docs/wbs.md)
- 공식 문서: [docs/official](docs/official)
- v3 아키텍처 아카이브: [docs/archive/v3-planner-era/](docs/archive/v3-planner-era/)

## GitHub 이슈 구조

- root roadmap: `#402`
- roadmap의 하위: `workstream`
- workstream의 하위: `task`
- 세부 작업 내용은 `task` 이슈 본문에만 작성한다.

## DORA Metrics

<!-- latest-dora.png는 워크플로우 첫 실행 후 자동 생성됩니다. 실시간 대시보드는 아래 Grafana 링크를 이용하세요. -->

| 지표 | 설명 | 수집 방식 |
|------|------|----------|
| Deployment Frequency | main 머지 PR 수 / 주 | GitHub API |
| Lead Time | PR 첫 커밋 → 머지 평균 시간 | GitHub API |
| Change Failure Rate | hotfix/revert 커밋 비율 | git log |
| MTTR | bug 이슈 open → close 평균 시간 | GitHub API |

- **실시간 대시보드**: [Grafana Cloud](https://umyunsang.grafana.net/d/govon-dora/govon-dora-metrics-dashboard?orgId=1&from=now-7d&to=now&timezone=Asia%2FSeoul)
- **주간 보고서**: [`metrics/reports/`](metrics/reports/)
- **수집 워크플로우**: [`.github/workflows/dora-metrics.yml`](.github/workflows/dora-metrics.yml)

## 개발 규칙

기여 전 아래 문서를 먼저 본다.

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [site/docs/guide/development.md](site/docs/guide/development.md)

브랜치는 GitHub Flow를 사용하고, 기본 대상 브랜치는 항상 `main`이다.
