# Work Breakdown Structure (WBS)
## GovOn CLI Shell MVP

**프로젝트 기간**: R1 기준 16주  
**작성일**: 2026-04-03  
**기준 문서**: [TO-BE Architecture SVG](https://govon-org.github.io/GovOn/govon-tobe-architecture.svg)

---

## 진행률 요약

| Workstream | 핵심 산출물 |
|----------|-------------|
| WS-1 | Civil-response adapter |
| WS-2 | Local daemon runtime + SQLite session store |
| WS-3 | LangGraph ToolNode + Tier 0/1 도구 + local RAG |
| WS-4 | LangGraph ReAct agent + approval gate runtime |
| WS-5 | Interactive CLI shell |
| WS-6 | 설치/패키징 |
| WS-7 | 테스트 및 품질 검증 |
| WS-8 | 문서화 및 최종 납품 |

---

## Milestone 1: Architecture Freeze and Runtime Basis

### 1.1 제품 경계 확정

- [ ] CLI-first MVP architecture freeze
- [ ] LangGraph 기반 approval-gated task loop specification
- [ ] shell control command scope freeze
- [ ] public-doc / classification exclusion confirmation

### 1.2 로컬 런타임 기반

- [ ] FastAPI local daemon contract 정의
- [ ] daemon 내부 LangGraph runtime lifecycle 정의
- [ ] daemon auto-start / reconnect 정책 정의
- [ ] SQLite session schema 정의
- [ ] runtime health/status contract 정의

### 1.3 Tool 경계 정의

- [ ] `adapters.yaml` 기반 동적 도구 등록 구조 정의
- [ ] Tier 0 도구 (`rag_search`, `api_lookup`, `stats_lookup` 등) contract 정의
- [ ] Tier 1 어댑터 도구 (`public_admin_adapter`, `legal_adapter`) contract 정의
- [ ] `bind_tools()` 스키마 → LLM tool_call 연동 정의
- [ ] approval_wait interrupt payload 정의

### Milestone 1 완료 기준

- [ ] canonical architecture 문서 승인
- [ ] PRD/WBS/ADR가 동일한 제품 경계를 설명
- [ ] LangGraph가 MVP 필수 의존으로 반영돼 있다.
- [ ] 정규식 기반 business router가 정본에서 제거돼 있다.
- [ ] roadmap / workstream / task 이슈 구조가 문서와 일치

---

## Milestone 2: Civil Drafting and Tooling MVP

### 2.1 Civil-response adapter

- [ ] civil-response adapter 학습 데이터 확보
- [ ] 데이터 전처리 및 검증
- [ ] 단일 adapter 학습 및 평가
- [ ] adapter attach policy 정의

### 2.2 Tool layer

- [ ] StructuredTool 팩토리 (`build_search_tools`, `build_analysis_tools`, `build_adapter_tools`) 구현
- [ ] `adapters.yaml`에서 어댑터 도구 자동 등록 구현
- [ ] local RAG ingestion / retrieval 구현
- [ ] `get_tool_approval_map()` — requires_approval 메타데이터 기반 Tier 분류 구현

### 2.3 Runtime loop

- [ ] agent 노드: `bind_tools()` + ReAct 루프 구현
- [ ] `route_agent()` — tool_calls 유무 및 Tier 분류 라우팅 구현
- [ ] `approval_wait` 노드: `interrupt()` + approve/reject 구현
- [ ] `route_after_approval()` — 승인 시 ToolNode, 거절 시 agent 재진입 구현

### Milestone 2 완료 기준

- [ ] 민원 답변 초안 생성이 동작한다.
- [ ] agent가 `bind_tools()`로 전달받은 스키마를 읽고 자율적으로 도구를 선택한다.
- [ ] Tier 1 도구 실행 전 승인 절차가 동작한다.
- [ ] 거절 시 agent가 대안을 제시한다.

---

## Milestone 3: CLI Shell and Evidence Augmentation

### 3.1 CLI shell

- [ ] interactive prompt 구현
- [ ] daemon attach / auto-start 구현
- [ ] 상태 표시 및 approval UI 구현
- [ ] `govon --session <id>` 재개 구현

### 3.2 Evidence 수집

- [ ] ToolMessage에서 evidence 자동 추출 구현 (persist 노드)
- [ ] 후속 질문 시 agent가 ReAct 루프로 검색 도구 재호출 구현
- [ ] RAG provenance를 `파일경로 + 페이지`로 정규화

### 3.3 RAG validation

- [ ] 샘플 문서 폴더 기반 ingestion 검증
- [ ] `pdf/hwp/docx/txt/html` 파서 검증
- [ ] 검색 정확성 및 인용 일관성 확인

### Milestone 3 완료 기준

- [ ] CLI에서 세션 시작/재개/종료가 가능하다.
- [ ] 후속 질문 시 agent가 검색 도구를 재호출하여 근거를 보강한다.
- [ ] 샘플 문서 기반 RAG가 mixed-format에서 동작한다.

---

## Milestone 4: Packaging, QA, Docs, Delivery

### 4.1 Packaging

- [ ] daemon + shell 설치 자산 정리
- [ ] 로컬 실행 runbook 작성
- [ ] 로그/설정/문서 경로 정리

### 4.2 Quality assurance

- [ ] approval-gated E2E 테스트
- [ ] session resume 테스트
- [ ] ReAct 루프 evidence 수집 테스트
- [ ] latency / stability benchmark

### 4.3 Documentation and delivery

- [ ] 사용자 가이드
- [ ] 운영 가이드
- [ ] architecture / ADR / PRD / WBS 정합성 확인
- [ ] demo package / release note / known issues 정리

### Milestone 4 완료 기준

- [ ] `govon` MVP 설치와 실행이 재현 가능하다.
- [ ] 핵심 E2E 시나리오가 통과한다.
- [ ] 문서와 실제 동작이 일치한다.
- [ ] v1.0.0 전달 패키지가 완성된다.

---

## 핵심 의존 관계

```text
Architecture freeze
    -> runtime/approval loop 정리
    -> tool layer 정리
    -> shell UX 구현
    -> packaging/QA/docs
```

## 주요 리스크

| 리스크 | 영향 | 대응 |
|--------|------|------|
| adapter 품질 불안정 | 초안 품질 편차 | 데이터 검증 범위를 민원 답변 중심으로 축소 |
| RAG 원문 부족 | 근거 보강 품질 저하 | 샘플 문서로 parser/retrieval 먼저 검증 후 운영 문서로 확장 |
| agent의 불필요한 도구 호출 | 승인 설명 품질 저하 및 오동작 | Tier 0/1 분류, approval gate, 도구 description 정밀화, E2E 회귀 테스트로 통제 |
| 승인 UX 미완성 | 사용자 신뢰 저하 | approval-gated E2E를 MVP 핵심 acceptance로 둠 |
| daemon/session 불안정 | resume 실패 | LangGraph checkpointer 세션 복원 테스트를 필수화 |

---

**작성자**: GovOn Team  
**최종 수정**: 2026-04-03
