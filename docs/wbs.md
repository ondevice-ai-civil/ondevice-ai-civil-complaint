# Work Breakdown Structure (WBS)
## GovOn CLI Shell MVP

**프로젝트 기간**: R1 기준 16주  
**작성일**: 2026-04-03  
**기준 문서**: `docs/architecture/GovOn-shell-mvp-architecture.md`

---

## 진행률 요약

| Workstream | 핵심 산출물 |
|----------|-------------|
| WS-1 | Civil-response adapter |
| WS-2 | Local daemon runtime + SQLite session store |
| WS-3 | `api_lookup` + local RAG + evidence augmentation |
| WS-4 | Approval-gated task orchestration |
| WS-5 | Interactive CLI shell |
| WS-6 | 설치/패키징 |
| WS-7 | 테스트 및 품질 검증 |
| WS-8 | 문서화 및 최종 납품 |

---

## Milestone 1: Architecture Freeze and Runtime Basis

### 1.1 제품 경계 확정

- [ ] CLI-first MVP architecture freeze
- [ ] approval-gated task loop specification
- [ ] shell control command scope freeze
- [ ] public-doc / classification exclusion confirmation

### 1.2 로컬 런타임 기반

- [ ] FastAPI local daemon contract 정의
- [ ] daemon auto-start / reconnect 정책 정의
- [ ] SQLite session schema 정의
- [ ] runtime health/status contract 정의

### 1.3 Tool 경계 정의

- [ ] unified `api_lookup` capability contract 정의
- [ ] local `rag_search` capability contract 정의
- [ ] `draft_civil_response` / `append_evidence` output contract 정의

### Milestone 1 완료 기준

- [ ] canonical architecture 문서 승인
- [ ] PRD/WBS/ADR가 동일한 제품 경계를 설명
- [ ] roadmap / workstream / task 이슈 구조가 문서와 일치

---

## Milestone 2: Civil Drafting and Tooling MVP

### 2.1 Civil-response adapter

- [ ] civil-response adapter 학습 데이터 확보
- [ ] 데이터 전처리 및 검증
- [ ] 단일 adapter 학습 및 평가
- [ ] adapter attach policy 정의

### 2.2 Tool layer

- [ ] external API wrapper 구현
- [ ] local RAG ingestion / retrieval 구현
- [ ] mixed evidence normalization 구현
- [ ] evidence summary -> final draft synthesis 구현

### 2.3 Runtime loop

- [ ] 한 작업 기준 task planning 구현
- [ ] 사람말 approval prompt 구현
- [ ] 승인 시 다중 tool 묶음 실행 구현
- [ ] 거절 시 완전 idle 복귀 동작 구현

### Milestone 2 완료 기준

- [ ] 민원 답변 초안 생성이 동작한다.
- [ ] tool 실행 전 승인 절차가 동작한다.
- [ ] adapter attach가 작성 task에서만 동작한다.

---

## Milestone 3: CLI Shell and Evidence Augmentation

### 3.1 CLI shell

- [ ] interactive prompt 구현
- [ ] daemon attach / auto-start 구현
- [ ] 상태 표시 및 approval UI 구현
- [ ] `govon --session <id>` 재개 구현

### 3.2 Evidence augmentation

- [ ] 초안 작성 후 근거 추가 요청 처리 구현
- [ ] 원 질문 + 생성 답변 기준 재검색 구현
- [ ] 기존 답변 아래 `근거/출처` 섹션 추가 구현
- [ ] RAG provenance를 `파일경로 + 페이지`로 정규화

### 3.3 RAG validation

- [ ] 샘플 문서 폴더 기반 ingestion 검증
- [ ] `pdf/hwp/docx/txt/html` 파서 검증
- [ ] 검색 정확성 및 인용 일관성 확인

### Milestone 3 완료 기준

- [ ] CLI에서 세션 시작/재개/종료가 가능하다.
- [ ] 후속 근거 보강 요청이 동작한다.
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
- [ ] evidence augmentation 테스트
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
| 승인 UX 미완성 | 사용자 신뢰 저하 | approval-gated E2E를 MVP 핵심 acceptance로 둠 |
| daemon/session 불안정 | resume 실패 | SQLite 세션 복원 테스트를 필수화 |

---

**작성자**: GovOn Team  
**최종 수정**: 2026-04-03
