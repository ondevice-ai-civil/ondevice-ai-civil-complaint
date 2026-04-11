# GovOn MMP Harness Migration — 공공업무형 하네스 설계

**Status**: Design (not yet implemented)
**Related**: [ADR-005](../adr/ADR-005-harness-migration.md), PRD v6.0
**Last Updated**: 2026-04-11

---

## 1. 배경

GovOn MVP의 하네스는 개발자형 에이전틱 시스템(Claude Code, Cursor 등)의 패턴을 차용했다. 즉 **단일 tool call 중심의 ReAct 루프**이며, 사용자는 개발자/기획자처럼 "요청 → 제안 → 승인 → 결과" 사이클을 반복한다.

MMP의 타깃 사용자는 **부처·기관·부서의 공무원**이고, 이들의 업무는 단일 tool call이 아니라 **법정 절차로 정의된 다단계 flow**다. 본 문서는 MVP 하네스를 건드리지 않으면서 그 위에 **공공업무형 하네스 레이어**를 덧씌우는 설계를 기술한다.

본 문서는 구현 PR이 아니라 **설계/계획 문서**다. 구현은 후속 이슈에서 단계적으로 진행한다.

---

## 2. 목표와 비목표

### 목표

- 공공업무 flow(인허가, 민원, 행정처분 등)를 **YAML로 선언적으로 표현**할 수 있다.
- 결재선·법정기한·역할·감사 이력을 **1급 엔터티**로 관리한다.
- 기존 Tier 0/1 capability와 ReAct 루프는 **그대로 재사용**한다.
- `adapters.yaml`을 **`harness.yaml`로 통합**하여 adapter·org·role·procedure가 단일 이름공간을 공유하도록 한다.
- MMP 샘플로 **인허가 프로세스 1건**을 end-to-end로 demo할 수 있는 뼈대를 확보한다.

### 비목표

- 기존 행정 시스템(새올/온나라/정부24 등) 법적 대체.
- 공무원의 결재 권한을 시스템적으로 "구속"하는 것. MMP의 결재선은 **advisory**다.
- Web UI, BPMN 편집기, drag-and-drop 워크플로 에디터.
- MVP ReAct 루프의 수정/치환.
- 정적 planner/executor/synthesis 노드로의 퇴행.

---

## 3. 핵심 개념 (도메인 모델)

### 3.1 Case

한 건의 업무 사안(=민원 1건, 인허가 신청 1건). Case는 다음을 갖는다.

| 필드 | 설명 |
|------|------|
| `case_id` | UUID. |
| `procedure_id` | 어느 procedure 하에서 처리 중인지. |
| `org` | 주관 부처/부서 (ex. `도시계획과`). |
| `applicant` | 민원인/신청인 참조 (개인 식별정보는 해시). |
| `opened_at` | 접수 시각. |
| `legal_due_at` | 법정 처리기한. procedure가 선언한 `sla_hours` 기준으로 계산. |
| `state` | `intake | in_progress | awaiting_approval | notified | closed | rejected`. |
| `evidence_refs` | 수집된 근거(Evidence) 리스트. |
| `approvals` | 결재 이력(승인자, 역할, 시각, 의견). |
| `trace_id` | 감사 추적용. 각 LLM 호출·tool 호출·approval 이벤트의 부모 키. |

Case는 `SessionContext`와 **다른** 영속 객체다. 하나의 Session에 여러 Case가 존재할 수 있고, 하나의 Case는 여러 Session에 걸쳐 재개될 수 있다.

### 3.2 Procedure

업무 flow의 템플릿. `procedures/<domain>/<name>.yaml`로 정의한다.

### 3.3 Role

결재·실행 권한의 최소 단위. MMP는 RBAC로 시작한다. 예: `civil_servant`, `team_lead`, `section_chief`.

### 3.4 Org

기관/부서 식별자. `harness.yaml`의 `orgs` 섹션에 선언된다. Role은 Org 내부에 귀속된다.

### 3.5 Step

Procedure의 실행 단위. 다음 타입을 갖는다.

- `capability` — 기존 Tier 0 capability 호출.
- `adapter` — 기존 Tier 1 도메인 어댑터 호출.
- `approval` — 결재선의 특정 role 승인 요구.
- `handoff` — 다른 기관/부서로 이관 (감사 기록).
- `notice` — 민원인/관계자 통보 (outbound).
- `wait` — 외부 이벤트/시간 조건 대기.

### 3.6 Agentic 자유도

Procedure는 **제약(constraint)** 을 선언할 뿐, Step 내부의 tool 선택은 여전히 LLM이 ReAct 루프에서 수행한다. 예를 들어 `legal_check` step은 "legal_adapter를 쓰라"가 아니라 "이 step이 끝나려면 법적 근거 evidence가 최소 1건 있어야 한다"는 post-condition만 선언한다.

---

## 4. 시스템 아키텍처

```
CLI (govon)
  │
  ▼
FastAPI runtime
  │
  ├── Procedure Runner  (신규 · Layer 4)
  │     - case store (SQLite/append-only)
  │     - state machine (step 전이/타임아웃)
  │     - approval router (role 매칭)
  │     - audit logger
  │
  ▼
ReAct Agent Loop  (기존 · Layer 3)
  │     - LangGraph StateGraph
  │     - bind_tools()
  │
  ▼
Capabilities / Adapters  (기존 · Layer 2 · +신규 primitives)
  │     - api_lookup, stats_lookup, ...
  │     - public_admin_adapter, legal_adapter
  │     - 신규: case_open, approval_route, notice_dispatch
  │
  ▼
harness.yaml  (신규 · Layer 1)
        adapters / orgs / roles / procedures
```

Procedure Runner는 **새 LangGraph 노드**로 추가되는 것이 아니라 **그래프 상위의 외부 오케스트레이터**다. 즉 한 Step 안에서는 LangGraph ReAct 루프가 평소처럼 돈다. Step 경계에서만 Procedure Runner가 개입해 상태 전이·결재 라우팅·감사 기록을 수행한다.

---

## 5. harness.yaml — 통합 설정 스키마

`config/adapters.yaml`을 제거하고 `config/harness.yaml`로 통합한다. 기존 adapter 로더는 harness.yaml의 `adapters` 섹션만 읽도록 리팩터링된다(MVP 동작 회귀 없음).

### 5.1 최상위 스키마

```yaml
version: 1

# --- 기존 adapter 정의 (MVP에서 이전) ---
adapters:
  public_admin: { ... }  # adapters.yaml 내용 그대로 이전
  legal:        { ... }

# --- 신규: 조직 ---
orgs:
  dong_a_city:
    display_name: 동아시
    departments:
      urban_planning:
        display_name: 도시계획과
        roles: [civil_servant, team_lead, section_chief]
      construction_permit:
        display_name: 건축허가과
        roles: [civil_servant, team_lead, section_chief]

# --- 신규: 역할 ---
roles:
  civil_servant:
    display_name: 담당자
    can: [open_case, draft_response, call_tool]
  team_lead:
    display_name: 팀장
    can: [approve_draft]
  section_chief:
    display_name: 과장
    can: [approve_final, assign_case]

# --- 신규: procedure index ---
procedures:
  - id: construction_permit_basic
    path: procedures/public_admin/construction_permit_basic.yaml
  - id: road_damage_complaint
    path: procedures/public_admin/road_damage_complaint.yaml
```

### 5.2 procedure 파일 스키마

```yaml
# procedures/public_admin/construction_permit_basic.yaml
procedure_id: construction_permit_basic
display_name: 일반 건축허가
domain: public_admin
owner_org: dong_a_city.construction_permit

legal_basis:
  - 건축법 제11조
  - 건축법 시행령 제9조
  - 민원처리법 제17조

sla_hours: 240        # 법정 처리기한 (advisory)

inputs:
  - name: applicant_id
    type: string
    required: true
  - name: site_address
    type: string
    required: true
  - name: building_type
    type: enum
    values: [residential, commercial, mixed]

steps:
  - id: intake
    type: capability
    uses: case_open        # 신규 primitive
    outputs: [case_id]

  - id: document_check
    type: capability
    uses: api_lookup
    goal: "신청서류 완비 여부 및 국토교통부 건축인허가 API 조회"
    postcondition:
      evidence_min: 1

  - id: legal_check
    type: adapter
    uses: legal_adapter
    goal: "건축법 제11조 및 관련 조례 확인"
    postcondition:
      evidence_min: 1

  - id: cross_dept_consult
    type: handoff
    to: dong_a_city.urban_planning
    condition: "building_type in [commercial, mixed]"
    timeout_hours: 72

  - id: draft_permit
    type: adapter
    uses: public_admin_adapter
    goal: "허가증 초안 생성"
    requires_role: civil_servant

  - id: supervisor_approval
    type: approval
    requires_role: team_lead
    on_reject:
      goto: draft_permit

  - id: final_approval
    type: approval
    requires_role: section_chief
    on_reject:
      goto: draft_permit

  - id: notify
    type: notice
    channel: govon_outbox   # 실제 전송은 기존 정부 시스템에 위임
    template: construction_permit_issued
```

주의: `goal`은 LLM에게 전달되는 자연어 instruction이다. `uses`는 **strong hint**이지 유일 강제가 아니다. LLM은 해당 step 안에서 필요하다면 다른 tool도 호출할 수 있으며, postcondition을 만족하면 step이 종료된다.

---

## 6. Procedure Runner 상태머신

### 6.1 상태

```
INTAKE → IN_PROGRESS → AWAITING_APPROVAL → (IN_PROGRESS | NOTIFIED) → CLOSED
                                          └── REJECTED ──────────────┘
```

- `INTAKE`: Case가 열렸고 첫 Step 실행 전.
- `IN_PROGRESS`: Step을 실행 중(내부에서 ReAct 루프 작동).
- `AWAITING_APPROVAL`: `approval` 타입 Step에서 결재자 입력 대기.
- `NOTIFIED`: `notice` Step이 성공적으로 outbox에 기록됨.
- `CLOSED`: 전체 procedure 성공 종료.
- `REJECTED`: 결재 거부 또는 postcondition 반복 실패.

### 6.2 전이 이벤트

- `step_complete(step_id, outputs)`
- `step_postcondition_failed(step_id, reason)` → 재시도 또는 REJECTED.
- `approval_granted(step_id, approver_role, approver_id, comment)`
- `approval_denied(step_id, approver_role, comment)` → `on_reject.goto` 또는 REJECTED.
- `sla_exceeded(case_id)` → advisory 경보, state는 유지.
- `handoff_completed(step_id, target_org)`

### 6.3 감사 로그

모든 전이 이벤트는 append-only 감사 저장소에 기록된다. 레코드 스키마:

```json
{
  "trace_id": "uuid",
  "case_id": "uuid",
  "procedure_id": "construction_permit_basic",
  "step_id": "draft_permit",
  "event": "step_complete",
  "actor": {"type": "llm" | "user", "id": "..."},
  "timestamp": "ISO8601",
  "payload_hash": "sha256(...)",
  "evidence_refs": ["..."]
}
```

MMP는 SQLite append-only 테이블로 시작하고, R2에서 외부 감사 싱크(정부 감사 시스템 또는 외부 WORM 저장소) 연동을 검토한다.

---

## 7. 신규 Capability Primitives

MVP Tier 0에 다음을 추가한다(구현은 후속 이슈).

| 이름 | 역할 | approval? |
|------|------|-----------|
| `case_open` | Case 객체 생성 + 법정기한 계산 + 감사 레코드 시작 | no |
| `case_assign` | Case를 특정 부서/담당자에게 할당 | no |
| `legal_basis_lookup` | 법령·조례·시행령 근거 조회 (기존 `legal_adapter`와 별개로, 구조화된 citation 반환) | no |
| `approval_route` | 결재선 다음 단계 결정 및 approval Step 트리거 | no |
| `notice_dispatch` | govon outbox에 통보 문서 기록(실제 전송은 외부 시스템) | yes |
| `case_handoff` | 다른 org/부서로의 이관 이벤트 기록 | yes |

### 기존 어댑터와의 관계

- `public_admin_adapter`, `legal_adapter`는 그대로 Tier 1 어댑터로 남는다.
- procedure가 `uses: public_admin_adapter`로 hint를 주면 ReAct agent는 해당 step 안에서 우선적으로 그 어댑터를 선택한다(강제는 아님, goal/postcondition이 우선).

---

## 8. 샘플 시나리오 — 인허가 프로세스 (MMP demo)

### 8.1 시나리오 개요

- **Case**: "동아시 해운대구 ○○번지, 주상복합 건축허가 신청"
- **Actor**: 건축허가과 담당자 (role=civil_servant)
- **Procedure**: `construction_permit_basic`

### 8.2 실행 흐름 (예상)

```
[govon> 주상복합 건축허가 신청 1건 접수됐어]
  │
  ▼
Procedure Runner: construction_permit_basic 기동
  → case_id=c-42, legal_due_at=+240h
  → state=INTAKE → IN_PROGRESS
  │
  ▼
Step 1 intake (case_open)
  → 감사 레코드 생성
  │
  ▼
Step 2 document_check (ReAct 루프)
  agent가 api_lookup 선택 → 국토부 인허가 API 조회
  → 서류 미비 1건 발견 → evidence 2건 수집
  postcondition 만족 (evidence_min=1)
  │
  ▼
Step 3 legal_check (ReAct 루프)
  agent가 legal_adapter 호출 (approval 필요 → 사용자 승인)
  → 건축법 제11조 citation 확보
  │
  ▼
Step 4 cross_dept_consult (building_type=mixed → 발동)
  → 도시계획과로 handoff 이벤트 기록, 72h 대기
  → 회신 수신 후 IN_PROGRESS 복귀
  │
  ▼
Step 5 draft_permit (public_admin_adapter)
  → 허가증 초안 생성 (approval 필요 → 승인)
  │
  ▼
Step 6 supervisor_approval
  → state=AWAITING_APPROVAL
  → 팀장 승인 입력 대기 → 승인 → IN_PROGRESS
  │
  ▼
Step 7 final_approval
  → state=AWAITING_APPROVAL
  → 과장 승인 → 통과
  │
  ▼
Step 8 notify (notice_dispatch)
  → govon outbox에 기록, state=NOTIFIED
  │
  ▼
state=CLOSED
```

MMP demo에서는 외부 기관 API와 결재자는 **모두 fake/mock**으로 구성된다. 실제 기관 연동은 R2 이후 과제.

---

## 9. 마이그레이션 단계

본 문서는 계획 단계이므로 실제 PR은 이슈로 쪼개서 진행한다. 각 단계는 **기존 MVP 동작을 깨지 않는다**는 것을 invariant로 둔다.

| 단계 | 산출물 | 기존 MVP 영향 |
|------|--------|---------------|
| **M0** (본 문서) | ADR-005, 본 설계 문서, 추적 이슈 | 없음 |
| **M1** | `harness.yaml` 스키마 정의 + loader, `adapters.yaml` 흡수 (BC 유지: 기존 loader는 harness.yaml의 adapters 섹션 읽음) | loader 리팩터링만, 런타임 동작 동일 |
| **M2** | Case 도메인 모델 + SQLite append-only 감사 테이블 | 없음 |
| **M3** | Procedure YAML 스키마 + 파서 + validator | 없음 |
| **M4** | Procedure Runner 상태머신 (ReAct 루프 외부) | 없음 |
| **M5** | 신규 primitives (`case_open`, `approval_route`, `notice_dispatch`, ...) | Tier 0 추가 |
| **M6** | 인허가 샘플 procedure + fake fixtures + e2e 테스트 | 없음 |
| **M7** | CLI 통합: `govon procedure run <id>` / 자연어 → procedure 매칭 | CLI UX 확장 |

각 단계는 독립 PR로 나누고, M6까지는 "MVP 회귀 없음"을 CI에서 증명한다.

---

## 10. 위험과 완화

| 위험 | 완화 |
|------|------|
| Procedure Runner와 ReAct 루프의 상태 이원화 | Case ↔ SessionContext는 `case_id` 단방향 참조만 허용. SessionContext는 Case를 모른다. |
| harness.yaml 비대화 | procedure는 `procedures/*.yaml`로 분리, harness.yaml은 index만. |
| 법령 해석 오류로 인한 책임 문제 | MMP 전 단계에서 **모든 법적 효력은 기존 시스템에 위임**. Procedure Runner는 advisory. UI/로그에 "본 결과는 행정 보조 참고용입니다" 고지. |
| LoRA 어댑터 간 지식 경계 오염 | 기존 adapter_registry 경계 유지. procedure는 adapter를 hint로만 주고 강제하지 않음. |
| 기존 사용자의 MVP flow 회귀 | M1~M6는 기존 entry point(`govon`, `govon "..."`)의 동작을 바꾸지 않음. procedure는 신규 명령(`govon procedure`)로만 노출. |
| 감사 저장소가 SQLite라 변조 가능 | append-only + row hash chain 도입, R2에서 외부 WORM 저장소 연계 검토. |

---

## 11. 열린 질문

- 결재자 신원 확인은 어떻게 할 것인가? MMP는 로컬 daemon이므로 단일 사용자 가정. 다중 사용자는 R2 이후(SSO, 기관 인증).
- 법정기한 초과 시 advisory 경보 외에 추가 조치가 필요한가?
- procedure간 상속/composition (공통 전처리 step 등)을 지원할 것인가? MMP는 단순 flat procedure만.
- cross-case correlation (동일 신청인의 여러 Case를 묶어보기)은 MMP 범위인가? → 아니오, R2로.

---

## 12. 참고

- [`docs/adr/ADR-005-harness-migration.md`](../adr/ADR-005-harness-migration.md)
- [`docs/prd.md`](../prd.md)
- [`config/adapters.yaml`](../../config/adapters.yaml) — 흡수 대상
- [`src/inference/graph/capabilities/registry.py`](../../src/inference/graph/capabilities/registry.py) — Tier 0 registry
- [`src/inference/graph/builder.py`](../../src/inference/graph/builder.py) — LangGraph StateGraph
