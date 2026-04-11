# ADR-005: 개발자형 하네스 → 공공업무형 하네스 마이그레이션

## Status

Proposed (MMP 단계 준비)

## Date

2026-04-11

## Context

GovOn은 MVP 단계에서 **LangGraph ReAct 루프 + Tier 0 Capability + Tier 1 도메인 어댑터** 구조로 동작하고 있다. 이 구조는 Claude Code, Cursor, Devin 류의 **개발자형 에이전틱 하네스**를 그대로 차용한 것이다.

- 하네스의 실행 단위는 **단일 tool call**이다.
- 사용자(개발자/기획자)는 자연어 요청을 주고, 에이전트가 tool을 자율 선택·승인받아 1회성 결과를 돌려준다.
- 상태는 `SessionContext`와 LangGraph `checkpointer`에 메시지 단위로 저장된다.
- 권한/결재/법정기한 같은 공공행정 고유 제약은 모델링되지 않는다.

그러나 GovOn의 MMP 타깃 사용자는 **중앙부처·지자체·공공기관의 실무자**이며, 이들의 실제 업무는 다음 성질을 갖는다.

1. **다단계 절차**: 민원 접수 → 담당 배정 → 법령 조회 → 유사 사례 검색 → 초안 작성 → 결재 → 통보 → 이의처리.
2. **법정 기한**: 민원처리법, 행정절차법, 인허가의제법 등으로 처리기한이 법적으로 고정돼 있다.
3. **결재선과 권한 분리**: 담당 → 팀장 → 과장 → (필요 시) 국장. 각 단계의 승인자 역할과 법적 권한이 다르다.
4. **기관·부서 간 협의**: 한 건의 인허가에도 여러 부처/부서/외부 기관 API가 얽힌다.
5. **감사 추적**: 모든 결정은 근거와 결재 이력이 영속적으로 남아야 한다.

즉, MVP의 "1 요청 = 1 tool chain" 가정은 공공업무 flow를 표현하기에 부족하다. MMP는 이 하네스 계층을 **공공업무형 하네스(Public Workflow Harness)**로 재정의하는 단계가 되어야 한다.

## 검토 후보

| 후보 | 장점 | 단점 |
|------|------|------|
| **(선택) Procedure Layer 도입 — 기존 ReAct 유지 위에 Case/Procedure/Role 계층을 새로 얹는다** | MVP 회귀 위험 최소, Tier 0/1 자산 재사용, YAML 기반 확장 경로 유지, 감사/결재/기한 모델링을 선언적으로 표현 | ReAct + Procedure 이중 런타임 복잡도, 두 실행 경로의 상태 일관성을 꾸준히 관리해야 함 |
| 단일 파일형 adapters.yaml만 확장 | 변경 범위가 작고 리스크 낮음 | 결재선·기한·역할·감사 같은 공공 도메인 1급 개념을 표현할 수 없음. MMP 요구사항을 만족하지 못함 |
| ReAct 루프 폐기 후 정적 Workflow DAG로 전환 | 공공업무 flow를 자연스럽게 표현, 예측 가능 | PRD Non-Goals에 명시된 "정적 planner/executor 구조"로의 퇴행. MVP 철학(모델 주도 tool 선택)과 충돌 |
| BPMN 엔진(Camunda 등) 외부 의존 | 성숙한 워크플로 엔진, 결재/기한/감사 기본 지원 | 파이썬 런타임·HF Space·승인 UI와 통합 비용이 크고, 로컬 daemon 경량성 훼손 |

## Decision

GovOn MMP 단계에서 **하네스 계층을 다음 4계층으로 재구성**한다. 기존 MVP ReAct 루프는 그대로 유지하고, 그 위에 **공공업무형 하네스 레이어**를 얹는다.

```
┌──────────────────────────────────────────────────────────┐
│  Layer 4. Procedure Runner  (신규)                        │
│    - Case/Procedure 단위 상태머신                          │
│    - 결재선 라우팅, 법정기한 카운터, 감사 로그             │
└──────────────────────────────────────────────────────────┘
                          │
┌──────────────────────────────────────────────────────────┐
│  Layer 3. ReAct Agent Loop  (MVP 유지)                    │
│    - LangGraph StateGraph: agent → approval_wait → Tool  │
│    - bind_tools() 기반 모델 주도 선택                      │
└──────────────────────────────────────────────────────────┘
                          │
┌──────────────────────────────────────────────────────────┐
│  Layer 2. Capability / Adapter  (MVP 유지 + 확장)         │
│    - Tier 0: api_lookup / stats / keyword / ...          │
│    - Tier 1: public_admin / legal LoRA adapters          │
│    - 신규 primitives: case_open, approval_route, ...     │
└──────────────────────────────────────────────────────────┘
                          │
┌──────────────────────────────────────────────────────────┐
│  Layer 1. harness.yaml  (신규 — adapters.yaml 통합 대상)  │
│    - adapters / orgs / roles / procedures 단일 소스       │
└──────────────────────────────────────────────────────────┘
```

### 결정 근거

1. **PRD 철학 유지**: "모델 주도 tool 선택"과 "정적 planner 금지"를 위반하지 않는다. Procedure는 **business constraint의 선언적 표현**일 뿐, tool 선택 자체는 여전히 LLM이 ReAct 루프에서 수행한다.
2. **자산 재사용**: 기존 `capabilities/`, `adapter_registry`, LangGraph builder, `SessionContext` 모두 그대로 사용된다. 상위에 Case·Procedure·Role 계층만 추가한다.
3. **감사·결재·기한**은 MVP 수준의 `SessionContext`로는 표현할 수 없으므로 **신규 1급 개념**으로 도입한다.
4. **설정 단일화**: `adapters.yaml`은 `harness.yaml`로 흡수된다. 하나의 YAML에서 adapter / org / role / procedure가 일관된 이름공간을 공유해야 감사와 결재선 해석이 모호하지 않다.
5. **확장성**: 16개 국가 도메인 로드맵에서 domain마다 procedure가 추가될 때, 파이썬 코드가 아니라 YAML 추가만으로 확장 가능하다.

## Consequences

### 긍정적 영향

- 공공업무 flow를 1급 엔터티로 표현 → 결재선/기한/감사가 테스트 가능한 대상이 된다.
- PRD Non-Goals를 훼손하지 않으면서 MVP → MMP 전환이 가능하다.
- 신규 도메인 어댑터 추가 시 procedure YAML만 작성하면 돼 기여 장벽이 낮다.
- 기존 Tier 0/1 capability 자산은 100% 보존된다.

### 부정적 영향

- 두 개의 실행 경로(순수 ReAct / Procedure Runner)가 공존하여 state 모델이 이원화된다. `SessionContext` ↔ `Case`의 일관성 유지 비용이 생긴다.
- `harness.yaml`이 단일 파일로 비대해질 수 있다. procedure는 별도 `procedures/*.yaml`로 분리할 여지를 남겨야 한다.
- 결재선·법정기한은 법령 해석 이슈가 섞여 있어, 잘못 구현할 경우 법적 책임 여지가 있다. MMP 단계에서는 **advisory(참고용)** 로만 동작시키고 법적 효력은 기존 시스템에 위임한다.

### 향후 고려사항

- **R2 이후**: procedure 실행 이력을 기존 행정 시스템(새올/온나라 등)에 전달하는 outbound connector 설계.
- **감사(audit) 저장소**: 현재 SQLite `db/` 로는 부족. append-only ledger 또는 외부 감사 로그 싱크 필요.
- **권한 모델**: RBAC/ABAC 중 선택. MMP는 RBAC로 시작하되 ABAC로 확장할 여지 남김.
- **테스트 하네스**: procedure end-to-end 실행을 위한 fake capability / fake LLM harness가 필요.

## References

- PRD v6.0 (`docs/prd.md`)
- 상세 설계: [`docs/architecture/harness-migration-mmp.md`](../architecture/harness-migration-mmp.md)
- 추적 이슈: govon-org/govon MMP Harness Migration (별도 이슈로 발행)
- MVP capability registry: `src/inference/graph/capabilities/registry.py`
- MVP adapters: `config/adapters.yaml`
