# ADR-003: vLLM 기반 로컬 추론 런타임 유지

## Status

Accepted

## Context

GovOn R1/MVP의 제품 형태는 웹 UI가 아니라 `govon` 대화형 CLI 셸이다. 사용자는 셸을 실행하면 즉시 대화에 들어가고, 내부에서는 로컬 FastAPI 데몬이 자동으로 기동되거나 기존 프로세스에 재연결되어야 한다. 이 런타임은 다음 조건을 만족해야 한다.

1. **로컬 데몬 통합**: CLI가 붙는 내부 런타임으로 자연스럽게 동작해야 한다.
2. **베이스 모델 추론 품질**: 의도 파악, 작업 계획, tool 선택, 답변 합성을 한 프로세스 안에서 안정적으로 처리해야 한다.
3. **어댑터 결합 용이성**: 민원 답변 작성 단계에서 LoRA adapter를 내부적으로 붙이거나 전환할 수 있어야 한다.
4. **스트리밍 가능성**: 이후 CLI에 단계별 상태와 생성 중간 출력을 표시할 수 있어야 한다.
5. **Python 런타임 결합성**: FastAPI, SQLite 세션 저장, RAG 검색기, tool registry와 같은 Python 계층과 직접 연결돼야 한다.

## 검토 후보

| 후보 | 장점 | 단점 |
|------|------|------|
| **vLLM** | Python 프로세스 안에서 직접 제어 가능, AWQ 모델 서빙 적합, Async 엔진 제공 | CUDA 의존성 큼, EXAONE 계열 호환성 패치 관리 필요 |
| Ollama | 사용이 단순함, 독립 서버로 다루기 쉬움 | 별도 서버 의존, Python 내부 제어 폭이 좁고 adapter/tool 결합이 제한적 |
| TGI | 고성능 서빙 엔진, 스트리밍 지원 | 별도 서비스 운영 필요, 로컬 daemon 내부 구성과 어긋남 |
| 단순 Transformers 추론 | 의존성 단순, 직접 제어 가능 | 배치/메모리/스트리밍 측면에서 비효율적, 운영 성능 열세 |

## Decision

GovOn의 내부 로컬 런타임은 **FastAPI + vLLM** 조합을 유지한다.

`vLLM`은 외부 공개 API 서버의 의미보다, `govon` CLI가 붙는 **로컬 추론 데몬의 엔진 계층**으로 사용한다. CLI는 사용자 진입점이고, FastAPI는 내부 runtime adapter 역할을 맡는다.

### 런타임 역할 분리

- `govon` CLI: 사용자 입력, 승인 UI, 세션 resume UX
- `FastAPI daemon`: 세션 제어, task loop 실행, tool orchestration, 상태 브로커
- `vLLM`: base model 추론 및 민원답변 adapter 사용 시 생성 실행

### 유지 이유

- EXAONE 계열 모델과 AWQ 경로를 유지하기 쉽다.
- FastAPI와 같은 Python 프로세스 안에서 tool 실행, RAG 검색, SQLite 세션 저장을 자연스럽게 연결할 수 있다.
- 향후 CLI 스트리밍 출력, 승인 대기, task trace 전송을 같은 런타임에서 처리할 수 있다.
- 민원답변 LoRA adapter를 내부 capability처럼 붙이는 구조를 설계하기 쉽다.

## Consequences

### 긍정적 영향

- `govon` 셸과 내부 runtime 사이의 경계가 명확해진다.
- base model 추론, tool 호출, 세션 저장, RAG 검색을 한 서비스 경계에서 통제할 수 있다.
- 이후 로컬 daemon 재사용으로 CLI 재실행 속도를 높이기 쉽다.
- 별도 모델 게이트웨이를 두지 않아 초기 MVP 구현 복잡도가 낮다.

### 부정적 영향

- vLLM과 EXAONE 계열의 버전 호환성 문제를 계속 관리해야 한다.
- GPU 없는 환경에서는 완전한 추론 검증이 어렵고, 테스트는 mock 또는 `SKIP_MODEL_LOAD` 경로에 의존하게 된다.
- FastAPI가 외부 제품 표면이 아니라 내부 runtime이라는 점을 문서와 코드에서 일관되게 유지해야 한다.

### 향후 고려사항

- adapter attach/swap이 vLLM 경계 안에서 어느 수준까지 가능한지 실제 모델 운영 방식으로 구체화해야 한다.
- CLI와 daemon 간 통신은 MVP에서 localhost HTTP/SSE를 기본으로 두되, 이후 Unix socket 전환 가능성을 열어둔다.
- 모델 안정성이 확인되면 daemon warm pool 또는 preload 전략을 도입할 수 있다.
