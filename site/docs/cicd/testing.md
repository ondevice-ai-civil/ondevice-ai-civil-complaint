# 테스트 전략

GovOn 테스트 전략은 shell-first MVP surface를 기준으로 계층화되어 있다. 중요한 점은 "웹 UI가 아니라 로컬 daemon runtime 계약"이 우선이라는 점이다.

## 테스트 레이어

| 레이어 | 목적 | 현재 도구 |
|--------|------|-----------|
| Unit / Contract | agent loop, session store, API 응답 구조 검증 | `pytest`, `pytest-asyncio`, `TestClient` |
| Runtime contract | 실제 FastAPI 서버와 OpenAPI surface 검증 | `Playwright` request fixture |
| Docs contract | 문서 링크와 nav 무결성 보장 | `mkdocs build --strict` |
| Container smoke | 이미지가 최소 실행 가능한지 검증 | Docker smoke script |

## 커버리지 정책

- PR-safe inference suite는 `src/inference` 기준 branch coverage를 측정한다.
- 기준치는 현재 `80%`다.
- 임계치를 낮추기보다 누락된 runtime contract 테스트를 추가하는 방향을 우선한다.

## 현재 포함되는 MVP 계약

- `/health`가 SQLite session store와 runtime feature flag를 보고한다.
- `/v1/search`가 보호된 엔드포인트로 동작한다.
- `/v1/generate`와 `/v1/generate-civil-response`가 shell client 호출 surface를 유지한다.
- `/v1/agent/run`, `/v1/agent/stream`이 에이전트 runtime surface로 노출된다.
- `/v1/classify`는 더 이상 MVP contract에 포함되지 않는다.

## 남은 갭

- 실제 `govon` CLI bootstrap/install flow는 아직 별도 acceptance로 남아 있다.
- approval-gated 상호작용은 daemon API와 shell UI를 함께 검증하는 상위 레인이 추가되어야 완성된다.
