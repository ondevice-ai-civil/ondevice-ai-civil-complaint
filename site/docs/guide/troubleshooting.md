# 트러블슈팅

## `mkdocs build --strict`가 실패할 때

- nav에 없는 페이지가 있는지 확인한다.
- 문서 내 상대 링크가 실제 파일 경로와 일치하는지 확인한다.
- 새 페이지를 추가했다면 `site/mkdocs.yml`에도 반영한다.

## E2E가 `/v1/classify` 관련 오류로 실패할 때

- 현재 MVP 계약에는 `/v1/classify`가 포함되지 않는다.
- `e2e/api-health.spec.ts`가 shell-first runtime endpoint를 기준으로 검증하는지 확인한다.

## `/v1/search`가 503을 반환할 때

- `SKIP_MODEL_LOAD=true` 또는 인덱스 미초기화 상태일 수 있다.
- 이 경우 contract test에서는 graceful degradation 메시지를 기대하는 것이 맞다.

## Cloud Run 데모 배포가 실패할 때

- `GCP_SA_KEY`, `GCP_PROJECT_ID` secret이 설정되어 있는지 확인한다.
- 수동 배포 레인은 demo/runtime 검증용이므로, 먼저 GHCR 이미지가 정상 발행되었는지 확인한다.
