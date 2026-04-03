# 워크플로우 상세

## 핵심 워크플로우

| 워크플로우 | 트리거 | 역할 |
|------------|--------|------|
| `PR Gate` | PR, `main` push, manual | 변경 범위 감지 후 lint/test/security/runtime-contract/docs/build 실행 |
| `Docker Publish` | `PR Gate` 성공 후 `main`, release tag, manual | GHCR 이미지 발행 및 컨테이너 스모크 테스트 |
| `Deploy Demo Runtime to Cloud Run` | manual | 데모용 Cloud Run 런타임 배포 및 헬스체크 |
| `Docs Portal` | docs PR, docs push to `main`, manual | MkDocs strict build 및 GitHub Pages 배포 |
| `Publish Package` | tag, manual | Python 배포 아티팩트 생성 |
| `Offline Package Build` | release, manual | 오프라인 컨테이너 번들 생성 |

## PR Gate 내부 레인

### detect-changes

- `dorny/paths-filter`로 runtime, runtime contract, docs, packaging 변경을 분리한다.
- docs-only 변경이면 docs strict build만 돌고, 불필요한 inference 테스트를 생략한다.

### lint

- `black`, `isort`, `flake8`를 차단형으로 실행한다.
- 더 이상 `continue-on-error`로 통과시키지 않는다.

### test

- `tests/test_inference` 기준 PR-safe suite를 실행한다.
- 현재 기본 매트릭스는 `ubuntu-latest` + Python 3.10/3.11이다.

### security

- `bandit`으로 `src/inference` 정적 분석을 수행한다.
- 보안 스캔은 deploy placeholder가 아니라 PR 품질 게이트에 포함된다.

### runtime-contract

- FastAPI 런타임을 띄운 뒤 Playwright request 기반으로 shell-first API 계약을 검증한다.
- 검증 범위는 `/health`, `/v1/search`, `/v1/generate`, `/v1/generate-civil-response`, `/v1/agent/*`다.

### docs-build

- `mkdocs build --strict`로 누락 페이지, 깨진 nav, 링크 경고를 PR 단계에서 차단한다.

### build

- `uv build`로 Python 아티팩트를 생성해 후속 packaging 레인에서 재사용한다.

## 배포 레인 해석

- Docker 이미지는 merge 후 artifact 발행이다.
- Cloud Run은 자동 production 배포가 아니라 수동 검증 레인이다.
- GitHub Pages는 docs surface를 위한 별도 deploy 레인이다.
