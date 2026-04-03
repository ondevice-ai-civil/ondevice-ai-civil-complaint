# GovOn Repository Instructions

이 저장소에서 작업할 때는 아래 규칙을 기본값으로 따른다.

## 우선 문서

- Git/PR 규칙의 기준 문서는 [site/docs/guide/development.md](site/docs/guide/development.md)와 [CONTRIBUTING.md](CONTRIBUTING.md)다.
- [docs/wiki/Development-Guide.md](docs/wiki/Development-Guide.md)와 충돌하면 위 두 문서를 우선한다. 위키 문서는 구버전 참고 자료로 본다.

## 브랜치 전략

- GitHub Flow를 사용한다.
- 기본 대상 브랜치는 항상 `main`이다.
- `main`에 직접 push하지 않는다.
- 작업 브랜치는 `<type>/<issue-number>-<kebab-summary>` 형식을 사용한다.
- 허용 prefix는 `feat`, `fix`, `docs`, `chore`, `refactor`, `test`다.
- git diff의 주된 성격이 문서/로드맵/정책 정리라면 `docs/` 브랜치를 사용한다.

예시:

- `feat/129-session-store`
- `docs/402-agentic-shell-roadmap`
- `refactor/410-tool-routing-guardrails`

## 커밋과 PR

- 커밋 메시지는 한글 Conventional Commits 형식을 사용한다.
- PR 제목도 같은 형식을 사용한다.
- PR 본문은 반드시 [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md)를 따른다.
- 관련 이슈는 `## 관련 이슈` 섹션에 `Closes #...` 또는 `Refs #...`로 연결한다.
- 내부 팀원/조직원이 `main` 대상으로 PR 생성을 요청한 경우, 사용자가 명시적으로 초안을 원하지 않는 한 `ready for review` PR을 기본값으로 사용한다.
- 테스트 미실행, 후속 작업 필요 등의 사유는 PR 본문에 명시하되, 그 이유만으로 자동으로 draft PR로 올리지 않는다.

예시:

- `docs: 에이전틱 셸 로드맵 및 이슈 계층 정리`
- `feat: GovOn 셸 세션 루프 추가`

## 현재 제품 방향

- 첫 릴리즈는 `GovOn Agentic Shell` 기준으로 진행한다.
- R1 범위에는 interactive terminal shell, agent runtime, graph-based decision/orchestration, install/offline package가 포함된다.
- 웹/앱 UI 고도화는 R1 이후 단계로 본다.

## GitHub 이슈 구조

- roadmap parent는 `#402`다.
- roadmap 작업은 `#402 -> initiative -> canonical task` 구조를 유지한다.
- 가능하면 GitHub 네이티브 parent-child/sub-issue 링크를 사용한다.

## 범위

- 이 파일은 저장소 전체에 적용된다.
- 하위 디렉터리에 별도 `AGENTS.md`가 있으면 해당 디렉터리 작업에는 그 지침을 추가로 따른다.
