# 보안 스캔

GovOn의 보안 스캔은 shell-first runtime에 직접 연결된 항목부터 막는다. 문서상 정책과 실제 워크플로가 어긋나지 않도록 PR Gate 안에 기본 보안 레인을 포함한다.

## 현재 보안 레인

| 레인 | 위치 | 목적 |
|------|------|------|
| `bandit` | `PR Gate` | Python runtime 코드의 위험 패턴 탐지 |
| `Trivy` | `Docker Publish` | 컨테이너 이미지의 고위험 취약점 스캔 |
| GitHub secret / env 관리 | repo settings | Cloud Run, GHCR, Pages 시크릿 분리 |

## 운영 기준

- `src/inference`는 사용자 입력과 tool orchestration이 만나는 지점이므로 PR 단계에서 반드시 스캔한다.
- 컨테이너 스캔은 artifact 발행 시점에 수행한다.
- demo Cloud Run 배포는 manual lane으로 제한해, 실패한 변경이 자동으로 외부 surface에 노출되지 않게 한다.

## 남은 작업

- dependency audit를 일정 기반 레인으로 분리할지 결정해야 한다.
- CLI approval UI가 들어오면 prompt injection / tool approval bypass 관련 규칙을 별도 추가해야 한다.
