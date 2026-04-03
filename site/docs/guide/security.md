# 보안 정책

## 기본 원칙

- GovOn runtime은 로컬 daemon을 우선한다.
- 외부 배포 surface는 merge 게이트와 분리한다.
- 민감한 설정은 코드가 아니라 GitHub secrets와 환경 변수로 관리한다.

## 개발 중 지켜야 할 항목

- API 키, GCP 자격 증명, 모델 경로를 저장소에 커밋하지 않는다.
- 승인 기반 orchestration을 우회하는 단축 경로를 추가하지 않는다.
- tool 결과를 그대로 외부로 노출할 때는 최소한의 검증과 마스킹을 거친다.

## CI/CD 관점

- `PR Gate`에서 `bandit`을 실행한다.
- `Docker Publish`에서 `Trivy`를 실행한다.
- Cloud Run은 수동 환경으로 제한한다.
