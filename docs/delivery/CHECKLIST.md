# GovOn 납품 전 최종 점검 체크리스트

> R1 릴리즈 전 하나씩 확인하는 점검 목록이다. 모든 항목이 통과해야 납품 가능하다.

---

## 점검 일자

- **점검자**: _______________
- **점검일**: 2026-04-___
- **결과**: [ ] 전체 통과 / [ ] 일부 미통과

---

## 1. 코드 품질

- [ ] `main` 브랜치에 미머지 PR 없음
- [ ] `main` 브랜치에 커밋 직접 push 없음 (모든 변경은 PR 경유)
- [ ] 모든 CI 워크플로우 green (lint, test, security)
- [ ] Black + isort + flake8 린트 통과
- [ ] pytest 통과 (Python 3.10 / 3.11 / 3.12 matrix)
- [ ] security scan 통과 (Dependabot, reusable-security-scan)
- [ ] dead code 및 v1 레거시 코드 제거 완료

## 2. 테스트

- [ ] E2E 27/27 시나리오 통과 (`gh workflow run e2e-hfspace.yml`)
  - [ ] Phase 1: Infrastructure (3/3)
  - [ ] Phase 2: v2 Pipeline (6/6)
  - [ ] Phase 3: v3 ReAct (10/10)
  - [ ] Phase 4: Cross-version (2/2)
  - [ ] Phase 5: Multi-turn (3/3)
  - [ ] Phase 6: Context Management (3/3)
- [ ] latency benchmark 실행 완료 (`gh workflow run benchmark.yml`)

## 3. 문서

- [ ] README.md가 현재 v4 아키텍처를 반영
- [ ] 사용자 가이드 (`docs/guide/user-guide.md`) 최신
- [ ] 운영 가이드 (`docs/guide/ops-guide.md`) 최신
- [ ] 데모 패키지 (`docs/demo/README.md`) 최신
- [ ] 프로젝트 회고 (`docs/retrospective.md`) 최신
- [ ] 납품 패키지 (`docs/delivery/README.md`) 작성 완료
- [ ] Docs Portal (GitHub Pages) 빌드 성공 및 접근 가능
- [ ] ADR 문서 아카이브에 stale 문서 없음

## 4. 패키지 배포

- [ ] PyPI: `pip install govon` 후 `govon --version` 정상 (v1.0.1)
- [ ] Homebrew: `brew install govon` 정상 (해당 시)
- [ ] npm: `npm install govon` 정상 (해당 시)
- [ ] Docker 이미지 빌드 성공 (`docker build .`)

## 5. 배포 (HF Space)

- [ ] HF Space 상태: RUNNING
- [ ] `/health` 엔드포인트 정상 응답
- [ ] `model_loaded: true` 확인
- [ ] `ADAPTER_PATHS`에 public_admin, legal 어댑터 정상 로드
- [ ] v3 `/v3/agent/run` 호출 정상 응답
- [ ] v4 `/v2/agent/stream` + `/v2/agent/approve` 승인 흐름 동작
- [ ] 멀티턴 대화 `session_id` 유지 확인

## 6. 보안

- [ ] Dependabot 보안 알림 0건 (또는 알려진 이슈로 문서화)
- [ ] `deploy/env/.env.example`의 `API_KEY`가 placeholder 상태 (실제 키 노출 없음)
- [ ] GitHub Secrets 모두 설정 완료 (HF_TOKEN, PYPI_TOKEN 등)
- [ ] `ALLOW_NO_AUTH=false` (프로덕션 환경)
- [ ] CORS 설정 적절함

## 7. 메트릭

- [ ] DORA 대시보드 접근 가능 ([Grafana Cloud](https://umyunsang.grafana.net/d/govon-dora/))
- [ ] DORA 주간 보고서 최신 (`metrics/reports/weekly-*.md`)
- [ ] Deployment Frequency, Lead Time, CFR, MTTR 값 정상

## 8. 산출물 완전성

- [ ] GitHub Repository 공개/접근 가능
- [ ] HF Space 런타임 접근 가능
- [ ] Civil Adapter (HF Hub) 접근 가능
- [ ] Legal Adapter (HF Hub) 접근 가능
- [ ] Civil Dataset (HF Hub) 접근 가능
- [ ] Legal Dataset (HF Hub) 접근 가능
- [ ] 공식 서식 (`docs/official/`) 포함

---

## 점검 결과 요약

| 영역 | 항목 수 | 통과 | 미통과 | 비고 |
|------|---------|------|--------|------|
| 코드 품질 | 7 | | | |
| 테스트 | 8 | | | |
| 문서 | 8 | | | |
| 패키지 배포 | 4 | | | |
| 배포 (HF Space) | 7 | | | |
| 보안 | 5 | | | |
| 메트릭 | 3 | | | |
| 산출물 완전성 | 7 | | | |
| **합계** | **49** | | | |

---

## 서명

| 역할 | 이름 | 서명 | 날짜 |
|------|------|------|------|
| 팀장 | | | |
| 팀원 | | | |
| 팀원 | | | |
| 지도교수 | | | |
