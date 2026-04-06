# DORA Metrics 수집 및 대시보드

## 개요

DORA(DevOps Research and Assessment) 4대 지표를 GitHub Actions로 자동 수집하고,
**Grafana Cloud**(무료 티어)에서 실시간 대시보드로 시각화한다.
수집 시마다 주간 보고서(마크다운 + Chart.js HTML + PNG)를 자동 생성한다.

- **Grafana 실시간 대시보드**: [GovOn DORA Dashboard](https://umyunsang.grafana.net/d/govon-dora/govon-dora-metrics-dashboard?orgId=1&from=now-7d&to=now&timezone=Asia%2FSeoul)
- **Chart.js 보고서**: [`metrics/reports/latest-dora.html`](reports/latest-dora.html)
- **주간 보고서**: [`metrics/reports/weekly-YYYYMMDD.md`](reports/)

## 아키텍처

```
GitHub Actions (매주 월요일 09:00 KST + main push)
    │
    ├── DORA 4대 지표 수집 (gh CLI)
    ├── JSON 아티팩트 저장 (metrics/dora/)
    ├── 주간 보고서 생성 (metrics/scripts/generate_report.py)
    │   ├── weekly-YYYYMMDD.md  (마크다운)
    │   ├── latest-dora.html    (Chart.js 대시보드)
    │   └── latest-dora.png     (matplotlib 이미지)
    └── Grafana Cloud Prometheus에 메트릭 전송 (InfluxDB line protocol)
            │
            └── Grafana Cloud Dashboard (공개 URL로 팀원/교수님 공유)
```

## DORA 4대 지표

| 지표 | 측정 방법 | PromQL |
|------|----------|--------|
| **배포 빈도** | main 머지 PR 수 / 주 | `dora_deployment_frequency{project="govon", branch="main"}` |
| **리드 타임** | PR 첫 커밋 → 머지 평균 시간 | `dora_lead_time_hours{project="govon", branch="main"}` |
| **변경 실패율** | hotfix/revert 커밋 비율 | `dora_change_failure_rate{project="govon", branch="main"}` |
| **복구 시간 (MTTR)** | bug 이슈 open → close 평균 시간 | `dora_mttr_hours{project="govon", branch="main"}` |

## 디렉토리 구조

```
metrics/
├── README.md
├── dora/                        # 수집된 JSON 데이터 (Actions 자동 생성)
│   └── dora-YYYYMMDD.json
├── reports/                     # 주간 보고서 (Actions 자동 생성)
│   ├── weekly-YYYYMMDD.md
│   ├── latest-dora.html
│   └── latest-dora.png
├── scripts/
│   └── generate_report.py       # 보고서 생성 스크립트
└── grafana-cloud/
    ├── setup-guide.md           # Grafana Cloud 설정 가이드
    └── dora-dashboard.json      # 대시보드 Import용 JSON
```

## 설정 방법

**상세 가이드**: [grafana-cloud/setup-guide.md](grafana-cloud/setup-guide.md)

### 빠른 시작

1. [Grafana Cloud 무료 가입](https://grafana.com/auth/sign-up/create-user)
2. Prometheus Endpoint URL, Username, Token 확인
3. GitHub Secrets 3개 등록:
   - `GRAFANA_CLOUD_URL` — Remote Write Endpoint
   - `GRAFANA_CLOUD_USER` — Instance ID
   - `GRAFANA_CLOUD_API_KEY` — Access Token
4. Actions → DORA Metrics Collector → Run workflow
5. Grafana Cloud에서 `dora-dashboard.json` Import
6. **Share → Public dashboard** 활성화하여 공유

### 수동 보고서 생성

```bash
pip install matplotlib
python metrics/scripts/generate_report.py
```

## 등급 기준

| 등급 | 배포 빈도 | 리드 타임 | 변경 실패율 | MTTR |
|------|----------|----------|-----------|------|
| Elite | 일 1회+ | < 1일 | < 15% | < 1시간 |
| High | 주 1회+ | < 1주 | 15~30% | < 24시간 |
| Medium | 월 1회+ | < 1개월 | 30~45% | < 1주 |
| Low | 월 1회 미만 | > 1개월 | > 45% | > 1주 |
