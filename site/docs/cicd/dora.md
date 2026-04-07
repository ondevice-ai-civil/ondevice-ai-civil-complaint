# DORA 메트릭

GovOn은 DORA(DevOps Research and Assessment) 4대 지표를 자동 수집하여
개발 프로세스 품질을 정량적으로 측정한다.

## 구현 현황

**상태: 구현 완료**

- GitHub Actions 워크플로우로 자동 수집 (`dora-metrics.yml`)
- Grafana Cloud Prometheus에 메트릭 전송 및 실시간 대시보드
- 주간 보고서 자동 생성 (마크다운 + Chart.js HTML + PNG)

## 아키텍처

```
GitHub Actions (매주 월요일 09:00 KST + main push 시)
    │
    ├─ gh CLI로 DORA 4대 지표 수집
    ├─ metrics/dora/dora-YYYYMMDD.json 저장
    ├─ generate_report.py 실행
    │   ├─ metrics/reports/weekly-YYYYMMDD.md
    │   ├─ metrics/reports/latest-dora.html
    │   └─ metrics/reports/latest-dora.png
    └─ Grafana Cloud Prometheus로 메트릭 push
```

## 4대 지표

| 지표 | 측정 방법 | 수집 소스 | Elite 기준 |
|------|----------|----------|-----------|
| **Deployment Frequency** | main 머지 PR 수 / 주 | GitHub API (`gh pr list`) | 일 1회+ |
| **Lead Time for Changes** | PR 첫 커밋 → 머지 평균 시간 | GitHub API (`gh api pulls/commits`) | < 24h |
| **Change Failure Rate** | hotfix/revert 커밋 비율 | `git log --grep` | < 15% |
| **MTTR** | bug 이슈 open → close 평균 시간 | GitHub API (`gh issue list`) | < 1h |

## 수집 주기

- **정기 수집**: 매주 월요일 09:00 KST (cron: `0 0 * * 1`)
- **이벤트 수집**: main 브랜치 push 시
- **수동 실행**: GitHub Actions workflow_dispatch

## 대시보드

- **Grafana Cloud 실시간**: [GovOn DORA Dashboard](https://umyunsang.grafana.net/d/govon-dora/govon-dora-metrics-dashboard?orgId=1&from=now-7d&to=now&timezone=Asia%2FSeoul)
- **Chart.js 보고서**: `metrics/reports/latest-dora.html`
- **대시보드 JSON**: `metrics/grafana-cloud/dora-dashboard.json`

## 보고서 생성

보고서는 GitHub Actions에서 자동 생성되지만, 수동으로도 실행할 수 있다:

```bash
pip install matplotlib
python metrics/scripts/generate_report.py
```

산출물:
- `metrics/reports/weekly-YYYYMMDD.md` — 전주 대비 비교 포함 마크다운 보고서
- `metrics/reports/latest-dora.html` — Chart.js 기반 인터랙티브 대시보드
- `metrics/reports/latest-dora.png` — matplotlib 기반 대시보드 이미지

## 데이터 스키마

`metrics/dora/dora-YYYYMMDD.json`:

```json
{
  "timestamp": "2026-04-06T00:00:00Z",
  "period": "last_30_days",
  "grade": "High",
  "primary_branch": "main",
  "main": {
    "deployment_frequency": {
      "weekly": 5,
      "monthly": 20,
      "avg_per_week": 5.0
    },
    "lead_time_for_changes": {
      "method": "first_commit_to_merge",
      "avg_seconds": 36000,
      "avg_hours": 10.0,
      "sample_count": 20
    },
    "change_failure_rate": {
      "rate_percent": 12.5,
      "failure_commits": 5,
      "total_commits": 40
    },
    "mean_time_to_recovery": {
      "avg_seconds": 3600,
      "avg_hours": 1.0,
      "sample_count": 3
    }
  }
}
```

## 등급 기준

| 등급 | 배포 빈도 | 리드 타임 | 변경 실패율 | MTTR |
|------|----------|----------|-----------|------|
| Elite | 일 1회+ | < 1일 | < 15% | < 1시간 |
| High | 주 1회+ | < 1주 | 15~30% | < 24시간 |
| Medium | 월 1회+ | < 1개월 | 30~45% | < 1주 |
| Low | 월 1회 미만 | > 1개월 | > 45% | > 1주 |

## 관련 파일

- 워크플로우: `.github/workflows/dora-metrics.yml`
- 보고서 생성: `metrics/scripts/generate_report.py`
- 대시보드 JSON: `metrics/grafana-cloud/dora-dashboard.json`
- 수집 데이터: `metrics/dora/`
- 보고서 출력: `metrics/reports/`
