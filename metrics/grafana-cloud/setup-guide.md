# Grafana Cloud DORA 대시보드 설정 가이드

## 1. Grafana Cloud 계정 생성 (무료)

1. https://grafana.com/auth/sign-up/create-user 접속
2. 무료 계정 생성 (신용카드 불필요)
3. 스택 생성 완료 후 Grafana 인스턴스 URL 확인
   - 예: `https://govon.grafana.net`

## 2. `GRAFANA_CLOUD_URL` & `GRAFANA_CLOUD_USER` 확인

Prometheus Remote Write Endpoint와 Instance ID는 같은 페이지에 있다:

1. **Connections** → **Data sources** → **grafanacloud-...-prom** 클릭
   - 또는 직접: `https://grafana.com/orgs/{your-org}/stacks` → **Details** → **Prometheus** 탭
2. 다음 2가지 정보를 메모:
   - `Remote Write Endpoint` → **`GRAFANA_CLOUD_URL`**
     - 예: `https://prometheus-prod-24-prod-us-central-0.grafana.net/api/prom/push`
   - `Username / Instance ID` (숫자) → **`GRAFANA_CLOUD_USER`**
     - 예: `1234567`

## 3. `GRAFANA_CLOUD_API_KEY` 생성

1. Grafana Cloud Portal: `https://grafana.com/orgs/{your-org}/api-keys`
2. **Add API Key** 클릭
3. Role: **MetricsPublisher** 선택 (`metrics:write` 권한 포함)
4. 생성된 토큰 값 → **`GRAFANA_CLOUD_API_KEY`**

> **주의**: 토큰은 생성 직후에만 표시되므로 바로 복사할 것.

## 4. GitHub Secrets 설정

레포지토리 → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

| Secret 이름 | 값 |
|-------------|-----|
| `GRAFANA_CLOUD_URL` | Remote Write Endpoint URL |
| `GRAFANA_CLOUD_USER` | Username / Instance ID |
| `GRAFANA_CLOUD_API_KEY` | 생성한 Token |

## 5. 워크플로우 실행

### 수동 실행 (최초)
1. GitHub → **Actions** → **DORA Metrics Collector**
2. **Run workflow** 클릭
3. 실행 완료 후 Step Summary에서 결과 확인

### 자동 실행
- 매주 월요일 09:00 KST 자동 실행
- main 브랜치 push 시마다 자동 실행

## 6. 대시보드 Import

1. Grafana Cloud 인스턴스 접속 (예: `https://govon.grafana.net`)
2. 좌측 메뉴 → **Dashboards** → **New** → **Import**
3. `metrics/grafana-cloud/dora-dashboard.json` 파일 업로드
4. 데이터소스에서 **grafanacloud-{스택이름}-prom** 선택
5. **Import** 클릭

## 7. 대시보드 공유

1. 대시보드 우측 상단 **Share** 버튼
2. **Public dashboard** 탭 → 활성화
3. 생성된 공개 URL을 팀원/교수님에게 공유

또는:
1. **Snapshot** 탭 → **Publish to snapshots.raintank.io**
2. 생성된 URL로 누구나 접근 가능 (Grafana 계정 불필요)

## 8. 대시보드 패널 구성

| 패널 | 타입 | PromQL |
|------|------|--------|
| DORA 종합 등급 | Stat | `dora_grade{project="govon"}` |
| 배포 빈도 | Stat + TimeSeries | `dora_deployment_frequency{project="govon"}` |
| 리드 타임 | Stat + TimeSeries | `dora_lead_time_hours{project="govon"}` |
| 변경 실패율 | Stat + TimeSeries | `dora_change_failure_rate{project="govon"}` |
| MTTR | Stat + TimeSeries | `dora_mttr_hours{project="govon"}` |
| 등급 기준표 | Text (Markdown) | - |

## 9. 등급 색상 기준

- **녹색 (Elite)**: 최고 수준, 지속적 배포 가능
- **파랑 (High)**: 우수, 주 단위 배포
- **노랑 (Medium)**: 보통, 월 단위 배포
- **빨강 (Low)**: 개선 필요

## 무료 티어 한도

| 항목 | 무료 한도 |
|------|----------|
| Prometheus 메트릭 | 10,000 시리즈 |
| 로그 | 50GB |
| 대시보드 | 무제한 |
| 사용자 | 무제한 (Viewer) |
| 보존 기간 | 13개월 |

DORA 6개 메트릭 × 1회/주 = 무료 한도의 0.06% 사용으로 충분합니다.
