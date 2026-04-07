#!/usr/bin/env python3
"""GovOn DORA Metrics 주간 보고서 생성 스크립트.

metrics/dora/dora-*.json 파일들을 읽어 다음 3가지 산출물을 생성한다:
  - metrics/reports/weekly-YYYYMMDD.md  (마크다운 주간 보고서)
  - metrics/reports/latest-dora.html    (Chart.js 대시보드 HTML)
  - metrics/reports/latest-dora.png     (matplotlib 대시보드 이미지)

외부 의존성: matplotlib (PNG 생성용, 없으면 PNG만 스킵)
"""

import glob
import html
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# matplotlib은 선택적 의존성
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

BASE_DIR = Path(__file__).resolve().parent.parent
DORA_DIR = BASE_DIR / "dora"
REPORTS_DIR = BASE_DIR / "reports"

GRAFANA_URL = (
    "https://umyunsang.grafana.net/d/govon-dora/"
    "govon-dora-metrics-dashboard?orgId=1&from=now-7d&to=now&timezone=Asia%2FSeoul"
)

# DORA Elite 기준
ELITE = {
    "deployment_frequency": 7,  # 일 1회+ = 주 7건+
    "lead_time_hours": 24,
    "change_failure_rate": 15,
    "mttr_hours": 1,
}


def extract_metrics(data: dict) -> dict:
    """JSON 데이터에서 DORA 지표를 추출. 3가지 스키마 호환."""
    # main 키 우선
    if isinstance(data.get("main"), dict):
        return data["main"]
    # primary_branch fallback
    branch = data.get("primary_branch")
    if isinstance(branch, str) and isinstance(data.get(branch), dict):
        return data[branch]
    # develop fallback
    if isinstance(data.get("develop"), dict):
        return data["develop"]
    # 플랫 구조: {"deployment_frequency": ...}
    if "deployment_frequency" in data:
        return data
    return {}


def normalize_metrics(raw: dict) -> dict:
    """다양한 중첩 구조를 평탄화하여 통일된 dict 반환."""
    df = raw.get("deployment_frequency", {})
    lt = raw.get("lead_time_for_changes", {})
    cfr = raw.get("change_failure_rate", {})
    mttr = raw.get("mean_time_to_recovery", {})

    # 중첩 구조 vs 플랫 구조 대응
    if isinstance(df, dict):
        df_weekly = df.get("weekly", df.get("avg_per_week", 0))
        df_monthly = df.get("monthly", 0)
    else:
        df_weekly = df
        df_monthly = 0

    if isinstance(lt, dict):
        lt_hours = lt.get("avg_hours", 0)
    else:
        lt_hours = lt

    if isinstance(cfr, dict):
        cfr_pct = cfr.get("rate_percent", 0)
    else:
        cfr_pct = cfr

    if isinstance(mttr, dict):
        mttr_hours = mttr.get("avg_hours", 0)
    else:
        mttr_hours = mttr

    return {
        "df_weekly": float(df_weekly),
        "df_monthly": float(df_monthly),
        "lead_time_hours": float(lt_hours),
        "cfr_percent": float(cfr_pct),
        "mttr_hours": float(mttr_hours),
    }


def compute_grade(m: dict) -> str:
    """DORA 등급 판정."""
    if (
        m["df_weekly"] >= ELITE["deployment_frequency"]
        and m["lead_time_hours"] < ELITE["lead_time_hours"]
    ):
        return "Elite"
    if m["df_weekly"] >= 1:
        return "High"
    if m["df_monthly"] > 0:
        return "Medium"
    return "Low"


def grade_color(grade: str) -> str:
    """등급별 색상 코드."""
    return {
        "Elite": "#28a745",
        "High": "#2188ff",
        "Medium": "#dbab09",
        "Low": "#cb2431",
    }.get(grade, "#6a737d")


def trend_arrow(current: float, previous: float, lower_is_better: bool = False) -> str:
    """전주 대비 변화 화살표."""
    diff = current - previous
    if abs(diff) < 0.01:
        return "-"
    if lower_is_better:
        arrow = "▼" if diff < 0 else "▲"
    else:
        arrow = "▲" if diff > 0 else "▼"
    return f"{arrow}{abs(diff):.1f}"


def load_all_data() -> list[tuple[str, dict]]:
    """dora-*.json 파일들을 날짜순 정렬하여 (날짜문자열, 정규화 메트릭) 리스트로 반환."""
    pattern = str(DORA_DIR / "dora-*.json")
    files = sorted(glob.glob(pattern))
    results = []
    for fp in files:
        fname = os.path.basename(fp)
        # dora-YYYYMMDD.json -> YYYYMMDD
        date_str = fname.replace("dora-", "").replace(".json", "")
        try:
            with open(fp) as f:
                data = json.load(f)
            raw = extract_metrics(data)
            if not raw:
                continue
            m = normalize_metrics(raw)
            m["grade"] = data.get("grade", compute_grade(m))
            m["date"] = date_str
            results.append((date_str, m))
        except (json.JSONDecodeError, KeyError, OSError) as e:
            print(f"  [WARN] {fname} 파싱 실패: {e}", file=sys.stderr)
    return results


def generate_markdown(
    current: dict,
    previous: dict | None,
    grade: str,
    today: str,
) -> str:
    """마크다운 주간 보고서 생성."""
    prev = previous or current

    df_trend = trend_arrow(current["df_weekly"], prev["df_weekly"])
    lt_trend = trend_arrow(
        current["lead_time_hours"], prev["lead_time_hours"], lower_is_better=True
    )
    cfr_trend = trend_arrow(current["cfr_percent"], prev["cfr_percent"], lower_is_better=True)
    mttr_trend = trend_arrow(current["mttr_hours"], prev["mttr_hours"], lower_is_better=True)

    return f"""# GovOn DORA Metrics Weekly Report

> 생성일: {today} | 기간: 최근 30일 | 브랜치: main

## 종합 등급: {grade}

| 지표 | 현재 | 이전 | 변화 | Elite 기준 |
|------|------|------|------|-----------|
| 배포 빈도 | 주 {current['df_weekly']:.1f}건 | 주 {prev['df_weekly']:.1f}건 | {df_trend} | 일 1회+ |
| 리드 타임 | {current['lead_time_hours']:.1f}h | {prev['lead_time_hours']:.1f}h | {lt_trend} | < 24h |
| 변경 실패율 | {current['cfr_percent']:.1f}% | {prev['cfr_percent']:.1f}% | {cfr_trend} | < 15% |
| MTTR | {current['mttr_hours']:.1f}h | {prev['mttr_hours']:.1f}h | {mttr_trend} | < 1h |

## 대시보드

- [Grafana 실시간]({GRAFANA_URL})
- [Chart.js 보고서](latest-dora.html)

## 수집 방식
- GitHub Actions: `.github/workflows/dora-metrics.yml`
- 주기: 매주 월요일 09:00 KST + main push 시
- 데이터: `metrics/dora/dora-YYYYMMDD.json`
"""


def generate_html(history: list[tuple[str, dict]], grade: str) -> str:
    """Chart.js 대시보드 HTML 생성."""
    dates = [h[0] for h in history]
    df_data = [h[1]["df_weekly"] for h in history]
    lt_data = [h[1]["lead_time_hours"] for h in history]
    cfr_data = [h[1]["cfr_percent"] for h in history]
    mttr_data = [h[1]["mttr_hours"] for h in history]

    current = history[-1][1] if history else {}
    prev = history[-2][1] if len(history) >= 2 else current

    grade_escaped = html.escape(grade)
    gc = grade_color(grade)

    # 레이더 차트 정규화 (0~100 스케일)
    def norm_df(v: float) -> float:
        return min(v / 7 * 100, 100)

    def norm_lt(v: float) -> float:
        return max(0, min(100, (1 - v / 168) * 100))

    def norm_cfr(v: float) -> float:
        return max(0, min(100, (1 - v / 45) * 100))

    def norm_mttr(v: float) -> float:
        return max(0, min(100, (1 - v / 24) * 100))

    radar_data = [
        norm_df(current.get("df_weekly", 0)),
        norm_lt(current.get("lead_time_hours", 0)),
        norm_cfr(current.get("cfr_percent", 0)),
        norm_mttr(current.get("mttr_hours", 0)),
    ]

    def trend_card(label: str, val: str, cur: float, prv: float, lower: bool) -> str:
        diff = cur - prv
        if abs(diff) < 0.01:
            arrow_html = '<span style="color:#6a737d">-</span>'
        elif (lower and diff < 0) or (not lower and diff > 0):
            arrow_html = (
                f'<span style="color:#28a745">{"▼" if diff < 0 else "▲"}{abs(diff):.1f}</span>'
            )
        else:
            arrow_html = (
                f'<span style="color:#cb2431">{"▲" if diff > 0 else "▼"}{abs(diff):.1f}</span>'
            )
        return f"""<div class="card">
          <div class="card-label">{html.escape(label)}</div>
          <div class="card-value">{html.escape(val)}</div>
          <div class="card-trend">{arrow_html} vs 이전</div>
        </div>"""

    cards = "".join(
        [
            trend_card(
                "배포 빈도",
                f"주 {current.get('df_weekly', 0):.1f}건",
                current.get("df_weekly", 0),
                prev.get("df_weekly", 0),
                False,
            ),
            trend_card(
                "리드 타임",
                f"{current.get('lead_time_hours', 0):.1f}h",
                current.get("lead_time_hours", 0),
                prev.get("lead_time_hours", 0),
                True,
            ),
            trend_card(
                "변경 실패율",
                f"{current.get('cfr_percent', 0):.1f}%",
                current.get("cfr_percent", 0),
                prev.get("cfr_percent", 0),
                True,
            ),
            trend_card(
                "MTTR",
                f"{current.get('mttr_hours', 0):.1f}h",
                current.get("mttr_hours", 0),
                prev.get("mttr_hours", 0),
                True,
            ),
        ]
    )

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GovOn DORA Metrics Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root {{
    --bg: #0d1117;
    --surface: #161b22;
    --border: #30363d;
    --text: #c9d1d9;
    --text-muted: #8b949e;
  }}
  @media (prefers-color-scheme: light) {{
    :root {{
      --bg: #ffffff;
      --surface: #f6f8fa;
      --border: #d0d7de;
      --text: #24292f;
      --text-muted: #57606a;
    }}
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 2rem;
    max-width: 1200px;
    margin: 0 auto;
  }}
  h1 {{ text-align: center; margin-bottom: 0.5rem; }}
  .badge {{
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 2rem;
    font-weight: 700;
    font-size: 1.1rem;
    color: #fff;
    background: {gc};
    margin: 0.5rem auto 1.5rem;
  }}
  .badge-wrap {{ text-align: center; }}
  .cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
  }}
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 0.75rem;
    padding: 1.2rem;
    text-align: center;
  }}
  .card-label {{ font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.3rem; }}
  .card-value {{ font-size: 1.8rem; font-weight: 700; }}
  .card-trend {{ font-size: 0.9rem; margin-top: 0.3rem; }}
  .chart-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
    gap: 1.5rem;
    margin-bottom: 2rem;
  }}
  .chart-box {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 0.75rem;
    padding: 1rem;
  }}
  .chart-box h3 {{ font-size: 0.95rem; margin-bottom: 0.5rem; color: var(--text-muted); }}
  canvas {{ max-height: 300px; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 0.75rem;
    overflow: hidden;
  }}
  th, td {{
    padding: 0.6rem 1rem;
    text-align: center;
    border-bottom: 1px solid var(--border);
  }}
  th {{ background: var(--border); font-size: 0.85rem; }}
  td {{ font-size: 0.9rem; }}
</style>
</head>
<body>
<h1>GovOn DORA Metrics</h1>
<div class="badge-wrap"><span class="badge">{grade_escaped}</span></div>

<div class="cards">
{cards}
</div>

<div class="chart-row">
  <div class="chart-box">
    <h3>4 Metrics Trend</h3>
    <canvas id="trendChart"></canvas>
  </div>
  <div class="chart-box">
    <h3>Deployment Frequency</h3>
    <canvas id="dfChart"></canvas>
  </div>
</div>
<div class="chart-row">
  <div class="chart-box">
    <h3>Grade Radar</h3>
    <canvas id="radarChart"></canvas>
  </div>
</div>

<h3 style="margin:1rem 0 0.5rem;color:var(--text-muted)">등급 기준</h3>
<table>
  <tr><th>등급</th><th>배포 빈도</th><th>리드 타임</th><th>변경 실패율</th><th>MTTR</th></tr>
  <tr><td>Elite</td><td>일 1회+</td><td>&lt; 1일</td><td>&lt; 15%</td><td>&lt; 1시간</td></tr>
  <tr><td>High</td><td>주 1회+</td><td>&lt; 1주</td><td>15~30%</td><td>&lt; 24시간</td></tr>
  <tr><td>Medium</td><td>월 1회+</td><td>&lt; 1개월</td><td>30~45%</td><td>&lt; 1주</td></tr>
  <tr><td>Low</td><td>월 1회 미만</td><td>&gt; 1개월</td><td>&gt; 45%</td><td>&gt; 1주</td></tr>
</table>

<script>
const labels = {json.dumps(dates)};
const dfData = {json.dumps(df_data)};
const ltData = {json.dumps(lt_data)};
const cfrData = {json.dumps(cfr_data)};
const mttrData = {json.dumps(mttr_data)};
const radarData = {json.dumps(radar_data)};

const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
const gridColor = isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)';
const textColor = isDark ? '#c9d1d9' : '#24292f';
Chart.defaults.color = textColor;

new Chart(document.getElementById('trendChart'), {{
  type: 'line',
  data: {{
    labels,
    datasets: [
      {{ label: 'DF (weekly)', data: dfData, borderColor: '#58a6ff', tension: 0.3 }},
      {{ label: 'Lead Time (h)', data: ltData, borderColor: '#d29922', tension: 0.3 }},
      {{ label: 'CFR (%)', data: cfrData, borderColor: '#f85149', tension: 0.3 }},
      {{ label: 'MTTR (h)', data: mttrData, borderColor: '#bc8cff', tension: 0.3 }},
    ]
  }},
  options: {{ responsive: true, scales: {{ y: {{ grid: {{ color: gridColor }} }}, x: {{ grid: {{ color: gridColor }} }} }} }}
}});

new Chart(document.getElementById('dfChart'), {{
  type: 'bar',
  data: {{
    labels,
    datasets: [{{ label: 'Deployments / week', data: dfData, backgroundColor: '#58a6ff' }}]
  }},
  options: {{ responsive: true, scales: {{ y: {{ grid: {{ color: gridColor }} }}, x: {{ grid: {{ color: gridColor }} }} }} }}
}});

new Chart(document.getElementById('radarChart'), {{
  type: 'radar',
  data: {{
    labels: ['배포 빈도', '리드 타임', '변경 실패율', 'MTTR'],
    datasets: [{{
      label: 'Current',
      data: radarData,
      backgroundColor: 'rgba(88,166,255,0.2)',
      borderColor: '#58a6ff',
      pointBackgroundColor: '#58a6ff'
    }}]
  }},
  options: {{
    responsive: true,
    scales: {{
      r: {{
        min: 0, max: 100,
        grid: {{ color: gridColor }},
        angleLines: {{ color: gridColor }},
        pointLabels: {{ color: textColor }}
      }}
    }}
  }}
}});
</script>
</body>
</html>"""


def generate_png(history: list[tuple[str, dict]], grade: str) -> None:
    """matplotlib 대시보드 이미지 생성."""
    if not HAS_MATPLOTLIB:
        print("  [INFO] matplotlib 미설치 — PNG 생성 스킵")
        return

    dates = [h[0] for h in history]
    df_data = [h[1]["df_weekly"] for h in history]
    lt_data = [h[1]["lead_time_hours"] for h in history]
    cfr_data = [h[1]["cfr_percent"] for h in history]
    mttr_data = [h[1]["mttr_hours"] for h in history]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"GovOn DORA Metrics  —  Grade: {grade}",
        fontsize=16,
        fontweight="bold",
        color=grade_color(grade),
    )

    # 배포 빈도 bar
    ax = axes[0, 0]
    ax.bar(dates, df_data, color="#58a6ff", alpha=0.85)
    ax.set_title("Deployment Frequency (weekly)")
    ax.set_ylabel("PRs / week")
    ax.tick_params(axis="x", rotation=45, labelsize=7)

    # 리드 타임 line
    ax = axes[0, 1]
    ax.plot(dates, lt_data, marker="o", color="#d29922", linewidth=2)
    ax.axhline(y=24, color="#28a745", linestyle="--", alpha=0.5, label="Elite (<24h)")
    ax.set_title("Lead Time for Changes")
    ax.set_ylabel("Hours")
    ax.legend(fontsize=7)
    ax.tick_params(axis="x", rotation=45, labelsize=7)

    # 변경 실패율 line
    ax = axes[1, 0]
    ax.plot(dates, cfr_data, marker="o", color="#f85149", linewidth=2)
    ax.axhline(y=15, color="#28a745", linestyle="--", alpha=0.5, label="Elite (<15%)")
    ax.set_title("Change Failure Rate")
    ax.set_ylabel("Percent (%)")
    ax.legend(fontsize=7)
    ax.tick_params(axis="x", rotation=45, labelsize=7)

    # MTTR line
    ax = axes[1, 1]
    ax.plot(dates, mttr_data, marker="o", color="#bc8cff", linewidth=2)
    ax.axhline(y=1, color="#28a745", linestyle="--", alpha=0.5, label="Elite (<1h)")
    ax.set_title("Mean Time to Recovery")
    ax.set_ylabel("Hours")
    ax.legend(fontsize=7)
    ax.tick_params(axis="x", rotation=45, labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = REPORTS_DIR / "latest-dora.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"  PNG: {out}")


def main() -> None:
    """메인 진입점."""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=== GovOn DORA Weekly Report Generator ===")
    history = load_all_data()
    if not history:
        print("  [ERROR] metrics/dora/dora-*.json 파일이 없습니다.")
        sys.exit(1)

    print(f"  데이터 {len(history)}건 로드 완료")

    current = history[-1][1]
    previous = history[-2][1] if len(history) >= 2 else None
    grade = current.get("grade", compute_grade(current))
    now_utc = datetime.now(timezone.utc)
    today = now_utc.strftime("%Y-%m-%d")
    today_compact = now_utc.strftime("%Y%m%d")

    # 1. 마크다운 보고서
    md = generate_markdown(current, previous, grade, today)
    md_path = REPORTS_DIR / f"weekly-{today_compact}.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"  Markdown: {md_path}")

    # 2. Chart.js HTML
    html_content = generate_html(history, grade)
    html_path = REPORTS_DIR / "latest-dora.html"
    html_path.write_text(html_content, encoding="utf-8")
    print(f"  HTML: {html_path}")

    # 3. matplotlib PNG
    generate_png(history, grade)

    print("=== 보고서 생성 완료 ===")


if __name__ == "__main__":
    main()
