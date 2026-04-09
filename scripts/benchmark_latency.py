#!/usr/bin/env python3
"""GovOn Latency Benchmark Script.

HF Space에 배포된 govon-runtime 서버 대상으로 주요 엔드포인트의
latency를 측정하고 min/avg/max/p95 통계를 리포트한다.

사용법:
    GOVON_RUNTIME_URL=https://<space>.hf.space python3 scripts/benchmark_latency.py
    GOVON_RUNTIME_URL=https://<space>.hf.space python3 scripts/benchmark_latency.py --runs 5
    GOVON_RUNTIME_URL=https://<space>.hf.space python3 scripts/benchmark_latency.py --output results.json
"""

import argparse
import asyncio
import json
import math
import os
import sys
import time
from typing import Optional
from urllib.parse import urlparse
from uuid import uuid4


def _validate_base_url(raw: str) -> str:
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid GOVON_RUNTIME_URL: {raw}")
    return raw.rstrip("/")


BASE_URL = _validate_base_url(os.environ.get("GOVON_RUNTIME_URL", "http://localhost:7860"))
API_KEY = os.environ.get("API_KEY")
DEFAULT_RUNS = 3
REQUEST_TIMEOUT = 300.0  # 단일 요청 최대 대기 (초)

# ---------------------------------------------------------------------------
# 테스트 쿼리
# ---------------------------------------------------------------------------
QUERIES = {
    "no_tool": "안녕하세요",
    "single_tool": "우리 지역 도로 파손 민원 현황 알려줘",
    "multi_tool": "최근 민원 통계와 주요 이슈를 분석해줘",
}

# ---------------------------------------------------------------------------
# HTTP 클라이언트 (httpx 우선, urllib fallback)
# ---------------------------------------------------------------------------
try:
    import httpx as _httpx

    _HTTP_BACKEND = "httpx"

    def _headers(accept: str = "application/json") -> dict:
        h = {"Content-Type": "application/json", "Accept": accept}
        if API_KEY:
            h["X-API-Key"] = API_KEY
        return h

    async def _get(path: str) -> tuple[int, float, dict]:
        url = BASE_URL + path
        t0 = time.perf_counter()
        async with _httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as c:
            r = await c.get(url, headers=_headers())
        elapsed = time.perf_counter() - t0
        try:
            return r.status_code, elapsed, r.json()
        except Exception:
            return r.status_code, elapsed, {"_raw": r.text[:200]}

    async def _post(path: str, body: dict) -> tuple[int, float, dict]:
        url = BASE_URL + path
        t0 = time.perf_counter()
        async with _httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as c:
            r = await c.post(url, json=body, headers=_headers())
        elapsed = time.perf_counter() - t0
        try:
            return r.status_code, elapsed, r.json()
        except Exception:
            return r.status_code, elapsed, {"_raw": r.text[:200]}

    async def _post_sse(path: str, body: dict) -> tuple[int, float, float]:
        """SSE 스트리밍: (status, ttft_s, total_s) 반환. ttft는 첫 data 이벤트까지."""
        url = BASE_URL + path
        h = _headers(accept="text/event-stream")
        t0 = time.perf_counter()
        ttft: Optional[float] = None
        async with _httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as c:
            async with c.stream("POST", url, json=body, headers=h) as resp:
                status = resp.status_code
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if line.startswith("data:") and ttft is None:
                        ttft = time.perf_counter() - t0
        total = time.perf_counter() - t0
        return status, ttft if ttft is not None else total, total

except ImportError:
    import urllib.error
    import urllib.request

    _HTTP_BACKEND = "urllib"

    def _headers(accept: str = "application/json") -> dict:
        h = {"Content-Type": "application/json", "Accept": accept}
        if API_KEY:
            h["X-API-Key"] = API_KEY
        return h

    async def _get(path: str) -> tuple[int, float, dict]:
        url = BASE_URL + path
        req = urllib.request.Request(url, headers=_headers(), method="GET")
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
                elapsed = time.perf_counter() - t0
                return r.status, elapsed, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            elapsed = time.perf_counter() - t0
            return e.code, elapsed, {}
        except Exception:
            return 0, time.perf_counter() - t0, {}

    async def _post(path: str, body: dict) -> tuple[int, float, dict]:
        url = BASE_URL + path
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, headers=_headers(), method="POST")
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
                elapsed = time.perf_counter() - t0
                return r.status, elapsed, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            elapsed = time.perf_counter() - t0
            return e.code, elapsed, {}
        except Exception:
            return 0, time.perf_counter() - t0, {}

    async def _post_sse(path: str, body: dict) -> tuple[int, float, float]:
        url = BASE_URL + path
        data = json.dumps(body).encode()
        h = _headers(accept="text/event-stream")
        req = urllib.request.Request(url, data=data, headers=h, method="POST")
        t0 = time.perf_counter()
        ttft: Optional[float] = None
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
                status = r.status
                for raw in r:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if line.startswith("data:") and ttft is None:
                        ttft = time.perf_counter() - t0
        except urllib.error.HTTPError as e:
            status = e.code
        total = time.perf_counter() - t0
        return status, ttft if ttft is not None else total, total


# ---------------------------------------------------------------------------
# 통계 계산
# ---------------------------------------------------------------------------


def _stats(samples: list[float]) -> dict:
    """min/avg/max/p95 (초 단위) 계산."""
    if not samples:
        return {"min": 0.0, "avg": 0.0, "max": 0.0, "p95": 0.0, "samples": []}
    s = sorted(samples)
    n = len(s)
    p95_idx = max(0, math.ceil(n * 0.95) - 1)
    return {
        "min": round(s[0], 3),
        "avg": round(sum(s) / n, 3),
        "max": round(s[-1], 3),
        "p95": round(s[p95_idx], 3),
        "samples": [round(x, 3) for x in samples],
    }


def _fmt_s(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def _fmt_stats(st: dict) -> tuple[str, str, str, str]:
    return _fmt_s(st["min"]), _fmt_s(st["avg"]), _fmt_s(st["max"]), _fmt_s(st["p95"])


# ---------------------------------------------------------------------------
# 각 시나리오 측정 함수
# ---------------------------------------------------------------------------


async def measure_health(runs: int) -> dict:
    samples: list[float] = []
    errors = 0
    for _ in range(runs):
        status, elapsed, _ = await _get("/health")
        if status == 200:
            samples.append(elapsed)
        else:
            errors += 1
    return {"label": "Health (GET /health)", "stats": _stats(samples), "errors": errors}


async def measure_v3_run(label: str, query: str, runs: int) -> dict:
    samples: list[float] = []
    errors = 0
    for _ in range(runs):
        body = {
            "query": query,
            "session_id": uuid4().hex,
            "max_iterations": 10,
            "max_tokens": 512,
            "temperature": 0.7,
        }
        status, elapsed, _ = await _post("/v3/agent/run", body)
        if status == 200:
            samples.append(elapsed)
        else:
            errors += 1
    return {"label": label, "stats": _stats(samples), "errors": errors}


async def measure_v3_sse_ttft(runs: int) -> tuple[dict, dict]:
    ttft_samples: list[float] = []
    total_samples: list[float] = []
    errors = 0
    for _ in range(runs):
        body = {
            "query": QUERIES["no_tool"],
            "session_id": uuid4().hex,
            "max_iterations": 10,
            "max_tokens": 512,
            "temperature": 0.7,
        }
        status, ttft, total = await _post_sse("/v3/agent/stream", body)
        if status == 200:
            ttft_samples.append(ttft)
            total_samples.append(total)
        else:
            errors += 1
    return (
        {"label": "v3 SSE TTFT", "stats": _stats(ttft_samples), "errors": errors},
        {"label": "v3 SSE Total", "stats": _stats(total_samples), "errors": errors},
    )


async def measure_v2_run(runs: int) -> dict:
    """v2 agent/run: approval 대기 없이 interrupt 직전까지 latency 측정."""
    samples: list[float] = []
    errors = 0
    for _ in range(runs):
        body = {
            "query": QUERIES["single_tool"],
            "session_id": uuid4().hex,
            "max_iterations": 10,
            "max_tokens": 512,
            "temperature": 0.7,
        }
        status, elapsed, _ = await _post("/v2/agent/run", body)
        # 200(completed) 또는 202(awaiting_approval) 모두 유효한 응답
        if status in (200, 202):
            samples.append(elapsed)
        else:
            errors += 1
    return {"label": "v2 agent/run", "stats": _stats(samples), "errors": errors}


# ---------------------------------------------------------------------------
# 테이블 출력
# ---------------------------------------------------------------------------


def _print_table(rows: list[dict]) -> None:
    col_w = [23, 8, 8, 8, 8]
    header = ["Scenario", "Min", "Avg", "Max", "P95"]
    sep = "|-" + "-|-".join("-" * w for w in col_w) + "-|"
    row_fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_w) + " |"

    print(row_fmt.format(*header))
    print(sep)
    for r in rows:
        st = r["stats"]
        err_note = f" (err:{r['errors']})" if r["errors"] else ""
        label = r["label"][: col_w[0]] + err_note
        mn, avg, mx, p95 = _fmt_stats(st)
        print(row_fmt.format(label, mn, avg, mx, p95))


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------


async def run_benchmark(runs: int, output: Optional[str]) -> None:
    print(f"\n=== GovOn Latency Benchmark ===")
    print(f"Runtime: {BASE_URL}")
    print(f"HTTP backend: {_HTTP_BACKEND}")
    print(f"Runs per scenario: {runs}")
    print()

    results: list[dict] = []

    scenarios = [
        ("health", lambda: measure_health(runs)),
        ("v3_no_tool", lambda: measure_v3_run("v3 no-tool", QUERIES["no_tool"], runs)),
        ("v3_single_tool", lambda: measure_v3_run("v3 single-tool", QUERIES["single_tool"], runs)),
        ("v3_multi_tool", lambda: measure_v3_run("v3 multi-tool", QUERIES["multi_tool"], runs)),
        ("v2_run", lambda: measure_v2_run(runs)),
    ]

    for key, fn in scenarios:
        print(f"  measuring {key}...", end=" ", flush=True)
        t_start = time.perf_counter()
        result = await fn()
        elapsed_total = time.perf_counter() - t_start
        results.append({"key": key, **result})
        print(f"done ({elapsed_total:.1f}s)")

    # SSE 측정 (TTFT + total)
    print(f"  measuring v3_sse...", end=" ", flush=True)
    t_start = time.perf_counter()
    sse_ttft, sse_total = await measure_v3_sse_ttft(runs)
    elapsed_total = time.perf_counter() - t_start
    results.append({"key": "v3_sse_ttft", **sse_ttft})
    results.append({"key": "v3_sse_total", **sse_total})
    print(f"done ({elapsed_total:.1f}s)")

    print()
    _print_table(results)
    print()

    if output:
        out_data = {
            "meta": {
                "runtime_url": BASE_URL,
                "runs_per_scenario": runs,
                "http_backend": _HTTP_BACKEND,
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            "results": results,
        }
        with open(output, "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)
        print(f"JSON 결과 저장: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GovOn Latency Benchmark")
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        metavar="N",
        help=f"시나리오당 반복 횟수 (기본: {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="JSON 결과 파일 경로 (예: results.json)",
    )
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be >= 1")

    if not os.environ.get("GOVON_RUNTIME_URL"):
        print(
            "[WARN] GOVON_RUNTIME_URL 환경변수가 없습니다. localhost:7860 사용합니다.",
            file=sys.stderr,
        )

    asyncio.run(run_benchmark(runs=args.runs, output=args.output))


if __name__ == "__main__":
    main()
