"""결과 리포트 생성 — JSON + 터미널 요약."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .config import BASE_URL, VALID_TOOLS
from .flow_tracker import LatencyAggregator
from .http_client import get_http_backend
from .logger import E2ELogger


def write_json_report(
    results: List[dict],
    output_path: str,
    run_id: str,
    cold_start_wait: float = 0.0,
    observed_tools: Optional[set] = None,
    latency_aggregator: Optional[LatencyAggregator] = None,
) -> None:
    """JSON 결과 파일을 출력한다."""
    _observed = observed_tools or set()
    passed = sum(1 for r in results if r.get("status") == "passed")
    failed = sum(1 for r in results if r.get("status") == "failed")
    skipped = sum(1 for r in results if r.get("status") == "skipped")

    tool_ratio = len(_observed) / len(VALID_TOOLS) if VALID_TOOLS else 0

    output: Dict[str, Any] = {
        "meta": {
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "target_url": BASE_URL,
            "cold_start_wait_seconds": cold_start_wait,
            "http_backend": get_http_backend(),
        },
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "tool_coverage": {
                "observed": sorted(_observed),
                "ratio": round(tool_ratio, 2),
            },
        },
        "scenarios": results,
    }

    if latency_aggregator:
        output["latency_stats"] = latency_aggregator.all_stats()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def print_summary(results: List[dict], logger: E2ELogger, observed_tools: set) -> None:
    """터미널에 결과 요약을 출력한다."""
    passed = sum(1 for r in results if r.get("status") == "passed")
    failed = sum(1 for r in results if r.get("status") == "failed")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    total = len(results)

    logger.info("=" * 60)
    logger.info(f"결과: {passed}/{total} 통과, {failed} 실패, {skipped} 스킵")

    tool_ratio = len(observed_tools) / len(VALID_TOOLS) if VALID_TOOLS else 0
    logger.info(f"도구 커버리지: {len(observed_tools)}/{len(VALID_TOOLS)} ({tool_ratio:.0%})")
    if observed_tools:
        logger.info(f"  관측된 도구: {sorted(observed_tools)}")
