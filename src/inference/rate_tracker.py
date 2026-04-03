"""요청 통계 추적기.

슬라이딩 윈도우 기반으로 요청 수를 카운팅하고 엔드포인트별 통계를 추적한다:
- 슬라이딩 윈도우 기반 요청 수 카운팅
- 엔드포인트별 요청 통계 (count, avg_latency, error_rate)
- 통계 리셋
- 동시성 안전 (asyncio.Lock)
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List, NamedTuple

from loguru import logger


@dataclass
class EndpointStats:
    """엔드포인트별 집계 통계."""

    request_count: int = 0
    error_count: int = 0
    avg_latency_ms: float = 0.0
    error_rate: float = 0.0


class _RequestRecord(NamedTuple):
    """개별 요청 기록."""

    timestamp: float
    latency_ms: float
    is_error: bool


class RequestRateTracker:
    """슬라이딩 윈도우 기반 요청 통계 추적기."""

    def __init__(self, window_seconds: float = 60.0) -> None:
        self._window_seconds = window_seconds
        self._records: Dict[str, List[_RequestRecord]] = {}
        self._lock = asyncio.Lock()

    async def record(self, endpoint: str, latency_ms: float, is_error: bool = False) -> None:
        """요청을 기록한다."""
        async with self._lock:
            if endpoint not in self._records:
                self._records[endpoint] = []
            self._records[endpoint].append(
                _RequestRecord(
                    timestamp=time.monotonic(),
                    latency_ms=latency_ms,
                    is_error=is_error,
                )
            )

    def _active_records(self, endpoint: str) -> List[_RequestRecord]:
        """윈도우 내 유효한 레코드만 반환한다."""
        now = time.monotonic()
        cutoff = now - self._window_seconds
        records = self._records.get(endpoint, [])
        return [r for r in records if r.timestamp > cutoff]

    async def get_request_count(self, endpoint: str) -> int:
        """윈도우 내 요청 수를 반환한다."""
        async with self._lock:
            return len(self._active_records(endpoint))

    async def get_endpoint_stats(self, endpoint: str) -> EndpointStats:
        """엔드포인트별 통계를 반환한다."""
        async with self._lock:
            active = self._active_records(endpoint)
            if not active:
                return EndpointStats()

            total = len(active)
            errors = sum(1 for r in active if r.is_error)
            avg_latency = sum(r.latency_ms for r in active) / total

            return EndpointStats(
                request_count=total,
                error_count=errors,
                avg_latency_ms=avg_latency,
                error_rate=errors / total,
            )

    async def get_all_stats(self) -> Dict[str, EndpointStats]:
        """전체 엔드포인트 통계를 반환한다."""
        async with self._lock:
            result: Dict[str, EndpointStats] = {}
            for endpoint in list(self._records.keys()):
                active = self._active_records(endpoint)
                if not active:
                    continue
                total = len(active)
                errors = sum(1 for r in active if r.is_error)
                avg_latency = sum(r.latency_ms for r in active) / total
                result[endpoint] = EndpointStats(
                    request_count=total,
                    error_count=errors,
                    avg_latency_ms=avg_latency,
                    error_rate=errors / total,
                )
            return result

    async def get_total_request_count(self) -> int:
        """전체 윈도우 내 요청 수를 반환한다."""
        async with self._lock:
            total = 0
            for endpoint in self._records:
                total += len(self._active_records(endpoint))
            return total

    async def reset(self) -> None:
        """전체 통계를 초기화한다."""
        async with self._lock:
            self._records.clear()
            logger.info("요청 통계 전체 리셋")

    async def reset_endpoint(self, endpoint: str) -> None:
        """특정 엔드포인트 통계를 초기화한다."""
        async with self._lock:
            self._records.pop(endpoint, None)
            logger.info(f"요청 통계 리셋: {endpoint}")
