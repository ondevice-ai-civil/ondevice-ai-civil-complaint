# TDD: Red-Green-Refactor cycle로 구현됨
# TDD Phase: RED - 테스트 먼저 작성
"""RequestRateTracker 단위 테스트.

요청 통계 추적기의 핵심 기능을 테스트한다:
- 슬라이딩 윈도우 기반 요청 수 카운팅
- 엔드포인트별 요청 통계
- 시간대별 집계
- 통계 리셋
- 동시성 안전
"""

import asyncio
import time

import pytest

from src.inference.rate_tracker import EndpointStats, RequestRateTracker


@pytest.fixture
def tracker() -> RequestRateTracker:
    return RequestRateTracker(window_seconds=60)


class TestRequestRecording:
    """요청 기록."""

    @pytest.mark.asyncio
    async def test_record_request(self, tracker: RequestRateTracker):
        await tracker.record("/api/generate", latency_ms=150.0, is_error=False)
        stats = await tracker.get_endpoint_stats("/api/generate")
        assert stats.request_count == 1

    @pytest.mark.asyncio
    async def test_record_multiple_requests(self, tracker: RequestRateTracker):
        for _ in range(5):
            await tracker.record("/api/generate", latency_ms=100.0, is_error=False)
        stats = await tracker.get_endpoint_stats("/api/generate")
        assert stats.request_count == 5

    @pytest.mark.asyncio
    async def test_record_error_request(self, tracker: RequestRateTracker):
        await tracker.record("/api/generate", latency_ms=50.0, is_error=True)
        stats = await tracker.get_endpoint_stats("/api/generate")
        assert stats.error_count == 1


class TestSlidingWindow:
    """슬라이딩 윈도우 기반 카운팅."""

    @pytest.mark.asyncio
    async def test_expired_requests_excluded(self):
        tracker = RequestRateTracker(window_seconds=1)
        await tracker.record("/api/generate", latency_ms=100.0, is_error=False)
        await asyncio.sleep(1.1)
        count = await tracker.get_request_count("/api/generate")
        assert count == 0

    @pytest.mark.asyncio
    async def test_active_requests_counted(self, tracker: RequestRateTracker):
        await tracker.record("/api/generate", latency_ms=100.0, is_error=False)
        count = await tracker.get_request_count("/api/generate")
        assert count == 1


class TestEndpointStats:
    """엔드포인트별 통계."""

    @pytest.mark.asyncio
    async def test_avg_latency_calculation(self, tracker: RequestRateTracker):
        await tracker.record("/api/generate", latency_ms=100.0, is_error=False)
        await tracker.record("/api/generate", latency_ms=200.0, is_error=False)
        await tracker.record("/api/generate", latency_ms=300.0, is_error=False)
        stats = await tracker.get_endpoint_stats("/api/generate")
        assert stats.avg_latency_ms == pytest.approx(200.0)

    @pytest.mark.asyncio
    async def test_error_rate_calculation(self, tracker: RequestRateTracker):
        await tracker.record("/api/generate", latency_ms=100.0, is_error=False)
        await tracker.record("/api/generate", latency_ms=100.0, is_error=True)
        stats = await tracker.get_endpoint_stats("/api/generate")
        assert stats.error_rate == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_separate_endpoints(self, tracker: RequestRateTracker):
        await tracker.record("/api/generate", latency_ms=100.0, is_error=False)
        await tracker.record("/api/search", latency_ms=200.0, is_error=False)
        gen_stats = await tracker.get_endpoint_stats("/api/generate")
        search_stats = await tracker.get_endpoint_stats("/api/search")
        assert gen_stats.request_count == 1
        assert search_stats.request_count == 1

    @pytest.mark.asyncio
    async def test_unknown_endpoint_returns_empty_stats(self, tracker: RequestRateTracker):
        stats = await tracker.get_endpoint_stats("/api/unknown")
        assert stats.request_count == 0
        assert stats.avg_latency_ms == 0.0
        assert stats.error_rate == 0.0


class TestTimeAggregation:
    """시간대별 집계."""

    @pytest.mark.asyncio
    async def test_get_all_stats(self, tracker: RequestRateTracker):
        await tracker.record("/api/generate", latency_ms=100.0, is_error=False)
        await tracker.record("/api/search", latency_ms=200.0, is_error=False)
        all_stats = await tracker.get_all_stats()
        assert "/api/generate" in all_stats
        assert "/api/search" in all_stats

    @pytest.mark.asyncio
    async def test_total_request_count(self, tracker: RequestRateTracker):
        await tracker.record("/api/generate", latency_ms=100.0, is_error=False)
        await tracker.record("/api/search", latency_ms=200.0, is_error=False)
        await tracker.record("/api/generate", latency_ms=150.0, is_error=False)
        total = await tracker.get_total_request_count()
        assert total == 3


class TestStatsReset:
    """통계 리셋."""

    @pytest.mark.asyncio
    async def test_reset_clears_all(self, tracker: RequestRateTracker):
        await tracker.record("/api/generate", latency_ms=100.0, is_error=False)
        await tracker.record("/api/search", latency_ms=200.0, is_error=False)
        await tracker.reset()
        all_stats = await tracker.get_all_stats()
        assert len(all_stats) == 0

    @pytest.mark.asyncio
    async def test_reset_endpoint(self, tracker: RequestRateTracker):
        await tracker.record("/api/generate", latency_ms=100.0, is_error=False)
        await tracker.record("/api/search", latency_ms=200.0, is_error=False)
        await tracker.reset_endpoint("/api/generate")
        gen_stats = await tracker.get_endpoint_stats("/api/generate")
        search_stats = await tracker.get_endpoint_stats("/api/search")
        assert gen_stats.request_count == 0
        assert search_stats.request_count == 1


class TestConcurrencySafety:
    """동시성 안전 (asyncio.Lock)."""

    @pytest.mark.asyncio
    async def test_concurrent_records(self, tracker: RequestRateTracker):
        async def record_batch():
            for _ in range(100):
                await tracker.record("/api/generate", latency_ms=50.0, is_error=False)

        await asyncio.gather(record_batch(), record_batch(), record_batch())
        stats = await tracker.get_endpoint_stats("/api/generate")
        assert stats.request_count == 300


class TestEndpointStatsDataclass:
    """EndpointStats dataclass 검증."""

    def test_endpoint_stats_fields(self):
        stats = EndpointStats(
            request_count=10,
            error_count=2,
            avg_latency_ms=150.0,
            error_rate=0.2,
        )
        assert stats.request_count == 10
        assert stats.error_count == 2
        assert stats.avg_latency_ms == 150.0
        assert stats.error_rate == 0.2
