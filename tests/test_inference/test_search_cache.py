# TDD: Red-Green-Refactor cycle로 구현됨
# TDD Phase: RED - 테스트 먼저 작성
"""SearchResultCache 단위 테스트.

TTL 기반 인메모리 캐시의 핵심 기능을 테스트한다:
- async get/set 인터페이스
- TTL 만료
- LRU eviction
- 캐시 히트/미스 통계
- 쿼리+doc_type 기반 캐시 키 생성
"""

import asyncio
import time

import pytest

from src.inference.search_cache import SearchResultCache


@pytest.fixture
def cache() -> SearchResultCache:
    return SearchResultCache(max_size=5, ttl_seconds=300)


@pytest.fixture
def small_cache() -> SearchResultCache:
    return SearchResultCache(max_size=3, ttl_seconds=1)


class TestAsyncGetSet:
    """async get/set 인터페이스."""

    @pytest.mark.asyncio
    async def test_set_and_get(self, cache: SearchResultCache):
        await cache.set("민원 질문", "case", [{"score": 0.9, "text": "결과"}])
        result = await cache.get("민원 질문", "case")
        assert result is not None
        assert result[0]["score"] == 0.9

    @pytest.mark.asyncio
    async def test_get_miss_returns_none(self, cache: SearchResultCache):
        result = await cache.get("존재하지 않는 쿼리", "case")
        assert result is None

    @pytest.mark.asyncio
    async def test_different_doc_types_separate(self, cache: SearchResultCache):
        await cache.set("질문", "case", [{"type": "case"}])
        await cache.set("질문", "law", [{"type": "law"}])
        case_result = await cache.get("질문", "case")
        law_result = await cache.get("질문", "law")
        assert case_result[0]["type"] == "case"
        assert law_result[0]["type"] == "law"


class TestTTLExpiration:
    """TTL 기반 만료."""

    @pytest.mark.asyncio
    async def test_expired_entry_returns_none(self, small_cache: SearchResultCache):
        await small_cache.set("질문", "case", [{"data": 1}])
        # TTL이 1초이므로 대기 후 만료 확인
        await asyncio.sleep(1.1)
        result = await small_cache.get("질문", "case")
        assert result is None

    @pytest.mark.asyncio
    async def test_not_expired_entry_returns_value(self, cache: SearchResultCache):
        await cache.set("질문", "case", [{"data": 1}])
        # TTL이 300초이므로 즉시 조회하면 유효
        result = await cache.get("질문", "case")
        assert result is not None


class TestLRUEviction:
    """LRU eviction 테스트."""

    @pytest.mark.asyncio
    async def test_evicts_oldest_when_full(self, small_cache: SearchResultCache):
        # max_size=3, TTL=1초로 생성된 small_cache 사용하되 TTL을 길게 재생성
        cache = SearchResultCache(max_size=3, ttl_seconds=300)
        await cache.set("q1", "case", [{"id": 1}])
        await cache.set("q2", "case", [{"id": 2}])
        await cache.set("q3", "case", [{"id": 3}])
        # 4번째 삽입 -> q1이 evict 되어야 함
        await cache.set("q4", "case", [{"id": 4}])

        assert await cache.get("q1", "case") is None
        assert await cache.get("q4", "case") is not None

    @pytest.mark.asyncio
    async def test_access_refreshes_lru_order(self):
        cache = SearchResultCache(max_size=3, ttl_seconds=300)
        await cache.set("q1", "case", [{"id": 1}])
        await cache.set("q2", "case", [{"id": 2}])
        await cache.set("q3", "case", [{"id": 3}])
        # q1을 조회하여 LRU 순서를 갱신
        await cache.get("q1", "case")
        # q4 삽입 -> q2가 evict 되어야 함 (q1은 최근 사용)
        await cache.set("q4", "case", [{"id": 4}])

        assert await cache.get("q1", "case") is not None
        assert await cache.get("q2", "case") is None


class TestCacheStatistics:
    """캐시 히트/미스 통계."""

    @pytest.mark.asyncio
    async def test_hit_count_incremented(self, cache: SearchResultCache):
        await cache.set("질문", "case", [{"data": 1}])
        await cache.get("질문", "case")
        await cache.get("질문", "case")
        stats = cache.get_stats()
        assert stats["hits"] == 2

    @pytest.mark.asyncio
    async def test_miss_count_incremented(self, cache: SearchResultCache):
        await cache.get("없는질문", "case")
        stats = cache.get_stats()
        assert stats["misses"] == 1

    @pytest.mark.asyncio
    async def test_hit_rate_calculation(self, cache: SearchResultCache):
        await cache.set("질문", "case", [{"data": 1}])
        await cache.get("질문", "case")  # hit
        await cache.get("없음", "case")  # miss
        stats = cache.get_stats()
        assert stats["hit_rate"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_stats_reset(self, cache: SearchResultCache):
        await cache.set("질문", "case", [{"data": 1}])
        await cache.get("질문", "case")
        cache.reset_stats()
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0


class TestCacheKeyGeneration:
    """쿼리+doc_type 기반 캐시 키 생성."""

    def test_same_query_different_type_different_key(self, cache: SearchResultCache):
        key1 = cache.make_key("질문", "case")
        key2 = cache.make_key("질문", "law")
        assert key1 != key2

    def test_same_inputs_same_key(self, cache: SearchResultCache):
        key1 = cache.make_key("질문", "case")
        key2 = cache.make_key("질문", "case")
        assert key1 == key2


class TestCacheClear:
    """캐시 클리어."""

    @pytest.mark.asyncio
    async def test_clear_removes_all_entries(self, cache: SearchResultCache):
        await cache.set("q1", "case", [{"id": 1}])
        await cache.set("q2", "law", [{"id": 2}])
        cache.clear()
        assert await cache.get("q1", "case") is None
        assert await cache.get("q2", "law") is None
        stats = cache.get_stats()
        assert stats["size"] == 0
