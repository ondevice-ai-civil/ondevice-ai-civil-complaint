"""검색 결과 캐싱.

TTL 기반 인메모리 캐시로 검색 결과를 캐싱한다:
- TTL 만료 자동 처리
- LRU eviction (최대 크기 제한)
- 캐시 히트/미스 통계
- 쿼리+doc_type 기반 캐시 키 생성
"""

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class _CacheEntry:
    """캐시 엔트리."""

    value: List[Dict[str, Any]]
    expires_at: float


class SearchResultCache:
    """TTL 기반 LRU 인메모리 검색 결과 캐시."""

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 300.0) -> None:
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0

    def make_key(self, query: str, doc_type: str) -> str:
        """쿼리와 doc_type으로 캐시 키를 생성한다."""
        raw = f"{query}:{doc_type}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    async def get(self, query: str, doc_type: str) -> Optional[List[Dict[str, Any]]]:
        """캐시에서 검색 결과를 조회한다."""
        key = self.make_key(query, doc_type)
        entry = self._cache.get(key)

        if entry is None:
            self._misses += 1
            return None

        # TTL 만료 체크
        if time.monotonic() > entry.expires_at:
            del self._cache[key]
            self._misses += 1
            return None

        # LRU: 최근 사용으로 이동
        self._cache.move_to_end(key)
        self._hits += 1
        return entry.value

    async def set(self, query: str, doc_type: str, results: List[Dict[str, Any]]) -> None:
        """캐시에 검색 결과를 저장한다."""
        key = self.make_key(query, doc_type)

        # 이미 존재하면 갱신
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = _CacheEntry(
                value=results,
                expires_at=time.monotonic() + self._ttl_seconds,
            )
            return

        # LRU eviction: 최대 크기 도달 시 가장 오래된 항목 제거
        while len(self._cache) >= self._max_size:
            evicted_key, _ = self._cache.popitem(last=False)
            logger.debug(f"캐시 LRU eviction: {evicted_key[:16]}...")

        self._cache[key] = _CacheEntry(
            value=results,
            expires_at=time.monotonic() + self._ttl_seconds,
        )

    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계를 반환한다."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "size": len(self._cache),
            "max_size": self._max_size,
        }

    def reset_stats(self) -> None:
        """통계를 초기화한다."""
        self._hits = 0
        self._misses = 0

    def clear(self) -> None:
        """캐시를 모두 비운다."""
        self._cache.clear()
        logger.info("검색 캐시 초기화 완료")
