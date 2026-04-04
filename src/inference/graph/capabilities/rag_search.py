"""rag_search capability — 로컬 문서 하이브리드 검색을 LangGraph capability로 표준화.

Issue #395: local RAG 검색을 LangGraph tool capability로 표준화.

ApiLookupCapability 패턴을 따라 파라미터 검증, 타임아웃, 결과 정규화,
fallback 정책(empty/low-confidence)을 구현한다.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from src.inference.index_manager import IndexType

from .base import CapabilityBase, CapabilityMetadata, LookupResult

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

LOW_CONFIDENCE_THRESHOLD: float = 0.3
"""저신뢰도 기본 임계값. 모든 결과 score가 이 값 미만이면 low_confidence로 분류."""

_VALID_SOURCE_TYPES: frozenset[str] = frozenset(t.value for t in IndexType)
_DEFAULT_SOURCE_TYPES: list[str] = [t.value for t in IndexType]


# ---------------------------------------------------------------------------
# 파라미터 validator
# ---------------------------------------------------------------------------


@dataclass
class RagSearchParams:
    """rag_search 호출 파라미터 — context에서 추출·정규화·검증."""

    query: str
    top_k: int = 5
    source_types: List[str] = field(default_factory=lambda: list(_DEFAULT_SOURCE_TYPES))
    min_confidence: float = LOW_CONFIDENCE_THRESHOLD

    @classmethod
    def from_context(
        cls,
        query: str,
        context: Dict[str, Any],
        default_min_confidence: float = LOW_CONFIDENCE_THRESHOLD,
    ) -> "RagSearchParams":
        """context에서 파라미터를 추출하고 alias를 정규화한다."""
        top_k = int(context.get("rag_top_k") or context.get("top_k") or context.get("count") or 5)

        raw_filters = context.get("filters") or context.get("source_types")
        if isinstance(raw_filters, list):
            source_types = [str(f).lower() for f in raw_filters]
        else:
            source_types = list(_DEFAULT_SOURCE_TYPES)

        min_confidence = float(
            context.get("rag_min_confidence")
            or context.get("min_confidence")
            or context.get("score_threshold")
            or default_min_confidence
        )

        return cls(
            query=query.strip(),
            top_k=max(1, min(50, top_k)),
            source_types=source_types,
            min_confidence=max(0.0, min(1.0, min_confidence)),
        )

    def validate(self) -> Optional[str]:
        """검증 실패 시 오류 메시지, 통과 시 None."""
        if not self.query:
            return "query가 비어있습니다"
        if len(self.query) > 2000:
            return f"query가 너무 깁니다 ({len(self.query)}자, 최대 2000자)"
        invalid = [t for t in self.source_types if t not in _VALID_SOURCE_TYPES]
        if invalid:
            return f"유효하지 않은 source_type: {invalid}"
        return None


# ---------------------------------------------------------------------------
# 결과 정규화 헬퍼
# ---------------------------------------------------------------------------


def _normalize_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    """raw SearchResult dict에 공통 스키마 필드를 보강한다.

    추가 필드: excerpt, file_path, page. 기존 필드는 그대로 유지.
    """
    content = raw.get("content", "")
    metadata = raw.get("metadata", {})
    result = dict(raw)
    result["excerpt"] = content[:500] if content else ""
    result["file_path"] = metadata.get("file_path", "")
    result["page"] = metadata.get("page", raw.get("chunk_index", 0))
    return result


# ---------------------------------------------------------------------------
# Capability
# ---------------------------------------------------------------------------


class RagSearchCapability(CapabilityBase):
    """로컬 문서(법령/매뉴얼/사례/공지) 하이브리드 검색 capability.

    기존 api_server의 _rag_search_tool closure를 주입받아
    CapabilityBase 인터페이스로 래핑하고, 파라미터 검증·타임아웃·
    결과 정규화·fallback 정책을 적용한다.

    Parameters
    ----------
    execute_fn : Callable
        ``async (query, context, session) -> dict`` 시그니처의 실행 함수.
    low_confidence_threshold : float
        저신뢰도 임계값. 모든 결과가 이 값 미만이면 ``low_confidence``로 분류.
    """

    def __init__(
        self,
        execute_fn: Callable[..., Any],
        low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._execute_fn = execute_fn
        self._low_confidence_threshold = low_confidence_threshold

    @property
    def metadata(self) -> CapabilityMetadata:
        return CapabilityMetadata(
            name="rag_search",
            description=(
                "내부 법령, 매뉴얼, 민원 사례, 공지사항 등 로컬 문서를 "
                "하이브리드 검색(BM25 + 벡터)으로 조회합니다."
            ),
            approval_summary="로컬 문서 DB에서 관련 법령/사례를 검색합니다.",
            provider="local_vectordb",
            timeout_sec=15.0,
        )

    async def execute(
        self,
        query: str,
        context: Dict[str, Any],
        session: Any,
    ) -> LookupResult:
        """검색 실행 — 파라미터 검증, 타임아웃, 정규화, fallback 적용."""
        provider = self.metadata.provider

        params = RagSearchParams.from_context(
            query, context, default_min_confidence=self._low_confidence_threshold
        )
        validation_error = params.validate()
        if validation_error:
            return LookupResult(
                success=False,
                query=query,
                provider=provider,
                error=validation_error,
                empty_reason="validation_error",
            )

        try:
            raw = await asyncio.wait_for(
                self._execute_fn(query=params.query, context=context, session=session),
                timeout=self.metadata.timeout_sec,
            )
        except asyncio.TimeoutError:
            logger.warning(f"[rag_search] 타임아웃 ({self.metadata.timeout_sec}s 초과)")
            return LookupResult(
                success=False,
                query=params.query,
                provider=provider,
                error=f"검색 타임아웃 ({self.metadata.timeout_sec}초 초과)",
                empty_reason="provider_error",
            )
        except Exception as exc:
            logger.error(f"[rag_search] 검색 오류: {exc}", exc_info=True)
            return LookupResult(
                success=False,
                query=params.query,
                provider=provider,
                error=str(exc),
                empty_reason="provider_error",
            )

        if not isinstance(raw, dict):
            raw = {}

        if raw.get("error"):
            return LookupResult(
                success=False,
                query=raw.get("query", params.query),
                provider=provider,
                error=raw["error"],
                empty_reason="provider_error",
            )

        raw_query = raw.get("query", params.query)
        raw_context_text = raw.get("context_text", "")
        raw_results = raw.get("results", [])
        normalized = [_normalize_result(r) for r in raw_results]

        if not normalized:
            return LookupResult(
                success=True,
                query=raw_query,
                provider=provider,
                empty_reason="no_match",
                context_text=raw_context_text,
            )

        confident = [r for r in normalized if r["score"] >= params.min_confidence]
        if not confident:
            logger.info(f"[rag_search] 모든 결과가 저신뢰도 (threshold={params.min_confidence})")
            return LookupResult(
                success=True,
                query=raw_query,
                results=normalized,
                context_text=raw_context_text,
                provider=provider,
                empty_reason="low_confidence",
            )

        citations = [
            {
                "source_type": r["source_type"],
                "doc_id": r["doc_id"],
                "title": r["title"],
                "score": r["score"],
                "excerpt": r["excerpt"][:200],
            }
            for r in confident
        ]

        return LookupResult(
            success=True,
            query=raw_query,
            results=confident,
            context_text=raw_context_text,
            citations=citations,
            provider=provider,
        )
