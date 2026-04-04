"""rag_search capability — 기존 closure를 CapabilityBase로 래핑."""

from __future__ import annotations

from typing import Any, Callable, Dict

from .base import CapabilityBase, CapabilityMetadata, LookupResult


class RagSearchCapability(CapabilityBase):
    """로컬 문서(법령/매뉴얼/사례/공지) 검색 capability.

    기존 api_server의 _rag_search_tool closure를 주입받아
    CapabilityBase 인터페이스로 래핑한다.

    Parameters
    ----------
    execute_fn : Callable
        ``async (query, context, session) -> dict`` 시그니처의 실행 함수.
    """

    def __init__(self, execute_fn: Callable[..., Any]) -> None:
        self._execute_fn = execute_fn

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
        """주입받은 함수에 위임하고 결과를 LookupResult로 변환한다."""
        raw = await self._execute_fn(query=query, context=context, session=session)

        if isinstance(raw, dict) and raw.get("error"):
            return LookupResult(
                success=False,
                query=raw.get("query", query),
                provider=self.metadata.provider,
                error=raw["error"],
                empty_reason="provider_error",
            )

        results = raw.get("results", []) if isinstance(raw, dict) else []
        return LookupResult(
            success=True,
            query=raw.get("query", query) if isinstance(raw, dict) else query,
            results=results,
            context_text=raw.get("context_text", "") if isinstance(raw, dict) else "",
            provider=self.metadata.provider,
            empty_reason="no_match" if not results else None,
        )
