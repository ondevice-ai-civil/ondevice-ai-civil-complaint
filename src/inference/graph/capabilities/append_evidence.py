"""append_evidence capability — 기존 closure를 CapabilityBase로 래핑."""

from __future__ import annotations

from typing import Any, Callable, Dict

from .base import CapabilityBase, CapabilityMetadata, LookupResult


class AppendEvidenceCapability(CapabilityBase):
    """근거/출처 보강 capability.

    기존 api_server의 _append_evidence_tool closure를 주입받아
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
            name="append_evidence",
            description=(
                "기존 답변에 법령 근거, 유사 사례, 외부 통계 등 " "추가 출처를 보강합니다."
            ),
            approval_summary="기존 답변에 법적 근거와 출처를 추가합니다.",
            provider="local_vectordb+data.go.kr",
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
                query=query,
                provider=self.metadata.provider,
                error=raw["error"],
                empty_reason="provider_error",
            )

        text = raw.get("text", "") if isinstance(raw, dict) else str(raw)
        citations = raw.get("api_citations", []) if isinstance(raw, dict) else []
        rag_results = raw.get("rag_results", []) if isinstance(raw, dict) else []
        return LookupResult(
            success=True,
            query=query,
            context_text=text,
            citations=citations,
            results=rag_results,
            provider=self.metadata.provider,
        )
