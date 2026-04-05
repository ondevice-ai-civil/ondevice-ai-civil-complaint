"""append_evidence capability — 기존 closure를 CapabilityBase로 래핑."""

from __future__ import annotations

from typing import Any, Callable, Dict

from .base import CapabilityBase, CapabilityMetadata, EvidenceEnvelope, EvidenceItem, LookupResult


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
                evidence=EvidenceEnvelope(
                    status="error",
                    errors=[raw["error"]],
                ),
            )

        text = raw.get("text", "") if isinstance(raw, dict) else str(raw)
        citations = raw.get("api_citations", []) if isinstance(raw, dict) else []
        rag_results = raw.get("rag_results", []) if isinstance(raw, dict) else []

        # 이전 단계의 evidence를 합산하여 EvidenceEnvelope 구성
        evidence_items: list[EvidenceItem] = []
        errors: list[str] = []

        # rag_results -> EvidenceItem 변환
        for item in rag_results:
            if not isinstance(item, dict):
                continue
            metadata = item.get("metadata", {}) or {}
            evidence_items.append(
                EvidenceItem(
                    source_type="rag",
                    title=item.get("title", ""),
                    excerpt=str(item.get("content", ""))[:500],
                    link_or_path=metadata.get("file_path", ""),
                    page=metadata.get("page"),
                    score=float(item.get("score", 0.0)),
                    provider_meta={"provider": "local_vectordb"},
                )
            )

        # api_citations -> EvidenceItem 변환
        for c in citations:
            if not isinstance(c, dict):
                continue
            title = c.get("title") or c.get("qnaTitle") or c.get("question", "")
            excerpt = c.get("content") or c.get("qnaContent") or c.get("qnaAnswer", "")
            link = c.get("url") or c.get("detailUrl", "")
            evidence_items.append(
                EvidenceItem(
                    source_type="api",
                    title=str(title),
                    excerpt=str(excerpt)[:500],
                    link_or_path=str(link),
                    score=float(c.get("score", 0)),
                    provider_meta={"provider": "data.go.kr"},
                )
            )

        if isinstance(raw, dict):
            raw_errors = raw.get("errors", [])
            if isinstance(raw_errors, list):
                errors = [str(e) for e in raw_errors]

        status: str
        if not evidence_items and errors:
            status = "error"
        elif not evidence_items:
            status = "empty"
        elif errors:
            status = "partial"
        else:
            status = "ok"

        envelope = EvidenceEnvelope(
            items=evidence_items,
            summary_text=text,
            status=status,
            errors=errors,
        )

        return LookupResult(
            success=True,
            query=query,
            context_text=text,
            citations=citations,
            results=rag_results,
            provider=self.metadata.provider,
            evidence=envelope,
        )
