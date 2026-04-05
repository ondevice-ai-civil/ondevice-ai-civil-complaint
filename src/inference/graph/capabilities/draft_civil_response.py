"""draft_civil_response capability — 기존 closure를 CapabilityBase로 래핑."""

from __future__ import annotations

from typing import Any, Callable, Dict

from .base import CapabilityBase, CapabilityMetadata, EvidenceEnvelope, EvidenceItem, LookupResult
from .defaults import get_timeout


class DraftCivilResponseCapability(CapabilityBase):
    """민원 답변 초안 생성 capability.

    기존 api_server의 _draft_civil_response_tool closure를 주입받아
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
            name="draft_civil_response",
            description=(
                "검색된 법령/사례와 외부 민원분석 결과를 종합하여 " "민원 답변 초안을 생성합니다."
            ),
            approval_summary="AI 모델이 검색 결과를 종합하여 민원 답변 초안을 생성합니다.",
            provider="local_llm",
            timeout_sec=get_timeout("draft_civil_response"),
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

        # draft에서 참조된 사례를 EvidenceItem으로 변환
        evidence_items: list[EvidenceItem] = []
        if isinstance(raw, dict):
            # raw에 포함된 citations/references를 EvidenceItem으로 변환
            for ref in raw.get("citations", []):
                if not isinstance(ref, dict):
                    continue
                title = ref.get("title") or ref.get("qnaTitle") or ref.get("question", "")
                excerpt = ref.get("content") or ref.get("qnaContent") or ref.get("qnaAnswer", "")
                link = ref.get("url") or ref.get("detailUrl", "")
                evidence_items.append(
                    EvidenceItem(
                        source_type="llm_generated",
                        title=str(title),
                        excerpt=str(excerpt)[:500],
                        link_or_path=str(link),
                        provider_meta={"provider": self.metadata.provider},
                    )
                )

        envelope = EvidenceEnvelope(
            items=evidence_items,
            summary_text=text,
            status="ok",
        )

        return LookupResult(
            success=True,
            query=query,
            context_text=text,
            provider=self.metadata.provider,
            # draft 결과는 results 대신 context_text에 담긴다
            results=[raw] if isinstance(raw, dict) else [],
            evidence=envelope,
        )
