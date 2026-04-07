"""keyword_analyzer capability — 핵심키워드+연관어 조합.

Issue #488: 민원 키워드 분석 도구.
2개 API(핵심키워드, 연관어)를 조합하여
키워드 분석 결과를 제공한다.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from loguru import logger

from .base import (
    CapabilityBase,
    CapabilityMetadata,
    EvidenceEnvelope,
    EvidenceItem,
    LookupResult,
)
from .defaults import get_timeout


class KeywordAnalyzerCapability(CapabilityBase):
    """민원 키워드 분석 capability.

    핵심키워드와 연관어를 조합하여 키워드 분석 결과를 제공한다.

    Parameters
    ----------
    action : Optional[MinwonAnalysisAction]
        API 호출용 Action 인스턴스. None이면 빈 결과 반환.
    """

    def __init__(self, action: Optional[Any] = None) -> None:
        self._action = action

    @property
    def metadata(self) -> CapabilityMetadata:
        return CapabilityMetadata(
            name="keyword_analyzer",
            description=("핵심키워드와 연관어를 조합하여 " "민원 키워드 분석 결과를 제공합니다."),
            approval_summary="공공데이터포털에서 민원 키워드를 분석합니다.",
            provider="data.go.kr",
            timeout_sec=get_timeout("keyword_analyzer"),
        )

    async def execute(
        self,
        query: str,
        context: Dict[str, Any],
        session: Any,
    ) -> LookupResult:
        """핵심키워드 + 연관어 API를 병렬 호출하고 결과를 조합한다."""
        provider = self.metadata.provider

        if not query or not query.strip():
            return LookupResult(
                success=False,
                query=query,
                provider=provider,
                error="query가 비어있습니다",
                empty_reason="validation_error",
                evidence=EvidenceEnvelope(status="error", errors=["query가 비어있습니다"]),
            )

        if self._action is None:
            logger.debug("[keyword_analyzer] action이 None - 빈 결과 반환")
            return LookupResult(
                success=True,
                query=query,
                provider=provider,
                empty_reason="no_match",
                evidence=EvidenceEnvelope(status="empty"),
            )

        date_from = context.get("date_from", "")
        date_to = context.get("date_to", "")
        searchword = context.get("searchword", "")
        result_count = int(context.get("result_count", 5))

        try:
            core_kw, related = await asyncio.wait_for(
                self._fetch_all(date_from, date_to, searchword, result_count),
                timeout=self.metadata.timeout_sec,
            )
        except asyncio.TimeoutError:
            msg = f"API 호출 타임아웃 ({self.metadata.timeout_sec}초 초과)"
            logger.warning(f"[keyword_analyzer] {msg}")
            return LookupResult(
                success=False,
                query=query,
                provider=provider,
                error=msg,
                empty_reason="provider_error",
                evidence=EvidenceEnvelope(status="error", errors=[msg]),
            )
        except Exception as exc:
            logger.error(f"[keyword_analyzer] API 호출 오류: {exc}", exc_info=True)
            return LookupResult(
                success=False,
                query=query,
                provider=provider,
                error=str(exc),
                empty_reason="provider_error",
                evidence=EvidenceEnvelope(status="error", errors=[str(exc)]),
            )

        all_results: List[Dict[str, Any]] = []
        evidence_items: List[EvidenceItem] = []
        errors: List[str] = []

        if core_kw is not None:
            for item in core_kw:
                item["_source_api"] = "core_keyword"
                all_results.append(item)
                evidence_items.append(
                    EvidenceItem(
                        source_type="api",
                        title=item.get("label", ""),
                        excerpt=f"핵심키워드: {item.get('label', '')}, "
                        f"점수={item.get('value', 0)}",
                        provider_meta={"provider": provider, "api": "core_keyword"},
                    )
                )
        else:
            errors.append("핵심키워드 API 실패")

        if related is not None:
            for item in related:
                item["_source_api"] = "related_word"
                all_results.append(item)
                evidence_items.append(
                    EvidenceItem(
                        source_type="api",
                        title=item.get("label", ""),
                        excerpt=f"연관어: {item.get('label', '')}, " f"점수={item.get('value', 0)}",
                        provider_meta={"provider": provider, "api": "related_word"},
                    )
                )
        else:
            if searchword:
                errors.append("연관어 API 실패")

        if not all_results:
            status = "error" if errors else "empty"
            return LookupResult(
                success=not errors,
                query=query,
                provider=provider,
                empty_reason="no_match" if not errors else "provider_error",
                error="; ".join(errors) if errors else None,
                evidence=EvidenceEnvelope(items=[], status=status, errors=errors),
            )

        context_text = self._build_context_text(core_kw, related)
        status = "ok" if not errors else "partial"

        return LookupResult(
            success=True,
            query=query,
            results=all_results,
            context_text=context_text,
            provider=provider,
            evidence=EvidenceEnvelope(
                items=evidence_items,
                summary_text=context_text,
                status=status,
                errors=errors,
            ),
        )

    async def _fetch_all(
        self,
        date_from: str,
        date_to: str,
        searchword: str,
        result_count: int,
    ) -> tuple:
        """핵심키워드 + 연관어를 병렬 호출한다."""
        tasks = []

        # 핵심키워드는 date_from/date_to가 있으면 항상 호출
        if date_from and date_to:
            tasks.append(
                self._safe_call(
                    self._action.get_core_keywords,
                    date_from=date_from,
                    date_to=date_to,
                    result_count=result_count,
                )
            )
        else:
            tasks.append(self._noop())

        # 연관어는 searchword가 있을 때만 호출
        if date_from and date_to and searchword:
            tasks.append(
                self._safe_call(
                    self._action.get_related_words,
                    date_from=date_from,
                    date_to=date_to,
                    searchword=searchword,
                    result_count=result_count,
                )
            )
        else:
            tasks.append(self._noop())

        return tuple(await asyncio.gather(*tasks))

    @staticmethod
    async def _safe_call(fn, **kwargs) -> Optional[List[Dict[str, Any]]]:
        """개별 API 호출을 안전하게 래핑한다."""
        try:
            return await fn(**kwargs)
        except Exception as exc:
            logger.warning(f"[keyword_analyzer] 개별 API 실패: {exc}")
            return None

    @staticmethod
    async def _noop() -> None:
        """빈 결과를 반환하는 no-op 코루틴."""
        return None

    @staticmethod
    def _build_context_text(
        core_kw: Optional[List],
        related: Optional[List],
    ) -> str:
        """조합 결과에서 자연어 요약을 생성한다."""
        parts: List[str] = []

        if core_kw:
            items = []
            for k in core_kw[:5]:
                label = k.get("label", "")
                value = k.get("value", 0)
                try:
                    value_f = float(value)
                    items.append(f"{label}({value_f:,.0f}건)")
                except (ValueError, TypeError):
                    items.append(f"{label}({value})")
            if items:
                parts.append(f"핵심 키워드: {', '.join(items)}")

        if related:
            items = []
            for r in related[:5]:
                label = r.get("label", "")
                value = r.get("value", 0)
                try:
                    value_f = float(value)
                    items.append(f"{label}({value_f:,.1f}점)")
                except (ValueError, TypeError):
                    items.append(f"{label}({value})")
            if items:
                parts.append(f"연관어: {', '.join(items)}")

        return ", ".join(parts) if parts else ""
