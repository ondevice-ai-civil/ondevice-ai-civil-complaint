"""issue_detector capability — 급증키워드+오늘이슈+최다키워드 조합.

Issue #486: 민원 이슈 탐지 도구.
3개 API(급증키워드, 오늘이슈, 최다키워드)를 조합하여
현재 주요 이슈를 탐지하고 자연어 요약을 생성한다.
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


class IssueDetectorCapability(CapabilityBase):
    """민원 이슈 탐지 capability.

    급증키워드, 오늘 이슈 토픽, 최다 키워드를 조합하여
    현재 주요 민원 이슈를 파악한다.

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
            name="issue_detector",
            description=(
                "급증키워드, 오늘이슈, 최다키워드를 조합하여 " "현재 주요 민원 이슈를 탐지합니다."
            ),
            approval_summary="공공데이터포털에서 민원 이슈 현황을 조회합니다.",
            provider="data.go.kr",
            timeout_sec=get_timeout("issue_detector"),
        )

    async def execute(
        self,
        query: str,
        context: Dict[str, Any],
        session: Any,
    ) -> LookupResult:
        """3개 API를 병렬 호출하고 결과를 조합한다."""
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
            logger.debug("[issue_detector] action이 None - 빈 결과 반환")
            return LookupResult(
                success=True,
                query=query,
                provider=provider,
                empty_reason="no_match",
                evidence=EvidenceEnvelope(status="empty"),
            )

        # 날짜 파라미터 추출 및 검증
        analysis_time = context.get("analysis_time", "")
        search_date = context.get("search_date", "")
        max_result = int(context.get("max_result", 5))

        if not analysis_time and not search_date:
            err = "analysis_time 또는 search_date 파라미터가 필요합니다"
            return LookupResult(
                success=False,
                query=query,
                provider=provider,
                error=err,
                empty_reason="validation_error",
                evidence=EvidenceEnvelope(status="error", errors=[err]),
            )

        try:
            rising, topics, top_kw = await asyncio.wait_for(
                self._fetch_all(analysis_time, search_date, max_result),
                timeout=self.metadata.timeout_sec,
            )
        except asyncio.TimeoutError:
            msg = f"API 호출 타임아웃 ({self.metadata.timeout_sec}초 초과)"
            logger.warning(f"[issue_detector] {msg}")
            return LookupResult(
                success=False,
                query=query,
                provider=provider,
                error=msg,
                empty_reason="provider_error",
                evidence=EvidenceEnvelope(status="error", errors=[msg]),
            )
        except Exception as exc:
            logger.error(f"[issue_detector] API 호출 오류: {exc}", exc_info=True)
            return LookupResult(
                success=False,
                query=query,
                provider=provider,
                error=str(exc),
                empty_reason="provider_error",
                evidence=EvidenceEnvelope(status="error", errors=[str(exc)]),
            )

        # 결과 조합
        all_results: List[Dict[str, Any]] = []
        evidence_items: List[EvidenceItem] = []
        errors: List[str] = []

        if rising is not None:
            for item in rising:
                item["_source_api"] = "rising_keyword"
                all_results.append(item)
                evidence_items.append(
                    EvidenceItem(
                        source_type="api",
                        title=item.get("keyword", ""),
                        excerpt=f"급증키워드: {item.get('keyword', '')}, "
                        f"빈도={item.get('df', 0)}, 전일대비={item.get('prevRatio', '')}%",
                        provider_meta={"provider": provider, "api": "rising_keyword"},
                    )
                )
        else:
            errors.append("급증키워드 API 실패")

        if topics is not None:
            for item in topics:
                item["_source_api"] = "today_topic"
                all_results.append(item)
                evidence_items.append(
                    EvidenceItem(
                        source_type="api",
                        title=item.get("topic", ""),
                        excerpt=f"오늘이슈: {item.get('topic', '')}, "
                        f"건수={item.get('count', 0)}",
                        provider_meta={"provider": provider, "api": "today_topic"},
                    )
                )
        else:
            errors.append("오늘이슈 API 실패")

        if top_kw is not None:
            for item in top_kw:
                item["_source_api"] = "top_keyword"
                all_results.append(item)
                evidence_items.append(
                    EvidenceItem(
                        source_type="api",
                        title=item.get("term", ""),
                        excerpt=f"최다키워드: {item.get('term', '')}, " f"빈도={item.get('df', 0)}",
                        provider_meta={"provider": provider, "api": "top_keyword"},
                    )
                )
        else:
            errors.append("최다키워드 API 실패")

        if not all_results:
            status = "error" if len(errors) == 3 else "empty"
            return LookupResult(
                success=len(errors) < 3,
                query=query,
                provider=provider,
                empty_reason="no_match" if len(errors) < 3 else "provider_error",
                error="; ".join(errors) if errors else None,
                evidence=EvidenceEnvelope(items=[], status=status, errors=errors),
            )

        context_text = self._build_context_text(rising, topics, top_kw)
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
        analysis_time: str,
        search_date: str,
        max_result: int,
    ) -> tuple:
        """3개 API를 병렬 호출한다."""
        tasks = [
            (
                self._safe_call(
                    self._action.get_rising_keywords,
                    analysis_time=analysis_time,
                    max_result=max_result,
                )
                if analysis_time
                else self._noop()
            ),
            (
                self._safe_call(
                    self._action.get_today_topics,
                    search_date=search_date,
                    top_n=max_result,
                )
                if search_date
                else self._noop()
            ),
            (
                self._safe_call(
                    self._action.get_top_keywords_by_period,
                    analysis_time=analysis_time or search_date,
                    max_result=max_result,
                )
                if (analysis_time or search_date)
                else self._noop()
            ),
        ]
        return await asyncio.gather(*tasks)

    @staticmethod
    async def _noop() -> None:
        """빈 결과를 반환하는 no-op 코루틴."""
        return None

    @staticmethod
    async def _safe_call(fn, **kwargs) -> Optional[List[Dict[str, Any]]]:
        """개별 API 호출을 안전하게 래핑한다."""
        try:
            return await fn(**kwargs)
        except Exception as exc:
            logger.warning(f"[issue_detector] 개별 API 실패: {exc}")
            return None

    @staticmethod
    def _build_context_text(
        rising: Optional[List],
        topics: Optional[List],
        top_kw: Optional[List],
    ) -> str:
        """조합 결과에서 자연어 요약을 생성한다."""
        parts: List[str] = []

        if rising:
            items = []
            for r in rising[:3]:
                kw = r.get("keyword", "")
                ratio = r.get("prevRatio", "")
                items.append(f"{kw}(+{ratio}%)" if ratio else kw)
            if items:
                parts.append(f"급증 이슈: {', '.join(items)}")

        if topics:
            items = []
            for t in topics[:3]:
                topic = t.get("topic", "")
                count = t.get("count", 0)
                items.append(f"{topic}({count:,}건)")
            if items:
                parts.append(f"오늘 핵심: {', '.join(items)}")

        if top_kw:
            items = []
            for k in top_kw[:3]:
                term = k.get("term", "")
                df = k.get("df", 0)
                items.append(f"{term}({df}건)")
            if items:
                parts.append(f"최다: {', '.join(items)}")

        return "; ".join(parts) if parts else ""
