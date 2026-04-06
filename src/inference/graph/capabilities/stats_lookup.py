"""stats_lookup capability — 맞춤형통계+트렌드+건수+기관순위+지역순위 조합.

Issue #487: 민원 통계 조회 도구.
5개 API를 조합하여 민원 통계 현황을 제공한다.
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


class StatsLookupCapability(CapabilityBase):
    """민원 통계 조회 capability.

    키워드가 있으면 건수+트렌드, 없으면 통계+기관순위+지역순위를 조합한다.

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
            name="stats_lookup",
            description=(
                "민원 통계, 트렌드, 건수, 기관/지역 순위를 조합하여 " "민원 현황 통계를 제공합니다."
            ),
            approval_summary="공공데이터포털에서 민원 통계 현황을 조회합니다.",
            provider="data.go.kr",
            timeout_sec=get_timeout("stats_lookup"),
        )

    async def execute(
        self,
        query: str,
        context: Dict[str, Any],
        session: Any,
    ) -> LookupResult:
        """조건에 따라 API를 조합 호출하고 결과를 반환한다."""
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
            logger.debug("[stats_lookup] action이 None - 빈 결과 반환")
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
        period = context.get("period", "DAILY")
        top_n = int(context.get("top_n", 5))

        try:
            results_map = await asyncio.wait_for(
                self._fetch_all(date_from, date_to, searchword, period, top_n),
                timeout=self.metadata.timeout_sec,
            )
        except asyncio.TimeoutError:
            msg = f"API 호출 타임아웃 ({self.metadata.timeout_sec}초 초과)"
            logger.warning(f"[stats_lookup] {msg}")
            return LookupResult(
                success=False,
                query=query,
                provider=provider,
                error=msg,
                empty_reason="provider_error",
                evidence=EvidenceEnvelope(status="error", errors=[msg]),
            )
        except Exception as exc:
            logger.error(f"[stats_lookup] API 호출 오류: {exc}", exc_info=True)
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

        for api_name, items in results_map.items():
            if items is None:
                errors.append(f"{api_name} API 실패")
                continue
            for item in items:
                item["_source_api"] = api_name
                all_results.append(item)
                evidence_items.append(
                    EvidenceItem(
                        source_type="api",
                        title=item.get("label", item.get("term", "")),
                        excerpt=self._format_item(api_name, item),
                        provider_meta={"provider": provider, "api": api_name},
                    )
                )

        if not all_results:
            status = "error" if len(errors) == len(results_map) else "empty"
            return LookupResult(
                success=len(errors) < len(results_map),
                query=query,
                provider=provider,
                empty_reason="no_match" if len(errors) < len(results_map) else "provider_error",
                error="; ".join(errors) if errors else None,
                evidence=EvidenceEnvelope(items=[], status=status, errors=errors),
            )

        context_text = self._build_context_text(results_map, date_from, date_to)
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
        period: str,
        top_n: int,
    ) -> Dict[str, Optional[List]]:
        """조건에 따라 적절한 API를 병렬 호출한다."""
        tasks: Dict[str, Any] = {}

        if searchword:
            # 키워드 기반: 건수 + 트렌드
            tasks["doc_count"] = self._safe_call(
                self._action.get_doc_count,
                date_from=date_from,
                date_to=date_to,
                searchword=searchword,
            )
            if date_from and date_to:
                tasks["trend"] = self._safe_call(
                    self._action.get_trend,
                    date_from=date_from + "00",
                    date_to=date_to + "23",
                    period=period,
                )
        else:
            # 일반 통계: 통계 + 기관순위 + 지역순위
            if date_from and date_to:
                tasks["statistics"] = self._safe_call(
                    self._action.get_statistics,
                    date_from=date_from,
                    date_to=date_to,
                    period=period,
                )
                tasks["org_ranking"] = self._safe_call(
                    self._action.get_org_ranking,
                    date_from=date_from,
                    date_to=date_to,
                    top_n=top_n,
                )
                tasks["region_ranking"] = self._safe_call(
                    self._action.get_region_ranking,
                    date_from=date_from,
                    date_to=date_to,
                    top_n=top_n,
                )

        if not tasks:
            return {}

        keys = list(tasks.keys())
        values = await asyncio.gather(*tasks.values())
        return dict(zip(keys, values))

    @staticmethod
    async def _safe_call(fn, **kwargs) -> Optional[List[Dict[str, Any]]]:
        """개별 API 호출을 안전하게 래핑한다."""
        try:
            return await fn(**kwargs)
        except Exception as exc:
            logger.warning(f"[stats_lookup] 개별 API 실패: {exc}")
            return None

    @staticmethod
    def _format_item(api_name: str, item: Dict[str, Any]) -> str:
        """개별 아이템의 요약 문자열을 생성한다."""
        label = item.get("label", "")
        hits = item.get("hits", "")
        if api_name == "doc_count":
            pttn = item.get("pttn", 0)
            dfpt = item.get("dfpt", 0)
            saeol = item.get("saeol", 0)
            return f"국민신문고={pttn}, 민원24={dfpt}, 새올={saeol}"
        if api_name == "trend":
            ratio = item.get("prebRatio", "")
            return f"{label}: {hits}건, 전일대비 {ratio}%"
        return f"{label}: {hits}건"

    @staticmethod
    def _build_context_text(
        results_map: Dict[str, Optional[List]],
        date_from: str,
        date_to: str,
    ) -> str:
        """조합 결과에서 자연어 요약을 생성한다."""
        parts: List[str] = []
        period_str = ""
        if date_from and date_to:
            period_str = (
                f"{date_from[:4]}/{date_from[4:6]}/{date_from[6:8]}~{date_to[4:6]}/{date_to[6:8]}"
            )

        doc_count = results_map.get("doc_count")
        if doc_count and len(doc_count) > 0:
            item = doc_count[0]
            try:
                pttn = int(item.get("pttn") or 0)
                dfpt = int(item.get("dfpt") or 0)
                saeol = int(item.get("saeol") or 0)
            except (ValueError, TypeError):
                pttn, dfpt, saeol = 0, 0, 0
            total = pttn + dfpt + saeol
            parts.append(f"{period_str} 총 {total:,}건" if period_str else f"총 {total:,}건")

        stats = results_map.get("statistics")
        if stats:
            total = sum(int(s.get("hits", 0)) for s in stats)
            parts.append(f"{period_str} 총 {total:,}건" if period_str else f"총 {total:,}건")

        trend = results_map.get("trend")
        if trend and len(trend) > 0:
            last = trend[-1]
            ratio = last.get("prebRatio", "")
            if ratio:
                parts.append(f"전일대비 {'+' if not ratio.startswith('-') else ''}{ratio}%")

        region = results_map.get("region_ranking")
        if region and len(region) > 0:
            top = region[0]
            parts.append(f"{top.get('label', '')} 최다({int(top.get('hits', 0)):,}건)")

        org = results_map.get("org_ranking")
        if org and len(org) > 0:
            top = org[0]
            parts.append(f"기관 최다: {top.get('label', '')}({int(top.get('hits', 0)):,}건)")

        return ", ".join(parts) if parts else ""
