"""demographics_lookup capability — 성별+연령+인구대비 조합.

Issue #489: 민원 인구통계 분석 도구.
3개 API(성별통계, 연령통계, 인구대비비율)를 조합하여
민원 인구통계 분석 결과를 제공한다.
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


class DemographicsLookupCapability(CapabilityBase):
    """민원 인구통계 분석 capability.

    성별, 연령, 인구대비 비율을 조합하여 인구통계 분석 결과를 제공한다.

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
            name="demographics_lookup",
            description=(
                "성별, 연령, 인구대비 비율을 조합하여 " "민원 인구통계 분석 결과를 제공합니다."
            ),
            approval_summary="공공데이터포털에서 민원 인구통계를 분석합니다.",
            provider="data.go.kr",
            timeout_sec=get_timeout("demographics_lookup"),
        )

    async def execute(
        self,
        query: str,
        context: Dict[str, Any],
        session: Any,
    ) -> LookupResult:
        """성별+연령+인구대비 API를 병렬 호출하고 결과를 조합한다."""
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
            logger.debug("[demographics_lookup] action이 None - 빈 결과 반환")
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
        top_n = int(context.get("top_n", 5))

        if not searchword:
            return LookupResult(
                success=False,
                query=query,
                provider=provider,
                error="인구통계 분석에는 searchword가 필요합니다",
                empty_reason="validation_error",
                evidence=EvidenceEnvelope(
                    status="error",
                    errors=["인구통계 분석에는 searchword가 필요합니다"],
                ),
            )

        try:
            gender, age, population = await asyncio.wait_for(
                self._fetch_all(date_from, date_to, searchword, top_n),
                timeout=self.metadata.timeout_sec,
            )
        except asyncio.TimeoutError:
            msg = f"API 호출 타임아웃 ({self.metadata.timeout_sec}초 초과)"
            logger.warning(f"[demographics_lookup] {msg}")
            return LookupResult(
                success=False,
                query=query,
                provider=provider,
                error=msg,
                empty_reason="provider_error",
                evidence=EvidenceEnvelope(status="error", errors=[msg]),
            )
        except Exception as exc:
            logger.error(f"[demographics_lookup] API 호출 오류: {exc}", exc_info=True)
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

        if gender is not None:
            for item in gender:
                item["_source_api"] = "gender"
                all_results.append(item)
                evidence_items.append(
                    EvidenceItem(
                        source_type="api",
                        title=item.get("label", ""),
                        excerpt=f"성별: {item.get('label', '')}, " f"건수={item.get('hits', 0)}",
                        provider_meta={"provider": provider, "api": "gender"},
                    )
                )
        else:
            errors.append("성별통계 API 실패")

        if age is not None:
            for item in age:
                item["_source_api"] = "age"
                all_results.append(item)
                evidence_items.append(
                    EvidenceItem(
                        source_type="api",
                        title=f"{item.get('label', '')}대",
                        excerpt=f"연령: {item.get('label', '')}대, " f"건수={item.get('hits', 0)}",
                        provider_meta={"provider": provider, "api": "age"},
                    )
                )
        else:
            errors.append("연령통계 API 실패")

        if population is not None:
            for item in population:
                item["_source_api"] = "population"
                all_results.append(item)
                evidence_items.append(
                    EvidenceItem(
                        source_type="api",
                        title=item.get("label", ""),
                        excerpt=f"인구대비: {item.get('label', '')}, "
                        f"비율={item.get('ratio', '')}",
                        provider_meta={"provider": provider, "api": "population"},
                    )
                )
        else:
            errors.append("인구대비 API 실패")

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

        context_text = self._build_context_text(gender, age, population)
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
        top_n: int,
    ) -> tuple:
        """3개 API를 병렬 호출한다."""
        tasks = [
            self._safe_call(
                self._action.get_gender_stats,
                date_from=date_from,
                date_to=date_to,
                searchword=searchword,
            ),
            self._safe_call(
                self._action.get_age_stats,
                date_from=date_from,
                date_to=date_to,
                searchword=searchword,
            ),
            self._safe_call(
                self._action.get_population_ratio,
                date_from=date_from,
                date_to=date_to,
                top_n=top_n,
            ),
        ]
        return tuple(await asyncio.gather(*tasks))

    @staticmethod
    async def _safe_call(fn, **kwargs) -> Optional[List[Dict[str, Any]]]:
        """개별 API 호출을 안전하게 래핑한다."""
        try:
            return await fn(**kwargs)
        except Exception as exc:
            logger.warning(f"[demographics_lookup] 개별 API 실패: {exc}")
            return None

    @staticmethod
    def _build_context_text(
        gender: Optional[List],
        age: Optional[List],
        population: Optional[List],
    ) -> str:
        """조합 결과에서 자연어 요약을 생성한다."""
        parts: List[str] = []

        if gender:
            total = sum(int(g.get("hits", 0)) for g in gender)
            if total > 0:
                items = []
                for g in gender:
                    label = g.get("label", "")
                    hits = int(g.get("hits", 0))
                    pct = (hits / total * 100) if total else 0
                    items.append(f"{label} {pct:.1f}%")
                parts.append(", ".join(items))

        if age:
            # 가장 높은 건수의 연령대
            sorted_age = sorted(age, key=lambda x: int(x.get("hits", 0)), reverse=True)
            if sorted_age:
                top = sorted_age[0]
                total = sum(int(a.get("hits", 0)) for a in age)
                hits = int(top.get("hits", 0))
                pct = (hits / total * 100) if total else 0
                parts.append(f"{top.get('label', '')}대 최다({pct:.1f}%)")

        if population:
            if len(population) > 0:
                top = population[0]
                ratio = top.get("ratio", "")
                label = top.get("label", "")
                try:
                    ratio_pct = float(ratio) * 100 if ratio else 0
                    parts.append(f"{label} 인구대비 {ratio_pct:.2f}%")
                except (ValueError, TypeError):
                    parts.append(f"{label} 인구대비 {ratio}")

        return ", ".join(parts) if parts else ""
