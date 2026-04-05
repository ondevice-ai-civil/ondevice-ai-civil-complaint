"""unified api_lookup capability — MinwonAnalysisAction 래핑."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional

from loguru import logger

from .base import CapabilityBase, CapabilityMetadata, EvidenceEnvelope, EvidenceItem, LookupResult

try:
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:
    httpx = None  # type: ignore
    _HTTPX_AVAILABLE = False


# ---------------------------------------------------------------------------
# 파라미터 validator
# ---------------------------------------------------------------------------


@dataclass
class ApiLookupParams:
    """api_lookup 호출 파라미터 — context에서 추출·정규화·검증."""

    query: str
    ret_count: int = 5
    min_score: int = 2

    @classmethod
    def from_context(cls, query: str, context: Dict[str, Any]) -> "ApiLookupParams":
        """context에서 파라미터를 추출하고 alias를 정규화한다."""
        ret_count = int(
            context.get("api_lookup_count") or context.get("ret_count") or context.get("count") or 5
        )
        min_score = int(
            context.get("api_lookup_min_score")
            or context.get("min_score")
            or context.get("score_threshold")
            or 2
        )
        return cls(
            query=query.strip(),
            ret_count=max(1, min(20, ret_count)),
            min_score=max(0, min(10, min_score)),
        )

    def validate(self) -> Optional[str]:
        """검증 실패 시 오류 메시지, 통과 시 None."""
        if not self.query:
            return "query가 비어있습니다"
        if len(self.query) > 500:
            return f"query가 너무 깁니다 ({len(self.query)}자, 최대 500자)"
        return None


# ---------------------------------------------------------------------------
# Capability
# ---------------------------------------------------------------------------


class ApiLookupCapability(CapabilityBase):
    """공공데이터포털 민원분석정보조회 API를 LangGraph capability로 래핑.

    Parameters
    ----------
    action : Optional[MinwonAnalysisAction]
        래핑할 기존 Action 인스턴스. None이면 빈 결과를 반환한다
        (SKIP_MODEL_LOAD 등 경량 환경 지원).
    """

    def __init__(self, action: Optional[Any] = None) -> None:
        self._action = action
        self._lock = asyncio.Lock()

    @property
    def metadata(self) -> CapabilityMetadata:
        return CapabilityMetadata(
            name="api_lookup",
            description="공공데이터포털 민원분석정보조회 API를 호출하여 유사 민원 사례를 검색합니다.",
            approval_summary="외부 API(data.go.kr)에서 유사 민원 사례를 조회합니다.",
            provider="data.go.kr",
            timeout_sec=10.0,
        )

    async def execute(
        self,
        query: str,
        context: Dict[str, Any],
        session: Any,
    ) -> LookupResult:
        """MinwonAnalysisAction.fetch_similar_cases를 래핑하여 LookupResult로 반환."""
        provider = self.metadata.provider

        # 파라미터 추출 및 검증 (action 유무와 무관하게 항상 수행)
        params = ApiLookupParams.from_context(query, context)
        validation_error = params.validate()
        if validation_error:
            return LookupResult(
                success=False,
                query=query,
                provider=provider,
                error=validation_error,
                empty_reason="validation_error",
                evidence=EvidenceEnvelope(
                    status="error",
                    errors=[validation_error],
                ),
            )

        # action이 없으면 빈 결과 (경량 환경)
        if self._action is None:
            logger.debug("[api_lookup] action이 None — 빈 결과 반환")
            return LookupResult(
                success=True,
                query=params.query,
                provider=provider,
                empty_reason="no_match",
                evidence=EvidenceEnvelope(status="empty"),
            )

        # action에 파라미터 반영 (Lock으로 동시 접근 직렬화)
        # shared state(_ret_count, _min_score) 변경과 API 호출을 원자적으로 수행
        try:
            async with self._lock:
                self._action._ret_count = params.ret_count
                self._action._min_score = params.min_score
                payload = await asyncio.wait_for(
                    self._action.fetch_similar_cases(params.query, context),
                    timeout=self.metadata.timeout_sec,
                )
        except asyncio.TimeoutError:
            timeout_msg = f"API 호출 타임아웃 ({self.metadata.timeout_sec}초 초과)"
            logger.warning(f"[api_lookup] 타임아웃 ({self.metadata.timeout_sec}s 초과)")
            return LookupResult(
                success=False,
                query=params.query,
                provider=provider,
                error=timeout_msg,
                empty_reason="provider_error",
                evidence=EvidenceEnvelope(
                    status="error",
                    errors=[timeout_msg],
                ),
            )
        except Exception as exc:
            if _HTTPX_AVAILABLE and isinstance(exc, httpx.HTTPError):
                logger.warning(f"[api_lookup] httpx 오류: {exc}")
            else:
                logger.error(f"[api_lookup] API 호출 오류: {exc}", exc_info=True)
            return LookupResult(
                success=False,
                query=params.query,
                provider=provider,
                error=str(exc),
                empty_reason="provider_error",
                evidence=EvidenceEnvelope(
                    status="error",
                    errors=[str(exc)],
                ),
            )

        # 결과 변환
        results = payload.get("results")
        if results is None:
            error_msg = "민원 분석 API 호출에 실패했습니다."
            return LookupResult(
                success=False,
                query=payload.get("query", params.query),
                provider=provider,
                error=error_msg,
                empty_reason="provider_error",
                evidence=EvidenceEnvelope(
                    status="error",
                    errors=[error_msg],
                ),
            )

        # citations를 dict 목록으로 정규화
        raw_citations = payload.get("citations", [])
        citations: list[Dict[str, Any]] = []
        for c in raw_citations:
            if isinstance(c, dict):
                citations.append(c)
            elif hasattr(c, "__dict__"):
                citations.append({k: v for k, v in c.__dict__.items() if not k.startswith("_")})

        if not results:
            return LookupResult(
                success=True,
                query=payload.get("query", params.query),
                provider=provider,
                empty_reason="no_match",
                evidence=EvidenceEnvelope(status="empty"),
            )

        # EvidenceItem으로 정규화
        evidence_items = []
        for item in results:
            if not isinstance(item, dict):
                continue
            title = item.get("title") or item.get("qnaTitle") or item.get("question", "")
            excerpt = item.get("content") or item.get("qnaContent") or item.get("qnaAnswer", "")
            link = item.get("url") or item.get("detailUrl", "")
            evidence_items.append(
                EvidenceItem(
                    source_type="api",
                    title=str(title),
                    excerpt=str(excerpt)[:500],
                    link_or_path=str(link),
                    score=float(item.get("score", 0)),
                    provider_meta={"provider": provider},
                )
            )
        # citations도 EvidenceItem으로 변환 (중복 제거를 위해 link_or_path 기반 dedup)
        seen_links: set[str] = {item.link_or_path for item in evidence_items}
        for c in citations:
            link = c.get("url") or c.get("detailUrl", "")
            if link in seen_links:
                continue
            seen_links.add(str(link))
            title = c.get("title") or c.get("qnaTitle") or c.get("question", "")
            excerpt = c.get("content") or c.get("qnaContent") or c.get("qnaAnswer", "")
            evidence_items.append(
                EvidenceItem(
                    source_type="api",
                    title=str(title),
                    excerpt=str(excerpt)[:500],
                    link_or_path=str(link),
                    score=float(c.get("score", 0)),
                    provider_meta={"provider": provider},
                )
            )

        envelope = EvidenceEnvelope(
            items=evidence_items,
            status="ok" if evidence_items else "empty",
        )

        return LookupResult(
            success=True,
            query=payload.get("query", params.query),
            results=results,
            context_text=payload.get("context_text", ""),
            citations=citations,
            provider=provider,
            evidence=envelope,
        )
