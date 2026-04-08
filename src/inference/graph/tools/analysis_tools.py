"""LangGraph ToolNode용 분석 도구 팩토리.

기존 IssueDetectorCapability, StatsLookupCapability,
KeywordAnalyzerCapability, DemographicsLookupCapability에 위임하여
StructuredTool 인스턴스를 동적 생성한다.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pydantic 스키마 — LLM이 생성하는 JSON 인자
# ---------------------------------------------------------------------------


class IssueDetectorInput(BaseModel):
    """issue_detector 도구 입력 스키마."""

    query: str = Field(..., description="이슈 탐지 대상 키워드 또는 질의문")
    analysis_time: Optional[str] = Field(
        None,
        description="분석 시간대 (YYYYMMDDHH 형식, 10자리). 예: '2026040814'",
    )
    max_result: int = Field(10, description="반환할 최대 결과 수", ge=1)


class StatsLookupInput(BaseModel):
    """stats_lookup 도구 입력 스키마."""

    query: str = Field(..., description="통계 조회 대상 키워드")
    date_from: Optional[str] = Field(
        None, description="조회 시작일 (YYYYMMDD 형식). 예: '20260101'"
    )
    date_to: Optional[str] = Field(None, description="조회 종료일 (YYYYMMDD 형식). 예: '20260408'")
    period: Optional[str] = Field(
        None, description="집계 기간 단위 (DAILY, WEEKLY, MONTHLY, YEARLY)"
    )


class KeywordAnalyzerInput(BaseModel):
    """keyword_analyzer 도구 입력 스키마."""

    query: str = Field(..., description="키워드 분석 대상 질의문")
    date_from: Optional[str] = Field(
        None, description="분석 시작일 (YYYYMMDD 형식). 예: '20260101'"
    )
    date_to: Optional[str] = Field(None, description="분석 종료일 (YYYYMMDD 형식). 예: '20260408'")
    result_count: int = Field(20, description="반환할 키워드 수", ge=1)


class DemographicsLookupInput(BaseModel):
    """demographics_lookup 도구 입력 스키마."""

    query: str = Field(..., description="인구통계 분석 대상 질의문")
    date_from: Optional[str] = Field(
        None, description="분석 시작일 (YYYYMMDD 형식). 예: '20260101'"
    )
    date_to: Optional[str] = Field(None, description="분석 종료일 (YYYYMMDD 형식). 예: '20260408'")


# ---------------------------------------------------------------------------
# 팩토리
# ---------------------------------------------------------------------------


def build_analysis_tools(
    api_lookup_action: Optional[Any] = None,
) -> list:
    """분석 관련 StructuredTool 목록을 생성한다.

    Parameters
    ----------
    api_lookup_action : Optional[MinwonAnalysisAction]
        공공데이터포털 API Action 인스턴스. None이면 빈 결과 반환.

    Returns
    -------
    list[StructuredTool]
        [issue_detector_tool, stats_lookup_tool, keyword_analyzer_tool, demographics_lookup_tool]
    """
    from src.inference.graph.capabilities.issue_detector import IssueDetectorCapability
    from src.inference.graph.capabilities.stats_lookup import StatsLookupCapability
    from src.inference.graph.capabilities.keyword_analyzer import KeywordAnalyzerCapability
    from src.inference.graph.capabilities.demographics_lookup import DemographicsLookupCapability

    # -- issue_detector --
    _issue_cap = IssueDetectorCapability(action=api_lookup_action)

    async def _issue_detector(
        query: str,
        analysis_time: Optional[str] = None,
        max_result: int = 10,
    ) -> str:
        context: dict[str, Any] = {"max_result": max_result}
        if analysis_time is not None:
            context["analysis_time"] = analysis_time
            # search_date를 analysis_time에서 자동 추출 (YYYYMMDDHH -> YYYYMMDD)
            if len(analysis_time) >= 8:
                context["search_date"] = analysis_time[:8]
        result = await _issue_cap.execute(query=query, context=context, session=None)
        return json.dumps(result.to_dict(), ensure_ascii=False)

    issue_detector_tool = StructuredTool.from_function(
        coroutine=_issue_detector,
        name="issue_detector",
        description=(
            "민원 데이터에서 반복되는 이슈 패턴과 트렌드를 탐지합니다. "
            "민원 급증, 반복 불만, 신규 이슈를 파악할 때 사용하세요. "
            "반환값: 탐지된 이슈 목록 (이슈명, 건수, 심각도)"
        ),
        args_schema=IssueDetectorInput,
        metadata={"requires_approval": False},
    )

    # -- stats_lookup --
    _stats_cap = StatsLookupCapability(action=api_lookup_action)

    async def _stats_lookup(
        query: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        period: Optional[str] = None,
    ) -> str:
        context: dict[str, Any] = {}
        if date_from is not None:
            context["date_from"] = date_from
        if date_to is not None:
            context["date_to"] = date_to
        if period is not None:
            context["period"] = period
        result = await _stats_cap.execute(query=query, context=context, session=None)
        return json.dumps(result.to_dict(), ensure_ascii=False)

    stats_lookup_tool = StructuredTool.from_function(
        coroutine=_stats_lookup,
        name="stats_lookup",
        description=(
            "민원 접수 통계를 기간별/유형별로 조회합니다. "
            "민원 현황 파악, 추이 분석에 사용하세요. "
            "반환값: 통계 데이터 (기간, 접수건수, 유형별 분포)"
        ),
        args_schema=StatsLookupInput,
        metadata={"requires_approval": False},
    )

    # -- keyword_analyzer --
    _kw_cap = KeywordAnalyzerCapability(action=api_lookup_action)

    async def _keyword_analyzer(
        query: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        result_count: int = 20,
    ) -> str:
        context: dict[str, Any] = {"result_count": result_count}
        if date_from is not None:
            context["date_from"] = date_from
        if date_to is not None:
            context["date_to"] = date_to
        # searchword를 query에서 자동 설정 (연관어 분석용)
        context["searchword"] = query
        result = await _kw_cap.execute(query=query, context=context, session=None)
        return json.dumps(result.to_dict(), ensure_ascii=False)

    keyword_analyzer_tool = StructuredTool.from_function(
        coroutine=_keyword_analyzer,
        name="keyword_analyzer",
        description=(
            "민원 텍스트에서 핵심 키워드와 빈도를 분석합니다. "
            "민원 이슈의 핵심어를 파악할 때 사용하세요. "
            "반환값: 키워드 목록 (키워드, 빈도, 관련도)"
        ),
        args_schema=KeywordAnalyzerInput,
        metadata={"requires_approval": False},
    )

    # -- demographics_lookup --
    _demo_cap = DemographicsLookupCapability(action=api_lookup_action)

    async def _demographics_lookup(
        query: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> str:
        context: dict[str, Any] = {"searchword": query}
        if date_from is not None:
            context["date_from"] = date_from
        if date_to is not None:
            context["date_to"] = date_to
        result = await _demo_cap.execute(query=query, context=context, session=None)
        return json.dumps(result.to_dict(), ensure_ascii=False)

    demographics_lookup_tool = StructuredTool.from_function(
        coroutine=_demographics_lookup,
        name="demographics_lookup",
        description=(
            "민원인의 인구통계 정보(연령대, 지역 등)를 조회합니다. "
            "민원 대상 분석에 사용하세요. "
            "반환값: 인구통계 분포 데이터"
        ),
        args_schema=DemographicsLookupInput,
        metadata={"requires_approval": False},
    )

    return [issue_detector_tool, stats_lookup_tool, keyword_analyzer_tool, demographics_lookup_tool]
