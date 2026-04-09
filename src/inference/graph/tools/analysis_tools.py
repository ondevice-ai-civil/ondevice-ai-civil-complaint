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
    """issue_detector tool input schema."""

    query: str = Field(..., description="Keywords or query to detect complaint issues")
    analysis_time: Optional[str] = Field(
        None,
        description="Analysis timestamp (YYYYMMDDHH, 10 digits). Example: '2026040814'",
    )
    max_result: int = Field(10, description="Maximum number of results to return", ge=1)


class StatsLookupInput(BaseModel):
    """stats_lookup tool input schema."""

    query: str = Field(..., description="Keywords for statistics lookup")
    date_from: Optional[str] = Field(
        None, description="Start date (YYYYMMDD format). Example: '20260101'"
    )
    date_to: Optional[str] = Field(None, description="End date (YYYYMMDD format). Example: '20260408'")
    period: Optional[str] = Field(
        None, description="Aggregation period (DAILY, WEEKLY, MONTHLY, YEARLY)"
    )


class KeywordAnalyzerInput(BaseModel):
    """keyword_analyzer tool input schema."""

    query: str = Field(..., description="Query text for keyword frequency analysis")
    date_from: Optional[str] = Field(
        None, description="Start date (YYYYMMDD format). Example: '20260101'"
    )
    date_to: Optional[str] = Field(None, description="End date (YYYYMMDD format). Example: '20260408'")
    result_count: int = Field(20, description="Number of keywords to return", ge=1)


class DemographicsLookupInput(BaseModel):
    """demographics_lookup tool input schema."""

    query: str = Field(..., description="Query for demographic analysis of complaint filers")
    date_from: Optional[str] = Field(
        None, description="Start date (YYYYMMDD format). Example: '20260101'"
    )
    date_to: Optional[str] = Field(None, description="End date (YYYYMMDD format). Example: '20260408'")


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
    from src.inference.graph.capabilities.demographics_lookup import DemographicsLookupCapability
    from src.inference.graph.capabilities.issue_detector import IssueDetectorCapability
    from src.inference.graph.capabilities.keyword_analyzer import KeywordAnalyzerCapability
    from src.inference.graph.capabilities.stats_lookup import StatsLookupCapability

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
        try:
            result = await _issue_cap.execute(query=query, context=context, session=None)
            return json.dumps(result.to_dict(), ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e), "success": False}, ensure_ascii=False)

    issue_detector_tool = StructuredTool.from_function(
        coroutine=_issue_detector,
        name="issue_detector",
        description=(
            "Detect recurring issue patterns and trends in civil complaint data. "
            "USE THIS TOOL when the user asks about complaint surges, repeated complaints, "
            "emerging issues, or trend analysis. "
            "Returns: list of detected issues with name, count, and severity score."
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
        try:
            result = await _stats_cap.execute(query=query, context=context, session=None)
            return json.dumps(result.to_dict(), ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e), "success": False}, ensure_ascii=False)

    stats_lookup_tool = StructuredTool.from_function(
        coroutine=_stats_lookup,
        name="stats_lookup",
        description=(
            "Query civil complaint filing statistics by period and category. "
            "USE THIS TOOL when the user asks about complaint volume, filing counts, "
            "category distribution, or time-series trends. "
            "Returns: statistical data including period, filing count, and category breakdown."
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
        try:
            result = await _kw_cap.execute(query=query, context=context, session=None)
            return json.dumps(result.to_dict(), ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e), "success": False}, ensure_ascii=False)

    keyword_analyzer_tool = StructuredTool.from_function(
        coroutine=_keyword_analyzer,
        name="keyword_analyzer",
        description=(
            "Analyze top keywords and their frequency in civil complaint texts. "
            "USE THIS TOOL when the user asks about trending topics, frequently mentioned terms, "
            "or wants to understand what citizens are complaining about. "
            "Returns: ranked keyword list with frequency and relevance scores."
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
        try:
            result = await _demo_cap.execute(query=query, context=context, session=None)
            return json.dumps(result.to_dict(), ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e), "success": False}, ensure_ascii=False)

    demographics_lookup_tool = StructuredTool.from_function(
        coroutine=_demographics_lookup,
        name="demographics_lookup",
        description=(
            "Look up demographic distribution (age group, region, gender) of civil complaint filers. "
            "USE THIS TOOL when the user asks about who is filing complaints, "
            "regional patterns, or age-based analysis. "
            "Returns: demographic distribution data."
        ),
        args_schema=DemographicsLookupInput,
        metadata={"requires_approval": False},
    )

    return [issue_detector_tool, stats_lookup_tool, keyword_analyzer_tool, demographics_lookup_tool]
