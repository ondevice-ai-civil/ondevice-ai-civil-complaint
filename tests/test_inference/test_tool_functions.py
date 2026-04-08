"""도구 팩토리 함수 단위 테스트.

build_search_tools, build_analysis_tools, build_all_tools가
올바른 수의 StructuredTool을 생성하고, 각 도구가 호출 가능하며
JSON 문자열을 반환하는지 검증한다.
"""

from __future__ import annotations

import importlib
import json
import pathlib
import sys
import types
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# graph 패키지 import chain 보호
# langgraph/langchain_core가 미설치된 CI 환경에서
# graph 패키지를 stub으로 등록하여 tools 하위 모듈만 독립 테스트한다.
# ---------------------------------------------------------------------------


_graph_pkg_path = str(pathlib.Path(__file__).resolve().parents[2] / "src" / "inference" / "graph")

if "src.inference.graph" not in sys.modules:
    _graph_stub = types.ModuleType("src.inference.graph")
    _graph_stub.__path__ = [_graph_pkg_path]  # type: ignore[attr-defined]
    sys.modules["src.inference.graph"] = _graph_stub
else:
    # 이미 로드되었으면 그대로 사용
    pass

from src.inference.graph.tools import build_all_tools, get_tool_approval_map
from src.inference.graph.tools.analysis_tools import build_analysis_tools

# tools 패키지를 정상 로드
from src.inference.graph.tools.search_tools import build_search_tools

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_api_action() -> MagicMock:
    """MinwonAnalysisAction mock."""
    action = MagicMock()

    # api_lookup용
    action.fetch_similar_cases = AsyncMock(
        return_value={
            "query": "테스트 민원",
            "results": [
                {
                    "title": "유사 민원 사례",
                    "content": "답변 내용",
                    "url": "https://example.com",
                    "score": 5,
                }
            ],
            "context_text": "유사 사례 1건",
            "citations": [],
        }
    )

    # issue_detector용
    action.get_rising_keywords = AsyncMock(
        return_value=[{"keyword": "도로", "df": 100, "prevRatio": "50"}]
    )
    action.get_today_topics = AsyncMock(return_value=[{"topic": "교통", "count": 200}])
    action.get_top_keywords_by_period = AsyncMock(return_value=[{"term": "소음", "df": 80}])

    # stats_lookup용
    action.get_doc_count = AsyncMock(return_value=[{"pttn": 100, "dfpt": 50, "saeol": 30}])
    action.get_trend = AsyncMock(
        return_value=[{"label": "2026-04-08", "hits": 180, "prebRatio": "10"}]
    )
    action.get_statistics = AsyncMock(return_value=[{"label": "2026-04", "hits": 500}])
    action.get_org_ranking = AsyncMock(return_value=[{"label": "서울시청", "hits": 1000}])
    action.get_region_ranking = AsyncMock(return_value=[{"label": "서울", "hits": 2000}])

    # keyword_analyzer용
    action.get_core_keywords = AsyncMock(return_value=[{"label": "민원", "value": 500}])
    action.get_related_words = AsyncMock(return_value=[{"label": "불만", "value": 0.8}])

    # demographics_lookup용
    action.get_gender_stats = AsyncMock(
        return_value=[{"label": "남성", "hits": 600}, {"label": "여성", "hits": 400}]
    )
    action.get_age_stats = AsyncMock(
        return_value=[{"label": "30", "hits": 300}, {"label": "40", "hits": 250}]
    )
    action.get_population_ratio = AsyncMock(return_value=[{"label": "서울", "ratio": 0.0012}])

    return action


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildSearchTools:
    """build_search_tools 테스트."""

    def test_returns_one_tool(self) -> None:
        tools = build_search_tools(api_lookup_action=None)
        assert len(tools) == 1

    def test_tool_names(self) -> None:
        tools = build_search_tools()
        names = {t.name for t in tools}
        assert names == {"api_lookup"}

    def test_descriptions_not_empty(self) -> None:
        tools = build_search_tools()
        for tool in tools:
            assert tool.description, f"{tool.name} description이 비어있음"

    @pytest.mark.asyncio
    async def test_api_lookup_returns_json(
        self,
        mock_api_action: MagicMock,
    ) -> None:
        tools = build_search_tools(mock_api_action)
        api_tool = next(t for t in tools if t.name == "api_lookup")
        result = await api_tool.ainvoke({"query": "테스트 민원"})
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "success" in parsed


class TestBuildAnalysisTools:
    """build_analysis_tools 테스트."""

    def test_returns_four_tools(self) -> None:
        tools = build_analysis_tools(api_lookup_action=None)
        assert len(tools) == 4

    def test_tool_names(self) -> None:
        tools = build_analysis_tools()
        names = {t.name for t in tools}
        assert names == {
            "issue_detector",
            "stats_lookup",
            "keyword_analyzer",
            "demographics_lookup",
        }

    def test_descriptions_not_empty(self) -> None:
        tools = build_analysis_tools()
        for tool in tools:
            assert tool.description, f"{tool.name} description이 비어있음"

    @pytest.mark.asyncio
    async def test_issue_detector_returns_json(self, mock_api_action: MagicMock) -> None:
        tools = build_analysis_tools(mock_api_action)
        tool = next(t for t in tools if t.name == "issue_detector")
        result = await tool.ainvoke({"query": "교통 민원", "analysis_time": "2026040814"})
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "success" in parsed

    @pytest.mark.asyncio
    async def test_stats_lookup_returns_json(self, mock_api_action: MagicMock) -> None:
        tools = build_analysis_tools(mock_api_action)
        tool = next(t for t in tools if t.name == "stats_lookup")
        result = await tool.ainvoke({"query": "소음 민원"})
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "success" in parsed

    @pytest.mark.asyncio
    async def test_keyword_analyzer_returns_json(self, mock_api_action: MagicMock) -> None:
        tools = build_analysis_tools(mock_api_action)
        tool = next(t for t in tools if t.name == "keyword_analyzer")
        result = await tool.ainvoke(
            {
                "query": "민원 키워드",
                "date_from": "20260101",
                "date_to": "20260408",
            }
        )
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "success" in parsed

    @pytest.mark.asyncio
    async def test_demographics_returns_json(self, mock_api_action: MagicMock) -> None:
        tools = build_analysis_tools(mock_api_action)
        tool = next(t for t in tools if t.name == "demographics_lookup")
        result = await tool.ainvoke(
            {
                "query": "소음 민원",
                "date_from": "20260101",
                "date_to": "20260408",
            }
        )
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "success" in parsed


class TestBuildAllTools:
    """build_all_tools 테스트."""

    def test_collects_all_tools(self) -> None:
        tools = build_all_tools(
            api_lookup_action=None,
        )
        # 1 search + 4 analysis = 5 (draft_response_fn 없음)
        assert len(tools) == 5

    def test_all_names_unique(self) -> None:
        tools = build_all_tools()
        names = [t.name for t in tools]
        assert len(names) == len(set(names)), f"중복 도구명: {names}"


class TestToolApprovalMap:
    """get_tool_approval_map 테스트."""

    def test_returns_map(self) -> None:
        tools = build_all_tools()
        approval_map = get_tool_approval_map(tools)
        assert isinstance(approval_map, dict)
        assert len(approval_map) == len(tools)
        # 현재 모든 도구는 requires_approval=False
        for name, requires in approval_map.items():
            assert requires is False, f"{name}이 approval이 True"
