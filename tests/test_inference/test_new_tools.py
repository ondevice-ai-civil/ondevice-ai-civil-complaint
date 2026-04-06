"""신규 4개 capability(issue_detector, stats_lookup, keyword_analyzer, demographics_lookup) 테스트.

Issue #486-489: 민원분석 API 4개 신규 도구 구현.
실제 API 호출 없이 mock으로 검증한다.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.inference.graph.capabilities.base import CapabilityBase, LookupResult
from src.inference.graph.capabilities.demographics_lookup import (
    DemographicsLookupCapability,
)
from src.inference.graph.capabilities.issue_detector import IssueDetectorCapability
from src.inference.graph.capabilities.keyword_analyzer import KeywordAnalyzerCapability
from src.inference.graph.capabilities.registry import (
    MVP_CAPABILITY_IDS,
    build_mvp_registry,
)
from src.inference.graph.capabilities.stats_lookup import StatsLookupCapability

# ---------------------------------------------------------------------------
# 공용 fixture
# ---------------------------------------------------------------------------


class FakeAction:
    """MinwonAnalysisAction mock."""

    def __init__(self, responses: Optional[Dict[str, Any]] = None) -> None:
        self._responses = responses or {}

    async def get_rising_keywords(self, **kwargs) -> Optional[List]:
        return self._responses.get("rising_keywords")

    async def get_today_topics(self, **kwargs) -> Optional[List]:
        return self._responses.get("today_topics")

    async def get_top_keywords_by_period(self, **kwargs) -> Optional[List]:
        return self._responses.get("top_keywords_by_period")

    async def get_statistics(self, **kwargs) -> Optional[List]:
        return self._responses.get("statistics")

    async def get_trend(self, **kwargs) -> Optional[List]:
        return self._responses.get("trend")

    async def get_doc_count(self, **kwargs) -> Optional[List]:
        return self._responses.get("doc_count")

    async def get_org_ranking(self, **kwargs) -> Optional[List]:
        return self._responses.get("org_ranking")

    async def get_region_ranking(self, **kwargs) -> Optional[List]:
        return self._responses.get("region_ranking")

    async def get_core_keywords(self, **kwargs) -> Optional[List]:
        return self._responses.get("core_keywords")

    async def get_related_words(self, **kwargs) -> Optional[List]:
        return self._responses.get("related_words")

    async def get_gender_stats(self, **kwargs) -> Optional[List]:
        return self._responses.get("gender_stats")

    async def get_age_stats(self, **kwargs) -> Optional[List]:
        return self._responses.get("age_stats")

    async def get_population_ratio(self, **kwargs) -> Optional[List]:
        return self._responses.get("population_ratio")


# ---------------------------------------------------------------------------
# Registry 등록 확인
# ---------------------------------------------------------------------------


class TestRegistry:
    """신규 capability가 registry에 정상 등록되는지 확인."""

    def test_mvp_capability_ids_include_new_tools(self):
        """MVP_CAPABILITY_IDS에 4개 신규 도구가 포함된다."""
        assert "issue_detector" in MVP_CAPABILITY_IDS
        assert "stats_lookup" in MVP_CAPABILITY_IDS
        assert "keyword_analyzer" in MVP_CAPABILITY_IDS
        assert "demographics_lookup" in MVP_CAPABILITY_IDS

    def test_build_mvp_registry_includes_new_tools(self):
        """build_mvp_registry가 8개 capability를 모두 반환한다."""

        async def dummy_fn(query="", context=None, session=None):
            return {}

        registry = build_mvp_registry(
            rag_search_fn=dummy_fn,
            api_lookup_action=None,
            draft_civil_response_fn=dummy_fn,
            append_evidence_fn=dummy_fn,
        )
        assert "issue_detector" in registry
        assert "stats_lookup" in registry
        assert "keyword_analyzer" in registry
        assert "demographics_lookup" in registry
        assert len(registry) == 8

    def test_all_capabilities_are_capability_base(self):
        """모든 등록된 capability가 CapabilityBase 인스턴스이다."""

        async def dummy_fn(query="", context=None, session=None):
            return {}

        registry = build_mvp_registry(
            rag_search_fn=dummy_fn,
            api_lookup_action=None,
            draft_civil_response_fn=dummy_fn,
            append_evidence_fn=dummy_fn,
        )
        for name, cap in registry.items():
            assert isinstance(cap, CapabilityBase), f"{name}이 CapabilityBase가 아닙니다"


# ---------------------------------------------------------------------------
# IssueDetectorCapability
# ---------------------------------------------------------------------------


class TestIssueDetector:
    """issue_detector capability 테스트."""

    @pytest.mark.asyncio
    async def test_action_none_returns_empty(self):
        """action이 None이면 빈 결과를 반환한다."""
        cap = IssueDetectorCapability(action=None)
        result = await cap.execute("이슈 조회", {}, None)
        assert result.success is True
        assert result.empty_reason == "no_match"

    @pytest.mark.asyncio
    async def test_empty_query_returns_error(self):
        """빈 쿼리는 에러를 반환한다."""
        cap = IssueDetectorCapability(action=FakeAction())
        result = await cap.execute("", {}, None)
        assert result.success is False
        assert result.empty_reason == "validation_error"

    @pytest.mark.asyncio
    async def test_successful_response(self):
        """3개 API 모두 성공하면 context_text가 생성된다."""
        action = FakeAction(
            {
                "rising_keywords": [
                    {"keyword": "아파트", "df": 4, "prevRatio": "400.00"},
                ],
                "today_topics": [
                    {"topic": "차량 불법 정차", "count": 351},
                ],
                "top_keywords_by_period": [
                    {"term": "변전소_부지", "df": "97"},
                ],
            }
        )
        cap = IssueDetectorCapability(action=action)
        context = {
            "analysis_time": "2021050614",
            "search_date": "20210506",
        }
        result = await cap.execute("이슈 조회", context, None)
        assert result.success is True
        assert "아파트" in result.context_text
        assert "차량 불법 정차" in result.context_text
        assert "변전소_부지" in result.context_text
        assert result.evidence is not None
        assert result.evidence.status == "ok"

    @pytest.mark.asyncio
    async def test_partial_failure(self):
        """일부 API가 실패해도 결과를 반환한다."""
        action = FakeAction(
            {
                "rising_keywords": None,
                "today_topics": [{"topic": "테스트", "count": 100}],
                "top_keywords_by_period": None,
            }
        )
        cap = IssueDetectorCapability(action=action)
        context = {"analysis_time": "2021050614", "search_date": "20210506"}
        result = await cap.execute("이슈 조회", context, None)
        assert result.success is True
        assert result.evidence is not None
        assert result.evidence.status == "partial"

    def test_metadata(self):
        """metadata가 올바르게 설정된다."""
        cap = IssueDetectorCapability()
        meta = cap.metadata
        assert meta.name == "issue_detector"
        assert meta.provider == "data.go.kr"


# ---------------------------------------------------------------------------
# StatsLookupCapability
# ---------------------------------------------------------------------------


class TestStatsLookup:
    """stats_lookup capability 테스트."""

    @pytest.mark.asyncio
    async def test_action_none_returns_empty(self):
        cap = StatsLookupCapability(action=None)
        result = await cap.execute("통계 조회", {}, None)
        assert result.success is True
        assert result.empty_reason == "no_match"

    @pytest.mark.asyncio
    async def test_keyword_based_query(self):
        """searchword가 있으면 건수+트렌드를 호출한다."""
        action = FakeAction(
            {
                "doc_count": [{"pttn": "769", "dfpt": "6", "saeol": "59"}],
                "trend": [
                    {"hits": 31717, "label": "20191111", "prebRatio": "37.1"},
                ],
            }
        )
        cap = StatsLookupCapability(action=action)
        context = {
            "date_from": "20210501",
            "date_to": "20210506",
            "searchword": "코로나",
        }
        result = await cap.execute("코로나 통계", context, None)
        assert result.success is True
        assert "834" in result.context_text  # 769+6+59
        assert result.evidence is not None

    @pytest.mark.asyncio
    async def test_general_stats_query(self):
        """searchword 없으면 통계+순위를 호출한다."""
        action = FakeAction(
            {
                "statistics": [
                    {"hits": 36604, "label": "20210501"},
                    {"hits": 47252, "label": "20210503"},
                ],
                "org_ranking": [{"hits": 6138, "label": "경기도 고양시"}],
                "region_ranking": [{"hits": 92887, "label": "경기도"}],
            }
        )
        cap = StatsLookupCapability(action=action)
        context = {"date_from": "20210501", "date_to": "20210506"}
        result = await cap.execute("통계 조회", context, None)
        assert result.success is True
        assert "경기도" in result.context_text

    def test_metadata(self):
        cap = StatsLookupCapability()
        meta = cap.metadata
        assert meta.name == "stats_lookup"
        assert meta.provider == "data.go.kr"


# ---------------------------------------------------------------------------
# KeywordAnalyzerCapability
# ---------------------------------------------------------------------------


class TestKeywordAnalyzer:
    """keyword_analyzer capability 테스트."""

    @pytest.mark.asyncio
    async def test_action_none_returns_empty(self):
        cap = KeywordAnalyzerCapability(action=None)
        result = await cap.execute("키워드 분석", {}, None)
        assert result.success is True
        assert result.empty_reason == "no_match"

    @pytest.mark.asyncio
    async def test_core_keywords_and_related(self):
        """핵심키워드+연관어 결과가 올바르게 조합된다."""
        action = FakeAction(
            {
                "core_keywords": [{"label": "대구시", "value": "1300.0"}],
                "related_words": [{"label": "건강기능식품", "value": 202.31987}],
            }
        )
        cap = KeywordAnalyzerCapability(action=action)
        context = {
            "date_from": "20210501",
            "date_to": "20210506",
            "searchword": "비타민",
        }
        result = await cap.execute("키워드 분석", context, None)
        assert result.success is True
        assert "대구시" in result.context_text
        assert "건강기능식품" in result.context_text

    @pytest.mark.asyncio
    async def test_no_searchword_skips_related(self):
        """searchword가 없으면 연관어를 호출하지 않는다."""
        action = FakeAction(
            {
                "core_keywords": [{"label": "대구시", "value": "1300.0"}],
            }
        )
        cap = KeywordAnalyzerCapability(action=action)
        context = {"date_from": "20210501", "date_to": "20210506"}
        result = await cap.execute("키워드 분석", context, None)
        assert result.success is True
        assert "대구시" in result.context_text

    def test_metadata(self):
        cap = KeywordAnalyzerCapability()
        assert cap.metadata.name == "keyword_analyzer"


# ---------------------------------------------------------------------------
# DemographicsLookupCapability
# ---------------------------------------------------------------------------


class TestDemographicsLookup:
    """demographics_lookup capability 테스트."""

    @pytest.mark.asyncio
    async def test_action_none_returns_empty(self):
        cap = DemographicsLookupCapability(action=None)
        result = await cap.execute("인구통계", {}, None)
        assert result.success is True
        assert result.empty_reason == "no_match"

    @pytest.mark.asyncio
    async def test_requires_searchword(self):
        """searchword가 없으면 에러를 반환한다."""
        action = FakeAction()
        cap = DemographicsLookupCapability(action=action)
        result = await cap.execute(
            "인구통계", {"date_from": "20220220", "date_to": "20220310"}, None
        )
        assert result.success is False
        assert "searchword" in result.error

    @pytest.mark.asyncio
    async def test_successful_demographics(self):
        """3개 API 모두 성공하면 context_text가 올바르게 생성된다."""
        action = FakeAction(
            {
                "gender_stats": [
                    {"hits": 6235, "label": "남성"},
                    {"hits": 4029, "label": "여성"},
                ],
                "age_stats": [
                    {"hits": 2746, "label": "40"},
                    {"hits": 2465, "label": "30"},
                ],
                "population_ratio": [
                    {
                        "hits": 92887,
                        "label": "경기도",
                        "population": 6844493,
                        "ratio": "0.01357",
                    },
                ],
            }
        )
        cap = DemographicsLookupCapability(action=action)
        context = {
            "date_from": "20220220",
            "date_to": "20220310",
            "searchword": "코로나",
        }
        result = await cap.execute("코로나 인구통계", context, None)
        assert result.success is True
        assert "남성" in result.context_text
        assert "40대" in result.context_text
        assert "경기도" in result.context_text
        assert result.evidence is not None
        assert result.evidence.status == "ok"

    @pytest.mark.asyncio
    async def test_partial_failure(self):
        """일부 API 실패 시 partial 결과를 반환한다."""
        action = FakeAction(
            {
                "gender_stats": [{"hits": 100, "label": "남성"}],
                "age_stats": None,
                "population_ratio": None,
            }
        )
        cap = DemographicsLookupCapability(action=action)
        context = {
            "date_from": "20220220",
            "date_to": "20220310",
            "searchword": "코로나",
        }
        result = await cap.execute("인구통계", context, None)
        assert result.success is True
        assert result.evidence.status == "partial"

    def test_metadata(self):
        cap = DemographicsLookupCapability()
        assert cap.metadata.name == "demographics_lookup"
        assert cap.metadata.provider == "data.go.kr"


# ---------------------------------------------------------------------------
# data_go_kr.py _call_api 헬퍼 테스트
# ---------------------------------------------------------------------------


class TestCallApiHelper:
    """MinwonAnalysisAction._call_api 공통 헬퍼 테스트."""

    @pytest.mark.asyncio
    async def test_top_level_array_response(self):
        """최상위 배열 응답을 올바르게 파싱한다."""
        from src.inference.actions.data_go_kr import MinwonAnalysisAction

        action = MinwonAnalysisAction(api_key="test_key")

        mock_response = MagicMock()
        mock_response.json.return_value = [{"label": "test", "value": 1}]
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.inference.actions.data_go_kr.httpx.AsyncClient", return_value=mock_client):
            result = await action._call_api("/testEndpoint", {"param": "value"})

        assert result == [{"label": "test", "value": 1}]

    @pytest.mark.asyncio
    async def test_return_object_wrapper(self):
        """returnObject 래핑 응답을 올바르게 파싱한다."""
        from src.inference.actions.data_go_kr import MinwonAnalysisAction

        action = MinwonAnalysisAction(api_key="test_key")

        mock_response = MagicMock()
        mock_response.json.return_value = {"returnObject": [{"keyword": "test", "df": 5}]}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.inference.actions.data_go_kr.httpx.AsyncClient", return_value=mock_client):
            result = await action._call_api("/testEndpoint", {})

        assert result == [{"keyword": "test", "df": 5}]

    @pytest.mark.asyncio
    async def test_error_response_returns_none(self):
        """서버 오류 코드가 있으면 None을 반환한다."""
        from src.inference.actions.data_go_kr import MinwonAnalysisAction

        action = MinwonAnalysisAction(api_key="test_key")

        mock_response = MagicMock()
        mock_response.json.return_value = {"code": "500", "message": "서버 오류"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("src.inference.actions.data_go_kr.httpx.AsyncClient", return_value=mock_client):
            result = await action._call_api("/testEndpoint", {})

        assert result is None
