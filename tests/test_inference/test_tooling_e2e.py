"""capability execute → LookupResult → to_dict() E2E 파이프라인 테스트.

각 capability가 mock action/execute_fn을 통해 올바른 LookupResult를 생성하고,
to_dict()가 정해진 스키마를 반환하는지 검증한다.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# import — 프로젝트 모듈 import 실패는 테스트 실패로 노출시킨다.
# (broad ImportError skip은 내부 리팩터링 회귀를 숨길 수 있으므로 사용하지 않는다.)
# ---------------------------------------------------------------------------
from src.inference.graph.capabilities.api_lookup import ApiLookupCapability
from src.inference.graph.capabilities.base import (
    EvidenceEnvelope,
    LookupResult,
)
from src.inference.graph.capabilities.demographics_lookup import DemographicsLookupCapability
from src.inference.graph.capabilities.issue_detector import IssueDetectorCapability
from src.inference.graph.capabilities.keyword_analyzer import KeywordAnalyzerCapability
from src.inference.graph.capabilities.stats_lookup import StatsLookupCapability

# ---------------------------------------------------------------------------
# 공통 헬퍼
# ---------------------------------------------------------------------------


def _assert_to_dict_schema(d: dict) -> None:
    """to_dict() 결과가 공통 필수 키를 보유하는지 확인한다."""
    required_keys = {
        "success",
        "query",
        "count",
        "results",
        "context_text",
        "citations",
        "provider",
        "error",
        "empty_reason",
        "latency_ms",
    }
    assert required_keys.issubset(d.keys()), f"누락된 키: {required_keys - d.keys()}"


# ===========================================================================
# TestApiLookupPipeline
# ===========================================================================


class TestApiLookupPipeline:
    """ApiLookupCapability execute → LookupResult → to_dict() 파이프라인 검증."""

    @pytest.mark.asyncio
    async def test_api_lookup_returns_lookup_result(self):
        """action.fetch_similar_cases 결과가 LookupResult로 올바르게 변환되는지 검증한다."""
        mock_action = MagicMock()
        mock_action.fetch_similar_cases = AsyncMock(
            return_value={
                "query": "민원 처리 절차",
                "results": [
                    {
                        "qnaTitle": "민원 처리 안내",
                        "qnaContent": "민원은 다음 절차로 처리됩니다.",
                        "qnaAnswer": "1단계: 접수 2단계: 검토 3단계: 처리",
                        "detailUrl": "https://example.go.kr/qna/001",
                        "score": 8,
                    }
                ],
                "context_text": "유사 민원 사례",
                "citations": [],
            }
        )

        cap = ApiLookupCapability(action=mock_action)
        result = await cap.execute("민원 처리 절차", {}, None)

        assert result.success is True
        assert len(result.results) > 0
        assert result.evidence is not None
        assert len(result.evidence.items) > 0
        assert result.evidence.items[0].source_type == "api"
        assert result.evidence.status == "ok"

        d = result.to_dict()
        _assert_to_dict_schema(d)
        assert d["success"] is True
        assert d["count"] >= 1
        assert "evidence" in d

    @pytest.mark.asyncio
    async def test_api_lookup_no_action_returns_empty(self):
        """action=None이면 success=True, empty_reason='no_match'를 반환한다."""
        cap = ApiLookupCapability(action=None)
        result = await cap.execute("민원 처리 절차", {}, None)

        assert result.success is True
        assert result.empty_reason == "no_match"
        assert len(result.results) == 0

        d = result.to_dict()
        _assert_to_dict_schema(d)
        assert d["success"] is True
        assert d["empty_reason"] == "no_match"
        assert d["count"] == 0


# ===========================================================================
# TestIssueDetectorPipeline
# ===========================================================================


class TestIssueDetectorPipeline:
    """IssueDetectorCapability execute → LookupResult → to_dict() 파이프라인 검증."""

    @pytest.mark.asyncio
    async def test_issue_detection_returns_combined(self):
        """3개 API 모두 성공 시 combined 결과와 evidence status='ok'를 검증한다."""
        mock_action = MagicMock()
        mock_action.get_rising_keywords = AsyncMock(
            return_value=[
                {"keyword": "도로파손", "df": 120, "prevRatio": "25"},
                {"keyword": "쓰레기수거", "df": 98, "prevRatio": "15"},
            ]
        )
        mock_action.get_today_topics = AsyncMock(
            return_value=[
                {"topic": "생활불편", "count": 340},
                {"topic": "교통민원", "count": 210},
            ]
        )
        mock_action.get_top_keywords_by_period = AsyncMock(
            return_value=[
                {"term": "도로", "df": 500},
                {"term": "민원", "df": 450},
            ]
        )

        context = {"analysis_time": "2026040814", "search_date": "20260408"}
        cap = IssueDetectorCapability(action=mock_action)
        result = await cap.execute("민원 이슈 탐지", context, None)

        assert result.success is True
        # 3개 소스에서 결과가 조합되어야 한다
        assert len(result.results) == 6  # rising 2 + topics 2 + top_kw 2
        source_apis = {r["_source_api"] for r in result.results}
        assert "rising_keyword" in source_apis
        assert "today_topic" in source_apis
        assert "top_keyword" in source_apis

        assert result.evidence is not None
        assert result.evidence.status == "ok"
        assert len(result.evidence.items) == 6

        d = result.to_dict()
        _assert_to_dict_schema(d)
        assert d["success"] is True
        assert d["count"] == 6
        assert "evidence" in d
        assert d["evidence"]["status"] == "ok"

    @pytest.mark.asyncio
    async def test_issue_detection_partial_api_failure(self):
        """today_topics API 실패 시 partial 결과, evidence.status='partial', errors 1개를 검증한다."""
        mock_action = MagicMock()
        mock_action.get_rising_keywords = AsyncMock(
            return_value=[
                {"keyword": "도로파손", "df": 120, "prevRatio": "25"},
            ]
        )
        mock_action.get_today_topics = AsyncMock(
            side_effect=Exception("오늘이슈 API 네트워크 오류")
        )
        mock_action.get_top_keywords_by_period = AsyncMock(
            return_value=[
                {"term": "도로", "df": 500},
            ]
        )

        context = {"analysis_time": "2026040814", "search_date": "20260408"}
        cap = IssueDetectorCapability(action=mock_action)
        result = await cap.execute("민원 이슈 탐지", context, None)

        # 1개 API 실패여도 나머지 결과로 성공 반환
        assert result.success is True
        assert len(result.results) == 2  # rising 1 + top_kw 1

        assert result.evidence is not None
        assert result.evidence.status == "partial"
        assert len(result.evidence.errors) == 1
        assert "오늘이슈" in result.evidence.errors[0]

        d = result.to_dict()
        _assert_to_dict_schema(d)
        assert d["success"] is True
        assert d["evidence"]["status"] == "partial"
        assert len(d["evidence"]["errors"]) == 1


# ===========================================================================
# TestStatsLookupPipeline
# ===========================================================================


class TestStatsLookupPipeline:
    """StatsLookupCapability execute → LookupResult → to_dict() 파이프라인 검증."""

    @pytest.mark.asyncio
    async def test_stats_lookup_with_keyword_uses_doc_count_and_trend(self):
        """searchword가 있으면 doc_count + trend API를 사용하여 결과를 반환한다."""
        mock_action = MagicMock()
        mock_action.get_doc_count = AsyncMock(
            return_value=[{"pttn": 150, "dfpt": 80, "saeol": 30, "label": "도로파손"}]
        )
        mock_action.get_trend = AsyncMock(
            return_value=[
                {"label": "20260408", "hits": 260, "prebRatio": "12"},
                {"label": "20260407", "hits": 232, "prebRatio": "-3"},
            ]
        )

        context = {
            "date_from": "20260101",
            "date_to": "20260408",
            "searchword": "도로파손",
            "period": "DAILY",
        }
        cap = StatsLookupCapability(action=mock_action)
        result = await cap.execute("도로파손 통계", context, None)

        assert result.success is True
        # doc_count 1개 + trend 2개
        assert len(result.results) == 3
        source_apis = {r["_source_api"] for r in result.results}
        assert "doc_count" in source_apis
        assert "trend" in source_apis

        assert result.evidence is not None
        assert result.evidence.status == "ok"

        d = result.to_dict()
        _assert_to_dict_schema(d)
        assert d["success"] is True
        assert d["count"] == 3
        assert "evidence" in d

        # 호출 계약 검증: searchword가 있을 때 doc_count + trend가 호출되었는지 확인
        mock_action.get_doc_count.assert_awaited_once()
        mock_action.get_trend.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stats_lookup_without_keyword_uses_statistics_and_rankings(self):
        """searchword 없이 date_from/date_to만 있으면 statistics + org_ranking + region_ranking을 사용한다."""
        mock_action = MagicMock()
        mock_action.get_statistics = AsyncMock(
            return_value=[
                {"label": "접수", "hits": 1200},
                {"label": "처리", "hits": 1100},
            ]
        )
        mock_action.get_org_ranking = AsyncMock(
            return_value=[
                {"label": "서울시", "hits": 300},
                {"label": "경기도", "hits": 250},
            ]
        )
        mock_action.get_region_ranking = AsyncMock(
            return_value=[
                {"label": "강남구", "hits": 150},
                {"label": "서초구", "hits": 120},
            ]
        )

        context = {
            "date_from": "20260101",
            "date_to": "20260408",
            "period": "MONTHLY",
            "top_n": 5,
        }
        cap = StatsLookupCapability(action=mock_action)
        result = await cap.execute("전체 민원 통계", context, None)

        assert result.success is True
        # statistics 2 + org_ranking 2 + region_ranking 2
        assert len(result.results) == 6
        source_apis = {r["_source_api"] for r in result.results}
        assert "statistics" in source_apis
        assert "org_ranking" in source_apis
        assert "region_ranking" in source_apis

        assert result.evidence is not None
        assert result.evidence.status == "ok"

        d = result.to_dict()
        _assert_to_dict_schema(d)
        assert d["success"] is True
        assert d["count"] == 6
        assert "evidence" in d
        assert d["evidence"]["status"] == "ok"

        # 호출 계약 검증: searchword 없을 때 statistics + org_ranking + region_ranking 호출
        mock_action.get_statistics.assert_awaited_once()
        mock_action.get_org_ranking.assert_awaited_once()
        mock_action.get_region_ranking.assert_awaited_once()


# ===========================================================================
# TestKeywordAnalyzerPipeline
# ===========================================================================


class TestKeywordAnalyzerPipeline:
    """KeywordAnalyzerCapability execute → LookupResult → to_dict() 파이프라인 검증."""

    @pytest.mark.asyncio
    async def test_keyword_analyzer_returns_core_and_related(self):
        """get_core_keywords + get_related_words 결과가 올바르게 조합되는지 검증한다."""
        mock_action = MagicMock()
        mock_action.get_core_keywords = AsyncMock(
            return_value=[
                {"label": "도로파손", "value": 450},
                {"label": "보수공사", "value": 320},
                {"label": "신고접수", "value": 280},
            ]
        )
        mock_action.get_related_words = AsyncMock(
            return_value=[
                {"label": "포트홀", "value": 0.92},
                {"label": "아스팔트", "value": 0.87},
            ]
        )

        context = {
            "date_from": "20260101",
            "date_to": "20260408",
            "searchword": "도로파손",
            "result_count": 5,
        }
        cap = KeywordAnalyzerCapability(action=mock_action)
        result = await cap.execute("도로파손 키워드 분석", context, None)

        assert result.success is True
        # core_keywords 3 + related_words 2
        assert len(result.results) == 5
        source_apis = {r["_source_api"] for r in result.results}
        assert "core_keyword" in source_apis
        assert "related_word" in source_apis

        assert result.evidence is not None
        assert result.evidence.status == "ok"
        assert len(result.evidence.items) == 5
        # context_text에 핵심 키워드와 연관어 요약이 포함되어야 한다
        assert "핵심 키워드" in result.context_text
        assert "연관어" in result.context_text

        d = result.to_dict()
        _assert_to_dict_schema(d)
        assert d["success"] is True
        assert d["count"] == 5
        assert "evidence" in d
        assert d["evidence"]["status"] == "ok"


# ===========================================================================
# TestDemographicsLookupPipeline
# ===========================================================================


class TestDemographicsLookupPipeline:
    """DemographicsLookupCapability execute → LookupResult → to_dict() 파이프라인 검증."""

    @pytest.mark.asyncio
    async def test_demographics_returns_gender_age_population(self):
        """get_gender_stats + get_age_stats + get_population_ratio 결과가 올바르게 조합되는지 검증한다."""
        mock_action = MagicMock()
        mock_action.get_gender_stats = AsyncMock(
            return_value=[
                {"label": "남성", "hits": 620},
                {"label": "여성", "hits": 380},
            ]
        )
        mock_action.get_age_stats = AsyncMock(
            return_value=[
                {"label": "30", "hits": 310},
                {"label": "40", "hits": 290},
                {"label": "50", "hits": 200},
            ]
        )
        mock_action.get_population_ratio = AsyncMock(
            return_value=[
                {"label": "강남구", "ratio": "0.0045"},
                {"label": "서초구", "ratio": "0.0038"},
            ]
        )

        context = {
            "searchword": "도로파손",
            "date_from": "20260101",
            "date_to": "20260408",
            "top_n": 5,
        }
        cap = DemographicsLookupCapability(action=mock_action)
        result = await cap.execute("도로파손 인구통계", context, None)

        assert result.success is True
        # gender 2 + age 3 + population 2
        assert len(result.results) == 7
        source_apis = {r["_source_api"] for r in result.results}
        assert "gender" in source_apis
        assert "age" in source_apis
        assert "population" in source_apis

        assert result.evidence is not None
        assert result.evidence.status == "ok"
        assert len(result.evidence.items) == 7
        # context_text에 성별/연령 요약이 포함되어야 한다
        assert result.context_text != ""

        d = result.to_dict()
        _assert_to_dict_schema(d)
        assert d["success"] is True
        assert d["count"] == 7
        assert "evidence" in d
        assert d["evidence"]["status"] == "ok"
        # evidence items 중 source_type이 "api"인지 확인
        for item in d["evidence"]["items"]:
            assert item["source_type"] == "api"
