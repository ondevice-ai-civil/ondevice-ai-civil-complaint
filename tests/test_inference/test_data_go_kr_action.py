"""민원분석 API action 및 api_lookup 중심 테스트."""


from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.inference.actions.data_go_kr import MinwonAnalysisAction
from src.inference.session_context import SessionContext

# 실제 API 응답 구조 (2026-04-06 테스트 기준 — 최상위 배열)
_SAMPLE_ITEMS = [
    {
        "title": "도로 포장 파손 민원",
        "content": "인근 도로 포장이 심하게 파손되어 차량 통행에 위험합니다.",
        "create_date": "20250115120000",
        "main_sub_name": "서울특별시 도로관리과",
        "dep_name": "도로보수팀",
    },
    {
        "title": "보도블록 파손 민원",
        "content": "보도블록이 깨져 보행자 안전이 우려됩니다.",
        "create_date": "20250210150000",
        "main_sub_name": "경기도 고양시",
        "dep_name": "건설과",
    },
]

# 실제 API는 최상위 배열로 응답 (resultCode/body 래핑 없음)
_SAMPLE_API_RESPONSE = _SAMPLE_ITEMS


class TestMinwonAnalysisAction:
    @pytest.mark.asyncio
    async def test_execute_returns_results_and_citations(self):
        action = MinwonAnalysisAction(api_key="test-key")
        session = SessionContext()

        mock_response = MagicMock()
        mock_response.json.return_value = _SAMPLE_API_RESPONSE
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("src.inference.actions.data_go_kr.httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            mock_httpx.TimeoutException = Exception
            mock_httpx.HTTPStatusError = Exception

            result = await action(query="도로 포장 파손", context={}, session=session)

        assert result["success"] is True
        assert result["data"]["count"] == 2
        assert len(result["citations"]) == 2
        assert "공공데이터포털 유사 민원 사례" in result["context_text"]

    @pytest.mark.asyncio
    async def test_timeout_returns_failure(self):
        action = MinwonAnalysisAction(api_key="test-key")
        session = SessionContext()

        class FakeTimeout(Exception):
            pass

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=FakeTimeout("timeout"))

        with patch("src.inference.actions.data_go_kr.httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            mock_httpx.TimeoutException = FakeTimeout
            mock_httpx.HTTPStatusError = Exception

            result = await action(query="도로 포장 파손", context={}, session=session)

        assert result["success"] is False
        assert result["error"] is not None

    @pytest.mark.asyncio
    async def test_enrich_query_uses_session_context_summary(self):
        action = MinwonAnalysisAction(api_key="test-key")
        session = SessionContext()
        session.add_turn("user", "원래 민원 요청")
        session.add_turn("assistant", "이전 답변")

        query = action._enrich_query(
            "근거 보여줘",
            {"session_context": session.build_context_summary()},
        )

        assert "근거 보여줘" in query
        assert "이전 답변" in query or "원래 민원 요청" in query

    def test_enrich_query_respects_prebuilt_api_variant(self):
        action = MinwonAnalysisAction(api_key="test-key")
        prepared = "도로 포장 파손 기존 초안 요약 유사 민원 사례 통계 최근 이슈"

        query = action._enrich_query(
            prepared,
            {
                "session_context": "### 최근 대화\n[사용자] 원래 민원 요청",
                "query_variants": {"api_lookup": prepared},
            },
        )

        assert query == prepared


