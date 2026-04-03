"""민원분석 API action 및 api_lookup 중심 테스트."""

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.inference.actions.data_go_kr import MinwonAnalysisAction
from src.inference.agent_loop import AgentLoop
from src.inference.session_context import SessionContext
from src.inference.tool_router import ToolRouter, ToolType

_SAMPLE_ITEMS = [
    {
        "title": "도로 포장 파손 민원",
        "content": "인근 도로 포장이 심하게 파손되어 차량 통행에 위험합니다.",
        "answer": "해당 구역 도로 보수 작업을 2주 내 완료하겠습니다.",
        "category": "교통",
        "regDate": "2025-01-15",
        "url": "https://example.com/case-1",
    },
    {
        "title": "보도블록 파손 민원",
        "content": "보도블록이 깨져 보행자 안전이 우려됩니다.",
        "answer": "보도블록 교체 공사를 진행할 예정입니다.",
        "category": "교통",
        "regDate": "2025-02-10",
        "url": "https://example.com/case-2",
    },
]

_SAMPLE_API_RESPONSE = {
    "resultCode": "00",
    "resultMsg": "NORMAL SERVICE.",
    "body": {"items": _SAMPLE_ITEMS, "totalCount": 2, "pageNo": 1},
}


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


class TestToolRouterApiLookup:
    def setup_method(self):
        self.router = ToolRouter()

    def test_lookup_route_for_statistics(self):
        plan = self.router.plan("민원 통계와 최근 이슈를 조회해줘")
        assert plan.tool_names == ["api_lookup"]

    def test_evidence_request_keeps_api_lookup_in_plan(self):
        plan = self.router.plan("이 답변의 출처를 붙여줘")
        assert ToolType.API_LOOKUP.value in plan.tool_names


class TestAgentLoopApiLookupIntegration:
    @pytest.mark.asyncio
    async def test_agent_loop_with_api_lookup_and_draft(self):
        async def mock_rag_search(query: str, context: dict, session: Any) -> dict:
            return {"results": [{"title": "매뉴얼", "content": "절차 안내"}], "count": 1}

        async def mock_api_lookup(query: str, context: dict, session: Any) -> dict:
            return {
                "results": _SAMPLE_ITEMS,
                "context_text": "### 공공데이터포털 유사 민원 사례\n1. 도로 포장 파손 민원",
                "count": 2,
            }

        async def mock_draft(query: str, context: dict, session: Any) -> dict:
            return {"text": "근거 요약\n- 로컬 문서 1건\n\n최종 초안\n보수 접수를 진행하겠습니다."}

        loop = AgentLoop(
            tool_registry={
                ToolType.RAG_SEARCH: mock_rag_search,
                ToolType.API_LOOKUP: mock_api_lookup,
                ToolType.DRAFT_CIVIL_RESPONSE: mock_draft,
            }
        )
        session = SessionContext()

        trace = await loop.run("민원 답변 작성", session)

        assert trace.error is None
        assert trace.plan.tool_names == ["rag_search", "api_lookup", "draft_civil_response"]
        assert trace.tool_results[1].tool == ToolType.API_LOOKUP
        assert "최종 초안" in trace.final_text
