"""MinwonAnalysisAction 단위/통합 테스트.

Issue: #394
"""

import json
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.inference.actions.base import ActionResult, Citation
from src.inference.actions.data_go_kr import MinwonAnalysisAction
from src.inference.agent_loop import AgentLoop
from src.inference.session_context import SessionContext
from src.inference.tool_router import ToolRouter, ToolType


# ---------------------------------------------------------------------------
# 헬퍼 — 유사 민원 API 응답 픽스처
# ---------------------------------------------------------------------------

_SAMPLE_ITEMS = [
    {
        "title": "도로 포장 파손 민원",
        "content": "인근 도로 포장이 심하게 파손되어 차량 통행에 위험합니다.",
        "answer": "해당 구역 도로 보수 작업을 2주 내 완료하겠습니다.",
        "category": "교통",
        "regDate": "2025-01-15",
    },
    {
        "title": "보도블록 파손 민원",
        "content": "보도블록이 깨져 보행자 안전이 우려됩니다.",
        "answer": "보도블록 교체 공사를 진행할 예정입니다.",
        "category": "교통",
        "regDate": "2025-02-10",
    },
]

_SAMPLE_API_RESPONSE = {
    "resultCode": "00",
    "resultMsg": "NORMAL SERVICE.",
    "body": {
        "items": _SAMPLE_ITEMS,
        "totalCount": 2,
        "pageNo": 1,
    },
}


# ---------------------------------------------------------------------------
# 테스트
# ---------------------------------------------------------------------------


class TestMinwonAnalysisActionSuccess:
    """정상 케이스 테스트."""

    @pytest.mark.asyncio
    async def test_similar_cases_success(self):
        """httpx 응답 mock → 유사 사례 반환, citations 포함 확인."""
        action = MinwonAnalysisAction(api_key="test-key-12345")
        session = SessionContext()
        context: Dict[str, Any] = {}

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

            result_dict = await action(
                query="도로 포장 파손",
                context=context,
                session=session,
            )

        assert result_dict["success"] is True
        assert result_dict["source"] == "data.go.kr"
        assert result_dict["data"]["count"] == 2
        assert len(result_dict["citations"]) > 0
        assert result_dict["context_text"] != ""

        # 첫 번째 citation 필드 확인
        first_cite = result_dict["citations"][0]
        assert "title" in first_cite
        assert first_cite["title"] != ""


class TestMinwonAnalysisActionTimeout:
    """타임아웃 및 오류 케이스 테스트."""

    @pytest.mark.asyncio
    async def test_timeout_graceful(self):
        """httpx.TimeoutException → success=False, 에러 메시지 반환."""
        action = MinwonAnalysisAction(api_key="test-key-12345")
        session = SessionContext()

        class FakeTimeoutException(Exception):
            pass

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=FakeTimeoutException("timeout"))

        with patch("src.inference.actions.data_go_kr.httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            mock_httpx.TimeoutException = FakeTimeoutException
            mock_httpx.HTTPStatusError = Exception

            result_dict = await action(
                query="도로 포장 파손",
                context={},
                session=session,
            )

        assert result_dict["success"] is False
        assert result_dict["error"] is not None

    @pytest.mark.asyncio
    async def test_api_error_graceful(self):
        """resultCode=500 → 에러 메시지 반환."""
        action = MinwonAnalysisAction(api_key="test-key-12345")
        session = SessionContext()

        error_response = {
            "resultCode": "500",
            "resultMsg": "SERVICE ERROR",
        }

        mock_response = MagicMock()
        mock_response.json.return_value = error_response
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("src.inference.actions.data_go_kr.httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            mock_httpx.TimeoutException = Exception
            mock_httpx.HTTPStatusError = Exception

            result_dict = await action(
                query="도로 포장 파손",
                context={},
                session=session,
            )

        assert result_dict["success"] is False
        assert result_dict["error"] is not None

    @pytest.mark.asyncio
    async def test_empty_results(self):
        """빈 배열 → 빈 결과 (success=True, count=0)."""
        action = MinwonAnalysisAction(api_key="test-key-12345")
        session = SessionContext()

        empty_response = {
            "resultCode": "00",
            "body": {
                "items": [],
                "totalCount": 0,
            },
        }

        mock_response = MagicMock()
        mock_response.json.return_value = empty_response
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("src.inference.actions.data_go_kr.httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            mock_httpx.TimeoutException = Exception
            mock_httpx.HTTPStatusError = Exception

            result_dict = await action(
                query="도로 포장 파손",
                context={},
                session=session,
            )

        assert result_dict["success"] is True
        assert result_dict["data"]["count"] == 0
        assert result_dict["context_text"] == ""


class TestToolRouterApiLookup:
    """ToolRouter API_LOOKUP 키워드 매칭 테스트."""

    def setup_method(self):
        self.router = ToolRouter()

    def test_tool_router_api_lookup(self):
        """'민원 분석 사례 조회' → api_lookup 포함."""
        plan = self.router.plan("민원 분석 사례 조회 해주세요")
        assert ToolType.API_LOOKUP.value in plan.tool_names

    def test_api_lookup_gongong_data(self):
        """'공공 데이터' 키워드 → api_lookup 포함."""
        plan = self.router.plan("공공 데이터에서 민원 현황 조회해줘")
        assert ToolType.API_LOOKUP.value in plan.tool_names

    def test_api_lookup_trend(self):
        """'트렌드' 키워드 → api_lookup 포함."""
        plan = self.router.plan("민원 트렌드 분석해줘")
        assert ToolType.API_LOOKUP.value in plan.tool_names

    def test_api_lookup_with_classify(self):
        """api_lookup + classify 함께 요청 시 classify가 먼저 배치."""
        plan = self.router.plan("이 민원을 분류하고 민원 통계 조회해줘")
        names = plan.tool_names
        assert "classify" in names
        assert "api_lookup" in names
        assert names.index("classify") < names.index("api_lookup")

    def test_no_api_lookup_plain_minwon(self):
        """일반 민원 텍스트에는 api_lookup이 포함되지 않는다."""
        plan = self.router.plan("도로 포장이 파손되어 위험합니다")
        # 전체 파이프라인 실행 (api_lookup 없음)
        assert ToolType.API_LOOKUP.value not in plan.tool_names


class TestAgentLoopApiLookupIntegration:
    """AgentLoop + api_lookup 통합 테스트."""

    @pytest.mark.asyncio
    async def test_agent_loop_integration(self):
        """mock api_lookup → AgentLoop 전체 실행 확인."""

        async def mock_classify(query: str, context: dict, session: Any) -> dict:
            return {
                "classification": {
                    "category": "traffic",
                    "confidence": 0.9,
                    "reason": "교통 관련 민원",
                }
            }

        async def mock_api_lookup(query: str, context: dict, session: Any) -> dict:
            return {
                "success": True,
                "data": {"results": _SAMPLE_ITEMS, "query": query, "count": 2},
                "error": None,
                "source": "data.go.kr",
                "citations": [
                    {
                        "title": "도로 포장 파손 민원",
                        "url": "",
                        "date": "2025-01-15",
                        "snippet": "인근 도로 포장이 심하게 파손...",
                        "metadata": {},
                    }
                ],
                "context_text": "### 공공데이터포털 유사 민원 사례\n1. [교통] 도로 포장 파손 민원",
            }

        registry = {
            ToolType.CLASSIFY: mock_classify,
            ToolType.API_LOOKUP: mock_api_lookup,
        }
        loop = AgentLoop(
            tool_registry=registry,
            tool_timeout=5.0,
        )
        session = SessionContext()

        trace = await loop.run(
            query="민원 분석 사례 조회",
            session=session,
            force_tools=[ToolType.CLASSIFY, ToolType.API_LOOKUP],
        )

        assert trace.error is None
        assert len(trace.tool_results) == 2

        # classify 성공
        assert trace.tool_results[0].tool == ToolType.CLASSIFY
        assert trace.tool_results[0].success is True

        # api_lookup 성공
        assert trace.tool_results[1].tool == ToolType.API_LOOKUP
        assert trace.tool_results[1].success is True

        # 최종 텍스트에 API 결과가 포함되어야 함
        assert trace.final_text != ""
        assert "공공데이터포털" in trace.final_text or "분류 결과" in trace.final_text
