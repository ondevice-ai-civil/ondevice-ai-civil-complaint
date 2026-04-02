"""AgentLoop 단위 테스트.

Issue: #393
"""

import asyncio
from typing import Any, Dict
from unittest.mock import AsyncMock

import pytest

from src.inference.agent_loop import AgentLoop, AgentTrace, ToolResult
from src.inference.session_context import SessionContext
from src.inference.tool_router import ToolRouter, ToolType

# ---------------------------------------------------------------------------
# Mock tool 함수
# ---------------------------------------------------------------------------


async def mock_classify(query: str, context: dict, session: Any) -> dict:
    return {
        "classification": {
            "category": "environment",
            "confidence": 0.95,
            "reason": "환경 관련 민원",
        }
    }


async def mock_search_similar(query: str, context: dict, session: Any) -> dict:
    return {
        "results": [
            {"doc_id": "d1", "title": "사례1", "content": "도로 포장 파손 사례", "score": 0.9},
            {"doc_id": "d2", "title": "사례2", "content": "보도블록 파손 사례", "score": 0.8},
        ]
    }


async def mock_generate_civil_response(query: str, context: dict, session: Any) -> dict:
    return {
        "text": "도로 포장 파손에 대해 해당 부서에 보수 요청을 접수하겠습니다.",
        "prompt_tokens": 100,
        "completion_tokens": 50,
    }


async def mock_failing_tool(query: str, context: dict, session: Any) -> dict:
    raise RuntimeError("외부 API 연결 실패")


async def mock_timeout_tool(query: str, context: dict, session: Any) -> dict:
    await asyncio.sleep(10)
    return {"result": "should not reach"}


# ---------------------------------------------------------------------------
# AgentLoop 테스트
# ---------------------------------------------------------------------------


class TestAgentLoop:
    """AgentLoop 단위 테스트."""

    def _make_loop(self, overrides: Dict[ToolType, Any] = None) -> AgentLoop:
        registry = {
            ToolType.CLASSIFY: mock_classify,
            ToolType.SEARCH_SIMILAR: mock_search_similar,
            ToolType.GENERATE_CIVIL_RESPONSE: mock_generate_civil_response,
        }
        if overrides:
            registry.update(overrides)
        return AgentLoop(tool_registry=registry, tool_timeout=2.0)

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """classify -> search_similar -> generate_civil_response 전체 파이프라인 정상 실행."""
        loop = self._make_loop()
        session = SessionContext()

        trace = await loop.run(
            query="도로 포장이 파손되어 위험합니다",
            session=session,
        )

        assert trace.error is None
        assert len(trace.tool_results) == 3
        assert all(r.success for r in trace.tool_results)
        assert "도로 포장" in trace.final_text
        assert trace.total_latency_ms > 0

    @pytest.mark.asyncio
    async def test_session_context_updated(self):
        """실행 후 세션 컨텍스트가 업데이트되는지 확인."""
        loop = self._make_loop()
        session = SessionContext()

        await loop.run(query="민원 처리해주세요", session=session)

        # 대화 기록
        assert len(session.conversations) == 2  # user + assistant
        assert session.conversations[0].role == "user"
        assert session.conversations[1].role == "assistant"

        # 검색 근거
        assert len(session.selected_evidences) > 0

        # 초안
        assert session.latest_draft is not None

    @pytest.mark.asyncio
    async def test_tool_failure_graceful_fallback(self):
        """tool 실패 시 graceful fallback 동작."""
        loop = self._make_loop(overrides={ToolType.CLASSIFY: mock_failing_tool})
        session = SessionContext()

        trace = await loop.run(query="민원 처리", session=session)

        # classify 실패
        assert trace.tool_results[0].success is False
        assert trace.tool_results[0].error

        # search_similar, generate_civil_response는 여전히 실행됨
        assert trace.tool_results[1].success is True
        assert trace.tool_results[2].success is True

        # 최종 텍스트는 생성 결과를 사용
        assert trace.final_text

    @pytest.mark.asyncio
    async def test_tool_timeout(self):
        """tool 타임아웃 처리."""
        loop = self._make_loop(overrides={ToolType.SEARCH_SIMILAR: mock_timeout_tool})
        session = SessionContext()

        trace = await loop.run(query="민원 처리", session=session)

        search_result = trace.tool_results[1]
        assert search_result.success is False
        assert "타임아웃" in search_result.error

    @pytest.mark.asyncio
    async def test_force_tools(self):
        """force_tools로 특정 tool만 실행."""
        loop = self._make_loop()
        session = SessionContext()

        trace = await loop.run(
            query="검색만",
            session=session,
            force_tools=[ToolType.SEARCH_SIMILAR],
        )

        assert len(trace.tool_results) == 1
        assert trace.tool_results[0].tool == ToolType.SEARCH_SIMILAR

    @pytest.mark.asyncio
    async def test_force_tools_with_custom_registry_tool(self):
        """문자열 기반 custom tool도 registry에서 실행할 수 있다."""

        async def custom_tool(query: str, context: dict, session: Any) -> dict:
            return {"text": f"custom::{query}"}

        loop = AgentLoop(tool_registry={"custom_lookup": custom_tool}, tool_timeout=2.0)
        session = SessionContext()

        trace = await loop.run(
            query="테스트",
            session=session,
            force_tools=["custom_lookup"],
        )

        assert len(trace.tool_results) == 1
        assert trace.tool_results[0].tool == "custom_lookup"
        assert trace.final_text == "custom::테스트"

    @pytest.mark.asyncio
    async def test_trace_to_dict(self):
        """AgentTrace.to_dict() 직렬화 확인."""
        loop = self._make_loop()
        session = SessionContext()

        trace = await loop.run(query="테스트", session=session)
        trace_dict = trace.to_dict()

        assert "request_id" in trace_dict
        assert "session_id" in trace_dict
        assert "plan" in trace_dict
        assert "tool_results" in trace_dict
        assert "total_latency_ms" in trace_dict

    @pytest.mark.asyncio
    async def test_extract_final_text_without_generate(self):
        """생성 없이 classify/search_similar 결과만으로 요약."""
        loop = self._make_loop()
        session = SessionContext()

        trace = await loop.run(
            query="이 민원을 분류하고 사례를 검색해주세요",
            session=session,
            force_tools=[ToolType.CLASSIFY, ToolType.SEARCH_SIMILAR],
        )

        assert "분류 결과" in trace.final_text
        assert "유사 민원 사례" in trace.final_text


class TestAgentLoopStream:
    """AgentLoop 스트리밍 테스트."""

    @pytest.mark.asyncio
    async def test_stream_events(self):
        """스트리밍 이벤트가 올바른 순서로 전달되는지 확인."""
        registry = {
            ToolType.CLASSIFY: mock_classify,
            ToolType.SEARCH_SIMILAR: mock_search_similar,
            ToolType.GENERATE_CIVIL_RESPONSE: mock_generate_civil_response,
        }
        loop = AgentLoop(tool_registry=registry)
        session = SessionContext()

        events = []
        async for event in loop.run_stream(query="민원 처리해주세요", session=session):
            events.append(event)

        event_types = [e["type"] for e in events]
        assert event_types[0] == "plan"
        assert "tool_start" in event_types
        assert "tool_result" in event_types
        assert event_types[-1] == "final"

        # final 이벤트에는 완성된 텍스트가 있어야 함
        final = events[-1]
        assert final["finished"] is True
        assert final["text"]

    @pytest.mark.asyncio
    async def test_stream_error_handling(self):
        """스트리밍 중 전체 오류 시 error 이벤트 전달."""

        async def bad_classify(query, context, session):
            raise Exception("치명적 오류")

        registry = {
            ToolType.CLASSIFY: bad_classify,
            ToolType.SEARCH_SIMILAR: mock_search_similar,
            ToolType.GENERATE_CIVIL_RESPONSE: mock_generate_civil_response,
        }
        loop = AgentLoop(tool_registry=registry)
        session = SessionContext()

        events = []
        async for event in loop.run_stream(query="민원", session=session):
            events.append(event)

        # tool_result에서 실패가 보고되어야 함
        tool_results = [e for e in events if e["type"] == "tool_result"]
        classify_result = tool_results[0]
        assert classify_result["success"] is False


class TestToolResult:
    """ToolResult 단위 테스트."""

    def test_to_dict(self):
        result = ToolResult(
            tool=ToolType.CLASSIFY,
            success=True,
            data={"classification": {"category": "test"}},
            latency_ms=15.678,
        )
        d = result.to_dict()
        assert d["tool"] == "classify"
        assert d["success"] is True
        assert d["latency_ms"] == 15.68
