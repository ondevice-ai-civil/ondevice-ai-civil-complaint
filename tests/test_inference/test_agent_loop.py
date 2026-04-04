"""GovOn MVP AgentLoop 단위 테스트."""

import asyncio
from typing import Any, Dict

import pytest

from src.inference.agent_loop import AgentLoop
from src.inference.session_context import SessionContext
from src.inference.tool_router import ToolType


async def mock_rag_search(query: str, context: dict, session: Any) -> dict:
    return {
        "query": query,
        "results": [
            {"title": "도로 보수 매뉴얼", "content": "보수 절차 안내", "metadata": {"page": 3}},
            {"title": "민원 처리 지침", "content": "담당 부서 이관 기준", "metadata": {"page": 5}},
        ],
        "count": 2,
    }


async def mock_api_lookup(query: str, context: dict, session: Any) -> dict:
    return {
        "query": query,
        "results": [
            {"title": "도로 파손 유사 민원", "content": "유사 사례 본문", "answer": "보수 예정"},
        ],
        "context_text": "### 공공데이터포털 유사 민원 사례\n1. 도로 파손 유사 민원",
        "count": 1,
    }


async def mock_draft_civil_response(query: str, context: dict, session: Any) -> dict:
    return {
        "text": "근거 요약\n- 로컬 문서 2건을 참고했습니다.\n\n최종 초안\n도로 보수 접수를 진행하겠습니다.",
        "draft_text": "도로 보수 접수를 진행하겠습니다.",
    }


async def mock_append_evidence(query: str, context: dict, session: Any) -> dict:
    return {
        "text": "이전 답변\n\n근거/출처\n[1] /tmp/sample.pdf (p.3)\n[2] 도로 파손 유사 민원 - https://example.com",
    }


async def mock_failing_tool(query: str, context: dict, session: Any) -> dict:
    raise RuntimeError("외부 조회 실패")


async def mock_timeout_tool(query: str, context: dict, session: Any) -> dict:
    await asyncio.sleep(10)
    return {"result": "timeout"}


class TestAgentLoop:
    def _make_loop(self, overrides: Dict[ToolType | str, Any] | None = None) -> AgentLoop:
        registry = {
            ToolType.RAG_SEARCH: mock_rag_search,
            ToolType.API_LOOKUP: mock_api_lookup,
            ToolType.DRAFT_CIVIL_RESPONSE: mock_draft_civil_response,
            ToolType.APPEND_EVIDENCE: mock_append_evidence,
        }
        if overrides:
            registry.update(overrides)
        return AgentLoop(tool_registry=registry, tool_timeout=2.0)

    @pytest.mark.asyncio
    async def test_default_drafting_loop_runs(self):
        loop = self._make_loop()
        session = SessionContext()

        trace = await loop.run("도로 포장이 파손되어 위험합니다", session)

        assert trace.error is None
        assert trace.plan.tool_names == ["rag_search", "api_lookup", "draft_civil_response"]
        assert len(trace.tool_results) == 3
        assert all(result.success for result in trace.tool_results)
        assert "최종 초안" in trace.final_text

    @pytest.mark.asyncio
    async def test_session_stores_turns_and_tool_runs(self):
        loop = self._make_loop()
        session = SessionContext()

        await loop.run("민원 답변 초안 작성해줘", session)

        assert len(session.conversations) == 2
        assert session.conversations[0].role == "user"
        assert session.conversations[1].role == "assistant"
        assert len(session.tool_runs) == 3
        assert len(session.graph_runs) == 1
        assert all(
            tool_run.graph_run_request_id == session.graph_runs[0].request_id
            for tool_run in session.tool_runs
        )
        assert session.graph_runs[0].approval_status == "not_requested"
        assert session.graph_runs[0].executed_capabilities == [
            "rag_search",
            "api_lookup",
            "draft_civil_response",
        ]
        assert session.graph_runs[0].status == "completed"

    @pytest.mark.asyncio
    async def test_tool_failure_does_not_abort_remaining_tools(self):
        loop = self._make_loop(overrides={ToolType.RAG_SEARCH: mock_failing_tool})
        session = SessionContext()

        trace = await loop.run("민원 답변 초안 작성", session)

        assert trace.tool_results[0].success is False
        assert trace.tool_results[1].success is True
        assert trace.tool_results[2].success is True
        assert "최종 초안" in trace.final_text
        assert session.graph_runs[0].status == "completed_with_errors"
        assert all(
            tool_run.graph_run_request_id == trace.request_id for tool_run in session.tool_runs
        )

    @pytest.mark.asyncio
    async def test_tool_timeout_is_recorded(self):
        loop = self._make_loop(overrides={ToolType.API_LOOKUP: mock_timeout_tool})
        session = SessionContext()

        trace = await loop.run("민원 답변 초안 작성", session)

        api_result = trace.tool_results[1]
        assert api_result.success is False
        assert "타임아웃" in api_result.error
        assert session.tool_runs[1].graph_run_request_id == trace.request_id

    @pytest.mark.asyncio
    async def test_force_tools_runs_single_tool(self):
        loop = self._make_loop()
        session = SessionContext()

        trace = await loop.run(
            "근거만 정리해줘",
            session,
            force_tools=[ToolType.APPEND_EVIDENCE],
        )

        assert len(trace.tool_results) == 1
        assert trace.tool_results[0].tool == ToolType.APPEND_EVIDENCE
        assert "근거/출처" in trace.final_text

    @pytest.mark.asyncio
    async def test_force_tools_accepts_custom_registry_tool(self):
        async def custom_tool(query: str, context: dict, session: Any) -> dict:
            return {"text": f"custom::{query}"}

        loop = AgentLoop(tool_registry={"custom_lookup": custom_tool}, tool_timeout=2.0)
        session = SessionContext()

        trace = await loop.run("테스트", session, force_tools=["custom_lookup"])

        assert len(trace.tool_results) == 1
        assert trace.final_text == "custom::테스트"

    @pytest.mark.asyncio
    async def test_fallback_summary_uses_retrieval_outputs(self):
        loop = self._make_loop()
        session = SessionContext()

        trace = await loop.run(
            "근거를 보여줘",
            session,
            force_tools=[ToolType.RAG_SEARCH, ToolType.API_LOOKUP],
        )

        assert "[로컬 문서 근거]" in trace.final_text
        assert "공공데이터포털" in trace.final_text

    @pytest.mark.asyncio
    async def test_follow_up_uses_context_aware_query_variants_for_search_tools(self):
        seen_queries: Dict[str, str] = {}

        async def capture_rag(query: str, context: dict, session: Any) -> dict:
            seen_queries["rag_search"] = query
            return {"query": query, "results": [], "count": 0}

        async def capture_api(query: str, context: dict, session: Any) -> dict:
            seen_queries["api_lookup"] = query
            return {"query": query, "results": [], "count": 0, "context_text": ""}

        async def capture_append(query: str, context: dict, session: Any) -> dict:
            seen_queries["append_evidence"] = query
            return {"text": f"append::{query}"}

        loop = AgentLoop(
            tool_registry={
                ToolType.RAG_SEARCH: capture_rag,
                ToolType.API_LOOKUP: capture_api,
                ToolType.APPEND_EVIDENCE: capture_append,
            },
            tool_timeout=2.0,
        )
        session = SessionContext()
        session.add_turn("user", "도로 포장이 파손되어 위험합니다")
        session.add_turn(
            "assistant",
            "도로 보수 접수를 진행하겠습니다. 담당 부서 검토 후 보수 일정을 안내드리겠습니다.",
        )

        trace = await loop.run("이 답변의 근거를 붙여줘", session)

        assert trace.error is None
        assert seen_queries["append_evidence"] == "이 답변의 근거를 붙여줘"
        assert "도로 포장이 파손되어 위험합니다" in seen_queries["rag_search"]
        assert "담당 부서 검토 후 보수 일정을 안내드리겠습니다." in seen_queries["rag_search"]
        assert "관련 법령 지침 매뉴얼 공지 내부 문서" in seen_queries["rag_search"]
        assert "유사 민원 사례 통계 최근 이슈" in seen_queries["api_lookup"]
        assert seen_queries["rag_search"] != seen_queries["api_lookup"]


class TestAgentLoopStream:
    @pytest.mark.asyncio
    async def test_stream_events_are_emitted_in_order(self):
        loop = AgentLoop(
            tool_registry={
                ToolType.RAG_SEARCH: mock_rag_search,
                ToolType.API_LOOKUP: mock_api_lookup,
                ToolType.DRAFT_CIVIL_RESPONSE: mock_draft_civil_response,
            }
        )
        session = SessionContext()

        events = []
        async for event in loop.run_stream("민원 답변 초안 작성", session):
            events.append(event)

        event_types = [event["type"] for event in events]
        assert event_types[0] == "plan"
        assert "tool_start" in event_types
        assert "tool_result" in event_types
        assert event_types[-1] == "final"
        assert events[-1]["finished"] is True
        assert "최종 초안" in events[-1]["text"]

    @pytest.mark.asyncio
    async def test_stream_reports_tool_failure(self):
        loop = AgentLoop(
            tool_registry={
                ToolType.RAG_SEARCH: mock_failing_tool,
                ToolType.API_LOOKUP: mock_api_lookup,
                ToolType.DRAFT_CIVIL_RESPONSE: mock_draft_civil_response,
            }
        )
        session = SessionContext()

        events = []
        async for event in loop.run_stream("민원 답변 초안 작성", session):
            events.append(event)

        tool_results = [event for event in events if event["type"] == "tool_result"]
        assert tool_results[0]["success"] is False
