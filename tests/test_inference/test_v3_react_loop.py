"""v3 ReAct LangGraph StateGraph 루프 테스트.

build_govon_graph_v3가 구성하는 ReAct 루프를 검증한다:
- 단일 iteration 완료 (agent → synthesize)
- 멀티 iteration (agent → tools → agent → synthesize)
- max_iterations 강제 종료
- 복수 도구 병렬 실행
- iteration_count 누적 정확성
- tool_call_history 메타데이터 정확성

테스트 격리:
  - StubLLM: 미리 구성된 AIMessage를 순서대로 반환하는 결정적 LLM.
  - make_test_tool(): StructuredTool 인스턴스를 생성.
  - SessionStore: 임시 파일 경로 SQLite DB 사용.
  - MemorySaver: LangGraph 내장 인메모리 checkpointer.

환경 호환성:
  langgraph 설치본이 tool_node.py에서 ServerInfo를 요구하는 경우
  모듈 임포트 전에 stub을 주입하여 우회한다.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# langgraph 환경 호환성 패치
# ---------------------------------------------------------------------------
from unittest.mock import MagicMock

try:
    import langgraph.runtime as _lgrt

    if not hasattr(_lgrt, "ServerInfo"):
        _lgrt.ServerInfo = MagicMock()
except ImportError:
    pass

import json
import tempfile
import uuid
from typing import Any, Dict, List, Optional

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# 기존 test_graph_e2e에서 공통 도구/LLM을 재사용
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver

from tests.test_inference.test_graph_e2e import (
    SimpleInput,
    SimpleToolNode,
    StubLLM,
    make_test_tool,
)

# ---------------------------------------------------------------------------
# 추가 테스트 도구 팩토리
# ---------------------------------------------------------------------------


def make_failing_tool(name: str) -> StructuredTool:
    """success=false를 반환하는 테스트용 도구."""

    async def _execute(query: str) -> str:
        return json.dumps(
            {"success": False, "error": f"{name} 실패: not found"}, ensure_ascii=False
        )

    return StructuredTool.from_function(
        coroutine=_execute,
        name=name,
        description=f"실패 도구: {name}",
        args_schema=SimpleInput,
    )


def make_plain_text_tool(name: str) -> StructuredTool:
    """JSON이 아닌 plain text를 반환하는 도구."""

    async def _execute(query: str) -> str:
        return f"plain text result for {query}"

    return StructuredTool.from_function(
        coroutine=_execute,
        name=name,
        description=f"텍스트 도구: {name}",
        args_schema=SimpleInput,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db_path():
    """임시 SQLite DB 경로."""
    import os

    with tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False) as f:
        path = f.name
    try:
        yield path
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


@pytest.fixture
def session_store(tmp_db_path):
    """임시 SQLite 기반 SessionStore."""
    from src.inference.session_context import SessionStore

    return SessionStore(db_path=tmp_db_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_v3_graph(llm, tools, session_store):
    """테스트용 v3 graph를 구성하고 컴파일한다."""
    import src.inference.graph.builder as _builder
    from src.inference.graph.builder import build_govon_graph_v3

    original_tool_node = _builder.ToolNode
    _builder.ToolNode = SimpleToolNode
    try:
        graph = build_govon_graph_v3(
            llm=llm,
            tools=tools,
            session_store=session_store,
            checkpointer=MemorySaver(),
        )
    finally:
        _builder.ToolNode = original_tool_node

    return graph


def _make_config(thread_id: Optional[str] = None) -> Dict[str, Any]:
    """LangGraph 실행 설정."""
    return {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}


def _initial_state(
    query: str,
    session_id: Optional[str] = None,
    max_iterations: int = 10,
) -> Dict[str, Any]:
    """v3 초기 graph 입력 state."""
    return {
        "session_id": session_id or str(uuid.uuid4()),
        "request_id": str(uuid.uuid4()),
        "messages": [HumanMessage(content=query)],
        "max_iterations": max_iterations,
        "iteration_count": 0,
        "tool_call_history": [],
        "pending_tool_calls": [],
    }


# ---------------------------------------------------------------------------
# TestV3BasicFlow
# ---------------------------------------------------------------------------


class TestV3BasicFlow:
    """v3 기본 ReAct 흐름 검증."""

    @pytest.mark.asyncio
    async def test_no_tool_calls_single_iteration(self, session_store):
        """agent가 첫 호출에서 final answer → tools 미호출 → synthesize.

        흐름: START → session_load → agent → (no tool_calls) → synthesize → END
        """
        plain_response = AIMessage(content="도구 없이 직접 답변합니다.")
        llm = StubLLM(responses=[plain_response])
        tools = [make_test_tool("api_lookup")]

        graph = _make_v3_graph(llm, tools, session_store)
        config = _make_config()
        state = _initial_state("직접 답변 테스트")

        result = await graph.ainvoke(state, config)

        assert result["final_text"] == "도구 없이 직접 답변합니다."
        assert result.get("iteration_count", 0) == 0
        assert result.get("tool_call_history", []) == []
        assert result.get("pending_tool_calls", []) == []

    @pytest.mark.asyncio
    async def test_single_tool_call_multi_iteration(self, session_store):
        """agent → tool_calls → tools(자동) → agent(re-think) → final.

        흐름: agent → (tool_calls) → tools → agent → (no tool_calls) → synthesize → END
        """
        tool_call_response = AIMessage(
            content="",
            tool_calls=[{"name": "api_lookup", "args": {"query": "민원 검색"}, "id": "call_1"}],
        )
        final_response = AIMessage(content="검색 결과를 기반으로 답변합니다.")
        llm = StubLLM(responses=[tool_call_response, final_response])
        tools = [make_test_tool("api_lookup")]

        graph = _make_v3_graph(llm, tools, session_store)
        config = _make_config()
        state = _initial_state("멀티 iteration 테스트")

        result = await graph.ainvoke(state, config)

        assert result["final_text"] == "검색 결과를 기반으로 답변합니다."
        assert result["iteration_count"] == 1
        # tool_call_history에 1건 기록
        history = result.get("tool_call_history", [])
        assert len(history) >= 1
        assert history[0]["tool"] == "api_lookup"
        assert history[0]["success"] is True
        # ToolMessage가 messages에 포함
        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_messages) >= 1


class TestV3MaxIterations:
    """max_iterations 강제 종료 검증."""

    @pytest.mark.asyncio
    async def test_max_iterations_forces_final_answer(self, session_store):
        """iteration_count >= max → synthesize (현재 결과 기반 최선 응답 생성).

        max_iterations=1로 설정하면 첫 tool_call 후 재진입 시 강제 종료.
        """
        tool_call_response = AIMessage(
            content="",
            tool_calls=[{"name": "api_lookup", "args": {"query": "검색"}, "id": "call_max_1"}],
        )
        # 두 번째 agent 호출: max_iterations 도달로 도구 없이 응답 강제
        # tool_calls=[] 명시: StubLLM.bind_tools가 self를 반환하므로 빈 리스트를 직접 지정
        forced_response = AIMessage(content="현재까지 수집한 정보로 답변합니다.", tool_calls=[])
        llm = StubLLM(responses=[tool_call_response, forced_response])
        tools = [make_test_tool("api_lookup")]

        graph = _make_v3_graph(llm, tools, session_store)
        config = _make_config()
        state = _initial_state("max_iterations 테스트", max_iterations=1)

        result = await graph.ainvoke(state, config)

        assert (
            result["final_text"] == "현재까지 수집한 정보로 답변합니다."
        ), f"expected forced final answer, got {result.get('final_text')!r}"
        assert (
            result["iteration_count"] == 1
        ), f"expected iteration_count=1 after forced stop, got {result.get('iteration_count')}"
        # 1회 도구 실행 후 강제 종료
        history = result.get("tool_call_history", [])
        assert len(history) >= 1


class TestV3MultiTool:
    """복수 도구 병렬 실행 검증."""

    @pytest.mark.asyncio
    async def test_multiple_tools_parallel_execution(self, session_store):
        """agent가 2+ 도구를 한번에 호출 → tools 노드가 병렬 실행 → agent."""
        tool_call_response = AIMessage(
            content="",
            tool_calls=[
                {"name": "api_lookup", "args": {"query": "첫 번째"}, "id": "call_m1"},
                {"name": "analysis_tool", "args": {"query": "두 번째"}, "id": "call_m2"},
            ],
        )
        final_response = AIMessage(content="두 도구 결과를 합산하여 답변합니다.")
        llm = StubLLM(responses=[tool_call_response, final_response])
        tools = [
            make_test_tool("api_lookup"),
            make_test_tool("analysis_tool"),
        ]

        graph = _make_v3_graph(llm, tools, session_store)
        config = _make_config()
        state = _initial_state("복수 도구 테스트")

        result = await graph.ainvoke(state, config)

        assert result["final_text"] == "두 도구 결과를 합산하여 답변합니다."
        assert result["iteration_count"] == 1
        # tool_call_history에 2건 기록
        history = result.get("tool_call_history", [])
        assert len(history) == 2
        executed_tools = {h["tool"] for h in history}
        assert "api_lookup" in executed_tools
        assert "analysis_tool" in executed_tools
        # ToolMessage가 2개
        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_messages) >= 2


class TestV3IterationTracking:
    """iteration_count 누적 및 메타데이터 정확성 검증."""

    @pytest.mark.asyncio
    async def test_three_iterations_count_accurate(self, session_store):
        """3회 루프 시 iteration_count가 정확히 3인지 확인."""
        call_1 = AIMessage(
            content="",
            tool_calls=[{"name": "api_lookup", "args": {"query": "1차"}, "id": "c1"}],
        )
        call_2 = AIMessage(
            content="",
            tool_calls=[{"name": "api_lookup", "args": {"query": "2차"}, "id": "c2"}],
        )
        call_3 = AIMessage(
            content="",
            tool_calls=[{"name": "api_lookup", "args": {"query": "3차"}, "id": "c3"}],
        )
        final = AIMessage(content="3번 검색 후 최종 답변.")
        llm = StubLLM(responses=[call_1, call_2, call_3, final])
        tools = [make_test_tool("api_lookup")]

        graph = _make_v3_graph(llm, tools, session_store)
        config = _make_config()
        state = _initial_state("iteration 추적 테스트", max_iterations=10)

        result = await graph.ainvoke(state, config)

        assert result["final_text"] == "3번 검색 후 최종 답변."
        assert result["iteration_count"] == 3
        history = result.get("tool_call_history", [])
        assert len(history) == 3
        # 각 항목의 iteration 번호 확인
        for i, entry in enumerate(history):
            assert entry["iteration"] == i + 1

    @pytest.mark.asyncio
    async def test_tool_call_history_count_matches_execution(self, session_store):
        """tool_call_history 항목 수 == 실제 실행 수."""
        call_1 = AIMessage(
            content="",
            tool_calls=[
                {"name": "api_lookup", "args": {"query": "a"}, "id": "c1"},
                {"name": "analysis_tool", "args": {"query": "b"}, "id": "c2"},
            ],
        )
        final = AIMessage(content="완료.")
        llm = StubLLM(responses=[call_1, final])
        tools = [make_test_tool("api_lookup"), make_test_tool("analysis_tool")]

        graph = _make_v3_graph(llm, tools, session_store)
        config = _make_config()
        state = _initial_state("history count 테스트")

        result = await graph.ainvoke(state, config)

        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        history = result.get("tool_call_history", [])
        assert len(history) == len(tool_messages)


class TestV3NodeLatencies:
    """node_latencies 기록 검증."""

    @pytest.mark.asyncio
    async def test_latencies_recorded(self, session_store):
        """agent, tools, synthesize 노드 레이턴시가 기록된다."""
        call = AIMessage(
            content="",
            tool_calls=[{"name": "api_lookup", "args": {"query": "q"}, "id": "c1"}],
        )
        final = AIMessage(content="답변.")
        llm = StubLLM(responses=[call, final])
        tools = [make_test_tool("api_lookup")]

        graph = _make_v3_graph(llm, tools, session_store)
        config = _make_config()
        state = _initial_state("latency 테스트")

        result = await graph.ainvoke(state, config)

        latencies = result.get("node_latencies", {})
        assert "session_load" in latencies
        assert "agent" in latencies
        assert "tools" in latencies
        assert "persist" in latencies  # synthesize → make_persist_node
        for v in latencies.values():
            assert isinstance(v, float)
            assert v >= 0


class TestV3EmptyToolCalls:
    """빈 tool_calls 엣지 케이스."""

    @pytest.mark.asyncio
    async def test_empty_tool_calls_goes_to_synthesize(self, session_store):
        """agent가 tool_calls 없이 응답 → 즉시 synthesize 진입."""
        response = AIMessage(content="바로 답변합니다.", tool_calls=[])
        llm = StubLLM(responses=[response])
        tools = [make_test_tool("api_lookup")]

        graph = _make_v3_graph(llm, tools, session_store)
        config = _make_config()
        state = _initial_state("빈 tool_calls 테스트")

        result = await graph.ainvoke(state, config)

        assert result["final_text"] == "바로 답변합니다."
        assert result.get("iteration_count", 0) == 0
        assert result.get("tool_call_history", []) == []


# ---------------------------------------------------------------------------
# TestV3ThreadIsolation
# ---------------------------------------------------------------------------


class TestV3ThreadIsolation:
    """thread_id 격리: tool_call_history가 thread별로 독립인지 검증."""

    @pytest.mark.asyncio
    async def test_different_thread_ids_have_independent_history(self, session_store):
        """서로 다른 thread_id로 2회 실행 → 각각 history length=1."""
        call = AIMessage(
            content="",
            tool_calls=[{"name": "api_lookup", "args": {"query": "q"}, "id": "c1"}],
        )
        final = AIMessage(content="답변.")

        # Run 1
        llm1 = StubLLM(responses=[call, final])
        tools = [make_test_tool("api_lookup")]
        graph1 = _make_v3_graph(llm1, tools, session_store)
        config1 = _make_config()  # new thread_id
        state1 = _initial_state("테스트 1")
        result1 = await graph1.ainvoke(state1, config1)

        # Run 2
        llm2 = StubLLM(responses=[call, final])
        graph2 = _make_v3_graph(llm2, tools, session_store)
        config2 = _make_config()  # different thread_id
        state2 = _initial_state("테스트 2")
        result2 = await graph2.ainvoke(state2, config2)

        assert len(result1.get("tool_call_history", [])) == 1
        assert len(result2.get("tool_call_history", [])) == 1

    @pytest.mark.asyncio
    async def test_same_thread_id_accumulates_history(self, session_store):
        """동일 thread_id로 2회 실행 → _append_list로 history 누적됨을 문서화."""
        call = AIMessage(
            content="",
            tool_calls=[{"name": "api_lookup", "args": {"query": "q"}, "id": "c1"}],
        )
        final = AIMessage(content="답변.")

        tools = [make_test_tool("api_lookup")]
        graph = _make_v3_graph(StubLLM(responses=[call, final, call, final]), tools, session_store)
        config = _make_config("shared-thread")

        # Run 1
        state1 = _initial_state("테스트 1")
        result1 = await graph.ainvoke(state1, config)
        assert len(result1.get("tool_call_history", [])) == 1

        # Run 2 with same config (same thread_id → same checkpoint)
        state2 = _initial_state("테스트 2")
        result2 = await graph.ainvoke(state2, config)
        # _append_list causes accumulation
        history = result2.get("tool_call_history", [])
        assert len(history) >= 2, f"Expected accumulation with same thread_id, got {len(history)}"


# ---------------------------------------------------------------------------
# TestV3ToolFailureReporting
# ---------------------------------------------------------------------------


class TestV3ToolFailureReporting:
    """도구 실행 실패 시 tool_call_history 기록 검증."""

    @pytest.mark.asyncio
    async def test_tool_failure_recorded_as_success_false(self, session_store):
        """도구가 {"success": false} 반환 → history.success=False."""
        call = AIMessage(
            content="",
            tool_calls=[{"name": "fail_tool", "args": {"query": "q"}, "id": "c1"}],
        )
        final = AIMessage(content="실패 후 답변.")
        llm = StubLLM(responses=[call, final])
        tools = [make_failing_tool("fail_tool")]

        graph = _make_v3_graph(llm, tools, session_store)
        result = await graph.ainvoke(_initial_state("실패 테스트"), _make_config())

        history = result.get("tool_call_history", [])
        assert len(history) >= 1
        assert history[0]["tool"] == "fail_tool"
        assert history[0]["success"] is False

    @pytest.mark.asyncio
    async def test_tool_non_json_response_defaults_success_true(self, session_store):
        """도구가 plain text 반환 → JSON parse 실패 → success=True (낙관적 기본값)."""
        call = AIMessage(
            content="",
            tool_calls=[{"name": "text_tool", "args": {"query": "q"}, "id": "c1"}],
        )
        final = AIMessage(content="텍스트 도구 후 답변.")
        llm = StubLLM(responses=[call, final])
        tools = [make_plain_text_tool("text_tool")]

        graph = _make_v3_graph(llm, tools, session_store)
        result = await graph.ainvoke(_initial_state("텍스트 테스트"), _make_config())

        history = result.get("tool_call_history", [])
        assert len(history) >= 1
        assert history[0]["tool"] == "text_tool"
        assert history[0]["success"] is True  # optimistic default on parse failure

    @pytest.mark.asyncio
    async def test_tool_exception_creates_error_tool_message(self, session_store):
        """도구 실행 중 예외 → ToolMessage에 error JSON 기록."""
        call = AIMessage(
            content="",
            tool_calls=[{"name": "error_tool", "args": {"query": "q"}, "id": "c1"}],
        )
        final = AIMessage(content="예외 후 답변.")
        llm = StubLLM(responses=[call, final])

        # 예외를 발생시키는 도구
        async def _raise(query: str) -> str:
            raise RuntimeError("도구 내부 오류")

        error_tool = StructuredTool.from_function(
            coroutine=_raise,
            name="error_tool",
            description="예외 도구",
            args_schema=SimpleInput,
        )

        graph = _make_v3_graph(llm, [error_tool], session_store)
        result = await graph.ainvoke(_initial_state("예외 테스트"), _make_config())

        # ToolMessage에 오류가 포함되어 있어야 한다
        tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_msgs) >= 1
        assert "오류" in tool_msgs[0].content or "error" in tool_msgs[0].content.lower()


# ---------------------------------------------------------------------------
# TestV3EvidenceCollection
# ---------------------------------------------------------------------------


class TestV3EvidenceCollection:
    """synthesize 노드의 evidence 수집 검증."""

    @pytest.mark.asyncio
    async def test_evidence_items_collected_from_tool_messages(self, session_store):
        """ToolMessage의 evidence.items가 evidence_items로 수집된다."""
        call = AIMessage(
            content="",
            tool_calls=[{"name": "api_lookup", "args": {"query": "민원"}, "id": "c1"}],
        )
        final = AIMessage(content="근거 기반 답변.")
        llm = StubLLM(responses=[call, final])
        tools = [make_test_tool("api_lookup")]

        graph = _make_v3_graph(llm, tools, session_store)
        result = await graph.ainvoke(_initial_state("evidence 테스트"), _make_config())

        evidence = result.get("evidence_items", [])
        assert len(evidence) >= 1
        assert evidence[0].get("source") == "api_lookup"

    @pytest.mark.asyncio
    async def test_evidence_max_10_items(self, session_store):
        """evidence_items는 최대 10개로 잘린다."""

        # 15개의 evidence item을 반환하는 도구
        async def _many_evidence(query: str) -> str:
            items = [
                {"source": f"src_{i}", "text": f"evidence {i}", "score": 0.9} for i in range(15)
            ]
            return json.dumps(
                {
                    "success": True,
                    "context_text": "many results",
                    "evidence": {"items": items},
                },
                ensure_ascii=False,
            )

        many_tool = StructuredTool.from_function(
            coroutine=_many_evidence,
            name="many_tool",
            description="다수 결과 도구",
            args_schema=SimpleInput,
        )

        call = AIMessage(
            content="",
            tool_calls=[{"name": "many_tool", "args": {"query": "q"}, "id": "c1"}],
        )
        final = AIMessage(content="결과 답변.")
        llm = StubLLM(responses=[call, final])

        graph = _make_v3_graph(llm, [many_tool], session_store)
        result = await graph.ainvoke(_initial_state("max evidence 테스트"), _make_config())

        evidence = result.get("evidence_items", [])
        assert len(evidence) <= 10, f"Expected max 10 evidence items, got {len(evidence)}"
