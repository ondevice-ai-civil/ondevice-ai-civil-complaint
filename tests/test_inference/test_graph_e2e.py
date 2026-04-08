"""v4 ReAct LangGraph StateGraph E2E 테스트.

GovOn의 graph/builder.py가 구성하는 StateGraph를 실제로 컴파일하고
실행하여 각 분기(no-tool-call, Tier0, approval, 거절, 취소)를 검증한다.

테스트 격리:
  - StubLLM: 미리 구성된 AIMessage를 순서대로 반환하는 결정적 LLM.
  - make_test_tool(): StructuredTool 인스턴스를 생성. metadata에 requires_approval 저장.
  - SessionStore: 임시 파일 경로 SQLite DB 사용.
  - MemorySaver: LangGraph 내장 인메모리 checkpointer.
  - conftest.py가 vllm/sentence_transformers/retriever/database.py 등을 이미 mock 처리.

환경 호환성:
  langgraph 설치본이 tool_node.py에서 ServerInfo를 요구하는 경우
  (패키지 파일 혼합 상태) 모듈 임포트 전에 stub을 주입하여 우회한다.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# langgraph 환경 호환성 패치
# ServerInfo가 langgraph.runtime에 없는 혼합 설치 환경에서
# tool_node.py 임포트가 실패하는 경우를 대비해 stub을 먼저 주입한다.
# ---------------------------------------------------------------------------
import sys
from unittest.mock import MagicMock

_langgraph_runtime = sys.modules.get("langgraph.runtime")
if _langgraph_runtime is not None and not hasattr(_langgraph_runtime, "ServerInfo"):
    _langgraph_runtime.ServerInfo = MagicMock()
else:
    # 아직 로드되지 않은 경우를 위해 직접 임포트하여 패치
    try:
        import langgraph.runtime as _lgrt

        if not hasattr(_lgrt, "ServerInfo"):
            _lgrt.ServerInfo = MagicMock()
    except ImportError:
        pass

import asyncio
import json
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Sequence

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# SimpleToolNode — 환경 호환성 문제를 우회하는 경량 ToolNode 대체 구현
#
# langgraph 설치본의 ToolNode가 런타임 config 인프라를 요구하여
# 독립 실행 시 동작하지 않는 환경에서 사용한다.
# build_govon_graph 호출 전 builder.ToolNode를 이 클래스로 교체한다.
# ---------------------------------------------------------------------------


def _make_simple_tool_node(tools: list):
    """ToolNode 대체 비동기 노드 함수를 생성한다.

    LangGraph StateGraph에 등록 가능한 async callable을 반환한다.
    AIMessage.tool_calls에 있는 모든 도구를 순차 실행하고
    각 결과를 ToolMessage로 반환한다.

    langgraph 설치본의 ToolNode가 런타임 config 인프라를 요구하여
    독립 실행 시 동작하지 않는 환경에서 사용한다.
    """
    tool_map = {t.name: t for t in tools}

    async def tool_node(state: dict) -> dict:
        messages = state.get("messages", [])
        if not messages:
            return {"messages": []}

        last = messages[-1]
        tool_calls = getattr(last, "tool_calls", []) or []

        async def _run_tool(tc):
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_id = tc.get("id", f"call_{tool_name}")
            tool = tool_map.get(tool_name)
            if tool is None:
                content = json.dumps({"error": f"도구를 찾을 수 없습니다: {tool_name}"})
            else:
                try:
                    # StructuredTool.coroutine는 async callable이다
                    coro_fn = getattr(tool, "coroutine", None) or getattr(tool, "func", None)
                    if coro_fn and asyncio.iscoroutinefunction(coro_fn):
                        raw = await coro_fn(**tool_args)
                    else:
                        raw = await tool.ainvoke(tool_args)
                    content = raw if isinstance(raw, str) else json.dumps(raw)
                except Exception as exc:
                    content = json.dumps({"error": str(exc)})
            return ToolMessage(content=content, tool_call_id=tool_id, name=tool_name)

        tool_messages = await asyncio.gather(*[_run_tool(tc) for tc in tool_calls])
        return {"messages": list(tool_messages)}

    return tool_node


# SimpleToolNode 클래스 별칭 — builder.ToolNode 교체 시 사용
# builder.py는 ToolNode(tools) 형태로 인스턴스화하므로
# 클래스처럼 호출하면 노드 함수를 반환하는 callable로 구현한다.
class SimpleToolNode:
    """builder.ToolNode 교체용 callable 클래스.

    ToolNode(tools) 호출 시 LangGraph 노드로 등록 가능한 async 함수를 반환한다.
    """

    def __new__(cls, tools: list):
        """ToolNode(tools) 패턴을 지원하는 async callable을 반환한다."""
        return _make_simple_tool_node(tools)


# ---------------------------------------------------------------------------
# StubLLM — 결정적 응답을 순서대로 반환하는 테스트용 LLM
# ---------------------------------------------------------------------------


class StubLLM(BaseChatModel):
    """미리 구성된 AIMessage 목록을 순서대로 반환하는 결정적 LLM.

    bind_tools() 호출 시 self를 반환하여 실제 도구 바인딩을 건너뛴다.
    responses 목록이 소진되면 마지막 응답을 반복 반환한다.
    """

    responses: List[Any]
    call_count: int = 0

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "stub"

    def _generate(
        self,
        messages,
        stop=None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        idx = min(self.call_count, len(self.responses) - 1)
        msg = self.responses[idx]
        self.call_count += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(
        self,
        messages,
        stop=None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        return self._generate(messages, stop, run_manager, **kwargs)

    def bind_tools(self, tools, **kwargs):
        """실제 도구 바인딩 대신 self를 그대로 반환한다.

        StubLLM의 응답은 이미 tool_calls 여부가 사전에 설정되어 있으므로
        bind_tools 결과가 응답에 영향을 주지 않는다.
        """
        return self


# ---------------------------------------------------------------------------
# 테스트 도구 팩토리
# ---------------------------------------------------------------------------


class SimpleInput(BaseModel):
    """단순 도구 입력 스키마."""

    query: str = Field(description="검색 쿼리")


def make_test_tool(name: str, requires_approval: bool = False) -> StructuredTool:
    """테스트용 StructuredTool을 생성한다.

    Parameters
    ----------
    name : str
        도구 이름.
    requires_approval : bool
        True이면 approval_wait 노드로 라우팅된다.

    Returns
    -------
    StructuredTool
        도구 인스턴스.
    """

    async def _execute(query: str) -> str:
        result = {
            "success": True,
            "context_text": f"{name} result for: {query}",
            "evidence": {"items": [{"source": name, "text": f"{name} 근거 텍스트", "score": 0.9}]},
        }
        return json.dumps(result, ensure_ascii=False)

    return StructuredTool.from_function(
        coroutine=_execute,
        name=name,
        description=f"테스트 도구: {name}",
        args_schema=SimpleInput,
        metadata={"requires_approval": requires_approval},
    )


# ---------------------------------------------------------------------------
# 공통 fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db_path():
    """임시 SQLite DB 경로를 생성하고 테스트 후 삭제한다."""
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
    """임시 SQLite 기반 SessionStore 인스턴스를 반환한다."""
    from src.inference.session_context import SessionStore

    return SessionStore(db_path=tmp_db_path)


def _make_graph(llm, tools, session_store):
    """테스트용 그래프를 구성하고 컴파일한다.

    MemorySaver를 checkpointer로 사용하여 인터럽트/재개를 지원한다.
    langgraph 환경 호환성 문제로 인해 builder.ToolNode를
    SimpleToolNode로 교체한 후 build_govon_graph를 호출한다.
    """
    import src.inference.graph.builder as _builder
    from src.inference.graph.builder import build_govon_graph

    # 환경 내 ToolNode가 런타임 config 인프라를 요구하여 독립 실행 불가능한 경우
    # SimpleToolNode로 교체하여 도구 실행을 정상 수행한다.
    original_tool_node = _builder.ToolNode
    _builder.ToolNode = SimpleToolNode
    try:
        checkpointer = MemorySaver()
        graph = build_govon_graph(
            llm=llm,
            tools=tools,
            session_store=session_store,
            checkpointer=checkpointer,
        )
    finally:
        _builder.ToolNode = original_tool_node

    return graph


def _make_config(thread_id: Optional[str] = None) -> Dict[str, Any]:
    """LangGraph 실행 설정(config)을 생성한다."""
    return {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}


async def _invoke_expecting_interrupt(graph, state_or_command, config) -> dict:
    """graph.ainvoke를 호출하고 interrupt가 발생했는지 확인한다.

    langgraph 버전에 따라 interrupt()는:
    - 구버전(≤0.2): GraphInterrupt를 raise
    - 신버전(1.1.x): __interrupt__ 키를 포함한 dict를 반환

    두 경우 모두 처리하며, interrupt가 발생한 result dict를 반환한다.
    interrupt가 발생하지 않은 경우 AssertionError를 raise한다.
    """
    from langgraph.errors import GraphInterrupt

    try:
        result = await graph.ainvoke(state_or_command, config)
    except GraphInterrupt:
        # 구버전 동작: state에서 현재 값 조회
        graph_state = await graph.aget_state(config)
        return dict(graph_state.values)

    if "__interrupt__" in result and result["__interrupt__"]:
        return result

    raise AssertionError(
        "interrupt가 발생해야 하지만 그래프가 정상 완료되었다. "
        f"결과: final_text={result.get('final_text')}, "
        f"approval_status={result.get('approval_status')}"
    )


def _initial_state(query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """초기 graph 입력 state를 생성한다."""
    return {
        "session_id": session_id or str(uuid.uuid4()),
        "request_id": str(uuid.uuid4()),
        "messages": [HumanMessage(content=query)],
    }


# ---------------------------------------------------------------------------
# TestGraphBasicFlow
# ---------------------------------------------------------------------------


class TestGraphBasicFlow:
    """기본 graph 흐름 검증."""

    @pytest.mark.asyncio
    async def test_no_tool_calls_goes_to_persist(self, session_store):
        """도구 호출 없는 순수 텍스트 응답 시 persist 노드를 통해 final_text가 설정된다.

        흐름: START → session_load → agent → (no tool_calls) → persist → END
        """
        plain_response = AIMessage(content="도구 없이 직접 답변합니다.")
        llm = StubLLM(responses=[plain_response])
        tools = [make_test_tool("rag_search", requires_approval=False)]

        graph = _make_graph(llm, tools, session_store)
        config = _make_config()
        state = _initial_state("직접 답변 테스트")

        result = await graph.ainvoke(state, config)

        assert result["final_text"] == "도구 없이 직접 답변합니다."
        assert isinstance(result["messages"], list)
        # 메시지 목록: HumanMessage(초기) + AIMessage(응답)
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert len(ai_messages) >= 1
        assert ai_messages[-1].content == "도구 없이 직접 답변합니다."

    @pytest.mark.asyncio
    async def test_tier0_tool_calls_skip_approval(self, session_store):
        """requires_approval=False인 도구 호출은 approval_wait를 거치지 않고 바로 실행된다.

        흐름: agent → (Tier0 tool_calls) → tools → agent → persist → END
        """
        tool_call_response = AIMessage(
            content="",
            tool_calls=[{"name": "rag_search", "args": {"query": "민원 검색"}, "id": "call_t0_1"}],
        )
        final_response = AIMessage(content="검색 결과를 기반으로 답변합니다.")
        llm = StubLLM(responses=[tool_call_response, final_response])
        tools = [make_test_tool("rag_search", requires_approval=False)]

        graph = _make_graph(llm, tools, session_store)
        config = _make_config()
        state = _initial_state("Tier0 도구 테스트")

        # GraphInterrupt 없이 완료되어야 한다
        result = await graph.ainvoke(state, config)

        assert result["final_text"] == "검색 결과를 기반으로 답변합니다."
        # ToolMessage가 messages에 포함되어 있어야 한다
        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_messages) >= 1
        assert tool_messages[0].name == "rag_search"

    @pytest.mark.asyncio
    async def test_approval_required_tool_interrupts(self, session_store):
        """requires_approval=True인 도구 호출 시 approval_wait에서 interrupt가 발생한다.

        흐름: agent → (approval tool_calls) → approval_wait → interrupt()
        langgraph 1.1.x: interrupt()는 GraphInterrupt를 raise하거나
        결과 dict에 __interrupt__ 키를 포함한다. 두 경우 모두 처리한다.
        interrupt 발생 후 graph.aget_state()의 next에 노드가 포함된다.
        """
        tool_call_response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "sensitive_tool",
                    "args": {"query": "민감 작업"},
                    "id": "call_approval_1",
                }
            ],
        )
        llm = StubLLM(responses=[tool_call_response])
        tools = [make_test_tool("sensitive_tool", requires_approval=True)]

        graph = _make_graph(llm, tools, session_store)
        config = _make_config()
        state = _initial_state("승인 필요 도구 테스트")

        # _invoke_expecting_interrupt: 버전 호환 방식으로 interrupt 발생 확인
        await _invoke_expecting_interrupt(graph, state, config)

        # interrupt 후 graph 상태를 확인한다
        graph_state = await graph.aget_state(config)
        assert graph_state.next  # 대기 중인 노드가 있어야 한다

    @pytest.mark.asyncio
    async def test_approve_resumes_to_tools(self, session_store):
        """interrupt 후 approved=True로 재개하면 tools → agent → persist로 완료된다.

        흐름: (interrupted) → Command(resume=approved) → tools → agent → persist → END
        """
        from langgraph.types import Command

        tool_call_response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "sensitive_tool",
                    "args": {"query": "승인 후 실행"},
                    "id": "call_approve_1",
                }
            ],
        )
        final_response = AIMessage(content="승인 후 도구를 실행하여 답변합니다.")
        llm = StubLLM(responses=[tool_call_response, final_response])
        tools = [make_test_tool("sensitive_tool", requires_approval=True)]

        graph = _make_graph(llm, tools, session_store)
        config = _make_config()
        state = _initial_state("승인 후 재개 테스트")

        # 첫 번째 호출: interrupt 발생 확인
        await _invoke_expecting_interrupt(graph, state, config)

        # 승인으로 재개
        result = await graph.ainvoke(Command(resume={"approved": True}), config)

        assert result["final_text"] == "승인 후 도구를 실행하여 답변합니다."
        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_messages) >= 1
        assert tool_messages[0].name == "sensitive_tool"

    @pytest.mark.asyncio
    async def test_reject_goes_back_to_agent(self, session_store):
        """interrupt 후 approved=False로 재개하면 agent 노드로 돌아가 대안을 제시한다.

        흐름: (interrupted) → Command(resume=rejected) → agent → persist → END
        approval_wait_node는 거절 메시지(HumanMessage)를 messages에 추가한다.
        """
        from langgraph.types import Command

        tool_call_response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "sensitive_tool",
                    "args": {"query": "거절될 작업"},
                    "id": "call_reject_1",
                }
            ],
        )
        # 두 번째 agent 호출: 거절 후 도구 없이 직접 답변
        fallback_response = AIMessage(content="도구 없이 직접 답변드립니다.")
        llm = StubLLM(responses=[tool_call_response, fallback_response])
        tools = [make_test_tool("sensitive_tool", requires_approval=True)]

        graph = _make_graph(llm, tools, session_store)
        config = _make_config()
        state = _initial_state("거절 테스트")

        # interrupt 발생 확인
        await _invoke_expecting_interrupt(graph, state, config)

        # 거절로 재개
        result = await graph.ainvoke(Command(resume={"approved": False}), config)

        assert result["final_text"] == "도구 없이 직접 답변드립니다."
        # 거절 메시지가 messages에 포함되어 있어야 한다
        human_messages = [m for m in result["messages"] if isinstance(m, HumanMessage)]
        rejection_msgs = [m for m in human_messages if "거부했습니다" in (m.content or "")]
        assert len(rejection_msgs) >= 1

    @pytest.mark.asyncio
    async def test_cancel_goes_to_persist(self, session_store):
        """interrupt 후 cancel=True로 재개하면 persist로 바로 이동한다.

        흐름: (interrupted) → Command(resume=cancel) → persist → END
        approval_status가 "cancelled"로 설정된다.
        """
        from langgraph.types import Command

        tool_call_response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "sensitive_tool",
                    "args": {"query": "취소될 작업"},
                    "id": "call_cancel_1",
                }
            ],
        )
        llm = StubLLM(responses=[tool_call_response])
        tools = [make_test_tool("sensitive_tool", requires_approval=True)]

        graph = _make_graph(llm, tools, session_store)
        config = _make_config()
        state = _initial_state("취소 테스트")

        # interrupt 발생 확인
        await _invoke_expecting_interrupt(graph, state, config)

        # 취소로 재개
        result = await graph.ainvoke(Command(resume={"cancel": True}), config)

        assert result["approval_status"] == "cancelled"
        # persist가 실행되어 node_latencies에 persist 레이턴시가 기록되어야 한다
        assert "persist" in result.get("node_latencies", {})


# ---------------------------------------------------------------------------
# TestGraphMultiTool
# ---------------------------------------------------------------------------


class TestGraphMultiTool:
    """복수 도구 호출 시나리오 검증."""

    @pytest.mark.asyncio
    async def test_multiple_tier0_tools_all_execute(self, session_store):
        """LLM이 Tier0 도구 2개를 동시에 호출하면 둘 다 실행된다.

        LangGraph ToolNode는 AIMessage.tool_calls에 있는 모든 도구를 병렬 실행한다.
        """
        tool_call_response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "rag_search",
                    "args": {"query": "첫 번째 검색"},
                    "id": "call_multi_1",
                },
                {
                    "name": "analysis_tool",
                    "args": {"query": "두 번째 분석"},
                    "id": "call_multi_2",
                },
            ],
        )
        final_response = AIMessage(content="두 도구 결과를 합산하여 답변합니다.")
        llm = StubLLM(responses=[tool_call_response, final_response])
        tools = [
            make_test_tool("rag_search", requires_approval=False),
            make_test_tool("analysis_tool", requires_approval=False),
        ]

        graph = _make_graph(llm, tools, session_store)
        config = _make_config()
        state = _initial_state("복수 도구 테스트")

        result = await graph.ainvoke(state, config)

        assert result["final_text"] == "두 도구 결과를 합산하여 답변합니다."
        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        # rag_search와 analysis_tool 모두 실행되어야 한다
        executed_tools = {m.name for m in tool_messages}
        assert "rag_search" in executed_tools
        assert "analysis_tool" in executed_tools

    @pytest.mark.asyncio
    async def test_mixed_approval_tiers(self, session_store):
        """Tier0 도구와 approval 필요 도구를 함께 호출하면 approval_wait가 트리거된다.

        tool_calls 중 하나라도 requires_approval=True이면 approval_wait로 라우팅된다.
        """
        mixed_tool_call_response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "rag_search",
                    "args": {"query": "공개 검색"},
                    "id": "call_mix_1",
                },
                {
                    "name": "sensitive_tool",
                    "args": {"query": "민감 작업"},
                    "id": "call_mix_2",
                },
            ],
        )
        llm = StubLLM(responses=[mixed_tool_call_response])
        tools = [
            make_test_tool("rag_search", requires_approval=False),
            make_test_tool("sensitive_tool", requires_approval=True),
        ]

        graph = _make_graph(llm, tools, session_store)
        config = _make_config()
        state = _initial_state("혼합 tier 테스트")

        # approval_wait에서 interrupt가 발생해야 한다
        await _invoke_expecting_interrupt(graph, state, config)

        graph_state = await graph.aget_state(config)
        # interrupt 후 아직 처리할 노드가 남아 있어야 한다
        assert graph_state.next


# ---------------------------------------------------------------------------
# TestGraphEdgeCases
# ---------------------------------------------------------------------------


class TestGraphEdgeCases:
    """엣지 케이스 및 특수 시나리오 검증."""

    @pytest.mark.asyncio
    async def test_consecutive_rejections_force_direct_response(self, session_store):
        """연속 2회 거절 후 agent가 도구 호출 없이 직접 응답한다.

        nodes.py의 _count_consecutive_rejections이 _MAX_CONSECUTIVE_REJECTIONS(2) 이상이면
        llm_without_tools를 사용하여 순수 텍스트 응답을 보장한다.
        """
        from langgraph.types import Command

        tool_call_1 = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "sensitive_tool",
                    "args": {"query": "첫 번째 시도"},
                    "id": "call_rej_1",
                }
            ],
        )
        tool_call_2 = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "sensitive_tool",
                    "args": {"query": "두 번째 시도"},
                    "id": "call_rej_2",
                }
            ],
        )
        # 두 번 거절 후 agent는 직접 답변을 생성한다
        direct_response = AIMessage(content="도구 없이 직접 답변드립니다.")
        llm = StubLLM(responses=[tool_call_1, tool_call_2, direct_response])
        tools = [make_test_tool("sensitive_tool", requires_approval=True)]

        graph = _make_graph(llm, tools, session_store)
        config = _make_config()
        state = _initial_state("연속 거절 테스트")

        # 첫 번째 interrupt 발생 확인
        await _invoke_expecting_interrupt(graph, state, config)

        # 첫 번째 거절 → 두 번째 agent 호출에서 다시 tool_call → interrupt
        await _invoke_expecting_interrupt(graph, Command(resume={"approved": False}), config)

        # 두 번째 거절 → agent가 연속 거절 감지 → 직접 응답
        result = await graph.ainvoke(Command(resume={"approved": False}), config)

        assert result["final_text"] == "도구 없이 직접 답변드립니다."
        # 최종 AIMessage에 tool_calls가 없어야 한다
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        last_ai = ai_messages[-1]
        assert not getattr(last_ai, "tool_calls", None)

    @pytest.mark.asyncio
    async def test_session_persist_stores_final_text(self, session_store):
        """그래프 완료 후 state의 final_text가 정확히 설정된다.

        persist_node는 messages에서 마지막 tool_calls 없는 AIMessage의 content를
        final_text로 추출한다.
        """
        expected_text = "최종 답변 텍스트입니다."
        plain_response = AIMessage(content=expected_text)
        llm = StubLLM(responses=[plain_response])
        tools = [make_test_tool("rag_search", requires_approval=False)]

        graph = _make_graph(llm, tools, session_store)
        config = _make_config()
        state = _initial_state("세션 저장 테스트")

        result = await graph.ainvoke(state, config)

        assert result["final_text"] == expected_text
        # node_latencies에 persist와 agent가 모두 기록되어야 한다
        latencies = result.get("node_latencies", {})
        assert "persist" in latencies
        assert "agent" in latencies
        assert "session_load" in latencies

    @pytest.mark.asyncio
    async def test_evidence_collected_from_tool_messages(self, session_store):
        """ToolMessage에 evidence JSON이 포함된 경우 persist_node가 evidence_items를 수집한다.

        make_test_tool의 실행 결과에는 evidence.items가 포함되어 있으며,
        persist_node가 이를 state의 evidence_items에 누적한다.
        """
        tool_call_response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "rag_search",
                    "args": {"query": "근거 수집 테스트"},
                    "id": "call_ev_1",
                }
            ],
        )
        final_response = AIMessage(content="근거를 포함하여 답변합니다.")
        llm = StubLLM(responses=[tool_call_response, final_response])
        tools = [make_test_tool("rag_search", requires_approval=False)]

        graph = _make_graph(llm, tools, session_store)
        config = _make_config()
        state = _initial_state("evidence 수집 테스트")

        result = await graph.ainvoke(state, config)

        # evidence_items가 수집되어야 한다
        evidence = result.get("evidence_items", [])
        assert len(evidence) >= 1
        # 각 item에 source 필드가 있어야 한다
        assert evidence[0].get("source") == "rag_search"
        assert evidence[0].get("text") == "rag_search 근거 텍스트"
