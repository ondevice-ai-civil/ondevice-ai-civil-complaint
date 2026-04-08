"""GovOn LangGraph StateGraph 빌더.

v4 아키텍처: ReAct + ToolNode 기반.
LLM이 자율적으로 도구 호출을 결정하며, 정적 planner/executor를 제거한다.

Graph topology:
  START → session_load → agent → [route_agent]
       ├── (no tool_calls)   → persist → END
       ├── (all Tier 0)      → tools → agent → ...
       └── (needs approval)  → approval_wait → [route_after_approval]
                                   ├── (approved)  → tools → agent → ...
                                   ├── (cancelled) → persist → END
                                   └── (rejected)  → agent → ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .nodes import (
    make_agent_node,
    make_agent_node_v3,
    make_approval_wait_node,
    make_persist_node,
    make_session_load_node,
    make_tools_node_v3,
    route_after_approval,
)
from .state import ApprovalStatus, GovOnGraphState

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from src.inference.session_context import SessionStore


def _make_route_agent(approval_map: dict[str, bool]):
    """agent 노드 이후 라우팅 함수를 생성한다.

    Parameters
    ----------
    approval_map : dict[str, bool]
        {tool_name: requires_approval} 매핑.

    Returns
    -------
    Callable
        state를 받아 다음 노드 이름을 반환하는 라우팅 함수.
    """

    def route_agent(state: GovOnGraphState) -> str:
        messages = state.get("messages", [])
        if not messages:
            return "persist"

        last = messages[-1]
        tool_calls = getattr(last, "tool_calls", None)
        if not tool_calls:
            return "persist"

        needs_approval = any(approval_map.get(tc["name"], False) for tc in tool_calls)
        return "approval_wait" if needs_approval else "tools"

    return route_agent


def build_govon_graph(
    *,
    llm,
    tools: list,
    session_store: "SessionStore",
    checkpointer: Optional[object] = None,
) -> "CompiledStateGraph":
    """GovOn ReAct StateGraph를 구성하고 컴파일한다.

    Parameters
    ----------
    llm : BaseChatModel
        도구 바인딩 가능한 LLM 인스턴스 (ChatOpenAI 등).
    tools : list[StructuredTool]
        build_all_tools()로 생성된 도구 목록.
    session_store : SessionStore
        세션 저장소. session_load / persist 노드에서 사용.
    checkpointer : optional
        LangGraph checkpoint 저장소.
        None이면 MemorySaver를 사용한다.

    Returns
    -------
    CompiledStateGraph
        컴파일된 LangGraph.
    """
    from langgraph.checkpoint.memory import MemorySaver

    from .tools import get_tool_approval_map

    tool_node = ToolNode(tools)
    approval_map = get_tool_approval_map(tools)

    graph = StateGraph(GovOnGraphState)

    # --- 노드 등록 ---
    graph.add_node("session_load", make_session_load_node(session_store))
    graph.add_node("agent", make_agent_node(llm, tools))
    graph.add_node("approval_wait", make_approval_wait_node(approval_map))
    graph.add_node("tools", tool_node)
    graph.add_node("persist", make_persist_node(session_store))

    # --- 엣지 ---
    graph.add_edge(START, "session_load")
    graph.add_edge("session_load", "agent")
    graph.add_conditional_edges(
        "agent",
        _make_route_agent(approval_map),
        {
            "tools": "tools",
            "approval_wait": "approval_wait",
            "persist": "persist",
        },
    )
    graph.add_conditional_edges(
        "approval_wait",
        route_after_approval,
        {
            "tools": "tools",
            "agent": "agent",
            "persist": "persist",
        },
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("persist", END)

    # --- 컴파일 ---
    saver = checkpointer if checkpointer is not None else MemorySaver()
    return graph.compile(checkpointer=saver)


# ---------------------------------------------------------------------------
# v3 ReAct graph builder
# ---------------------------------------------------------------------------


def _route_agent_v3(state: GovOnGraphState) -> str:
    """v3 agent 노드 이후 라우팅.

    1. pending_tool_calls 존재 → "tools" (자동 실행)
    2. pending_tool_calls 비어있음 (final answer 또는 강제 종료) → "synthesize"

    max_iterations 도달 시 강제 종료는 agent_node_v3 내부에서 처리한다.
    agent가 pending_tool_calls=[]를 반환하면 synthesize로 이동한다.

    현재 모든 도구가 low risk이므로 approval 분기 없음.
    향후 고위험 도구 추가 시 여기에 분기 조건을 추가한다.
    """
    pending = state.get("pending_tool_calls", [])
    if pending:
        return "tools"
    return "synthesize"


def build_govon_graph_v3(
    *,
    llm,
    tools: list,
    session_store: "SessionStore",
    checkpointer: Optional[object] = None,
) -> "CompiledStateGraph":
    """v3 ReAct StateGraph를 구성하고 컴파일한다.

    v2와의 차이:
    - agent 노드가 iteration_count를 추적하고 max_iterations에서 강제 종료
    - tools 노드가 tool_call_history에 실행 메타데이터를 기록
    - approval_wait 노드 없음 (모든 도구 자동 실행)
    - synthesize 노드가 persist의 역할을 겸함

    Graph topology:
      START → session_load → agent → [route_agent_v3]
           ├── (no tool_calls / max_iter)  → synthesize → END
           └── (tool_calls 존재)            → tools → agent → ...
    """
    from langgraph.checkpoint.memory import MemorySaver

    tool_node = ToolNode(tools)

    graph = StateGraph(GovOnGraphState)

    # --- 노드 등록 ---
    graph.add_node("session_load", make_session_load_node(session_store))
    graph.add_node("agent", make_agent_node_v3(llm, tools))
    graph.add_node("tools", make_tools_node_v3(tool_node))
    graph.add_node("synthesize", make_persist_node(session_store))

    # --- 엣지 ---
    graph.add_edge(START, "session_load")
    graph.add_edge("session_load", "agent")
    graph.add_conditional_edges(
        "agent",
        _route_agent_v3,
        {
            "tools": "tools",
            "synthesize": "synthesize",
        },
    )
    graph.add_edge("tools", "agent")  # ReAct 루프 백엣지
    graph.add_edge("synthesize", END)

    # --- 컴파일 ---
    saver = checkpointer if checkpointer is not None else MemorySaver()
    return graph.compile(checkpointer=saver)
