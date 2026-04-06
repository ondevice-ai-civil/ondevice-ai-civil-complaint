"""GovOn LangGraph StateGraph 빌더.

Issue #415: LangGraph runtime 기반 및 planner/executor adapter 구성.

`build_govon_graph()` 함수가 6-node StateGraph를 조립하고
컴파일된 graph를 반환한다.

Graph topology:
  START -> session_load -> planner -> approval_wait
               -> [approved] tool_execute -> synthesis -> persist -> END
               -> [rejected] persist -> END
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.utils.runnable import RunnableCallable

from .executor_adapter import ExecutorAdapter
from .nodes import (
    approval_wait_node,
    persist_node,
    planner_node,
    session_load_node,
    synthesis_node,
    tool_execute_node,
)
from .planner_adapter import PlannerAdapter
from .state import ApprovalStatus, GovOnGraphState

if TYPE_CHECKING:
    from src.inference.session_context import SessionStore


def route_after_approval(state: GovOnGraphState) -> str:
    """approval_wait 이후 분기 조건.

    `approval_status` 값에 따라 다음 노드를 결정한다.

    Parameters
    ----------
    state : GovOnGraphState
        현재 graph state.

    Returns
    -------
    str
        "tool_execute" (승인) 또는 "persist" (거절).
    """
    if state.get("approval_status") == ApprovalStatus.APPROVED.value:
        return "tool_execute"
    return "persist"


def build_govon_graph(
    planner_adapter: PlannerAdapter,
    executor_adapter: ExecutorAdapter,
    session_store: "SessionStore",
    checkpointer: Optional[object] = None,
) -> object:
    """GovOn MVP StateGraph를 구성하고 컴파일한다.

    6개 노드를 조립하고 conditional edge로 approval gate를 연결한다.
    checkpointer가 None이면 `MemorySaver`를 사용한다.

    Parameters
    ----------
    planner_adapter : PlannerAdapter
        planner 어댑터 인스턴스.
        운영 환경에서는 `LLMPlannerAdapter`를 사용한다.
        CI 환경에서는 `RegexPlannerAdapter`가 fallback으로 동작한다.
    executor_adapter : ExecutorAdapter
        tool executor 어댑터 인스턴스.
    session_store : SessionStore
        GovOn 세션 저장소. session_load와 persist 노드에서 사용한다.
    checkpointer : optional
        LangGraph checkpoint 저장소.
        None이면 MemorySaver를 사용한다 (메모리에만 저장, 재시작 시 소멸).
        프로덕션에서는 `AsyncSqliteSaver`를 주입한다.

    Returns
    -------
    CompiledGraph
        컴파일된 LangGraph. `ainvoke()`, `aget_state()` 등을 사용할 수 있다.
    """
    from langgraph.checkpoint.memory import MemorySaver

    graph = StateGraph(GovOnGraphState)

    # --- 노드 등록 (closure로 adapter와 session_store 주입) ---

    async def _session_load(state: GovOnGraphState) -> dict:
        return await session_load_node(state, session_store=session_store)

    async def _planner(state: GovOnGraphState) -> dict:
        return await planner_node(state, planner_adapter=planner_adapter)

    async def _tool_execute(state: GovOnGraphState) -> dict:
        return await tool_execute_node(state, executor_adapter=executor_adapter)

    async def _synthesis(state: GovOnGraphState) -> dict:
        return await synthesis_node(state)

    async def _persist(state: GovOnGraphState) -> dict:
        return await persist_node(state, session_store=session_store)

    graph.add_node("session_load", _session_load)
    graph.add_node("planner", _planner)
    # Preserve sync execution for interrupt() on Python 3.10.
    # The default add_node(sync_fn) path auto-generates an async executor wrapper,
    # which breaks LangGraph interrupt context under ainvoke().
    graph.add_node(
        "approval_wait",
        RunnableCallable(approval_wait_node, name="approval_wait"),
    )
    graph.add_node("tool_execute", _tool_execute)
    graph.add_node("synthesis", _synthesis)
    graph.add_node("persist", _persist)

    # --- 엣지 ---
    graph.add_edge(START, "session_load")
    graph.add_edge("session_load", "planner")
    graph.add_edge("planner", "approval_wait")
    graph.add_conditional_edges(
        "approval_wait",
        route_after_approval,
        {
            "tool_execute": "tool_execute",
            "persist": "persist",
        },
    )
    graph.add_edge("tool_execute", "synthesis")
    graph.add_edge("synthesis", "persist")
    graph.add_edge("persist", END)

    # --- 컴파일 ---
    saver = checkpointer if checkpointer is not None else MemorySaver()
    compiled = graph.compile(checkpointer=saver)

    return compiled
