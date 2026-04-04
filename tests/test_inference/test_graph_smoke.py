"""LangGraph graph 초기화 및 기본 사이클 smoke test.

Issue #415: LangGraph runtime 기반 및 planner/executor adapter 구성.

SKIP_MODEL_LOAD=true 환경에서 LLM 없이 실행 가능해야 한다.
4개 시나리오:
  1. graph 컴파일 테스트 (노드 존재 확인)
  2. interrupt 동작 테스트 (approval_wait에서 멈추는지)
  3. 승인 경로 테스트 (approved=True -> tool_execute 진행)
  4. 거절 경로 테스트 (approved=False -> END)
"""

from __future__ import annotations

import os
import tempfile

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.inference.graph.builder import build_govon_graph
from src.inference.graph.executor_adapter import ExecutorAdapter
from src.inference.graph.planner_adapter import RegexPlannerAdapter
from src.inference.graph.state import ApprovalStatus, GovOnGraphState
from src.inference.session_context import SessionStore

os.environ.setdefault("SKIP_MODEL_LOAD", "true")


class StubExecutorAdapter(ExecutorAdapter):
    """테스트용 스텁 executor.

    모든 tool 호출에 고정된 성공 결과를 반환한다.
    LLM이나 외부 API 없이도 graph를 완전히 실행할 수 있게 한다.
    """

    async def execute(
        self,
        tool_name: str,
        query: str,
        context: dict,
    ) -> dict:
        return {
            "success": True,
            "text": f"[stub] {tool_name} result for: {query}",
            "latency_ms": 1.0,
        }

    def list_tools(self) -> list[str]:
        return ["rag_search", "api_lookup", "draft_civil_response", "append_evidence"]


@pytest.fixture
def session_store(tmp_path):
    """임시 디렉터리에 SessionStore를 생성한다.

    각 테스트마다 격리된 SQLite 파일을 사용한다.
    """
    db_path = str(tmp_path / "test_sessions.sqlite3")
    return SessionStore(db_path=db_path)


@pytest.fixture
def graph(session_store):
    """RegexPlannerAdapter + StubExecutorAdapter로 graph를 구성한다.

    MemorySaver를 checkpointer로 사용하므로 LangGraph checkpoint 파일이 생성되지 않는다.
    임시 SessionStore를 사용하여 테스트 간 격리를 보장한다.
    """
    planner = RegexPlannerAdapter()
    executor = StubExecutorAdapter()
    return build_govon_graph(
        planner_adapter=planner,
        executor_adapter=executor,
        session_store=session_store,
        checkpointer=MemorySaver(),
    )


class TestGraphSmoke:
    """Graph 초기화 및 기본 사이클 smoke test."""

    def test_graph_compiles(self, graph):
        """graph가 에러 없이 컴파일된다."""
        assert graph is not None

    def test_graph_has_expected_nodes(self, graph):
        """graph에 6개 필수 노드가 존재한다."""
        node_names = set(graph.get_graph().nodes.keys())
        expected = {
            "session_load",
            "planner",
            "approval_wait",
            "tool_execute",
            "synthesis",
            "persist",
        }
        # START와 END는 별도이므로 expected가 node_names의 부분집합인지 확인
        assert expected.issubset(node_names), f"누락된 노드: {expected - node_names}"

    def test_graph_runs_to_approval_interrupt(self, graph):
        """graph가 approval_wait 노드에서 interrupt된다.

        planner 실행 후 graph가 approval_wait에서 멈추고
        graph_state.next가 비어있지 않아야 한다.
        """
        config = {"configurable": {"thread_id": "smoke-interrupt-1"}}
        initial = {
            "session_id": "test-session-interrupt",
            "request_id": "test-request-interrupt",
            "messages": [HumanMessage(content="이 민원에 대한 답변 초안 작성해줘")],
        }

        graph.invoke(initial, config=config)

        # interrupt 상태 확인: graph가 approval_wait에서 멈춰있어야 한다
        state = graph.get_state(config)
        assert state.next, "graph가 approval_wait에서 interrupt되어야 합니다"

    def test_graph_completes_after_approval(self, graph):
        """승인 후 graph가 끝까지 실행되고 final_text가 생성된다.

        1단계: interrupt까지 실행
        2단계: approved=True로 resume
        """
        from langgraph.types import Command

        config = {"configurable": {"thread_id": "smoke-approve-1"}}
        initial = {
            "session_id": "test-session-approve",
            "request_id": "test-request-approve",
            "messages": [HumanMessage(content="답변 초안 작성해줘")],
        }

        # 1단계: interrupt까지 실행
        graph.invoke(initial, config=config)

        # 2단계: 승인으로 resume
        result = graph.invoke(
            Command(resume={"approved": True}),
            config=config,
        )

        assert result.get("final_text"), "승인 후 final_text가 생성되어야 합니다"
        assert (
            result.get("approval_status") == ApprovalStatus.APPROVED.value
        ), f"approval_status가 APPROVED여야 합니다. 실제: {result.get('approval_status')}"

    def test_graph_ends_on_rejection(self, graph):
        """거절 시 graph가 tool_execute 없이 종료된다.

        1단계: interrupt까지 실행
        2단계: approved=False로 resume -> tool_results가 없어야 한다
        """
        from langgraph.types import Command

        config = {"configurable": {"thread_id": "smoke-reject-1"}}
        initial = {
            "session_id": "test-session-reject",
            "request_id": "test-request-reject",
            "messages": [HumanMessage(content="답변 초안 작성해줘")],
        }

        # 1단계: interrupt까지 실행
        graph.invoke(initial, config=config)

        # 2단계: 거절로 resume
        result = graph.invoke(
            Command(resume={"approved": False}),
            config=config,
        )

        assert (
            result.get("approval_status") == ApprovalStatus.REJECTED.value
        ), f"approval_status가 REJECTED여야 합니다. 실제: {result.get('approval_status')}"
        assert not result.get("tool_results"), "거절 후 tool_results가 비어있어야 합니다"

    def test_persist_logs_graph_run(self, graph, session_store):
        """승인 후 전체 실행 완료 시 SessionStore에 graph_run 레코드가 기록된다.

        plan_summary, approval_status, executed_capabilities가 포함되어야 한다.
        """
        from langgraph.types import Command

        config = {"configurable": {"thread_id": "smoke-graph-run-1"}}
        session_id = "test-session-graph-run"
        request_id = "test-request-graph-run"
        initial = {
            "session_id": session_id,
            "request_id": request_id,
            "messages": [HumanMessage(content="답변 초안 작성해줘")],
        }

        # 1단계: interrupt까지 실행
        graph.invoke(initial, config=config)

        # 2단계: 승인으로 resume
        graph.invoke(
            Command(resume={"approved": True}),
            config=config,
        )

        # graph_run 레코드 검증
        session = session_store.get_or_create(session_id)
        assert len(session.recent_graph_runs) > 0, "graph_run 레코드가 기록되어야 합니다"

        run = session.recent_graph_runs[0]
        assert run.request_id == request_id
        assert run.approval_status == ApprovalStatus.APPROVED.value
        assert run.plan_summary, "plan_summary가 비어있지 않아야 합니다"
        assert len(run.executed_capabilities) > 0, "executed_capabilities가 있어야 합니다"
        assert run.status == "completed"

    def test_persist_logs_tool_runs_with_request_id(self, graph, session_store):
        """tool_runs가 올바른 graph_run_request_id로 기록된다."""
        from langgraph.types import Command

        config = {"configurable": {"thread_id": "smoke-tool-run-1"}}
        session_id = "test-session-tool-run"
        request_id = "test-request-tool-run"
        initial = {
            "session_id": session_id,
            "request_id": request_id,
            "messages": [HumanMessage(content="답변 초안 작성해줘")],
        }

        # 1단계: interrupt까지 실행
        graph.invoke(initial, config=config)

        # 2단계: 승인으로 resume
        graph.invoke(
            Command(resume={"approved": True}),
            config=config,
        )

        # tool_run 레코드 검증
        session = session_store.get_or_create(session_id)
        tool_runs = session.recent_tool_runs
        assert len(tool_runs) > 0, "tool_run 레코드가 기록되어야 합니다"

        for tr in tool_runs:
            assert tr.graph_run_request_id == request_id, (
                f"tool_run의 graph_run_request_id가 '{request_id}'여야 합니다. "
                f"실제: {tr.graph_run_request_id}"
            )

    def test_persist_logs_graph_run_on_rejection(self, graph, session_store):
        """거절 시에도 graph_run 레코드가 기록된다.

        거절 경로도 persist 노드를 거치므로 graph_run이 남아야 한다.
        """
        from langgraph.types import Command

        config = {"configurable": {"thread_id": "smoke-reject-persist-1"}}
        session_id = "test-session-reject-persist"
        request_id = "test-request-reject-persist"
        initial = {
            "session_id": session_id,
            "request_id": request_id,
            "messages": [HumanMessage(content="답변 초안 작성해줘")],
        }

        # 1단계: interrupt까지 실행
        graph.invoke(initial, config=config)

        # 2단계: 거절로 resume
        graph.invoke(
            Command(resume={"approved": False}),
            config=config,
        )

        # graph_run 레코드 검증
        session = session_store.get_or_create(session_id)
        assert len(session.recent_graph_runs) > 0, "거절 시에도 graph_run이 기록되어야 합니다"

        run = session.recent_graph_runs[0]
        assert run.request_id == request_id
        assert run.approval_status == ApprovalStatus.REJECTED.value
        assert run.status == "rejected"
        assert len(run.executed_capabilities) == 0, "거절 시 executed_capabilities가 비어야 합니다"
