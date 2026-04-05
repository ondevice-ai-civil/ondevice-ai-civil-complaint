"""LangGraph approval-gated orchestration E2E 검증 테스트.

Issue #417: LangGraph orchestration E2E 검증.

Graph topology (6 nodes):
  START -> session_load -> planner -> approval_wait
               -> [approved] tool_execute -> synthesis -> persist -> END
               -> [rejected] persist -> END

각 테스트는 고유한 thread_id와 session_id를 사용하여 완전히 격리된다.
SKIP_MODEL_LOAD=true 환경에서 LLM 없이 실행 가능하다.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence

import pytest
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.inference.graph.builder import build_govon_graph
from src.inference.graph.executor_adapter import ExecutorAdapter
from src.inference.graph.planner_adapter import PlannerAdapter
from src.inference.graph.state import ApprovalStatus, TaskType, ToolPlan
from src.inference.session_context import SessionStore

os.environ.setdefault("SKIP_MODEL_LOAD", "true")


# ---------------------------------------------------------------------------
# Stub adapters
# ---------------------------------------------------------------------------


class ConfigurableStubPlanner(PlannerAdapter):
    """테스트용 고정 출력 planner.

    생성 시 주어진 task_type, goal, reason, tools를 그대로 반환하는
    ToolPlan을 생성한다. 각 테스트가 planner 출력을 완전히 제어할 수 있게 한다.
    """

    def __init__(
        self,
        task_type: TaskType,
        goal: str,
        reason: str,
        tools: List[str],
    ) -> None:
        self._task_type = task_type
        self._goal = goal
        self._reason = reason
        self._tools = tools

    async def plan(
        self,
        messages: Sequence[AnyMessage],
        context: Dict[str, Any],
    ) -> ToolPlan:
        return ToolPlan(
            task_type=self._task_type,
            goal=self._goal,
            reason=self._reason,
            tools=list(self._tools),
        )


class TrackingStubExecutor(ExecutorAdapter):
    """tool 호출 기록 및 설정 가능한 결과를 반환하는 테스트용 executor.

    `results` dict로 tool별 반환값을 설정할 수 있다.
    기본값: {"success": True, "text": f"[stub] {tool_name} result", "latency_ms": 1.0}
    모든 호출은 self.calls 리스트에 (tool_name, query) 튜플로 기록된다.
    """

    def __init__(self, results: Dict[str, dict] | None = None) -> None:
        self.calls: List[tuple[str, str]] = []
        self._results: Dict[str, dict] = results or {}

    async def execute(
        self,
        tool_name: str,
        query: str,
        context: Dict[str, Any],
    ) -> dict:
        self.calls.append((tool_name, query))
        if tool_name in self._results:
            result = dict(self._results[tool_name])
            result.setdefault("latency_ms", 1.0)
            return result
        return {
            "success": True,
            "text": f"[stub] {tool_name} result",
            "latency_ms": 1.0,
        }

    def list_tools(self) -> list[str]:
        return ["rag_search", "api_lookup", "draft_civil_response", "append_evidence"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def session_store(tmp_path):
    """임시 디렉터리에 격리된 SessionStore를 생성한다."""
    db_path = str(tmp_path / "test_e2e.sqlite3")
    return SessionStore(db_path=db_path)


@pytest.fixture
def make_graph(session_store):
    """planner와 executor를 받아 graph를 생성하는 팩토리 픽스처."""

    def _make(planner: PlannerAdapter, executor: ExecutorAdapter):
        return build_govon_graph(
            planner_adapter=planner,
            executor_adapter=executor,
            session_store=session_store,
            checkpointer=MemorySaver(),
        )

    return _make


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run_to_interrupt(graph, session_id: str, thread_id: str, query: str, request_id: str):
    """graph를 approval_wait interrupt까지 실행한다."""
    config = {"configurable": {"thread_id": thread_id}}
    initial = {
        "session_id": session_id,
        "request_id": request_id,
        "messages": [HumanMessage(content=query)],
    }
    graph.invoke(initial, config=config)
    return config


def _approve(graph, config):
    """승인 Command로 graph를 재개한다."""
    return graph.invoke(Command(resume={"approved": True}), config=config)


def _reject(graph, config):
    """거절 Command로 graph를 재개한다."""
    return graph.invoke(Command(resume={"approved": False}), config=config)


# ---------------------------------------------------------------------------
# TestClass 1: TestApprovalExecuteE2E
# ---------------------------------------------------------------------------


class TestApprovalExecuteE2E:
    """승인 경로 E2E 테스트."""

    def test_approve_full_path_produces_final_text(self, make_graph):
        """승인 후 graph가 끝까지 실행되고 final_text가 생성된다."""
        planner = ConfigurableStubPlanner(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="민원 답변 초안 작성",
            reason="사용자가 답변 초안을 요청했습니다",
            tools=["rag_search", "draft_civil_response"],
        )
        executor = TrackingStubExecutor(
            results={
                "draft_civil_response": {
                    "success": True,
                    "text": "작성된 민원 답변 초안입니다.",
                    "latency_ms": 1.0,
                }
            }
        )
        graph = make_graph(planner, executor)

        config = _run_to_interrupt(
            graph,
            session_id="e2e-approve-full-sess-1",
            thread_id="e2e-approve-full-1",
            query="민원 답변 초안 작성해줘",
            request_id="e2e-approve-full-req-1",
        )
        result = _approve(graph, config)

        assert result.get("final_text"), "승인 후 final_text가 생성되어야 합니다"
        assert result.get("approval_status") == ApprovalStatus.APPROVED.value

    def test_approve_executes_all_planned_tools(self, make_graph):
        """승인 후 planned_tools 목록의 모든 tool이 실행된다."""
        planned = ["rag_search", "draft_civil_response"]
        planner = ConfigurableStubPlanner(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="민원 답변 초안 작성",
            reason="사용자가 답변 초안을 요청했습니다",
            tools=planned,
        )
        executor = TrackingStubExecutor()
        graph = make_graph(planner, executor)

        config = _run_to_interrupt(
            graph,
            session_id="e2e-approve-tools-sess-1",
            thread_id="e2e-approve-tools-1",
            query="민원 답변 초안 작성해줘",
            request_id="e2e-approve-tools-req-1",
        )
        result = _approve(graph, config)

        tool_results = result.get("tool_results", {})
        for tool_name in planned:
            assert tool_name in tool_results, f"tool_results에 {tool_name}이 없습니다"

        executed_names = [call[0] for call in executor.calls]
        for tool_name in planned:
            assert tool_name in executed_names, f"executor.calls에 {tool_name}이 없습니다"

    def test_approve_accumulated_context_has_tool_outputs(self, make_graph):
        """승인 후 accumulated_context에 각 tool의 결과가 포함된다."""
        planned = ["rag_search", "draft_civil_response"]
        planner = ConfigurableStubPlanner(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="민원 답변 초안 작성",
            reason="사용자가 답변 초안을 요청했습니다",
            tools=planned,
        )
        executor = TrackingStubExecutor()
        graph = make_graph(planner, executor)

        config = _run_to_interrupt(
            graph,
            session_id="e2e-approve-ctx-sess-1",
            thread_id="e2e-approve-ctx-1",
            query="민원 답변 초안 작성해줘",
            request_id="e2e-approve-ctx-req-1",
        )
        result = _approve(graph, config)

        accumulated = result.get("accumulated_context", {})
        for tool_name in planned:
            assert tool_name in accumulated, f"accumulated_context에 {tool_name} 결과가 없습니다"
            entry = accumulated[tool_name]
            assert entry, f"accumulated_context[{tool_name}]이 비어있습니다"


# ---------------------------------------------------------------------------
# TestClass 2: TestRejectIdleE2E
# ---------------------------------------------------------------------------


class TestRejectIdleE2E:
    """거절 경로 E2E 테스트."""

    def test_reject_skips_tool_execute(self, make_graph):
        """거절 시 tool_execute가 실행되지 않는다."""
        planner = ConfigurableStubPlanner(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="민원 답변 초안 작성",
            reason="사용자가 답변 초안을 요청했습니다",
            tools=["rag_search", "draft_civil_response"],
        )
        executor = TrackingStubExecutor()
        graph = make_graph(planner, executor)

        config = _run_to_interrupt(
            graph,
            session_id="e2e-reject-skip-sess-1",
            thread_id="e2e-reject-skip-1",
            query="민원 답변 초안 작성해줘",
            request_id="e2e-reject-skip-req-1",
        )
        result = _reject(graph, config)

        assert not result.get("tool_results"), "거절 후 tool_results가 비어있어야 합니다"
        assert len(executor.calls) == 0, "거절 후 executor.calls가 비어있어야 합니다"

    def test_reject_graph_reaches_idle(self, make_graph):
        """거절 후 graph가 idle 상태(pending 노드 없음)로 복귀한다."""
        planner = ConfigurableStubPlanner(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="민원 답변 초안 작성",
            reason="사용자가 답변 초안을 요청했습니다",
            tools=["rag_search", "draft_civil_response"],
        )
        executor = TrackingStubExecutor()
        graph = make_graph(planner, executor)

        config = _run_to_interrupt(
            graph,
            session_id="e2e-reject-idle-sess-1",
            thread_id="e2e-reject-idle-1",
            query="민원 답변 초안 작성해줘",
            request_id="e2e-reject-idle-req-1",
        )
        _reject(graph, config)

        state = graph.get_state(config)
        assert not state.next, "거절 후 graph가 idle 상태여야 합니다 (next가 비어야 함)"

    def test_reject_then_new_run_on_same_session(self, make_graph, session_store):
        """거절 후 같은 session에서 새 thread로 실행하면 두 번째 실행이 성공한다."""
        planner = ConfigurableStubPlanner(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="민원 답변 초안 작성",
            reason="사용자가 답변 초안을 요청했습니다",
            tools=["rag_search", "draft_civil_response"],
        )
        executor1 = TrackingStubExecutor()
        graph1 = make_graph(planner, executor1)

        session_id = "e2e-reject-rerun-sess-1"

        # 첫 번째 실행: 거절
        config1 = _run_to_interrupt(
            graph1,
            session_id=session_id,
            thread_id="e2e-reject-rerun-thread-1",
            query="민원 답변 초안 작성해줘",
            request_id="e2e-reject-rerun-req-1",
        )
        _reject(graph1, config1)

        # 두 번째 실행: 새 thread_id, 동일 session_id, 승인
        planner2 = ConfigurableStubPlanner(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="민원 답변 초안 작성",
            reason="사용자가 답변 초안을 요청했습니다",
            tools=["rag_search", "draft_civil_response"],
        )
        executor2 = TrackingStubExecutor(
            results={
                "draft_civil_response": {
                    "success": True,
                    "text": "두 번째 실행 민원 답변 초안입니다.",
                    "latency_ms": 1.0,
                }
            }
        )
        graph2 = make_graph(planner2, executor2)

        config2 = _run_to_interrupt(
            graph2,
            session_id=session_id,
            thread_id="e2e-reject-rerun-thread-2",
            query="민원 답변 초안 작성해줘",
            request_id="e2e-reject-rerun-req-2",
        )
        result2 = _approve(graph2, config2)

        assert result2.get("final_text"), "두 번째 실행에서 final_text가 생성되어야 합니다"
        assert result2.get("approval_status") == ApprovalStatus.APPROVED.value

        # 세션에 두 run이 모두 기록되어야 한다
        session = session_store.get_or_create(session_id)
        assert (
            len(session.recent_graph_runs) >= 2
        ), "세션에 2개 이상의 graph_run이 기록되어야 합니다"


# ---------------------------------------------------------------------------
# TestClass 3: TestEvidenceAugmentationE2E
# ---------------------------------------------------------------------------


class TestEvidenceAugmentationE2E:
    """근거 보강(APPEND_EVIDENCE) 경로 E2E 테스트."""

    def test_append_evidence_executes_all_evidence_tools(self, make_graph):
        """APPEND_EVIDENCE 작업에서 3개 tool이 모두 실행된다."""
        planner = ConfigurableStubPlanner(
            task_type=TaskType.APPEND_EVIDENCE,
            goal="민원 답변 근거 보강",
            reason="사용자가 근거 보강을 요청했습니다",
            tools=["rag_search", "api_lookup", "append_evidence"],
        )
        executor = TrackingStubExecutor(
            results={
                "rag_search": {
                    "success": True,
                    "text": "RAG 결과",
                    "results": [{"title": "관련 법령", "content": "도로법 제3조"}],
                    "latency_ms": 1.0,
                },
                "api_lookup": {
                    "success": True,
                    "text": "API 결과",
                    "context_text": "유사 민원 통계 데이터",
                    "latency_ms": 1.0,
                },
                "append_evidence": {
                    "success": True,
                    "text": "보강된 근거 텍스트",
                    "latency_ms": 1.0,
                },
            }
        )
        graph = make_graph(planner, executor)

        config = _run_to_interrupt(
            graph,
            session_id="e2e-evidence-all-sess-1",
            thread_id="e2e-evidence-all-1",
            query="근거를 보강해줘",
            request_id="e2e-evidence-all-req-1",
        )
        result = _approve(graph, config)

        executed_names = [call[0] for call in executor.calls]
        for tool_name in ["rag_search", "api_lookup", "append_evidence"]:
            assert tool_name in executed_names, f"{tool_name}이 실행되지 않았습니다"

        final_text = result.get("final_text", "")
        assert (
            "보강된 근거 텍스트" in final_text
        ), f"final_text에 append_evidence 결과가 포함되어야 합니다. 실제: {final_text!r}"

    def test_evidence_accumulated_context_chains(self, make_graph):
        """append_evidence 실행 시 accumulated_context에 이전 tool 결과가 포함된다."""
        planner = ConfigurableStubPlanner(
            task_type=TaskType.APPEND_EVIDENCE,
            goal="민원 답변 근거 보강",
            reason="사용자가 근거 보강을 요청했습니다",
            tools=["rag_search", "api_lookup", "append_evidence"],
        )
        executor = TrackingStubExecutor(
            results={
                "rag_search": {
                    "success": True,
                    "text": "RAG 결과",
                    "results": [{"title": "관련 법령", "content": "도로법 제3조"}],
                    "latency_ms": 1.0,
                },
                "api_lookup": {
                    "success": True,
                    "text": "API 결과",
                    "context_text": "유사 민원 통계 데이터",
                    "latency_ms": 1.0,
                },
                "append_evidence": {
                    "success": True,
                    "text": "보강된 근거 텍스트",
                    "latency_ms": 1.0,
                },
            }
        )
        graph = make_graph(planner, executor)

        config = _run_to_interrupt(
            graph,
            session_id="e2e-evidence-chain-sess-1",
            thread_id="e2e-evidence-chain-1",
            query="근거를 보강해줘",
            request_id="e2e-evidence-chain-req-1",
        )
        result = _approve(graph, config)

        # executor.calls 순서 검증: rag_search -> api_lookup -> append_evidence
        executed_names = [call[0] for call in executor.calls]
        assert executed_names.index("rag_search") < executed_names.index(
            "append_evidence"
        ), "rag_search는 append_evidence보다 먼저 실행되어야 합니다"
        assert executed_names.index("api_lookup") < executed_names.index(
            "append_evidence"
        ), "api_lookup은 append_evidence보다 먼저 실행되어야 합니다"

        # accumulated_context에 이전 tool 결과가 포함되어 있어야 한다
        accumulated = result.get("accumulated_context", {})
        assert "rag_search" in accumulated, "accumulated_context에 rag_search 결과가 있어야 합니다"
        assert "api_lookup" in accumulated, "accumulated_context에 api_lookup 결과가 있어야 합니다"
        assert (
            "append_evidence" in accumulated
        ), "accumulated_context에 append_evidence 결과가 있어야 합니다"

    def test_draft_then_evidence_follow_up(self, make_graph, session_store):
        """두 번의 연속 graph run: 첫 번째 DRAFT_RESPONSE, 두 번째 APPEND_EVIDENCE."""
        session_id = "e2e-draft-evidence-sess-1"

        # 첫 번째 실행: DRAFT_RESPONSE
        planner1 = ConfigurableStubPlanner(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="민원 답변 초안 작성",
            reason="사용자가 답변 초안을 요청했습니다",
            tools=["rag_search", "draft_civil_response"],
        )
        executor1 = TrackingStubExecutor(
            results={
                "draft_civil_response": {
                    "success": True,
                    "text": "초안 답변 텍스트",
                    "latency_ms": 1.0,
                }
            }
        )
        graph1 = make_graph(planner1, executor1)

        config1 = _run_to_interrupt(
            graph1,
            session_id=session_id,
            thread_id="e2e-draft-evidence-thread-1",
            query="민원 답변 초안 작성해줘",
            request_id="e2e-draft-evidence-req-1",
        )
        result1 = _approve(graph1, config1)
        assert result1.get("final_text"), "첫 번째 실행에서 final_text가 생성되어야 합니다"

        # 두 번째 실행: APPEND_EVIDENCE
        planner2 = ConfigurableStubPlanner(
            task_type=TaskType.APPEND_EVIDENCE,
            goal="민원 답변 근거 보강",
            reason="사용자가 근거 보강을 요청했습니다",
            tools=["rag_search", "api_lookup", "append_evidence"],
        )
        executor2 = TrackingStubExecutor(
            results={
                "rag_search": {
                    "success": True,
                    "text": "RAG 결과",
                    "latency_ms": 1.0,
                },
                "api_lookup": {
                    "success": True,
                    "text": "API 결과",
                    "context_text": "통계 데이터",
                    "latency_ms": 1.0,
                },
                "append_evidence": {
                    "success": True,
                    "text": "보강된 근거 텍스트 두 번째",
                    "latency_ms": 1.0,
                },
            }
        )
        graph2 = make_graph(planner2, executor2)

        config2 = _run_to_interrupt(
            graph2,
            session_id=session_id,
            thread_id="e2e-draft-evidence-thread-2",
            query="근거를 보강해줘",
            request_id="e2e-draft-evidence-req-2",
        )
        result2 = _approve(graph2, config2)
        assert result2.get("final_text"), "두 번째 실행에서 final_text가 생성되어야 합니다"

        # 세션에 두 run이 모두 기록되어야 한다
        session = session_store.get_or_create(session_id)
        graph_runs = session.recent_graph_runs
        assert len(graph_runs) >= 2, "세션에 2개 이상의 graph_run이 기록되어야 합니다"

        request_ids = {run.request_id for run in graph_runs}
        assert "e2e-draft-evidence-req-1" in request_ids, "첫 번째 graph_run이 기록되어야 합니다"
        assert "e2e-draft-evidence-req-2" in request_ids, "두 번째 graph_run이 기록되어야 합니다"


# ---------------------------------------------------------------------------
# TestClass 4: TestSessionTraceConsistency
# ---------------------------------------------------------------------------


class TestSessionTraceConsistency:
    """세션 추적 일관성 E2E 테스트."""

    def test_graph_run_and_tool_runs_share_request_id(self, make_graph, session_store):
        """graph_run의 request_id와 모든 tool_run의 graph_run_request_id가 일치한다."""
        session_id = "e2e-trace-req-sess-1"
        request_id = "e2e-trace-req-1"

        planner = ConfigurableStubPlanner(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="민원 답변 초안 작성",
            reason="사용자가 답변 초안을 요청했습니다",
            tools=["rag_search", "draft_civil_response"],
        )
        executor = TrackingStubExecutor()
        graph = make_graph(planner, executor)

        config = _run_to_interrupt(
            graph,
            session_id=session_id,
            thread_id="e2e-trace-req-thread-1",
            query="민원 답변 초안 작성해줘",
            request_id=request_id,
        )
        _approve(graph, config)

        session = session_store.get_or_create(session_id)
        graph_runs = session.recent_graph_runs
        assert len(graph_runs) > 0, "graph_run이 기록되어야 합니다"

        run = graph_runs[0]
        assert run.request_id == request_id, "graph_run.request_id가 일치해야 합니다"

        tool_runs = session.recent_tool_runs
        assert len(tool_runs) > 0, "tool_run이 기록되어야 합니다"
        for tr in tool_runs:
            assert tr.graph_run_request_id == request_id, (
                f"tool_run의 graph_run_request_id가 '{request_id}'여야 합니다. "
                f"실제: {tr.graph_run_request_id}"
            )

    def test_multi_turn_accumulates_graph_runs(self, make_graph, session_store):
        """같은 세션에서 두 번의 graph run이 모두 기록된다."""
        session_id = "e2e-multi-turn-sess-1"

        planner = ConfigurableStubPlanner(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="민원 답변 초안 작성",
            reason="사용자가 답변 초안을 요청했습니다",
            tools=["rag_search", "draft_civil_response"],
        )

        # 첫 번째 실행
        graph1 = make_graph(planner, TrackingStubExecutor())
        config1 = _run_to_interrupt(
            graph1,
            session_id=session_id,
            thread_id="e2e-multi-turn-thread-1",
            query="첫 번째 질문",
            request_id="e2e-multi-turn-req-1",
        )
        _approve(graph1, config1)

        # 두 번째 실행
        planner2 = ConfigurableStubPlanner(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="민원 답변 초안 작성",
            reason="사용자가 답변 초안을 요청했습니다",
            tools=["rag_search", "draft_civil_response"],
        )
        graph2 = make_graph(planner2, TrackingStubExecutor())
        config2 = _run_to_interrupt(
            graph2,
            session_id=session_id,
            thread_id="e2e-multi-turn-thread-2",
            query="두 번째 질문",
            request_id="e2e-multi-turn-req-2",
        )
        _approve(graph2, config2)

        session = session_store.get_or_create(session_id)
        assert (
            len(session.recent_graph_runs) >= 2
        ), f"세션에 2개 이상의 graph_run이 기록되어야 합니다. 실제: {len(session.recent_graph_runs)}"

    def test_reject_graph_run_has_zero_executed_capabilities(self, make_graph, session_store):
        """거절 시 graph_run.executed_capabilities가 빈 리스트이고 status가 'rejected'이다."""
        session_id = "e2e-reject-cap-sess-1"
        request_id = "e2e-reject-cap-req-1"

        planner = ConfigurableStubPlanner(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="민원 답변 초안 작성",
            reason="사용자가 답변 초안을 요청했습니다",
            tools=["rag_search", "draft_civil_response"],
        )
        executor = TrackingStubExecutor()
        graph = make_graph(planner, executor)

        config = _run_to_interrupt(
            graph,
            session_id=session_id,
            thread_id="e2e-reject-cap-thread-1",
            query="민원 답변 초안 작성해줘",
            request_id=request_id,
        )
        _reject(graph, config)

        session = session_store.get_or_create(session_id)
        graph_runs = session.recent_graph_runs
        assert len(graph_runs) > 0, "거절 시에도 graph_run이 기록되어야 합니다"

        run = graph_runs[0]
        assert (
            run.executed_capabilities == []
        ), f"거절 시 executed_capabilities가 빈 리스트여야 합니다. 실제: {run.executed_capabilities}"
        assert (
            run.status == "rejected"
        ), f"거절 시 status가 'rejected'여야 합니다. 실제: {run.status}"

    def test_session_messages_reflect_graph_io(self, make_graph, session_store):
        """승인 후 세션에 사용자 메시지와 어시스턴트 메시지가 모두 기록된다."""
        session_id = "e2e-messages-sess-1"
        user_query = "이 민원에 대한 답변 초안 작성해줘"

        planner = ConfigurableStubPlanner(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="민원 답변 초안 작성",
            reason="사용자가 답변 초안을 요청했습니다",
            tools=["rag_search", "draft_civil_response"],
        )
        executor = TrackingStubExecutor(
            results={
                "draft_civil_response": {
                    "success": True,
                    "text": "작성된 민원 답변 초안입니다.",
                    "latency_ms": 1.0,
                }
            }
        )
        graph = make_graph(planner, executor)

        config = _run_to_interrupt(
            graph,
            session_id=session_id,
            thread_id="e2e-messages-thread-1",
            query=user_query,
            request_id="e2e-messages-req-1",
        )
        _approve(graph, config)

        session = session_store.get_or_create(session_id)
        messages = session.recent_history

        assert (
            len(messages) >= 2
        ), f"세션에 사용자와 어시스턴트 메시지가 모두 기록되어야 합니다. 실제: {len(messages)}"

        roles = [msg.role for msg in messages]
        assert "user" in roles, "사용자 메시지가 기록되어야 합니다"
        assert "assistant" in roles, "어시스턴트 메시지가 기록되어야 합니다"

        user_msgs = [msg for msg in messages if msg.role == "user"]
        assert any(
            user_query in msg.content for msg in user_msgs
        ), "사용자 메시지에 원래 쿼리가 포함되어야 합니다"

        assistant_msgs = [msg for msg in messages if msg.role == "assistant"]
        assert any(
            msg.content for msg in assistant_msgs
        ), "어시스턴트 메시지가 비어있지 않아야 합니다"
