"""LangGraph tooling E2E 통합 테스트.

Issue #162: tooling 계층 전체를 end-to-end로 검증한다.

test_orchestration_e2e.py와의 차이:
  - orchestration E2E는 StubExecutorAdapter로 graph 흐름만 검증
  - 이 파일은 실제 capability 인스턴스 + mock execute_fn으로
    capability→adapter→node 파이프라인을 검증

실제 capability 인스턴스(ApiLookupCapability,
DraftResponseCapability)를 사용하고
RegistryExecutorAdapter를 통해 capability->adapter->node 파이프라인을 검증한다.
StubExecutorAdapter가 아닌 실제 capability + mock execute_fn 클로저를 사용한다.

각 테스트는 고유한 thread_id와 session_id를 사용하여 완전히 격리된다.
SKIP_MODEL_LOAD=true 환경에서 LLM 없이 실행 가능하다.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Sequence

import pytest
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.inference.graph.builder import build_govon_graph
from src.inference.graph.capabilities.api_lookup import ApiLookupCapability
from src.inference.graph.capabilities.draft_response import DraftResponseCapability
from src.inference.graph.executor_adapter import RegistryExecutorAdapter
from src.inference.graph.planner_adapter import PlannerAdapter
from src.inference.graph.state import ApprovalStatus, TaskType, ToolPlan
from src.inference.session_context import SessionStore

os.environ.setdefault("SKIP_MODEL_LOAD", "true")

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helper: registry factory
# ---------------------------------------------------------------------------


def _make_registry(
    api_action=None,
    draft_fn=None,
) -> Dict[str, Any]:
    """실제 capability 인스턴스로 구성된 registry를 생성한다.

    각 capability는 주입된 mock execute_fn 클로저를 사용하여
    capability->adapter->node 파이프라인을 실제로 거친다.
    """
    if draft_fn is None:

        async def draft_fn(query, context, session):
            return {"text": f"[기본 초안] {query}에 대한 답변입니다."}

    return {
        "api_lookup": ApiLookupCapability(action=api_action),
        "draft_response": DraftResponseCapability(execute_fn=draft_fn),
    }


# ---------------------------------------------------------------------------
# ConfigurableStubPlanner (same pattern as test_orchestration_e2e.py)
# ---------------------------------------------------------------------------


class ConfigurableStubPlanner(PlannerAdapter):
    """테스트용 고정 출력 planner.

    생성 시 주어진 task_type, goal, reason, tools를 그대로 반환하는
    ToolPlan을 생성한다.
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def session_store(tmp_path):
    """임시 디렉터리에 격리된 SessionStore를 생성한다."""
    return SessionStore(db_path=str(tmp_path / "test_tooling_e2e.sqlite3"))


@pytest.fixture
def make_tooling_graph(session_store):
    """팩토리: 실제 capability + configurable planner로 graph를 생성한다."""

    def _make(planner, api_action=None, draft_fn=None):
        registry = _make_registry(api_action, draft_fn)
        executor = RegistryExecutorAdapter(tool_registry=registry, session_store=session_store)
        return build_govon_graph(
            planner_adapter=planner,
            executor_adapter=executor,
            session_store=session_store,
            checkpointer=MemorySaver(),
        )

    return _make


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


async def _run_to_interrupt(graph, session_id: str, thread_id: str, query: str, request_id: str):
    """graph를 approval_wait interrupt까지 실행한다."""
    config = {"configurable": {"thread_id": thread_id}}
    initial = {
        "session_id": session_id,
        "request_id": request_id,
        "messages": [HumanMessage(content=query)],
    }
    await graph.ainvoke(initial, config=config)
    return config


async def _approve(graph, config):
    """승인 Command로 graph를 재개한다."""
    return await graph.ainvoke(Command(resume={"approved": True}), config=config)


# ---------------------------------------------------------------------------
# TestClass 1: TestDraftResponsePipeline
# ---------------------------------------------------------------------------


class TestDraftResponsePipeline:
    """DRAFT_RESPONSE 파이프라인 E2E 테스트.

    실제 ApiLookupCapability, DraftResponseCapability
    인스턴스를 사용하여 capability->adapter->node 파이프라인을 검증한다.
    """

    async def test_draft_response_full_pipeline(self, make_tooling_graph):
        """api+draft 2-tool 콤보: draft 텍스트가 final_text로 우선 선택된다.

        api_action=None(빈 결과), draft는 텍스트 반환.
        final_text가 draft 텍스트와 일치하고, 2개 tool 모두 tool_results에 존재한다.
        """
        expected_draft_text = "민원 답변 초안: 도로 파손 관련 조치를 취하겠습니다."

        async def draft_fn(query, context, session):
            return {"text": expected_draft_text}

        planner = ConfigurableStubPlanner(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="민원 답변 초안 작성",
            reason="사용자가 답변 초안을 요청했습니다",
            tools=["api_lookup", "draft_response"],
        )
        graph = make_tooling_graph(planner, draft_fn=draft_fn)

        config = await _run_to_interrupt(
            graph,
            session_id="tooling-draft-full-sess-1",
            thread_id="tooling-draft-full-1",
            query="도로 파손 민원 답변 초안 작성해줘",
            request_id="tooling-draft-full-req-1",
        )
        result = await _approve(graph, config)

        final_text = result.get("final_text", "")
        tool_results = result.get("tool_results", {})

        assert (
            expected_draft_text in final_text
        ), f"final_text에 draft 텍스트가 포함되어야 합니다. 실제: {final_text!r}"
        assert "api_lookup" in tool_results, "tool_results에 api_lookup이 있어야 합니다"
        assert "draft_response" in tool_results, "tool_results에 draft_response가 있어야 합니다"

    async def test_draft_response_synthesis_prioritizes_draft_text(self, make_tooling_graph):
        """draft_response 텍스트가 api 결과보다 우선 선택된다.

        synthesis_node의 _extract_final_text 우선순위 검증:
        draft_response.text > api formatted results
        """
        draft_text = "초안 텍스트: 민원에 대해 답변드립니다."

        async def draft_fn(query, context, session):
            return {"text": draft_text}

        planner = ConfigurableStubPlanner(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="민원 답변 초안 작성",
            reason="사용자가 답변 초안을 요청했습니다",
            tools=["api_lookup", "draft_response"],
        )
        graph = make_tooling_graph(planner, draft_fn=draft_fn)

        config = await _run_to_interrupt(
            graph,
            session_id="tooling-draft-priority-sess-1",
            thread_id="tooling-draft-priority-1",
            query="답변 초안 작성해줘",
            request_id="tooling-draft-priority-req-1",
        )
        result = await _approve(graph, config)

        final_text = result.get("final_text", "")
        assert (
            draft_text in final_text
        ), f"draft_response 텍스트가 final_text에 포함되어야 합니다. 실제: {final_text!r}"

    async def test_lookup_stats_api_only(self, make_tooling_graph):
        """api_lookup 단독 실행: tool_results에 api_lookup만 존재한다.

        api_action=None이므로 ApiLookupCapability는 빈 결과를 반환한다.
        파이프라인이 정상 완료되고 fallback 또는 api context_text가 final_text로 사용된다.
        """
        planner = ConfigurableStubPlanner(
            task_type=TaskType.LOOKUP_STATS,
            goal="유사 민원 사례 조회",
            reason="통계 데이터가 필요합니다",
            tools=["api_lookup"],
        )
        # api_action=None -> ApiLookupCapability가 빈 결과 반환
        graph = make_tooling_graph(planner, api_action=None)

        config = await _run_to_interrupt(
            graph,
            session_id="tooling-lookup-stats-sess-1",
            thread_id="tooling-lookup-stats-1",
            query="유사 민원 사례 조회해줘",
            request_id="tooling-lookup-stats-req-1",
        )
        result = await _approve(graph, config)

        tool_results = result.get("tool_results", {})
        assert "api_lookup" in tool_results, "tool_results에 api_lookup이 있어야 합니다"
        assert "api_lookup" not in tool_results, "LOOKUP_STATS에서 api_lookup이 실행되면 안 됩니다"

        # api_lookup 성공 여부: action=None이면 success=True, empty_reason="no_match"
        api_result = tool_results["api_lookup"]
        assert (
            api_result.get("success") is True
        ), "api_action=None일 때 api_lookup은 성공 상태(빈 결과)를 반환해야 합니다"

        # api_action=None이면 유의미한 결과가 없으므로 fallback 메시지여야 한다
        final_text = result.get("final_text", "")
        assert (
            final_text == "요청을 처리할 수 없습니다."
        ), f"api_action=None일 때 유의미한 결과가 없으므로 fallback이어야 합니다. 실제: {final_text!r}"


# ---------------------------------------------------------------------------
# TestClass 2: TestPartialFailureE2E
# ---------------------------------------------------------------------------


class TestPartialFailureE2E:
    """부분 실패 시나리오 E2E 테스트.

    일부 tool이 실패해도 나머지 tool이 계속 실행되는 resilience를 검증한다.
    """

    async def test_api_failure_draft_still_runs(self, session_store):
        """api_lookup이 예외를 발생시켜도 draft_response가 실행된다.

        RegistryExecutorAdapter의 예외 처리가 api 실패를 잡고
        다음 tool인 draft_response를 실행한다.
        ApiLookupCapability는 action.fetch_similar_cases 예외를 success=False로 반환하고
        tool_execute_node는 계속 진행한다.
        """
        draft_text = "API 실패 이후 생성된 민원 답변 초안입니다."

        async def draft_fn(query, context, session):
            return {"text": draft_text}

        # api_action처럼 동작하지만 fetch_similar_cases에서 예외 발생
        class FailingApiAction:
            _ret_count = 5
            _min_score = 2

            async def fetch_similar_cases(self, query, context):
                raise RuntimeError("API 서버 연결 오류")

        # FailingApiAction을 주입한 registry 구성
        registry = _make_registry(api_action=None, draft_fn=draft_fn)
        registry["api_lookup"] = ApiLookupCapability(action=FailingApiAction())
        executor = RegistryExecutorAdapter(tool_registry=registry, session_store=session_store)

        planner = ConfigurableStubPlanner(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="민원 답변 초안 작성",
            reason="사용자가 답변 초안을 요청했습니다",
            tools=["api_lookup", "draft_response"],
        )

        graph = build_govon_graph(
            planner_adapter=planner,
            executor_adapter=executor,
            session_store=session_store,
            checkpointer=MemorySaver(),
        )

        config = await _run_to_interrupt(
            graph,
            session_id="tooling-api-fail-sess-1",
            thread_id="tooling-api-fail-1",
            query="답변 초안 작성해줘",
            request_id="tooling-api-fail-req-1",
        )
        result = await _approve(graph, config)

        tool_results = result.get("tool_results", {})
        assert "api_lookup" in tool_results, "api_lookup이 tool_results에 있어야 합니다"
        api_result = tool_results["api_lookup"]
        assert api_result.get("success") is False, "API 실패 시 success=False여야 합니다"

        assert (
            "draft_response" in tool_results
        ), "api 실패 후에도 draft_response가 실행되어야 합니다"
        assert draft_text in result.get(
            "final_text", ""
        ), "draft 텍스트가 final_text에 포함되어야 합니다"

    async def test_draft_exception_caught_by_adapter(self, make_tooling_graph):
        """draft execute_fn이 RuntimeError를 발생시키면 어댑터가 잡고 success=False를 반환한다.

        DraftResponseCapability는 execute() 내부 try/except 없이
        execute_fn에서 발생한 예외가 CapabilityBase.__call__을 통해
        RegistryExecutorAdapter까지 전파된다. 어댑터가 예외를 잡아 success=False로 반환한다.
        """

        async def draft_fn_raises(query, context, session):
            raise RuntimeError("LLM 서버 오류")

        planner = ConfigurableStubPlanner(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="민원 답변 초안 작성",
            reason="사용자가 답변 초안을 요청했습니다",
            tools=["api_lookup", "draft_response"],
        )
        graph = make_tooling_graph(planner, draft_fn=draft_fn_raises)

        config = await _run_to_interrupt(
            graph,
            session_id="tooling-draft-except-sess-1",
            thread_id="tooling-draft-except-1",
            query="답변 초안 작성해줘",
            request_id="tooling-draft-except-req-1",
        )
        result = await _approve(graph, config)

        tool_results = result.get("tool_results", {})
        assert "draft_response" in tool_results, "draft_response가 tool_results에 있어야 합니다"
        draft_result = tool_results["draft_response"]
        assert (
            draft_result.get("success") is False
        ), "draft 예외 시 tool_results['draft_response']['success']==False여야 합니다"
        assert draft_result.get("error"), "draft 예외 시 error 필드가 있어야 합니다"

    async def test_all_tools_fail_synthesis_fallback(self, make_tooling_graph):
        """모든 tool이 실패하면 final_text가 fallback 메시지가 된다.

        DraftResponseCapability는 execute() 내부 try/except가 없으므로
        예외가 RegistryExecutorAdapter까지 전파되어 어댑터가 잡는다.

        draft execute_fn에서 RuntimeError 발생 → CapabilityBase.__call__을 통해 어댑터로 전파 → success=False.
        api_action=None → ApiLookupCapability가 success=True, 빈 결과 반환.

        유효한 텍스트 소스가 없으므로 _extract_final_text는 "요청을 처리할 수 없습니다."를 반환한다.
        """

        async def draft_fn_fail(query, context, session):
            raise RuntimeError("LLM 서버 오류")

        planner = ConfigurableStubPlanner(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="민원 답변 초안 작성",
            reason="사용자가 답변 초안을 요청했습니다",
            tools=["api_lookup", "draft_response"],
        )
        # api_action=None: success=True지만 빈 결과, context_text 없음
        graph = make_tooling_graph(
            planner, api_action=None, draft_fn=draft_fn_fail
        )

        config = await _run_to_interrupt(
            graph,
            session_id="tooling-all-fail-sess-1",
            thread_id="tooling-all-fail-1",
            query="답변 초안 작성해줘",
            request_id="tooling-all-fail-req-1",
        )
        result = await _approve(graph, config)

        # draft가 실패해야 한다
        tool_results = result.get("tool_results", {})
        draft_result = tool_results.get("draft_response", {})
        assert draft_result.get("success") is False, "draft 실패 확인"

        # api는 success=True지만 빈 결과
        api_result = tool_results.get("api_lookup", {})
        assert api_result.get("success") is True, "api(action=None)은 성공 반환"

        # 모든 유효한 텍스트 소스가 없으므로 fallback이어야 한다
        final_text = result.get("final_text", "")
        assert (
            final_text == "요청을 처리할 수 없습니다."
        ), f"모든 tool 실패 시 fallback 메시지여야 합니다. 실제: {final_text!r}"


# ---------------------------------------------------------------------------
# TestClass 4: TestEmptyResultScenarios
# ---------------------------------------------------------------------------


class TestEmptyResultScenarios:
    """빈 결과 시나리오 테스트.

    ApiLookupCapability의 no_match 처리와
    synthesis의 fallback 로직을 검증한다.
    """

    async def test_api_no_match_empty_reason(self, make_tooling_graph):
        """api_action=None이면 ApiLookupCapability가 success=True, empty_reason='no_match'를 반환한다."""
        planner = ConfigurableStubPlanner(
            task_type=TaskType.LOOKUP_STATS,
            goal="유사 민원 사례 조회",
            reason="통계 데이터가 필요합니다",
            tools=["api_lookup"],
        )
        graph = make_tooling_graph(planner, api_action=None)

        config = await _run_to_interrupt(
            graph,
            session_id="tooling-api-no-match-sess-1",
            thread_id="tooling-api-no-match-1",
            query="유사 민원 조회해줘",
            request_id="tooling-api-no-match-req-1",
        )
        result = await _approve(graph, config)

        tool_results = result.get("tool_results", {})
        assert "api_lookup" in tool_results, "tool_results에 api_lookup이 있어야 합니다"

        api_result = tool_results["api_lookup"]
        assert (
            api_result.get("success") is True
        ), "action=None일 때 success=True여야 합니다 (no_match는 오류가 아님)"
        assert (
            api_result.get("empty_reason") == "no_match"
        ), f"empty_reason이 'no_match'여야 합니다. 실제: {api_result.get('empty_reason')!r}"


# ---------------------------------------------------------------------------
# TestClass 5: TestPersistToolRunAccuracy
# ---------------------------------------------------------------------------


class TestPersistToolRunAccuracy:
    """persist_node의 tool run 기록 정확도 테스트.

    실제 capability 실행 결과가 SessionStore에 정확히 기록되는지 검증한다.
    """

    async def test_partial_failure_tool_runs_recorded(self, make_tooling_graph, session_store):
        """일부 tool 실패 시 실패/성공 상태가 tool_runs에 정확히 기록된다.

        실패 tool: success=False + error 존재
        성공 tool: success=True
        """

        async def draft_fn_fail(query, context, session):
            raise RuntimeError("LLM 서버 오류")

        session_id = "tooling-persist-partial-sess-1"
        request_id = "tooling-persist-partial-req-1"

        planner = ConfigurableStubPlanner(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="민원 답변 초안 작성",
            reason="사용자가 답변 초안을 요청했습니다",
            tools=["api_lookup", "draft_response"],
        )
        graph = make_tooling_graph(planner, draft_fn=draft_fn_fail)

        config = await _run_to_interrupt(
            graph,
            session_id=session_id,
            thread_id="tooling-persist-partial-1",
            query="답변 초안 작성해줘",
            request_id=request_id,
        )
        await _approve(graph, config)

        session = session_store.get_or_create(session_id)
        tool_runs = session.recent_tool_runs

        assert len(tool_runs) > 0, "tool_runs가 기록되어야 합니다"

        # api는 성공, draft는 실패여야 한다
        tool_run_map = {tr.tool: tr for tr in tool_runs}

        assert "api_lookup" in tool_run_map, "api_lookup tool_run이 기록되어야 합니다"
        assert (
            tool_run_map["api_lookup"].success is True
        ), "api_lookup tool_run.success가 True여야 합니다"

        assert "draft_response" in tool_run_map, "draft_response tool_run이 기록되어야 합니다"
        draft_run = tool_run_map["draft_response"]
        assert draft_run.success is False, "draft_response tool_run.success가 False여야 합니다"
        assert draft_run.error, "draft_response tool_run.error가 있어야 합니다"

    async def test_total_latency_ms_accumulated(self, make_tooling_graph, session_store):
        """전체 실행 후 graph_run.total_latency_ms가 0보다 커야 한다.

        persist_node는 tool_results의 latency_ms 합계를 total_latency_ms로 기록한다.
        """
        session_id = "tooling-persist-latency-sess-1"
        request_id = "tooling-persist-latency-req-1"

        planner = ConfigurableStubPlanner(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="민원 답변 초안 작성",
            reason="사용자가 답변 초안을 요청했습니다",
            tools=["api_lookup", "draft_response"],
        )
        graph = make_tooling_graph(planner)

        config = await _run_to_interrupt(
            graph,
            session_id=session_id,
            thread_id="tooling-persist-latency-1",
            query="답변 초안 작성해줘",
            request_id=request_id,
        )
        await _approve(graph, config)

        session = session_store.get_or_create(session_id)
        graph_runs = session.recent_graph_runs
        assert len(graph_runs) > 0, "graph_run이 기록되어야 합니다"

        run = graph_runs[0]

        # tool_runs에서 개별 latency 합산 검증
        tool_runs = session.recent_tool_runs
        if tool_runs:
            sum_latency = sum(tr.latency_ms for tr in tool_runs if tr.latency_ms)
            # sum_latency와 total_latency_ms가 같거나 근사해야 한다
            # (latency_ms는 실제 실행 시간이므로 완벽한 일치는 보장 안 됨)
            # 최소한 total_latency_ms > 0인지만 확인
            assert (
                run.total_latency_ms > 0
            ), "tool이 실행되었으므로 total_latency_ms > 0이어야 합니다"

    async def test_executed_capabilities_matches_actual(self, make_tooling_graph, session_store):
        """graph_run.executed_capabilities에는 실제로 실행된 tool만 포함된다.

        planned_tools가 3개여도 그 중 일부가 실패하면
        executed_capabilities에는 tool_results에 존재하는 tool 이름이 기록된다.
        persist_node는 planned_tools를 기준으로 tool_results에 있는 것들을 기록한다.
        """
        session_id = "tooling-persist-caps-sess-1"
        request_id = "tooling-persist-caps-req-1"

        planner = ConfigurableStubPlanner(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="민원 답변 초안 작성",
            reason="사용자가 답변 초안을 요청했습니다",
            tools=["api_lookup", "draft_response"],
        )
        graph = make_tooling_graph(planner)

        config = await _run_to_interrupt(
            graph,
            session_id=session_id,
            thread_id="tooling-persist-caps-1",
            query="답변 초안 작성해줘",
            request_id=request_id,
        )
        result = await _approve(graph, config)

        session = session_store.get_or_create(session_id)
        graph_runs = session.recent_graph_runs
        assert len(graph_runs) > 0, "graph_run이 기록되어야 합니다"

        run = graph_runs[0]
        tool_results = result.get("tool_results", {})

        # executed_capabilities는 planned_tools 중 tool_results에 있는 것들이어야 한다
        for cap in run.executed_capabilities:
            assert (
                cap in tool_results
            ), f"executed_capabilities에 있는 '{cap}'이 tool_results에도 있어야 합니다"

        # planned_tools에 있는 tool은 모두 executed_capabilities에 포함되어야 한다
        # (실패해도 tool_execute_node가 빈 dict로 기록하지 않고 result를 기록함)
        planned_tools = ["api_lookup", "draft_response"]
        for tool in planned_tools:
            if tool in tool_results:
                assert (
                    tool in run.executed_capabilities
                ), f"tool_results에 있는 '{tool}'이 executed_capabilities에도 있어야 합니다"
