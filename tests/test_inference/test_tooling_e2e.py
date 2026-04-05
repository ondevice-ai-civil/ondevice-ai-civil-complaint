"""LangGraph tooling E2E 통합 테스트.

Issue #162: tooling 계층 전체를 end-to-end로 검증한다.

test_orchestration_e2e.py와의 차이:
  - orchestration E2E는 StubExecutorAdapter로 graph 흐름만 검증
  - 이 파일은 실제 capability 인스턴스 + mock execute_fn으로
    capability→adapter→node 파이프라인을 검증

실제 capability 인스턴스(RagSearchCapability, ApiLookupCapability,
DraftCivilResponseCapability, AppendEvidenceCapability)를 사용하고
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
from src.inference.graph.capabilities.append_evidence import AppendEvidenceCapability
from src.inference.graph.capabilities.draft_civil_response import DraftCivilResponseCapability
from src.inference.graph.capabilities.rag_search import RagSearchCapability
from src.inference.graph.executor_adapter import RegistryExecutorAdapter
from src.inference.graph.planner_adapter import PlannerAdapter
from src.inference.graph.state import ApprovalStatus, TaskType, ToolPlan
from src.inference.session_context import SessionStore

os.environ.setdefault("SKIP_MODEL_LOAD", "true")


# ---------------------------------------------------------------------------
# Helper: registry factory
# ---------------------------------------------------------------------------


def _make_registry(
    rag_fn=None,
    api_action=None,
    draft_fn=None,
    evidence_fn=None,
) -> Dict[str, Any]:
    """실제 capability 인스턴스로 구성된 registry를 생성한다.

    각 capability는 주입된 mock execute_fn 클로저를 사용하여
    capability->adapter->node 파이프라인을 실제로 거친다.
    """
    if rag_fn is None:

        async def rag_fn(query, context, session):
            return {
                "results": [
                    {
                        "title": "테스트 문서",
                        "content": "테스트 내용입니다.",
                        "score": 0.9,
                        "source_type": "local",
                        "doc_id": "test-doc-001",
                    }
                ],
                "context_text": "",
                "query": query,
            }

    if draft_fn is None:

        async def draft_fn(query, context, session):
            return {"text": f"[기본 초안] {query}에 대한 답변입니다."}

    if evidence_fn is None:

        async def evidence_fn(query, context, session):
            return {"text": f"[기본 근거] {query}에 대한 근거입니다."}

    return {
        "rag_search": RagSearchCapability(execute_fn=rag_fn),
        "api_lookup": ApiLookupCapability(action=api_action),
        "draft_civil_response": DraftCivilResponseCapability(execute_fn=draft_fn),
        "append_evidence": AppendEvidenceCapability(execute_fn=evidence_fn),
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

    def _make(planner, rag_fn=None, api_action=None, draft_fn=None, evidence_fn=None):
        registry = _make_registry(rag_fn, api_action, draft_fn, evidence_fn)
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


# ---------------------------------------------------------------------------
# TestClass 1: TestDraftResponsePipeline
# ---------------------------------------------------------------------------


class TestDraftResponsePipeline:
    """DRAFT_RESPONSE 파이프라인 E2E 테스트.

    실제 RagSearchCapability, ApiLookupCapability, DraftCivilResponseCapability
    인스턴스를 사용하여 capability->adapter->node 파이프라인을 검증한다.
    """

    def test_draft_response_full_pipeline(self, make_tooling_graph):
        """rag+api+draft 3-tool 콤보: draft 텍스트가 final_text로 우선 선택된다.

        rag는 결과 반환, api_action=None(빈 결과), draft는 텍스트 반환.
        final_text가 draft 텍스트와 일치하고, 3개 tool 모두 tool_results에 존재한다.
        """
        expected_draft_text = "민원 답변 초안: 도로 파손 관련 조치를 취하겠습니다."

        async def draft_fn(query, context, session):
            return {"text": expected_draft_text}

        planner = ConfigurableStubPlanner(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="민원 답변 초안 작성",
            reason="사용자가 답변 초안을 요청했습니다",
            tools=["rag_search", "api_lookup", "draft_civil_response"],
        )
        graph = make_tooling_graph(planner, draft_fn=draft_fn)

        config = _run_to_interrupt(
            graph,
            session_id="tooling-draft-full-sess-1",
            thread_id="tooling-draft-full-1",
            query="도로 파손 민원 답변 초안 작성해줘",
            request_id="tooling-draft-full-req-1",
        )
        result = _approve(graph, config)

        final_text = result.get("final_text", "")
        tool_results = result.get("tool_results", {})

        assert (
            expected_draft_text in final_text
        ), f"final_text에 draft 텍스트가 포함되어야 합니다. 실제: {final_text!r}"
        assert "rag_search" in tool_results, "tool_results에 rag_search가 있어야 합니다"
        assert "api_lookup" in tool_results, "tool_results에 api_lookup이 있어야 합니다"
        assert (
            "draft_civil_response" in tool_results
        ), "tool_results에 draft_civil_response가 있어야 합니다"

    def test_draft_response_synthesis_prioritizes_draft_text(self, make_tooling_graph):
        """draft_civil_response 텍스트가 rag 결과보다 우선 선택된다.

        synthesis_node의 _extract_final_text 우선순위 검증:
        draft_civil_response.text > rag formatted results
        """
        draft_text = "초안 텍스트: 민원에 대해 답변드립니다."

        async def draft_fn(query, context, session):
            return {"text": draft_text}

        async def rag_fn(query, context, session):
            return {
                "results": [
                    {
                        "title": "법령 문서",
                        "content": "관련 법령 내용입니다.",
                        "score": 0.95,
                        "source_type": "local",
                        "doc_id": "test-doc-001",
                    }
                ],
                "context_text": "",
                "query": query,
            }

        planner = ConfigurableStubPlanner(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="민원 답변 초안 작성",
            reason="사용자가 답변 초안을 요청했습니다",
            tools=["rag_search", "draft_civil_response"],
        )
        graph = make_tooling_graph(planner, rag_fn=rag_fn, draft_fn=draft_fn)

        config = _run_to_interrupt(
            graph,
            session_id="tooling-draft-priority-sess-1",
            thread_id="tooling-draft-priority-1",
            query="답변 초안 작성해줘",
            request_id="tooling-draft-priority-req-1",
        )
        result = _approve(graph, config)

        final_text = result.get("final_text", "")
        assert (
            draft_text in final_text
        ), f"draft_civil_response 텍스트가 final_text에 포함되어야 합니다. 실제: {final_text!r}"
        # rag 결과가 아닌 draft 텍스트가 최우선이어야 한다
        assert (
            "[로컬 문서 근거]" not in final_text
        ), "draft가 성공했을 때 rag 포맷 출력이 사용되면 안 됩니다"

    def test_lookup_stats_api_only(self, make_tooling_graph):
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

        config = _run_to_interrupt(
            graph,
            session_id="tooling-lookup-stats-sess-1",
            thread_id="tooling-lookup-stats-1",
            query="유사 민원 사례 조회해줘",
            request_id="tooling-lookup-stats-req-1",
        )
        result = _approve(graph, config)

        tool_results = result.get("tool_results", {})
        assert "api_lookup" in tool_results, "tool_results에 api_lookup이 있어야 합니다"
        assert "rag_search" not in tool_results, "LOOKUP_STATS에서 rag_search가 실행되면 안 됩니다"

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
# TestClass 2: TestEvidenceAugmentationPipeline
# ---------------------------------------------------------------------------


class TestEvidenceAugmentationPipeline:
    """APPEND_EVIDENCE 파이프라인 E2E 테스트.

    실제 capability 인스턴스를 사용하여 rag+api+append_evidence 체인을 검증한다.
    """

    def test_append_evidence_merges_rag_and_api(self, make_tooling_graph):
        """3-tool 콤보: rag 결과와 api context가 append_evidence execute_fn에 전달된다.

        append_evidence의 execute_fn이 병합된 텍스트를 반환하고,
        final_text가 그 텍스트와 일치한다.
        """
        evidence_text = "RAG + API 병합 근거: 관련 법령 제3조에 따라 처리됩니다."

        async def rag_fn(query, context, session):
            return {
                "results": [
                    {
                        "title": "관련 법령",
                        "content": "법령 내용입니다.",
                        "score": 0.85,
                        "source_type": "local",
                        "doc_id": "test-doc-001",
                    }
                ],
                "context_text": "법령 검색 컨텍스트",
                "query": query,
            }

        async def evidence_fn(query, context, session):
            return {"text": evidence_text}

        planner = ConfigurableStubPlanner(
            task_type=TaskType.APPEND_EVIDENCE,
            goal="민원 답변 근거 보강",
            reason="사용자가 근거 보강을 요청했습니다",
            tools=["rag_search", "api_lookup", "append_evidence"],
        )
        graph = make_tooling_graph(planner, rag_fn=rag_fn, evidence_fn=evidence_fn)

        config = _run_to_interrupt(
            graph,
            session_id="tooling-evidence-merge-sess-1",
            thread_id="tooling-evidence-merge-1",
            query="근거를 보강해줘",
            request_id="tooling-evidence-merge-req-1",
        )
        result = _approve(graph, config)

        final_text = result.get("final_text", "")
        assert (
            evidence_text in final_text
        ), f"final_text에 append_evidence 텍스트가 포함되어야 합니다. 실제: {final_text!r}"

    def test_evidence_context_chaining(self, make_tooling_graph):
        """append_evidence execute_fn 호출 시 context에 rag_search/api_lookup 결과가 포함된다.

        tool_execute_node는 도구를 순차 실행하며 누적 컨텍스트에 이전 결과를 반영한다.
        accumulated_context에 rag_search와 api_lookup 결과가 존재하는지 검증한다.
        """
        received_context: Dict[str, Any] = {}

        async def rag_fn(query, context, session):
            return {
                "results": [
                    {
                        "title": "법령 문서",
                        "content": "관련 법령",
                        "score": 0.9,
                        "source_type": "local",
                        "doc_id": "test-doc-001",
                    }
                ],
                "context_text": "법령 컨텍스트",
                "query": query,
            }

        async def evidence_fn(query, context, session):
            # context를 캡처하여 rag/api 결과가 포함되는지 확인
            received_context.update(context)
            return {"text": "근거 보강 완료"}

        planner = ConfigurableStubPlanner(
            task_type=TaskType.APPEND_EVIDENCE,
            goal="민원 답변 근거 보강",
            reason="사용자가 근거 보강을 요청했습니다",
            tools=["rag_search", "api_lookup", "append_evidence"],
        )
        graph = make_tooling_graph(planner, rag_fn=rag_fn, evidence_fn=evidence_fn)

        config = _run_to_interrupt(
            graph,
            session_id="tooling-evidence-chain-sess-1",
            thread_id="tooling-evidence-chain-1",
            query="근거를 보강해줘",
            request_id="tooling-evidence-chain-req-1",
        )
        _approve(graph, config)

        # append_evidence execute_fn이 호출될 때 rag_search 결과가 누적 컨텍스트에 있어야 한다
        assert (
            "rag_search" in received_context
        ), "append_evidence 호출 시 context에 rag_search 결과가 있어야 합니다"
        assert isinstance(
            received_context["rag_search"], dict
        ), "rag_search 결과는 dict이어야 합니다"
        assert (
            "api_lookup" in received_context
        ), "append_evidence 호출 시 context에 api_lookup 결과가 있어야 합니다"
        assert isinstance(
            received_context["api_lookup"], dict
        ), "api_lookup 결과는 dict이어야 합니다"

    def test_evidence_with_empty_rag(self, make_tooling_graph):
        """rag 결과가 없을 때도 append_evidence 파이프라인이 완료된다.

        rag가 빈 결과를 반환해도 api_lookup과 append_evidence는 실행되고
        파이프라인이 정상 완료된다.
        """
        evidence_text = "API 결과만으로 보강된 근거입니다."

        async def rag_fn_empty(query, context, session):
            # 빈 결과 반환 (no_match)
            return {"results": [], "context_text": "", "query": query}

        async def evidence_fn(query, context, session):
            return {"text": evidence_text}

        planner = ConfigurableStubPlanner(
            task_type=TaskType.APPEND_EVIDENCE,
            goal="민원 답변 근거 보강",
            reason="사용자가 근거 보강을 요청했습니다",
            tools=["rag_search", "api_lookup", "append_evidence"],
        )
        graph = make_tooling_graph(planner, rag_fn=rag_fn_empty, evidence_fn=evidence_fn)

        config = _run_to_interrupt(
            graph,
            session_id="tooling-evidence-empty-rag-sess-1",
            thread_id="tooling-evidence-empty-rag-1",
            query="근거를 보강해줘",
            request_id="tooling-evidence-empty-rag-req-1",
        )
        result = _approve(graph, config)

        # 파이프라인이 정상 완료되어야 한다
        assert result.get("approval_status") == ApprovalStatus.APPROVED.value
        final_text = result.get("final_text", "")
        assert final_text, "rag가 비어도 final_text가 생성되어야 합니다"

        # append_evidence가 실행되어 evidence_text가 final_text에 포함되어야 한다
        assert (
            evidence_text in final_text
        ), f"append_evidence 텍스트가 final_text에 포함되어야 합니다. 실제: {final_text!r}"


# ---------------------------------------------------------------------------
# TestClass 3: TestPartialFailureE2E
# ---------------------------------------------------------------------------


class TestPartialFailureE2E:
    """부분 실패 시나리오 E2E 테스트.

    일부 tool이 실패해도 나머지 tool이 계속 실행되는 resilience를 검증한다.
    """

    def test_rag_timeout_other_tools_continue(self, make_tooling_graph):
        """rag execute_fn이 타임아웃되어도 draft가 실행되고 final_text가 생성된다.

        RagSearchCapability는 execute_fn에서 asyncio.TimeoutError가 발생하면
        success=False, empty_reason='provider_error'를 반환한다.
        tool_execute_node는 실패한 tool 이후에도 draft 실행을 계속 진행한다.
        """
        draft_text = "타임아웃 이후 생성된 민원 답변 초안입니다."

        async def rag_fn_timeout(query, context, session):
            # asyncio.TimeoutError를 직접 발생시켜 타임아웃 시뮬레이션
            raise asyncio.TimeoutError("RAG 검색 타임아웃")

        async def draft_fn(query, context, session):
            return {"text": draft_text}

        planner = ConfigurableStubPlanner(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="민원 답변 초안 작성",
            reason="사용자가 답변 초안을 요청했습니다",
            tools=["rag_search", "draft_civil_response"],
        )
        graph = make_tooling_graph(planner, rag_fn=rag_fn_timeout, draft_fn=draft_fn)

        config = _run_to_interrupt(
            graph,
            session_id="tooling-rag-timeout-sess-1",
            thread_id="tooling-rag-timeout-1",
            query="답변 초안 작성해줘",
            request_id="tooling-rag-timeout-req-1",
        )
        result = _approve(graph, config)

        tool_results = result.get("tool_results", {})

        # rag가 실패해야 한다 (타임아웃)
        assert "rag_search" in tool_results, "rag_search가 tool_results에 있어야 합니다"
        rag_result = tool_results["rag_search"]
        assert rag_result.get("success") is False, "rag 타임아웃 시 success=False여야 합니다"

        # draft가 실행되어 final_text가 생성되어야 한다
        assert (
            "draft_civil_response" in tool_results
        ), "draft_civil_response가 tool_results에 있어야 합니다"
        draft_result = tool_results["draft_civil_response"]
        assert draft_result.get("success") is True, "draft가 성공해야 합니다"
        assert draft_text in result.get(
            "final_text", ""
        ), "draft 텍스트가 final_text에 포함되어야 합니다"

    def test_api_failure_draft_still_runs(self, session_store):
        """api_lookup이 예외를 발생시켜도 draft_civil_response가 실행된다.

        RegistryExecutorAdapter의 예외 처리가 api 실패를 잡고
        다음 tool인 draft_civil_response를 실행한다.
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
        registry = _make_registry(draft_fn=draft_fn)
        registry["api_lookup"] = ApiLookupCapability(action=FailingApiAction())
        executor = RegistryExecutorAdapter(tool_registry=registry, session_store=session_store)

        planner = ConfigurableStubPlanner(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="민원 답변 초안 작성",
            reason="사용자가 답변 초안을 요청했습니다",
            tools=["rag_search", "api_lookup", "draft_civil_response"],
        )

        graph = build_govon_graph(
            planner_adapter=planner,
            executor_adapter=executor,
            session_store=session_store,
            checkpointer=MemorySaver(),
        )

        config = _run_to_interrupt(
            graph,
            session_id="tooling-api-fail-sess-1",
            thread_id="tooling-api-fail-1",
            query="답변 초안 작성해줘",
            request_id="tooling-api-fail-req-1",
        )
        result = _approve(graph, config)

        tool_results = result.get("tool_results", {})
        assert "api_lookup" in tool_results, "api_lookup이 tool_results에 있어야 합니다"
        api_result = tool_results["api_lookup"]
        assert api_result.get("success") is False, "API 실패 시 success=False여야 합니다"

        assert (
            "draft_civil_response" in tool_results
        ), "api 실패 후에도 draft_civil_response가 실행되어야 합니다"
        assert draft_text in result.get(
            "final_text", ""
        ), "draft 텍스트가 final_text에 포함되어야 합니다"

    def test_draft_exception_caught_by_adapter(self, make_tooling_graph):
        """draft execute_fn이 RuntimeError를 발생시키면 어댑터가 잡고 success=False를 반환한다.

        DraftCivilResponseCapability는 execute() 내부 try/except 없이
        execute_fn에서 발생한 예외가 CapabilityBase.__call__을 통해
        RegistryExecutorAdapter까지 전파된다. 어댑터가 예외를 잡아 success=False로 반환한다.
        RagSearchCapability와 달리 capability 자체에서 예외를 흡수하지 않는다.
        """

        async def draft_fn_raises(query, context, session):
            raise RuntimeError("LLM 서버 오류")

        planner = ConfigurableStubPlanner(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="민원 답변 초안 작성",
            reason="사용자가 답변 초안을 요청했습니다",
            tools=["rag_search", "draft_civil_response"],
        )
        graph = make_tooling_graph(planner, draft_fn=draft_fn_raises)

        config = _run_to_interrupt(
            graph,
            session_id="tooling-draft-except-sess-1",
            thread_id="tooling-draft-except-1",
            query="답변 초안 작성해줘",
            request_id="tooling-draft-except-req-1",
        )
        result = _approve(graph, config)

        tool_results = result.get("tool_results", {})
        assert (
            "draft_civil_response" in tool_results
        ), "draft_civil_response가 tool_results에 있어야 합니다"
        draft_result = tool_results["draft_civil_response"]
        assert (
            draft_result.get("success") is False
        ), "draft 예외 시 tool_results['draft_civil_response']['success']==False여야 합니다"
        assert draft_result.get("error"), "draft 예외 시 error 필드가 있어야 합니다"

    def test_all_tools_fail_synthesis_fallback(self, make_tooling_graph):
        """모든 tool이 실패하면 final_text가 fallback 메시지가 된다.

        RagSearchCapability는 execute() 내부에 자체 try/except가 있어서
        execute_fn에서 발생한 예외를 success=False LookupResult로 변환하고
        RegistryExecutorAdapter까지 전파하지 않는다.
        DraftCivilResponseCapability는 execute() 내부 try/except가 없으므로
        예외가 RegistryExecutorAdapter까지 전파되어 어댑터가 잡는다.

        rag execute_fn에서 RuntimeError 발생 → RagSearchCapability.execute()가 잡아 success=False 반환.
        draft execute_fn에서 RuntimeError 발생 → CapabilityBase.__call__을 통해 어댑터로 전파 → success=False.
        api_action=None → ApiLookupCapability가 success=True, 빈 결과 반환.

        유효한 텍스트 소스가 없으므로 _extract_final_text는 "요청을 처리할 수 없습니다."를 반환한다.
        """

        async def rag_fn_fail(query, context, session):
            raise RuntimeError("RAG 엔진 오류")

        async def draft_fn_fail(query, context, session):
            raise RuntimeError("LLM 서버 오류")

        planner = ConfigurableStubPlanner(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="민원 답변 초안 작성",
            reason="사용자가 답변 초안을 요청했습니다",
            tools=["rag_search", "api_lookup", "draft_civil_response"],
        )
        # api_action=None: success=True지만 빈 결과, context_text 없음
        graph = make_tooling_graph(
            planner, rag_fn=rag_fn_fail, api_action=None, draft_fn=draft_fn_fail
        )

        config = _run_to_interrupt(
            graph,
            session_id="tooling-all-fail-sess-1",
            thread_id="tooling-all-fail-1",
            query="답변 초안 작성해줘",
            request_id="tooling-all-fail-req-1",
        )
        result = _approve(graph, config)

        # rag와 draft가 실패해야 한다
        tool_results = result.get("tool_results", {})
        rag_result = tool_results.get("rag_search", {})
        draft_result = tool_results.get("draft_civil_response", {})
        assert rag_result.get("success") is False, "rag 실패 확인"
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
    """빈 결과 / 저신뢰도 시나리오 테스트.

    RagSearchCapability의 no_match, low_confidence 처리와
    synthesis의 fallback 로직을 검증한다.
    """

    def test_rag_no_match_empty_reason(self, make_tooling_graph):
        """rag execute_fn이 빈 결과를 반환하면 LookupResult.empty_reason='no_match'가 된다.

        RagSearchCapability는 results가 없으면 success=True, empty_reason='no_match'를 반환한다.
        tool_results에서 이 empty_reason을 확인한다.
        """

        async def rag_fn_empty(query, context, session):
            return {"results": [], "context_text": "", "query": query}

        planner = ConfigurableStubPlanner(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="민원 답변 초안 작성",
            reason="사용자가 답변 초안을 요청했습니다",
            tools=["rag_search"],
        )
        graph = make_tooling_graph(planner, rag_fn=rag_fn_empty)

        config = _run_to_interrupt(
            graph,
            session_id="tooling-rag-no-match-sess-1",
            thread_id="tooling-rag-no-match-1",
            query="법령 검색해줘",
            request_id="tooling-rag-no-match-req-1",
        )
        result = _approve(graph, config)

        tool_results = result.get("tool_results", {})
        assert "rag_search" in tool_results, "tool_results에 rag_search가 있어야 합니다"

        rag_result = tool_results["rag_search"]
        assert (
            rag_result.get("success") is True
        ), "빈 결과일 때 success=True여야 합니다 (no_match는 오류가 아님)"
        assert (
            rag_result.get("empty_reason") == "no_match"
        ), f"empty_reason이 'no_match'여야 합니다. 실제: {rag_result.get('empty_reason')!r}"

    def test_rag_low_confidence_includes_results(self, make_tooling_graph):
        """모든 결과의 score가 0.3 미만이면 empty_reason='low_confidence'지만 results는 있다.

        RagSearchCapability는 low_confidence 케이스에서도 results를 반환한다.
        """

        async def rag_fn_low_conf(query, context, session):
            return {
                "results": [
                    {"title": "관련 없는 문서", "content": "내용", "score": 0.1},
                    {"title": "관련 없는 문서2", "content": "내용2", "score": 0.15},
                ],
                "context_text": "",
                "query": query,
            }

        planner = ConfigurableStubPlanner(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="민원 답변 초안 작성",
            reason="사용자가 답변 초안을 요청했습니다",
            tools=["rag_search"],
        )
        graph = make_tooling_graph(planner, rag_fn=rag_fn_low_conf)

        config = _run_to_interrupt(
            graph,
            session_id="tooling-rag-low-conf-sess-1",
            thread_id="tooling-rag-low-conf-1",
            query="법령 검색해줘",
            request_id="tooling-rag-low-conf-req-1",
        )
        result = _approve(graph, config)

        tool_results = result.get("tool_results", {})
        rag_result = tool_results.get("rag_search", {})

        assert rag_result.get("success") is True, "low_confidence일 때 success=True여야 합니다"
        assert (
            rag_result.get("empty_reason") == "low_confidence"
        ), f"empty_reason이 'low_confidence'여야 합니다. 실제: {rag_result.get('empty_reason')!r}"
        # low_confidence여도 results는 포함되어야 한다
        assert (
            len(rag_result.get("results", [])) > 0
        ), "low_confidence일 때도 results가 반환되어야 합니다"

    def test_synthesis_with_only_rag_results(self, make_tooling_graph):
        """rag만 성공하고 draft가 실패할 때 final_text가 rag 포맷 출력을 포함한다.

        _extract_final_text의 legacy fallback:
        draft가 없고 evidence도 없으면 rag_search.results 기반으로 출력한다.
        """

        async def rag_fn(query, context, session):
            return {
                "results": [
                    {
                        "title": "도로법 제3조",
                        "content": "도로 유지 보수에 관한 사항",
                        "score": 0.9,
                        "source_type": "local",
                        "doc_id": "test-doc-001",
                    }
                ],
                "context_text": "",
                "query": query,
            }

        async def draft_fn_fail(query, context, session):
            raise RuntimeError("LLM 서버 오류")

        planner = ConfigurableStubPlanner(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="민원 답변 초안 작성",
            reason="사용자가 답변 초안을 요청했습니다",
            tools=["rag_search", "draft_civil_response"],
        )
        graph = make_tooling_graph(planner, rag_fn=rag_fn, draft_fn=draft_fn_fail)

        config = _run_to_interrupt(
            graph,
            session_id="tooling-only-rag-sess-1",
            thread_id="tooling-only-rag-1",
            query="법령 찾아줘",
            request_id="tooling-only-rag-req-1",
        )
        result = _approve(graph, config)

        tool_results = result.get("tool_results", {})
        rag_result = tool_results.get("rag_search", {})
        assert rag_result.get("success") is True, "rag가 성공해야 합니다"
        assert (
            tool_results.get("draft_civil_response", {}).get("success") is False
        ), "draft가 실패해야 합니다"

        final_text = result.get("final_text", "")
        # rag 결과가 있고 draft가 실패했으므로 rag 기반 출력이 있어야 한다
        # _extract_final_text는 evidence/draft가 없으면 [로컬 문서 근거] 포맷 사용
        assert (
            "[로컬 문서 근거]" in final_text
        ), f"rag만 성공 시 '[로컬 문서 근거]' 포맷이어야 합니다. 실제: {final_text!r}"
        assert (
            "도로법" in final_text
        ), f"rag 결과의 제목이 final_text에 포함되어야 합니다. 실제: {final_text!r}"


# ---------------------------------------------------------------------------
# TestClass 5: TestPersistToolRunAccuracy
# ---------------------------------------------------------------------------


class TestPersistToolRunAccuracy:
    """persist_node의 tool run 기록 정확도 테스트.

    실제 capability 실행 결과가 SessionStore에 정확히 기록되는지 검증한다.
    """

    def test_partial_failure_tool_runs_recorded(self, make_tooling_graph, session_store):
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
            tools=["rag_search", "draft_civil_response"],
        )
        graph = make_tooling_graph(planner, draft_fn=draft_fn_fail)

        config = _run_to_interrupt(
            graph,
            session_id=session_id,
            thread_id="tooling-persist-partial-1",
            query="답변 초안 작성해줘",
            request_id=request_id,
        )
        _approve(graph, config)

        session = session_store.get_or_create(session_id)
        tool_runs = session.recent_tool_runs

        assert len(tool_runs) > 0, "tool_runs가 기록되어야 합니다"

        # rag는 성공, draft는 실패여야 한다
        tool_run_map = {tr.tool: tr for tr in tool_runs}

        assert "rag_search" in tool_run_map, "rag_search tool_run이 기록되어야 합니다"
        assert (
            tool_run_map["rag_search"].success is True
        ), "rag_search tool_run.success가 True여야 합니다"

        assert (
            "draft_civil_response" in tool_run_map
        ), "draft_civil_response tool_run이 기록되어야 합니다"
        draft_run = tool_run_map["draft_civil_response"]
        assert (
            draft_run.success is False
        ), "draft_civil_response tool_run.success가 False여야 합니다"
        assert draft_run.error, "draft_civil_response tool_run.error가 있어야 합니다"

    def test_total_latency_ms_accumulated(self, make_tooling_graph, session_store):
        """전체 실행 후 graph_run.total_latency_ms가 0보다 커야 한다.

        persist_node는 tool_results의 latency_ms 합계를 total_latency_ms로 기록한다.
        """
        session_id = "tooling-persist-latency-sess-1"
        request_id = "tooling-persist-latency-req-1"

        planner = ConfigurableStubPlanner(
            task_type=TaskType.DRAFT_RESPONSE,
            goal="민원 답변 초안 작성",
            reason="사용자가 답변 초안을 요청했습니다",
            tools=["rag_search", "draft_civil_response"],
        )
        graph = make_tooling_graph(planner)

        config = _run_to_interrupt(
            graph,
            session_id=session_id,
            thread_id="tooling-persist-latency-1",
            query="답변 초안 작성해줘",
            request_id=request_id,
        )
        _approve(graph, config)

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

    def test_executed_capabilities_matches_actual(self, make_tooling_graph, session_store):
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
            tools=["rag_search", "api_lookup", "draft_civil_response"],
        )
        graph = make_tooling_graph(planner)

        config = _run_to_interrupt(
            graph,
            session_id=session_id,
            thread_id="tooling-persist-caps-1",
            query="답변 초안 작성해줘",
            request_id=request_id,
        )
        result = _approve(graph, config)

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
        planned_tools = ["rag_search", "api_lookup", "draft_civil_response"]
        for tool in planned_tools:
            if tool in tool_results:
                assert (
                    tool in run.executed_capabilities
                ), f"tool_results에 있는 '{tool}'이 executed_capabilities에도 있어야 합니다"
