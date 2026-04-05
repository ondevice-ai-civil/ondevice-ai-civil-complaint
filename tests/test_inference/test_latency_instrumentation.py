"""레이턴시 계측 및 병렬 실행 테스트.

Issue #163: node-level latency tracking과 parallel tool execution 검증.
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest

os.environ.setdefault("SKIP_MODEL_LOAD", "true")

from src.inference.graph.nodes import (
    persist_node,
    session_load_node,
    synthesis_node,
    tool_execute_node,
)
from src.inference.graph.state import ApprovalStatus


# ---------------------------------------------------------------------------
# Helper stubs
# ---------------------------------------------------------------------------


class SlowExecutor:
    """각 tool에 지정된 시간만큼 sleep하는 executor."""

    def __init__(self, latencies: dict[str, float] | None = None):
        self.latencies = latencies or {
            "rag_search": 0.2,
            "api_lookup": 0.2,
            "draft_civil_response": 0.1,
        }

    async def execute(self, tool_name: str, query: str, context: dict) -> dict:
        delay = self.latencies.get(tool_name, 0.05)
        await asyncio.sleep(delay)
        return {"success": True, "text": f"[stub] {tool_name}", "latency_ms": delay * 1000}

    def list_tools(self) -> list[str]:
        return list(self.latencies.keys())


class StubSessionStore:
    """최소한의 SessionStore 스텁."""

    class _Session:
        session_id = "test-session"

        def add_turn(self, role, content):
            pass

        def add_graph_run(self, **kwargs):
            pass

        def add_tool_run(self, **kwargs):
            pass

    def get_or_create(self, session_id=None):
        s = self._Session()
        if session_id:
            s.session_id = session_id
        return s


# ---------------------------------------------------------------------------
# Tests: node latency instrumentation
# ---------------------------------------------------------------------------


class TestNodeLatencyInstrumentation:
    """각 노드가 _node_latency_ms를 반환하는지 확인."""

    @pytest.mark.asyncio
    async def test_session_load_node_returns_latency(self):
        state = {
            "session_id": "test",
            "messages": [],
        }
        result = await session_load_node(state, session_store=StubSessionStore())
        assert "_node_latency_ms" in result
        assert isinstance(result["_node_latency_ms"], float)
        assert result["_node_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_synthesis_node_returns_latency(self):
        state = {
            "accumulated_context": {},
            "task_type": "",
        }
        result = await synthesis_node(state)
        assert "_node_latency_ms" in result
        assert isinstance(result["_node_latency_ms"], float)

    @pytest.mark.asyncio
    async def test_persist_node_returns_latency(self):
        state = {
            "session_id": "test",
            "request_id": "req-1",
            "messages": [],
            "approval_status": ApprovalStatus.APPROVED.value,
            "planned_tools": [],
            "tool_results": {},
            "final_text": "",
        }
        result = await persist_node(state, session_store=StubSessionStore())
        assert "_node_latency_ms" in result
        assert isinstance(result["_node_latency_ms"], float)

    @pytest.mark.asyncio
    async def test_tool_execute_returns_latency_and_per_tool(self):
        state = {
            "approval_status": ApprovalStatus.APPROVED.value,
            "planned_tools": ["rag_search"],
            "accumulated_context": {"query": "test"},
        }
        executor = SlowExecutor({"rag_search": 0.05})
        result = await tool_execute_node(state, executor_adapter=executor)
        assert "_node_latency_ms" in result
        assert "_tool_latencies" in result
        assert "rag_search" in result["_tool_latencies"]


# ---------------------------------------------------------------------------
# Tests: parallel tool execution
# ---------------------------------------------------------------------------


class TestParallelToolExecution:
    """독립 도구가 병렬로 실행되어 총 시간이 줄어드는지 확인."""

    @pytest.mark.asyncio
    async def test_independent_tools_run_in_parallel(self):
        """rag_search + api_lookup이 병렬로 실행되면 총 시간 < 두 개의 합."""
        latencies = {"rag_search": 0.2, "api_lookup": 0.2}
        executor = SlowExecutor(latencies)

        state = {
            "approval_status": ApprovalStatus.APPROVED.value,
            "planned_tools": ["rag_search", "api_lookup"],
            "accumulated_context": {"query": "test query"},
        }

        t0 = time.monotonic()
        result = await tool_execute_node(state, executor_adapter=executor)
        elapsed = time.monotonic() - t0

        # 병렬 실행이면 ~0.2초, 순차 실행이면 ~0.4초
        # 0.35초 미만이면 병렬로 간주
        assert elapsed < 0.35, (
            f"독립 도구가 병렬로 실행되어야 합니다. "
            f"소요: {elapsed:.3f}s (합: 0.4s, 기대: <0.35s)"
        )
        assert "rag_search" in result["tool_results"]
        assert "api_lookup" in result["tool_results"]

    @pytest.mark.asyncio
    async def test_dependent_tools_run_after_independent(self):
        """draft_civil_response는 independent 도구 이후 순차 실행된다."""
        executor = SlowExecutor({
            "rag_search": 0.05,
            "api_lookup": 0.05,
            "draft_civil_response": 0.05,
        })

        state = {
            "approval_status": ApprovalStatus.APPROVED.value,
            "planned_tools": ["rag_search", "api_lookup", "draft_civil_response"],
            "accumulated_context": {"query": "test"},
        }

        result = await tool_execute_node(state, executor_adapter=executor)

        assert len(result["tool_results"]) == 3
        assert "draft_civil_response" in result["tool_results"]
        # draft_civil_response는 rag/api 결과가 accumulated된 후 실행
        assert "rag_search" in result["accumulated_context"]
        assert "api_lookup" in result["accumulated_context"]

    @pytest.mark.asyncio
    async def test_parallel_execution_handles_exceptions(self):
        """병렬 실행 중 하나가 실패해도 나머지는 정상 처리된다."""

        class FailingExecutor:
            async def execute(self, tool_name, query, context):
                if tool_name == "api_lookup":
                    raise RuntimeError("API down")
                await asyncio.sleep(0.01)
                return {"success": True, "text": f"ok {tool_name}"}

            def list_tools(self):
                return ["rag_search", "api_lookup"]

        state = {
            "approval_status": ApprovalStatus.APPROVED.value,
            "planned_tools": ["rag_search", "api_lookup"],
            "accumulated_context": {"query": "test"},
        }

        result = await tool_execute_node(state, executor_adapter=FailingExecutor())
        # rag_search는 성공, api_lookup은 예외로 건너뜀
        assert "rag_search" in result["tool_results"]
        assert "api_lookup" not in result["tool_results"]


# ---------------------------------------------------------------------------
# Tests: defaults module
# ---------------------------------------------------------------------------


class TestCapabilityDefaults:
    """capability defaults 모듈 테스트."""

    def test_get_timeout_returns_default(self):
        from src.inference.graph.capabilities.defaults import get_timeout

        assert get_timeout("rag_search") == 15.0
        assert get_timeout("api_lookup") == 10.0
        assert get_timeout("draft_civil_response") == 30.0

    def test_get_timeout_env_override(self, monkeypatch):
        from src.inference.graph.capabilities.defaults import get_timeout

        monkeypatch.setenv("GOVON_TOOL_TIMEOUT_RAG_SEARCH", "25")
        assert get_timeout("rag_search") == 25.0

    def test_get_timeout_unknown_capability(self):
        from src.inference.graph.capabilities.defaults import get_timeout

        assert get_timeout("unknown_tool") == 10.0

    def test_get_max_retries(self):
        from src.inference.graph.capabilities.defaults import get_max_retries

        assert get_max_retries("api_lookup") == 1
        assert get_max_retries("rag_search") == 0
        assert get_max_retries("unknown") == 0
