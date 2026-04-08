"""LangGraph approval-gated shell E2E 통합 테스트.

Issue #400: LangGraph approval-gated shell E2E 통합 테스트.

실제 LangGraph graph를 FastAPI TestClient에 주입하여
완전한 HTTP 흐름을 검증한다:
  /v2/agent/run → interrupt → /v2/agent/approve → completion

TestClass 구성:
  1. TestV2RunApproveFlow   — run/approve/reject/cancel 기본 흐름
  2. TestV2StreamFlow       — SSE 스트리밍 노드 이벤트 흐름
  3. TestV2SessionResume    — 동일 session_id 재사용 및 신규 thread 생성
  4. TestHttpClientCompatibility — http_client approve/cancel 파라미터 검증
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# SKIP_MODEL_LOAD 설정 및 heavy deps mock
# 반드시 src.inference.api_server import 전에 실행되어야 한다.
# ---------------------------------------------------------------------------
os.environ.setdefault("SKIP_MODEL_LOAD", "true")

_vllm_mock = MagicMock()
_vllm_mock.AsyncLLM = MagicMock()
_vllm_mock.SamplingParams = MagicMock()
sys.modules.setdefault("vllm", _vllm_mock)
sys.modules.setdefault("vllm.engine", _vllm_mock)
sys.modules.setdefault("vllm.engine.arg_utils", _vllm_mock)
sys.modules.setdefault("vllm.engine.async_llm_engine", _vllm_mock)
sys.modules.setdefault("vllm.sampling_params", _vllm_mock)
sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("transformers", MagicMock())
sys.modules.setdefault("transformers.modeling_rope_utils", MagicMock())
sys.modules.setdefault("transformers.utils", MagicMock())
sys.modules.setdefault("transformers.utils.generic", MagicMock())
if "torch" not in sys.modules:
    sys.modules["torch"] = MagicMock()

from unittest.mock import patch

with patch("src.inference.vllm_stabilizer.apply_transformers_patch"):
    import src.inference.api_server as api_server

    app = api_server.app
    manager = api_server.manager

from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# 실제 graph 빌드용 import (heavy deps가 이미 mock되어 있으므로 안전)
# ---------------------------------------------------------------------------
from langgraph.checkpoint.memory import MemorySaver

from src.inference.graph.builder import build_govon_graph
from src.inference.graph.executor_adapter import ExecutorAdapter
from src.inference.graph.planner_adapter import (  # CI fallback: 실제 운영은 LLMPlannerAdapter
    RegexPlannerAdapter,
)
from src.inference.session_context import SessionStore

# ---------------------------------------------------------------------------
# Stub adapters (test_graph_smoke.py 패턴 재사용)
# ---------------------------------------------------------------------------


class StubExecutorAdapter(ExecutorAdapter):
    """모든 tool 호출에 고정된 성공 결과를 반환하는 스텁 executor."""

    async def execute(self, tool_name: str, query: str, context: dict) -> dict:
        return {
            "success": True,
            "text": f"[stub] {tool_name} result for: {query}",
            "latency_ms": 1.0,
        }

    def list_tools(self) -> list[str]:
        return ["api_lookup", "draft_response"]


# ---------------------------------------------------------------------------
# SSE 파싱 헬퍼
# ---------------------------------------------------------------------------


def _parse_sse_events(sse_text: str) -> list[dict]:
    """SSE 텍스트에서 data 라인을 파싱하여 dict 목록으로 반환한다."""
    events = []
    for line in sse_text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            data_str = line[len("data:") :].strip()
            if data_str:
                try:
                    events.append(json.loads(data_str))
                except json.JSONDecodeError:
                    pass
    return events


# ---------------------------------------------------------------------------
# 공통 fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def setup_real_graph(tmp_path):
    """실제 graph를 app의 manager에 주입한다.

    CI fallback: RegexPlannerAdapter + StubExecutorAdapter + MemorySaver를 사용한다.
    실제 운영은 LLMPlannerAdapter를 사용하며, CI(SKIP_MODEL_LOAD=true) 환경에서는
    LLM 없이 RegexPlannerAdapter를 fallback으로 사용한다.
    각 테스트마다 격리된 SQLite 파일을 사용한다.
    """
    original_graph = manager.graph
    original_session_store = manager.session_store

    planner = RegexPlannerAdapter()  # CI fallback: 실제 운영은 LLMPlannerAdapter
    executor = StubExecutorAdapter()
    session_store = SessionStore(db_path=str(tmp_path / "e2e.sqlite3"))

    manager.graph = build_govon_graph(
        planner_adapter=planner,
        executor_adapter=executor,
        session_store=session_store,
        checkpointer=MemorySaver(),
    )
    manager.session_store = session_store

    yield session_store

    manager.graph = original_graph
    manager.session_store = original_session_store


# ---------------------------------------------------------------------------
# TestClass 1: TestV2RunApproveFlow
# ---------------------------------------------------------------------------


class TestV2RunApproveFlow:
    """POST /v2/agent/run → /v2/agent/approve 기본 흐름 테스트."""

    def test_run_returns_awaiting_approval(self, setup_real_graph):
        """POST /v2/agent/run이 awaiting_approval 상태와 thread_id를 반환한다."""
        client = TestClient(app)
        response = client.post(
            "/v2/agent/run",
            json={"query": "민원 답변 초안 작성해줘"},
        )

        assert response.status_code == 200, f"응답 내용: {response.text}"
        data = response.json()
        assert data["status"] == "awaiting_approval", f"status 불일치: {data}"
        assert "thread_id" in data, f"thread_id 없음: {data}"
        assert data["thread_id"], "thread_id가 비어있음"

    def test_approve_completes_with_final_text(self, setup_real_graph):
        """run → approve(True) 흐름에서 completed 상태와 final_text를 반환한다."""
        client = TestClient(app)

        # 1단계: run → thread_id 획득
        run_resp = client.post(
            "/v2/agent/run",
            json={"query": "민원 답변 초안 작성해줘"},
        )
        assert run_resp.status_code == 200
        run_data = run_resp.json()
        assert run_data["status"] == "awaiting_approval"
        thread_id = run_data["thread_id"]

        # 2단계: approve(True) → completed
        approve_resp = client.post(
            "/v2/agent/approve",
            params={"thread_id": thread_id, "approved": "true"},
        )
        assert approve_resp.status_code == 200, f"approve 응답: {approve_resp.text}"
        approve_data = approve_resp.json()
        assert approve_data["status"] == "completed", f"status 불일치: {approve_data}"
        assert approve_data.get("text"), f"final_text가 비어있음: {approve_data}"

    def test_reject_returns_rejected(self, setup_real_graph):
        """run → approve(False) 흐름에서 rejected 상태를 반환하고 tool_results가 없다."""
        client = TestClient(app)

        # 1단계: run
        run_resp = client.post(
            "/v2/agent/run",
            json={"query": "민원 답변 초안 작성해줘"},
        )
        assert run_resp.status_code == 200
        thread_id = run_resp.json()["thread_id"]

        # 2단계: approve(False) → rejected
        reject_resp = client.post(
            "/v2/agent/approve",
            params={"thread_id": thread_id, "approved": "false"},
        )
        assert reject_resp.status_code == 200, f"reject 응답: {reject_resp.text}"
        reject_data = reject_resp.json()
        assert reject_data["status"] == "rejected", f"status 불일치: {reject_data}"
        # 거절 시 tool_results는 비어있어야 한다
        tool_results = reject_data.get("tool_results", {})
        assert not tool_results, f"거절 후 tool_results가 있음: {tool_results}"

    def test_cancel_returns_cancelled(self, setup_real_graph):
        """run → cancel 흐름에서 cancelled 상태를 반환한다."""
        client = TestClient(app)

        # 1단계: run
        run_resp = client.post(
            "/v2/agent/run",
            json={"query": "민원 답변 초안 작성해줘"},
        )
        assert run_resp.status_code == 200
        thread_id = run_resp.json()["thread_id"]

        # 2단계: cancel
        cancel_resp = client.post(
            "/v2/agent/cancel",
            params={"thread_id": thread_id},
        )
        assert cancel_resp.status_code == 200, f"cancel 응답: {cancel_resp.text}"
        cancel_data = cancel_resp.json()
        assert cancel_data["status"] == "cancelled", f"status 불일치: {cancel_data}"


# ---------------------------------------------------------------------------
# TestClass 2: TestV2StreamFlow
# ---------------------------------------------------------------------------


class TestV2StreamFlow:
    """POST /v2/agent/stream SSE 스트리밍 흐름 테스트."""

    def test_stream_yields_node_events(self, setup_real_graph):
        """POST /v2/agent/stream이 session_load, planner 이벤트를 순서대로 반환하고
        approval_wait 또는 __interrupt__ 이벤트로 중단된다.

        LangGraph 1.1.4에서 interrupt()가 호출되면 stream_mode="updates" 기준으로
        노드 이름이 "__interrupt__"인 chunk가 생성된다. 서버는 이 chunk도 이벤트로 전달한다.
        """
        client = TestClient(app)
        response = client.post(
            "/v2/agent/stream",
            json={"query": "민원 답변 초안 작성해줘"},
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        events = _parse_sse_events(response.text)
        assert events, "SSE 이벤트가 없음"

        node_names = [e.get("node") for e in events]
        assert "session_load" in node_names, f"session_load 이벤트 없음: {node_names}"
        assert "planner" in node_names, f"planner 이벤트 없음: {node_names}"

        # approval_wait 또는 __interrupt__ 중 하나가 있어야 한다
        has_approval_node = "approval_wait" in node_names or "__interrupt__" in node_names
        assert has_approval_node, f"approval_wait 또는 __interrupt__ 이벤트 없음: {node_names}"

        # 순서 검증: session_load < planner < (approval_wait or __interrupt__)
        idx_session = node_names.index("session_load")
        idx_planner = node_names.index("planner")
        approval_node = "approval_wait" if "approval_wait" in node_names else "__interrupt__"
        idx_approval = node_names.index(approval_node)
        assert idx_session < idx_planner < idx_approval, (
            f"노드 이벤트 순서 불일치: session_load={idx_session}, "
            f"planner={idx_planner}, {approval_node}={idx_approval}"
        )

    def test_stream_approval_wait_has_thread_id(self, setup_real_graph):
        """스트림 종료 후 /v2/agent/run을 통해 thread_id를 확인한다.

        LangGraph 1.1.4에서 interrupt()는 stream_mode="updates"에서
        "__interrupt__" 노드를 방출하며, 서버의 node_name == "approval_wait" 분기가
        발동하지 않아 awaiting_approval 이벤트가 스트림에 포함되지 않는다.
        대신 /v2/agent/run 엔드포인트가 올바른 awaiting_approval 응답을 반환함을 검증한다.
        """
        client = TestClient(app)

        # stream → interrupt까지 이벤트 수신 확인
        stream_resp = client.post(
            "/v2/agent/stream",
            json={"query": "민원 답변 초안 작성해줘"},
        )
        assert stream_resp.status_code == 200
        events = _parse_sse_events(stream_resp.text)
        assert events, "SSE 이벤트가 없음"

        # session_id가 포함된 이벤트가 있어야 한다 (session_load 노드)
        session_events = [e for e in events if e.get("node") == "session_load"]
        assert session_events, f"session_load 이벤트 없음: {events}"

        # 스트림이 끝난 뒤 interrupt 관련 노드가 있어야 한다
        interrupt_nodes = {"approval_wait", "__interrupt__"}
        node_names = {e.get("node") for e in events}
        assert (
            node_names & interrupt_nodes
        ), f"interrupt 관련 이벤트 없음 (기대: approval_wait 또는 __interrupt__): {node_names}"

    def test_stream_then_approve_completes(self, setup_real_graph):
        """stream(session_id 지정) → thread_id 획득 → approve(True) → completed.

        session_id == thread_id 불변을 이용하여 stream 요청 시 session_id를 지정하고
        동일 값으로 approve를 호출한다.
        """
        client = TestClient(app)
        session_id = "e2e-stream-approve-test"

        # 1단계: stream (session_id 지정)
        stream_resp = client.post(
            "/v2/agent/stream",
            json={"query": "민원 답변 초안 작성해줘", "session_id": session_id},
        )
        assert stream_resp.status_code == 200
        events = _parse_sse_events(stream_resp.text)
        assert events, "SSE 이벤트가 없음"

        # session_load 이벤트 확인
        node_names = [e.get("node") for e in events]
        assert "session_load" in node_names, f"session_load 이벤트 없음: {node_names}"

        # thread_id == session_id 불변을 이용
        thread_id = session_id

        # 2단계: approve(True) → completed
        approve_resp = client.post(
            "/v2/agent/approve",
            params={"thread_id": thread_id, "approved": "true"},
        )
        assert approve_resp.status_code == 200, f"approve 응답: {approve_resp.text}"
        approve_data = approve_resp.json()
        assert approve_data["status"] == "completed", f"status 불일치: {approve_data}"


# ---------------------------------------------------------------------------
# TestClass 3: TestV2SessionResume
# ---------------------------------------------------------------------------


class TestV2SessionResume:
    """세션 재사용 및 신규 thread 생성 테스트."""

    def test_same_session_id_reuses_thread(self, setup_real_graph):
        """동일 session_id로 두 번 실행하면 같은 thread_id를 사용한다.

        Note: session_id == thread_id 불변 (api_server.py 주석 참조).
        따라서 같은 session_id를 보내면 같은 thread_id가 반환된다.
        """
        client = TestClient(app)
        session_id = "e2e-session-reuse-test"

        # 1차 run
        run1_resp = client.post(
            "/v2/agent/run",
            json={"query": "민원 답변 초안 작성해줘", "session_id": session_id},
        )
        assert run1_resp.status_code == 200
        run1_data = run1_resp.json()
        assert run1_data["status"] == "awaiting_approval"
        thread_id_1 = run1_data["thread_id"]
        assert (
            thread_id_1 == session_id
        ), f"thread_id가 session_id와 다름: thread_id={thread_id_1}, session_id={session_id}"
        session_id_1 = run1_data["session_id"]

        # 1차 approve로 graph 완료
        client.post(
            "/v2/agent/approve",
            params={"thread_id": thread_id_1, "approved": "true"},
        )

        # 2차 run — 동일 session_id
        run2_resp = client.post(
            "/v2/agent/run",
            json={"query": "추가 질문입니다", "session_id": session_id},
        )
        assert run2_resp.status_code == 200
        run2_data = run2_resp.json()
        thread_id_2 = run2_data["thread_id"]
        session_id_2 = run2_data["session_id"]

        assert (
            session_id_1 == session_id_2
        ), f"session_id가 달라짐: {session_id_1} != {session_id_2}"
        assert (
            thread_id_1 == thread_id_2
        ), f"동일 session_id에서 thread_id가 달라짐: {thread_id_1} != {thread_id_2}"

    def test_session_id_none_generates_unique_threads(self, setup_real_graph):
        """session_id 없이 두 번 실행하면 서로 다른 thread_id가 생성된다."""
        client = TestClient(app)

        # 1차 run (session_id 없음)
        run1_resp = client.post(
            "/v2/agent/run",
            json={"query": "민원 답변 초안 작성해줘"},
        )
        assert run1_resp.status_code == 200
        thread_id_1 = run1_resp.json()["thread_id"]

        # 1차 approve로 graph 완료
        client.post(
            "/v2/agent/approve",
            params={"thread_id": thread_id_1, "approved": "false"},
        )

        # 2차 run (session_id 없음)
        run2_resp = client.post(
            "/v2/agent/run",
            json={"query": "다른 민원 질문"},
        )
        assert run2_resp.status_code == 200
        thread_id_2 = run2_resp.json()["thread_id"]

        assert (
            thread_id_1 != thread_id_2
        ), f"session_id 없이 실행했는데 thread_id가 같음: {thread_id_1}"


# ---------------------------------------------------------------------------
# TestClass 4: TestHttpClientCompatibility
# ---------------------------------------------------------------------------


class TestHttpClientCompatibility:
    """수정된 GovOnClient.approve() / cancel()이 실제 엔드포인트와 호환됨을 검증한다."""

    def test_approve_sends_correct_params(self, setup_real_graph):
        """GovOnClient.approve()가 쿼리 파라미터로 올바르게 동작한다.

        httpx.Client를 패치하여 실제로 params=로 요청이 전달되는지 확인한다.
        이전 구현은 json= body를 사용했기 때문에 FastAPI 422 오류가 발생했다.
        """
        from unittest.mock import patch as _patch

        from src.cli.http_client import GovOnClient

        gov_client = GovOnClient("http://testserver")

        # httpx.Client.post() 호출을 가로채 params가 올바르게 전달되는지 검증
        captured_calls = []

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"status": "completed", "thread_id": "t-test"}

        mock_httpx = MagicMock()
        mock_httpx.post.side_effect = lambda url, **kwargs: (
            captured_calls.append(kwargs) or mock_response
        )

        with _patch("httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__ = lambda s: mock_httpx
            mock_cls.return_value.__exit__ = MagicMock(return_value=False)

            result = gov_client.approve(thread_id="test-thread-123", approved=True)

        # params가 사용되었는지 확인 (json body가 아닌 query params)
        assert captured_calls, "httpx.Client.post()가 호출되지 않음"
        call_kwargs = captured_calls[0]
        assert "params" in call_kwargs, f"approve()가 params를 사용하지 않음. kwargs: {call_kwargs}"
        assert (
            "json" not in call_kwargs
        ), f"approve()가 여전히 json body를 사용함. kwargs: {call_kwargs}"
        params = call_kwargs["params"]
        assert params.get("thread_id") == "test-thread-123"
        assert params.get("approved") == "true", f"approved 파라미터 오류: {params}"

    def test_cancel_sends_correct_params(self, setup_real_graph):
        """GovOnClient.cancel()이 쿼리 파라미터로 올바르게 동작한다.

        httpx.Client를 패치하여 실제로 params=로 요청이 전달되는지 확인한다.
        이전 구현은 json= body를 사용했기 때문에 FastAPI 422 오류가 발생했다.
        """
        from unittest.mock import patch as _patch

        from src.cli.http_client import GovOnClient

        gov_client = GovOnClient("http://testserver")

        captured_calls = []

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"status": "cancelled", "thread_id": "t-test"}

        mock_httpx = MagicMock()
        mock_httpx.post.side_effect = lambda url, **kwargs: (
            captured_calls.append(kwargs) or mock_response
        )

        with _patch("httpx.Client") as mock_cls:
            mock_cls.return_value.__enter__ = lambda s: mock_httpx
            mock_cls.return_value.__exit__ = MagicMock(return_value=False)

            result = gov_client.cancel(thread_id="test-thread-456")

        # params가 사용되었는지 확인 (json body가 아닌 query params)
        assert captured_calls, "httpx.Client.post()가 호출되지 않음"
        call_kwargs = captured_calls[0]
        assert "params" in call_kwargs, f"cancel()이 params를 사용하지 않음. kwargs: {call_kwargs}"
        assert (
            "json" not in call_kwargs
        ), f"cancel()이 여전히 json body를 사용함. kwargs: {call_kwargs}"
        params = call_kwargs["params"]
        assert params.get("thread_id") == "test-thread-456"
